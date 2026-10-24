import os
import sys
from pathlib import Path

import pytest

from estate_agent.tools import ToolError
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from estate_agent.tools.rules import BM25, RULES_DIR, RulesIndex, search_rules, split_document, tokenize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fetch_rules  # noqa: E402

# Invented test text in the same format fetch_rules.py writes.
NOTICE = """# Example Tenancy Law

Source: https://example.test

## Article (1) Notice

Either party wishing to change the tenancy terms must give the other written notice
ninety days before the contract ends.

## Article (2) Deposit

The landlord holds the security deposit and returns it when the tenant leaves the unit
in good condition.
"""


@pytest.fixture
def rules_dir(tmp_path):
    (tmp_path / "example-law.md").write_text(NOTICE)
    (tmp_path / "README.md").write_text("# Not a rule document\n\nsecurity deposit notice")
    return tmp_path


def test_one_passage_per_article_with_citable_ids(rules_dir):
    passages = split_document(rules_dir / "example-law.md")
    assert [p.passage_id for p in passages] == ["example-law#art1", "example-law#art2"]
    assert [p.chunk_id for p in passages] == ["example-law#art1", "example-law#art2"]
    assert passages[0].section == "Article (1) Notice"
    assert passages[0].title == "Example Tenancy Law"
    assert "Source:" not in passages[0].text


def test_long_sections_are_chunked_but_cited_whole(tmp_path):
    path = tmp_path / "long.md"
    path.write_text("# Long\n\n## Article (7)\n\n" + "\n\n".join(f"part{i} " * 60 for i in range(4)))
    passages = split_document(path, max_words=130)
    assert [p.chunk_id for p in passages] == ["long#art7.1", "long#art7.2"]
    assert {p.passage_id for p in passages} == {"long#art7"}
    assert all(p.section_text.count("part") == 240 for p in passages)


def test_search_returns_each_section_once_with_full_text(tmp_path, embedding):
    path = tmp_path / "long.md"
    path.write_text("# Long\n\n## Article (7)\n\n" + "\n\n".join("rent rent increase " * 20 for _ in range(4))
                    + "\n\n## Article (8)\n\nrent deposit")
    index = RulesIndex(split_document(path, max_words=130), embedding)
    results = search_rules(index, "rent increase", k=2)["results"]
    assert [r["passage_id"] for r in results] == ["long#art7", "long#art8"]
    assert results[0]["text"].count("increase") == 80


def test_document_without_sections_is_one_passage(tmp_path):
    path = tmp_path / "plain.txt"
    path.write_text("# Plain\n\nSome rule text.")
    [passage] = split_document(path)
    assert passage.passage_id == "plain#s1" and passage.text == "Some rule text."


def test_finds_relevant_article(rules_dir, embedding):
    index = RulesIndex.from_directory(rules_dir, embedding)
    result = search_rules(index, "How much notice before the tenancy contract ends?", k=2)
    top = result["results"][0]
    assert top["passage_id"] == "example-law#art1"
    assert top["section"] == "Article (1) Notice"
    assert "ninety days" in top["text"]
    assert result["results"][0]["similarity"] > result["results"][1]["similarity"]


def test_readme_is_not_indexed(rules_dir, embedding):
    index = RulesIndex.from_directory(rules_dir, embedding)
    assert {p.document for p in index.passages} == {"example-law.md"}


def test_empty_index_and_bad_query():
    empty = search_rules(RulesIndex([]), "deposit")
    assert empty["results"] == [] and "No rule documents" in empty["note"]
    with pytest.raises(ToolError):
        search_rules(RulesIndex([]), "   ")


def test_k_is_capped_and_unrelated_sections_dropped(rules_dir, embedding):
    index = RulesIndex.from_directory(rules_dir, embedding)
    # The whole document as the query is similar to both sections.
    assert len(search_rules(index, NOTICE, k=50)["results"]) == 2
    unrelated = search_rules(index, "swimming pool", k=4)
    assert unrelated["results"] == []
    assert "similar enough" in unrelated["note"]


class TopicEmbedding(EmbeddingFunction[Documents]):
    """Ranks by one topic only: texts about deposits ('deposit' or 'money') point one way."""

    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        return [[1.0, 0.0] if ("deposit" in text.lower() or "money" in text.lower()) else [0.6, 0.8]
                for text in input]

    @staticmethod
    def name() -> str:
        return "topic-test"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "TopicEmbedding":
        return TopicEmbedding()


def test_tokenize_cuts_words_to_a_shared_prefix():
    assert tokenize("Registration of the registered Tenancy") == ["regis", "regis", "tenan"]


def test_bm25_prefers_the_section_with_the_rare_term():
    scores = BM25(["rent is paid yearly", "contracts are registered with RERA", "rent rules"]).scores("registration")
    assert scores[1] > 0 and scores[0] == scores[2] == 0


def test_keyword_match_is_added_when_meaning_misses_it(tmp_path):
    (tmp_path / "law.md").write_text(
        "# Law\n\n## Article (1) Deposit\n\nThe deposit is returned at the end.\n\n"
        "## Article (2) Contracts\n\nAll contracts will be registered with RERA.\n")
    index = RulesIndex.from_directory(tmp_path, TopicEmbedding())
    # "money" finds Article 1 by meaning only; "registration" finds Article 2 by keyword only.
    results = search_rules(index, "money registration", k=1)["results"]
    assert [(r["passage_id"], r["matched_by"]) for r in results] == [("law#art1", "meaning"), ("law#art2", "keywords")]


def test_keyword_match_already_found_by_meaning_is_not_duplicated(tmp_path):
    (tmp_path / "law.md").write_text("# Law\n\n## Article (1) Deposit\n\nThe deposit is returned.\n")
    results = search_rules(RulesIndex.from_directory(tmp_path, TopicEmbedding()), "deposit", k=3)["results"]
    assert [(r["passage_id"], r["matched_by"]) for r in results] == [("law#art1", "meaning and keywords")]


# --- committed legislation -------------------------------------------------


def test_committed_legislation_is_consolidated_and_complete():
    law = split_document(RULES_DIR / "law-26-2007.md")
    articles = {p.section.split(")")[0] + ")" for p in law}
    assert articles == {f"Article ({n})" for n in range(1, 38)}
    text = (RULES_DIR / "law-26-2007.md").read_text()
    assert text.count("*Text as superseded by Law No. 33 of 2008.*") == 11
    decree = {p.section for p in split_document(RULES_DIR / "decree-43-2013.md")}
    assert "Article (1) Percentages of Increase" in decree


def test_superseded_list_parsing():
    intro = "Articles (2), (3), (25), and (36) of the Original Law are hereby superseded by the following:"
    assert fetch_rules.superseded_numbers(intro) == [2, 3, 25, 36]


def test_converter_reproduces_committed_rules_when_raw_html_present():
    if not all((fetch_rules.RAW / f"{name}.html").exists() for name in fetch_rules.SOURCES):
        pytest.skip("raw HTML not downloaded (run scripts/fetch_rules.py)")
    before = {p.name: p.read_text() for p in RULES_DIR.glob("*.md")}
    fetch_rules.main()
    assert {p.name: p.read_text() for p in RULES_DIR.glob("*.md")} == before


# --- real model ------------------------------------------------------------

model = pytest.mark.skipif(not os.environ.get("RUN_MODEL_TESTS"), reason="set RUN_MODEL_TESTS=1")


@pytest.fixture(scope="module")
def real_index():
    return RulesIndex.from_directory()


@model
@pytest.mark.model
@pytest.mark.parametrize("question, expected", [
    ("How much can my landlord increase the rent when I renew?", "decree-43-2013#art1"),
    ("How many days notice must be given before changing the lease terms at renewal?", "law-26-2007#art14"),
    ("Can the landlord evict me because he wants to sell the property?", "law-26-2007#art25"),
    ("Who pays for maintenance and repairs of the property?", "law-26-2007#art16"),
    # Missed by meaning-only search in evaluation run 1 (case R7):
    ("registration of tenancy contract Ejari", "law-26-2007#art4"),
    ("Does a tenancy contract have to be registered?", "law-26-2007#art4"),
    ("when is rent due if not specified in the lease", "law-26-2007#art12"),
])
def test_real_model_retrieves_expected_article(real_index, question, expected):
    ids = [r["passage_id"] for r in search_rules(real_index, question, k=4)["results"]]
    assert expected in ids, ids


@model
@pytest.mark.model
def test_real_model_returns_nothing_for_unrelated_question(real_index):
    assert search_rules(real_index, "What is the capital of Mars?")["results"] == []
