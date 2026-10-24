import pytest

from estate_agent.tools import ToolError
from estate_agent.tools.rules import RulesIndex, search_rules, split_document, tokenize

# Invented test text; real documents are added in milestone 2.
NOTICE = """# Example Tenancy Notice Rules

Either party wishing to change the tenancy terms must give the other written notice
ninety days before the contract ends.
"""
DEPOSIT = """# Example Deposit Rules

The landlord holds the security deposit and returns it when the tenant leaves the unit
in good condition.
"""


@pytest.fixture
def rules_dir(tmp_path):
    (tmp_path / "notice.md").write_text(NOTICE)
    (tmp_path / "deposit.md").write_text(DEPOSIT)
    (tmp_path / "README.md").write_text("# Not a rule document\n\nsecurity deposit notice")
    return tmp_path


def test_tokenize_drops_stopwords_and_folds_plurals():
    assert tokenize("What are the tenants' Rights?") == ["tenant", "right"]
    assert tokenize("Glass access") == ["glass", "access"]


def test_finds_relevant_passage_with_citation(rules_dir):
    result = search_rules(RulesIndex.from_directory(rules_dir), "How much notice before the tenancy ends?")
    top = result["results"][0]
    assert top["passage_id"] == "notice#1"
    assert top["title"] == "Example Tenancy Notice Rules"
    assert "ninety days" in top["text"]


def test_readme_is_not_indexed(rules_dir):
    index = RulesIndex.from_directory(rules_dir)
    assert {p.document for p in index.passages} == {"notice.md", "deposit.md"}


def test_unmatched_and_empty_index(rules_dir):
    assert search_rules(RulesIndex.from_directory(rules_dir), "swimming pool")["note"]
    empty = search_rules(RulesIndex([]), "deposit")
    assert empty["results"] == [] and "No rule documents" in empty["note"]
    with pytest.raises(ToolError):
        search_rules(RulesIndex([]), "   ")


def test_long_documents_split_into_passages(tmp_path):
    path = tmp_path / "long.md"
    path.write_text("# Long\n\n" + "\n\n".join(f"Paragraph {i} " + "word " * 50 for i in range(4)))
    passages = split_document(path)
    assert [p.passage_id for p in passages] == ["long#1", "long#2"]
    assert all(p.title == "Long" for p in passages)
