"""search_rules: hybrid keyword + semantic search over official tenancy documents.

Documents in data/rules/ are split into passages (one per "## " section, such as
an article of a law). Each query is ranked two ways:

- by meaning: chunks embedded in an in-memory ChromaDB collection (default model
  all-MiniLM-L6-v2, ONNX, downloaded by Chroma on first use; tests pass a small
  offline embedding function instead);
- by keywords: BM25 over whole sections, with words cut to a short prefix so
  "registration" matches "registered".

Results are the top-k sections by meaning, plus the strongest keyword match if
it is not already among them. Embeddings alone missed literal terms (evaluation
run 1, case R7: "registration of tenancy contract" did not find Article 4, "will
be registered with RERA"). Adding, rather than re-ranking, means a keyword match
can never push out a section the meaning search found: an even rank fusion did
exactly that on other queries (docs/EVALUATION.md).
"""

import math
import re
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.api.types import EmbeddingFunction
from chromadb.utils import embedding_functions

from estate_agent.tools.base import ToolError

RULES_DIR = Path(__file__).resolve().parents[3] / "data" / "rules"
# MiniLM reads at most 256 word pieces, so long sections are embedded in chunks.
# Results always return the whole section, so a list is never cut in half.
PASSAGE_MAX_WORDS = 170
MAX_K = 8
# Below this cosine similarity a section is not a meaning match.
MIN_SIMILARITY = 0.2
STEM_PREFIX = 5
STOPWORDS = frozenset(
    "a an and any are as at be by can do does for from has have how i if in is it its may my "
    "of on or the this to what when which who will with".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase words without stopwords, cut to STEM_PREFIX letters (a crude stemmer)."""
    return [w[:STEM_PREFIX] for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._terms = [Counter(tokenize(d)) for d in documents]
        self._lengths = [sum(t.values()) for t in self._terms]
        self._average = sum(self._lengths) / len(documents) if documents else 0
        frequency = Counter(term for terms in self._terms for term in terms)
        n = len(documents)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in frequency.items()}

    def scores(self, query: str) -> list[float]:
        query_terms = set(tokenize(query))
        result = []
        for terms, length in zip(self._terms, self._lengths):
            norm = self.k1 * (1 - self.b + self.b * length / self._average)
            result.append(sum(
                self._idf[t] * terms[t] * (self.k1 + 1) / (terms[t] + norm)
                for t in query_terms & terms.keys()
            ))
        return result


@dataclass(frozen=True)
class Passage:
    """An embedded chunk of a section. passage_id identifies the section for citation."""

    chunk_id: str
    passage_id: str
    document: str
    title: str
    section: str
    text: str  # this chunk
    section_text: str  # the whole section


def _chunks(paragraphs: list[str], max_words: int) -> list[list[str]]:
    """Group paragraphs into chunks of at most max_words (a longer paragraph stands alone)."""
    chunks, current, size = [], [], 0
    for paragraph in paragraphs:
        words = len(paragraph.split())
        if current and size + words > max_words:
            chunks.append(current)
            current, size = [], 0
        current.append(paragraph)
        size += words
    if current:
        chunks.append(current)
    return chunks


def _section_key(section: str, number: int) -> str:
    article = re.match(r"Article \((\d+)\)", section)
    return f"art{article[1]}" if article else f"s{number}"


def split_document(path: Path, max_words: int = PASSAGE_MAX_WORDS) -> list[Passage]:
    """One passage per "## " section, or several chunks of it if long.

    Text before the first section (title, source lines) is not indexed. A file
    with no sections is indexed as a single untitled section.
    """
    text = path.read_text(encoding="utf-8")
    title = next((line[2:].strip() for line in text.splitlines() if line.startswith("# ")), path.stem)
    sections = re.split(r"^## (.+)$", text, flags=re.M)
    if len(sections) == 1:
        body = re.sub(r"^# .*$", "", text, count=1, flags=re.M)
        pairs = [("", body)]
    else:
        pairs = list(zip(sections[1::2], sections[2::2]))

    passages = []
    for number, (section, body) in enumerate(pairs, 1):
        section = section.strip()
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        key = _section_key(section, number)
        chunks = _chunks(paragraphs, max_words)
        passage_id = f"{path.stem}#{key}"
        for part, chunk in enumerate(chunks, 1):
            chunk_id = f"{passage_id}.{part}" if len(chunks) > 1 else passage_id
            passages.append(Passage(
                chunk_id, passage_id, path.name, title, section, "\n\n".join(chunk), "\n\n".join(paragraphs)
            ))
    return passages


def default_embedding_function() -> EmbeddingFunction:
    return embedding_functions.DefaultEmbeddingFunction()


class RulesIndex:
    def __init__(self, passages: list[Passage], embedding_function: EmbeddingFunction | None = None):
        self.passages = passages
        self._by_id = {p.chunk_id: p for p in passages}
        if len(self._by_id) != len(passages):
            raise ValueError("Duplicate chunk IDs")
        self._collection = None
        if passages:
            client = chromadb.EphemeralClient()
            self._collection = client.create_collection(
                f"rules-{uuid.uuid4().hex}",
                embedding_function=embedding_function or default_embedding_function(),
                configuration={"hnsw": {"space": "cosine"}},
            )
            # The section title is embedded with its text so "Article (25) Eviction
            # Cases" contributes to matching; results show the text alone.
            self._collection.add(
                ids=[p.chunk_id for p in passages],
                documents=[f"{p.section}\n\n{p.text}".strip() for p in passages],
            )
        # Keyword index over whole sections (one entry per passage_id).
        self._sections = list({p.passage_id: p for p in passages}.values())
        self._bm25 = BM25([f"{p.section}\n\n{p.section_text}" for p in self._sections])

    @classmethod
    def from_directory(cls, path: Path = RULES_DIR, embedding_function: EmbeddingFunction | None = None) -> "RulesIndex":
        files = sorted(
            f for f in path.glob("*") if f.suffix in {".md", ".txt"} and f.name.lower() != "readme.md"
        )
        return cls([p for f in files for p in split_document(f)], embedding_function)

    def search(self, query: str, k: int = 4) -> list["Match"]:
        """Top-k sections by meaning, plus the best keyword match if not among them."""
        if self._collection is None:
            return []
        # Similarity of each section = its best chunk's cosine similarity.
        found = self._collection.query(query_texts=[query], n_results=len(self.passages))
        similarity: dict[str, float] = {}
        for chunk_id, distance in zip(found["ids"][0], found["distances"][0]):
            similarity.setdefault(self._by_id[chunk_id].passage_id, 1 - distance)
        keyword = dict(zip((p.passage_id for p in self._sections), self._bm25.scores(query)))
        section = {p.passage_id: p for p in self._sections}

        by_meaning = [pid for pid in sorted(similarity, key=lambda p: (-similarity[p], p))
                      if similarity[pid] >= MIN_SIMILARITY][:k]
        chosen = [(pid, "meaning") for pid in by_meaning]
        best_keyword = max(keyword, key=lambda p: (keyword[p], p), default=None)
        if best_keyword is not None and keyword[best_keyword] > 0:
            if best_keyword in by_meaning:
                chosen[by_meaning.index(best_keyword)] = (best_keyword, "meaning and keywords")
            else:
                chosen.append((best_keyword, "keywords"))
        return [Match(section[pid], similarity.get(pid, 0.0), keyword[pid], how) for pid, how in chosen]


@dataclass(frozen=True)
class Match:
    passage: Passage
    similarity: float
    keyword_score: float
    matched_by: str  # "meaning", "keywords" or "meaning and keywords"


def search_rules(index: RulesIndex, query: str, k: int = 4) -> dict:
    """Return the sections most related to query, each citable by passage_id."""
    if not query.strip():
        raise ToolError("query must not be empty")
    k = max(1, min(k, MAX_K))
    result = {
        "source": "Official Dubai legislation in data/rules (English versions; Arabic text prevails)",
        "query": query,
        "results": [],
    }
    if not index.passages:
        result["note"] = "No rule documents have been added yet"
        return result
    for match in index.search(query, k):
        passage = match.passage
        result["results"].append({
            "passage_id": passage.passage_id,
            "document": passage.document,
            "title": passage.title,
            "section": passage.section,
            "similarity": round(match.similarity, 3),
            "keyword_score": round(match.keyword_score, 3),
            "matched_by": match.matched_by,
            "text": passage.section_text,
        })
    if result["results"]:
        result["note"] = ("Found by meaning and keywords, not verified relevance; "
                          "check each passage answers the question")
    else:
        result["note"] = "No passage is similar enough to the query; the documents may not cover it"
    return result
