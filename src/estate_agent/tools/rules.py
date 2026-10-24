"""search_rules: semantic search over official tenancy documents.

Documents in data/rules/ are split into passages (one per "## " section, such as
an article of a law) and embedded into an in-memory ChromaDB collection. The
default embedding model is all-MiniLM-L6-v2 (ONNX), which Chroma downloads on
first use; tests pass a small offline embedding function instead.
"""

import re
import uuid
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
# Below this cosine similarity a section is treated as unrelated to the query.
MIN_SIMILARITY = 0.2


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

    @classmethod
    def from_directory(cls, path: Path = RULES_DIR, embedding_function: EmbeddingFunction | None = None) -> "RulesIndex":
        files = sorted(
            f for f in path.glob("*") if f.suffix in {".md", ".txt"} and f.name.lower() != "readme.md"
        )
        return cls([p for f in files for p in split_document(f)], embedding_function)

    def search(self, query: str, k: int = 4) -> list[tuple[Passage, float]]:
        """Top-k sections, each scored by its best chunk's cosine similarity."""
        if self._collection is None:
            return []
        # Fetch extra chunks so k distinct sections survive de-duplication.
        found = self._collection.query(query_texts=[query], n_results=min(k * 3, len(self.passages)))
        best: dict[str, tuple[Passage, float]] = {}
        for chunk_id, distance in zip(found["ids"][0], found["distances"][0]):
            passage = self._by_id[chunk_id]
            if passage.passage_id not in best:  # results arrive best first
                best[passage.passage_id] = (passage, 1 - distance)
        return list(best.values())[:k]


def search_rules(index: RulesIndex, query: str, k: int = 4) -> dict:
    """Return the passages most similar to query, each citable by passage_id."""
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
    for passage, score in index.search(query, k):
        if score < MIN_SIMILARITY:
            continue
        result["results"].append({
            "passage_id": passage.passage_id,
            "document": passage.document,
            "title": passage.title,
            "section": passage.section,
            "similarity": round(score, 3),
            "text": passage.section_text,
        })
    if result["results"]:
        result["note"] = "Ranked by meaning, not verified relevance; check each passage answers the question"
    else:
        result["note"] = "No passage is similar enough to the query; the documents may not cover it"
    return result
