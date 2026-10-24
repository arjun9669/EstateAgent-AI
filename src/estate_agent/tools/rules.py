"""search_rules: find passages in official tenancy and mortgage documents.

Milestone 1 ranks passages with BM25 keyword scoring. Milestone 2 replaces this
with embeddings in ChromaDB behind the same search() interface.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from estate_agent.tools.base import ToolError

RULES_DIR = Path(__file__).resolve().parents[3] / "data" / "rules"
PASSAGE_MIN_WORDS = 80
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i if in is it its may my "
    "of on or the this to what when which who will with".split()
)


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    # Light plural folding so "tenants" matches "tenant".
    return [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
            for w in words if w not in STOPWORDS]


@dataclass(frozen=True)
class Passage:
    passage_id: str
    document: str
    title: str
    text: str


def split_document(path: Path) -> list[Passage]:
    text = path.read_text(encoding="utf-8")
    title = next((line[2:].strip() for line in text.splitlines() if line.startswith("# ")), path.stem)
    passages, buffer = [], []

    def flush():
        if buffer:
            passages.append(Passage(f"{path.stem}#{len(passages) + 1}", path.name, title, "\n\n".join(buffer)))
            buffer.clear()

    for paragraph in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if paragraph:
            buffer.append(paragraph)
            if sum(len(p.split()) for p in buffer) >= PASSAGE_MIN_WORDS:
                flush()
    flush()
    return passages


class RulesIndex:
    def __init__(self, passages: list[Passage], k1: float = 1.5, b: float = 0.75):
        self.passages = passages
        self.k1, self.b = k1, b
        self._terms = [Counter(tokenize(p.text)) for p in passages]
        self._lengths = [sum(t.values()) for t in self._terms]
        self._avg_length = sum(self._lengths) / len(passages) if passages else 0
        df = Counter(term for terms in self._terms for term in terms)
        n = len(passages)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    @classmethod
    def from_directory(cls, path: Path = RULES_DIR) -> "RulesIndex":
        files = sorted(
            f for f in path.glob("*") if f.suffix in {".md", ".txt"} and f.name.lower() != "readme.md"
        )
        return cls([p for f in files for p in split_document(f)])

    def search(self, query: str, k: int = 4) -> list[tuple[Passage, float]]:
        query_terms = set(tokenize(query))
        scored = []
        for passage, terms, length in zip(self.passages, self._terms, self._lengths):
            score = 0.0
            for term in query_terms & terms.keys():
                tf = terms[term]
                norm = self.k1 * (1 - self.b + self.b * length / self._avg_length)
                score += self._idf[term] * tf * (self.k1 + 1) / (tf + norm)
            if score > 0:
                scored.append((passage, score))
        scored.sort(key=lambda item: (-item[1], item[0].passage_id))
        return scored[:k]


def search_rules(index: RulesIndex, query: str, k: int = 4) -> dict:
    """Return the passages most relevant to query, each citable by passage_id."""
    if not query.strip():
        raise ToolError("query must not be empty")
    k = max(1, min(k, 8))
    result = {"source": "Official documents in data/rules", "query": query, "results": []}
    if not index.passages:
        result["note"] = "No rule documents have been added yet"
        return result
    for passage, score in index.search(query, k):
        result["results"].append({
            "passage_id": passage.passage_id,
            "document": passage.document,
            "title": passage.title,
            "score": round(score, 3),
            "text": passage.text,
        })
    if not result["results"]:
        result["note"] = "No passage matched the query terms"
    return result
