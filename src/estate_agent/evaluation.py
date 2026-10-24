"""Score recorded agent answers against the frozen evaluation set (eval/cases-v1.json).

Each case is scored on four dimensions; a case passes only if all four pass.

  tool_choice   every required tool ran successfully; no forbidden tool was called
  correctness   every expected fact is in the answer; no forbidden phrasing is
  citations     required listing/law IDs are cited, every cited ID came from a tool,
                and every quotation appears word for word in the tool results
  grounding     the agent's own grounding check passed (no unsupported numbers)

Numbers are matched like the grounding check: "AED 2.3 million" matches 2,277,500.
"""

import json
import re
from collections import Counter
from pathlib import Path

from estate_agent.grounding import LISTING_ID, PASSAGE_ID, _matches, numbers_in

DIMENSIONS = ("tool_choice", "correctness", "citations", "grounding")
QUOTE = re.compile(r"[\"“]([^\"”]{12,})[\"”]")


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _strings(value) -> list[str]:
    """All string values inside a JSON-like structure."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def _has_number(answer: str, expected: float) -> bool:
    prose = LISTING_ID.sub(" ", PASSAGE_ID.sub(" ", answer))
    return any(_matches(number, expected) for number in numbers_in(prose))


def _fact_found(fact: dict, answer: str) -> bool:
    text = _norm(answer)
    if "number" in fact:
        return _has_number(answer, fact["number"]) or (
            "or_text" in fact and _norm(fact["or_text"]) in text)
    if "text" in fact:
        return _norm(fact["text"]) in text
    return any(_norm(option) in text for option in fact["any_of"])


def _describe(fact: dict) -> str:
    if "number" in fact:
        return str(fact["number"])
    return fact.get("text") or " / ".join(fact["any_of"])


def score_case(case: dict, record: dict) -> dict:
    """Score one recorded answer. record has: answer, steps, grounding."""
    answer, steps = record["answer"], record["steps"]
    problems = {dimension: [] for dimension in DIMENSIONS}

    succeeded = {s["tool"] for s in steps if s["ok"]}
    called = {s["tool"] for s in steps}
    for tool in case.get("tools_required", []):
        if tool not in succeeded:
            problems["tool_choice"].append(f"did not run {tool}")
    for tool in case.get("tools_forbidden", []):
        if tool in called:
            problems["tool_choice"].append(f"called {tool}")

    for fact in case.get("facts", []):
        if not _fact_found(fact, answer):
            problems["correctness"].append(f"missing {_describe(fact)}")
    for phrase in case.get("must_not_contain", []):
        if _norm(phrase) in _norm(answer):
            problems["correctness"].append(f"says '{phrase}'")

    cited = set(LISTING_ID.findall(answer)) | set(PASSAGE_ID.findall(answer))
    for citation in case.get("citations_required", []):
        if citation not in cited:
            problems["citations"].append(f"does not cite {citation}")
    if case.get("no_citations") and cited:
        problems["citations"].append(f"cites {', '.join(sorted(cited))} where nothing should be cited")
    for citation in record["grounding"].get("unknown_citations", []):
        problems["citations"].append(f"cites {citation}, which no tool returned")
    evidence = _norm(" ".join([record.get("question", case["question"])]
                              + [s for step in steps if step["ok"] for s in _strings(step["result"])]))
    for quote in QUOTE.findall(answer):
        if _norm(quote).strip(" .,;:") not in evidence:
            problems["citations"].append(f"quote not found word for word: \"{quote[:60]}\"")

    if record["grounding"].get("ungrounded_numbers"):
        problems["grounding"].append("unsupported numbers: " + ", ".join(record["grounding"]["ungrounded_numbers"]))

    return {
        "id": case["id"],
        "category": case["category"],
        "passed": not any(problems.values()),
        **{dimension: not problems[dimension] for dimension in DIMENSIONS},
        "problems": [f"{d}: {p}" for d in DIMENSIONS for p in problems[d]],
    }


def summarize(scores: list[dict]) -> dict:
    total = len(scores)
    by_category = Counter(s["category"] for s in scores)
    passed_by_category = Counter(s["category"] for s in scores if s["passed"])
    return {
        "cases": total,
        "passed": sum(s["passed"] for s in scores),
        **{dimension: sum(s[dimension] for s in scores) for dimension in DIMENSIONS},
        "by_category": {c: f"{passed_by_category[c]}/{n}" for c, n in sorted(by_category.items())},
    }


def load_cases(path: Path) -> list[dict]:
    return json.loads(path.read_text())["cases"]


def score_run(cases: list[dict], run: dict) -> tuple[list[dict], dict]:
    """Score a recorded run (as written by scripts/run_eval.py) against the cases."""
    records = {r["id"]: r for r in run["results"]}
    missing = [c["id"] for c in cases if c["id"] not in records]
    if missing:
        raise ValueError(f"Run has no result for cases: {', '.join(missing)}")
    scores = [score_case(case, records[case["id"]]) for case in cases]
    return scores, summarize(scores)
