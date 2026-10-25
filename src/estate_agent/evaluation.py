"""Score recorded agent answers against a frozen evaluation set (eval/cases-v*.json).

Each case is scored on four dimensions; a case passes only if all four pass.

  tool_choice   every required tool ran successfully; no forbidden tool was called
  correctness   every expected fact is in the answer; no forbidden phrasing is
  citations     required listing/law IDs are cited, every cited ID came from a tool,
                and every quotation appears word for word in the tool results
  grounding     the agent's own grounding check passed (no unsupported numbers)

Numbers are matched like the grounding check: "AED 2.3 million" matches 2,277,500.

Scorer versions. Recorded runs keep the version they were scored with, so their
stored results can always be reproduced; newer versions are reported alongside.

  1.0  first version (runs 1 and 2)
  1.1  quotation marks are paired in order. In 1.0 a quote shorter than 12
       characters was skipped and the search restarted at its closing mark, so
       the text *between* two quotes ("2138.9" AED. This is a "5.3") was checked
       as if it were a quote.
  1.2  quotations are also found in the names of tool result fields, not only
       their values. Once the mortgage tool used plain-English names, the model
       quoted them ("DLD transfer fee (4% of price)") and 1.1 failed those
       accurate quotes.
"""

import json
import re
from collections import Counter
from pathlib import Path

from estate_agent.grounding import LISTING_ID, PASSAGE_ID, _matches, numbers_in

DIMENSIONS = ("tool_choice", "correctness", "citations", "grounding")
SCORER_VERSIONS = ("1.0", "1.1", "1.2")
LATEST_SCORER = "1.2"
MIN_QUOTE_CHARS = 12
_QUOTE_1_0 = re.compile(r"[\"“]([^\"”]{12,})[\"”]")
_QUOTE_PAIR = re.compile(r"[\"“]([^\"”]*)[\"”]")


def quotations(answer: str, scorer: str = LATEST_SCORER) -> list[str]:
    """Quoted passages of at least MIN_QUOTE_CHARS characters."""
    if scorer == "1.0":
        return _QUOTE_1_0.findall(answer)
    return [q for q in _QUOTE_PAIR.findall(answer) if len(q) >= MIN_QUOTE_CHARS]


CODE_NAME = re.compile(r"\b[a-z]+(?:_[a-z]+)+\b")


def field_name_leaks(answer: str) -> list[str]:
    """Code-style names (like within_debt_burden_cap) shown to the user. Not part of
    pass/fail; reported separately so changes to it don't alter recorded scores."""
    return sorted(set(CODE_NAME.findall(answer)))


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _strings(value, keys: bool = False) -> list[str]:
    """All string values inside a JSON-like structure, and its keys if keys=True."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for k, v in value.items() for s in ([k] if keys else []) + _strings(v, keys)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v, keys)]
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


def score_case(case: dict, record: dict, scorer: str = LATEST_SCORER) -> dict:
    """Score one recorded answer. record has: answer, steps, grounding."""
    if scorer not in SCORER_VERSIONS:
        raise ValueError(f"Unknown scorer version {scorer!r}")
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
                              + [s for step in steps if step["ok"]
                                 for s in _strings(step["result"], keys=scorer >= "1.2")]))
    for quote in quotations(answer, scorer):
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


EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"


def load_cases(path: Path) -> list[dict]:
    return json.loads(path.read_text())["cases"]


def cases_for_run(run: dict) -> list[dict]:
    """The case set a recorded run was made with (runs before v2 did not say: v1)."""
    return load_cases(EVAL_DIR / run.get("cases_file", "cases-v1.json"))


def score_run(cases: list[dict], run: dict, scorer: str | None = None) -> tuple[list[dict], dict]:
    """Score a recorded run (as written by scripts/run_eval.py) against the cases.

    scorer defaults to the version the run was recorded with (1.0 if not stated).
    """
    scorer = scorer or run.get("scorer_version", "1.0")
    records = {r["id"]: r for r in run["results"]}
    missing = [c["id"] for c in cases if c["id"] not in records]
    if missing:
        raise ValueError(f"Run has no result for cases: {', '.join(missing)}")
    scores = [score_case(case, records[case["id"]], scorer) for case in cases]
    return scores, summarize(scores)
