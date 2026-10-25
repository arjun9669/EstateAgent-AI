"""Run the frozen evaluation set once against the configured model and record it.

    python scripts/run_eval.py                          # cases-v1, writes eval/runs/<date>-<model>-<commit>.json
    python scripts/run_eval.py --cases eval/cases-v2.json
    python scripts/run_eval.py --only R1 R2             # a few cases, printed but not saved

A saved run is a permanent record: answers, every tool call and result, the
grounding report, timings and scores. Do not overwrite or edit saved runs; a
new run gets a new file. CI re-scores saved runs (tests/test_evaluation.py).
"""

import argparse
import inspect
import json
import platform
import subprocess
import sys
import time
from datetime import date
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from estate_agent import config  # noqa: E402
from estate_agent.agent import SYSTEM_PROMPT, ask  # noqa: E402
from estate_agent.evaluation import LATEST_SCORER, load_cases, score_run  # noqa: E402
from estate_agent.presentation import default_agent  # noqa: E402

DEFAULT_CASES = ROOT / "eval" / "cases-v1.json"
RUNS = ROOT / "eval" / "runs"


def git_state() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    # New run files from earlier repeats of the same code don't count as changes.
    changes = git("status", "--porcelain", "--", ".", ":(exclude)eval/runs")
    return {"commit": git("rev-parse", "HEAD"), "uncommitted_changes": bool(changes)}


# Multi-turn cases list earlier questions under "context". They are asked first,
# in order, in the same conversation; the case is scored on the final answer. An
# agent without conversation memory answers each question on its own.
AGENT_HAS_MEMORY = "history" in inspect.signature(ask).parameters


def _turn(result, question: str, seconds: float) -> dict:
    return {"question": question, "answer": result.answer, "steps": result.steps,
            "grounding": result.grounding, "checks": result.checks, "revisions": result.revisions,
            "seconds": round(seconds, 1)}


def run_case(agent, case: dict) -> dict:
    start = time.time()
    questions = case.get("context", []) + [case["question"]]
    turns, history = [], []
    try:
        for question in questions:
            turn_start = time.time()
            result = ask(agent, question, history=history) if AGENT_HAS_MEMORY else ask(agent, question)
            history.append(result)
            turns.append(_turn(result, question, time.time() - turn_start))
    except Exception as error:  # record the failure; it scores as a fail
        return {"id": case["id"], "question": case["question"], "answer": "", "steps": [],
                "grounding": {}, "checks": [], "error": str(error), "seconds": round(time.time() - start, 1),
                **({"turns": turns} if case.get("context") else {})}
    record = {"id": case["id"], **turns[-1], "seconds": round(time.time() - start, 1)}
    if case.get("context"):
        record["turns"] = turns
        record["memory"] = AGENT_HAS_MEMORY
    return record


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES, help="frozen case set to run")
    parser.add_argument("--only", nargs="+", help="case IDs to run; nothing is saved")
    args = parser.parse_args(argv)

    cases = load_cases(args.cases)
    if args.only:
        cases = [c for c in cases if c["id"] in set(args.only)]
    try:
        agent = default_agent()
    except RuntimeError as error:
        print(error)
        return 1

    code = git_state()
    if code["uncommitted_changes"] and not args.only:
        print("Commit your changes first: a recorded run must point to the exact code it ran.")
        return 1
    run = {
        "cases_file": args.cases.name,
        "git_commit": code["commit"],
        "scorer_version": LATEST_SCORER,
        "date": date.today().isoformat(),
        "provider": config.provider(),
        "model": config.model_name(),
        "temperature": 0,
        "system_prompt": SYSTEM_PROMPT,
        "versions": {p: version(p) for p in ("langgraph", "langchain-core", "langchain-ollama", "chromadb")}
                    | {"python": platform.python_version()},
        "results": [],
    }
    for case in cases:
        record = run_case(agent, case)
        run["results"].append(record)
        print(f"{case['id']:>3} {record['seconds']:5.1f}s  {len(record['steps'])} tool calls"
              + (f"  ERROR {record['error']}" if record.get("error") else ""), flush=True)

    scores, summary = score_run(cases, run, LATEST_SCORER)
    run["scores"], run["summary"] = scores, summary
    for score in scores:
        mark = "PASS" if score["passed"] else "FAIL"
        print(f"{mark} {score['id']:>3}  " + "; ".join(score["problems"]))
    print(json.dumps(summary, indent=1))

    if not args.only:
        RUNS.mkdir(exist_ok=True)
        model = run["model"].replace(":", "-").replace("/", "-")
        # Repeat runs of the same code get -2, -3, ... (never overwrite a record).
        case_set = args.cases.stem.removeprefix("cases-")
        stem = f"{run['date']}-{model}-{code['commit'][:7]}" + ("" if case_set == "v1" else f"-{case_set}")
        path, repeat = RUNS / f"{stem}.json", 1
        while path.exists():
            repeat += 1
            path = RUNS / f"{stem}-{repeat}.json"
        path.write_text(json.dumps(run, indent=1, ensure_ascii=False) + "\n")
        print(f"Saved {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
