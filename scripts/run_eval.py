"""Run the frozen evaluation set once against the configured model and record it.

    python scripts/run_eval.py                 # writes eval/runs/<date>-<model>.json
    python scripts/run_eval.py --only R1 R2    # a few cases, printed but not saved

A saved run is a permanent record: answers, every tool call and result, the
grounding report, timings and scores. Do not overwrite or edit saved runs; a
new run gets a new file. CI re-scores saved runs (tests/test_evaluation.py).
"""

import argparse
import json
import platform
import sys
import time
from datetime import date
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from estate_agent import config  # noqa: E402
from estate_agent.agent import SYSTEM_PROMPT, ask  # noqa: E402
from estate_agent.evaluation import load_cases, score_run  # noqa: E402
from estate_agent.presentation import default_agent  # noqa: E402

CASES = ROOT / "eval" / "cases-v1.json"
RUNS = ROOT / "eval" / "runs"


def run_case(agent, case: dict) -> dict:
    start = time.time()
    try:
        result = ask(agent, case["question"])
    except Exception as error:  # record the failure; it scores as a fail
        return {"id": case["id"], "question": case["question"], "answer": "", "steps": [],
                "grounding": {}, "checks": [], "error": str(error), "seconds": round(time.time() - start, 1)}
    return {
        "id": case["id"],
        "question": case["question"],
        "answer": result.answer,
        "steps": result.steps,
        "grounding": result.grounding,
        "checks": result.checks,
        "revisions": result.revisions,
        "seconds": round(time.time() - start, 1),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", help="case IDs to run; nothing is saved")
    args = parser.parse_args(argv)

    cases = load_cases(CASES)
    if args.only:
        cases = [c for c in cases if c["id"] in set(args.only)]
    try:
        agent = default_agent()
    except RuntimeError as error:
        print(error)
        return 1

    run = {
        "cases_file": CASES.name,
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

    scores, summary = score_run(cases, run)
    run["scores"], run["summary"] = scores, summary
    for score in scores:
        mark = "PASS" if score["passed"] else "FAIL"
        print(f"{mark} {score['id']:>3}  " + "; ".join(score["problems"]))
    print(json.dumps(summary, indent=1))

    if not args.only:
        RUNS.mkdir(exist_ok=True)
        path = RUNS / f"{run['date']}-{run['model'].replace(':', '-').replace('/', '-')}.json"
        if path.exists():
            print(f"{path.name} already exists; not overwriting a recorded run.")
            return 1
        path.write_text(json.dumps(run, indent=1, ensure_ascii=False) + "\n")
        print(f"Saved {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
