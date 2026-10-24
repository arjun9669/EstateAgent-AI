"""Summarize variation across repeated evaluation runs of the same code.

    python scripts/variation.py eval/runs/A.json eval/runs/B.json eval/runs/C.json

Scores every run with the latest scorer, then reports the spread of totals and
which cases did not give the same result every time.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from estate_agent.evaluation import DIMENSIONS, LATEST_SCORER, load_cases, score_run  # noqa: E402


def variation(paths: list[Path]) -> dict:
    cases = load_cases(ROOT / "eval" / "cases-v1.json")
    runs = [json.loads(p.read_text()) for p in paths]
    scored = [score_run(cases, run, LATEST_SCORER) for run in runs]
    totals = {key: [summary[key] for _, summary in scored] for key in ("passed", *DIMENSIONS)}
    passes = {c["id"]: [next(s for s in scores if s["id"] == c["id"])["passed"] for scores, _ in scored]
              for c in cases}
    return {
        "scorer_version": LATEST_SCORER,
        "runs": [p.name for p in paths],
        "commits": sorted({run.get("git_commit", "?")[:7] for run in runs}),
        "totals": {key: {"values": v, "min": min(v), "max": max(v)} for key, v in totals.items()},
        "always_pass": sum(all(v) for v in passes.values()),
        "never_pass": [cid for cid, v in passes.items() if not any(v)],
        "unstable": {cid: "".join("P" if x else "F" for x in v) for cid, v in passes.items() if 0 < sum(v) < len(v)},
    }


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    print(json.dumps(variation([Path(a) for a in argv]), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
