"""Re-score every recorded run with the latest scorer, next to its original score.

    python scripts/rescore_runs.py          # writes eval/rescored-latest.json

Recorded run files are never changed. A test checks that the committed output
matches what this script produces, so it cannot drift from the scorer.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from estate_agent.evaluation import LATEST_SCORER, cases_for_run, score_run  # noqa: E402

RUNS = ROOT / "eval" / "runs"
OUTPUT = ROOT / "eval" / "rescored-latest.json"


def rescore() -> dict:
    report = {"scorer_version": LATEST_SCORER, "runs": []}
    for path in sorted(RUNS.glob("*.json")):
        run = json.loads(path.read_text())
        cases = cases_for_run(run)
        original_version = run.get("scorer_version", "1.0")
        original_scores, original = score_run(cases, run)
        new_scores, new = score_run(cases, run, LATEST_SCORER)
        changed = [
            {"id": new_s["id"], "was": "PASS" if old_s["passed"] else "FAIL",
             "now": "PASS" if new_s["passed"] else "FAIL", "problems_now": new_s["problems"]}
            for old_s, new_s in zip(original_scores, new_scores) if old_s["passed"] != new_s["passed"]
        ]
        report["runs"].append({
            "run": path.name,
            "cases_file": run.get("cases_file", "cases-v1.json"),
            "cases": len(cases),
            "original_scorer": original_version,
            "original": {"passed": original["passed"], "citations": original["citations"]},
            "latest": {"passed": new["passed"], "citations": new["citations"]},
            "changed_cases": changed,
        })
    return report


def main() -> int:
    report = rescore()
    OUTPUT.write_text(json.dumps(report, indent=1) + "\n")
    for run in report["runs"]:
        n = run["cases"]
        print(f"{run['run']}: scorer {run['original_scorer']} {run['original']['passed']}/{n} -> "
              f"{report['scorer_version']} {run['latest']['passed']}/{n}  "
              + ", ".join(f"{c['id']} {c['was']}->{c['now']}" for c in run["changed_cases"]))
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
