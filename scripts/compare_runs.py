"""Compare two recorded evaluation runs case by case.

    python scripts/compare_runs.py eval/runs/OLD.json eval/runs/NEW.json
"""

import json
import sys
from pathlib import Path

DIMENSIONS = ("tool_choice", "correctness", "citations", "grounding")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    old, new = (json.loads(Path(p).read_text()) for p in argv)
    old_scores = {s["id"]: s for s in old["scores"]}
    print(f"{'':6}{Path(argv[0]).name:>34}  {Path(argv[1]).name:>34}")
    for key in ("passed", *DIMENSIONS):
        print(f"{key:<12}{old['summary'][key]:>28}/{old['summary']['cases']}  {new['summary'][key]:>32}/{new['summary']['cases']}")
    changed = [(s, old_scores[s["id"]]) for s in new["scores"] if s["passed"] != old_scores[s["id"]]["passed"]]
    print("\nChanged cases:" if changed else "\nNo case changed pass/fail.")
    for now, before in changed:
        arrow = "FAIL -> PASS" if now["passed"] else "PASS -> FAIL"
        print(f"  {now['id']:>3} {arrow}  {'; '.join(now['problems'] or before['problems'])}")
    still = [s for s in new["scores"] if not s["passed"] and not old_scores[s["id"]]["passed"]]
    for score in still:
        print(f"  {score['id']:>3} still FAIL  {'; '.join(score['problems'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
