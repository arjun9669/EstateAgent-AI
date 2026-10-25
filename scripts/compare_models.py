"""Compare models on the frozen evaluation sets, from recorded runs.

    python scripts/compare_models.py            # writes eval/model-comparison.json

Uses every recorded run made with the agent code of MODEL_RUNS' commits, so all
models are compared on the same agent, tools, prompt and latest scorer.
"""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from estate_agent.evaluation import DIMENSIONS, LATEST_SCORER, cases_for_run, field_name_leaks, score_run  # noqa: E402

RUNS = ROOT / "eval" / "runs"
OUTPUT = ROOT / "eval" / "model-comparison.json"
# The agent code (prompt, tools, graph) is identical at these commits; c8a8d8b
# changed only the scorer and docs since 9e71263. Runs made at them are comparable.
AGENT_COMMITS = ("9e71263", "c8a8d8b")
SETS = ("cases-v1.json", "cases-v2.json", "cases-v3.json")


def model_runs() -> dict[str, list[tuple[Path, dict]]]:
    by_model: dict[str, list[tuple[Path, dict]]] = {}
    for path in sorted(RUNS.glob("*.json")):
        run = json.loads(path.read_text())
        if run.get("git_commit", "")[:7] in AGENT_COMMITS:
            by_model.setdefault(f"{run['provider']}:{run['model']}", []).append((path, run))
    return by_model


def compare() -> dict:
    report = {"scorer_version": LATEST_SCORER, "agent_commits": AGENT_COMMITS, "models": {}}
    for model, runs in sorted(model_runs().items()):
        entry = {"sets": {}, "answers_with_field_names": 0, "answers": 0, "errors": 0, "seconds": []}
        for path, run in runs:
            cases = cases_for_run(run)
            _, summary = score_run(cases, run, LATEST_SCORER)
            name = run.get("cases_file", "cases-v1.json")
            s = entry["sets"].setdefault(name, {"cases": len(cases), "runs": [], **{d: [] for d in DIMENSIONS}})
            s["runs"].append(summary["passed"])
            for d in DIMENSIONS:
                s[d].append(summary[d])
            for r in run["results"]:
                entry["answers"] += 1
                entry["errors"] += bool(r.get("error"))
                entry["answers_with_field_names"] += bool(field_name_leaks(r["answer"]))
                entry["seconds"].append(r["seconds"])
        seconds = entry.pop("seconds")
        entry["median_seconds_per_question"] = round(statistics.median(seconds), 1)
        entry["max_seconds_per_question"] = round(max(seconds), 1)
        report["models"][model] = entry
    return report


def main() -> int:
    report = compare()
    OUTPUT.write_text(json.dumps(report, indent=1) + "\n")
    for model, entry in report["models"].items():
        sets = ", ".join(f"{k.removeprefix('cases-').removesuffix('.json')} {v['runs']}/{v['cases']}"
                         for k, v in sorted(entry["sets"].items()))
        print(f"{model}: {sets}; median {entry['median_seconds_per_question']}s; "
              f"field names {entry['answers_with_field_names']}/{entry['answers']}; errors {entry['errors']}")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
