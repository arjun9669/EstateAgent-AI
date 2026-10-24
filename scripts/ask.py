"""Ask the agent a question from the command line.

    python scripts/ask.py "Find 2-bedroom apartments in Dubai Marina under AED 170k a year"

Needs GROQ_API_KEY in the environment or in the project's .env file.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from estate_agent import config  # noqa: E402
from estate_agent.agent import ask, build_agent  # noqa: E402
from estate_agent.tools import Toolbox  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    config.load_env()
    try:
        model = config.make_groq_model()
    except RuntimeError as e:
        print(e)
        return 1

    graph = build_agent(model, Toolbox.load())
    result = ask(graph, argv[0])
    for number, step in enumerate(result.steps, 1):
        status = "ok" if step["ok"] else f"error: {step['error']}"
        print(f"[{number}] {step['tool']}({json.dumps(step['arguments'])}) -> {status}")
    print()
    print(result.answer)
    if result.warning:
        print(f"\n⚠️  {result.warning}")
    print(f"\n(model: {config.groq_model_name()}, revisions: {result.revisions})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
