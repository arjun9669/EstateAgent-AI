"""Run the chat interface: python app.py, then open http://127.0.0.1:7860.

Hugging Face Spaces runs this file directly. See docs/HOSTING.md.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from estate_agent.ui import build_app  # noqa: E402

# One request at a time: Ollama's free cloud plan allows a single concurrent request.
demo = build_app().queue(default_concurrency_limit=1, max_size=20)

if __name__ == "__main__":
    demo.launch()
