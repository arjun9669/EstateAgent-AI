"""Streamlit entry point (Streamlit Community Cloud runs this file).

    streamlit run streamlit_app.py

See docs/HOSTING.md. The Gradio version for local use is app.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from estate_agent.web import main  # noqa: E402

main()
