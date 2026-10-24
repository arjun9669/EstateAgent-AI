"""Settings from the environment, with an optional .env file in the project root."""

import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
DEFAULT_MODEL = "llama-3.3-70b-versatile"


def load_env(path: Path | None = None) -> None:
    """Read KEY=VALUE lines into os.environ without overriding existing variables."""
    path = path or ENV_FILE
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def groq_model_name() -> str:
    return os.environ.get("GROQ_MODEL", DEFAULT_MODEL)


def make_groq_model():
    """Chat model on Groq's API. Needs GROQ_API_KEY (see README)."""
    from langchain_groq import ChatGroq

    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Create a free key at console.groq.com and put "
            "GROQ_API_KEY=... in the project's .env file (it is git-ignored)."
        )
    return ChatGroq(model=groq_model_name(), temperature=0, max_retries=2)
