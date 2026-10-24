"""Settings from the environment, with an optional .env file in the project root.

LLM_PROVIDER chooses the chat model:
  ollama  a free local model through Ollama (no key; the default)
  groq    Groq's free cloud API (needs GROQ_API_KEY)
"""

import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
PROVIDERS = ("ollama", "groq")
DEFAULT_PROVIDER = "ollama"
DEFAULT_MODELS = {"ollama": "qwen3:4b", "groq": "llama-3.3-70b-versatile"}
DEFAULT_OLLAMA_URL = "http://localhost:11434"
# Ollama's default context window is too small for tool results plus the prompt.
OLLAMA_CONTEXT_TOKENS = 8192


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


def provider() -> str:
    name = os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    if name not in PROVIDERS:
        raise RuntimeError(f"LLM_PROVIDER must be one of {', '.join(PROVIDERS)}; got {name!r}")
    return name


def model_name() -> str:
    name = provider()
    return os.environ.get(f"{name.upper()}_MODEL") or DEFAULT_MODELS[name]


def make_model():
    """The chat model for the configured provider."""
    if provider() == "groq":
        from langchain_groq import ChatGroq

        if not os.environ.get("GROQ_API_KEY"):
            raise RuntimeError(
                "GROQ_API_KEY is not set. Create a free key at console.groq.com and put "
                "GROQ_API_KEY=... in the project's .env file (it is git-ignored), "
                "or use LLM_PROVIDER=ollama for a local model with no key."
            )
        return ChatGroq(model=model_name(), temperature=0, max_retries=2)

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model_name(),
        base_url=os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL),
        temperature=0,
        num_ctx=OLLAMA_CONTEXT_TOKENS,
        reasoning=False,  # skip "thinking" output; the agent loop does the planning
    )


def check_ollama() -> str | None:
    """None if Ollama is running and has the model; otherwise what to do about it."""
    import ollama

    client = ollama.Client(host=os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL))
    try:
        installed = {m.model for m in client.list().models}
    except Exception:
        return "Ollama is not running. Install it from https://ollama.com and open the Ollama app."
    wanted = model_name()
    if wanted not in installed and f"{wanted}:latest" not in installed:
        return f"Model {wanted!r} is not downloaded. Run: ollama pull {wanted}"
    return None
