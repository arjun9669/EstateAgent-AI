"""Settings from the environment, with an optional .env file in the project root.

LLM_PROVIDER chooses the chat model:
  ollama  a model served by the Ollama app (the default). Either a local model
          (e.g. qwen3:4b, downloaded with `ollama pull`) or an Ollama cloud model
          (e.g. gemma4:cloud), which runs on Ollama's servers after `ollama signin`.
          No API key in this project either way.
          On a server without the Ollama app (e.g. Streamlit Community Cloud), set
          OLLAMA_BASE_URL=https://ollama.com, OLLAMA_API_KEY and a direct model
          name such as gemma4:31b.
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

    api_key = os.environ.get("OLLAMA_API_KEY")
    return ChatOllama(
        model=model_name(),
        base_url=ollama_url(),
        client_kwargs={"headers": {"Authorization": f"Bearer {api_key}"}} if api_key else {},
        temperature=0,
        num_ctx=OLLAMA_CONTEXT_TOKENS,
        reasoning=False,  # skip "thinking" output; the agent loop does the planning
    )


def ollama_url() -> str:
    return os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL).rstrip("/")


def uses_ollama_api() -> bool:
    """True when calling ollama.com directly instead of the local Ollama app."""
    return ollama_url().startswith("https://ollama.com")


def is_ollama_cloud_model(name: str) -> bool:
    """Cloud models are named like gemma4:cloud or gemma4:31b-cloud."""
    return name.endswith((":cloud", "-cloud"))


def check_ollama() -> str | None:
    """None if Ollama is running and has the model; otherwise what to do about it.

    Cloud models need no download, so only the server is checked for them; a
    missing sign-in shows up as an error on the first request.
    """
    import ollama

    if uses_ollama_api():
        return None if os.environ.get("OLLAMA_API_KEY") else (
            "OLLAMA_API_KEY is not set. Create one at https://ollama.com/settings/keys "
            "and add it as a secret, never in the repository."
        )
    client = ollama.Client(host=ollama_url())
    try:
        installed = {m.model for m in client.list().models}
    except Exception:
        return (
            f"Ollama is not running at {ollama_url()}. On your computer: install it from "
            "https://ollama.com and open the Ollama app. On a hosted app: set "
            "OLLAMA_BASE_URL=https://ollama.com and OLLAMA_API_KEY in its secrets."
        )
    wanted = model_name()
    if is_ollama_cloud_model(wanted):
        return None
    if wanted not in installed and f"{wanted}:latest" not in installed:
        return f"Model {wanted!r} is not downloaded. Run: ollama pull {wanted}"
    return None
