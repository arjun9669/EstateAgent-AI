# Hosting the demo for free on Streamlit Community Cloud

Status: **ready to deploy; not deployed yet.** The Streamlit page
(`streamlit_app.py`) is tested and runs locally with `streamlit run streamlit_app.py`.

## Why Streamlit Community Cloud

Hugging Face Spaces was the first plan, but creating a Gradio Space on the free
CPU tier now fails with "402 Payment Required": Gradio and Docker Spaces need a
PRO subscription, and only static Spaces stay free
([Hugging Face docs](https://huggingface.co/docs/huggingface_hub/en/guides/manage-spaces)).
Render's free tier has 512 MB of memory, too little for ChromaDB, the embedding
model and a web app together. Streamlit Community Cloud runs Python apps from a
public GitHub repository for free, with roughly 690 MB to 2.7 GB of memory
([limits as of February 2024](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app);
Streamlit says they can change).

The Gradio app (`app.py`) is kept for local use; both share the agent, tools and
`presentation.py`.

## What runs where

```
visitor's browser ──▶ Streamlit Community Cloud (free)          ──▶ Ollama cloud API
                       streamlit_app.py: page, LangGraph agent,      gemma4:31b
                       the five tools, ChromaDB + MiniLM             (the only part
                       over the committed data                        that is an LLM)
```

Everything except the language model runs on Streamlit's servers. The model is
called over Ollama's hosted API with your key.

## Limits that shape the design

| Limit | Effect |
|---|---|
| Ollama free plan: a monthly "starter amount" of cloud usage, 1 concurrent request ([pricing](https://ollama.com/pricing)) | Questions from all visitors run one at a time (a shared lock), and each browser session gets at most 10 questions (`MAX_QUESTIONS_PER_SESSION` in `web.py`) |
| Streamlit apps sleep when unused | The first visitor after a quiet period waits for it to wake up |
| No persistent disk | MiniLM (about 80 MB) downloads again after each restart, on the first question |
| Direct API calls use model names like `gemma4:31b`, not `gemma4:cloud` ([Ollama API auth](https://docs.ollama.com/api/authentication.md)) | Set `OLLAMA_MODEL=gemma4:31b` in the app's secrets. Ollama's hosted model list includes `gemma4:31b` (checked 24 Oct 2026) |

## Settings

Streamlit secrets are TOML; root-level secrets are also exposed as environment
variables ([Streamlit docs](https://docs.streamlit.io/develop/concepts/connections/secrets-management)),
which is what `config.py` reads:

```toml
LLM_PROVIDER = "ollama"
OLLAMA_BASE_URL = "https://ollama.com"
OLLAMA_MODEL = "gemma4:31b"
OLLAMA_API_KEY = "paste your key here, in Streamlit's settings only"
```

`config.make_model()` sends the key as `Authorization: Bearer …`. The key lives
only in Streamlit's secret store, never in the repository.

## Steps (yours)

1. Create an Ollama API key at [ollama.com/settings/keys](https://ollama.com/settings/keys).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and
   allow Streamlit to read your repositories.
3. **Create app → Deploy a public app from GitHub**:
   repository `arjun9669/EstateAgent-AI`, branch `main`, main file `streamlit_app.py`.
   Pick an app URL, for example `estateagent-ai`.
4. Open **Advanced settings**: choose **Python 3.12** and paste the TOML above into
   **Secrets**, with your real key.
5. **Deploy.** Streamlit installs `requirements.txt` (exact versions) and starts the
   app; later pushes to `main` redeploy it automatically.

## Not verified yet

- The hosted-API path (`OLLAMA_BASE_URL=https://ollama.com` with a key) is covered
  by unit tests but has not been called for real; locally the app uses the
  signed-in Ollama app instead.
- How much of Ollama's monthly free usage a public demo consumes.
- Visitors' questions are sent to Ollama's cloud; the page says questions leave
  the app. Ollama states it does not train on cloud prompts.
