# Hosting the demo for free on Hugging Face Spaces

Status: **planned, not deployed.** The app runs locally (`python app.py`); this is
the plan for a public demo link at no cost.

## What runs where

```
visitor's browser ──▶ Hugging Face Space (free CPU)            ──▶ Ollama cloud API
                       app.py: Gradio UI, LangGraph agent,           gemma4:31b
                       the five tools, ChromaDB + MiniLM             (the only part
                       over the committed data                        that is an LLM)
```

Everything except the language model runs inside the Space. The model is called
over Ollama's hosted API, because a Space has no Ollama app and the free CPU is
too slow for a useful local model.

## Free limits that shape the design

| Limit | Value | Source | Effect |
|---|---|---|---|
| Space hardware | CPU basic: 2 vCPU, 16 GB RAM, 50 GB disk that is not persistent | [Spaces overview](https://huggingface.co/docs/hub/spaces-overview) | Enough for the tools, ChromaDB and MiniLM; not for running an LLM |
| Space sleep | Paused after 48 h without visitors; first visit then waits for a restart | Spaces docs | Acceptable for a portfolio; the README says so |
| Ollama free plan | A monthly "starter amount" of cloud usage; 1 concurrent request, others queued | [ollama.com/pricing](https://ollama.com/pricing) | The app queues one question at a time (`default_concurrency_limit=1`). Heavy traffic could use up the month's quota |
| Model name | Direct API calls use names like `gemma4:31b`, not `gemma4:cloud` | [Ollama API authentication](https://docs.ollama.com/api/authentication.md) | Set `OLLAMA_MODEL=gemma4:31b` on the Space |

## Configuration on the Space

| Name | Kind | Value |
|---|---|---|
| `LLM_PROVIDER` | Variable | `ollama` |
| `OLLAMA_BASE_URL` | Variable | `https://ollama.com` |
| `OLLAMA_MODEL` | Variable | `gemma4:31b` |
| `OLLAMA_API_KEY` | **Secret** | Created by you at [ollama.com/settings/keys](https://ollama.com/settings/keys) |

`config.make_model()` sends the key as `Authorization: Bearer …` when
`OLLAMA_API_KEY` is set, and skips the local "is Ollama running" check when the
base URL is `https://ollama.com`. The key lives only in the Space's secret store,
never in the repository. Groq works the same way (`LLM_PROVIDER=groq`,
`GROQ_API_KEY` secret) if Ollama's quota is not enough.

## Getting the code onto the Space

A Space is its own git repository and reads its settings from YAML at the top
of its `README.md`. Putting that YAML in this repository's README would show up
as a table on GitHub, so the plan is:

1. `deploy/space/README.md` holds the Space README with its YAML settings.
2. A GitHub Actions workflow, on each push to `main` after tests pass, copies
   the app (`app.py`, `requirements.txt`, `src/`, `data/sample/`, `data/rules/`)
   plus `deploy/space/README.md` as `README.md` to the Space's repository.
3. It authenticates with a Hugging Face **write token** stored as a GitHub
   Actions secret (`HF_TOKEN`), which you create.

The Space installs `requirements.txt` (exact versions) and runs `app.py`.
MiniLM (about 80 MB) downloads on the first question after each restart.

## What you need to do (when ready)

1. Create a free account at [huggingface.co](https://huggingface.co) and an
   empty Space named `EstateAgent-AI` with the **Gradio** SDK and **CPU basic** hardware.
2. In the Space's **Settings → Variables and secrets**, add the four values above.
   Create the Ollama API key yourself; do not paste it into chat or code.
3. Create a Hugging Face access token with **write** access and add it to this
   GitHub repository under **Settings → Secrets and variables → Actions** as `HF_TOKEN`.

Then the sync workflow can be added and the first deployment checked.

## Open questions

- How much of Ollama's monthly free usage a public demo consumes. If it runs out,
  the chat shows the model error; a fallback to the recorded
  [sample run](SAMPLE_RUN.md) would keep the page useful.
- Whether to add a per-visitor limit (for example, a few questions per session)
  to protect the quota.
- Visitors' questions are sent to Ollama's cloud. Ollama states it does not
  train on cloud prompts; the page should say questions leave the Space.
