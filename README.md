# EstateAgent AI

**An agentic RAG assistant for the Dubai property market.** Ask a question in
plain language; an AI agent decides which tools to use (listing search, market
statistics, mortgage maths, official rules), runs them, and answers with sources.

> *"Find 2-bedroom apartments in Dubai Marina under AED 170k a year, compare
> them with the area's typical prices, and explain how rent increases are regulated."*

🚧 **Status: milestone 3 of 6.** The five tools, semantic search over official
Dubai tenancy law and the LangGraph agent are built and tested. The chat
interface and live demo come next. See [PLAN.md](PLAN.md).

## Why agentic?

A plain RAG chatbot retrieves some text and hopes the answer is in it. Real
property questions mix several kinds of work: filtering listings, computing
statistics, doing loan maths and reading regulations. Here the language model
only **plans and explains**. Every number comes from a deterministic tool, so
figures can be checked rather than trusted.

| Tool | Example question | Backed by |
|---|---|---|
| `search_listings` | "2BR in JVC under 90k/year, furnished" | SQLite listings table |
| `compare_properties` | "Compare L0435 and L0503" | Price per sqft against similar listings |
| `area_price_stats` | "Are Dubai Marina prices rising?" | Median sale prices, this year vs. last |
| `mortgage_calculator` | "Monthly payment on AED 1.8M?" | UAE Central Bank loan-to-value caps, DLD fees |
| `search_rules` | "How much can my landlord raise the rent?" | Semantic search (MiniLM embeddings, ChromaDB) over official Dubai legislation |

Each tool returns JSON with a `source` field, and errors come back as messages
the agent can read and correct, not crashes. The JSON schemas for LLM tool
calling are in [`tools/__init__.py`](src/estate_agent/tools/__init__.py), and a
test keeps them in sync with the Python functions.

## How the agent works

Built with [LangGraph](https://github.com/langchain-ai/langgraph)
([`agent.py`](src/estate_agent/agent.py)). The model runs locally through
[Ollama](https://ollama.com) with no API key, or on Groq's free cloud API:

```
question ─▶ agent ──tool calls──▶ tools ──results──▶ agent   (up to 6 rounds)
              │  └── tool limit hit ──▶ force_answer ──┐
              ▼ answer                                  ▼
            check ◀─────────────────────────────────────┘
              ├── figures not in evidence (once) ──▶ agent
              ▼
            answer + tool steps + grounding report
```

1. The model plans and calls tools, as many rounds as the question needs. Tool
   errors go back to it as messages, so it can fix its arguments.
2. If it keeps calling tools past the limit, it is made to answer with what it has.
3. A **grounding check** ([`grounding.py`](src/estate_agent/grounding.py)) compares
   every number and ID in the answer with the tool results and the question.
   "AED 2.3 million" passes if a tool returned 2,277,500; "AED 2.5 million" does
   not. The model gets one chance to fix anything unsupported; whatever is still
   unsupported is shown as a warning next to the answer.

## Try the tools

Requires Python 3.11+.

```bash
git clone https://github.com/arjun9669/EstateAgent-AI.git
cd EstateAgent-AI
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                      # offline: uses a small stand-in embedder
RUN_MODEL_TESTS=1 pytest -m model   # real retrieval checks; downloads the ~80 MB model once
```

```python
from estate_agent.tools import Toolbox

tools = Toolbox.load()
tools.run("search_listings", {"purpose": "rent", "area": "marina", "bedrooms": 2, "max_price": 170000})
tools.run("area_price_stats", {"area": "Dubai Marina"})
# -> median AED 2,139/sqft over the last 12 months, +5.3% on the year before (sample data)
tools.run("mortgage_calculator", {"property_price": 1_800_000, "annual_rate_pct": 4.5})
# -> 20% minimum down payment for an expat's first home, AED 8,004/month over 25 years
tools.run("search_rules", {"query": "How much can my landlord increase the rent when I renew?"})
# -> decree-43-2013#art1 "Percentages of Increase": 0-20% depending on how far
#    the rent is below the RERA Rental Index average
```

`search_rules` returns whole articles with IDs like `law-26-2007#art14`, so the
agent can cite them, and returns nothing when no article is similar enough
(for example, "What is the capital of Mars?").

## Ask the agent

**With Ollama, free, no API key in the project (default).** Install
[Ollama](https://ollama.com/download) and open it, then pick one:

- **Ollama cloud model (fast on any computer):** run `ollama signin` once and set
  `OLLAMA_MODEL=gemma4:cloud` in `.env`. It runs on Ollama's servers with tool
  calling and no download; the free plan includes a monthly starter amount of
  usage and one request at a time.
- **Local model (offline, fully private):** `ollama pull qwen3:4b` (about 2.5 GB).
  Small enough for an 8 GB Mac, but slower and a weaker planner; with 16 GB or
  more, `qwen3:8b` does better.

Then ask:

```bash
python scripts/ask.py "Find 2-bedroom apartments in Dubai Marina under AED 170k a year and explain how rent increases are regulated"
```

**On Groq's free cloud API:** create a key at [console.groq.com](https://console.groq.com),
`cp .env.example .env`, set `LLM_PROVIDER=groq` and `GROQ_API_KEY=...`. `.env` is
git-ignored. Groq's `llama-3.3-70b-versatile` is stronger than a small local
model; the free tier has rate limits.

The script prints each tool call, the answer, and a warning if any figure
could not be verified. See a [sample run on six questions](docs/SAMPLE_RUN.md)
with `gemma4:cloud`. Small local models make more planning mistakes; the
grounding check is there to catch the figures they get wrong.

The agent tests use a scripted stand-in model, so `pytest` needs no model or key.

## Project structure

```
src/estate_agent/
  areas.py          Dubai communities and aliases (JVC, JLT, JBR…)
  data.py           Loads CSVs into in-memory SQLite
  agent.py          LangGraph agent: plan, call tools, answer, check
  grounding.py      Checks every figure and ID in an answer against the evidence
  config.py         .env loading; Ollama (local) or Groq (cloud) model
  dld.py            Converts the Dubai Land Department transactions export
  tools/            The five agent tools and their schemas
scripts/
  ask.py                    Ask the agent from the command line
  generate_sample_data.py   Reproducible synthetic data (CI checks it)
  fetch_rules.py            Official legislation -> article text
  load_dld_transactions.py  Your DLD download -> transactions.csv
data/sample/        Synthetic listings and transactions
data/rules/         Dubai tenancy legislation, one section per article
docs/DATA_SOURCES.md
```

## Data

- **Legislation:** Law No. 26 of 2007 (consolidated with Law No. 33 of 2008)
  and Decree No. 43 of 2013, from the official Dubai Legislation Portal.
- **Listings:** **synthetic**. Area names are real; every listing, price and ID
  is invented.
- **Sales transactions:** synthetic by default. A loader converts the Dubai Land
  Department open dataset if you download it yourself; it is not redistributed
  here because its licence is unspecified.

Details, sources and known gaps: [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).

## Roadmap

1. ✅ Tools, sample data, tests, CI
2. ✅ Official legislation, embeddings + ChromaDB for `search_rules`, DLD loader
3. ✅ LangGraph agent on a local Ollama model or Groq's free API, with checks that answers match tool output
4. Gradio chat showing each agent step, hosted free on Hugging Face Spaces
5. Evaluation set for tool choice, correctness and citations
6. Demo GIF, architecture diagram, release

Built to run entirely on free tiers.

## Disclaimer

For information and learning only, not legal, financial or investment advice.
Mortgage rules change and lenders may be stricter; check with a lender and the
official sources.

## License

[MIT](LICENSE) © Arjun Kumar
