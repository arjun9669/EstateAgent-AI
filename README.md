# EstateAgent AI

**An agentic RAG assistant for the Dubai property market.** Ask a question in
plain language; an AI agent decides which tools to use (listing search, market
statistics, mortgage maths, official rules), runs them, and answers with sources.

> *"Find 2-bedroom apartments in Dubai Marina under AED 170k a year, compare
> them with the area's typical prices, and explain how rent increases are regulated."*

**[▶ Try the live demo](https://appapppy-hadkfldjgvu3gw24nfdysg.streamlit.app/)** (free hosting; the first visit after a quiet
period takes a moment to wake up, and each visitor gets 10 questions).

🚧 **Status: milestone 5 of 6 done.** The tools, semantic search over official Dubai
tenancy law, the LangGraph agent and the chat interface are built, tested and
[hosted free on Streamlit Community Cloud](docs/HOSTING.md). On a frozen
30-question evaluation, **27/30 pass**, with tool choice, correctness and grounding
all 30/30 ([results](docs/EVALUATION.md)). Polish comes next. See [PLAN.md](PLAN.md).

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
| `search_rules` | "How much can my landlord raise the rent?" | Hybrid search over official Dubai legislation: MiniLM embeddings in ChromaDB plus BM25 keywords |

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

## Evaluation

A frozen set of 30 questions ([`eval/cases-v1.json`](eval/cases-v1.json)) scores
each answer on tool choice, correctness, citation support and grounding, with
`gemma4:cloud`:

| Run | Tool choice | Correctness | Citations | Grounding | All four |
|---|---:|---:|---:|---:|---:|
| 1: meaning-only law search | 30/30 | 30/30 | 27/30 | 30/30 | **27/30** |
| 2: hybrid search + quotation rule | 30/30 | 28/30 | 29/30 | 30/30 | **27/30** |

Run 2 fixed all three run-1 failures (a law-search miss and two loose quotations;
law questions now 7/7) but failed three others on completeness and style.
Every result is recorded and explained, not just the best one:
[docs/EVALUATION.md](docs/EVALUATION.md).

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

**Chat interface:** `streamlit run streamlit_app.py` (the hosted version), or
`python app.py` for the Gradio version at http://127.0.0.1:7860. Each tool call
appears as a step you can open to see its arguments and full result, followed by
the grounding check and the answer.

The command-line script prints each tool call, the answer, and a warning if any figure
could not be verified. See a [sample run on six questions](docs/SAMPLE_RUN.md)
with `gemma4:cloud`. Small local models make more planning mistakes; the
grounding check is there to catch the figures they get wrong.

The agent tests use a scripted stand-in model, so `pytest` needs no model or key.

## Project structure

```
streamlit_app.py    Streamlit chat page (the hosted demo)
app.py              Gradio chat app (local alternative)
src/estate_agent/
  web.py            Streamlit page: streams each agent step, check and answer
  ui.py             The same in Gradio
  presentation.py   Text and step formatting shared by both
  areas.py          Dubai communities and aliases (JVC, JLT, JBR…)
  data.py           Loads CSVs into in-memory SQLite
  agent.py          LangGraph agent: plan, call tools, answer, check
  grounding.py      Checks every figure and ID in an answer against the evidence
  config.py         .env loading; Ollama (local) or Groq (cloud) model
  evaluation.py     Scores recorded answers on the four evaluation dimensions
  dld.py            Converts the Dubai Land Department transactions export
  tools/            The five agent tools and their schemas
scripts/
  ask.py                    Ask the agent from the command line
  run_eval.py               Run the frozen evaluation set and record the run
  generate_sample_data.py   Reproducible synthetic data (CI checks it)
  fetch_rules.py            Official legislation -> article text
  load_dld_transactions.py  Your DLD download -> transactions.csv
data/sample/        Synthetic listings and transactions
data/rules/         Dubai tenancy legislation, one section per article
eval/               Frozen evaluation set and recorded runs
docs/               DATA_SOURCES.md, EVALUATION.md, HOSTING.md, SAMPLE_RUN.md
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
4. ✅ Chat showing each agent step, [hosted free on Streamlit Community Cloud](https://appapppy-hadkfldjgvu3gw24nfdysg.streamlit.app/)
5. ✅ Evaluation set for tool choice, correctness and citations ([results](docs/EVALUATION.md))
6. Demo GIF, architecture diagram, release

Built to run entirely on free tiers.

## Disclaimer

For information and learning only, not legal, financial or investment advice.
Mortgage rules change and lenders may be stricter; check with a lender and the
official sources.

## License

[MIT](LICENSE) © Arjun Kumar
