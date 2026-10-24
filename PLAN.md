# EstateAgent AI — build plan

An agentic RAG assistant for the Dubai property market. A user asks a question
in plain language; an agent plans which tools to call, calls them, checks the
results and answers with sources.

> Example: *"Find 2-bedroom apartments in Dubai Marina under AED 170k a year,
> compare them with the area's typical rent, and explain how rent increases are
> regulated."*

Everything must stay free to build, run and host.

## Architecture

```
            ┌──────────────────────────────┐
 question → │ Agent (LangGraph, Llama via   │ → answer + sources
            │ Groq free API)                │
            └──────┬───────────────────────┘
                   │ tool calls (JSON)
   ┌───────────────┼──────────────┬───────────────┬──────────────────┐
   ▼               ▼              ▼               ▼                  ▼
search_listings  compare_     area_price_    mortgage_          search_rules
(SQLite)         properties   stats          calculator         (embeddings in
                 (SQLite)     (SQLite)       (pure maths)       ChromaDB over laws)
```

The tools are deterministic Python. Only the agent uses a language model, so
every number in an answer comes from data or maths, not from the model.

## Tools

| Tool | Answers | Data |
|---|---|---|
| `search_listings` | "2BR in JVC under 90k/year, furnished" | Sample listings |
| `compare_properties` | "Compare L0012 and L0031" | Sample listings |
| `area_price_stats` | "Are Business Bay prices rising?" | Sales transactions |
| `mortgage_calculator` | "Monthly payment on AED 1.8M with 20% down?" | UAE lending rules |
| `search_rules` | "How much can my landlord raise the rent?" | Official documents |

Each tool returns JSON with a `source` field so the agent can cite it.

## Milestones

1. ✅ **Foundation**: repository, data sources, sample data, the five tools with
   unit tests and CI. No language model yet.
2. ✅ **Real data**: official tenancy legislation converted to citable articles;
   `search_rules` on MiniLM embeddings in ChromaDB; loader for the Dubai Land
   Department transactions export (not redistributed: licence unspecified).
   Open: verify the loader and DLD area-name mapping on the real file.
3. ✅ **Agent**: LangGraph loop with tool calling on Groq, a six-round tool
   limit, and a grounding check that every figure and ID in the answer appears
   in a tool result or the question (one revision, then a visible warning).
   Runs on a local Ollama model (default, no key) or Groq. Tested with a
   scripted model; live quality on each provider still to be measured.
4. ✅ **Interface and hosting**: chat pages in Streamlit (`streamlit_app.py`,
   hosted) and Gradio (`app.py`, local) that stream each tool call as an expandable
   step, then the grounding check and the answer. Hugging Face Spaces turned out to
   need PRO for Gradio apps, so the demo runs on Streamlit Community Cloud with the
   model over Ollama's hosted API: [live demo](https://appapppy-hadkfldjgvu3gw24nfdysg.streamlit.app/), [notes](docs/HOSTING.md).
5. ✅ **Evaluation**: 30 frozen questions scored on tool choice, correctness,
   citation support and grounding; recorded runs are re-scored in CI. First run
   with gemma4:cloud: 27/30 ([docs/EVALUATION.md](docs/EVALUATION.md)). Found a
   retrieval miss (contract registration) and two loose quotations to fix next.
6. **Polish** *(next)*: README with demo GIF, architecture diagram, release tag.

## Ground rules

- Listings are synthetic and labelled as such everywhere they appear.
- No scraping of property portals.
- The app gives information, not legal or financial advice.
- API keys never go into the repository.
