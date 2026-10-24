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
   Tested with a scripted model. Gemini as a backup provider is deferred.
4. **Interface and hosting** *(next)*: Gradio chat that shows each agent step, deployed
   to Hugging Face Spaces with the API key as a Space secret.
5. **Evaluation**: a frozen question set measuring tool choice, answer
   correctness and citation support, run in CI against recorded tool outputs.
6. **Polish**: README with demo GIF, architecture diagram, release tag.

## Ground rules

- Listings are synthetic and labelled as such everywhere they appear.
- No scraping of property portals.
- The app gives information, not legal or financial advice.
- API keys never go into the repository.
