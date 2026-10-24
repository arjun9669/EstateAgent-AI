# EstateAgent AI

**An agentic RAG assistant for the Dubai property market.** Ask a question in
plain language; an AI agent decides which tools to use (listing search, market
statistics, mortgage maths, official rules), runs them, and answers with sources.

> *"Find 2-bedroom apartments in Dubai Marina under AED 170k a year, compare
> them with the area's typical prices, and explain how rent increases are regulated."*

🚧 **Status: milestone 1 of 6, foundation.** The five tools work and are tested.
The language-model agent, chat interface and live demo come next. See [PLAN.md](PLAN.md).

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
| `search_rules` | "How much can my landlord raise the rent?" | Retrieval over official Dubai documents |

Each tool returns JSON with a `source` field, and errors come back as messages
the agent can read and correct, not crashes. The JSON schemas for LLM tool
calling are in [`tools/__init__.py`](src/estate_agent/tools/__init__.py), and a
test keeps them in sync with the Python functions.

## Try the tools

Requires Python 3.11+. No other dependencies yet.

```bash
git clone https://github.com/arjun9669/EstateAgent-AI.git
cd EstateAgent-AI
pip install -e ".[dev]"
pytest
```

```python
from estate_agent.tools import Toolbox

tools = Toolbox.load()
tools.run("search_listings", {"purpose": "rent", "area": "marina", "bedrooms": 2, "max_price": 170000})
tools.run("area_price_stats", {"area": "Dubai Marina"})
# -> median AED 2,139/sqft over the last 12 months, +5.3% on the year before (sample data)
tools.run("mortgage_calculator", {"property_price": 1_800_000, "annual_rate_pct": 4.5})
# -> 20% minimum down payment for an expat's first home, AED 8,004/month over 25 years
```

## Project structure

```
src/estate_agent/
  areas.py          Dubai communities and aliases (JVC, JLT, JBR…)
  data.py           Loads CSVs into in-memory SQLite
  tools/            The five agent tools and their schemas
scripts/
  generate_sample_data.py   Reproducible synthetic data (CI checks it)
data/sample/        Synthetic listings and transactions
data/rules/         Official documents for search_rules (milestone 2)
docs/DATA_SOURCES.md
```

## Data

The listings and transactions are **synthetic**. Area names are real; every
listing, price and ID is invented. Milestone 2 adds Dubai Land Department open
transaction data and official tenancy documents after checking their terms.
Details and sources: [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).

## Roadmap

1. ✅ Tools, sample data, tests, CI
2. Real DLD data and official documents; embeddings + ChromaDB for `search_rules`
3. LangGraph agent on Groq's free API, with checks that answers match tool output
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
