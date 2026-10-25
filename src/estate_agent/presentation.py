"""Text and step formatting shared by the Gradio and Streamlit interfaces."""

import json

from estate_agent import config
from estate_agent.agent import build_agent
from estate_agent.tools import Toolbox

TITLE = "EstateAgent AI"
INTRO = """# 🏙️ EstateAgent AI
An **agentic RAG** assistant for the Dubai property market. Ask a question; the agent
decides which tools to use, runs them, and answers with sources. Open each step to see
exactly what it called and what came back. Every figure in the answer is checked
against those results.

**Listings and sales data are synthetic.** Legislation is the official English text of
Dubai Law 26/2007 (as amended) and Decree 43/2013. Information only, not legal or
financial advice. Each question is answered on its own; the agent does not remember
earlier questions. Questions are sent to the language model's provider.
[Source code](https://github.com/arjun9669/EstateAgent-AI)"""
EXAMPLES = [
    "Find 2-bedroom apartments in Dubai Marina under AED 170k a year",
    "Are Business Bay prices rising?",
    "What is the monthly payment on a AED 1.8M flat with the minimum down payment at 4.5%? "
    "I'm an expat buying my first home.",
    "How much can my landlord raise the rent when I renew?",
    "I want a furnished 1-bedroom in JVC under AED 90k. Compare the two cheapest, and tell me "
    "how much notice is needed before changing the lease terms.",
]
TOOL_ICONS = {
    "search_listings": "🏠",
    "compare_properties": "⚖️",
    "area_price_stats": "📈",
    "mortgage_calculator": "🧮",
    "search_rules": "📜",
}
MAX_RESULT_CHARS = 3000


def summarize_step(step: dict) -> str:
    """One line describing what a tool call returned."""
    if not step["ok"]:
        return f"error: {step['error']}"
    result, tool = step["result"], step["tool"]
    if tool == "search_listings":
        ids = ", ".join(r["listing_id"] for r in result["results"][:5])
        total = result["total_matches"]
        return f"{total} match{'' if total == 1 else 'es'}" + (f" ({ids})" if ids else "")
    if tool == "compare_properties":
        return "compared " + ", ".join(item["listing_id"] for item in result["listings"])
    if tool == "area_price_stats":
        change = result.get("price_per_sqft_change_pct")
        current = result["current_period"]["transactions"]
        return f"{current} sale{'' if current == 1 else 's'} in the last period" + (f", price/sqft {change:+}%" if change is not None else "")
    if tool == "mortgage_calculator":
        return f"AED {result['monthly payment (AED)']:,} a month"
    if tool == "search_rules":
        ids = ", ".join(r["passage_id"] for r in result["results"])
        return ids or result.get("note", "no matching passage")
    return "done"


def step_title(step: dict) -> str:
    return f"{TOOL_ICONS.get(step['tool'], '🔧')} {step['tool']} → {summarize_step(step)}"


def step_json(step: dict) -> tuple[str, str]:
    """The call's arguments and its result (or error) as JSON, the result truncated if long."""
    body = json.dumps(step["result"] if step["ok"] else {"error": step["error"]}, indent=1, ensure_ascii=False)
    if len(body) > MAX_RESULT_CHARS:
        body = body[:MAX_RESULT_CHARS] + "\n… (truncated)"
    return json.dumps(step["arguments"], ensure_ascii=False), body


def check_title(check: dict) -> str:
    if check["ok"]:
        return "✅ Grounding check passed: every figure and ID is in the tool results"
    return "⚠️ Grounding check flagged: " + ", ".join(check["ungrounded_numbers"] + check["unknown_citations"])


def default_agent():
    """Agent on the configured model; raises RuntimeError saying what is missing."""
    config.load_env()
    problem = config.check_ollama() if config.provider() == "ollama" else None
    if problem:
        raise RuntimeError(problem)
    return build_agent(config.make_model(), Toolbox.load())
