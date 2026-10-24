"""Gradio chat interface that shows each agent step as it happens."""

import json
from collections.abc import Callable, Iterator

import gradio as gr

from estate_agent import config
from estate_agent.agent import AgentResult, CheckEvent, StepEvent, build_agent, stream
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
        return f"AED {result['monthly_payment']:,} a month"
    if tool == "search_rules":
        ids = ", ".join(r["passage_id"] for r in result["results"])
        return ids or result.get("note", "no matching passage")
    return "done"


def step_message(step: dict) -> gr.ChatMessage:
    body = json.dumps(step["result"] if step["ok"] else {"error": step["error"]}, indent=1, ensure_ascii=False)
    if len(body) > MAX_RESULT_CHARS:
        body = body[:MAX_RESULT_CHARS] + "\n… (truncated)"
    content = (
        f"**Arguments**\n```json\n{json.dumps(step['arguments'], ensure_ascii=False)}\n```\n"
        f"**Result**\n```json\n{body}\n```"
    )
    icon = TOOL_ICONS.get(step["tool"], "🔧")
    return gr.ChatMessage(
        content=content,
        metadata={"title": f"{icon} {step['tool']} → {summarize_step(step)}", "status": "done"},
    )


def check_message(check: dict) -> gr.ChatMessage:
    if check["ok"]:
        title = "✅ Grounding check passed: every figure and ID is in the tool results"
        content = ""
    else:
        problems = ", ".join(check["ungrounded_numbers"] + check["unknown_citations"])
        title = f"⚠️ Grounding check flagged: {problems}"
        content = f"Draft that was checked:\n\n{check['draft']}"
    return gr.ChatMessage(content=content, metadata={"title": title, "status": "done"})


def answer_message(result: AgentResult) -> gr.ChatMessage:
    content = result.answer
    if result.warning:
        content += f"\n\n> ⚠️ {result.warning}"
    return gr.ChatMessage(content=content)


def make_responder(agent_factory: Callable[[], object]):
    """Chat handler that streams agent steps into the history. Builds the agent on first use."""
    agent = None

    def respond(question: str, history: list) -> Iterator[tuple[str, list]]:
        nonlocal agent
        question = (question or "").strip()
        if not question:
            yield "", history
            return
        history = history + [gr.ChatMessage(role="user", content=question)]
        working = gr.ChatMessage(content="", metadata={"title": "🤔 Planning…", "status": "pending"})
        yield "", history + [working]
        try:
            if agent is None:
                agent = agent_factory()
            for event in stream(agent, question):
                if isinstance(event, StepEvent):
                    history.append(step_message(event.step))
                elif isinstance(event, CheckEvent):
                    history.append(check_message(event.check))
                else:
                    history.append(answer_message(event))
                    yield "", history
                    return
                yield "", history + [working]
        except Exception as error:  # show model/network problems in the chat, not a stack trace
            history.append(gr.ChatMessage(content=f"⚠️ The model could not be reached: {error}"))
            yield "", history

    return respond


def default_agent():
    config.load_env()
    problem = config.check_ollama() if config.provider() == "ollama" else None
    if problem:
        raise RuntimeError(problem)
    return build_agent(config.make_model(), Toolbox.load())


def build_app(agent_factory: Callable[[], object] = default_agent) -> gr.Blocks:
    respond = make_responder(agent_factory)
    with gr.Blocks(title=TITLE) as app:
        gr.Markdown(INTRO)
        chat = gr.Chatbot(height=560, show_label=False, placeholder="Ask about listings, prices, mortgages or tenancy rules.")
        with gr.Row():
            box = gr.Textbox(placeholder="e.g. Are Business Bay prices rising?", show_label=False, scale=8,
                             autofocus=True, submit_btn=True)
            clear = gr.Button("Clear", scale=1)
        gr.Examples(EXAMPLES, inputs=box, label="Try one")
        box.submit(respond, [box, chat], [box, chat])
        clear.click(lambda: ("", []), None, [box, chat])
    return app
