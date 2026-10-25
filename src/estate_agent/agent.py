"""The LangGraph agent: plan, call tools, answer, then check the answer.

    question ─▶ agent ──tool calls──▶ tools ──results──▶ agent   (up to MAX_TOOL_ROUNDS)
                  │  └── tool limit hit ──▶ force_answer ──┐
                  ▼ answer                                  ▼
                check ◀─────────────────────────────────────┘
                  ├── figures not in evidence (once) ──▶ agent
                  ▼
                result

The model only plans and writes. Every number it states must appear in a tool
result or the question (see grounding.py); otherwise it is asked once to fix
the answer, and any remaining problems are reported alongside it.
"""

import json
import operator
from dataclasses import dataclass
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from estate_agent import grounding
from estate_agent.tools import TOOL_SPECS, Toolbox

MAX_TOOL_ROUNDS = 6
MAX_REVISIONS = 1
# Follow-up questions see this many earlier turns of the conversation.
MAX_HISTORY_TURNS = 3

SYSTEM_PROMPT = """You are EstateAgent AI, an assistant for the Dubai property market.

Rules:
- Use the tools for every fact: listings, prices, market statistics, mortgage figures and rules. Never state a number or rule that is not in a tool result or the user's question.
- Call as many tools as the question needs, one after another. If a tool returns an error, read it, fix the arguments and try again.
- Cite the listing IDs (like L0435) and rule passage IDs (like law-26-2007#art14) behind each fact.
- When you state a legal rule, keep the law's own wording for who must act ("either party", "the Landlord", "the Tenant"), and for its conditions and exceptions ("unless otherwise agreed"). Quote the key phrase. Do not narrow a rule to fit the question: if the user asks what the landlord must do and the law says either party, say either party.
- Put quotation marks only around text copied word for word from a tool result. Never shorten a quote with "..."; quote a shorter exact phrase instead. Never put tool field names (like within_debt_burden_cap) in quotation marks; describe them in plain words.
- Listings, and sales data unless a tool says otherwise, are synthetic sample data. Say so when you use them.
- Rent prices are per year, in AED.
- If a tool result contains a warning or a check that fails (for example that prices are not directly comparable, that there are too few sales to rely on, or that a down payment or debt burden is outside the limits), tell the user plainly, even if they did not ask.
- If the tools cannot answer part of the question, say what is missing. Do not guess.
- Earlier questions and answers in this conversation are context. Use them to understand follow-ups such as "what about JLT?", "compare the two cheapest" or "what if I were a national?": keep the earlier details (area, bedrooms, budget, price, buyer) unless the user changes them. Call the tools again for the figures in your answer. If the new question is unrelated, ignore the earlier turns.
- Be concise. End with: Information only, not legal or financial advice."""

TOOL_LIMIT_NOTE = "Tool limit reached; this call was not run."
FORCE_ANSWER_PROMPT = (
    "You have used the maximum number of tool calls. Answer now using only the tool "
    "results above, and say what is still unknown."
)
REVISION_PROMPT = (
    "Your answer contains figures or IDs that are not in any tool result or in my question: {problems}. "
    "Check them with the tools or remove them, then give the full corrected answer."
)

TOOLS = [{"type": "function", "function": spec} for spec in TOOL_SPECS]


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    steps: Annotated[list[dict], operator.add]
    tool_rounds: int
    revisions: int
    grounding: dict
    checks: Annotated[list[dict], operator.add]
    prior_evidence: list[str]  # earlier turns' tool calls and results


@dataclass
class AgentResult:
    question: str
    answer: str
    steps: list[dict]  # each tool call: tool, arguments, ok, result or error
    grounding: dict
    revisions: int
    checks: list[dict]  # every checked draft with its grounding report, in order

    @property
    def warning(self) -> str | None:
        if self.grounding.get("ok", True):
            return None
        problems = self.grounding["ungrounded_numbers"] + self.grounding["unknown_citations"]
        return "Could not verify against tool results: " + ", ".join(problems)


def _compact(data) -> str:
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)


def _evidence(messages: list[AnyMessage]) -> list[str]:
    """Texts an answer may draw on: the user's messages, tool arguments and results."""
    texts = []
    for message in messages:
        if isinstance(message, HumanMessage) and not message.additional_kwargs.get("revision"):
            texts.append(message.content)
        elif isinstance(message, ToolMessage):
            texts.append(message.content)
        elif isinstance(message, AIMessage):
            texts += [_compact(call["args"]) for call in message.tool_calls]
    return texts


def build_agent(model, toolbox: Toolbox, max_tool_rounds: int = MAX_TOOL_ROUNDS,
                max_revisions: int = MAX_REVISIONS):
    """Compile the graph. `model` is a LangChain chat model that supports bind_tools."""
    planner = model.bind_tools(TOOLS)
    finisher = model.bind_tools(TOOLS, tool_choice="none")

    def agent(state: AgentState) -> dict:
        return {"messages": [planner.invoke(state["messages"])]}

    def tools(state: AgentState) -> dict:
        messages, steps = [], []
        for call in state["messages"][-1].tool_calls:
            response = toolbox.run(call["name"], call["args"])
            steps.append({"tool": call["name"], "arguments": call["args"], **response})
            messages.append(ToolMessage(_compact(response), tool_call_id=call["id"], name=call["name"]))
        return {"messages": messages, "steps": steps, "tool_rounds": state["tool_rounds"] + 1}

    def force_answer(state: AgentState) -> dict:
        # Every tool call needs a reply before the model can be called again.
        unanswered = [
            ToolMessage(_compact({"ok": False, "error": TOOL_LIMIT_NOTE}), tool_call_id=c["id"], name=c["name"])
            for c in state["messages"][-1].tool_calls
        ]
        prompt = HumanMessage(FORCE_ANSWER_PROMPT, additional_kwargs={"revision": True})
        reply = finisher.invoke(state["messages"] + unanswered + [prompt])
        return {"messages": unanswered + [prompt, reply]}

    def check(state: AgentState) -> dict:
        answer = state["messages"][-1].content
        report = grounding.check(answer, _evidence(state["messages"]) + state.get("prior_evidence", []))
        update = {"grounding": report.as_dict(), "checks": [{"draft": answer, **report.as_dict()}]}
        if not report.ok and state["revisions"] < max_revisions:
            problems = ", ".join(report.ungrounded_numbers + report.unknown_citations)
            update["messages"] = [HumanMessage(REVISION_PROMPT.format(problems=problems),
                                               additional_kwargs={"revision": True})]
            update["revisions"] = state["revisions"] + 1
        return update

    def after_agent(state: AgentState) -> str:
        if not state["messages"][-1].tool_calls:
            return "check"
        return "tools" if state["tool_rounds"] < max_tool_rounds else "force_answer"

    def after_check(state: AgentState) -> str:
        return "agent" if isinstance(state["messages"][-1], HumanMessage) else END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tools)
    graph.add_node("force_answer", force_answer)
    graph.add_node("check", check)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", after_agent, ["tools", "force_answer", "check"])
    graph.add_edge("tools", "agent")
    graph.add_edge("force_answer", "check")
    graph.add_conditional_edges("check", after_check, ["agent", END])
    return graph.compile()


@dataclass
class StepEvent:
    """A tool call that has just run."""

    step: dict


@dataclass
class CheckEvent:
    """A draft answer that has just been checked; ok=False means a revision follows if allowed."""

    check: dict


def _history_messages(history: list[AgentResult]) -> list[AnyMessage]:
    """Earlier turns as question/answer pairs. Their tool output is not replayed;
    it is used only as evidence for the grounding check."""
    messages = []
    for turn in history[-MAX_HISTORY_TURNS:]:
        messages += [HumanMessage(turn.question), AIMessage(turn.answer)]
    return messages


def stream(graph, question: str, history: list[AgentResult] | None = None):
    """Yield StepEvent and CheckEvent as the agent works, then the AgentResult.

    history holds earlier AgentResults of the same conversation, oldest first.
    """
    history = (history or [])[-MAX_HISTORY_TURNS:]
    state = None
    seen_steps = seen_checks = 0
    for state in graph.stream(
        {
            "messages": [SystemMessage(SYSTEM_PROMPT), *_history_messages(history), HumanMessage(question)],
            "steps": [],
            "tool_rounds": 0,
            "revisions": 0,
            "grounding": {},
            "checks": [],
            "prior_evidence": [_compact(step) for turn in history for step in turn.steps],
        },
        # Each tool round is two graph steps; leave room for revisions.
        {"recursion_limit": 4 * (MAX_TOOL_ROUNDS + 2) * (MAX_REVISIONS + 1)},
        stream_mode="values",
    ):
        for step in state["steps"][seen_steps:]:
            yield StepEvent(step)
        for check in state["checks"][seen_checks:]:
            yield CheckEvent(check)
        seen_steps, seen_checks = len(state["steps"]), len(state["checks"])
    yield _result(question, state)


def ask(graph, question: str, history: list[AgentResult] | None = None) -> AgentResult:
    *_, result = stream(graph, question, history)
    return result


def _result(question: str, state: dict) -> AgentResult:
    return AgentResult(
        question=question,
        answer=state["messages"][-1].content,
        steps=state["steps"],
        grounding=state["grounding"],
        revisions=state["revisions"],
        checks=state["checks"],
    )
