"""Gradio chat interface that shows each agent step as it happens."""

from collections.abc import Callable, Iterator

import gradio as gr

from estate_agent.agent import AgentResult, CheckEvent, StepEvent, stream
from estate_agent.presentation import (
    EXAMPLES, INTRO, TITLE, check_title, default_agent, step_json, step_title,
)


def step_message(step: dict) -> gr.ChatMessage:
    arguments, result = step_json(step)
    content = f"**Arguments**\n```json\n{arguments}\n```\n**Result**\n```json\n{result}\n```"
    return gr.ChatMessage(content=content, metadata={"title": step_title(step), "status": "done"})


def check_message(check: dict) -> gr.ChatMessage:
    content = "" if check["ok"] else f"Draft that was checked:\n\n{check['draft']}"
    return gr.ChatMessage(content=content, metadata={"title": check_title(check), "status": "done"})


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
