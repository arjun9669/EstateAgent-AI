"""Streamlit chat interface, used for the hosted demo on Streamlit Community Cloud.

Shows each tool call as an expandable step while the agent works, then the
grounding check and the answer. Run locally with `streamlit run streamlit_app.py`.
"""

import os
import threading

import streamlit as st

from estate_agent import presentation
from estate_agent.agent import CheckEvent, StepEvent, stream

# The free Ollama cloud plan serves one request at a time, so questions from all
# visitors run one after another, and each visitor gets a limited number.
MAX_QUESTIONS_PER_SESSION = 10
SETTING_NAMES = ("LLM_PROVIDER", "OLLAMA_BASE_URL", "OLLAMA_MODEL", "OLLAMA_API_KEY", "GROQ_API_KEY")


def load_secrets_into_environment() -> None:
    """Copy root-level text secrets into os.environ, where config.py reads settings.

    Streamlit parses secrets only when st.secrets is first used, so without this
    the hosted app never sees settings like OLLAMA_BASE_URL. Existing environment
    variables win. No secrets file (e.g. when run locally) is fine.
    """
    try:
        st.secrets.load_if_toml_exists()
        secrets = {key: st.secrets[key] for key in st.secrets}
    except Exception:  # missing or unreadable secrets: fall back to the environment and .env
        return
    for key, value in secrets.items():
        if isinstance(value, str):
            os.environ.setdefault(key, value)


def settings_report() -> str:
    """Which model settings are present, by name only (values may be secret)."""
    try:
        secret_names = sorted(st.secrets.keys())
    except Exception:
        secret_names = []
    found = [name for name in SETTING_NAMES if os.environ.get(name)]
    return (f"Settings found: {', '.join(found) or 'none'}. "
            f"Top-level secret names: {', '.join(secret_names) or 'none'}.")


def make_agent():
    """Replaced in tests with an agent on a scripted model."""
    load_secrets_into_environment()
    return presentation.default_agent()


@st.cache_resource(show_spinner="Loading the agent and its search index…")
def shared_agent():
    return make_agent()


@st.cache_resource
def model_lock() -> threading.Lock:
    return threading.Lock()


def show_step(step: dict) -> None:
    arguments, result = presentation.step_json(step)
    with st.expander(presentation.step_title(step)):
        st.markdown("**Arguments**")
        st.code(arguments, language="json")
        st.markdown("**Result**")
        st.code(result, language="json")


def show_check(check: dict) -> None:
    with st.expander(presentation.check_title(check)):
        if check["ok"]:
            st.caption("Every number and listing or law ID in the answer appears in a tool result or the question.")
        else:
            st.caption("Draft that was checked:")
            st.markdown(check["draft"])


def show_answer(turn: dict) -> None:
    st.markdown(turn["answer"])
    if turn.get("warning"):
        st.warning(turn["warning"])


def show_turn(turn: dict) -> None:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        for item in turn["items"]:
            (show_step if item["kind"] == "step" else show_check)(item["data"])
        if turn.get("error"):
            st.error(turn["error"])
        else:
            show_answer(turn)


def answer(question: str, earlier: list) -> dict:
    """Run the agent, drawing each step as it arrives; return the turn for the history.

    earlier holds the AgentResults of previous turns, so follow-ups have context.
    """
    turn = {"question": question, "items": []}
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        status = st.empty()
        try:
            status.caption("Loading…")
            agent = shared_agent()
            lock = model_lock()
            if not lock.acquire(blocking=False):
                status.caption("⏳ Waiting for another visitor's question to finish…")
                lock.acquire()
            try:
                status.caption("🤔 Planning…")
                for event in stream(agent, question, earlier):
                    if isinstance(event, StepEvent):
                        turn["items"].append({"kind": "step", "data": event.step})
                        show_step(event.step)
                    elif isinstance(event, CheckEvent):
                        turn["items"].append({"kind": "check", "data": event.check})
                        show_check(event.check)
                    else:
                        turn["answer"], turn["warning"] = event.answer, event.warning
                        turn["result"] = event
                    status.caption("🤔 Working…")
            finally:
                lock.release()
            status.empty()
            show_answer(turn)
        except Exception as error:  # model or network problem: show it, keep the page working
            status.empty()
            turn["error"] = f"The model could not be reached: {error}\n\n{settings_report()}"
            st.error(turn["error"])
    return turn


def main() -> None:
    st.set_page_config(page_title=presentation.TITLE, page_icon="🏙️")
    st.markdown(presentation.INTRO)
    history = st.session_state.setdefault("history", [])
    for turn in history:
        show_turn(turn)

    question = None
    if not history:
        st.caption("Try one:")
        for number, example in enumerate(presentation.EXAMPLES):
            if st.button(example, key=f"example-{number}", width="stretch"):
                question = example

    remaining = MAX_QUESTIONS_PER_SESSION - len(history)
    typed = st.chat_input(
        "Ask about listings, prices, mortgages or tenancy rules" if remaining > 0
        else "Question limit for this session reached",
        disabled=remaining <= 0,
    )
    question = question or typed
    if question and remaining > 0:
        earlier = [turn["result"] for turn in history if "result" in turn]
        history.append(answer(question.strip(), earlier))
        if len(history) == 1:
            st.rerun()  # hide the example buttons now that the chat has started
