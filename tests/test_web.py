from pathlib import Path

import pytest
import streamlit as st
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

from estate_agent import web
from estate_agent.agent import build_agent
from test_agent import GOOD_ANSWER, LISTINGS_CALL, RULES_CALL, ScriptedModel, tool_turn, toolbox  # noqa: F401

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


@pytest.fixture
def use_agent(monkeypatch):
    """Point the app at a given agent factory, with Streamlit's caches cleared."""
    def install(factory):
        st.cache_resource.clear()
        monkeypatch.setattr(web, "make_agent", factory)
    yield install
    st.cache_resource.clear()


def labels(at):
    return [e.label for e in at.expander]


def test_first_load_shows_intro_and_examples_without_loading_a_model(use_agent):
    calls = []
    use_agent(lambda: calls.append(1))
    at = AppTest.from_file(APP).run()
    assert not at.exception
    assert "EstateAgent AI" in at.markdown[0].value
    assert len(at.button) == 5
    assert calls == []


def test_question_shows_steps_check_and_answer(use_agent, toolbox):
    model = ScriptedModel([tool_turn(LISTINGS_CALL), tool_turn(RULES_CALL), AIMessage(GOOD_ANSWER)])
    use_agent(lambda: build_agent(model, toolbox))
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("2BR in Marina under 130k?").run()

    assert not at.exception
    assert labels(at) == [
        "🏠 search_listings → 1 match (L0002)",
        "📜 search_rules → rent-law#art1",
        "✅ Grounding check passed: every figure and ID is in the tool results",
    ]
    assert any(GOOD_ANSWER in m.value for m in at.markdown)
    assert len(at.button) == 0  # examples are hidden once the chat starts
    assert at.session_state["history"][0]["answer"] == GOOD_ANSWER


def test_example_button_asks_that_question(use_agent, toolbox):
    model = ScriptedModel([AIMessage("Prices data needs a tool. Information only, not legal or financial advice.")])
    use_agent(lambda: build_agent(model, toolbox))
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.button[1].click().run()
    assert at.session_state["history"][0]["question"] == "Are Business Bay prices rising?"


def test_model_errors_are_shown_and_the_page_keeps_working(use_agent):
    def broken():
        raise RuntimeError("OLLAMA_API_KEY is not set.")

    use_agent(broken)
    at = AppTest.from_file(APP).run()
    at.chat_input[0].set_value("hello").run()
    assert not at.exception
    assert "OLLAMA_API_KEY is not set." in at.error[0].value
    assert at.session_state["history"][0]["error"]


def test_flagged_draft_and_warning_are_shown(use_agent, toolbox):
    wrong = "L0002 rents for AED 99,000 a year."
    model = ScriptedModel([tool_turn(LISTINGS_CALL), AIMessage(wrong), AIMessage(wrong)])
    use_agent(lambda: build_agent(model, toolbox))
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("rent?").run()
    assert sum(label.startswith("⚠️ Grounding check flagged: 99,000") for label in labels(at)) == 2
    assert "Could not verify" in at.warning[0].value


def test_session_question_limit(use_agent, toolbox, monkeypatch):
    monkeypatch.setattr(web, "MAX_QUESTIONS_PER_SESSION", 1)
    model = ScriptedModel([AIMessage("First answer.")])
    use_agent(lambda: build_agent(model, toolbox))
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("one").run()
    assert at.chat_input[0].disabled
    assert len(at.session_state["history"]) == 1
