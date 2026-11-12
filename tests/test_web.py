import os
from pathlib import Path

import pytest
import streamlit as st
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

from estate_agent import presentation, web
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
        raise presentation.SetupError("OLLAMA_API_KEY is not set.")

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


def test_hosted_secrets_reach_the_model_settings(monkeypatch):
    """Streamlit secrets must become environment variables before the agent is built."""
    for name in ("OLLAMA_BASE_URL", "OLLAMA_MODEL", "OLLAMA_API_KEY"):
        monkeypatch.setenv(name, "")  # so pytest restores the original value afterwards
        monkeypatch.delenv(name)
    seen = {}

    def fake_default_agent():
        seen.update({k: os.environ.get(k) for k in ("OLLAMA_BASE_URL", "OLLAMA_MODEL", "OLLAMA_API_KEY")})
        raise presentation.SetupError("stop here")

    st.cache_resource.clear()
    monkeypatch.setattr(web.presentation, "default_agent", fake_default_agent)
    at = AppTest.from_file(APP)
    at.secrets["OLLAMA_BASE_URL"] = "https://ollama.com"
    at.secrets["OLLAMA_MODEL"] = "gemma4:31b"
    at.secrets["OLLAMA_API_KEY"] = "test-key-not-real"
    at.run()
    at.chat_input[0].set_value("hello").run()
    st.cache_resource.clear()
    assert seen == {"OLLAMA_BASE_URL": "https://ollama.com", "OLLAMA_MODEL": "gemma4:31b",
                    "OLLAMA_API_KEY": "test-key-not-real"}
    assert "stop here" in at.error[0].value
    report = at.error[0].value
    assert "Settings found: OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_API_KEY" in report
    assert "test-key-not-real" not in report  # names only, never values


def test_streamlit_follow_up_passes_earlier_turns(use_agent, toolbox):
    model = ScriptedModel([tool_turn(LISTINGS_CALL), AIMessage("L0002 rents for AED 120,000 a year (synthetic)."),
                           AIMessage("L0002 is the cheapest, at AED 120,000 a year.")])
    use_agent(lambda: build_agent(model, toolbox))
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("2BR in Marina under 130k?").run()
    at.chat_input[0].set_value("Which is cheapest?").run()
    assert not at.exception
    _, messages = model.calls[-1]
    assert "2BR in Marina under 130k?" in [m.content for m in messages]
    second = at.session_state["history"][1]
    assert second["answer"] == "L0002 is the cheapest, at AED 120,000 a year." and not second.get("warning")


class ProviderError(Exception):
    """Like ollama.ResponseError: the provider's own text and an HTTP status."""

    def __init__(self, text, status_code):
        super().__init__(text)
        self.status_code = status_code


class FailingModel:
    def __init__(self, error):
        self.error = error

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages):
        raise self.error

    stream = invoke


@pytest.mark.parametrize("error, shown", [
    (ProviderError("Add usage credits or subscribe and save up to 66%: https://ollama.com/plans", 402),
     "used up its free model allowance"),
    (ProviderError("Rate limit reached for model", 429), "used up its free model allowance"),
    (ProviderError("upstream connect error, internal ref 7f3a", 503), "try again in a few minutes"),
])
def test_provider_errors_are_explained_without_their_raw_text(use_agent, toolbox, error, shown):
    use_agent(lambda: build_agent(FailingModel(error), toolbox))
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("hello").run()
    assert not at.exception
    message = at.error[0].value
    assert shown in message
    assert str(error) not in message and "Settings found" not in message  # no upsell, no setting names


def test_quota_errors_are_recognised():
    assert presentation.is_quota_error(ProviderError("anything", 402))
    assert presentation.is_quota_error(RuntimeError("You have exceeded your quota"))
    assert not presentation.is_quota_error(ProviderError("Bad gateway", 502))
