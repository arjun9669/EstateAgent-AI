import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from estate_agent import config
from estate_agent.agent import FORCE_ANSWER_PROMPT, TOOL_LIMIT_NOTE, TOOLS, ask, build_agent
from estate_agent.data import connect
from estate_agent.tools import Toolbox
from estate_agent.tools.rules import RulesIndex

RULE = """# Example Rent Law

## Article (1) Rent Increase

On renewal the landlord may increase the rent by at most five percent (5%).
"""


class ScriptedModel:
    """Stands in for a chat model: returns scripted replies and records what it saw."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []  # (tool_choice, messages)

    def bind_tools(self, tools, tool_choice=None):
        assert tools == TOOLS
        model = self

        class Bound:
            def invoke(self, messages):
                model.calls.append((tool_choice, list(messages)))
                return model.replies.pop(0)

        return Bound()


def call(name, args, call_id):
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def tool_turn(*calls):
    return AIMessage(content="", tool_calls=list(calls))


@pytest.fixture
def toolbox(data_dir, tmp_path, embedding):
    rules = tmp_path / "rules"
    rules.mkdir()
    (rules / "rent-law.md").write_text(RULE)
    return Toolbox(connect(data_dir), RulesIndex.from_directory(rules, embedding))


LISTINGS_CALL = call("search_listings", {"purpose": "rent", "area": "Dubai Marina", "bedrooms": 2,
                                         "max_price": 130000}, "c1")
RULES_CALL = call("search_rules", {"query": "rent increase on renewal"}, "c2")
GOOD_ANSWER = ("L0002 is a 2-bedroom flat in Dubai Marina at AED 120,000 a year (synthetic sample data). "
               "On renewal the rent can rise by at most 5% (rent-law#art1). "
               "Information only, not legal or financial advice.")


def test_plans_multiple_tools_and_answers_with_grounded_citations(toolbox):
    model = ScriptedModel([tool_turn(LISTINGS_CALL), tool_turn(RULES_CALL), AIMessage(GOOD_ANSWER)])
    result = ask(build_agent(model, toolbox), "2BR in Marina under 130k, and how much can rent rise?")

    assert result.answer == GOOD_ANSWER
    assert [s["tool"] for s in result.steps] == ["search_listings", "search_rules"]
    assert all(s["ok"] for s in result.steps)
    assert result.steps[0]["result"]["results"][0]["listing_id"] == "L0002"
    assert result.grounding["ok"] and result.revisions == 0 and result.warning is None
    # The model saw each tool result before planning the next step.
    _, last_messages = model.calls[-1]
    tool_messages = [m for m in last_messages if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in tool_messages] == ["c1", "c2"]
    assert json.loads(tool_messages[0].content)["ok"] is True


def test_tool_errors_are_returned_so_the_model_can_retry(toolbox):
    bad = call("area_price_stats", {"area": "Atlantis"}, "c1")
    good = call("area_price_stats", {"area": "Business Bay"}, "c2")
    answer = "Business Bay sale prices rose 10.0% to AED 1,100 per sq ft (synthetic sample data)."
    model = ScriptedModel([tool_turn(bad), tool_turn(good), AIMessage(answer)])
    result = ask(build_agent(model, toolbox), "Are Business Bay prices rising?")

    assert [s["ok"] for s in result.steps] == [False, True]
    assert "Unknown area" in result.steps[0]["error"]
    assert result.grounding["ok"]


def test_ungrounded_figure_triggers_one_revision(toolbox):
    wrong = "L0002 rents for AED 99,000 a year."
    fixed = "L0002 rents for AED 120,000 a year (synthetic sample data)."
    model = ScriptedModel([tool_turn(LISTINGS_CALL), AIMessage(wrong), AIMessage(fixed)])
    result = ask(build_agent(model, toolbox), "2BR in Marina under 130k?")

    assert result.revisions == 1
    assert result.answer == fixed
    assert result.grounding["ok"]
    _, revision_messages = model.calls[-1]
    assert isinstance(revision_messages[-1], HumanMessage)
    assert "99,000" in revision_messages[-1].content


def test_unfixed_answer_is_returned_with_a_warning(toolbox):
    wrong = "L0002 rents for AED 99,000 a year; see also L9999."
    model = ScriptedModel([tool_turn(LISTINGS_CALL), AIMessage(wrong), AIMessage(wrong)])
    result = ask(build_agent(model, toolbox), "2BR in Marina under 130k?")

    assert result.revisions == 1
    assert result.grounding == {"ok": False, "ungrounded_numbers": ["99,000"], "unknown_citations": ["L9999"]}
    assert "99,000" in result.warning and "L9999" in result.warning


def test_rule_figures_need_a_rules_lookup(toolbox):
    # GOOD_ANSWER's 5% and rent-law#art1 come from search_rules, which was never called.
    model = ScriptedModel([tool_turn(LISTINGS_CALL), AIMessage(GOOD_ANSWER), AIMessage(GOOD_ANSWER)])
    result = ask(build_agent(model, toolbox), "2BR in Marina under 130k?")
    assert result.grounding["unknown_citations"] == ["rent-law#art1"]
    assert result.grounding["ungrounded_numbers"] == ["5"]


def test_revision_prompt_does_not_count_as_evidence(toolbox):
    # Echoing the flagged figure back must not make it grounded.
    wrong = "Rent is AED 99,000."
    model = ScriptedModel([AIMessage(wrong), AIMessage(wrong)])
    result = ask(build_agent(model, toolbox), "What is the rent?")
    assert result.grounding["ungrounded_numbers"] == ["99,000"]


def test_tool_limit_forces_an_answer_without_tools(toolbox):
    looping = [tool_turn(call("search_rules", {"query": "rent"}, f"c{i}")) for i in range(3)]
    final = "I could not finish checking. Rent can rise by at most 5% (rent-law#art1)."
    model = ScriptedModel(looping + [AIMessage(final)])
    result = ask(build_agent(model, toolbox, max_tool_rounds=2), "Tell me everything about rent")

    assert len(result.steps) == 2
    assert result.answer == final
    tool_choice, messages = model.calls[-1]
    assert tool_choice == "none"
    assert messages[-1].content == FORCE_ANSWER_PROMPT
    unanswered = messages[-2]
    assert isinstance(unanswered, ToolMessage) and unanswered.tool_call_id == "c2"
    assert TOOL_LIMIT_NOTE in unanswered.content


def test_question_numbers_count_as_evidence(toolbox):
    model = ScriptedModel([AIMessage("A budget of AED 250,000 is above every listing I can see.")])
    result = ask(build_agent(model, toolbox), "Is AED 250,000 a year enough?")
    assert result.grounding["ok"] and result.steps == []


def test_missing_api_key_explains_setup(monkeypatch, tmp_path):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(config, "ENV_FILE", tmp_path / ".env")
    config.load_env()
    with pytest.raises(RuntimeError, match="console.groq.com"):
        config.make_groq_model()


def test_env_file_is_read_without_overriding(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("# comment\nGROQ_MODEL='some-model'\nGROQ_API_KEY=from-file\n")
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "from-environment")
    config.load_env(env)
    assert config.groq_model_name() == "some-model"
    import os
    assert os.environ["GROQ_API_KEY"] == "from-environment"
