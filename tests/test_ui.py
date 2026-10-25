import gradio as gr
from langchain_core.messages import AIMessage

from estate_agent.agent import build_agent
from estate_agent.presentation import summarize_step
from estate_agent.ui import build_app, check_message, make_responder, step_message
from test_agent import GOOD_ANSWER, LISTINGS_CALL, RULES_CALL, ScriptedModel, tool_turn, toolbox  # noqa: F401


def titles(history):
    return [m.metadata.get("title") if m.metadata else None for m in history]


def test_streams_steps_check_and_answer_into_the_chat(toolbox):
    model = ScriptedModel([tool_turn(LISTINGS_CALL), tool_turn(RULES_CALL), AIMessage(GOOD_ANSWER)])
    respond = make_responder(lambda: build_agent(model, toolbox))
    updates = list(respond("2BR in Marina under 130k?", []))

    box, final = updates[-1]
    assert box == ""
    assert final[0].role == "user" and final[0].content == "2BR in Marina under 130k?"
    assert titles(final)[1] == "🏠 search_listings → 1 match (L0002)"
    assert titles(final)[2].startswith("📜 search_rules → rent-law#art1")
    assert titles(final)[3].startswith("✅ Grounding check passed")
    assert final[4].content == GOOD_ANSWER and not final[4].metadata
    # While working, a pending "Planning…" message is shown after the latest step.
    assert titles(updates[1][1])[-1] == "🤔 Planning…"
    assert all(titles(u[1])[-1] != "🤔 Planning…" for u in updates[-1:])


def test_flagged_draft_is_shown_and_warning_appended(toolbox):
    wrong = "L0002 rents for AED 99,000 a year."
    model = ScriptedModel([tool_turn(LISTINGS_CALL), AIMessage(wrong), AIMessage(wrong)])
    *_, (_, final) = make_responder(lambda: build_agent(model, toolbox))("rent?", [])

    flagged = [m for m in final if (m.metadata or {}).get("title", "").startswith("⚠️ Grounding check flagged")]
    assert len(flagged) == 2 and "99,000" in flagged[0].metadata["title"]
    assert wrong in flagged[0].content
    assert "Could not verify" in final[-1].content


def test_model_errors_are_shown_in_the_chat():
    def broken():
        raise RuntimeError("Ollama is not running.")

    *_, (_, final) = make_responder(broken)("hello", [])
    assert "Ollama is not running." in final[-1].content


def test_empty_question_does_nothing():
    assert list(make_responder(lambda: None)("   ", [])) == [("", [])]


def test_step_summaries():
    listing = {"tool": "search_listings", "ok": True, "arguments": {},
               "result": {"total_matches": 0, "results": []}}
    assert summarize_step(listing) == "0 matches"
    error = {"tool": "area_price_stats", "ok": False, "arguments": {"area": "X"}, "error": "Unknown area 'X'"}
    assert summarize_step(error) == "error: Unknown area 'X'"
    assert "Unknown area" in step_message(error).content
    mortgage = {"tool": "mortgage_calculator", "ok": True, "arguments": {}, "result": {"monthly payment (AED)": 8004}}
    assert summarize_step(mortgage) == "AED 8,004 a month"
    stats = {"tool": "area_price_stats", "ok": True, "arguments": {},
             "result": {"current_period": {"transactions": 12}, "price_per_sqft_change_pct": 10.0}}
    assert summarize_step(stats) == "12 sales in the last period, price/sqft +10.0%"
    rules = {"tool": "search_rules", "ok": True, "arguments": {}, "result": {"results": [], "note": "No passage"}}
    assert summarize_step(rules) == "No passage"


def test_long_results_are_truncated():
    step = {"tool": "search_rules", "ok": True, "arguments": {"query": "q"},
            "result": {"results": [{"passage_id": "a#art1", "text": "x" * 10_000}]}}
    assert "(truncated)" in step_message(step).content


def test_check_message_titles():
    assert check_message({"ok": True, "ungrounded_numbers": [], "unknown_citations": [], "draft": "d"}) \
        .metadata["title"].startswith("✅")


def test_app_builds_without_contacting_a_model():
    calls = []
    app = build_app(lambda: calls.append(1))
    assert isinstance(app, gr.Blocks) and calls == []
