import hashlib
import json
from pathlib import Path

import pytest

from estate_agent.data import connect
from estate_agent.evaluation import load_cases, quotations, score_case, score_run, summarize
from estate_agent.tools.listings import compare_properties, search_listings
from estate_agent.tools.market import area_price_stats
from estate_agent.tools.mortgage import mortgage_calculator

ROOT = Path(__file__).resolve().parents[1]
CASES_FILE = ROOT / "eval" / "cases-v1.json"
# Changing the frozen set must be deliberate: add cases-v2.json instead.
CASES_SHA256 = (ROOT / "eval" / "cases-v1.sha256").read_text().split()[0]

CASE = {
    "id": "T1", "category": "listings", "question": "2BR in Marina under 170k?",
    "tools_required": ["search_listings"], "tools_forbidden": ["mortgage_calculator"],
    "facts": [{"number": 148000}, {"text": "synthetic"}, {"any_of": ["per year", "a year"]}],
    "citations_required": ["L0435"], "must_not_contain": ["guaranteed"],
}
STEP = {"tool": "search_listings", "ok": True, "arguments": {},
        "result": {"results": [{"listing_id": "L0435", "price_aed": 148000, "title": "A quiet flat near the marina walk"}]}}
OK = {"ok": True, "ungrounded_numbers": [], "unknown_citations": []}


def record(answer, steps=(STEP,), grounding=OK):
    return {"answer": answer, "steps": list(steps), "grounding": grounding}


def test_frozen_cases_are_unchanged():
    assert hashlib.sha256(CASES_FILE.read_bytes()).hexdigest() == CASES_SHA256


def test_case_ids_are_unique_and_well_formed():
    cases = load_cases(CASES_FILE)
    assert len({c["id"] for c in cases}) == len(cases) == 30
    for case in cases:
        assert case["question"] and case["category"]
        assert case.get("facts"), case["id"]


def test_passing_answer():
    score = score_case(CASE, record("L0435 costs AED 148,000 a year (synthetic sample data)."))
    assert score["passed"] and score["problems"] == []


def test_rounded_figures_count():
    case = dict(CASE, facts=[{"number": 2277500}])
    assert score_case(case, record("About AED 2.28 million, L0435, synthetic."))["correctness"]


@pytest.mark.parametrize("answer, steps, grounding, failed, problem", [
    ("L0435: AED 148,000 a year, synthetic.", [], OK, "tool_choice", "did not run search_listings"),
    ("L0435: AED 148,000 a year, synthetic.", [STEP, dict(STEP, tool="mortgage_calculator")], OK,
     "tool_choice", "called mortgage_calculator"),
    ("L0435: AED 150,000 a year, synthetic.", [STEP], OK, "correctness", "missing 148000"),
    ("L0435: AED 148,000 a year, synthetic, guaranteed.", [STEP], OK, "correctness", "says 'guaranteed'"),
    ("AED 148,000 a year, synthetic.", [STEP], OK, "citations", "does not cite L0435"),
    ("L0435 at AED 148,000 a year, synthetic, is \"a quiet flat by the beach\".", [STEP], OK,
     "citations", "quote not found word for word"),
    ("L0435: AED 148,000 a year, synthetic.", [STEP],
     {"ok": False, "ungrounded_numbers": ["99"], "unknown_citations": []}, "grounding", "unsupported numbers: 99"),
])
def test_each_dimension_fails_independently(answer, steps, grounding, failed, problem):
    score = score_case(CASE, record(answer, steps, grounding))
    assert not score["passed"] and not score[failed]
    assert any(problem in p for p in score["problems"])
    assert all(score[d] for d in ("tool_choice", "correctness", "citations", "grounding") if d != failed)


def test_quotes_from_tool_results_pass():
    answer = "L0435 at AED 148,000 a year, synthetic, is \"A quiet flat near the marina walk\"."
    assert score_case(CASE, record(answer))["citations"]


def test_no_citations_and_unknown_citations():
    case = {"id": "T2", "category": "out-of-scope", "question": "q", "facts": [{"text": "cannot"}], "no_citations": True}
    assert not score_case(case, record("I cannot say; see L0435."))["citations"]
    bad = {"ok": False, "ungrounded_numbers": [], "unknown_citations": ["L9999"]}
    assert "which no tool returned" in score_case(CASE, record("L0435 AED 148,000 a year synthetic", grounding=bad))["problems"][0]


def test_summary_and_missing_results():
    scores = [score_case(CASE, record("L0435: AED 148,000 a year, synthetic.")),
              score_case(CASE, record("nothing"))]
    summary = summarize(scores)
    assert summary["passed"] == 1 and summary["by_category"] == {"listings": "1/2"}
    with pytest.raises(ValueError, match="no result for cases: T1"):
        score_run([CASE], {"results": []})


def test_expected_values_match_the_tools():
    """The frozen expectations must agree with what the tools return for the committed data."""
    conn = connect()
    marina = search_listings(conn, purpose="rent", area="Dubai Marina", bedrooms=2, max_price=170000)
    assert [r["price_aed"] for r in marina["results"]] == [148000, 165000, 169000]
    assert search_listings(conn, purpose="rent", area="International City", bedrooms=0, limit=1)["results"][0]["listing_id"] == "L0562"
    assert search_listings(conn, purpose="rent", area="Deira", bedrooms=3, max_price=50000)["total_matches"] == 0
    assert compare_properties(conn, ["L0562", "L0149", "L0249"])["lowest_price_per_sqft"] == "L0149"
    assert "warning" in compare_properties(conn, ["L0435", "L0167"])
    assert area_price_stats(conn, "Business Bay")["price_per_sqft_change_pct"] == 7.5
    jvc = area_price_stats(conn, "JVC", property_type="apartment", bedrooms=2)
    assert jvc["price_per_sqft_change_pct"] == 2.2 and "warning" in jvc
    assert mortgage_calculator(property_price=1_800_000, annual_rate_pct=4.5)["monthly_payment"] == 8004
    assert mortgage_calculator(property_price=6_000_000, annual_rate_pct=5, buyer="national")["monthly_payment"] == 26307
    assert mortgage_calculator(property_price=2_500_000, annual_rate_pct=4.5, monthly_income=25000)["debt_burden_pct"] == 44.5
    cheapest = search_listings(conn, purpose="sale", area="JVC", bedrooms=1, limit=1)["results"][0]
    assert (cheapest["listing_id"], cheapest["price_aed"]) == ("L0401", 700000)
    assert mortgage_calculator(property_price=700_000, annual_rate_pct=4.5)["monthly_payment"] == 3113


def test_recorded_runs_rescore_identically():
    """Recorded runs are re-scored in CI; the stored summary must match today's scorer."""
    cases = load_cases(CASES_FILE)
    runs = sorted((ROOT / "eval" / "runs").glob("*.json"))
    for path in runs:
        run = json.loads(path.read_text())
        _, summary = score_run(cases, run)
        assert summary == run["summary"], path.name


def test_scorer_1_1_pairs_quotation_marks_in_order():
    answer = 'The median is "2138.9" AED. This represents a "5.3" percent increase.'
    assert quotations(answer, "1.0") == [" AED. This represents a "]  # the 1.0 bug
    assert quotations(answer, "1.1") == []  # both quotes are under 12 characters
    assert quotations('He said "registered with RERA" and "no".', "1.1") == ["registered with RERA"]
    assert quotations("Curly “will be registered with RERA” works.", "1.1") == ["will be registered with RERA"]


def test_scorer_versions_score_the_bug_case_differently():
    case = {"id": "T3", "category": "market", "question": "q", "facts": [{"number": 2138.9}]}
    answer = 'Median "2138.9" AED. This represents a "5.3" percent rise.'
    step = {"tool": "area_price_stats", "ok": True, "arguments": {}, "result": {"median": 2138.9, "change": 5.3}}
    assert not score_case(case, record(answer, [step]), "1.0")["citations"]
    assert score_case(case, record(answer, [step]), "1.1")["citations"]
    with pytest.raises(ValueError, match="Unknown scorer"):
        score_case(case, record(answer, [step]), "9.9")


def test_rescored_report_matches_the_latest_scorer():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import rescore_runs

    committed = json.loads((ROOT / "eval" / "rescored-latest.json").read_text())
    assert committed == json.loads(json.dumps(rescore_runs.rescore()))
