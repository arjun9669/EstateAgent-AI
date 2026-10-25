import hashlib
import json
from pathlib import Path

import pytest

from estate_agent.data import connect
from estate_agent.evaluation import cases_for_run, field_name_leaks, load_cases, quotations, score_case, score_run, summarize
from estate_agent.tools.listings import compare_properties, search_listings
from estate_agent.tools.market import area_price_stats
from estate_agent.tools.mortgage import calculate_mortgage as mortgage_calculator  # figures with code-style keys

ROOT = Path(__file__).resolve().parents[1]
CASES_FILE = ROOT / "eval" / "cases-v1.json"
CASE_SETS = sorted((ROOT / "eval").glob("cases-v*.json"))

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


@pytest.mark.parametrize("path", CASE_SETS, ids=lambda p: p.name)
def test_frozen_cases_are_unchanged(path):
    """Changing a frozen set must be deliberate: add a new version instead."""
    pinned = path.with_suffix(".sha256").read_text().split()[0]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == pinned


@pytest.mark.parametrize("path, expected_count", [(ROOT / "eval" / "cases-v1.json", 30),
                                                  (ROOT / "eval" / "cases-v2.json", 20),
                                                  (ROOT / "eval" / "cases-v3.json", 14),
                                                  (ROOT / "eval" / "cases-v4.json", 12)])
def test_case_ids_are_unique_and_well_formed(path, expected_count):
    cases = load_cases(path)
    assert len({c["id"] for c in cases}) == len(cases) == expected_count
    for case in cases:
        assert case["question"] and case["category"]
        assert case.get("facts"), case["id"]


def test_later_sets_ask_new_questions():
    seen = set()
    for name in ("cases-v1.json", "cases-v2.json", "cases-v3.json", "cases-v4.json"):
        questions = {c["question"] for c in load_cases(ROOT / "eval" / name)}
        assert not seen & questions, name
        seen |= questions


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
    """Recorded runs are re-scored in CI with their own case set and scorer version."""
    runs = sorted((ROOT / "eval" / "runs").glob("*.json"))
    for path in runs:
        run = json.loads(path.read_text())
        _, summary = score_run(cases_for_run(run), run)
        assert summary == run["summary"], path.name


def test_v2_expected_values_match_the_tools():
    conn = connect()
    assert "warning" in compare_properties(conn, ["L0521", "L0583"])
    downtown = area_price_stats(conn, "Downtown Dubai", property_type="apartment", bedrooms=0)
    assert downtown["price_per_sqft_change_pct"] == 8.0 and "warning" in downtown
    jbr = area_price_stats(conn, "JBR", property_type="apartment", bedrooms=3)
    assert jbr["price_per_sqft_change_pct"] == 8.8 and "warning" in jbr
    low = mortgage_calculator(property_price=2_000_000, annual_rate_pct=4.5, down_payment=200_000)
    assert low["meets_ltv_cap"] is False and low["minimum_down_payment"] == 400_000
    stretched = mortgage_calculator(property_price=2_500_000, annual_rate_pct=4.5, monthly_income=15_000)
    assert stretched["debt_burden_pct"] == 74.1 and stretched["within_debt_burden_cap"] is False
    assert [r["listing_id"] for r in search_listings(conn, purpose="rent", area="Mirdif",
                                                     property_type="villa", limit=3)["results"]] == ["L0361", "L0473", "L0187"]
    jlt = search_listings(conn, purpose="rent", area="JLT", bedrooms=2, furnished=True)
    assert [r["price_aed"] for r in jlt["results"]] == [102000, 117000, 152000]
    assert area_price_stats(conn, "Deira")["price_per_sqft_change_pct"] == 7.4
    assert mortgage_calculator(property_price=3_000_000, annual_rate_pct=4, buyer="national",
                               first_home=False)["monthly_payment"] == 10293
    assert mortgage_calculator(property_price=1_200_000, annual_rate_pct=4.25, off_plan=True)["minimum_down_payment"] == 600_000
    hills = search_listings(conn, purpose="sale", area="Dubai Hills", property_type="townhouse", limit=1)["results"][0]
    assert (hills["listing_id"], hills["price_aed"]) == ("L0027", 3665000)
    assert mortgage_calculator(property_price=3_665_000, annual_rate_pct=4.5)["monthly_payment"] == 16297


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


def test_v3_expected_values_match_the_tools():
    conn = connect()
    assert mortgage_calculator(property_price=1_500_000, annual_rate_pct=4, down_payment=150_000)["minimum_down_payment"] == 300_000
    national = mortgage_calculator(property_price=4_000_000, annual_rate_pct=4.5, buyer="national", down_payment=400_000)
    assert (national["minimum_down_payment"], national["max_loan_to_value_pct"]) == (600_000, 85.0)
    over = mortgage_calculator(property_price=1_600_000, annual_rate_pct=4.5, monthly_income=12_000)
    assert (over["monthly_payment"], over["debt_burden_pct"]) == (7115, 59.3)
    within = mortgage_calculator(property_price=3_000_000, annual_rate_pct=4, monthly_income=40_000)
    assert (within["monthly_payment"], within["debt_burden_pct"]) == (12668, 31.7)
    assert mortgage_calculator(property_price=900_000, annual_rate_pct=4.5, off_plan=True)["minimum_down_payment"] == 450_000
    assert mortgage_calculator(property_price=2_000_000, annual_rate_pct=4.5, off_plan=True)["max_loan_to_value_pct"] == 50.0
    second = mortgage_calculator(property_price=1_800_000, annual_rate_pct=4.25, first_home=False)
    assert (second["minimum_down_payment"], second["monthly_payment"]) == (720_000, 5851)
    dso = search_listings(conn, purpose="sale", area="DSO", bedrooms=2, limit=1)["results"][0]
    assert (dso["listing_id"], dso["price_aed"]) == ("L0063", 935000)
    assert compare_properties(conn, ["L0130", "L0143"])["lowest_price_per_sqft"] == "L0130"
    assert area_price_stats(conn, "Al Barsha", property_type="villa")["price_per_sqft_change_pct"] == 13.6
    assert mortgage_calculator(property_price=935_000, annual_rate_pct=4.5)["monthly_payment"] == 4158


def test_field_name_leaks():
    assert field_name_leaks('It is "within_debt_burden_cap"; dld_transfer_fee AED 4, ok.') == [
        "dld_transfer_fee", "within_debt_burden_cap"]
    assert field_name_leaks("The DLD transfer fee is AED 72,000.") == []


def test_scorer_1_2_finds_quotes_in_field_names():
    case = {"id": "T4", "category": "mortgage", "question": "q", "facts": [{"number": 60000}]}
    step = {"tool": "mortgage_calculator", "ok": True, "arguments": {},
            "result": {"upfront costs (AED)": {"DLD transfer fee (4% of price)": 60000}}}
    answer = 'The "DLD transfer fee (4% of price)" is AED 60,000.'
    assert not score_case(case, record(answer, [step]), "1.1")["citations"]
    assert score_case(case, record(answer, [step]), "1.2")["citations"]


def test_model_comparison_matches_the_recorded_runs():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import compare_models

    committed = json.loads((ROOT / "eval" / "model-comparison.json").read_text())
    assert committed == json.loads(json.dumps(compare_models.compare()))


def test_multi_turn_cases_take_evidence_from_earlier_turns():
    case = {"id": "T5", "category": "follow-up", "context": ["2BR in Marina?"], "question": "Compare the first two.",
            "tools_required": ["compare_properties"], "facts": [{"number": 143.7}], "citations_required": ["L0435"]}
    first = {"question": "2BR in Marina?", "steps": [STEP], "answer": "L0435 ..."}
    compare = {"tool": "compare_properties", "ok": True, "arguments": {}, "result": {"price_per_sqft": 143.7}}
    final = {"question": case["question"], "answer": 'L0435 is "A quiet flat near the marina walk" at 143.7 per sq ft.',
             "steps": [compare], "grounding": OK, "turns": [first, {"steps": [compare]}]}
    score = score_case(case, final)
    assert score["passed"], score["problems"]  # the quote comes from the first turn's tool result
    single = dict(final, turns=None)
    assert not score_case(case, single)["citations"]  # without earlier turns it is not evidence


def test_v4_expected_values_match_the_tools():
    conn = connect()
    jlt = search_listings(conn, purpose="rent", area="JLT", bedrooms=2, max_price=170000)
    assert [r["listing_id"] for r in jlt["results"][:2]] == ["L0566", "L0403"]
    assert compare_properties(conn, ["L0435", "L0503"])["lowest_price_per_sqft"] == "L0435"
    assert area_price_stats(conn, "Deira")["price_per_sqft_change_pct"] == 7.4
    afford = mortgage_calculator(property_price=1_800_000, annual_rate_pct=4.5, monthly_income=20_000)
    assert (afford["debt_burden_pct"], afford["within_debt_burden_cap"]) == (40.0, True)
    best = compare_properties(conn, ["L0403", "L0545", "L0198"])
    assert best["lowest_price_per_sqft"] == "L0545"
    deira = search_listings(conn, purpose="rent", area="Deira", bedrooms=0, limit=1)["results"][0]
    assert deira["listing_id"] == "L0576"
    pair = {x["listing_id"]: x["price_per_sqft"] for x in compare_properties(conn, ["L0562", "L0576"])["listings"]}
    assert pair == {"L0562": 59.3, "L0576": 52.4}
    national = mortgage_calculator(property_price=2_500_000, annual_rate_pct=4.5, buyer="national")
    assert (national["minimum_down_payment"], national["monthly_payment"]) == (375_000, 11811)
    assert area_price_stats(conn, "Palm Jumeirah", property_type="villa")["price_per_sqft_change_pct"] == 11.0
