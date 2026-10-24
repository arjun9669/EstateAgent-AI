import pytest

from estate_agent.grounding import check, numbers_in

EVIDENCE = ['{"price_aed":2277500,"median_price_per_sqft":2138.9,"change_pct":5.3,'
            '"listing_id":"L0435","passage_id":"decree-43-2013#art1","date":"2026-09-30"}',
            "Find flats under AED 170k"]


@pytest.mark.parametrize("answer", [
    "The median is AED 2,277,500.",
    "That is about AED 2.28 million, or AED 2.3M.",
    "Prices are AED 2,139 per sq ft, up 5.3%.",
    "Your budget of 170k is enough.",
    "Data as of 2026-09-30.",
    "See L0435 and decree-43-2013#art1.",
    "1. First option\n2. Second option, 3 bedrooms",
])
def test_grounded_answers(answer):
    assert check(answer, EVIDENCE).ok


@pytest.mark.parametrize("answer, flagged", [
    ("The median is AED 2,500,000.", ["2,500,000"]),
    ("Prices rose 7.1%.", ["7.1"]),
    ("About AED 2.4 million.", ["2.4 million"]),
    ("Rent caps start at 40%.", ["40"]),
    ("Rent can rise by 7%.", ["7"]),
    ("Rent can rise by 7 %.", ["7"]),  # 5% would match 5.3 rounded
])
def test_ungrounded_numbers_are_flagged(answer, flagged):
    report = check(answer, EVIDENCE)
    assert report.ungrounded_numbers == flagged and not report.ok


def test_unknown_citations_are_flagged():
    report = check("See L9999 and law-26-2007#art99.", EVIDENCE)
    assert report.unknown_citations == ["law-26-2007#art99", "L9999"]
    assert report.ungrounded_numbers == []


def test_digits_inside_ids_are_not_checked_as_numbers():
    report = check("L0435 under law-26-2007#art14 costs 1,200", ["1,200 L0435 law-26-2007#art14"])
    assert report.ok


def test_numbers_in_parses_suffixes_and_decimals():
    parsed = [(n.raw, n.scale, n.decimals) for n in numbers_in("AED 2.28 million, 170k, 1,200.5")]
    assert parsed == [(2.28, 1e6, 2), (170, 1e3, 0), (1200.5, 1, 1)]


def test_duplicates_reported_once():
    assert check("AED 9,999 and again AED 9,999", EVIDENCE).ungrounded_numbers == ["9,999"]
