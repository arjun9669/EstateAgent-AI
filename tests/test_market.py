from datetime import date

import pytest

from estate_agent.tools import ToolError
from estate_agent.tools.market import _months_before, area_price_stats


def test_trend_compares_adjacent_periods(conn):
    result = area_price_stats(conn, "Business Bay")
    assert result["data_as_of"] == "2026-09-30"
    current, previous = result["current_period"], result["previous_period"]
    assert (current["from"], current["to"]) == ("2025-09-30", "2026-09-30")
    assert (previous["from"], previous["to"]) == ("2024-09-30", "2025-09-30")
    assert current["transactions"] == 12 and previous["transactions"] == 12
    assert current["median_price_per_sqft"] == 1100.0
    assert previous["median_price_per_sqft"] == 1000.0
    assert result["price_per_sqft_change_pct"] == 10.0
    assert "warning" not in result


def test_sparse_data_is_flagged(conn):
    result = area_price_stats(conn, "Deira")
    assert result["current_period"]["transactions"] == 1
    assert result["previous_period"]["transactions"] == 0
    assert "price_per_sqft_change_pct" not in result
    assert "warning" in result


def test_filters_narrow_results(conn):
    result = area_price_stats(conn, "Business Bay", property_type="villa")
    assert result["current_period"]["transactions"] == 0


def test_months_before_handles_short_months():
    assert _months_before(date(2026, 3, 31), 1) == date(2026, 2, 28)
    assert _months_before(date(2026, 1, 15), 13) == date(2024, 12, 15)


@pytest.mark.parametrize("kwargs, message", [
    ({"area": "Nowhere"}, "Unknown area"),
    ({"area": "Deira", "months": 1}, "between 3 and 24"),
    ({"area": "Deira", "property_type": "castle"}, "property_type"),
])
def test_rejects_bad_input(conn, kwargs, message):
    with pytest.raises(ToolError, match=message):
        area_price_stats(conn, **kwargs)
