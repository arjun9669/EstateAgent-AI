import pytest

from estate_agent.tools import ToolError
from estate_agent.tools.listings import compare_properties, search_listings


def ids(result):
    return [r["listing_id"] for r in result["results"]]


def test_filters_by_area_alias_bedrooms_and_budget(conn):
    result = search_listings(conn, purpose="rent", area="marina", bedrooms=2, max_price=150000)
    assert result["filters"]["area"] == "Dubai Marina"
    assert ids(result) == ["L0002", "L0001"]
    assert result["total_matches"] == 2
    assert result["price_unit"] == "AED per year"
    assert result["results"][0]["price_per_sqft"] == 100.0


def test_furnished_filter_and_sorting(conn):
    assert ids(search_listings(conn, purpose="rent", area="Dubai Marina", furnished=True)) == ["L0001"]
    newest = search_listings(conn, purpose="rent", area="Dubai Marina", sort="newest")
    assert ids(newest)[0] == "L0003"


def test_limit_caps_results_but_not_total(conn):
    result = search_listings(conn, purpose="rent", limit=1)
    assert len(result["results"]) == 1
    assert result["total_matches"] == 4


def test_no_matches_is_not_an_error(conn):
    result = search_listings(conn, purpose="rent", area="Deira")
    assert result["total_matches"] == 0 and result["results"] == []


@pytest.mark.parametrize("kwargs, message", [
    ({"purpose": "lease"}, "purpose must be one of"),
    ({"purpose": "rent", "area": "Atlantis City"}, "Unknown area"),
    ({"purpose": "rent", "bedrooms": 2, "min_bedrooms": 1}, "not both"),
    ({"purpose": "rent", "min_price": 200, "max_price": 100}, "greater than"),
    ({"purpose": "rent", "max_price": -1}, "negative"),
    ({"purpose": "rent", "sort": "random"}, "sort must be one of"),
])
def test_rejects_bad_input(conn, kwargs, message):
    with pytest.raises(ToolError, match=message):
        search_listings(conn, **kwargs)


def test_compare_against_peer_median(conn):
    result = compare_properties(conn, ["l0001", "L0002"])
    first, second = result["listings"]
    # Marina rental apartments: 150, 100 and 125 AED/sqft -> median 125.
    assert first["peer_count"] == 3
    assert first["peer_median_price_per_sqft"] == 125.0
    assert first["vs_peers_pct"] == 20.0
    assert second["vs_peers_pct"] == -20.0
    assert result["lowest_price_per_sqft"] == "L0002"


def test_compare_warns_when_mixing_rent_and_sale(conn):
    result = compare_properties(conn, ["L0001", "L0004"])
    assert "warning" in result and "lowest_price_per_sqft" not in result


@pytest.mark.parametrize("listing_ids, message", [
    (["L0001"], "between 2 and 5"),
    (["L0001", "L0001"], "between 2 and 5"),
    (["L0001", "L9999"], "not found: L9999"),
])
def test_compare_rejects_bad_ids(conn, listing_ids, message):
    with pytest.raises(ToolError, match=message):
        compare_properties(conn, listing_ids)
