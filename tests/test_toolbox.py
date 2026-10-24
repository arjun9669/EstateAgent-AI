import inspect
import json

import pytest

from estate_agent.tools import TOOL_SPECS, Toolbox


@pytest.fixture(scope="module")
def toolbox():
    return Toolbox.load()  # committed sample data


@pytest.mark.parametrize("spec", TOOL_SPECS, ids=lambda s: s["name"])
def test_specs_match_function_signatures(toolbox, spec):
    tool, context = toolbox._tools[spec["name"]]
    params = list(inspect.signature(tool).parameters.values())[len(context):]
    assert set(spec["parameters"]["properties"]) == {p.name for p in params}
    assert set(spec["parameters"]["required"]) == {p.name for p in params if p.default is p.empty}


def test_every_tool_runs_on_sample_data_and_returns_json(toolbox):
    calls = [
        ("search_listings", {"purpose": "rent", "area": "JVC", "bedrooms": 2}),
        ("compare_properties", {"listing_ids": ["L0001", "L0002"]}),
        ("area_price_stats", {"area": "Dubai Marina"}),
        ("mortgage_calculator", {"property_price": 1_500_000, "annual_rate_pct": 4.25}),
        ("search_rules", {"query": "rent increase"}),
    ]
    for name, args in calls:
        response = toolbox.run(name, args)
        assert response["ok"], response
        assert "source" in response["result"]
        json.dumps(response)


def test_errors_are_returned_not_raised(toolbox):
    assert "Unknown tool" in toolbox.run("book_viewing", {})["error"]
    assert "Invalid arguments" in toolbox.run("search_listings", {"purpose": "rent", "colour": "red"})["error"]
    assert "Invalid arguments" in toolbox.run("search_listings", {})["error"]
    assert "Unknown area" in toolbox.run("area_price_stats", {"area": "Atlantis"})["error"]
