"""The agent's tools and their JSON schemas for LLM tool calling."""

import inspect
import sqlite3
from pathlib import Path

from estate_agent.areas import area_names
from estate_agent.data import SAMPLE_DIR, connect
from estate_agent.tools.base import ToolError
from estate_agent.tools.listings import PROPERTY_TYPES, SORTS, compare_properties, search_listings
from estate_agent.tools.market import area_price_stats
from estate_agent.tools.mortgage import BUYERS, MAX_YEARS, mortgage_calculator
from estate_agent.tools.rules import RULES_DIR, RulesIndex, search_rules

__all__ = ["TOOL_SPECS", "ToolError", "Toolbox"]

_AREA = {"type": "string", "description": "Dubai community, e.g. " + ", ".join(area_names()[:5])}
_TYPE = {"type": "string", "enum": list(PROPERTY_TYPES)}

TOOL_SPECS = [
    {
        "name": "search_listings",
        "description": "Search property listings for rent (price is annual rent in AED) or sale.",
        "parameters": {
            "type": "object",
            "properties": {
                "purpose": {"type": "string", "enum": ["rent", "sale"]},
                "area": _AREA,
                "property_type": _TYPE,
                "bedrooms": {"type": "integer", "description": "Exact bedrooms; 0 = studio"},
                "min_bedrooms": {"type": "integer"},
                "min_price": {"type": "integer", "description": "AED"},
                "max_price": {"type": "integer", "description": "AED"},
                "furnished": {"type": "boolean"},
                "sort": {"type": "string", "enum": list(SORTS)},
                "limit": {"type": "integer", "description": "1-20, default 10"},
            },
            "required": ["purpose"],
        },
    },
    {
        "name": "compare_properties",
        "description": "Compare 2-5 listings by ID, including price per sqft against similar listings.",
        "parameters": {
            "type": "object",
            "properties": {"listing_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["listing_ids"],
        },
    },
    {
        "name": "area_price_stats",
        "description": "Median sale prices in an area for recent months and the change from the period before.",
        "parameters": {
            "type": "object",
            "properties": {
                "area": _AREA,
                "property_type": _TYPE,
                "bedrooms": {"type": "integer"},
                "months": {"type": "integer", "description": "Period length, 3-24, default 12"},
            },
            "required": ["area"],
        },
    },
    {
        "name": "mortgage_calculator",
        "description": (
            "UAE mortgage maths: monthly payment, loan-to-value cap check, upfront fees and "
            "optional debt-burden check. Uses the minimum down payment if none is given."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "property_price": {"type": "number", "description": "AED"},
                "annual_rate_pct": {"type": "number", "description": "e.g. 4.5"},
                "down_payment": {"type": "number", "description": "AED"},
                "years": {"type": "integer", "description": f"1-{MAX_YEARS}, default {MAX_YEARS}"},
                "buyer": {"type": "string", "enum": list(BUYERS)},
                "first_home": {"type": "boolean"},
                "off_plan": {"type": "boolean"},
                "monthly_income": {"type": "number", "description": "AED, for affordability"},
            },
            "required": ["property_price", "annual_rate_pct"],
        },
    },
    {
        "name": "search_rules",
        "description": "Search official Dubai tenancy and UAE mortgage documents. Cite passage_id.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "description": "Passages to return, 1-8, default 4"},
            },
            "required": ["query"],
        },
    },
]


class Toolbox:
    """Holds the data the tools need and runs tool calls by name."""

    def __init__(self, conn: sqlite3.Connection, rules: RulesIndex):
        self.conn, self.rules = conn, rules
        # name -> (function, leading arguments the agent does not supply)
        self._tools = {
            "search_listings": (search_listings, (conn,)),
            "compare_properties": (compare_properties, (conn,)),
            "area_price_stats": (area_price_stats, (conn,)),
            "mortgage_calculator": (mortgage_calculator, ()),
            "search_rules": (search_rules, (rules,)),
        }

    @classmethod
    def load(cls, data_dir: Path = SAMPLE_DIR, rules_dir: Path = RULES_DIR) -> "Toolbox":
        return cls(connect(data_dir), RulesIndex.from_directory(rules_dir))

    def run(self, name: str, arguments: dict) -> dict:
        """Run a tool call. Errors come back as data so the agent can correct itself."""
        if name not in self._tools:
            return {"ok": False, "error": f"Unknown tool {name!r}"}
        tool, context = self._tools[name]
        try:
            inspect.signature(tool).bind(*context, **arguments)
        except TypeError as e:
            # Wrong or missing argument names from the model.
            return {"ok": False, "error": f"Invalid arguments for {name}: {e}"}
        try:
            return {"ok": True, "result": tool(*context, **arguments)}
        except ToolError as e:
            return {"ok": False, "error": str(e)}
