"""area_price_stats: sale price levels and trend from recorded transactions."""

import sqlite3
from calendar import monthrange
from datetime import date
from statistics import median

from estate_agent.data import source_label
from estate_agent.tools.base import ToolError, pct_change, require_area, require_choice
from estate_agent.tools.listings import PROPERTY_TYPES

MIN_RELIABLE_COUNT = 10


def _months_before(day: date, months: int) -> date:
    year, month = divmod(day.year * 12 + day.month - 1 - months, 12)
    month += 1
    return date(year, month, min(day.day, monthrange(year, month)[1]))


def _window(conn, where: str, params: list, start: date, end: date) -> dict:
    rows = conn.execute(
        f"SELECT price_aed, size_sqft FROM transactions WHERE {where} AND date > ? AND date <= ?",
        [*params, start.isoformat(), end.isoformat()],
    ).fetchall()
    stats = {"from": start.isoformat(), "to": end.isoformat(), "transactions": len(rows)}
    if rows:
        stats["median_price_aed"] = round(median(r["price_aed"] for r in rows))
        stats["median_price_per_sqft"] = round(median(r["price_aed"] / r["size_sqft"] for r in rows), 1)
    return stats


def area_price_stats(
    conn: sqlite3.Connection,
    area: str,
    property_type: str | None = None,
    bedrooms: int | None = None,
    months: int = 12,
) -> dict:
    """Median sale prices over the last `months`, compared with the period before."""
    if not 3 <= months <= 24:
        raise ToolError("months must be between 3 and 24")
    filters: dict = {"area": require_area(area).name}
    clauses, params = ["area = ?"], [filters["area"]]
    if property_type:
        filters["property_type"] = require_choice("property_type", property_type, PROPERTY_TYPES)
        clauses.append("property_type = ?")
        params.append(filters["property_type"])
    if bedrooms is not None:
        filters["bedrooms"] = bedrooms
        clauses.append("bedrooms = ?")
        params.append(bedrooms)
    where = " AND ".join(clauses)

    latest = conn.execute("SELECT MAX(date) FROM transactions").fetchone()[0]
    if latest is None:
        raise ToolError("No transaction data loaded")
    as_of = date.fromisoformat(latest)
    start = _months_before(as_of, months)
    current = _window(conn, where, params, start, as_of)
    previous = _window(conn, where, params, _months_before(as_of, 2 * months), start)

    result = {
        "source": source_label(conn, "transactions"),
        "filters": filters,
        "data_as_of": latest,
        "current_period": current,
        "previous_period": previous,
    }
    if current["transactions"] and previous["transactions"]:
        result["price_per_sqft_change_pct"] = pct_change(
            current["median_price_per_sqft"], previous["median_price_per_sqft"]
        )
    if min(current["transactions"], previous["transactions"]) < MIN_RELIABLE_COUNT:
        result["warning"] = (
            f"Fewer than {MIN_RELIABLE_COUNT} transactions in a period; treat figures as rough"
        )
    return result
