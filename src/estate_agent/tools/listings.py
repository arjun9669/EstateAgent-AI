"""search_listings and compare_properties."""

import sqlite3
from statistics import median

from estate_agent.data import SAMPLE_SOURCE
from estate_agent.tools.base import ToolError, pct_change, require_area, require_choice

PURPOSES = ("rent", "sale")
PROPERTY_TYPES = ("apartment", "townhouse", "villa")
SORTS = {
    "price_asc": "price_aed ASC",
    "price_desc": "price_aed DESC",
    "newest": "listed_date DESC",
    "largest": "size_sqft DESC",
}
MAX_LIMIT = 20


def _listing(row: sqlite3.Row) -> dict:
    item = dict(row)
    item["furnished"] = bool(item["furnished"])
    item["price_per_sqft"] = round(item["price_aed"] / item["size_sqft"], 1)
    return item


def search_listings(
    conn: sqlite3.Connection,
    purpose: str,
    area: str | None = None,
    property_type: str | None = None,
    bedrooms: int | None = None,
    min_bedrooms: int | None = None,
    min_price: int | None = None,
    max_price: int | None = None,
    furnished: bool | None = None,
    sort: str = "price_asc",
    limit: int = 10,
) -> dict:
    """Find listings for rent (price = annual rent) or sale. Bedrooms 0 = studio."""
    filters: dict = {"purpose": require_choice("purpose", purpose, PURPOSES)}
    clauses, params = ["purpose = ?"], [filters["purpose"]]

    if area:
        filters["area"] = require_area(area).name
        clauses.append("area = ?")
        params.append(filters["area"])
    if property_type:
        filters["property_type"] = require_choice("property_type", property_type, PROPERTY_TYPES)
        clauses.append("property_type = ?")
        params.append(filters["property_type"])
    if bedrooms is not None and min_bedrooms is not None:
        raise ToolError("Give either bedrooms or min_bedrooms, not both")
    for name, value, op in (
        ("bedrooms", bedrooms, "bedrooms = ?"),
        ("min_bedrooms", min_bedrooms, "bedrooms >= ?"),
        ("min_price", min_price, "price_aed >= ?"),
        ("max_price", max_price, "price_aed <= ?"),
    ):
        if value is not None:
            if value < 0:
                raise ToolError(f"{name} cannot be negative")
            filters[name] = value
            clauses.append(op)
            params.append(value)
    if min_price is not None and max_price is not None and min_price > max_price:
        raise ToolError("min_price is greater than max_price")
    if furnished is not None:
        filters["furnished"] = furnished
        clauses.append("furnished = ?")
        params.append(int(furnished))
    if sort not in SORTS:
        raise ToolError(f"sort must be one of {', '.join(SORTS)}")
    limit = max(1, min(limit, MAX_LIMIT))

    where = " AND ".join(clauses)
    total = conn.execute(f"SELECT COUNT(*) FROM listings WHERE {where}", params).fetchone()[0]
    rows = conn.execute(
        f"SELECT * FROM listings WHERE {where} ORDER BY {SORTS[sort]}, listing_id LIMIT ?",
        [*params, limit],
    ).fetchall()
    return {
        "source": SAMPLE_SOURCE,
        "filters": filters,
        "price_unit": "AED per year" if filters["purpose"] == "rent" else "AED",
        "total_matches": total,
        "results": [_listing(r) for r in rows],
    }


def compare_properties(conn: sqlite3.Connection, listing_ids: list[str]) -> dict:
    """Compare 2-5 listings side by side, including value against similar listings."""
    ids = list(dict.fromkeys(i.strip().upper() for i in listing_ids))
    if not 2 <= len(ids) <= 5:
        raise ToolError("Give between 2 and 5 different listing IDs")
    rows = conn.execute(
        f"SELECT * FROM listings WHERE listing_id IN ({', '.join('?' for _ in ids)})", ids
    ).fetchall()
    found = {r["listing_id"]: r for r in rows}
    missing = [i for i in ids if i not in found]
    if missing:
        raise ToolError(f"Listing IDs not found: {', '.join(missing)}")

    compared = []
    for listing_id in ids:
        item = _listing(found[listing_id])
        # Peers: listings with the same purpose, area and property type.
        peers = conn.execute(
            "SELECT price_aed * 1.0 / size_sqft FROM listings"
            " WHERE purpose = ? AND area = ? AND property_type = ?",
            (item["purpose"], item["area"], item["property_type"]),
        ).fetchall()
        peer_median = median(p[0] for p in peers)
        item["peer_count"] = len(peers)
        item["peer_median_price_per_sqft"] = round(peer_median, 1)
        item["vs_peers_pct"] = pct_change(item["price_per_sqft"], peer_median)
        compared.append(item)

    purposes = {c["purpose"] for c in compared}
    result = {"source": SAMPLE_SOURCE, "listings": compared}
    if len(purposes) > 1:
        result["warning"] = "Mixes rentals and sales; prices are not directly comparable"
    else:
        result["lowest_price_per_sqft"] = min(compared, key=lambda c: c["price_per_sqft"])["listing_id"]
    return result
