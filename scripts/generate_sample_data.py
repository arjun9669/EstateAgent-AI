"""Generate the synthetic sample listings and transactions.

Output is deterministic (fixed seed) so CI can check the committed CSVs match.
Everything generated here is invented; see docs/DATA_SOURCES.md.

    python scripts/generate_sample_data.py
"""

import csv
import random
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from estate_agent.areas import AREAS  # noqa: E402

SEED = 20261024
OUT = ROOT / "data" / "sample"
LISTING_COUNT = 600
TRANSACTION_COUNT = 2400
LISTINGS_FROM, LISTINGS_TO = date(2026, 7, 1), date(2026, 10, 15)
SALES_FROM, SALES_TO = date(2024, 1, 1), date(2026, 9, 30)

# bedrooms -> (min, max) size in sqft
SIZES = {
    "apartment": {0: (380, 550), 1: (650, 950), 2: (1000, 1450), 3: (1450, 2100), 4: (2200, 3200)},
    "townhouse": {2: (1400, 1900), 3: (1900, 2600), 4: (2500, 3300)},
    "villa": {3: (2600, 3600), 4: (3400, 5000), 5: (4500, 7500)},
}
BEDROOM_WEIGHTS = {
    "apartment": (0.15, 0.30, 0.30, 0.18, 0.07),
    "townhouse": (0.30, 0.45, 0.25),
    "villa": (0.35, 0.40, 0.25),
}


def random_day(rng, start, end):
    return start + timedelta(days=rng.randrange((end - start).days + 1))


def random_unit(rng, area):
    kind = rng.choice(area.property_types)
    bedrooms = rng.choices(list(SIZES[kind]), BEDROOM_WEIGHTS[kind])[0]
    low, high = SIZES[kind][bedrooms]
    return kind, bedrooms, rng.randrange(low, high + 1, 5)


def bedroom_label(bedrooms):
    return "Studio" if bedrooms == 0 else f"{bedrooms}-bedroom"


def listings(rng):
    rows = []
    for i in range(1, LISTING_COUNT + 1):
        area = rng.choice(AREAS)
        kind, bedrooms, size = random_unit(rng, area)
        value = size * area.sale_price_per_sqft * rng.uniform(0.85, 1.15)
        purpose = "rent" if rng.random() < 0.55 else "sale"
        if purpose == "rent":
            price = round(value * area.gross_yield * rng.uniform(0.9, 1.1), -3)
        else:
            price = round(value / 5000) * 5000
        furnished = rng.random() < (0.45 if purpose == "rent" else 0.25)
        rows.append({
            "listing_id": f"L{i:04d}",
            "title": f"{bedroom_label(bedrooms)} {kind} in {area.name}",
            "area": area.name,
            "property_type": kind,
            "purpose": purpose,
            "bedrooms": bedrooms,
            "bathrooms": max(1, bedrooms + (1 if kind != "apartment" or bedrooms > 1 else 0)),
            "size_sqft": size,
            "price_aed": int(price),
            "furnished": int(furnished),
            "listed_date": random_day(rng, LISTINGS_FROM, LISTINGS_TO).isoformat(),
        })
    return rows


def transactions(rng):
    # Each area gets its own monthly growth rate so trends differ between areas.
    growth = {a.name: rng.uniform(0.002, 0.010) for a in AREAS}
    last_month = (SALES_TO.year - SALES_FROM.year) * 12 + SALES_TO.month - SALES_FROM.month
    rows = []
    for _ in range(TRANSACTION_COUNT):
        area = rng.choice(AREAS)
        kind, bedrooms, size = random_unit(rng, area)
        day = random_day(rng, SALES_FROM, SALES_TO)
        month = (day.year - SALES_FROM.year) * 12 + day.month - SALES_FROM.month
        # Price levels in AREAS are "today"; earlier sales are discounted by growth.
        level = area.sale_price_per_sqft * (1 + growth[area.name]) ** (month - last_month)
        rows.append({
            "date": day.isoformat(),
            "area": area.name,
            "property_type": kind,
            "bedrooms": bedrooms,
            "size_sqft": size,
            "price_aed": round(size * level * rng.uniform(0.88, 1.12) / 1000) * 1000,
        })
    rows.sort(key=lambda r: (r["date"], r["area"], r["price_aed"]))
    for i, row in enumerate(rows, 1):
        row["transaction_id"] = f"T{i:05d}"
    return rows


def write(path, rows, fields):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    listing_rows = listings(rng)
    write(OUT / "listings.csv", listing_rows, list(listing_rows[0]))
    sale_rows = transactions(rng)
    write(OUT / "transactions.csv", sale_rows,
          ["transaction_id", "date", "area", "property_type", "bedrooms", "size_sqft", "price_aed"])
    print(f"Wrote {len(listing_rows)} listings and {len(sale_rows)} transactions to {OUT}")


if __name__ == "__main__":
    main()
