"""Convert the Dubai Land Department transactions export to this project's schema.

Input: Transactions.csv from Dubai Pulse (dataset dld_transactions-open), which
you download yourself after accepting its terms. It is large (about 1 GB) and
is read row by row. Output: a transactions.csv with the columns in data.SCHEMA,
keeping residential sales in areas the tools know.

Neither file is committed (data/raw/ is git-ignored); the dataset's licence is
not specified. See docs/DATA_SOURCES.md.
"""

import csv
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from estate_agent.areas import resolve_area

SOURCE = "Dubai Land Department transactions (Dubai Pulse open data)"
SQM_TO_SQFT = 10.7639
REQUIRED_COLUMNS = (
    "transaction_id", "instance_date", "trans_group_en", "property_usage_en",
    "property_sub_type_en", "area_name_en", "rooms_en", "procedure_area", "actual_worth",
)
PROPERTY_TYPES = {"flat": "apartment", "villa": "villa", "townhouse": "townhouse"}
# DLD registers communities under official area names that often differ from
# marketing names. UNVERIFIED: check against the dld_lkp_areas lookup before
# relying on these, and add more as the conversion report shows unmatched names.
DLD_AREA_NAMES = {
    "marsa dubai": "Dubai Marina",
    "burj khalifa": "Downtown Dubai",
    "al barsha south fourth": "Jumeirah Village Circle",
}
OUTPUT_COLUMNS = ("transaction_id", "date", "area", "property_type", "bedrooms", "size_sqft", "price_aed")
DATE_FORMATS = ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S", "%d-%m-%Y %H:%M:%S")


@dataclass
class Report:
    rows_read: int = 0
    rows_written: int = 0
    skipped: Counter = field(default_factory=Counter)
    unmatched_areas: Counter = field(default_factory=Counter)


def parse_date(text: str) -> date | None:
    text = text.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def parse_bedrooms(text: str) -> int | None:
    """'Studio' -> 0, '2 B/R' -> 2; anything else (e.g. 'Office', 'PENTHOUSE') -> None."""
    text = text.strip().lower()
    if text == "studio":
        return 0
    first = text.split(" ", 1)[0]
    return int(first) if first.isdigit() and "b/r" in text else None


def area_name(dld_name: str) -> str | None:
    area = resolve_area(dld_name) or resolve_area(DLD_AREA_NAMES.get(dld_name.strip().lower(), ""))
    return area.name if area else None


def convert_row(row: dict, report: Report) -> dict | None:
    if row["trans_group_en"].strip().lower() != "sales":
        report.skipped["not a sale"] += 1
        return None
    if row["property_usage_en"].strip().lower() != "residential":
        report.skipped["not residential"] += 1
        return None
    kind = PROPERTY_TYPES.get(row["property_sub_type_en"].strip().lower())
    if kind is None:
        report.skipped["other property type"] += 1
        return None
    area = area_name(row["area_name_en"])
    if area is None:
        report.unmatched_areas[row["area_name_en"].strip()] += 1
        return None
    day = parse_date(row["instance_date"])
    try:
        size_sqm, price = float(row["procedure_area"]), float(row["actual_worth"])
    except ValueError:
        size_sqm = price = 0
    if day is None or size_sqm <= 0 or price <= 0:
        report.skipped["missing date, size or price"] += 1
        return None
    return {
        "transaction_id": row["transaction_id"].strip(),
        "date": day.isoformat(),
        "area": area,
        "property_type": kind,
        "bedrooms": parse_bedrooms(row["rooms_en"]),
        "size_sqft": round(size_sqm * SQM_TO_SQFT),
        "price_aed": round(price),
    }


def convert(source: Path, destination: Path) -> Report:
    report = Report()
    with source.open(newline="", encoding="utf-8-sig") as f_in:
        reader = csv.DictReader(f_in)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"{source} is missing columns {missing}; found {reader.fieldnames}. "
                "The export format may have changed: update REQUIRED_COLUMNS and convert_row."
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
            writer.writeheader()
            for row in reader:
                report.rows_read += 1
                converted = convert_row(row, report)
                if converted is not None:
                    writer.writerow(converted)
                    report.rows_written += 1
    return report
