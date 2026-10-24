import csv
import sys
from pathlib import Path

import pytest

from estate_agent import dld
from estate_agent.tools import Toolbox

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import load_dld_transactions  # noqa: E402

# Invented rows in the DLD export's column layout (only the columns we read,
# plus one we ignore).
COLUMNS = list(dld.REQUIRED_COLUMNS) + ["building_name_en"]


def row(tid, day="15-03-2026", group="Sales", usage="Residential", sub_type="Flat",
        area="Marsa Dubai", rooms="2 B/R", sqm="100", worth="2000000"):
    return dict(zip(COLUMNS, [tid, day, group, usage, sub_type, area, rooms, sqm, worth, "Tower"]))


ROWS = [
    row("1"),
    row("2", day="2026-04-01", area="Business Bay", rooms="Studio", sqm="40", worth="900000"),
    row("3", sub_type="Villa", area="Palm Jumeirah", rooms="", worth="15000000", sqm="500"),
    row("4", group="Mortgages"),
    row("5", usage="Commercial"),
    row("6", sub_type="Office"),
    row("7", area="Al Warsan First"),
    row("8", worth=""),
    row("9", day="not a date"),
]


@pytest.fixture
def export(tmp_path):
    path = tmp_path / "Transactions.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(ROWS)
    return path


def test_converts_residential_sales_in_known_areas(export, tmp_path):
    out = tmp_path / "out.csv"
    report = dld.convert(export, out)
    rows = list(csv.DictReader(out.open()))
    assert [r["transaction_id"] for r in rows] == ["1", "2", "3"]
    assert rows[0] == {
        "transaction_id": "1", "date": "2026-03-15", "area": "Dubai Marina",
        "property_type": "apartment", "bedrooms": "2", "size_sqft": "1076", "price_aed": "2000000",
    }
    assert rows[1]["bedrooms"] == "0" and rows[1]["date"] == "2026-04-01"
    assert rows[2]["property_type"] == "villa" and rows[2]["bedrooms"] == ""
    assert report.rows_read == 9 and report.rows_written == 3
    assert report.skipped == {
        "not a sale": 1, "not residential": 1, "other property type": 1, "missing date, size or price": 2,
    }
    assert report.unmatched_areas == {"Al Warsan First": 1}


def test_missing_columns_explain_the_problem(tmp_path):
    path = tmp_path / "Transactions.csv"
    path.write_text("transaction_id,instance_date\n1,2026-01-01\n")
    with pytest.raises(ValueError, match="missing columns"):
        dld.convert(path, tmp_path / "out.csv")


@pytest.mark.parametrize("text, expected", [
    ("Studio", 0), ("1 B/R", 1), ("4 b/r", 4), ("Office", None), ("", None), ("PENTHOUSE", None),
])
def test_parse_bedrooms(text, expected):
    assert dld.parse_bedrooms(text) == expected


def test_converted_file_feeds_area_price_stats(export, tmp_path, embedding):
    out = tmp_path / "out.csv"
    dld.convert(export, out)
    toolbox = Toolbox.load(rules_dir=tmp_path, embedding_function=embedding,
                           transactions_file=out, transactions_source=dld.SOURCE)
    result = toolbox.run("area_price_stats", {"area": "Palm Jumeirah"})["result"]
    assert result["source"] == dld.SOURCE
    assert result["current_period"]["transactions"] == 1
    # Listings stay synthetic and say so.
    assert "Synthetic" in toolbox.run("search_listings", {"purpose": "rent"})["result"]["source"]


def test_script_reports_missing_download(tmp_path, capsys):
    assert load_dld_transactions.main(["--input", str(tmp_path / "nope.csv")]) == 1
    assert "Download it first" in capsys.readouterr().out
