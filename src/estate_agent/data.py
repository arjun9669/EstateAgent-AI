"""Load listings and transactions into an in-memory SQLite database."""

import csv
import sqlite3
from pathlib import Path

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "data" / "sample"

SAMPLE_SOURCE = "Synthetic sample data (not real listings or sales)"

SCHEMA = """
CREATE TABLE listings (
    listing_id    TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    area          TEXT NOT NULL,
    property_type TEXT NOT NULL,   -- apartment | townhouse | villa
    purpose       TEXT NOT NULL,   -- rent | sale
    bedrooms      INTEGER NOT NULL, -- 0 = studio
    bathrooms     INTEGER NOT NULL,
    size_sqft     INTEGER NOT NULL,
    price_aed     INTEGER NOT NULL, -- annual rent, or sale price
    furnished     INTEGER NOT NULL, -- 0 | 1
    listed_date   TEXT NOT NULL     -- ISO date
);
CREATE TABLE transactions (
    transaction_id TEXT PRIMARY KEY,
    date           TEXT NOT NULL,
    area           TEXT NOT NULL,
    property_type  TEXT NOT NULL,
    bedrooms       INTEGER,         -- NULL when the source does not say
    size_sqft      INTEGER NOT NULL,
    price_aed      INTEGER NOT NULL
);
CREATE TABLE sources (
    table_name TEXT PRIMARY KEY,
    label      TEXT NOT NULL
);
"""

_INT_COLUMNS = {"bedrooms", "bathrooms", "size_sqft", "price_aed", "furnished"}


def _load(conn: sqlite3.Connection, table: str, path: Path) -> None:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    columns = list(rows[0])
    placeholders = ", ".join("?" for _ in columns)
    conn.executemany(
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
        ([_value(c, r[c]) for c in columns] for r in rows),
    )


def _value(column: str, text: str):
    if column not in _INT_COLUMNS:
        return text
    return int(text) if text != "" else None


def connect(
    data_dir: Path = SAMPLE_DIR,
    transactions_file: Path | None = None,
    transactions_source: str = SAMPLE_SOURCE,
) -> sqlite3.Connection:
    """Return a database holding listings.csv and transactions.csv from data_dir.

    transactions_file replaces the sample transactions, e.g. with DLD data
    converted by scripts/load_dld_transactions.py; transactions_source labels it.
    """
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _load(conn, "listings", data_dir / "listings.csv")
    _load(conn, "transactions", transactions_file or data_dir / "transactions.csv")
    conn.executemany(
        "INSERT INTO sources VALUES (?, ?)",
        [("listings", SAMPLE_SOURCE), ("transactions", transactions_source)],
    )
    conn.commit()
    return conn


def source_label(conn: sqlite3.Connection, table: str) -> str:
    return conn.execute("SELECT label FROM sources WHERE table_name = ?", (table,)).fetchone()[0]
