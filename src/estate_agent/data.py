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
    bedrooms       INTEGER NOT NULL,
    size_sqft      INTEGER NOT NULL,
    price_aed      INTEGER NOT NULL
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
        ([int(r[c]) if c in _INT_COLUMNS else r[c] for c in columns] for r in rows),
    )


def connect(data_dir: Path = SAMPLE_DIR) -> sqlite3.Connection:
    """Return a database holding listings.csv and transactions.csv from data_dir."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _load(conn, "listings", data_dir / "listings.csv")
    _load(conn, "transactions", data_dir / "transactions.csv")
    conn.commit()
    return conn
