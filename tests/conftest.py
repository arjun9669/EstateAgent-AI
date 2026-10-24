import csv
import hashlib
import re

import numpy as np
import pytest
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from estate_agent.data import connect

LISTING_FIELDS = ["listing_id", "title", "area", "property_type", "purpose", "bedrooms",
                  "bathrooms", "size_sqft", "price_aed", "furnished", "listed_date"]
TRANSACTION_FIELDS = ["transaction_id", "date", "area", "property_type", "bedrooms",
                      "size_sqft", "price_aed"]


def listing(listing_id, area, purpose, bedrooms, size, price, kind="apartment", furnished=0,
            listed="2026-09-01"):
    return dict(zip(LISTING_FIELDS, [listing_id, f"{listing_id} title", area, kind, purpose,
                                     bedrooms, max(1, bedrooms), size, price, furnished, listed]))


LISTINGS = [
    listing("L0001", "Dubai Marina", "rent", 2, 1000, 150000, furnished=1),
    listing("L0002", "Dubai Marina", "rent", 2, 1200, 120000),
    listing("L0003", "Dubai Marina", "rent", 1, 800, 100000, listed="2026-10-01"),
    listing("L0004", "Dubai Marina", "sale", 2, 1100, 2200000),
    listing("L0005", "Jumeirah Village Circle", "rent", 2, 1100, 90000),
    listing("L0006", "Arabian Ranches", "sale", 4, 3000, 4500000, kind="villa"),
]


def transaction(i, day, area, size, price, kind="apartment", bedrooms=2):
    return dict(zip(TRANSACTION_FIELDS, [f"T{i:05d}", day, area, kind, bedrooms, size, price]))


# Business Bay: 1,000/sqft in the previous 12 months, 1,100/sqft in the latest 12.
TRANSACTIONS = (
    [transaction(i, f"2024-{m:02d}-15", "Business Bay", 1000, 1_000_000) for i, m in enumerate(range(10, 13))]
    + [transaction(10 + i, f"2025-{m:02d}-15", "Business Bay", 1000, 1_000_000) for i, m in enumerate(range(1, 10))]
    + [transaction(30 + i, f"2025-{m:02d}-15", "Business Bay", 1000, 1_100_000) for i, m in enumerate(range(10, 13))]
    + [transaction(40 + i, f"2026-{m:02d}-15", "Business Bay", 1000, 1_100_000) for i, m in enumerate(range(1, 10))]
    + [transaction(60, "2026-09-30", "Deira", 800, 720_000)]
)


def write_csv(path, fields, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def data_dir(tmp_path):
    write_csv(tmp_path / "listings.csv", LISTING_FIELDS, LISTINGS)
    write_csv(tmp_path / "transactions.csv", TRANSACTION_FIELDS, TRANSACTIONS)
    return tmp_path


@pytest.fixture
def conn(data_dir):
    return connect(data_dir)


class HashingEmbedding(EmbeddingFunction[Documents]):
    """Offline stand-in for the real model: hashed bag of words, so tests need no download."""

    DIMENSIONS = 256

    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        vectors = []
        for text in input:
            vector = np.zeros(self.DIMENSIONS, dtype=np.float32)
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIMENSIONS] += 1
            vectors.append(vector)
        return vectors

    @staticmethod
    def name() -> str:
        return "hashing-test"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "HashingEmbedding":
        return HashingEmbedding()

    def default_space(self):
        return "cosine"


@pytest.fixture(scope="session")
def embedding():
    return HashingEmbedding()
