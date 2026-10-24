"""Convert a downloaded DLD Transactions.csv for use by area_price_stats.

1. On https://www.dubaipulse.gov.ae/data/dld-transactions/dld_transactions-open
   read the terms, then download Transactions.csv into data/raw/dld/.
2. python scripts/load_dld_transactions.py
3. Load it:  Toolbox.load(transactions_file=Path("data/raw/dld/transactions.csv"),
                          transactions_source=dld.SOURCE)

Nothing here is committed: data/raw/ is git-ignored.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from estate_agent import dld  # noqa: E402

RAW = ROOT / "data" / "raw" / "dld"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=RAW / "Transactions.csv")
    parser.add_argument("--output", type=Path, default=RAW / "transactions.csv")
    args = parser.parse_args(argv)
    if not args.input.exists():
        print(f"Not found: {args.input}\nDownload it first; see the instructions at the top of this script.")
        return 1

    report = dld.convert(args.input, args.output)
    print(f"Read {report.rows_read:,} rows, wrote {report.rows_written:,} to {args.output}")
    for reason, count in report.skipped.most_common():
        print(f"  skipped {count:,}: {reason}")
    unmatched = sum(report.unmatched_areas.values())
    if unmatched:
        print(f"  skipped {unmatched:,}: area not in the tools' area list. Most common:")
        for name, count in report.unmatched_areas.most_common(15):
            print(f"    {count:>8,}  {name}")
        print("  Add verified DLD names to DLD_AREA_NAMES in src/estate_agent/dld.py to include them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
