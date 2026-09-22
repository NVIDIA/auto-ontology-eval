#!/usr/bin/env python3
"""Merge eval shard CSVs into one file ordered by ``row_index``.

Later files win on duplicate ``row_index`` so a re-run shard supersedes an
earlier partial one.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Eval rows carry full result sets; the 128 KiB default field limit is too small.
csv.field_size_limit(sys.maxsize)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge eval shard CSVs.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()

    by_index: dict[int, dict[str, str]] = {}
    fieldnames: list[str] | None = None

    for path in args.inputs:
        if not path.exists():
            print(f"  skip (missing): {path}")
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames:
                fieldnames = list(reader.fieldnames)
            rows = list(reader)
        for row in rows:
            by_index[int(row["row_index"])] = row
        print(f"  {path.name}: {len(rows)} rows")

    if fieldnames is None:
        raise SystemExit("No input rows found; nothing to merge.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in sorted(by_index):
            writer.writerow(by_index[index])

    print(f"\nmerged {len(by_index)} unique rows -> {args.output}")
    expected = set(range(min(by_index), max(by_index) + 1))
    gaps = sorted(expected - set(by_index))
    if gaps:
        print(f"WARNING: missing row_index values: {gaps}")


if __name__ == "__main__":
    main()
