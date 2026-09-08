#!/usr/bin/env python3
"""Re-execute gold and agent SQL to see how much of the score is strictness.

Execution match requires identical shape and identical value order within each
row. This re-runs both queries per question and re-compares under progressively
looser criteria, so a low score can be attributed to genuinely wrong SQL rather
than to the comparison being stricter than the benchmark intends.
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
csv.field_size_limit(sys.maxsize)

from ontology_sql_eval.retrieval.scoring import (  # noqa: E402
    _canonical,
    _execute_sql,
)


def rows_of(df) -> list[tuple]:
    return [tuple(_canonical(v) for v in row) for row in df.values.tolist()]


def sort_key(row: tuple) -> tuple:
    """None-safe ordering; plain sorted() raises on None mixed with str."""
    return tuple((v is None, "" if v is None else str(v)) for v in row)


def has_null(rows: list[tuple]) -> bool:
    return any(v is None for row in rows for v in row)


def eq(a: list[tuple], b: list[tuple]) -> bool:
    return sorted(a, key=sort_key) == sorted(b, key=sort_key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()

    from gsf.connectors.registry import get_connectors

    connectors = {
        name: c
        for c in get_connectors()
        if (name := getattr(c, "database_name", None))
    }
    connector = connectors.get("dw") or next(iter(connectors.values()))

    rows = list(csv.DictReader(args.csv.open(encoding="utf-8", newline="")))
    verdict = collections.Counter()

    for r in rows:
        gold_df, gold_err = _execute_sql(connector, r["expected_sql"])
        got_df, got_err = _execute_sql(connector, r["returned_sql"])

        if gold_df is None:
            verdict["gold query failed"] += 1
            continue
        if got_df is None:
            verdict["agent query failed or empty"] += 1
            continue

        g, a = rows_of(gold_df), rows_of(got_df)

        if eq(g, a):
            verdict["MATCH (values identical)"] += 1
            if has_null(g):
                verdict["  ^ of those, blocked by NULL sort bug"] += 1
        elif eq(
            [tuple(sorted((str(v) for v in t))) for t in g],
            [tuple(sorted((str(v) for v in t))) for t in a],
        ):
            verdict["same values, column order differs"] += 1
        elif set(g) == set(a):
            verdict["same row set, duplicate counts differ"] += 1
        elif gold_df.shape == got_df.shape:
            verdict["same shape, different values"] += 1
        elif got_df.shape[0] == 0:
            verdict["agent returned zero rows"] += 1
        elif gold_df.shape[1] != got_df.shape[1]:
            verdict["different column count"] += 1
        else:
            verdict["different row count"] += 1

    total = sum(verdict.values())
    print(f"re-executed {total} questions\n")
    for label, count in verdict.most_common():
        print(f"  {label:<38} {count}")


if __name__ == "__main__":
    main()
