# SPDX-FileCopyrightText: Copyright (c) 2024 bird_sql
# SPDX-License-Identifier: MIT
#
# The comparison rule is ported from the Spider2-lite official evaluator
# (``spider2-lite/evaluation_suite/evaluate.py``) in
# https://github.com/xlang-ai/Spider2, used under the MIT License.
#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spider2-lite's official result comparison, without pandas.

Spider2 has no gold SQL to execute for most questions; it ships gold *result*
CSVs instead, often several acceptable ones per question (``<id>_a.csv``,
``<id>_b.csv``, ...), and a prediction scores if it matches any of them.
Matching is column-wise: every gold column -- or only the question's
``condition_cols`` -- must equal *some* predicted column, numbers within an
absolute 0.01, so extra predicted columns and column order do not matter.

Upstream compares pandas frames, and the frames are not the raw rows: the
prediction is fetched with ``pd.read_sql_query`` in 500-row chunks, written to
CSV and read back with ``pd.read_csv``, the same reader the gold CSVs go
through. That round trip decides what equals what -- ``'007'`` becomes ``7``,
an integer column holding a NULL becomes floats, ``'NA'`` becomes missing and
missing compares as ``0`` -- so it is reproduced here value by value rather
than approximated. ``tests/gym/spider2_reference.py`` keeps upstream's pandas
code and ``tests/gym/test_spider2_match.py`` checks the two agree; the
schema-only server's venv has no pandas, which is why this copy exists.

Upstream's quirks are kept, since they decide official scores. The one that
matters most: with a single ``<id>_a.csv`` and a flat ``condition_cols`` such as
``[2, 3]``, only gold column 2 is checked, and with ``[0, 1]`` every gold column
is (``compare_multi_pandas_table`` indexes the flat list per gold file).
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Sequence

__all__ = [
    "GoldResults",
    "Table",
    "gold_is_empty",
    "load_gold_csv",
    "prediction_table",
    "results_match",
]

TOLERANCE = 1e-2

# pd.read_sql_query(..., chunksize=500), as upstream's get_sqlite_result calls it.
# Column dtypes are inferred per chunk, so a chunk boundary can change how a
# value is written to the intermediate CSV.
_READ_SQL_CHUNK = 500

# pandas' default na_values (pandas._libs.parsers.STR_NA_VALUES).
_NA_STRINGS = frozenset(
    {
        "",
        "#N/A",
        "#N/A N/A",
        "#NA",
        "-1.#IND",
        "-1.#QNAN",
        "-NaN",
        "-nan",
        "1.#IND",
        "1.#QNAN",
        "<NA>",
        "N/A",
        "NA",
        "NULL",
        "NaN",
        "None",
        "n/a",
        "nan",
        "null",
    }
)
_TRUE_STRINGS = frozenset({"True", "TRUE", "true"})
_FALSE_STRINGS = frozenset({"False", "FALSE", "false"})
_INT_RE = re.compile(r"[+-]?\d+")
_FLOAT_RE = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_INF_STRINGS = frozenset({"inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"})
_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1
_UINT64_MAX = 2**64 - 1

_NUMERIC_KINDS = frozenset({"int64", "uint64", "float64"})


@dataclass(frozen=True)
class Table:
    """A frame as upstream sees it after ``read_csv``: one dtype per column."""

    kinds: tuple[str, ...]
    columns: tuple[tuple[Any, ...], ...]

    @property
    def row_count(self) -> int:
        return len(self.columns[0]) if self.columns else 0


@dataclass(frozen=True)
class GoldResults:
    """One question's gold result files and how they are compared.

    ``single`` is upstream's distinction between one ``<id>.csv`` (compared
    directly with ``condition_cols``) and ``<id>_<letter>.csv`` alternatives
    (compared through ``compare_multi_pandas_table``, even when there is one).
    """

    paths: tuple[Path, ...]
    single: bool
    condition_cols: Any = None
    ignore_order: bool = False

    @classmethod
    def from_task(
        cls,
        dataset_dir: Path,
        gold_results: Sequence[str],
        condition_cols: Any = None,
        ignore_order: bool = False,
    ) -> "GoldResults":
        paths = tuple(dataset_dir / rel for rel in gold_results)
        single = len(paths) == 1 and not re.search(r"_[a-z]$", paths[0].stem)
        return cls(paths, single, condition_cols, bool(ignore_order))


def _isna(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _parse_column(cells: Sequence[Optional[str]]) -> tuple[str, tuple[Any, ...]]:
    """Infer one column the way ``pd.read_csv`` does. ``None`` is a missing field."""
    if not cells:
        return "object", ()
    na = [c is None or c in _NA_STRINGS for c in cells]
    present = [c for c, missing in zip(cells, na) if not missing]
    if not present:
        return "float64", tuple(math.nan for _ in cells)
    has_na = any(na)
    stripped = [c.strip(" \t") for c in present]

    if all(_INT_RE.fullmatch(s) for s in stripped):
        ints = [int(s) for s in stripped]
        if has_na:
            it = iter(ints)
            return "float64", tuple(
                math.nan if missing else float(next(it)) for missing in na
            )
        if all(_INT64_MIN <= i <= _INT64_MAX for i in ints):
            return "int64", tuple(ints)
        if all(0 <= i <= _UINT64_MAX for i in ints):
            return "uint64", tuple(ints)
        return "object", tuple(cells)

    if all(_FLOAT_RE.fullmatch(s) or s.lower() in _INF_STRINGS for s in stripped):
        it = iter(stripped)
        return "float64", tuple(
            math.nan if missing else float(next(it)) for missing in na
        )

    if all(c in _TRUE_STRINGS or c in _FALSE_STRINGS for c in present):
        values = tuple(
            math.nan if missing else (c in _TRUE_STRINGS)
            for c, missing in zip(cells, na)
        )
        return ("object" if has_na else "bool"), values

    return "object", tuple(math.nan if missing else c for c, missing in zip(cells, na))


def _parse_rows(header_width: int, rows: Sequence[Sequence[Optional[str]]]) -> Table:
    for row in rows:
        if len(row) > header_width:
            raise ValueError("data row has more fields than the header")
    padded = [list(row) + [None] * (header_width - len(row)) for row in rows]
    parsed = [_parse_column([row[i] for row in padded]) for i in range(header_width)]
    return Table(
        tuple(kind for kind, _ in parsed), tuple(values for _, values in parsed)
    )


@lru_cache(maxsize=None)
def load_gold_csv(path: Path) -> Table:
    """Read a gold CSV as ``pd.read_csv`` would (blank lines skipped)."""
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = [row for row in csv.reader(fh) if row]
    if not rows:
        raise ValueError(f"{path} has no header")
    return _parse_rows(len(rows[0]), rows[1:])


def _chunk_cells(values: Sequence[Any]) -> list[str]:
    """How one column of one read_sql chunk is written by ``DataFrame.to_csv``."""
    kinds = {type(v) for v in values if v is not None}
    has_none = any(v is None for v in values)
    if kinds and kinds <= {int} and not has_none:
        return [str(v) for v in values]
    if kinds and kinds <= {int, float}:
        return ["" if v is None else repr(float(v)) for v in values]
    return ["" if v is None else str(v) for v in values]


def prediction_table(column_count: int, rows: Sequence[Sequence[Any]]) -> Table:
    """The predicted frame upstream compares: SQLite rows after the CSV round trip."""
    cells: list[list[str]] = [[] for _ in rows]
    for start in range(0, len(rows), _READ_SQL_CHUNK):
        chunk = rows[start : start + _READ_SQL_CHUNK]
        for col in range(column_count):
            rendered = _chunk_cells([row[col] for row in chunk])
            for offset, cell in enumerate(rendered):
                cells[start + offset].append(cell)
    return _parse_rows(column_count, cells)


def gold_is_empty(gold: GoldResults) -> bool:
    return all(load_gold_csv(path).row_count == 0 for path in gold.paths)


def _column_vectors(table: Table, positions: Optional[list[int]]) -> list[list[Any]]:
    """``frame.iloc[:, positions].transpose().values.tolist()``.

    The transpose interleaves every selected column into one array, so dtypes
    are unified across columns first: ints beside floats become floats, while
    any bool or object column leaves every value its own Python type.
    """
    if positions is None:
        kinds, columns = list(table.kinds), list(table.columns)
    else:
        width = len(table.columns)
        for p in positions:
            if not -width <= p < width:
                raise IndexError(f"condition column {p} out of range")
        kinds = [table.kinds[p] for p in positions]
        columns = [table.columns[p] for p in positions]

    distinct = set(kinds)
    if len(distinct) > 1 and distinct <= _NUMERIC_KINDS:
        return [[float(v) for v in column] for column in columns]
    return [list(column) for column in columns]


def _normalize(value: Any) -> Any:
    return 0 if _isna(value) else value


def _vectors_match(v1: list, v2: list, ignore_order: bool) -> bool:
    v1 = [_normalize(x) for x in v1]
    v2 = [_normalize(x) for x in v2]
    if ignore_order:
        key = lambda x: (x is None, str(x), isinstance(x, (int, float)))  # noqa: E731
        v1 = sorted(v1, key=key)
        v2 = sorted(v2, key=key)
    if len(v1) != len(v2):
        return False
    for a, b in zip(v1, v2):
        if _isna(a) and _isna(b):
            continue
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if not math.isclose(float(a), float(b), abs_tol=TOLERANCE):
                return False
        elif a != b:
            return False
    return True


def _compare_table(
    pred: Table, gold: Table, condition_cols: Any, ignore_order: bool
) -> bool:
    """Upstream ``compare_pandas_table``."""
    positions: Optional[list[int]] = None
    if condition_cols:
        if not isinstance(condition_cols, (list, tuple)):
            condition_cols = [condition_cols]
        positions = [int(c) for c in condition_cols]
    gold_vectors = _column_vectors(gold, positions)
    pred_vectors = _column_vectors(pred, None)
    return all(
        any(_vectors_match(g, p, ignore_order) for p in pred_vectors)
        for g in gold_vectors
    )


def _compare_multi(
    pred: Table, golds: list[Table], condition_cols: Any, ignore_order: bool
) -> bool:
    """Upstream ``compare_multi_pandas_table``."""
    if not golds:
        return False
    if condition_cols in (None, [], [[]], [None]):
        condition_cols = [[] for _ in golds]
    elif len(golds) > 1 and not all(isinstance(c, list) for c in condition_cols):
        condition_cols = [condition_cols for _ in golds]
    return any(
        _compare_table(pred, gold, condition_cols[i], ignore_order)
        for i, gold in enumerate(golds)
    )


def results_match(pred: Table, gold: GoldResults) -> bool:
    """Score a predicted frame against a question's gold results.

    A comparison that raises -- a condition column the gold does not have, a
    malformed CSV -- scores 0, as upstream's ``Python Script Error`` path does.
    """
    try:
        golds = [load_gold_csv(path) for path in gold.paths]
        if gold.single:
            return _compare_table(
                pred, golds[0], gold.condition_cols, gold.ignore_order
            )
        return _compare_multi(pred, golds, gold.condition_cols, gold.ignore_order)
    except (ValueError, IndexError, TypeError):
        return False
