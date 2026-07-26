# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scoring and result-parsing helpers for the text-to-SQL chatbot evaluation.

These functions compare the agent's SQL/answer against the expected values:
executing both SQL queries and comparing row sets, parsing DB/markdown results
into DataFrames, and computing text/numeric similarity. They are pure and have
no dependency on the eval driver, CLI, or CSV layout.
"""

from __future__ import annotations

import difflib
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from nemo_retriever.tabular_data.sql_database import SQLDatabase


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_SCORE_STR_LIMIT = 8000
_STRINGIFY_ROW_LIMIT = 200


def normalize_text(s: str) -> str:
    """Lowercase, collapse whitespace, drop trailing semicolons."""
    if s is None:
        return ""
    s = str(s).strip().rstrip(";")
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def _sql_text_similarity(expected: str, actual: str) -> float:
    a = normalize_text(expected)
    b = normalize_text(actual)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _df_values_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Compare two DataFrames by row-multiset of values, ignoring column names/order."""
    try:
        if a.shape != b.shape:
            return False
        a_rows = sorted(tuple(_canonical(v) for v in row) for row in a.values.tolist())
        b_rows = sorted(tuple(_canonical(v) for v in row) for row in b.values.tolist())
        return a_rows == b_rows
    except Exception:
        return False


def _canonical(value: Any) -> Any:
    """Make a value hashable and comparable across small numeric/string drift."""
    if value is None:
        return None
    if isinstance(value, float):
        # Round to mitigate float jitter from aggregations
        return round(value, 4)
    return str(value).strip().lower()


def _execute_sql(
    connector: SQLDatabase, sql: str, schema: str = ""
) -> Tuple[Optional[pd.DataFrame], str]:
    if not sql or not sql.strip():
        return None, "empty SQL"
    try:
        if schema and getattr(connector, "dialect", "") == "duckdb":
            connector.execute(f"SET schema = '{schema}'")
        df = connector.execute(sql)
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        return df, ""
    except Exception as exc:  # pragma: no cover - tooling script
        return None, f"{type(exc).__name__}: {exc}"


def score_sql(
    connector: SQLDatabase, expected: str, actual: str, schema: str = ""
) -> Dict[str, Any]:
    text_sim = _sql_text_similarity(expected, actual)
    expected_df, expected_err = _execute_sql(connector, expected, schema=schema)
    actual_df, actual_err = _execute_sql(connector, actual, schema=schema)
    exec_match = 0
    if expected_df is not None and actual_df is not None:
        exec_match = 1 if _df_values_equal(expected_df, actual_df) else 0
    expected_result = stringify_db_result(expected_df) if expected_df is not None else ""
    return {
        "sql_text_similarity": round(text_sim, 4),
        "sql_exec_match": exec_match,
        "expected_sql_error": expected_err,
        "returned_sql_error": actual_err,
        "expected_sql_result": expected_result,
    }


def _extract_numbers(text: str) -> List[float]:
    if not text:
        return []
    out = []
    for tok in _NUM_RE.findall(str(text)):
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


def _parse_markdown_table(md: str) -> Optional[pd.DataFrame]:
    """Parse a simple markdown table into a DataFrame, or None on failure."""
    if not md:
        return None
    lines = [ln.strip() for ln in md.strip().splitlines() if ln.strip()]
    # Need at least header + separator + one data row
    if len(lines) < 3:
        return None
    data_lines = [ln for ln in lines if not re.match(r"^\|[\s:_-]+\|$", ln)]
    if len(data_lines) < 2:
        return None
    header = [c.strip() for c in data_lines[0].strip("|").split("|")]
    rows = []
    for line in data_lines[1:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == len(header):
            rows.append(cells)
    if not rows:
        return None
    return pd.DataFrame(rows, columns=pd.Index(header))


def _db_result_to_df(value: str) -> Optional[pd.DataFrame]:
    """Try to parse the stringified DB result into a DataFrame."""
    if not value:
        return None
    text = str(value).strip()
    # Unwrap outer list wrapper like ['[{"count":712}]']
    if text.startswith("[") and text.endswith("]"):
        try:
            outer = json.loads(text)
            if (
                isinstance(outer, list)
                and len(outer) == 1
                and isinstance(outer[0], str)
            ):
                text = outer[0]
        except (json.JSONDecodeError, TypeError):
            pass
    # Try JSON array of objects
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return pd.DataFrame(parsed)
        if isinstance(parsed, dict):
            return pd.DataFrame([parsed])
    except (json.JSONDecodeError, TypeError):
        pass
    # Try CSV
    try:
        from io import StringIO

        df = pd.read_csv(StringIO(text))
        if not df.empty:
            return df
    except Exception:
        pass
    return None


def score_answer(expected_raw: str, returned_db_str: str) -> Dict[str, Any]:
    """Score the agent's answer against the expected ``answer_raw`` markdown table.

    Strategy (in priority order):
    1. Parse both sides into DataFrames and compare row-multisets (structural match).
    2. Compare the multiset of numeric values (survives formatting differences).
    3. Fall back to fuzzy text similarity.
    """
    expected_df = _parse_markdown_table(expected_raw)
    if expected_df is None:
        expected_df = _db_result_to_df(expected_raw)
    actual_df = _db_result_to_df(returned_db_str)

    structural_match = 0
    if expected_df is not None and actual_df is not None:
        structural_match = 1 if _df_values_equal(expected_df, actual_df) else 0

    # Truncate large strings before expensive text operations
    haystack = str(returned_db_str or "")[:_SCORE_STR_LIMIT]
    expected_capped = str(expected_raw or "")[:_SCORE_STR_LIMIT]

    expected_nums = sorted(round(n, 4) for n in _extract_numbers(expected_capped))
    actual_nums = sorted(round(n, 4) for n in _extract_numbers(haystack))
    if not expected_nums and not actual_nums:
        nums_match = 1
    else:
        nums_match = 1 if expected_nums and expected_nums == actual_nums else 0

    sim = (
        difflib.SequenceMatcher(
            None, normalize_text(expected_capped), normalize_text(haystack)
        ).ratio()
        if expected_capped and haystack
        else 0.0
    )
    if structural_match:
        sim = 1.0
    elif nums_match:
        sim = max(sim, 1.0)

    return {
        "answer_text_similarity": round(sim, 4),
        "answer_numbers_match": nums_match,
    }


def stringify_db_result(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, pd.DataFrame):
        return value.head(_STRINGIFY_ROW_LIMIT).to_csv(index=False)
    # Some agent paths return executed rows as a plain ``list[dict]`` — serialise
    # as JSON so the downstream parser hits the JSON branch (rather than
    # ``str(...)`` which uses single quotes and breaks json.loads).
    if isinstance(value, list) and value and isinstance(value[0], dict):
        try:
            return json.dumps(value[:_STRINGIFY_ROW_LIMIT], default=str)
        except (TypeError, ValueError):
            return str(value[:_STRINGIFY_ROW_LIMIT])
    return str(value)[:_SCORE_STR_LIMIT]
