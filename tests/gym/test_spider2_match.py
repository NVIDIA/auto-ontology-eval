# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Gym verifier's Spider2-lite rule must agree with the official evaluator.

``ontology_sql_eval.gym.spider2`` reimplements upstream's pandas comparison --
including the read_sql -> to_csv -> read_csv round trip that decides how values
are typed -- because the schema-only server's venv has no pandas.
``tests/gym/spider2_reference.py`` keeps upstream's code; this test is what
stops the two drifting apart.
"""

from __future__ import annotations

import asyncio
import csv
import json
import random
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from ontology_sql_eval.gym.exec_match import (
    GoldResults,
    SqliteExecutor,
    execute_and_compare,
)
from ontology_sql_eval.gym.tasks import DATASETS, build_records
from ontology_sql_eval.paths import REPO_ROOT
from tests.gym.spider2_reference import score_sqlite

DB_ID = "db"


def _make_db(root: Path, rows: list[tuple]) -> Path:
    """A table ``t`` whose untyped columns hold *rows* exactly as given."""
    path = root / DB_ID / f"{DB_ID}.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    width = len(rows[0]) if rows else 1
    with sqlite3.connect(path) as conn:
        conn.execute(f"CREATE TABLE t ({', '.join(f'c{i}' for i in range(width))})")
        conn.executemany(f"INSERT INTO t VALUES ({', '.join('?' * width)})", rows)
    return path


def _write_gold(root: Path, name: str, text: str) -> Path:
    path = root / "gold" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _ours(root: Path, sql: str, gold: GoldResults) -> bool:
    match, err, _ = asyncio.run(
        execute_and_compare(SqliteExecutor(root), DB_ID, "", sql, "spider2_lite", gold)
    )
    assert err is None, err
    return match


def _official(db: Path, sql: str, gold: GoldResults) -> bool:
    return bool(
        score_sqlite(
            db,
            sql,
            list(gold.paths),
            gold.single,
            gold.condition_cols,
            gold.ignore_order,
        )
    )


# (label, rows in t, query, gold CSV texts, condition_cols, ignore_order).
# One gold text is scored as a single <id>.csv; several as <id>_a, _b, ...
CASES: list[tuple[str, list[tuple], str, list[str], Any, bool]] = [
    (
        "exact",
        [(1, "a"), (2, "b")],
        "SELECT c0, c1 FROM t",
        ["x,y\n1,a\n2,b\n"],
        None,
        True,
    ),
    (
        "row order kept",
        [(1, "a"), (2, "b")],
        "SELECT c0, c1 FROM t ORDER BY c0 DESC",
        ["x,y\n1,a\n2,b\n"],
        None,
        False,
    ),
    (
        "row order ignored",
        [(1, "a"), (2, "b")],
        "SELECT c0, c1 FROM t ORDER BY c0 DESC",
        ["x,y\n1,a\n2,b\n"],
        None,
        True,
    ),
    (
        "extra pred column",
        [(1, "a", 9)],
        "SELECT c2, c1, c0 FROM t",
        ["x,y\n1,a\n"],
        None,
        True,
    ),
    ("missing pred column", [(1, "a")], "SELECT c0 FROM t", ["x,y\n1,a\n"], None, True),
    ("within tolerance", [(1.004,)], "SELECT c0 FROM t", ["x\n1\n"], None, True),
    ("outside tolerance", [(1.02,)], "SELECT c0 FROM t", ["x\n1\n"], None, True),
    (
        "null equals zero",
        [(None,), (2,)],
        "SELECT c0 FROM t",
        ["x\n0\n2\n"],
        None,
        True,
    ),
    ("leading zeros parse", [("007",)], "SELECT c0 FROM t", ["x\n7\n"], None, True),
    ("NA string is missing", [("NA",)], "SELECT c0 FROM t", ["x\n0\n"], None, True),
    ("case sensitive", [("Apple",)], "SELECT c0 FROM t", ["x\napple\n"], None, True),
    (
        "bool strings",
        [("True",), ("false",)],
        "SELECT c0 FROM t",
        ["x\nTrue\nFalse\n"],
        None,
        True,
    ),
    (
        "int beside float",
        [(1, 2.5)],
        "SELECT c0, c1 FROM t",
        ["x,y\n1.0,2.5\n"],
        None,
        True,
    ),
    (
        "mixed str int",
        [(1, "x"), ("y", 2)],
        "SELECT c0, c1 FROM t",
        ["a,b\n1,x\ny,2\n"],
        None,
        True,
    ),
    ("condition cols", [(1, "a")], "SELECT c0 FROM t", ["x,y\n1,zzz\n"], [0], True),
    ("condition col out of range", [(1,)], "SELECT c0 FROM t", ["x\n1\n"], [3], True),
    ("empty vs empty gold", [(1,)], "SELECT c0 FROM t WHERE 0", ["x\n"], None, True),
    ("empty vs gold rows", [(1,)], "SELECT c0 FROM t WHERE 0", ["x\n1\n"], None, True),
    ("any alternative", [(2,)], "SELECT c0 FROM t", ["x\n1\n", "x\n2\n"], None, True),
    ("no alternative", [(3,)], "SELECT c0 FROM t", ["x\n1\n", "x\n2\n"], None, True),
    # One _a file with a flat condition_cols: upstream checks cols[0] only.
    (
        "flat cols one alternative",
        [(5, 1)],
        "SELECT c0 FROM t",
        ["x,y\n9,5\n"],
        [1, 0],
        True,
    ),
    (
        "flat cols zero first",
        [(5, 1)],
        "SELECT c0 FROM t",
        ["x,y\n5,9\n"],
        [0, 1],
        True,
    ),
    (
        "nested cols",
        [(5, 1)],
        "SELECT c0 FROM t",
        ["x,y\n9,9\n", "x,y\n9,5\n"],
        [[0], [1]],
        True,
    ),
    (
        "huge int",
        [("99999999999999999999",)],
        "SELECT c0 FROM t",
        ["x\n99999999999999999999\n"],
        None,
        True,
    ),
    ("quoted newline", [("a\nb",)], "SELECT c0 FROM t", ['x\n"a\nb"\n'], None, True),
    ("blank gold line", [(1,), (2,)], "SELECT c0 FROM t", ["x\n1\n\n2\n"], None, True),
    ("unicode", [("café",)], "SELECT c0 FROM t", ["x\ncafé\n"], None, True),
]


@pytest.mark.parametrize(
    "label,rows,sql,golds,condition_cols,ignore_order",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_matches_official_evaluator(
    tmp_path, label, rows, sql, golds, condition_cols, ignore_order
):
    db = _make_db(tmp_path, rows)
    if len(golds) == 1 and not label.startswith("flat cols"):
        paths = [_write_gold(tmp_path, "q.csv", golds[0])]
    else:
        paths = [
            _write_gold(tmp_path, f"q_{chr(ord('a') + i)}.csv", text)
            for i, text in enumerate(golds)
        ]
    gold = GoldResults.from_task(
        tmp_path,
        [str(p.relative_to(tmp_path)) for p in paths],
        condition_cols,
        ignore_order,
    )
    official = _official(db, sql, gold)
    assert _ours(tmp_path, sql, gold) is official, f"{label}: official={official}"


def test_chunk_boundary_changes_how_values_are_written(tmp_path):
    """read_sql infers dtypes per 500-row chunk, and the CSV keeps the result.

    The first chunk holds a string, so its ints are written as ``5``; the
    second holds a NULL, so its ints are written as ``5.0``. The column stays
    textual, and ``'5.0'`` then differs from a gold ``5``.
    """
    rows = [("x",)] + [(5,)] * 499 + [(None,)] + [(5,)] * 99
    db = _make_db(tmp_path, rows)
    gold_text = "v\nx\n" + "5\n" * 499 + "\n" + "5\n" * 99
    path = _write_gold(tmp_path, "q.csv", gold_text.replace("\n\n", '\n""\n'))
    gold = GoldResults.from_task(
        tmp_path, [str(path.relative_to(tmp_path))], None, False
    )
    sql = "SELECT c0 FROM t ORDER BY rowid"
    assert _ours(tmp_path, sql, gold) is _official(db, sql, gold) is False


_POOL: list[Any] = [
    0,
    1,
    -3,
    7,
    10,
    2**40,
    2.0,
    0.1,
    1.005,
    2.5,
    -0.0,
    1e16,
    "a",
    "B",
    "007",
    " 5",
    "NA",
    "None",
    "True",
    "false",
    "",
    "1e5",
    "inf",
    "x,y",
    'q"q',
    "1_000",
    None,
]


def _render(value: Any, rng: random.Random) -> Any:
    """A gold cell for *value*: usually faithful, sometimes perturbed."""
    roll = rng.random()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if roll < 0.15:
            return float(value) + rng.choice([0.004, 0.05])
        if roll < 0.3:
            return float(value)
    if value is None and roll < 0.5:
        return 0
    if roll > 0.95:
        return rng.choice(_POOL)
    return value


def _fuzz_case(rng: random.Random):
    width = rng.randint(1, 3)
    kinds = [rng.choice(["int", "float", "str", "mixed"]) for _ in range(width)]

    def cell(kind: str) -> Any:
        if rng.random() < 0.1:
            return None
        if kind == "int":
            return rng.choice([0, 1, -3, 7, 10, 2**40])
        if kind == "float":
            return rng.choice([2.0, 0.1, 1.005, 2.5, 1e16, 3])
        if kind == "str":
            return rng.choice(["a", "B", "007", " 5", "NA", "True", "false", "x,y"])
        return rng.choice(_POOL)

    n_rows = rng.choice([0, 1, 2, 3, 5, 8])
    rows = [tuple(cell(k) for k in kinds) for _ in range(n_rows)]
    gold_rows = [tuple(_render(v, rng) for v in row) for row in rows]
    if gold_rows and rng.random() < 0.15:
        gold_rows.pop(rng.randrange(len(gold_rows)))
    if rng.random() < 0.5:
        rng.shuffle(gold_rows)
    gold_cols = list(range(width))
    if width > 1 and rng.random() < 0.3:
        gold_cols = rng.sample(gold_cols, rng.randint(1, width))
    condition_cols = rng.choice(
        [None, [], [0], list(range(len(gold_cols)))[-1:], [[0], [0]], [width + 1]]
    )
    return (
        rows,
        gold_rows,
        gold_cols,
        condition_cols,
        rng.random() < 0.7,
        rng.choice([1, 1, 2]),
    )


def _csv_text(header: list[str], rows: list[tuple]) -> str:
    import io

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow(["" if v is None else v for v in row])
    return buf.getvalue()


@pytest.mark.parametrize("seed", range(400))
def test_fuzz_agrees_with_official_evaluator(tmp_path, seed):
    rng = random.Random(seed)
    rows, gold_rows, gold_cols, condition_cols, ignore_order, n_gold = _fuzz_case(rng)
    width = len(rows[0]) if rows else 1
    db = _make_db(tmp_path, rows or [tuple([1] * width)])
    where = "" if rows else " WHERE 0"
    sql = f"SELECT {', '.join(f'c{i}' for i in range(width))} FROM t{where} ORDER BY rowid"
    text = _csv_text(
        [f"g{i}" for i in gold_cols],
        [tuple(r[i] for i in gold_cols) for r in gold_rows],
    )
    if n_gold == 1 and rng.random() < 0.5:
        paths = [_write_gold(tmp_path, "q.csv", text)]
    else:
        paths = [_write_gold(tmp_path, "q_a.csv", text)]
        if n_gold == 2:
            paths.append(_write_gold(tmp_path, "q_b.csv", _csv_text(["z"], [(1,)])))
    gold = GoldResults.from_task(
        tmp_path,
        [str(p.relative_to(tmp_path)) for p in paths],
        condition_cols,
        ignore_order,
    )
    official = _official(db, sql, gold)
    assert _ours(tmp_path, sql, gold) is official, (
        f"seed {seed}: official={official} rows={rows} gold={text!r} "
        f"cols={condition_cols} ignore_order={ignore_order}"
    )


def test_missing_gold_file_excludes_the_task(tmp_path):
    _make_db(tmp_path, [(1,)])
    gold = GoldResults.from_task(tmp_path, ["gold/absent_a.csv"], None, True)
    match, err, gold_result = asyncio.run(
        execute_and_compare(
            SqliteExecutor(tmp_path), DB_ID, "", "SELECT 1", "spider2_lite", gold
        )
    )
    assert (match, err, gold_result) == (False, "gold_sql_error", None)


def test_failed_prediction_is_a_scored_miss(tmp_path):
    _make_db(tmp_path, [(1,)])
    path = _write_gold(tmp_path, "q.csv", "x\n1\n")
    gold = GoldResults.from_task(tmp_path, [str(path.relative_to(tmp_path))])
    executor = SqliteExecutor(tmp_path)
    for sql, tag in (("SELEC nope", "pred_sql_error"), (None, "pred_sql_missing")):
        match, err, gold_result = asyncio.run(
            execute_and_compare(executor, DB_ID, "", sql, "spider2_lite", gold)
        )
        assert (match, err, gold_result) == (False, tag, "nonempty_gold")


def test_prediction_cannot_write_to_the_database(tmp_path):
    db = _make_db(tmp_path, [(1,)])
    path = _write_gold(tmp_path, "q.csv", "x\n1\n")
    gold = GoldResults.from_task(tmp_path, [str(path.relative_to(tmp_path))])
    match, err, _ = asyncio.run(
        execute_and_compare(
            SqliteExecutor(tmp_path), DB_ID, "", "DELETE FROM t", "spider2_lite", gold
        )
    )
    assert (match, err) == (False, "pred_sql_error")
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT count(*) FROM t").fetchone() == (1,)


def test_records_carry_the_gold_results(tmp_path):
    question = {
        "question_id": "local002",
        "db_id": "e_commerce",
        "question": "q?",
        "evidence": "",
        "SQL": "",
        "gold_results": ["gold/local002_a.csv", "gold/local002_b.csv"],
        "condition_cols": [[0], [1]],
        "ignore_order": True,
    }
    path = tmp_path / "datasets" / "spider2_lite" / "evaluation.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps([question]))
    [record] = build_records(DATASETS["spider2_lite"], tmp_path, "auto_ontology")
    assert record["task_id"] == "spider2_lite-local002"
    assert record["gold_results"] == question["gold_results"]
    assert record["condition_cols"] == [[0], [1]]
    assert record["ignore_order"] is True
    assert GoldResults.from_task(tmp_path, record["gold_results"]).single is False
    assert GoldResults.from_task(tmp_path, ["gold/local010.csv"]).single is True


_SEEDED = REPO_ROOT / "datasets" / "spider2_lite" / "evaluation.json"


@pytest.mark.skipif(not _SEEDED.is_file(), reason="spider2_lite is not seeded")
def test_seeded_gold_sql_agrees_with_official_evaluator():
    """Upstream's own gold SQL, and two ways of getting it wrong, on the real data."""
    dataset_dir = _SEEDED.parent
    executor = SqliteExecutor(dataset_dir, timeout_s=300)
    checked = 0
    for q in json.loads(_SEEDED.read_text()):
        if not q["SQL"]:
            continue
        db = dataset_dir / q["db_id"] / f"{q['db_id']}.sqlite"
        gold = GoldResults.from_task(
            dataset_dir, q["gold_results"], q["condition_cols"], q["ignore_order"]
        )
        base = q["SQL"].strip().rstrip(";")
        for sql in (
            base,
            f"SELECT * FROM ({base}) LIMIT 1",
            f"SELECT 1 AS one FROM ({base})",
        ):
            official = _official(db, sql, gold)
            match, err, _ = asyncio.run(
                execute_and_compare(executor, q["db_id"], "", sql, "spider2_lite", gold)
            )
            if err is not None:
                assert not official, (
                    f"{q['question_id']}: we failed ({err}), official scored"
                )
                continue
            assert match is official, (
                f"{q['question_id']}: official={official} sql={sql[:80]}"
            )
            checked += 1
    assert checked
