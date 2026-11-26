# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the TaskGen pilot's synthetic data, ontology and oracle in agreement."""

import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "analytics_shipping"
QUESTIONS = json.loads((FIXTURE / "questions.json").read_text())
ORACLE = json.loads((FIXTURE / "oracle.json").read_text())


@pytest.fixture
def data():
    db = sqlite3.connect(":memory:")
    try:
        db.executescript((FIXTURE / "source.sql").read_text())
        db.execute("PRAGMA query_only=ON")
        yield db
    finally:
        db.close()


def rows(db, sql):
    return [list(row) for row in db.execute(sql)]


def test_questions_and_oracle_are_separate_and_have_the_same_task_ids():
    cases = QUESTIONS["cases"]
    assert QUESTIONS["fixture_version"] == ORACLE["fixture_version"] == 1
    assert {case["task_id"] for case in cases} == ORACLE["cases"].keys()
    assert len(cases) == len(ORACLE["cases"])
    assert all(set(case) == {"task_id", "question"} for case in cases)


@pytest.mark.parametrize("task_id", ["paid_revenue", "shipped_revenue"])
def test_grounded_answers_match_independent_expected_rows(data, task_id):
    gold = ORACLE["cases"][task_id]
    assert gold["outcome"] == "answer"
    assert rows(data, gold["sql"]) == gold["rows"]


def test_oracle_detects_double_counting_and_the_wrong_time_axis(data):
    gold = ORACLE["cases"]["shipped_revenue"]
    # One West order has two parcels; a detail join counts its amount twice.
    naive = gold["sql"].replace(
        "FROM orders o", "FROM orders o JOIN shipments x ON x.order_id = o.id"
    )
    assert rows(data, naive) == [["East", 20000], ["West", 25000]]
    # The qualifying East order shipped in November, but was placed in October.
    shipment_month = gold["sql"].replace(
        "s.order_id = o.id", "s.order_id = o.id AND s.shipped_date < '2026-11-01'"
    )
    assert rows(data, shipment_month) == [["West", 12500]]
    assert rows(data, naive) != gold["rows"] != rows(data, shipment_month)


def test_ambiguity_has_distinct_verified_resolutions_and_missing_cost_is_not_zero(data):
    case = ORACLE["cases"]["clarify_month"]
    assert case["outcome"] == "clarify" and case["missing_slot"] == "reporting_month"
    assert "sql" not in case
    resolutions = [
        rows(data, alternative["sql"]) for alternative in case["alternatives"]
    ]
    assert resolutions == [alternative["rows"] for alternative in case["alternatives"]]
    assert resolutions == [[[20000]], [[3000]]]
    case = ORACLE["cases"]["missing_cost"]
    assert case == {"outcome": "unanswerable", "missing_input": "cost of goods sold"}
    tables = [
        row[0]
        for row in data.execute("SELECT name FROM sqlite_schema WHERE type = 'table'")
    ]
    assert set(tables) == {"orders", "shipments"}
    for table in ("orders", "shipments"):
        assert all(
            "cost" not in row[1] for row in data.execute(f"PRAGMA table_info({table})")
        )


def test_native_ontology_document_matches_the_physical_and_semantic_contract(data):
    # The adapter tests stub the auto_ontology package during collection.
    # Validate against the real model in a fresh interpreter, not those stubs.
    validated = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, yaml; "
            "from pathlib import Path; "
            "from auto_ontology.server.model_interchange.schemas import AutoOntologyModelDocument; "
            "data = yaml.safe_load(Path(sys.argv[1]).read_text()); "
            "print(AutoOntologyModelDocument.model_validate(data).model_dump_json())",
            str(FIXTURE / "model.yaml"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    doc = json.loads(validated.stdout, object_hook=lambda data: SimpleNamespace(**data))
    (database,) = doc.data_layer.databases
    assert database.dialect == "sqlite"
    (schema,) = database.schemas
    for table in schema.tables:
        assert table.name in {"orders", "shipments"}
        actual = list(data.execute(f"PRAGMA table_info({table.name})"))
        assert [
            (column.name, column.type, column.is_nullable) for column in table.columns
        ] == [(row[1], row[2], not row[3]) for row in actual]
        assert table.pk == [row[1] for row in actual if row[5]]
    (fk,) = doc.data_layer.foreign_keys
    assert (fk.source_column_id, fk.target_column_id) == (
        "shipment-order-id",
        "order-id",
    )
    assert [row[2:5] for row in data.execute("PRAGMA foreign_key_list(shipments)")] == [
        ("orders", "order_id", "id")
    ]
    (measure,) = doc.semantic_layer.sql_attributes.manual
    assert measure.term_id == "order-term"
    assert measure.sql_column_is == ["order-net-cents", "order-status"]
    assert {
        case["required_metric"]
        for case in ORACLE["cases"].values()
        if case["outcome"] == "answer"
    } == {measure.id}
    assert rows(data, measure.sql) == [[59000]]
    assert rows(data, measure.sql + " AND 1 = 0") == [[None]]
