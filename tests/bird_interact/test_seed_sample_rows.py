# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "bird_interact"
    / "seed_bird_interact.py"
)
_spec = importlib.util.spec_from_file_location("seed_bird_interact", _SCRIPT)
seed = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(seed)

SCHEMA = """\
"CREATE" TABLE "RiskManagement" (
"RISKTRACE" bigint NOT NULL,
risk_profile jsonb NULL,
    "PRIMARY" KEY (RISKTRACE)
);



"First" 3 rows:
  RISKTRACE  risk_profile
-----------  -----------------------------------------------------------------------------
          1  {'assessment': {'risk_score': 75.89, 'level': 'Medium'}, 'ratio': '1.95 USD/km'}
          2  {'assessment': {'risk_score': None, 'level': 'High'}, 'tags': ['a', 'b'], 'rat
...
"""


def test_sample_rows_reads_each_table_block_by_column_span():
    rows = seed._sample_rows(SCHEMA)

    assert rows["riskmanagement"]["risktrace"] == ["1", "2"]
    assert rows["riskmanagement"]["risk_profile"][1].endswith("'rat")


def test_json_leaf_values_keeps_complete_leaves_of_a_truncated_repr():
    cell = "{'a': {'b': 1.5, 'c': 'x y'}, 'd': [1, 2], 'e': None, 'f': 'cut"

    assert seed._json_leaf_values(cell) == {"a.b": "1.5", "a.c": "'x y'", "e": "None"}


def test_json_leaf_values_drops_a_number_the_cut_may_have_shortened():
    assert seed._json_leaf_values("{'a': 1, 'b': 27") == {"a": "1"}


def test_field_samples_lists_distinct_non_null_values_in_row_order():
    cells = seed._sample_rows(SCHEMA)["riskmanagement"]["risk_profile"]

    assert seed._field_samples(cells) == {
        "assessment.risk_score": ["75.89"],
        "assessment.level": ["'Medium'", "'High'"],
        "ratio": ["'1.95 USD/km'"],
    }


def test_column_meaning_text_appends_stored_values_to_each_field_line():
    meaning = {
        "column_meaning": "JSONB column.",
        "fields_meaning": {
            "assessment": {"level": "TEXT. Risk level."},
            "ratio": "REAL. Cost density. Example: 1.95.",
        },
    }
    samples = {"ratio": ["'1.95 USD/km'"]}

    assert seed._column_meaning_text(meaning, samples) == (
        "JSONB column.\n"
        "assessment.level: TEXT. Risk level.\n"
        "ratio: REAL. Cost density. Example: 1.95. "
        "Stored values in sample rows: '1.95 USD/km'."
    )


def test_column_meaning_text_leaves_plain_meanings_unchanged():
    assert seed._column_meaning_text("TEXT. A name.", {"x": ["1"]}) == "TEXT. A name."
