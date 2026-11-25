from __future__ import annotations

import csv
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


def _import_seed():
    """Import the seeder against the real ``auto_ontology``.

    ``tests/bird_interact/test_server.py`` replaces ``auto_ontology`` in
    ``sys.modules`` at collection time, and its tests need the stubs back.
    """
    stubbed = "auto_ontology" in sys.modules and not hasattr(
        sys.modules["auto_ontology"], "__path__"
    )
    stubs = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if stubbed and name.split(".")[0] == "auto_ontology"
    }
    try:
        return importlib.import_module("scripts.seed.seed_spider2_lite_bq")
    finally:
        if stubs:
            for name in [m for m in sys.modules if m.split(".")[0] == "auto_ontology"]:
                del sys.modules[name]
            sys.modules.update(stubs)


seed = _import_seed()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _ddl(dataset: str, table: str, body: str) -> str:
    return f"CREATE TABLE `{dataset}.{table}`\n(\n{body}\n);"


def _fake_upstream(root: Path) -> Path:
    """One GA4-style database: a sharded events table and a plain one."""
    lite = root / "spider2-lite"
    _write_jsonl(
        lite / "spider2-lite.jsonl",
        [
            {
                "instance_id": "ga900",
                "db": "ga4",
                "question": "Q",
                "external_knowledge": None,
            },
            {
                "instance_id": "local001",
                "db": "f1",
                "question": "local",
                "external_knowledge": None,
            },
        ],
    )
    gold = lite / "evaluation_suite" / "gold"
    _write_jsonl(
        gold / "spider2lite_eval.jsonl",
        [{"instance_id": "ga900", "condition_cols": [[0], [1]], "ignore_order": True}],
    )
    (gold / "sql").mkdir(parents=True)
    (gold / "sql" / "ga900.sql").write_text("SELECT 1\n")
    (gold / "exec_result").mkdir()
    (gold / "exec_result" / "ga900_a.csv").write_text("x\n1\n")
    (gold / "exec_result" / "ga900_b.csv").write_text("x,y\n1,2\n")
    (lite / "resource" / "documents").mkdir(parents=True)

    dataset = "bigquery-public-data.ga4_sample"
    ds_dir = lite / "resource" / "databases" / "bigquery" / "ga4" / dataset
    ds_dir.mkdir(parents=True)
    body = "  event_date STRING,\n  event_params ARRAY<STRUCT<key STRING>>"
    with (ds_dir / "DDL.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["table_name", "ddl"])
        for shard in ("events_20210101", "events_20210102"):
            writer.writerow([shard, _ddl(dataset, shard, body)])
        writer.writerow(["users", _ddl(dataset, "users", "  user_id STRING")])
    (ds_dir / "events_20210102.json").write_text(
        json.dumps(
            {
                "table_name": "events_20210102",
                "table_fullname": f"{dataset}.events_20210102",
                "column_names": ["event_date", "event_params"],
                "column_types": ["STRING", "ARRAY<STRUCT<key STRING>>"],
                "nested_column_names": [
                    "event_date",
                    "event_params",
                    "event_params.key",
                ],
                "nested_column_types": [
                    "STRING",
                    "ARRAY<STRUCT<key STRING>>",
                    "STRING",
                ],
                "description": ["Event day.", "Parameters.", "Parameter name."],
                "sample_rows": [{"event_date": "20210102", "event_params": []}],
            }
        )
    )
    return root


class SeedSpider2LiteBqTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.upstream = _fake_upstream(self.tmp / "Spider2")
        self.dest = self.tmp / "dest"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _seed(self, **kwargs) -> list[str]:
        return seed.seed_spider2_lite_bq(
            dest=self.dest,
            upstream_root=self.upstream,
            billing_project="billing",
            write_env=False,
            **kwargs,
        )

    def test_seeds_only_bigquery_questions(self) -> None:
        self.assertEqual(self._seed(), ["ga4"])
        rows = json.loads((self.dest / "evaluation.json").read_text())
        self.assertEqual([r["question_id"] for r in rows], ["ga900"])
        row = rows[0]
        self.assertEqual(row["SQL"], "SELECT 1")
        self.assertEqual(row["gold_results"], ["gold/ga900_a.csv", "gold/ga900_b.csv"])
        self.assertEqual(row["condition_cols"], [[0], [1]])
        self.assertEqual(row["bq_datasets"], ["bigquery-public-data.ga4_sample"])
        self.assertTrue((self.dest / "gold" / "ga900_b.csv").is_file())

    def test_collapses_shards_into_a_wildcard_table(self) -> None:
        self._seed()
        schema = (self.dest / "ga4" / "schema.sql").read_text()
        self.assertIn("CREATE TABLE `bigquery-public-data.ga4_sample.events_*`", schema)
        self.assertNotIn("events_20210101`", schema)
        self.assertIn("_TABLE_SUFFIX", schema)
        self.assertIn("events_*`.event_params.key: Parameter name.", schema)

        metadata = json.loads((self.dest / "ga4" / "metadata.json").read_text())
        self.assertEqual(set(metadata), {"events_*"})
        columns = {c["name"]: c for c in metadata["events_*"]["columns"]}
        self.assertEqual(columns["event_date"]["description"], "Event day.")
        self.assertEqual(
            columns["event_params"]["description"],
            "Parameters. Fields: key: Parameter name.",
        )
        self.assertEqual(
            columns["_TABLE_SUFFIX"]["value_examples"], ["20210101", "20210102"]
        )

    def test_omits_unused_allowlist_and_metadata_types(self) -> None:
        self._seed()
        database = self.dest / "ga4"
        self.assertFalse((database / "allowlist.json").exists())

        metadata = json.loads((database / "metadata.json").read_text())
        columns = [
            column for table in metadata.values() for column in table.get("columns", [])
        ]
        self.assertTrue(columns)
        self.assertTrue(all("data_type" not in column for column in columns))

    def test_schema_budget_lists_the_rest_by_name(self) -> None:
        tables = seed._read_dataset(
            self.upstream
            / "spider2-lite/resource/databases/bigquery/ga4/bigquery-public-data.ga4_sample"
        )
        full, listed = seed.render_schema("ga4", tables, budget=10**6)
        self.assertEqual(listed, 0)
        users_at = full.index("CREATE TABLE `bigquery-public-data.ga4_sample.users`")
        text, listed = seed.render_schema("ga4", tables, budget=users_at + 10)
        self.assertEqual(listed, 1)
        self.assertIn("1 more table(s), columns omitted for length", text)
        self.assertIn("`bigquery-public-data.ga4_sample.users`", text)

    def test_writes_manifest(self) -> None:
        self._seed()
        manifest = json.loads((self.dest / "manifest.json").read_text())
        self.assertEqual(manifest["question_count"], 1)
        self.assertEqual(manifest["databases"][0]["collapsed_shards"], 1)


if __name__ == "__main__":
    unittest.main()


def test_ddl_columns_reads_top_level_columns_and_types() -> None:
    ddl = _ddl(
        "d",
        "t_2020",
        '  id INT64 NOT NULL OPTIONS(description="key"),\n'
        "  hits ARRAY<STRUCT<n INT64, page STRING>>",
    )

    assert seed._ddl_columns(ddl) == {
        "id": "INT64",
        "hits": "ARRAY<STRUCT<n INT64, page STRING>>",
    }


def test_a_shard_with_a_column_the_newest_lacks_is_its_own_table(
    tmp_path: Path,
) -> None:
    with (tmp_path / "DDL.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["table_name", "ddl"])
        writer.writeheader()
        for table, body in [
            ("tree_census_1995", "  recordid INT64"),
            ("tree_census_2005", "  tree_id INT64"),
            ("tree_census_2015", "  tree_id INT64,\n  block_id INT64"),
        ]:
            writer.writerow({"table_name": table, "ddl": _ddl("d", table, body)})

    names = sorted(t.name for t in seed._read_dataset(tmp_path))

    assert names == ["tree_census_*", "tree_census_1995"]
