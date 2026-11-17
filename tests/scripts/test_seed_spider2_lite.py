from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
import zipfile
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from scripts.seed import seed_spider2_lite


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _fake_upstream(root: Path) -> Path:
    """A minimal Spider2 checkout: two local questions, one cloud question."""
    lite = root / "spider2-lite"
    _write_jsonl(
        lite / "spider2-lite.jsonl",
        [
            {
                "instance_id": "local001",
                "db": "Db-IMDB",
                "question": "Q1",
                "external_knowledge": "rule.md",
            },
            {
                "instance_id": "local002",
                "db": "f1",
                "question": "Q2",
                "external_knowledge": None,
            },
            {
                "instance_id": "bq001",
                "db": "ga360",
                "question": "cloud",
                "external_knowledge": None,
            },
        ],
    )
    gold = lite / "evaluation_suite" / "gold"
    _write_jsonl(
        gold / "spider2lite_eval.jsonl",
        [
            {"instance_id": "local001", "condition_cols": [0], "ignore_order": True},
            {
                "instance_id": "local002",
                "condition_cols": [[1], [0]],
                "ignore_order": False,
            },
        ],
    )
    (gold / "sql").mkdir(parents=True)
    (gold / "sql" / "local001.sql").write_text("SELECT 1;\n")
    (gold / "exec_result").mkdir()
    (gold / "exec_result" / "local001.csv").write_text("a\n1\n")
    (gold / "exec_result" / "local002_a.csv").write_text("a,b\n1,2\n")
    (gold / "exec_result" / "local002_b.csv").write_text("a\n1\n")
    # A prefix collision: must not be picked up as a variant of local002.
    (gold / "exec_result" / "local0020.csv").write_text("x\n")

    docs = lite / "resource" / "documents"
    docs.mkdir(parents=True)
    (docs / "rule.md").write_text("  Use the rule.  \n")

    tables = lite / "resource" / "databases" / "sqlite"
    (tables / "DB_IMDB").mkdir(parents=True)
    (tables / "DB_IMDB" / "movie.json").write_text(
        json.dumps(
            {
                "table_name": "movie",
                "table_fullname": "movie",
                "column_names": ["id", "title", "meta"],
                "column_types": ["INTEGER", "TEXT", "TEXT"],
                "description": ["", "Movie title", ""],
                "sample_rows": [
                    {"id": 1, "title": "A", "meta": {"k": 1}},
                    {"id": 1, "title": None, "meta": {"k": 1}},
                    {"id": 2, "title": "B", "meta": None},
                ],
            }
        )
    )
    (tables / "f1").mkdir()
    (tables / "f1" / "races.json").write_text(
        json.dumps(
            {
                "table_name": "races",
                "column_names": ["race_id"],
                "sample_rows": [{"race_id": 7}],
            }
        )
    )
    return root


def _sqlite_bytes(tmp: Path, name: str) -> bytes:
    path = tmp / f"{name}.build.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE t (x INTEGER)")
    data = path.read_bytes()
    path.unlink()
    return data


class SeedSpider2LiteTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.upstream = _fake_upstream(self.tmp / "Spider2")
        self.dest = self.tmp / "dest"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _seed(self, **kwargs) -> list[str]:
        return seed_spider2_lite.seed_spider2_lite(
            dest=self.dest, upstream_root=self.upstream, write_env=False, **kwargs
        )

    def test_builds_local_questions_with_gold_results(self) -> None:
        self._seed(skip_databases=True)

        rows = json.loads((self.dest / "evaluation.json").read_text())
        self.assertEqual([r["question_id"] for r in rows], ["local001", "local002"])

        first, second = rows
        self.assertEqual(first["db_id"], "db_imdb")
        self.assertEqual(first["spider2_db"], "Db-IMDB")
        self.assertEqual(first["SQL"], "SELECT 1;")
        self.assertEqual(first["evidence"], "Use the rule.")
        self.assertEqual(first["gold_results"], ["gold/local001.csv"])
        self.assertEqual(first["condition_cols"], [0])
        self.assertTrue(first["ignore_order"])

        self.assertEqual(second["SQL"], "")
        self.assertEqual(
            second["gold_results"], ["gold/local002_a.csv", "gold/local002_b.csv"]
        )
        self.assertEqual(second["condition_cols"], [[1], [0]])
        self.assertFalse(second["ignore_order"])

        copied = sorted(p.name for p in (self.dest / "gold").iterdir())
        self.assertEqual(copied, ["local001.csv", "local002_a.csv", "local002_b.csv"])

    def test_metadata_matches_apply_metadata_shape(self) -> None:
        self._seed(skip_databases=True)

        metadata = json.loads((self.dest / "db_imdb" / "metadata.json").read_text())
        columns = {c["name"]: c for c in metadata["movie"]["columns"]}
        self.assertEqual(columns["id"]["value_examples"], [1, 2])
        self.assertIsNone(columns["id"]["description"])
        self.assertEqual(columns["title"]["description"], "Movie title")
        self.assertEqual(columns["title"]["value_examples"], ["A", "B"])
        self.assertEqual(columns["meta"]["value_examples"], ['{"k": 1}'])

    def test_installs_sqlite_from_archive_by_normalized_name(self) -> None:
        archive = self.tmp / "pack.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("sqlite/DB_IMDB.sqlite", _sqlite_bytes(self.tmp, "imdb"))
            zf.writestr("__MACOSX/sqlite/._f1.sqlite", b"junk")
            zf.writestr("sqlite/unrelated.sqlite", _sqlite_bytes(self.tmp, "other"))

        def fake_download(url: str, dest_path: Path, *, force: bool = False) -> None:
            dest_path.write_bytes(archive.read_bytes())

        with patch.object(seed_spider2_lite, "_download", side_effect=fake_download):
            installed = self._seed()

        self.assertEqual(installed, ["db_imdb"])
        self.assertTrue((self.dest / "db_imdb" / "db_imdb.sqlite").is_file())
        self.assertFalse((self.dest / "f1" / "f1.sqlite").exists())
        self.assertFalse((self.dest / "unrelated").exists())
        self.assertFalse(list(self.dest.glob(".*.zip")))

        manifest = json.loads((self.dest / "manifest.json").read_text())
        status = {d["db_id"]: d["sqlite_installed"] for d in manifest["databases"]}
        self.assertEqual(status, {"db_imdb": True, "f1": False})

    def test_installed_databases_are_not_downloaded_again(self) -> None:
        root = self.tmp / "localdb"
        root.mkdir()
        (root / "Db-IMDB.sqlite").write_bytes(_sqlite_bytes(self.tmp, "a"))
        (root / "f1.sqlite").write_bytes(_sqlite_bytes(self.tmp, "b"))
        self._seed(sqlite_root=root)

        with patch.object(seed_spider2_lite, "_download") as download:
            installed = self._seed()

        download.assert_not_called()
        self.assertEqual(installed, ["db_imdb", "f1"])

    def test_drops_views_that_cannot_compile(self) -> None:
        root = self.tmp / "localdb"
        root.mkdir()
        db_path = root / "f1.sqlite"
        with closing(sqlite3.connect(db_path)) as conn, conn:
            conn.execute("CREATE TABLE periods (start_ TEXT)")
            conn.execute("CREATE VIEW good AS SELECT start_ FROM periods")
            conn.execute("CREATE VIEW bad AS SELECT start_date FROM periods")
            conn.execute("CREATE VIEW on_bad AS SELECT * FROM bad")
        (root / "Db-IMDB.sqlite").write_bytes(_sqlite_bytes(self.tmp, "a"))

        self._seed(sqlite_root=root)

        installed = self.dest / "f1" / "f1.sqlite"
        with closing(sqlite3.connect(installed)) as conn:
            views = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'view'"
                )
            }
        self.assertEqual(views, {"good"})

        manifest = json.loads((self.dest / "manifest.json").read_text())
        dropped = {d["db_id"]: d["dropped_views"] for d in manifest["databases"]}
        self.assertEqual(dropped, {"db_imdb": [], "f1": ["bad", "on_bad"]})


if __name__ == "__main__":
    unittest.main()
