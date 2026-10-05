from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import seed_bird


class SeedBirdTrainCorpusTest(unittest.TestCase):
    def test_dev_also_installs_train_questions(self) -> None:
        dev_rows = [{"question_id": 0, "db_id": "cards"}]
        train_rows = [{"question_id": 0, "db_id": "train_db"}]

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(seed_bird, "_migrate_legacy_flat_layout"),
            patch.object(seed_bird, "_cleanup_legacy_flat_layout"),
            patch.object(seed_bird, "_download") as download,
            patch.object(seed_bird, "_extract"),
            patch.object(
                seed_bird,
                "_load_train_questions",
                return_value=train_rows,
            ) as load_train,
            patch.object(seed_bird, "_organize_extracted", return_value=["cards"]),
            patch.object(seed_bird, "_load_eval_rows", return_value=dev_rows),
            patch.object(seed_bird, "_write_evaluation_file"),
            patch.object(seed_bird, "_write_train_file") as write_train,
            patch.object(seed_bird, "_print_summary"),
        ):
            target = Path(tmp)
            self.assertEqual(
                seed_bird.download_bird(splits=["dev"], dest=target, write_env=False),
                ["cards"],
            )

        train_source = target / f".{seed_bird.TRAIN_QUESTIONS_CACHE_NAME}"
        download.assert_any_call(
            seed_bird.TRAIN_QUESTIONS_URL, train_source, force=False
        )
        load_train.assert_called_once_with(train_source)
        write_train.assert_called_once_with(target / "train", train_rows)

    def test_mini_dev_does_not_install_train_questions(self) -> None:
        mini_rows = [{"question_id": 0, "db_id": "cards"}]

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(seed_bird, "_migrate_legacy_flat_layout"),
            patch.object(seed_bird, "_cleanup_legacy_flat_layout"),
            patch.object(seed_bird, "_download") as download,
            patch.object(seed_bird, "_extract"),
            patch.object(seed_bird, "_load_train_questions") as load_train,
            patch.object(seed_bird, "_organize_extracted", return_value=["cards"]),
            patch.object(seed_bird, "_load_eval_rows", return_value=mini_rows),
            patch.object(seed_bird, "_write_evaluation_file"),
            patch.object(seed_bird, "_write_train_file") as write_train,
            patch.object(seed_bird, "_print_summary"),
        ):
            target = Path(tmp)
            seed_bird.download_bird(splits=["mini-dev"], dest=target, write_env=False)

        self.assertEqual(download.call_count, 1)
        load_train.assert_not_called()
        write_train.assert_not_called()


class SeedBirdDescriptionTest(unittest.TestCase):
    def test_value_description_is_stored_separately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "schools.csv"
            csv_path.write_text(
                "original_column_name,column_name,column_description,"
                "value_description,data_format\n"
                "Status,Status,School status,"
                "A = active; C = closed,text\n"
                "OpaqueId,OpaqueId,Opaque identifier,not useful,integer\n",
                encoding="utf-8",
            )

            columns = seed_bird._read_description_csv(csv_path)

        self.assertEqual(
            columns,
            [
                {
                    "name": "Status",
                    "description": "School status",
                    "value_description": "A = active; C = closed",
                    "value_examples": None,
                },
                {
                    "name": "OpaqueId",
                    "description": "Opaque identifier",
                    "value_description": "not useful",
                    "value_examples": None,
                },
            ],
        )


class SeedBirdLegacyMigrationTest(unittest.TestCase):
    def test_migration_preserves_unrecognized_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            (dest / "evaluation.json").write_text(
                json.dumps([{"db_id": "bird_db"}]), encoding="utf-8"
            )

            bird_db = dest / "bird_db"
            bird_db.mkdir()
            (bird_db / "bird_db.sqlite").touch()

            unrelated = dest / "unrelated"
            unrelated.mkdir()
            (unrelated / "keep.txt").write_text("keep", encoding="utf-8")

            other_db = dest / "other_db"
            other_db.mkdir()
            (other_db / "other_db.sqlite").touch()

            seed_bird._migrate_legacy_flat_layout(dest)

            self.assertFalse(bird_db.exists())
            self.assertTrue((dest / "dev" / "bird_db" / "bird_db.sqlite").is_file())
            self.assertEqual(
                (unrelated / "keep.txt").read_text(encoding="utf-8"), "keep"
            )
            self.assertTrue((other_db / "other_db.sqlite").is_file())


if __name__ == "__main__":
    unittest.main()
