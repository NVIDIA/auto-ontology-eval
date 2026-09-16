from __future__ import annotations

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
            patch.object(seed_bird, "_extract_train_questions") as extract_train,
            patch.object(seed_bird, "_organize_extracted", return_value=["cards"]),
            patch.object(
                seed_bird, "_load_eval_rows", side_effect=[dev_rows, train_rows]
            ),
            patch.object(seed_bird, "_write_evaluation_file"),
            patch.object(seed_bird, "_write_train_file") as write_train,
            patch.object(seed_bird, "_print_summary"),
        ):
            target = Path(tmp)
            self.assertEqual(
                seed_bird.download_bird(splits=["dev"], dest=target, write_env=False),
                ["cards"],
            )

        train_archive = target / f".{seed_bird._TRAIN_CONFIG.archive_name}"
        download.assert_any_call(
            seed_bird._TRAIN_CONFIG.url, train_archive, force=False
        )
        extract_train.assert_called_once()
        write_train.assert_called_once_with(target / "train", train_rows)

    def test_mini_dev_does_not_install_train_questions(self) -> None:
        mini_rows = [{"question_id": 0, "db_id": "cards"}]

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(seed_bird, "_migrate_legacy_flat_layout"),
            patch.object(seed_bird, "_cleanup_legacy_flat_layout"),
            patch.object(seed_bird, "_download") as download,
            patch.object(seed_bird, "_extract"),
            patch.object(seed_bird, "_extract_train_questions") as extract_train,
            patch.object(seed_bird, "_organize_extracted", return_value=["cards"]),
            patch.object(seed_bird, "_load_eval_rows", return_value=mini_rows),
            patch.object(seed_bird, "_write_evaluation_file"),
            patch.object(seed_bird, "_write_train_file") as write_train,
            patch.object(seed_bird, "_print_summary"),
        ):
            target = Path(tmp)
            seed_bird.download_bird(splits=["mini-dev"], dest=target, write_env=False)

        self.assertEqual(download.call_count, 1)
        extract_train.assert_not_called()
        write_train.assert_not_called()


if __name__ == "__main__":
    unittest.main()
