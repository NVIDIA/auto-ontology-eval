from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from ontology_sql_eval.ingestion import enrich_graph


class EnrichmentArtifactResolutionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.datasets = root / "datasets"
        self.annotations = root / "annotations"
        self.datasets.mkdir()
        self.annotations.mkdir()
        self.original_datasets = enrich_graph.DEFAULT_DIR
        self.original_annotations = enrich_graph.ANNOTATIONS_DIR
        enrich_graph.DEFAULT_DIR = self.datasets
        enrich_graph.ANNOTATIONS_DIR = self.annotations

    def tearDown(self) -> None:
        enrich_graph.DEFAULT_DIR = self.original_datasets
        enrich_graph.ANNOTATIONS_DIR = self.original_annotations
        self.temp_dir.cleanup()

    @staticmethod
    def _write(path: Path, content: str = "{}") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_metadata_resolves_supported_dataset_layouts(self) -> None:
        cases = (
            ("standalone", None, self.datasets / "standalone" / "metadata.json"),
            (
                "database_a",
                "benchmark_a",
                self.datasets / "benchmark_a" / "database_a" / "metadata.json",
            ),
            (
                "database_b",
                "benchmark_b",
                self.datasets / "benchmark_b" / "dev" / "database_b" / "metadata.json",
            ),
        )
        for database, dataset, expected in cases:
            with self.subTest(database=database, dataset=dataset):
                self._write(expected)
                self.assertEqual(
                    enrich_graph.metadata_json_path(database, dataset), expected
                )

    def test_annotation_custom_analyses_override_dataset_copy(self) -> None:
        dataset_copy = self._write(
            self.datasets / "bird" / "dev" / "cards" / "custom_analyses.json",
            json.dumps([{"name": "dataset"}]),
        )
        annotation_copy = self._write(
            self.annotations / "bird" / "custom_analyses" / "cards.json",
            json.dumps([{"name": "annotation"}]),
        )

        self.assertNotEqual(dataset_copy, annotation_copy)
        self.assertEqual(
            enrich_graph.custom_analyses_json_path("cards", "bird"),
            annotation_copy,
        )

    def test_lookup_without_dataset_keeps_legacy_fallback(self) -> None:
        expected = self._write(
            self.datasets / "fdabench" / "database_c" / "metadata.json"
        )
        self.assertEqual(enrich_graph.metadata_json_path("database_c"), expected)

    def test_missing_artifacts_return_none(self) -> None:
        self.assertIsNone(enrich_graph.metadata_json_path("missing", "benchmark"))
        self.assertIsNone(
            enrich_graph.custom_analyses_json_path("missing", "benchmark")
        )

    def test_saved_descriptions_filter_by_database(self) -> None:
        path = self.annotations / "bird" / "semantic_descriptions.csv"
        path.parent.mkdir(parents=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "database",
                    "table",
                    "column",
                    "column_description",
                    "column_attribute",
                    "column_attribute_description",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "database": "database_a",
                    "table": "orders",
                    "column": "id",
                    "column_description": "Order identifier",
                    "column_attribute": "Order ID",
                    "column_attribute_description": "Identifier for an order",
                }
            )
            writer.writerow(
                {
                    "database": "database_b",
                    "table": "users",
                    "column": "id",
                    "column_description": "User identifier",
                }
            )

        self.assertEqual(
            enrich_graph.saved_descriptions_csv_path("database_a", "bird"), path
        )
        self.assertEqual(
            enrich_graph._read_saved_descriptions(path, "database_a"),
            {
                ("orders", "id"): {
                    "column_description": "Order identifier",
                    "attribute_name": "Order ID",
                    "attribute_description": "Identifier for an order",
                }
            },
        )

    def test_saved_descriptions_validate_required_columns(self) -> None:
        path = self._write(
            self.annotations / "bird" / "semantic_descriptions.csv",
            "database,table,column\n",
        )
        with self.assertRaisesRegex(ValueError, "missing column"):
            enrich_graph._read_saved_descriptions(path, "database_a")


if __name__ == "__main__":
    unittest.main()
