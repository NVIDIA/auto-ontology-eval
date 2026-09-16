from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import main
from ontology_sql_eval.ingestion import ingest, semantic


class AnnotationPipelineTest(unittest.TestCase):
    @patch.object(semantic, "apply_saved_descriptions")
    @patch.object(semantic, "run_semantic_compilation", return_value=7)
    def test_semantic_compile_automatically_applies_saved_descriptions(
        self,
        compile_semantic,
        apply_descriptions,
    ) -> None:
        self.assertEqual(semantic.run_semantic("cards", benchmark_name="bird"), 7)
        compile_semantic.assert_called_once_with("cards")
        apply_descriptions.assert_called_once_with("cards", "bird")

    @patch.object(semantic, "apply_saved_descriptions")
    @patch.object(semantic, "run_semantic_compilation", return_value=7)
    def test_semantic_defaults_benchmark_to_database_name(
        self,
        compile_semantic,
        apply_descriptions,
    ) -> None:
        self.assertEqual(semantic.run_semantic("wideworldimporters"), 7)
        compile_semantic.assert_called_once_with("wideworldimporters")
        apply_descriptions.assert_called_once_with(
            "wideworldimporters", "wideworldimporters"
        )

    @patch.object(ingest, "run_ingest")
    def test_main_passes_benchmark_name_to_each_ingest(self, run_ingest) -> None:
        connection_strings = "sqlite:///first.sqlite,sqlite:///second.sqlite"
        with patch.dict(os.environ, {"CONNECTION_STRINGS": connection_strings}):
            main.stage_ingest("bird")

        self.assertEqual(run_ingest.call_count, 2)
        run_ingest.assert_any_call("sqlite:///first.sqlite", "bird")
        run_ingest.assert_any_call("sqlite:///second.sqlite", "bird")

    @patch.object(semantic, "run_semantic")
    @patch.object(ingest, "database_name_for", return_value="california_schools")
    def test_main_separates_database_from_benchmark_for_one_connection(
        self, database_name_for, run_semantic
    ) -> None:
        connection_string = "sqlite:///california_schools.sqlite"
        with patch.dict(os.environ, {"CONNECTION_STRINGS": connection_string}):
            main.stage_semantic("bird")

        database_name_for.assert_called_once_with(connection_string)
        run_semantic.assert_called_once_with("california_schools", "bird")


if __name__ == "__main__":
    unittest.main()
