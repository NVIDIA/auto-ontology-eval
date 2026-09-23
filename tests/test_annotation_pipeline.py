from __future__ import annotations

import importlib
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import main


class AnnotationPipelineTest(unittest.TestCase):
    @staticmethod
    def _semantic_module():
        compile_semantic = MagicMock(return_value=7)
        gsf_semantic = types.ModuleType("auto_ontology.semantic")
        gsf_compile = types.ModuleType("auto_ontology.semantic.compile")
        gsf_compile.run_semantic_compilation = compile_semantic
        module_name = "ontology_sql_eval.ingestion.semantic"
        sys.modules.pop(module_name, None)
        with patch.dict(
            sys.modules,
            {
                "auto_ontology.semantic": gsf_semantic,
                "auto_ontology.semantic.compile": gsf_compile,
            },
        ):
            semantic = importlib.import_module(module_name)
        return semantic, compile_semantic

    def test_semantic_compile_automatically_applies_saved_descriptions(self) -> None:
        semantic, compile_semantic = self._semantic_module()
        with patch.object(semantic, "apply_saved_descriptions") as apply_descriptions:
            self.assertEqual(semantic.run_semantic("cards", benchmark_name="bird"), 7)
        compile_semantic.assert_called_once_with("cards")
        apply_descriptions.assert_called_once_with("cards", "bird")

    def test_semantic_preserves_global_annotation_fallback(self) -> None:
        semantic, compile_semantic = self._semantic_module()
        with patch.object(semantic, "apply_saved_descriptions") as apply_descriptions:
            self.assertEqual(semantic.run_semantic("wideworldimporters"), 7)
        compile_semantic.assert_called_once_with("wideworldimporters")
        apply_descriptions.assert_called_once_with("wideworldimporters", None)

    def test_main_passes_benchmark_name_to_each_ingest(self) -> None:
        ingest = types.ModuleType("ontology_sql_eval.ingestion.ingest")
        ingest.run_ingest = MagicMock()
        connection_strings = "sqlite:///first.sqlite,sqlite:///second.sqlite"
        with (
            patch.dict(os.environ, {"CONNECTION_STRINGS": connection_strings}),
            patch.dict(sys.modules, {"ontology_sql_eval.ingestion.ingest": ingest}),
        ):
            main.stage_ingest("bird")

        self.assertEqual(ingest.run_ingest.call_count, 2)
        ingest.run_ingest.assert_any_call("sqlite:///first.sqlite", "bird")
        ingest.run_ingest.assert_any_call("sqlite:///second.sqlite", "bird")

    def test_main_separates_database_from_benchmark_for_one_connection(self) -> None:
        ingest = types.ModuleType("ontology_sql_eval.ingestion.ingest")
        ingest.database_name_for = MagicMock(return_value="california_schools")
        semantic = types.ModuleType("ontology_sql_eval.ingestion.semantic")
        semantic.run_semantic = MagicMock()
        connection_string = "sqlite:///california_schools.sqlite"
        with (
            patch.dict(os.environ, {"CONNECTION_STRINGS": connection_string}),
            patch.dict(
                sys.modules,
                {
                    "ontology_sql_eval.ingestion.ingest": ingest,
                    "ontology_sql_eval.ingestion.semantic": semantic,
                },
            ),
        ):
            main.stage_semantic("bird")

        ingest.database_name_for.assert_called_once_with(connection_string)
        semantic.run_semantic.assert_called_once_with("california_schools", "bird")


if __name__ == "__main__":
    unittest.main()
