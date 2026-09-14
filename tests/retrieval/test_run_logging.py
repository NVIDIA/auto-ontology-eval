# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the run instrumentation. No external services needed."""

from __future__ import annotations

from ontology_sql_eval.retrieval.run_logging import RunLogger


class _ExplodingStream:
    """Stand-in for a jsonl handle on a full disk / closed file."""

    name = "<exploding>"

    def write(self, _text: str) -> None:
        raise OSError("simulated disk full")

    def flush(self) -> None:
        raise OSError("simulated disk full")

    def close(self) -> None:
        pass


def test_write_failures_never_reach_the_caller(tmp_path):
    """``event`` and ``question`` are both called from ``_evaluate_question``'s
    finally block, so a failed write must not propagate out through the
    worker's future and abort the whole run."""
    run = RunLogger(tmp_path, "run-write-failure")
    try:
        run._events = _ExplodingStream()
        run._questions = _ExplodingStream()

        run.event("phase_start", phase="retrieval")
        run.question({"qid": "q1", "row_index": 3})

        # The record is still aggregated for summary.json even though its
        # jsonl line could not be persisted.
        assert run._question_records == [{"qid": "q1", "row_index": 3}]
    finally:
        run.close()
