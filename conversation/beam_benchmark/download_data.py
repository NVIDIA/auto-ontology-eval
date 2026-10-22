#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download the BEAM benchmark into ``conversation/data/beam/``.

Source: https://huggingface.co/datasets/Mohammadta/BEAM (CC BY-SA 4.0), with
the 10M conversations in a second repo because of their size.

Produces one directory per scale::

    data/beam/100K/
      conversations/<conversation_id>.json   the dialogue, sessions of turns
      questions.csv                          one row per probing question

A conversation is the unit of evaluation, so it is the unit on disk: a run
reads one file, builds a memory from it, and answers only that file's
questions. Keeping them in one table would invite exactly the pooling the
benchmark is designed to catch.

``questions.csv`` carries the rubric as a JSON list in a single column rather
than exploding it into rows. Nuggets are scored individually but a question's
score is their mean, so the question is what has to stay addressable.

Upstream ships the rubrics inside a ``probing_questions`` string written by
``repr`` rather than ``json.dumps`` — single quotes, Python ``None`` — so it
is parsed with ``ast.literal_eval`` and re-emitted as real JSON here. Nothing
downstream should have to know that.

Usage
-----
    uv run python -m conversation.beam_benchmark.download_data
    uv run python -m conversation.beam_benchmark.download_data --scale 1M
    uv run python -m conversation.beam_benchmark.download_data --scale all
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

#: Everything this writes, outside the harness that writes it.
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "beam"

#: Where each scale lives, and how much comes down with it. The 10M chats are
#: in a repo of their own and split across two files; the rest share one.
#:
#: The paper calls the smallest scale 128K and the dataset files call it 100K.
#: The file name wins here, since that is what has to be fetched.
SCALES: dict[str, tuple[str, list[str]]] = {
    "100K": ("Mohammadta/BEAM", ["data/100K-00000-of-00001.parquet"]),
    "500K": ("Mohammadta/BEAM", ["data/500K-00000-of-00001.parquet"]),
    "1M": ("Mohammadta/BEAM", ["data/1M-00000-of-00001.parquet"]),
    "10M": (
        "Mohammadta/BEAM-10M",
        [
            "data/10M-00000-of-00002.parquet",
            "data/10M-00001-of-00002.parquet",
        ],
    ),
}

#: Roughly what each scale costs to fetch, so ``--scale all`` is an informed
#: choice rather than a surprise.
DOWNLOAD_MB = {"100K": 5, "500K": 34, "1M": 66, "10M": 344}

#: The ten memory abilities, which are the keys of ``probing_questions``.
#: Listed rather than discovered so that a scale missing one is noticed.
ABILITIES = (
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
)

QUESTION_FIELDS = [
    "conversation_id",
    "ability",
    "question_index",
    "question",
    "ideal_response",
    "difficulty",
    #: The atomic claims a response is scored against, as a JSON list.
    "rubric",
]


def _read_parquet(repo: str, names: list[str], columns: list[str]) -> list[dict]:
    """Read *names* from the Hub as one list of rows, *columns* only.

    Projected rather than downloaded whole: the conversations and the
    questions live in the same file, and a caller that wants the questions
    should not pay for ten million tokens of dialogue to get them.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    filesystem = HfFileSystem()
    rows: list[dict] = []
    for name in names:
        remote = f"datasets/{repo}/{name}"
        logger.info("  reading %s", name)
        with filesystem.open(remote, "rb") as handle:
            rows.extend(pq.read_table(handle, columns=columns).to_pylist())
    return rows


def _questions(conversation_id: str, raw: str) -> list[dict]:
    """The probing questions for one conversation, as rows.

    *raw* is a Python literal, not JSON — see the module docstring.
    """
    try:
        parsed = ast.literal_eval(raw) if raw else {}
    except (ValueError, SyntaxError):
        logger.warning("conversation %s: unreadable probing_questions", conversation_id)
        return []

    missing = [a for a in ABILITIES if a not in parsed]
    if missing:
        logger.warning("conversation %s: no %s", conversation_id, ", ".join(missing))

    rows = []
    for ability in ABILITIES:
        for index, question in enumerate(parsed.get(ability) or []):
            rows.append(
                {
                    "conversation_id": conversation_id,
                    "ability": ability,
                    "question_index": index,
                    "question": question.get("question", ""),
                    "ideal_response": question.get("ideal_response", ""),
                    "difficulty": question.get("difficulty", ""),
                    "rubric": json.dumps(question.get("rubric") or []),
                }
            )
    return rows


def _write_scale(scale: str, force: bool) -> tuple[int, int]:
    """Fetch one scale and write it out, as ``(conversations, questions)``."""
    repo, names = SCALES[scale]
    target = DATA_DIR / scale
    conversations = target / "conversations"
    questions_csv = target / "questions.csv"

    if questions_csv.exists() and not force:
        logger.info("%s: already downloaded, skipping (--force to replace)", scale)
        existing = len(list(conversations.glob("*.json")))
        with questions_csv.open() as handle:
            return existing, sum(1 for _ in handle) - 1

    logger.info("%s: downloading ~%d MB", scale, DOWNLOAD_MB.get(scale, 0))
    rows = _read_parquet(repo, names, ["conversation_id", "chat", "probing_questions"])

    conversations.mkdir(parents=True, exist_ok=True)
    question_rows: list[dict] = []
    for row in rows:
        conversation_id = str(row["conversation_id"])
        # The dialogue keeps its nesting: BEAM's turns are grouped into
        # sessions, and several of the abilities — event ordering, temporal
        # reasoning — are asked about where a turn sits in that sequence.
        (conversations / f"{conversation_id}.json").write_text(
            json.dumps({"conversation_id": conversation_id, "chat": row["chat"]})
        )
        question_rows.extend(_questions(conversation_id, row["probing_questions"]))

    with questions_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUESTION_FIELDS)
        writer.writeheader()
        writer.writerows(question_rows)

    logger.info(
        "%s: %d conversation(s), %d question(s) → %s",
        scale,
        len(rows),
        len(question_rows),
        target,
    )
    return len(rows), len(question_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scale",
        default="100K",
        choices=[*SCALES, "all"],
        help="Which conversation length to fetch. Default: 100K, the cheapest "
        "and the only scale with enough conversations to measure anything "
        "narrow on.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download a scale already on disk."
    )
    args = parser.parse_args()

    scales = list(SCALES) if args.scale == "all" else [args.scale]
    logger.info("Writing to %s", DATA_DIR)

    totals = [0, 0]
    for scale in scales:
        got, asked = _write_scale(scale, args.force)
        totals = [totals[0] + got, totals[1] + asked]

    logger.info("Ready: %d conversation(s), %d question(s).", *totals)


if __name__ == "__main__":
    main()
