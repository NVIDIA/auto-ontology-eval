# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The vendored BEAM and cognee prompts are the text they were checked against.

BEAM's files are verbatim from ``mohammadtavakoli78/BEAM@b2da22e``
(``src/prompts.py`` and ``src/evaluation/compute_metrics.py``); cognee's from
``topoteretes/cognee`` ``eval_framework/beam``, with ``cognee_judge.txt``'s
``str.format`` fields written as ``<name>``. A score is only comparable to a
published one while these stay put, so an edit has to change a hash here too.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

PROMPTS = (
    Path(__file__).resolve().parents[2] / "conversation" / "beam_benchmark" / "prompts"
)

SHA256 = {
    "answer/beam_answer.txt": "ed89b98d1024432a46bd297345961cf89d2429746daec37befb50db46e0c914a",
    "answer/cognee/abstention.txt": "90487f81172d34953cd213e7032e1c1b2669941bbcf742f0d0c821210132a41f",
    "answer/cognee/context_for_question.txt": "5266f6608b552e729fe873bd61880d2d89401896f1107fc54834b52b83b29802",
    "answer/cognee/contradiction_resolution.txt": "127e5ab066deac7e09ba72e82d749ed4abb67318e803f6c200dd264ba24ba4d1",
    "answer/cognee/default.txt": "41eb561a9104fd756f333bf37a6e0d0f9fbd7c8806454350622846e03a5ad4ee",
    "answer/cognee/event_ordering.txt": "ea7019908f29d09b5dc0c2ffbd589a11e4d752c399e360746e9319c9dc260087",
    "answer/cognee/information_extraction.txt": "b646547f91d7610221b32a4663a1cde68a7cfc3ccebaefc0b8b7edeed6badf01",
    "answer/cognee/instruction_following.txt": "720a4962d8727c67b98c75a6c71c3222a0502488c18eb0857650916a62d479de",
    "answer/cognee/knowledge_update.txt": "8679bfd36b33877c50735c14c109bf4003d8a4018e07ced0b02a92f2267c730d",
    "answer/cognee/multi_session_reasoning.txt": "2c5ed0d451b52a0ea468617854968380a12f7b4d9a7ca7aea9ba6b1a24155a06",
    "answer/cognee/preference_following.txt": "e51df6b42ca84186b6c1e57a4fadf348adaeaee553476ebc039a188155a83b4a",
    "answer/cognee/summarization.txt": "de25b0272ebd818484d184571dc12635d15cb5fcf44db532583e7c610f34f870",
    "answer/cognee/temporal_reasoning.txt": "85d2a704582d7319324c221c606f96f5070121d10a91865fcf0bbb91f9a7ab69",
    "judge/beam_equivalence_system.txt": "0863c78c1b27d36c97bb7bfd2d4d6b88ef69b093e92bb4719e665527722413e2",
    "judge/beam_equivalence_user.txt": "e950adf8d75378eb37ce8e5c30456098701ec209c78983a89d79056cd1697570",
    "judge/beam_judge.txt": "593373c642a288a7b590577d8a8fc92c3f9a2b70e2f64ad6e59a040a6c56b7f5",
    "judge/cognee_judge.txt": "f9028c0b2ea9064b786a2c62120b90f91f5cb2b4c5715975546c019134eb0eca",
}


def test_every_vendored_prompt_is_listed() -> None:
    on_disk = {str(p.relative_to(PROMPTS)) for p in PROMPTS.rglob("*.txt")}
    assert on_disk == set(SHA256)


def test_every_vendored_prompt_matches_its_hash() -> None:
    changed = [
        name
        for name, digest in SHA256.items()
        if hashlib.sha256((PROMPTS / name).read_bytes()).hexdigest() != digest
    ]
    assert changed == []
