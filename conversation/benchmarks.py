# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How each conversation benchmark is scored, as data keyed by its name.

The SQL side already works this way. A dataset's evaluator is not something the
command line can choose: ``exec_match.matcher_for`` hands BEAVER's result
comparison to ``beaverbench`` and BIRD's to everything else, and
``timeout_for`` does the same for the query timeout. Nothing anyone types can
score a dataset with another dataset's rule.

The conversation side needs it as much. BEAM is scored here the way cognee
scored it, because their published **0.794** on conversation 2 at 100K is the
figure we compare against: their judge prompt, their judge model (``gpt-5``),
their reader prompts on the vector arm, and their four rounds. A run that
changed any one of those would not be wrong, it would be unanswerable, so none
of them is a command-line flag.

Adding a conversation benchmark means adding an entry, not a flag.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["BENCHMARKS", "BenchmarkSpec", "is_conversation_benchmark", "spec_for"]


@dataclass(frozen=True)
class BenchmarkSpec:
    """One benchmark's scoring instrument and the retrieval it was measured with.

    Every field is a property of the published setup we are comparing against,
    which is why none of them is a flag. The fields we do vary between runs --
    which arm retrieves, how many rounds, where output lands -- stay on the
    command line, in the same way the SQL side keeps ``--arm`` and ``--limit``
    while keeping the evaluator in :mod:`ontology_sql_eval.gym.exec_match`.
    """

    name: str

    # Scored with. These govern every arm, because they are how an answer is
    # turned into a number rather than how it was produced.

    #: The judge model. Changing it means the score is not the same instrument.
    judge_model: str
    #: How many times the whole answer-and-judge pass is repeated, the mean of
    #: which is the score. Part of the instrument rather than a convenience:
    #: the rubric judge is stochastic, run-to-run spread on this tier is about
    #: 0.035, and a single round therefore cannot separate two arms. What the
    #: repetition reports is run-to-run noise on a fixed question set, not a
    #: confidence interval over questions.
    rounds: int = 1

    # Answered with, on ``--retrieval vector`` only. The agent arm reads
    # nothing below this line: it retrieves through its own tool and answers
    # as its own reader, and its budget comes from the environment the whole
    # library shares -- see ``react/config.py``. Nothing here could pin it
    # without making this benchmark score a configuration no other caller
    # runs, so the run records that budget instead (``agent_limits``).

    #: The reader, or ``None`` for the configured default. Pinned because the
    #: reader is part of the measurement rather than an implementation detail.
    reader_model: str | None = None
    #: Turns retrieved per question. Not the library's 10.
    top_k: int = 5


#: Keyed by the ``--database-name`` that selects it. The transcripts are read
#: from ``conversation/data/beam/<scale>/``.
BENCHMARKS: dict[str, BenchmarkSpec] = {
    # Cognee's instrument: their judge prompt and judge model.
    "beam": BenchmarkSpec(
        name="beam",
        judge_model="azure/openai/gpt-5",
        reader_model="aws/anthropic/bedrock-claude-opus-4-8",
        # Four, as cognee ran this tier. One round against a stochastic judge
        # is not the measurement 0.794 was, so it is not an option here.
        rounds=4,
    ),
}


def spec_for(database_name: str) -> BenchmarkSpec | None:
    """The benchmark *database_name* selects, or ``None`` if it is a SQL dataset."""
    return BENCHMARKS.get(database_name)


def is_conversation_benchmark(database_name: str) -> bool:
    """Whether ``--database-name`` routes to the conversation stages."""
    return database_name in BENCHMARKS
