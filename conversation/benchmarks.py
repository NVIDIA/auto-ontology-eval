# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How each conversation benchmark is scored, as data keyed by its name.

The SQL side already works this way. A dataset's evaluator is not something the
command line can choose: ``exec_match.matcher_for`` hands BEAVER's result
comparison to ``beaverbench`` and BIRD's to everything else, and
``timeout_for`` does the same for the query timeout. Nothing anyone types can
score a dataset with another dataset's rule.

The conversation side needs it more, not less, because BEAM has been published
under two different instruments and their numbers are not on one scale. The
paper's own judge, which never shows the judge the question, scored the paper's
best RAG setup **0.323**. Cognee's edited judge -- it supplies the question,
snaps every verdict to 0/0.5/1, and scores a judge failure 0.0 instead of
raising -- scored their system **0.794**. The bar to beat is **0.79**, so the
instrument that decides whether we have won is cognee's, and under it we score
**0.856**.

Both are kept, because a score means nothing apart from the published figure it
is set beside. They are two entries here rather than two command-line flags,
and that is the point of the file: while the instrument was spelled out in
arguments, the defaults quietly combined the paper's judge, the paper's reader
prompt and a turn budget of 10, which reproduces neither 0.323 nor 0.856 but a
third number nobody has published. A run that forgot one flag was not a worse
run, it was an unanswerable one.

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

    #: Which rubric prompt and scoring rule the judge uses.
    judge_flavour: str
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

    #: Whose reader prompt answers from the retrieved turns.
    reader_style: str = "beam"
    #: The reader, or ``None`` for the configured default. Worth pinning: across
    #: the paper's four readers the same RAG setup scored 0.269 to 0.323, so the
    #: reader is part of the measurement rather than an implementation detail.
    reader_model: str | None = None
    #: Turns retrieved per question. Not the library's 10: the published runs
    #: this is compared against read five.
    top_k: int = 5


#: Keyed by the ``--database-name`` that selects it. Both entries read the same
#: transcripts from ``conversation/data/beam/<scale>/``; they differ only in how
#: an answer is produced and scored, which is what makes them two benchmarks
#: over one dataset rather than one benchmark with options.
BENCHMARKS: dict[str, BenchmarkSpec] = {
    # Cognee's instrument, and the setup behind our 0.856. The default, because
    # 0.79 is the bar and this is the only instrument that can be held against
    # it.
    "beam": BenchmarkSpec(
        name="beam",
        judge_flavour="cognee",
        judge_model="azure/openai/gpt-4.1-mini",
        reader_style="cognee",
        reader_model="aws/anthropic/bedrock-claude-opus-4-8",
        # Four, as cognee ran this tier. One round against a stochastic judge
        # is not the measurement 0.794 was, so it is not an option here.
        rounds=4,
    ),
    # The paper's own instrument, verbatim, and the only one that can be held
    # against its 0.323. Kept reachable so a regression can be checked against
    # the original as well as against cognee.
    "beam-paper": BenchmarkSpec(
        name="beam-paper",
        judge_flavour="beam",
        judge_model="azure/openai/gpt-4.1-mini",
        reader_style="beam",
        reader_model=None,
        # The paper reports a single pass.
        rounds=1,
    ),
}


def spec_for(database_name: str) -> BenchmarkSpec | None:
    """The benchmark *database_name* selects, or ``None`` if it is a SQL dataset."""
    return BENCHMARKS.get(database_name)


def is_conversation_benchmark(database_name: str) -> bool:
    """Whether ``--database-name`` routes to the conversation stages."""
    return database_name in BENCHMARKS
