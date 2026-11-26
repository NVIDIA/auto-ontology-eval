# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How cognee ran BEAM, as settings rather than as prose.

Four values, and every one of them is theirs: the judge model, the reader, how
many turns it sees, how many rounds are averaged. They sit here, beside the
judge and the prompts, because this directory is the instrument their published
**0.794** was measured with and these are the knobs it was set to. 0.79 is the
bar, and a score produced with any of them changed is not a score that may be
put next to it.

That is also why none of them is a command-line flag. The things that do vary
between runs -- which tier, which transcripts, which arm, where output lands --
are flags in ``main.py``; the things that define the measurement are here.

The modules that read these values are spread across the benchmark -- the judge
model is used in ``cognee/judge.py``, the reader and ``top_k`` in
``beam/reader.py``, ``rounds`` in ``beam/run.py``. They are collected here
anyway, because what makes them one thing is not who reads them but who chose
them.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["BEAM", "PublishedSetup"]


@dataclass(frozen=True)
class PublishedSetup:
    """One benchmark's published measurement setup.

    Frozen because a run that rewrote one of these mid-flight would report a
    number for a configuration it did not use.
    """

    name: str

    #: Which corpus the transcripts come from, as
    #: :mod:`auto_ontology_eval.ingestion.conversation_ingest` keys them. Named
    #: rather than assumed equal to :attr:`name`: they happen to match for BEAM,
    #: and a benchmark run against a corpus it was not measured on would be
    #: scored without complaint.
    corpus: str

    # Scored with. These govern every arm, because they are how an answer is
    # turned into a number rather than how it was produced.

    #: The judge model. Changing it means the score is not the same instrument:
    #: re-judging one byte-identical set of answers with ``gpt-4.1-mini`` moved
    #: it from 0.842 to 0.807, so the judge is worth about 0.035 on its own.
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


#: Cognee's instrument, as published for conversation 2 at the 100K tier:
#: their judge prompt and judge model, their reader prompts on the vector arm,
#: and their four rounds.
BEAM = PublishedSetup(
    name="beam",
    corpus="beam",
    judge_model="azure/openai/gpt-5",
    reader_model="aws/anthropic/bedrock-claude-opus-4-8",
    # Four, as cognee ran this tier. One round against a stochastic judge is
    # not the measurement 0.794 was, so it is not an option here.
    rounds=4,
)
