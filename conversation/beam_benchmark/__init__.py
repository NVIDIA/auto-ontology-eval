# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BEAM: long-term memory over conversations of 128K to 10M tokens.

    paper   https://arxiv.org/abs/2510.27246   (ICLR 2026)
    code    https://github.com/mohammadtavakoli78/BEAM
    data    https://huggingface.co/datasets/Mohammadta/BEAM

100 conversations and 2,000 questions, twenty per conversation: two for each
of ten memory abilities, from plain recall through to detecting that two
widely separated turns contradict each other, and abstaining when the
conversation never said.

Each conversation is its own memory and is scored alone. Pooling them would
not merely be unfaithful, it would break the task — two conversations describe
different users and may state opposing facts, so a system that could see both
would be marked wrong for answering either.

Answers are scored by rubric rather than by string overlap. Every question
carries *nuggets*, atomic claims a response must make, and a judge marks each
0, 0.5 or 1. A question's score is the mean over its nuggets, so partial
credit is the normal case and a perfect score is hard to reach.

``download_data.py`` fetches and normalises the corpus. Everything downstream
reads what it writes.
"""
