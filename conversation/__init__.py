# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conversational memory: benchmarks over what a system remembers of a dialogue.

Where the unstructured work asks what can be retrieved from a corpus of
documents, this asks what can be recalled from a conversation — the same
retrieval question with the evidence arriving as turns rather than pages, and
with the additional demands that come from that: facts that are revised later,
statements that contradict earlier ones, and questions whose honest answer is
that the conversation never said.

``beam_benchmark/`` holds the harness for BEAM, and ``data/`` the corpora it
downloads. The two are kept apart because the data is large, regenerable and
untracked, while the harness is small and is the thing under review.
"""
