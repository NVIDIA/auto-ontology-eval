# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The request the ``auto_ontology`` arm sends for one question.

Kept out of the resources server so a test can build it without starting Gym,
and hand it to the installed library's ``_build_state``. CI installs the library
from ``uv.lock``, so that test runs against the pinned commit rather than
whatever checkout a developer has installed in editable mode.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_request(
    *,
    question: str,
    evidence: str,
    connector: Any,
    sql_examples: Optional[List[Dict[str, str]]] = None,
    value_anchors: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Return the ``AgentRequest`` fields for one BIRD-style question."""
    request: Dict[str, Any] = {
        "question": question,
        "evidence": evidence,
        # Pins the intent to CALCULATION. Every BIRD question is answered by
        # running SQL, so letting the intent classifier route elsewhere only
        # creates ways to lose.
        "calculation_only": True,
        "connectors": [connector],
        "path_state": {},
        "custom_prompts": "",
        "acronyms": [],
        # Appends "Return exactly the requested output fields and NO others"
        # to Auto Ontology's projection rules. Not cosmetic: extra columns break the
        # set(gold) == set(pred) comparison outright, so omitting this scores
        # Auto Ontology with a handicap the legacy harness no longer carries.
        "shorten_answer": True,
        # Enables the validate_sql_values node, which checks filter literals
        # against the database before the SQL is accepted. Costly -- it is the
        # single largest contributor to wall clock -- but the legacy harness
        # runs with it, so omitting it makes the two arms incomparable.
        "validate_sql_values": True,
    }
    # Only set when this question has an entry: an empty list would tell the
    # agent "no exemplars exist", which is a different instruction from
    # leaving the key off, and the min_score gate deliberately withholds
    # exemplars from questions whose best match is too weak.
    if sql_examples:
        request["sql_examples"] = sql_examples
    if value_anchors:
        request["value_anchors"] = value_anchors
    return request
