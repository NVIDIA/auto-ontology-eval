# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""auto-ontology-eval: LLM-powered re-scoring of Text-to-SQL evaluation CSVs."""

from ontology_sql_eval.judge.models import LLM_SCORE_FIELDS, SqlScore
from ontology_sql_eval.judge.runner import run

__all__ = ["run", "SqlScore", "LLM_SCORE_FIELDS"]
__version__ = "0.1.0"
