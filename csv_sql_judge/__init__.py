# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""csv-sql-judge: LLM-powered re-scoring of Text-to-SQL evaluation CSVs."""

from csv_sql_judge.models import LLM_SCORE_FIELDS, SqlScore
from csv_sql_judge.runner import run

__all__ = ["run", "SqlScore", "LLM_SCORE_FIELDS"]
__version__ = "0.1.0"
