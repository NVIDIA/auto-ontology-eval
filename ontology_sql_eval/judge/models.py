# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data models and the prompt used for LLM-based SQL scoring."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

SQL_SCORING_PROMPT = """\
You are an expert SQL evaluator. Your task is to score SQL code based on three separate criteria:

1. **Logic Match**: How well does the SQL logic answer the given question?
2. **Semantic Match**: How well do the SQL response types match the expected types for the question?
3. **Ground Truth Similarity**: How similar is the SQL to the provided ground truth SQL?

**Logic Match Scoring Guidelines:**
- Scores should be between 0.0 and 1.0
- Evaluate ONLY based on the question — do NOT compare to the ground truth SQL.
- Ask: does this SQL correctly answer the question on its own merits?
  - Does the SQL use appropriate tables, columns, and filters?
  - Does the query logic match the question's requirements?
  - Are the joins, aggregations, and conditions correct?
  - Does it capture the business logic behind the question?
- Provide short text in logic_issues explaining what reduced the score 
(e.g., "Missing WHERE clause for date filter", "Wrong aggregation function", "Incorrect table join")

**Semantic Match Scoring Guidelines:**
- Scores should be between 0.0 and 1.0
- Evaluate if the SQL response types match what the question expects:
  - **Format Appropriateness**: Does the response format match what the question is asking for?
    * Questions asking for "the earliest date" or "the maximum value" should return
     a single value, not a table of multiple values
    * Questions asking for "top 5" should return exactly 5 rows (or fewer if data doesn't exist)
    * Questions asking for specific single values should not return multiple rows
  - **Data Completeness**: Does the response include all relevant information requested?
    * Questions asking for "sales with and without discount" should include BOTH categories in results
    * Questions asking for comparisons should include all relevant comparison groups
    * Questions asking for detailed breakdowns should include all requested dimensions
  - **Important**: If the SQL expected types include the user's question expected types, it is acceptable

**Final Weighted Score:**
- Calculate as: (logic_match * 0.5) + (semantic_match * 0.5)

**Ground Truth Similarity Guidelines:**
- Compare the SQL structure, logic, tables used, and expected results
- Be lenient with minor syntax differences or equivalent approaches
- Focus on semantic similarity rather than exact text matching

**Question:** {question}

**SQL Code to Evaluate:**
{sql_code}

**Ground Truth SQL:**
{ground_truth_sql}

**SQL Result Preview (if available):**
{sql_result_preview}

Please provide all scores, logic issues text, and boolean flags based on your comprehensive evaluation.
"""

# CSV column names appended to the output for each LLM score field.
LLM_SCORE_FIELDS = [
    "llm_logic_match",
    "llm_semantic_match",
    "llm_final_weighted_score",
    "llm_sql_vs_ground_truth",
    "llm_is_valid_sql",
    "llm_is_sql_returns_data",
    "llm_logic_issues",
]


class SqlScore(BaseModel):
    """Structured output returned by the LLM scorer."""

    # Azure's strict structured-output mode requires the generated JSON schema to
    # set ``additionalProperties: false``; ``extra="forbid"`` makes Pydantic emit it.
    model_config = ConfigDict(extra="forbid")

    logic_match: float = Field(ge=0.0, le=1.0)
    logic_issues: str
    semantic_match: float = Field(ge=0.0, le=1.0)
    final_weighted_score: float = Field(ge=0.0, le=1.0)
    sql_compared_to_ground_truth_score: float = Field(ge=0.0, le=1.0)
    is_valid_sql: bool
    is_sql_returns_data: bool

    def as_csv_row(self) -> dict[str, object]:
        """Map the score onto the ``LLM_SCORE_FIELDS`` CSV columns."""
        return {
            "llm_logic_match": self.logic_match,
            "llm_semantic_match": self.semantic_match,
            "llm_final_weighted_score": self.final_weighted_score,
            "llm_sql_vs_ground_truth": self.sql_compared_to_ground_truth_score,
            "llm_is_valid_sql": self.is_valid_sql,
            "llm_is_sql_returns_data": self.is_sql_returns_data,
            "llm_logic_issues": self.logic_issues,
        }
