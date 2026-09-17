# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM-backed SQL scoring."""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import TYPE_CHECKING

from ontology_sql_eval.judge.config import Settings
from ontology_sql_eval.judge.models import SQL_SCORING_PROMPT, SqlScore

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

_RESULT_PREVIEW_LIMIT = 500
_MAX_ATTEMPTS = 3
_BACKOFF_BASE_S = 5.0


@lru_cache(maxsize=None)
def _build_llm(settings: Settings) -> "BaseChatModel":
    """Construct the chat model lazily and reuse it across rows."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.model_name,
        api_key=settings.api_key,
        base_url=settings.base_url,
        max_tokens=10240,
    )


def score_sql(
    settings: Settings,
    question: str,
    sql_code: str,
    ground_truth_sql: str,
    sql_result_preview: str,
) -> SqlScore | None:
    """Score a single SQL query, returning ``None`` if scoring is skipped/fails."""
    if not sql_code:
        return None

    prompt = SQL_SCORING_PROMPT.format(
        question=question,
        sql_code=sql_code,
        ground_truth_sql=ground_truth_sql or "(none provided)",
        sql_result_preview=(
            sql_result_preview[:_RESULT_PREVIEW_LIMIT]
            if sql_result_preview
            else "(none)"
        ),
    )

    result = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            llm = _build_llm(settings)
            result = llm.with_structured_output(SqlScore).invoke(prompt)
            break
        except Exception as exc:  # noqa: BLE001 - scoring is best-effort per row
            if attempt == _MAX_ATTEMPTS:
                logger.warning(
                    "LLM scoring failed after %d attempts for question %r: %s",
                    _MAX_ATTEMPTS,
                    question[:60],
                    exc,
                )
                return None
            delay = _BACKOFF_BASE_S * (2 ** (attempt - 1))
            logger.info(
                "LLM scoring attempt %d/%d failed for question %r (%s); "
                "retrying in %.0fs",
                attempt,
                _MAX_ATTEMPTS,
                question[:60],
                exc,
                delay,
            )
            time.sleep(delay)

    if isinstance(result, SqlScore):
        return result
    try:
        return SqlScore.model_validate(result)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not parse LLM score for question %r: %s", question[:60], exc
        )
        return None
