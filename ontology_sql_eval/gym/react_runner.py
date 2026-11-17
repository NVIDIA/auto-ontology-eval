# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one Auto Ontology ReAct episode, inside the Auto Ontology agent's sandbox.

    python -m ontology_sql_eval.gym.react_runner TASK_JSON RESULT_JSON

The Auto Ontology Agent Server (``responses_api_agents/auto_ontology_agent``)
writes TASK_JSON into the sandbox and starts this module there. The sandbox's
image holds Auto Ontology, the source databases and the catalog (see
``docker/auto_ontology_agent``), and its environment names them.

The episode is the one Auto Ontology's own ``ReactSQLAgent`` runs, with the
settings this repository uses for a graded benchmark: SQL answers only, the
question as asked and its evidence apart. Its model is Auto Ontology's
reasoning client, pointed at the Gym Model Server URL in TASK_JSON, so every
model call goes through Gym on the rollout's route.

RESULT_JSON gets the submitted query, or none, and Auto Ontology's record of
the episode.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Auto Ontology's client needs a key; the Gym Model Server holds the provider's.
MODEL_SERVER_API_KEY = "gym-model-server"


def episode_environment(task: dict[str, Any]) -> dict[str, str]:
    """The settings Auto Ontology reads at import: the task's, then its model."""
    return {
        **{str(k): str(v) for k, v in (task.get("environment") or {}).items()},
        "REASONING_ENDPOINT": task["model_base_url"],
        "REASONING_MODEL": task["model"],
        "REASONING_API_KEY": MODEL_SERVER_API_KEY,
    }


def run(task: dict[str, Any]) -> dict[str, Any]:
    """Answer the task's question with one ReAct episode."""
    os.environ.update(episode_environment(task))

    from ontology_sql_eval.env import load_env

    load_env()

    # ruff: noqa: E402 - Auto Ontology reads its settings when it is imported.
    from auto_ontology.connectors import get_connectors
    from auto_ontology.retrieval.text_to_sql.react.context import ToolContext
    from auto_ontology.retrieval.text_to_sql.react.loop import run_episode
    from auto_ontology.retrieval.text_to_sql.react.run_sql import connector_for
    from auto_ontology.utils import (
        get_data_objects_retriever,
        get_semantic_objects_retriever,
    )
    from auto_ontology.utils.llm_invoke import get_llm_client

    database = task["database"]
    # The connectors the catalog was ingested with (CONNECTION_STRINGS), so
    # their database names match the catalog's.
    connectors = {c.database_name: c for c in get_connectors()}
    if database not in connectors:
        raise KeyError(f"no source database {database!r}; have {sorted(connectors)}")

    data_retriever = get_data_objects_retriever()
    ctx = ToolContext(
        retriever=get_semantic_objects_retriever() or data_retriever,
        data_retriever=data_retriever,
        connectors=[connectors[database]],
        database_name=database,
        numeric_type_alignment=False,
        information_answers=False,
        question=task["question"],
    )
    connector = connector_for(ctx)
    result = run_episode(
        get_llm_client(),
        ctx,
        question=task["question"],
        evidence=task.get("evidence") or None,
        dialect=getattr(connector, "dialect", None),
    )
    submission = result.submission
    ran = submission is not None and submission.db_response is not None
    return {
        "sql": submission.sql if ran else None,
        "reason": result.reason,
        "tool_calls": result.tool_calls,
        "trajectory": result.trajectory,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    task_path, result_path = (Path(arg) for arg in argv)
    result = run(json.loads(task_path.read_text()))
    result_path.write_text(json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
