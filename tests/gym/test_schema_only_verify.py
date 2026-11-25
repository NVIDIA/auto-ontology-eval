# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import MagicMock

import pytest
from nemo_gym.server_utils import ServerClient

from ontology_sql_eval.gym.exec_match import FailureCode
from resources_servers.schema_only_sql.app import (
    SchemaOnlySqlResourcesServer,
    SchemaOnlySqlResourcesServerConfig,
    SchemaOnlySqlVerifyRequest,
)


class FixtureExecutor:
    timeout_s = 1.0

    def __init__(self, gold_rows):
        self.gold_rows = gold_rows
        self.calls = []

    async def execute(self, db_id, sql, *, timeout_s=None):
        self.calls.append(sql)
        assert db_id == "fixture"
        if sql == "SELECT 1":
            return [(1,)], ""
        if sql == "":
            return [], ""
        assert sql == "SELECT COUNT(*) FROM fixture"
        return self.gold_rows, ""


def verify(dataset, gold_rows, answer):
    executor = FixtureExecutor(gold_rows)
    server = SchemaOnlySqlResourcesServer(
        config=SchemaOnlySqlResourcesServerConfig(
            host="127.0.0.1", port=18000, entrypoint="app.py"
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    server._executors[dataset] = executor
    body = SchemaOnlySqlVerifyRequest.model_validate(
        {
            "responses_create_params": {"input": []},
            "response": {
                "id": "fixture",
                "created_at": 0.0,
                "model": "fixture",
                "object": "response",
                "output": [
                    {
                        "id": "msg-fixture",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": answer, "annotations": []}
                        ],
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            },
            "question": "How many rows are in this fixture?",
            "gt_sql": "SELECT COUNT(*) FROM fixture",
            "db_id": "fixture",
            "dataset": dataset,
        }
    )
    return asyncio.run(server.verify(body)), executor.calls


@pytest.mark.parametrize(
    ("answer", "had_block"),
    [
        ("I don't have SQL to submit.", False),
        ("```sql\n-- No query was produced\n```", True),
    ],
)
def test_bird_missing_sql_cannot_match_a_one_row_gold(answer, had_block):
    result, calls = verify("bird", [(1,)], answer)

    assert result.reward == 0.0
    assert not result.execution_match
    assert result.failure_reason == FailureCode.NO_SQL_EXTRACTED
    assert result.extracted_sql is None
    assert result.had_codeblock is had_block
    assert result.gold_result == "nonempty_gold"
    assert result.instance_config == {"mask_sample": False}
    assert calls == ["SELECT COUNT(*) FROM fixture"]


def test_beaver_missing_sql_can_match_empty_gold():
    result, calls = verify("beaverbench", [], "No query to submit.")

    assert result.reward == 1.0
    assert result.execution_match
    assert result.failure_reason == FailureCode.NO_SQL_EXTRACTED
    assert result.extracted_sql is None
    assert result.gold_result == "empty_gold"
    assert calls == ["SELECT COUNT(*) FROM fixture"]


def test_bird_submitted_select_one_can_match_a_one_row_gold():
    result, calls = verify("bird", [(1,)], "```sql\nSELECT 1\n```")

    assert result.reward == 1.0
    assert result.execution_match
    assert result.failure_reason == FailureCode.NONE
    assert result.extracted_sql == "SELECT 1"
    assert result.had_codeblock
    assert calls == ["SELECT 1", "SELECT COUNT(*) FROM fixture"]
