# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema-only text-to-SQL resources server: the control arm.

The model gets the question plus a raw schema dump and answers in one shot. No
tools, no retrieval, no ontology -- whatever this arm scores is what a competent
LLM achieves from the schema alone, and the Auto Ontology arm's lift is measured against
it.

Deliberately close to NeMo Gym's own ``bird_sql`` server so the baseline stays
comparable to published BIRD numbers. It differs in three ways, all forced by
our corpora spanning more than BIRD dev:

* executors are chosen per dataset, so Postgres (WideWorldImporters) works
  alongside SQLite (BIRD, FDABench);
* ``task_id`` is a string, because FDABench's ids are not integers;
* a ``no_model_output`` failure code is reported and surfaced as its own metric.
"""

from __future__ import annotations

import logging
import os
from urllib.parse import urlparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.reward_profile import (
    compute_pass_majority_metrics,
    compute_subset_metrics,
    highest_k_metrics,
)

from ontology_sql_eval.gym.exec_match import (
    drop_unusable_tasks,
    FailureCode,
    MySqlExecutor,
    PostgresExecutor,
    SqliteExecutor,
    execute_and_compare,
    gold_is_runnable,
    extract_sql,
    has_sql_codeblock,
)
from ontology_sql_eval.gym.tasks import DATASETS

logger = logging.getLogger(__name__)

_GOLD_FAILURE = {
    "gold_sql_timeout": FailureCode.GOLD_EXECUTION_TIMEOUT,
    "gold_sql_error": FailureCode.GOLD_EXECUTION_ERROR,
}


def mysql_dsns_by_db(connection_strings: str) -> Dict[str, str]:
    """Map ``db_id`` -> MySQL DSN from a CONNECTION_STRINGS value.

    BEAVER runs one physical database per ``db_id``, so the DSNs are keyed by the
    database name in each URI rather than collapsed to a single connection.
    """
    out: Dict[str, str] = {}
    for raw in connection_strings.split(","):
        dsn = raw.strip()
        if not dsn.startswith("mysql://"):
            continue
        database = urlparse(dsn).path.lstrip("/")
        if database:
            out[database] = dsn
    return out


# Failures that mean the benchmark, not the model, is at fault.
_UNUSABLE = {FailureCode.GOLD_EXECUTION_ERROR, FailureCode.GOLD_EXECUTION_TIMEOUT}


class SchemaOnlySqlResourcesServerConfig(BaseResourcesServerConfig):
    name: str = "schema_only_sql"
    datasets_dir: str = "datasets"
    max_concurrency: int = 32
    sql_execution_timeout_s: float = 30.0
    # Only needed for the Postgres-backed dataset; falls back to the env var the
    # rest of the harness already uses.
    postgres_dsn: Optional[str] = None


class SchemaOnlySqlVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    question: str
    gt_sql: str
    db_id: str
    dataset: str
    dialect: str = "sqlite"
    difficulty: Optional[str] = None
    task_id: Optional[str] = None


class SchemaOnlySqlVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    # Read by nemo-gym's token-id capture to keep a sample out of training data.
    # It does *not* affect pass@k -- reward_profile ignores it -- so the accuracy
    # denominator is corrected separately in compute_metrics.
    instance_config: Dict[str, Any] = Field(
        default_factory=lambda: {"mask_sample": False}
    )

    question: str
    gt_sql: str
    db_id: str
    dataset: str
    dialect: str = "sqlite"
    difficulty: Optional[str] = None
    task_id: Optional[str] = None
    model_output: str
    extracted_sql: Optional[str] = None
    had_codeblock: bool = False
    execution_match: bool = False
    failure_reason: Optional[FailureCode] = None


class SchemaOnlySqlResourcesServer(SimpleResourcesServer):
    config: SchemaOnlySqlResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._root = Path(__file__).resolve().parent.parent.parent
        self._executors: Dict[str, Any] = {}

    def _executor(self, dataset: str) -> Any:
        """Executor for a dataset, built once and reused.

        Built lazily rather than up front so a run over the SQLite datasets does
        not require a reachable Postgres instance.
        """
        if dataset in self._executors:
            return self._executors[dataset]

        spec = DATASETS.get(dataset)
        if spec is None:
            raise KeyError(f"unknown dataset {dataset!r}")

        if spec.dialect == "mysql":
            dsns = mysql_dsns_by_db(os.environ.get("CONNECTION_STRINGS", ""))
            if not dsns:
                raise RuntimeError(
                    f"no mysql:// entry in CONNECTION_STRINGS for dataset {dataset!r}; "
                    "seed it with scripts/seed_beaverbench.py --import-mysql"
                )
            executor: Any = MySqlExecutor(
                dsns, timeout_s=self.config.sql_execution_timeout_s
            )
        elif spec.dialect == "postgres":
            dsn = self.config.postgres_dsn or os.environ.get("CONNECTION_STRINGS", "")
            dsn = dsn.split(",")[0].strip()
            if not dsn:
                raise RuntimeError(f"no Postgres DSN available for dataset {dataset!r}")
            executor = PostgresExecutor(
                dsn,
                timeout_s=self.config.sql_execution_timeout_s,
            )
        else:
            executor = SqliteExecutor(
                self._root / self.config.datasets_dir / spec.db_root,
                max_concurrency=self.config.max_concurrency,
                timeout_s=self.config.sql_execution_timeout_s,
            )

        self._executors[dataset] = executor
        return executor

    async def verify(
        self, body: SchemaOnlySqlVerifyRequest
    ) -> SchemaOnlySqlVerifyResponse:
        generated = body.response.output_text or ""

        carried = body.model_dump()
        # sql_context is the entire CREATE TABLE dump (~16k chars); echoing it
        # into every rollout line multiplies the output by --num-repeats for no
        # diagnostic value, since it is reproducible from the task JSONL.
        carried.pop("sql_context", None)
        for field in (
            "question",
            "gt_sql",
            "db_id",
            "dataset",
            "dialect",
            "difficulty",
            "task_id",
        ):
            carried.pop(field, None)

        def _response(
            reward: float, failure: FailureCode, **kwargs: Any
        ) -> SchemaOnlySqlVerifyResponse:
            return SchemaOnlySqlVerifyResponse(
                **carried,
                reward=reward,
                # Masked when the *gold* query never ran: the rollout says
                # nothing about the policy, so it should not train on it.
                instance_config={
                    "mask_sample": failure in _UNUSABLE,
                },
                question=body.question,
                gt_sql=body.gt_sql,
                db_id=body.db_id,
                dataset=body.dataset,
                dialect=body.dialect,
                difficulty=body.difficulty,
                task_id=body.task_id,
                model_output=generated,
                failure_reason=failure,
                **kwargs,
            )

        # An empty response is not a wrong answer -- it usually means the call
        # never succeeded (rate limiting, timeout). Kept as its own failure code
        # so an infrastructure outage cannot be read as a low benchmark score.
        if not generated.strip():
            # Check the gold query even though there is nothing to compare: a
            # task whose benchmark query is broken must leave the denominator
            # rather than be recorded as a miss.
            gold_bad = await gold_is_runnable(
                self._executor(body.dataset), body.db_id, body.gt_sql
            )
            if gold_bad:
                return _response(0.0, _GOLD_FAILURE[gold_bad], execution_match=False)
            return _response(0.0, FailureCode.NO_MODEL_OUTPUT, execution_match=False)

        extracted = extract_sql(generated)
        had_block = has_sql_codeblock(generated)

        try:
            match, err = await execute_and_compare(
                self._executor(body.dataset),
                body.db_id,
                body.gt_sql,
                extracted,
                body.dataset,
            )
        except Exception as exc:
            logger.exception(
                "verify failed task_id=%s db_id=%s: %s", body.task_id, body.db_id, exc
            )
            return _response(
                0.0,
                FailureCode.UNKNOWN_ERROR,
                extracted_sql=extracted,
                had_codeblock=had_block,
                execution_match=False,
            )

        if err == "gold_sql_timeout":
            failure = FailureCode.GOLD_EXECUTION_TIMEOUT
        elif err == "gold_sql_error":
            failure = FailureCode.GOLD_EXECUTION_ERROR
        elif err == "pred_sql_timeout":
            failure = FailureCode.EXECUTION_TIMEOUT
        elif err == "pred_sql_error":
            failure = (
                FailureCode.EXECUTION_ERROR
                if had_block
                else FailureCode.NO_SQL_EXTRACTED
            )
        elif match:
            failure = FailureCode.NONE
        else:
            # The query ran and returned rows, they just did not match. That is
            # an ordinary wrong answer, not an execution failure.
            failure = (
                FailureCode.WRONG_RESULT if had_block else FailureCode.NO_SQL_EXTRACTED
            )

        return _response(
            1.0 if match else 0.0,
            failure,
            extracted_sql=extracted,
            had_codeblock=had_block,
            execution_match=match,
        )

    @staticmethod
    def _score_fn(r: dict) -> Dict[str, float]:
        return {"accuracy": float(r.get("reward", 0.0) > 0)}

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Overall pass@k, plus per-difficulty, per-dataset, and health counters."""
        # Accuracy is computed over tasks whose gold query actually ran. A
        # benchmark query that errors or times out says nothing about the model,
        # so counting it as a miss understates every arm equally and invisibly.
        usable, dropped = drop_unusable_tasks(tasks)
        metrics, *_ = compute_pass_majority_metrics(
            usable, score_fn=self._score_fn, answer_key="extracted_sql"
        )
        for subset in ("difficulty", "dataset"):
            metrics.update(
                compute_subset_metrics(usable, subset, self._score_fn, "extracted_sql")
            )
        metrics["health/excluded_unusable_gold_count"] = dropped

        # Health counters. A rate-limited run and a genuinely weak model both
        # score near zero; these are what tell them apart, so they are reported
        # even when they are zero.
        rollouts = [r for task in tasks for r in task]
        total = len(rollouts) or 1
        for code in (
            FailureCode.NO_MODEL_OUTPUT,
            FailureCode.GOLD_EXECUTION_ERROR,
            FailureCode.GOLD_EXECUTION_TIMEOUT,
            # Without this, an arm that errors on every task (a db_id the
            # connectors do not know, say) reports 0% with every health
            # counter green, and compare_arms.py publishes it.
            FailureCode.UNKNOWN_ERROR,
        ):
            n = sum(1 for r in rollouts if r.get("failure_reason") == code.value)
            metrics[f"health/{code.value}_rate"] = n / total
            metrics[f"health/{code.value}_count"] = n
        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        key: Dict[str, Any] = {}
        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]
        for name, value in agent_metrics.items():
            if name.startswith("health/"):
                key[name] = value
        key.update(
            highest_k_metrics(
                agent_metrics, "pass@1[avg-of-{k}]", score_names=["accuracy"]
            )
        )
        key.update(
            highest_k_metrics(agent_metrics, "pass@{k}", score_names=["accuracy"])
        )
        return key


if __name__ == "__main__":
    SchemaOnlySqlResourcesServer.run_webserver()
