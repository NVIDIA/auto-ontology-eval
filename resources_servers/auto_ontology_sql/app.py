# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auto Ontology-grounded text-to-SQL resources server: the treatment arm.

Drives Auto Ontology's LangGraph agent **in-process**, exactly as
``ontology_sql_eval.retrieval.eval_chatbot`` does, so this arm behaves
identically to the harness it replaces. Gym's policy model is not in the loop:
``verify`` runs the agent itself and ignores the rollout's model output. That
costs RL-trainability and buys an apples-to-apples comparison.

Grading is shared with the schema-only arm -- same executor, same BIRD
set-equality rule, same failure codes -- so a difference in the number can only
come from the SQL.

Sharing one interpreter with ``nemo_gym`` took two dependency decisions, both
recorded in ``overrides.txt`` and ``pyproject.toml``: an override lifting
nemo-retriever's stale ``prometheus-fastapi-instrumentator<8`` cap (it pins
``starlette<1.0.0``, which collides with nemo-gym's ray[serve]), and a
``langchain-openai<1.3.5`` bound (nemo-gym pins ``openai==2.44.0``).

Three couplings with Auto Ontology this module must respect:

* ``auto_ontology.retrieval.text_to_sql.main`` builds its LLM client and compiles the
  graph *at import time*, so the environment is loaded before the gsf imports.
* Auto Ontology is imported from a checkout on ``PYTHONPATH``; it is not installed here.
* ``stream_agent_response`` is a blocking generator, so it runs on a worker
  thread under a semaphore rather than on the event loop.
"""

from __future__ import annotations

import asyncio
import logging
import os
from urllib.parse import urlparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import ConfigDict, Field

# Must precede the gsf imports: Auto Ontology resolves endpoints and constructs an LLM
# client while its modules are being imported.
load_dotenv()

from nemo_gym.base_resources_server import (  # noqa: E402
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.reward_profile import (  # noqa: E402
    compute_pass_majority_metrics,
    compute_subset_metrics,
    highest_k_metrics,
)

from auto_ontology.connectors import get_connectors  # noqa: E402
from auto_ontology.retrieval.text_to_sql.main import stream_agent_response  # noqa: E402
from auto_ontology.retrieval.text_to_sql.state import TextToSQLPayload  # noqa: E402
from auto_ontology.utils import (  # noqa: E402
    get_data_objects_retriever,
    get_semantic_objects_retriever,
)

from ontology_sql_eval.gym.exec_match import (  # noqa: E402
    drop_unusable_tasks,
    FailureCode,
    MySqlExecutor,
    PostgresExecutor,
    SqliteExecutor,
    execute_and_compare,
    gold_is_runnable,
)
from ontology_sql_eval.gym.tasks import DATASETS  # noqa: E402

logger = logging.getLogger(__name__)


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

_GOLD_FAILURE = {
    "gold_sql_timeout": FailureCode.GOLD_EXECUTION_TIMEOUT,
    "gold_sql_error": FailureCode.GOLD_EXECUTION_ERROR,
}

# Older Auto Ontology builds fold the BIRD-style hint into the question text instead of
# taking it as its own field. Probe rather than pin a version, so one checkout of
# this harness runs against both.
_SUPPORTS_EVIDENCE = "evidence" in getattr(TextToSQLPayload, "__annotations__", {})


class AutoOntologySqlResourcesServerConfig(BaseResourcesServerConfig):
    name: str = "auto_ontology_sql"
    datasets_dir: str = "datasets"
    # Sized against the *model endpoint's* rate limit, not CPU. A 60-question run at 8
    # concurrent questions against bedrock-claude-opus-4-8 came back 66% HTTP
    # 429; Auto Ontology's retry turned that into empty SQL and a plausible-looking 13.3%
    # score. Three is clean.
    max_concurrency: int = 3
    sql_execution_timeout_s: float = 30.0
    postgres_dsn: Optional[str] = None


class AutoOntologySqlVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    question: str
    gt_sql: str
    db_id: str
    dataset: str
    dialect: str = "sqlite"
    difficulty: Optional[str] = None
    task_id: Optional[str] = None
    evidence: str = ""


class AutoOntologySqlVerifyResponse(BaseVerifyResponse):
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
    extracted_sql: Optional[str] = None
    agent_answer: str = ""
    agent_error: str = ""
    agent_nodes: List[str] = []
    execution_match: bool = False
    failure_reason: Optional[FailureCode] = None
    policy_output_ignored: bool = True


class AutoOntologySqlResourcesServer(SimpleResourcesServer):
    config: AutoOntologySqlResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._root = Path(__file__).resolve().parent.parent.parent
        self._executors: Dict[str, Any] = {}
        self._semaphore = asyncio.Semaphore(self.config.max_concurrency)

        inflight = (os.environ.get("LLM_MAX_INFLIGHT") or "0").strip()
        if inflight != "0":
            # This process-wide bound inside Auto Ontology silently capped throughput at 6
            # once before; if it is set here it will distort this arm's timings.
            logger.warning(
                "LLM_MAX_INFLIGHT=%s bounds Auto Ontology's in-flight LLM calls "
                "process-wide and will distort this arm; 0 disables it.",
                inflight,
            )

        # Built once: Auto Ontology's retrievers are process-wide singletons and the
        # connectors hold real database handles.
        self._data_retriever = get_data_objects_retriever()
        self._semantic_retriever = get_semantic_objects_retriever()
        self._connectors = {c.database_name: c for c in get_connectors()}
        logger.info("Auto Ontology connectors available: %s", sorted(self._connectors))

    def _executor(self, dataset: str) -> Any:
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
            dsn = (
                (self.config.postgres_dsn or os.environ.get("CONNECTION_STRINGS", ""))
                .split(",")[0]
                .strip()
            )
            if not dsn:
                raise RuntimeError(f"no Postgres DSN available for dataset {dataset!r}")
            executor = PostgresExecutor(
                dsn, timeout_s=self.config.sql_execution_timeout_s
            )
        else:
            executor = SqliteExecutor(
                self._root / self.config.datasets_dir / spec.db_root,
                timeout_s=self.config.sql_execution_timeout_s,
            )

        self._executors[dataset] = executor
        return executor

    def _run_agent(self, question: str, evidence: str, db_id: str) -> Dict[str, Any]:
        """Drive Auto Ontology's graph to completion. Blocking; call on a worker thread."""
        connector = self._connectors.get(db_id)
        if connector is None:
            raise KeyError(
                f"no Auto Ontology connector for db_id {db_id!r}; have {sorted(self._connectors)}"
            )

        payload: Dict[str, Any] = {
            "question": question,
            "data_retriever": self._data_retriever,
            "semantic_retriever": self._semantic_retriever,
            "connectors": [connector],
            "path_state": {},
            "custom_prompts": "",
            "acronyms": [],
            # Appends "Return exactly the requested output fields and NO others"
            # to Auto Ontology's projection rules. Not cosmetic: extra columns break the
            # set(gold) == set(pred) comparison outright, so omitting this scores
            # Auto Ontology with a handicap the legacy harness no longer carries.
            #
            # Set unconditionally, as ontology_sql_eval.retrieval.eval_chatbot
            # does. TextToSQLPayload is a TypedDict, so on a Auto Ontology checkout that
            # predates the parameter this is an inert extra key rather than an
            # error -- it simply has no effect.
            "shorten_answer": True,
        }
        if _SUPPORTS_EVIDENCE:
            payload["evidence"] = evidence
        elif evidence:
            payload["question"] = f"{question}\n\nEvidence: {evidence}"

        nodes: List[str] = []
        answer: Dict[str, Any] = {}
        error = ""
        for event in stream_agent_response(TextToSQLPayload(**payload)):
            kind = event.get("type")
            if kind == "step" and event.get("phase") == "end":
                nodes.append(event.get("node") or "")
            elif kind == "sql" and event.get("sql"):
                answer.setdefault("sql_code", event["sql"])
            elif kind == "result":
                answer = event.get("answer") or answer
            elif kind == "error":
                error = str(event.get("message") or "agent error")
                answer = event.get("partial_answer") or answer

        return {"answer": answer, "nodes": nodes, "error": error}

    async def verify(
        self, body: AutoOntologySqlVerifyRequest
    ) -> AutoOntologySqlVerifyResponse:
        carried = body.model_dump()
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
        ) -> AutoOntologySqlVerifyResponse:
            return AutoOntologySqlVerifyResponse(
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
                failure_reason=failure,
                **kwargs,
            )

        try:
            async with self._semaphore:
                result = await asyncio.to_thread(
                    self._run_agent, body.question, body.evidence, body.db_id
                )
        except Exception as exc:
            logger.exception(
                "Auto Ontology agent failed task_id=%s: %s", body.task_id, exc
            )
            return _response(0.0, FailureCode.UNKNOWN_ERROR, agent_error=str(exc))

        answer = result["answer"]
        sql = str(answer.get("sql_code") or "").strip()
        nodes = result["nodes"]

        # Auto Ontology treats "cannot build a query" as a normal terminal state, not an
        # error, so an exhausted-retry rate limit arrives here as empty SQL with
        # nothing on the error channel. Counting it separately is what stops an
        # outage from reading as a low score.
        if not sql:
            # As above: a broken gold query must leave the denominator rather
            # than be recorded as the agent producing a wrong answer.
            gold_bad = await gold_is_runnable(
                self._executor(body.dataset), body.db_id, body.gt_sql
            )
            if gold_bad:
                return _response(0.0, _GOLD_FAILURE[gold_bad], agent_nodes=nodes)
            return _response(
                0.0,
                FailureCode.NO_MODEL_OUTPUT,
                agent_answer=str(answer.get("response") or ""),
                agent_error=result["error"],
                agent_nodes=nodes,
            )

        try:
            match, err = await execute_and_compare(
                self._executor(body.dataset), body.db_id, body.gt_sql, sql, body.dataset
            )
        except Exception as exc:
            logger.exception("verify failed task_id=%s: %s", body.task_id, exc)
            return _response(
                0.0, FailureCode.UNKNOWN_ERROR, extracted_sql=sql, agent_nodes=nodes
            )

        if err == "gold_sql_timeout":
            failure = FailureCode.GOLD_EXECUTION_TIMEOUT
        elif err == "gold_sql_error":
            failure = FailureCode.GOLD_EXECUTION_ERROR
        elif err == "pred_sql_timeout":
            failure = FailureCode.EXECUTION_TIMEOUT
        elif err == "pred_sql_error":
            failure = FailureCode.EXECUTION_ERROR
        else:
            # Ran and returned rows, but the wrong ones -- a wrong answer, not
            # an execution failure.
            failure = FailureCode.NONE if match else FailureCode.WRONG_RESULT

        return _response(
            1.0 if match else 0.0,
            failure,
            extracted_sql=sql,
            agent_answer=str(answer.get("response") or ""),
            agent_error=result["error"],
            agent_nodes=nodes,
            execution_match=match,
        )

    @staticmethod
    def _score_fn(r: dict) -> Dict[str, float]:
        return {"accuracy": float(r.get("reward", 0.0) > 0)}

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
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
        key: Dict[str, Any] = {
            name: value
            for name, value in agent_metrics.items()
            if name.startswith("health/")
        }
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
    AutoOntologySqlResourcesServer.run_webserver()
