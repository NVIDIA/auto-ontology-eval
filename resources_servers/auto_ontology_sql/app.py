# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auto Ontology-grounded text-to-SQL resources server: the treatment arm.

Drives Auto Ontology's LangGraph agent **in-process**, exactly as the
single-arm harness it replaced did, so the two behave identically. Gym's policy
model is not in the loop: ``verify`` runs the agent itself and ignores the
rollout's model output. That costs RL-trainability and buys an apples-to-apples
comparison.

Grading is shared with the schema-only arm -- same executor, same BIRD
set-equality rule, same failure codes -- so a difference in the number can only
come from the SQL.

Auto Ontology's own LLM calls go through two Gym model servers rather than
straight to the endpoint (see ``ontology_sql_eval.gym.model_routing``), each
tagged with the rollout it belongs to. With ``+observability_enabled=true`` Gym
captures them per rollout, and ``verify`` reports the question's token totals.

Three couplings with Auto Ontology this module must respect:

* ``auto_ontology.retrieval.text_to_sql.main`` builds its LLM client and compiles the
  graph *at import time*, so the environment is loaded -- and the endpoints
  repointed at Gym -- before Auto Ontology imports.
* Auto Ontology is installed from the parent project's pinned Git dependency.
* ``stream_agent_response`` is a blocking generator, so it runs on a worker
  thread under a semaphore rather than on the event loop.
"""

from __future__ import annotations

import asyncio
import logging
import os
from urllib.parse import urlparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from pydantic import ConfigDict, Field

# Must precede Auto Ontology imports: it resolves endpoints and constructs an LLM
# client while its modules are being imported.
load_dotenv()

from ontology_sql_eval.gym.model_routing import (  # noqa: E402
    captured_usage,
    route_through_gym,
    tag_rollout_requests,
)

route_through_gym()
tag_rollout_requests()

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
from nemo_gym.rollout_correlation import current_rollout_id  # noqa: E402

from auto_ontology.connectors import get_connectors  # noqa: E402
from auto_ontology.retrieval.text_to_sql.main import stream_agent_response  # noqa: E402
from auto_ontology.retrieval.text_to_sql.state import TextToSQLPayload  # noqa: E402
from auto_ontology.utils import (  # noqa: E402
    get_data_objects_retriever,
    get_semantic_objects_retriever,
)

from ontology_sql_eval.prompt_inputs import (  # noqa: E402
    load_sql_examples,
    load_value_anchors,
)

from ontology_sql_eval.gym.exec_match import (  # noqa: E402
    drop_unusable_tasks,
    FailureCode,
    MySqlExecutor,
    PostgresExecutor,
    SqliteExecutor,
    execute_and_compare,
    timeout_for,
)
from ontology_sql_eval.gym.tasks import DATASETS  # noqa: E402

logger = logging.getLogger(__name__)

# Gym configures no logging, so a module logger has no handler and Python's
# last-resort handler drops anything below WARNING. Every startup diagnostic
# this server emits -- which connectors resolved, how many exemplars and
# anchors loaded -- was therefore invisible. Attach a handler to our own
# loggers only: the root logger would also surface httpx, which logs a line per
# request and runs ~23 of them per question.
_log_handler = logging.StreamHandler()
_log_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
for _logger_name in (__name__, "ontology_sql_eval"):
    _own_logger = logging.getLogger(_logger_name)
    _own_logger.addHandler(_log_handler)
    _own_logger.setLevel(logging.INFO)
    _own_logger.propagate = False


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
# taking it as its own field. Probe rather than pin a version, so one revision of
# this harness runs against both.
_SUPPORTS_EVIDENCE = "evidence" in getattr(TextToSQLPayload, "__annotations__", {})

# Same probe for the two prompt inputs. On a build that predates either
# parameter the payload key would be silently ignored, so a run that asked for
# exemplars would quietly get none; refuse at startup instead.
_SUPPORTS_SQL_EXAMPLES = "sql_examples" in getattr(
    TextToSQLPayload, "__annotations__", {}
)
_SUPPORTS_VALUE_ANCHORS = "value_anchors" in getattr(
    TextToSQLPayload, "__annotations__", {}
)


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

    # Question-keyed prompt inputs, both optional and both off by default.
    # Paths are relative to the repo root. These change what the agent is told,
    # so a run that sets them is NOT comparable to a schema-only control that
    # did not get the equivalent -- the delta then covers the exemplars and
    # anchors as well as the ontology.
    sql_examples_csv: Optional[str] = None
    sql_examples_k: int = 5
    sql_examples_min_score: float = 0.0
    value_anchors_csv: Optional[str] = None


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
    # The dataset's own question_id, which is what the prompt-input CSVs are
    # keyed by. Int on BIRD/WWI but a string like "FDA0002" on FDABench, so it
    # is accepted as either and stringified at lookup.
    id: Optional[Union[int, str]] = None


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
    # "empty_gold" / "nonempty_gold", or None when gold never ran. Grouped on in
    # compute_metrics for BEAVER's second headline number, accuracy over
    # nonempty_gold tasks only.
    gold_result: Optional[str] = None
    policy_output_ignored: bool = True
    # How many exemplars and anchors this question's payload actually carried.
    # Recorded per rollout because a mistyped path or a stale question_id key
    # yields zero of them without failing, and the run would then be filed as
    # "with prompt inputs" while having had none.
    sql_examples_used: int = 0
    value_anchors_used: int = 0
    # Auto Ontology's own LLM usage for this question, as Gym captured it. None
    # rather than 0 when nothing was captured (observability off), so a run
    # without capture reports no token metrics instead of a mean of zero.
    agent_llm_calls: Optional[int] = None
    agent_llm_calls_without_usage: Optional[int] = None
    agent_input_tokens: Optional[int] = None
    agent_cached_input_tokens: Optional[int] = None
    agent_output_tokens: Optional[int] = None
    agent_total_tokens: Optional[int] = None


# The per-question token fields, surfaced in key metrics as their means.
_TOKEN_FIELDS = (
    "agent_llm_calls",
    "agent_input_tokens",
    "agent_output_tokens",
    "agent_total_tokens",
)


class AutoOntologySqlResourcesServer(SimpleResourcesServer):
    config: AutoOntologySqlResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._root = Path(__file__).resolve().parent.parent.parent
        self._executors: Dict[str, Any] = {}
        self._semaphore = asyncio.Semaphore(self.config.max_concurrency)
        # Logged because this is the arm's throughput ceiling and it is set by a
        # Hydra override that silently does nothing if the key path is wrong --
        # leaving the run 3x slower than intended with nothing to show why.
        logger.info(
            "Agent concurrency ceiling: max_concurrency=%d",
            self.config.max_concurrency,
        )

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

        self._sql_examples: Dict[str, List[Dict[str, str]]] = {}
        if self.config.sql_examples_csv and self.config.sql_examples_k > 0:
            if not _SUPPORTS_SQL_EXAMPLES:
                raise RuntimeError(
                    "sql_examples_csv is set but this Auto Ontology build takes no "
                    "'sql_examples' payload field; the examples would be dropped."
                )
            self._sql_examples = load_sql_examples(
                self._root / self.config.sql_examples_csv,
                self.config.sql_examples_k,
                self.config.sql_examples_min_score,
            )

        self._value_anchors: Dict[str, List[Dict[str, str]]] = {}
        if self.config.value_anchors_csv:
            if not _SUPPORTS_VALUE_ANCHORS:
                raise RuntimeError(
                    "value_anchors_csv is set but this Auto Ontology build takes no "
                    "'value_anchors' payload field; the anchors would be dropped."
                )
            self._value_anchors = load_value_anchors(
                self._root / self.config.value_anchors_csv
            )

        logger.info(
            "Prompt inputs: sql_examples=%s (k=%d, min_score=%.2f, %d questions "
            "keyed) value_anchors=%s (%d questions keyed)",
            self.config.sql_examples_csv or "off",
            self.config.sql_examples_k,
            self.config.sql_examples_min_score,
            len(self._sql_examples),
            self.config.value_anchors_csv or "off",
            len(self._value_anchors),
        )

    def _executor(self, dataset: str) -> Any:
        if dataset in self._executors:
            return self._executors[dataset]

        spec = DATASETS.get(dataset)
        if spec is None:
            raise KeyError(f"unknown dataset {dataset!r}")
        timeout_s = timeout_for(dataset, self.config.sql_execution_timeout_s)

        if spec.dialect == "mysql":
            dsns = mysql_dsns_by_db(os.environ.get("CONNECTION_STRINGS", ""))
            if not dsns:
                raise RuntimeError(
                    f"no mysql:// entry in CONNECTION_STRINGS for dataset {dataset!r}; "
                    "seed it with scripts/seed_beaverbench.py --import-mysql"
                )
            executor: Any = MySqlExecutor(dsns, timeout_s=timeout_s)
        elif spec.dialect == "postgres":
            dsn = (
                (self.config.postgres_dsn or os.environ.get("CONNECTION_STRINGS", ""))
                .split(",")[0]
                .strip()
            )
            if not dsn:
                raise RuntimeError(f"no Postgres DSN available for dataset {dataset!r}")
            executor = PostgresExecutor(dsn, timeout_s=timeout_s)
        else:
            executor = SqliteExecutor(
                self._root / self.config.datasets_dir / spec.db_root,
                timeout_s=timeout_s,
            )

        self._executors[dataset] = executor
        return executor

    def _run_agent(
        self, question: str, evidence: str, db_id: str, question_id: str
    ) -> Dict[str, Any]:
        """Drive Auto Ontology's graph to completion. Blocking; call on a worker thread."""
        connector = self._connectors.get(db_id)
        if connector is None:
            raise KeyError(
                f"no Auto Ontology connector for db_id {db_id!r}; have {sorted(self._connectors)}"
            )

        payload: Dict[str, Any] = {
            "question": question,
            # Pins the intent to CALCULATION. Every BIRD question is answered by
            # running SQL, so letting the intent classifier route elsewhere only
            # creates ways to lose.
            "calculation_only": True,
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
            # Set unconditionally, as the single-arm harness this arm replaced
            # did. TextToSQLPayload is a TypedDict, so on an Auto Ontology revision that
            # predates the parameter this is an inert extra key rather than an
            # error -- it simply has no effect.
            "shorten_answer": True,
            # Enables the validate_sql_values node, which checks filter literals
            # against the database before the SQL is accepted. Costly -- it is the
            # single largest contributor to wall clock -- but the legacy harness
            # runs with it, so omitting it makes the two arms incomparable.
            "validate_sql_values": True,
        }
        if _SUPPORTS_EVIDENCE:
            payload["evidence"] = evidence
        elif evidence:
            payload["question"] = f"{question}\n\nEvidence: {evidence}"

        # Only set when this question has an entry: an empty list would tell the
        # agent "no exemplars exist", which is a different instruction from
        # leaving the key off, and the min_score gate deliberately withholds
        # exemplars from questions whose best match is too weak.
        examples = self._sql_examples.get(question_id)
        if examples:
            payload["sql_examples"] = examples
        anchors = self._value_anchors.get(question_id)
        if anchors:
            payload["value_anchors"] = anchors

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

        return {
            "answer": answer,
            "nodes": nodes,
            "error": error,
            "sql_examples_used": len(examples or []),
            "value_anchors_used": len(anchors or []),
        }

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

        rollout_id = current_rollout_id()
        try:
            async with self._semaphore:
                result = await asyncio.to_thread(
                    self._run_agent,
                    body.question,
                    body.evidence,
                    body.db_id,
                    "" if body.id is None else str(body.id),
                )
        except Exception as exc:
            logger.exception(
                "Auto Ontology agent failed task_id=%s: %s", body.task_id, exc
            )
            # A failed agent has usually spent tokens before failing. Recorded
            # first: _without_prediction builds its response through _response,
            # which reads carried at call time.
            carried.update(await self._usage(rollout_id, body.task_id))
            return await self._without_prediction(
                body, _response, FailureCode.UNKNOWN_ERROR, agent_error=str(exc)
            )

        # Read by _response at call time, like the counts below.
        carried.update(await self._usage(rollout_id, body.task_id))

        answer = result["answer"]
        sql = str(answer.get("sql_code") or "").strip()
        nodes = result["nodes"]

        # Read by _response at call time, so every path below reports them. The
        # pre-agent error path above correctly leaves both at zero.
        carried["sql_examples_used"] = result["sql_examples_used"]
        carried["value_anchors_used"] = result["value_anchors_used"]

        # Auto Ontology treats "cannot build a query" as a normal terminal state, not an
        # error, so an exhausted-retry rate limit arrives here as empty SQL with
        # nothing on the error channel. Counting it separately is what stops an
        # outage from reading as a low score.
        if not sql:
            return await self._without_prediction(
                body,
                _response,
                FailureCode.NO_MODEL_OUTPUT,
                agent_answer=str(answer.get("response") or ""),
                agent_error=result["error"],
                agent_nodes=nodes,
            )

        try:
            match, err, gold_result = await execute_and_compare(
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
            gold_result=gold_result,
        )

    async def _without_prediction(
        self,
        body: AutoOntologySqlVerifyRequest,
        respond: Any,
        failure: FailureCode,
        **kwargs: Any,
    ) -> AutoOntologySqlVerifyResponse:
        """Score a task the agent produced no SQL for.

        Gold is run even though there is nothing to compare: a task whose
        benchmark query is broken must leave the denominator rather than be
        recorded as the agent producing a wrong answer -- and on BEAVER, whose
        official evaluator scores a missing prediction as an empty result, this
        can match. ``failure`` is kept either way, so the health gate sees it.
        """
        try:
            match, err, gold_result = await execute_and_compare(
                self._executor(body.dataset),
                body.db_id,
                body.gt_sql,
                None,
                body.dataset,
            )
        except Exception as exc:
            # Often the same fault that broke the agent (an unknown dataset, an
            # unreachable database); the original failure is the one to report.
            logger.exception("gold check failed task_id=%s: %s", body.task_id, exc)
            return respond(0.0, failure, **kwargs)
        if err in _GOLD_FAILURE:
            # An empty answer over broken gold says nothing; an agent crash
            # still does, so it keeps its own code for the health counters.
            if failure is FailureCode.NO_MODEL_OUTPUT:
                return respond(0.0, _GOLD_FAILURE[err], **kwargs)
            return respond(0.0, failure, **kwargs)
        return respond(
            1.0 if match else 0.0,
            failure,
            execution_match=match,
            gold_result=gold_result,
            **kwargs,
        )

    async def _usage(
        self, rollout_id: Optional[str], task_id: Optional[str]
    ) -> Dict[str, int]:
        """This rollout's captured token totals, logged per question. Never raises."""
        try:
            usage = await asyncio.to_thread(captured_usage, rollout_id)
        except Exception:
            logger.warning(
                "Could not read token usage for task_id=%s", task_id, exc_info=True
            )
            return {}
        if usage:
            logger.info(
                "task_id=%s tokens: %d in (%d cached), %d out, %d total over %d "
                "LLM calls",
                task_id,
                usage["agent_input_tokens"],
                usage["agent_cached_input_tokens"],
                usage["agent_output_tokens"],
                usage["agent_total_tokens"],
                usage["agent_llm_calls"],
            )
        return usage

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
        for subset in ("difficulty", "dataset", "gold_result"):
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
        # On BEAVER a gold query that errors or times out is scored as an empty
        # result rather than excluded, so these tasks stay in the denominator;
        # this is the official evaluator's "gold SQL errors/timeouts" line.
        metrics["health/gold_failure_scored_as_empty_count"] = sum(
            1 for r in rollouts if r.get("gold_result") == "failed_gold"
        )
        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        key: Dict[str, Any] = {
            name: value
            for name, value in agent_metrics.items()
            if name.startswith("health/")
        }
        for field in _TOKEN_FIELDS:
            if f"mean/{field}" in agent_metrics:
                key[f"mean/{field}"] = agent_metrics[f"mean/{field}"]
        key.update(
            highest_k_metrics(
                agent_metrics, "pass@1[avg-of-{k}]", score_names=["accuracy"]
            )
        )
        key.update(
            highest_k_metrics(agent_metrics, "pass@{k}", score_names=["accuracy"])
        )
        # BEAVER reports accuracy both with and without empty-gold questions;
        # this is the "without". Present for every dataset, since the split is
        # just as informative elsewhere.
        key.update(
            highest_k_metrics(
                agent_metrics,
                "nonempty_gold/pass@1[avg-of-{k}]",
                score_names=["accuracy"],
            )
        )
        return key


if __name__ == "__main__":
    AutoOntologySqlResourcesServer.run_webserver()
