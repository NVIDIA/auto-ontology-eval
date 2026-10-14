# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route the Auto Ontology arm's LLM calls through NeMo Gym model servers.

Auto Ontology talks to two chat models: a *reasoning* one (SQL generation from
the semantic layer) and a *non-reasoning* one (nearly every other node). Left
alone it calls them directly, so Gym neither serves nor counts any of the ~19
calls a question costs. This module puts a Gym model server in front of each.

Nothing about the models is configured here. Both halves defer to Auto
Ontology's own resolution (``<ROLE>_ENDPOINT`` / ``_MODEL`` / ``_API_KEY``, then
``DEFAULT_MODELS_*``, then its built-in defaults), so a ``.env`` override or a
change to Auto Ontology's defaults carries through without an edit here:

* :func:`launch_env`, run by ``scripts/run_gym_arm.sh`` before Gym starts,
  resolves each role and exports it as ``AUTO_ONTOLOGY_GYM_<ROLE>_<FIELD>``. The
  arm's config builds one ``inference_provider`` model server per role from
  those variables.
* :func:`route_through_gym`, run inside the resources server before Auto
  Ontology is imported, repoints ``<ROLE>_ENDPOINT`` at that role's Gym server.
  Auto Ontology still resolves the model name itself, which keeps its client
  choice (``ChatOpenAI`` vs ``ChatNVIDIA``) the same as a direct run; the Gym
  server substitutes its configured model regardless.
* :func:`tag_rollout_requests`, also run inside the resources server, puts the
  current rollout's ``/ng-rollout/<id>`` prefix on every request to those
  servers. Gym keys token capture by that prefix, so without it the calls are
  served but counted against no question.
* :func:`captured_usage` reads one rollout's captured calls back, so ``verify``
  can report the question's token totals as rollout fields.

Embedding and reranking are untouched: Gym has no server type for either, so
retrieval still calls its endpoints directly.
"""

from __future__ import annotations

import logging
import os
import shlex
from typing import Any

__all__ = [
    "GYM_MODEL_SERVERS",
    "captured_usage",
    "launch_env",
    "route_through_gym",
    "tag_rollout_requests",
]

logger = logging.getLogger(__name__)

# Auto Ontology triplet prefix -> the Gym model server that stands in for it.
# The names must match the top-level keys in
# resources_servers/auto_ontology_sql/configs/auto_ontology_sql.yaml.
GYM_MODEL_SERVERS = {
    "REASONING": "auto_ontology_reasoning_model",
    "NON_REASONING": "auto_ontology_non_reasoning_model",
}

_FIELDS = ("ENDPOINT", "MODEL", "API_KEY")
_LAUNCH_PREFIX = "AUTO_ONTOLOGY_GYM_"

# Root URLs (scheme://host:port) of the servers route_through_gym pointed Auto
# Ontology at. Only requests to these are rewritten by tag_rollout_requests.
_routed_roots: tuple[str, ...] = ()


def launch_env() -> dict[str, str]:
    """Auto Ontology's resolved model triplets, keyed for the arm's Gym config.

    Reads the current environment, so ``.env`` must already be loaded.
    """
    # Imported here: it pulls in the auto_ontology package, which the rest of
    # this package must not require.
    from auto_ontology.utils.model_config import resolve

    env: dict[str, str] = {}
    for role in GYM_MODEL_SERVERS:
        for field in _FIELDS:
            value = resolve(role, field)
            if not value:
                raise SystemExit(
                    f"Auto Ontology resolves no {role}_{field}; set it, or "
                    f"DEFAULT_MODELS_{field}, in .env"
                )
            env[f"{_LAUNCH_PREFIX}{role}_{field}"] = value
    return env


def route_through_gym() -> None:
    """Point each Auto Ontology role's endpoint at its Gym model server.

    Must run before Auto Ontology is imported: its text-to-SQL module builds
    both chat clients at import time.
    """
    from nemo_gym.server_utils import get_server_url

    global _routed_roots
    roots = []
    for role, server in GYM_MODEL_SERVERS.items():
        root = get_server_url(server).rstrip("/")
        os.environ[f"{role}_ENDPOINT"] = f"{root}/v1"
        roots.append(root)
    _routed_roots = tuple(roots)


def with_rollout_prefix(url: str, rollout_id: str | None) -> str:
    """*url* moved under ``/ng-rollout/<rollout_id>`` if it targets a routed server.

    Anything else -- embedding and reranking endpoints, an already-prefixed
    path, a call made outside any rollout -- is returned unchanged.
    """
    from nemo_gym.config_types import ROLLOUT_PATH_PREFIX
    from nemo_gym.server_utils import apply_rollout_prefix

    for root in _routed_roots:
        if not url.startswith(f"{root}/"):
            continue
        rest = url[len(root) :]
        if rest.startswith(f"/{ROLLOUT_PATH_PREFIX}/"):
            return url
        if not rollout_id:
            # Served, but captured against no rollout: the question's token
            # count comes up short with nothing else to show for it.
            logger.warning(
                "Auto Ontology called %s outside any rollout; its tokens will "
                "not be counted against a question",
                url,
            )
            return url
        return apply_rollout_prefix(root, rollout_id) + rest
    return url


def tag_rollout_requests() -> None:
    """Prefix every request to a routed server with the current rollout id.

    Patched at the HTTP layer because Auto Ontology builds its chat clients in
    many places, some per call, and with two libraries: ``ChatOpenAI`` sends
    through httpx, ``ChatNVIDIA`` through requests. The rollout id comes from
    Gym's context variable, which ``asyncio.to_thread`` and LangGraph's
    executor both copy onto the threads that make the calls. Idempotent.
    """
    import httpx
    import requests
    from nemo_gym.rollout_correlation import current_rollout_id

    if getattr(httpx.Client.send, "_gym_rollout_tagged", False):
        return

    def _tag_httpx(request: httpx.Request) -> None:
        tagged = with_rollout_prefix(str(request.url), current_rollout_id())
        if tagged != str(request.url):
            request.url = httpx.URL(tagged)

    def _wrap(send: Any, tag: Any) -> Any:
        def wrapped(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
            tag(request)
            return send(self, request, *args, **kwargs)

        wrapped._gym_rollout_tagged = True  # type: ignore[attr-defined]
        return wrapped

    def _tag_requests(request: requests.PreparedRequest) -> None:
        request.url = with_rollout_prefix(str(request.url), current_rollout_id())

    httpx.Client.send = _wrap(httpx.Client.send, _tag_httpx)  # type: ignore[method-assign]
    httpx.AsyncClient.send = _wrap(httpx.AsyncClient.send, _tag_httpx)  # type: ignore[method-assign]
    requests.Session.send = _wrap(requests.Session.send, _tag_requests)  # type: ignore[method-assign]


def captured_usage(rollout_id: str | None) -> dict[str, int]:
    """Token totals Gym captured for this rollout's Auto Ontology calls.

    Returned as flat ints so ``verify`` can put them on the rollout, where Gym
    aggregates every numeric field into ``mean/...`` metrics. Gym's own
    per-rollout ``ng_model_call_capture`` block is dropped before metrics are
    computed, so without this the totals would be in the rollouts file only.

    The policy model's call is captured under the same rollout and is left
    out: the Auto Ontology arm ignores its output. Empty when observability is
    off or nothing was captured.
    """
    if not rollout_id:
        return {}
    from nemo_gym.base_responses_api_model import (
        CaptureStore,
        model_call_capture_dirs_from_config,
        read_available_model_call_records,
    )
    from nemo_gym.global_config import get_global_config_dict

    servers = set(GYM_MODEL_SERVERS.values())
    calls = []
    for directory in model_call_capture_dirs_from_config(get_global_config_dict()):
        records, _ = read_available_model_call_records(
            CaptureStore(directory), rollout_id
        )
        calls += [
            c
            for c in records
            if c.model_ref is not None and c.model_ref.name in servers
        ]
    if not calls:
        return {}

    def total(attr: str) -> int:
        return sum(getattr(c, attr) or 0 for c in calls)

    input_tokens = sum(_input_tokens(c) or 0 for c in calls)
    output_tokens = total("tokens_out")
    return {
        "agent_llm_calls": len(calls),
        # Calls Gym saw but got no token counts for (an error, an unparsable
        # body). Nonzero means the totals below are a floor.
        "agent_llm_calls_without_usage": sum(
            1 for c in calls if _input_tokens(c) is None
        ),
        "agent_input_tokens": input_tokens,
        "agent_cached_input_tokens": total("cached_tokens"),
        "agent_output_tokens": output_tokens,
        "agent_total_tokens": input_tokens + output_tokens,
    }


def _input_tokens(call: Any) -> int | None:
    """One call's prompt tokens, from the provider's own total where it gave one.

    Gym's ``tokens_in`` double-counts the inference-api gateway's Claude
    responses. Their usage is OpenAI-shaped -- ``prompt_tokens`` already
    includes cache writes and reads -- but also carries Anthropic's top-level
    ``cache_creation_input_tokens``, which Gym takes as the native Anthropic
    shape and adds again: 2972 prompt tokens with 2968 written to cache are
    recorded as 5940. ``total_tokens`` comes straight from the provider, so
    total minus output is the prompt size whichever shape the usage had.
    """
    if call.tokens_total is not None and call.tokens_out is not None:
        return call.tokens_total - call.tokens_out
    return call.tokens_in


if __name__ == "__main__":
    # Emitted as shell for `eval`, so the API keys travel through the
    # environment rather than argv or a file.
    for key, value in launch_env().items():
        print(f"export {key}={shlex.quote(value)}")
