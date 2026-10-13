# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Auto Ontology arm's Gym model servers must follow Auto Ontology's config.

The arm's YAML builds its model servers from ``AUTO_ONTOLOGY_GYM_*`` variables,
so these tests pin the contract on both sides: the launcher exports exactly what
Auto Ontology itself would resolve, and the server repoints the endpoints the
YAML's servers stand in for.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import yaml

from ontology_sql_eval.gym import model_routing
from ontology_sql_eval.gym.model_routing import (
    GYM_MODEL_SERVERS,
    launch_env,
    route_through_gym,
)

_ROLES_AND_FIELDS = [
    f"{role}_{field}"
    for role in GYM_MODEL_SERVERS
    for field in ("ENDPOINT", "MODEL", "API_KEY")
]

_CONFIG = (
    Path(__file__).resolve().parent.parent
    / "resources_servers/auto_ontology_sql/configs/auto_ontology_sql.yaml"
)


@pytest.fixture
def clean_env(monkeypatch):
    for name in _ROLES_AND_FIELDS:
        monkeypatch.delenv(name, raising=False)
    for legacy in ("NVIDIA_API_KEY", "BASE_URL", "MODEL_NAME"):
        monkeypatch.delenv(legacy, raising=False)
    for field in ("ENDPOINT", "MODEL", "API_KEY"):
        monkeypatch.delenv(f"DEFAULT_MODELS_{field}", raising=False)
    return monkeypatch


@pytest.fixture
def real_auto_ontology(monkeypatch):
    """Hide sibling tests' ``auto_ontology`` stubs for the duration of a test.

    ``tests/bird_interact/test_server.py`` replaces ``auto_ontology`` and
    ``auto_ontology.utils`` in ``sys.modules`` at collection time, so without
    this the import below would resolve against those stubs.
    """
    for name in [m for m in sys.modules if m.split(".")[0] == "auto_ontology"]:
        monkeypatch.delitem(sys.modules, name)
    pytest.importorskip("auto_ontology.utils.model_config")


def test_launch_env_matches_auto_ontology_resolution(clean_env, real_auto_ontology):
    from auto_ontology.utils.model_config import resolve

    clean_env.setenv("DEFAULT_MODELS_API_KEY", "sk-test")
    clean_env.setenv("NON_REASONING_MODEL", "aws/example/override")

    env = launch_env()

    for role in GYM_MODEL_SERVERS:
        for field in ("ENDPOINT", "MODEL", "API_KEY"):
            assert env[f"AUTO_ONTOLOGY_GYM_{role}_{field}"] == resolve(role, field)
    # An explicit override wins, and the other role keeps Auto Ontology's default.
    assert env["AUTO_ONTOLOGY_GYM_NON_REASONING_MODEL"] == "aws/example/override"
    assert env["AUTO_ONTOLOGY_GYM_REASONING_MODEL"] != "aws/example/override"


def test_route_through_gym_points_each_role_at_its_server(clean_env):
    urls = {
        "auto_ontology_reasoning_model": "http://127.0.0.1:1111",
        "auto_ontology_non_reasoning_model": "http://127.0.0.1:2222",
    }
    clean_env.setattr("nemo_gym.server_utils.get_server_url", lambda name: urls[name])
    clean_env.setenv("REASONING_ENDPOINT", "https://set-in-dotenv.example/v1")

    route_through_gym()

    assert os.environ["REASONING_ENDPOINT"] == "http://127.0.0.1:1111/v1"
    assert os.environ["NON_REASONING_ENDPOINT"] == "http://127.0.0.1:2222/v1"


def test_config_declares_a_server_per_role_from_launch_env():
    config = yaml.safe_load(_CONFIG.read_text())
    for role, server in model_routing.GYM_MODEL_SERVERS.items():
        provider = config[server]["responses_api_models"]["inference_provider"]
        assert provider["base_url"] == f"${{oc.env:AUTO_ONTOLOGY_GYM_{role}_ENDPOINT}}"
        assert provider["model"] == f"${{oc.env:AUTO_ONTOLOGY_GYM_{role}_MODEL}}"
        assert provider["api_key"] == f"${{oc.env:AUTO_ONTOLOGY_GYM_{role}_API_KEY}}"


@pytest.fixture
def routed(monkeypatch):
    """Two routed Gym servers, as route_through_gym would leave them."""
    monkeypatch.setattr(
        model_routing,
        "_routed_roots",
        ("http://127.0.0.1:1111", "http://127.0.0.1:2222"),
    )


def test_rollout_prefix_goes_only_on_routed_calls(routed):
    tag = model_routing.with_rollout_prefix
    assert (
        tag("http://127.0.0.1:1111/v1/chat/completions", "3-0")
        == "http://127.0.0.1:1111/ng-rollout/3-0/v1/chat/completions"
    )
    # Embedding endpoints, already-tagged calls and out-of-rollout calls pass.
    assert tag("https://embed.example/v1/embeddings", "3-0") == (
        "https://embed.example/v1/embeddings"
    )
    already = "http://127.0.0.1:2222/ng-rollout/3-0/v1/chat/completions"
    assert tag(already, "3-0") == already
    assert tag("http://127.0.0.1:2222/v1/chat/completions", None) == (
        "http://127.0.0.1:2222/v1/chat/completions"
    )
    # A port that merely starts with a routed one is a different server.
    assert tag("http://127.0.0.1:11110/v1/x", "3-0") == "http://127.0.0.1:11110/v1/x"


def test_both_http_stacks_carry_the_rollout(routed):
    """ChatOpenAI sends through httpx and ChatNVIDIA through requests."""
    import httpx
    import requests
    from nemo_gym.rollout_correlation import rollout_context

    model_routing.tag_rollout_requests()
    model_routing.tag_rollout_requests()  # idempotent: no double prefix

    seen = []
    transport = httpx.MockTransport(
        lambda request: (seen.append(str(request.url)), httpx.Response(200))[1]
    )

    class Adapter(requests.adapters.BaseAdapter):
        def send(self, request, **kwargs):
            seen.append(request.url)
            response = requests.Response()
            response.status_code = 200
            return response

        def close(self):
            pass

    session = requests.Session()
    session.mount("http://", Adapter())
    with rollout_context("7-1"):
        httpx.Client(transport=transport).post("http://127.0.0.1:1111/v1/chat")
        session.post("http://127.0.0.1:2222/v1/chat")

    assert seen == [
        "http://127.0.0.1:1111/ng-rollout/7-1/v1/chat",
        "http://127.0.0.1:2222/ng-rollout/7-1/v1/chat",
    ]


def test_captured_usage_counts_only_auto_ontology_calls(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import nemo_gym.base_responses_api_model as model_api
    import nemo_gym.global_config as global_config

    def call(server, tokens_in, tokens_out, cached=0, tokens_total="sum"):
        if tokens_total == "sum":
            tokens_total = None if tokens_in is None else tokens_in + tokens_out
        return SimpleNamespace(
            model_ref=SimpleNamespace(name=server),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tokens_total=tokens_total,
            cached_tokens=cached,
        )

    calls = [
        call("policy_model", 1000, 50),  # ignored by this arm
        call("auto_ontology_reasoning_model", 400, 100, cached=300),
        # As Gym records the gateway's Claude usage: prompt_tokens=150 with
        # 140 cache-written, which Gym adds again into tokens_in. The
        # provider's total_tokens (150 + 20) is the one to trust.
        call("auto_ontology_non_reasoning_model", 290, 20, tokens_total=170),
        call("auto_ontology_non_reasoning_model", 50, 0, tokens_total=None),
        call("auto_ontology_non_reasoning_model", None, None),  # e.g. a 429
    ]
    monkeypatch.setattr(
        global_config,
        "get_global_config_dict",
        lambda: {"observability_enabled": True, "model_call_capture_dir": tmp_path},
    )
    monkeypatch.setattr(
        model_api, "read_available_model_call_records", lambda store, rid: (calls, 0)
    )

    assert model_routing.captured_usage("3-0") == {
        "agent_llm_calls": 4,
        "agent_llm_calls_without_usage": 1,
        "agent_input_tokens": 600,
        "agent_cached_input_tokens": 300,
        "agent_output_tokens": 120,
        "agent_total_tokens": 720,
    }
    assert model_routing.captured_usage(None) == {}
