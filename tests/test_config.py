# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for judge LLM settings resolution."""

from __future__ import annotations

import pytest

from evaluation.judge import config
from evaluation.judge.config import load_settings

_ALL_VARS = [
    "JUDGE_MODEL_NAME",
    "JUDGE_BASE_URL",
    "JUDGE_API_KEY",
    "MODEL_NAME",
    "BASE_URL",
    "NVIDIA_API_KEY",
]


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise .env loading and clear all relevant vars for each test."""
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: False)
    for var in _ALL_VARS:
        monkeypatch.delenv(var, raising=False)


def test_defaults_when_nothing_set() -> None:
    settings = load_settings()
    assert settings.model_name == config._DEFAULT_MODEL
    assert settings.base_url == config._DEFAULT_BASE_URL
    assert settings.api_key == ""


def test_judge_vars_take_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JUDGE_MODEL_NAME", "judge-model")
    monkeypatch.setenv("JUDGE_BASE_URL", "https://judge.example/v1")
    monkeypatch.setenv("JUDGE_API_KEY", "judge-key")
    # Shared vars set too, but JUDGE_* must win.
    monkeypatch.setenv("MODEL_NAME", "agent-model")
    monkeypatch.setenv("BASE_URL", "https://agent.example/v1")
    monkeypatch.setenv("NVIDIA_API_KEY", "agent-key")

    settings = load_settings()

    assert settings.model_name == "judge-model"
    assert settings.base_url == "https://judge.example/v1"
    assert settings.api_key == "judge-key"


def test_falls_back_to_shared_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_NAME", "agent-model")
    monkeypatch.setenv("BASE_URL", "https://agent.example/v1")
    monkeypatch.setenv("NVIDIA_API_KEY", "agent-key")

    settings = load_settings()

    assert settings.model_name == "agent-model"
    assert settings.base_url == "https://agent.example/v1"
    assert settings.api_key == "agent-key"
