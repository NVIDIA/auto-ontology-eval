# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment-backed configuration for the LLM scorer.

Replaces the original ``auto_ontology.server.env`` dependency with a standalone
``python-dotenv`` loader so the tool can run on its own.

Resolution mirrors Auto Ontology's ``auto_ontology.utils.model_config``: each
``JUDGE_*`` field
falls back to ``DEFAULT_MODELS_*``, then the legacy shared names
(``NVIDIA_API_KEY`` / ``BASE_URL`` / ``MODEL_NAME``), then a built-in default
chosen by whether the effective API key is an ``sk-`` or ``nvapi-`` key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Built-in endpoint/model defaults, selected by the judge API key prefix.
# API keys are intentionally absent (they have no safe hardcoded default).
_DEFAULTS_BY_KEY_PREFIX: dict[str, dict[str, str]] = {
    # inference-api.nvidia.com (sk-... keys).
    "sk-": {
        "JUDGE_BASE_URL": "https://inference-api.nvidia.com/v1",
        "JUDGE_MODEL_NAME": "nvidia/nvidia/Nemotron-3-Nano-30B-A3B",
    },
    # integrate.api.nvidia.com / build.nvidia.com (nvapi-... keys).
    "nvapi-": {
        "JUDGE_BASE_URL": "https://integrate.api.nvidia.com/v1",
        "JUDGE_MODEL_NAME": "nvidia/nemotron-3-nano-30b-a3b",
    },
}

# Default set to use when the API key matches no known prefix (or is unset).
_FALLBACK_KEY_PREFIX = "sk-"

# Shared-default field → legacy (pre-triplet) env var name.
_LEGACY_ENV: dict[str, str] = {
    "API_KEY": "NVIDIA_API_KEY",
    "ENDPOINT": "BASE_URL",
    "MODEL": "MODEL_NAME",
}

# Judge-specific env var → shared DEFAULT_MODELS field name.
_JUDGE_FIELD: dict[str, str] = {
    "JUDGE_API_KEY": "API_KEY",
    "JUDGE_BASE_URL": "ENDPOINT",
    "JUDGE_MODEL_NAME": "MODEL",
}


@dataclass(frozen=True)
class Settings:
    """Resolved LLM connection settings."""

    model_name: str = ""
    base_url: str = ""
    api_key: str = ""


def _shared_default(field: str) -> str:
    """Shared default for *field*: ``DEFAULT_MODELS_<field>`` then the legacy name."""
    value = os.environ.get(f"DEFAULT_MODELS_{field}", "")
    if not value and field in _LEGACY_ENV:
        value = os.environ.get(_LEGACY_ENV[field], "")
    return value


def _resolve_api_key() -> str:
    """Effective judge API key (own env var, else the shared default)."""
    return os.environ.get("JUDGE_API_KEY", "") or _shared_default("API_KEY")


def _builtin_default(key: str) -> str:
    """Built-in default for a ``JUDGE_*`` field, chosen by the key prefix."""
    api_key = _resolve_api_key()
    for key_prefix, defaults in _DEFAULTS_BY_KEY_PREFIX.items():
        if api_key.startswith(key_prefix):
            return defaults.get(key, "")
    return _DEFAULTS_BY_KEY_PREFIX[_FALLBACK_KEY_PREFIX].get(key, "")


def _resolve(key: str) -> str:
    """Resolve one judge field.

    Order of precedence: the ``JUDGE_*`` env var, then the shared
    ``DEFAULT_MODELS_<field>`` (or legacy name), then a built-in default chosen
    by whether the judge API key is an ``sk-`` or ``nvapi-`` key.
    """
    shared_field = _JUDGE_FIELD[key]
    return (
        os.environ.get(key, "")
        or _shared_default(shared_field)
        or _builtin_default(key)
    )


def load_settings() -> Settings:
    """Load the judge's LLM settings from a ``.env`` file and the environment.

    The judge uses its own ``JUDGE_*`` variables so the scoring model can differ
    from the agent LLM. Each ``JUDGE_*`` var falls back to ``DEFAULT_MODELS_*``,
    then the legacy shared vars (``NVIDIA_API_KEY`` / ``BASE_URL`` /
    ``MODEL_NAME``), then a built-in default selected by API-key prefix.
    """
    load_dotenv()
    return Settings(
        model_name=_resolve("JUDGE_MODEL_NAME"),
        base_url=_resolve("JUDGE_BASE_URL"),
        api_key=_resolve_api_key(),
    )
