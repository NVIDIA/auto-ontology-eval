# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment-backed configuration for the LLM scorer.

Replaces the original ``gsf.server.env`` dependency with a standalone
``python-dotenv`` loader so the tool can run on its own.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

_DEFAULT_MODEL = "nvidia/nvidia/Nemotron-3-Nano-30B-A3B"
_DEFAULT_BASE_URL = "https://inference-api.nvidia.com/v1"


@dataclass(frozen=True)
class Settings:
    """Resolved LLM connection settings."""

    model_name: str = _DEFAULT_MODEL
    base_url: str = _DEFAULT_BASE_URL
    api_key: str = ""


def load_settings() -> Settings:
    """Load the judge's LLM settings from a ``.env`` file and the environment.

    The judge uses its own ``JUDGE_*`` variables so the scoring model can differ
    from the agent LLM (``MODEL_NAME`` / ``BASE_URL`` / ``NVIDIA_API_KEY``) used
    during retrieval eval. Each ``JUDGE_*`` var falls back to its shared
    counterpart when unset.
    """
    load_dotenv()
    return Settings(
        model_name=os.environ.get(
            "JUDGE_MODEL_NAME", os.environ.get("MODEL_NAME", _DEFAULT_MODEL)
        ),
        base_url=os.environ.get(
            "JUDGE_BASE_URL", os.environ.get("BASE_URL", _DEFAULT_BASE_URL)
        ),
        api_key=os.environ.get("JUDGE_API_KEY", os.environ.get("NVIDIA_API_KEY", "")),
    )
