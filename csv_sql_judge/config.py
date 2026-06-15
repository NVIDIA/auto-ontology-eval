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
    """Load settings from a ``.env`` file (if present) and the environment."""
    load_dotenv()
    return Settings(
        model_name=os.environ.get("MODEL_NAME", _DEFAULT_MODEL),
        base_url=os.environ.get("BASE_URL", _DEFAULT_BASE_URL),
        api_key=os.environ.get("NVIDIA_API_KEY", ""),
    )
