# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The root ``[tool.uv]`` overrides also reach the Gym server venvs.

Gym builds each server venv with ``uv pip install`` run from inside this repo,
and uv walks up the tree to the root ``pyproject.toml``, so its
``override-dependencies`` apply there as well as in the parent venv. That is how
the server venvs get the CVE floors, from one list. It also means an override
can silently defeat a pin Gym itself passes on the command line. Both bugs this
guards against were silent: a Ray mismatch stops the cluster with an error that
names neither venv, and an openai past Gym's pin rejects every response from
the inference-api gateway.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _override_names() -> set[str]:
    with (ROOT / "pyproject.toml").open("rb") as fh:
        pyproject = tomllib.load(fh)
    listed = pyproject["tool"]["uv"]["override-dependencies"]
    return {re.split(r"[\[<>=!~ ;]", item, maxsplit=1)[0].lower() for item in listed}


@pytest.mark.parametrize(
    ("package", "why"),
    [
        (
            "ray",
            "Gym pins ray[default]==<head version> into every server venv; an "
            "override replaces that pin and the server fails to join the cluster",
        ),
        (
            "openai",
            "Gym pins the parent's openai into every server venv, and Auto "
            "Ontology pins openai==2.44.0 to match nemo-gym; an override replaces "
            "both, and openai 2.45+ rejects the inference-api gateway's responses",
        ),
    ],
)
def test_no_override_defeats_a_gym_pin(package: str, why: str) -> None:
    assert package not in _override_names(), f"do not override {package}: {why}"


def test_no_per_server_overrides_file() -> None:
    """Overrides live in one place.

    Gym also honours an ``overrides.txt`` beside a server's requirements.txt.
    A second list drifts from the first, and the root list already applies.
    """
    stray = sorted(
        str(path.relative_to(ROOT))
        for path in (ROOT / "resources_servers").glob("*/overrides.txt")
    )
    assert not stray, f"move these into pyproject.toml's [tool.uv]: {stray}"


def test_server_requirements_pin_the_parents_nemo_gym() -> None:
    """Each server venv must run the Gym version the head runs.

    The servers' requirements.txt must name nemo-gym explicitly (see
    resources_servers/README.md), so the version lives in three places.
    """
    with (ROOT / "pyproject.toml").open("rb") as fh:
        pyproject = tomllib.load(fh)
    parent = [
        dep
        for dep in pyproject["project"]["dependencies"]
        if dep.startswith("nemo-gym")
    ]
    assert len(parent) == 1 and parent[0].startswith("nemo-gym=="), (
        f"pyproject.toml must pin nemo-gym exactly, got {parent!r}"
    )
    for requirements in sorted((ROOT / "resources_servers").glob("*/requirements.txt")):
        pins = [
            line.strip()
            for line in requirements.read_text().splitlines()
            if line.strip().startswith("nemo-gym")
        ]
        assert pins == parent, (
            f"{requirements.relative_to(ROOT)} pins {pins!r}, pyproject.toml "
            f"{parent!r}; bump them together."
        )
