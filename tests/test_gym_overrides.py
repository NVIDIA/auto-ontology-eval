# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Gym server's overrides must stay in step with the parent environment.

``resources_servers/auto_ontology_sql/overrides.txt`` restates pins that also
live in ``uv.lock`` and the root ``pyproject.toml``. The duplication is forced:
Gym builds that server's venv under ``UV_NO_CONFIG=1``, which is what stops the
parent's ``override-dependencies`` leaking in and clobbering the server's own
``openai`` pin -- but it switches off the parent's other overrides at the same
time, so they have to be repeated.

Duplication that nothing checks is duplication that drifts, and both of the
bugs these pins exist to prevent were silent: a Ray mismatch stops the cluster
with an error that names neither venv, and a newer nemo-retriever leaves the
ontology arm retrieving nothing at all while still reporting a score. These
tests turn "bump both together" from a comment into a rule.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OVERRIDES = ROOT / "resources_servers" / "auto_ontology_sql" / "overrides.txt"


def _overrides() -> dict[str, str]:
    """Requirement lines from overrides.txt, keyed by package name."""
    entries: dict[str, str] = {}
    for raw in OVERRIDES.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # "nemo-retriever[tabular] @ git+..." and "ray==2.59.0" alike: the name
        # is everything before the first extra, comparison or " @ ".
        name = re.split(r"[\[<>=!~ ]", line, maxsplit=1)[0]
        entries[name] = line
    return entries


def _locked() -> dict[str, dict]:
    """``[[package]]`` entries from uv.lock, keyed by name."""
    with (ROOT / "uv.lock").open("rb") as fh:
        lock = tomllib.load(fh)
    return {pkg["name"]: pkg for pkg in lock["package"]}


def _pyproject_overrides() -> dict[str, str]:
    with (ROOT / "pyproject.toml").open("rb") as fh:
        pyproject = tomllib.load(fh)
    listed = pyproject["tool"]["uv"]["override-dependencies"]
    return {re.split(r"[\[<>=!~ ]", item, maxsplit=1)[0]: item for item in listed}


def test_nemo_retriever_commit_matches_the_lockfile() -> None:
    """The server must embed with the same revision the legacy harness uses.

    nemo-retriever is an *unpinned* Git dependency, so the parent venv takes the
    locked commit while this venv -- built with ``uv pip install -e ../..``,
    which ignores the lock -- would otherwise take whatever HEAD is. A newer
    HEAD rejects the gateway-style embedding model name Auto Ontology passes,
    and ``get_relevant_tables`` turns that into an empty table list rather than
    an error.
    """
    override = _overrides()["nemo-retriever"]
    pinned = re.search(r"@([0-9a-f]{40})#", override)
    assert pinned, f"nemo-retriever override must pin a full commit sha: {override}"

    source = _locked()["nemo-retriever"]["source"]["git"]
    locked = re.search(r"#([0-9a-f]{40})$", source)
    assert locked, f"could not read the locked nemo-retriever commit from {source!r}"

    assert pinned.group(1) == locked.group(1), (
        "overrides.txt and uv.lock disagree on the nemo-retriever commit. "
        "Bump both together, and re-run the ontology arm: a mismatch here is "
        "not visible in the rollouts, only in the node trace."
    )


def test_ray_is_pinned_to_the_locked_version() -> None:
    """Gym's cluster needs the driver and the server's workers on one version.

    A lower bound is not enough: the parent's version is frozen by uv.lock while
    this venv resolves freely, so ``ray>=`` agrees only until the next release.
    """
    override = _overrides()["ray"]
    assert override.startswith("ray=="), (
        f"ray must be pinned exactly so parent and server agree, got {override!r}"
    )
    assert override == f"ray=={_locked()['ray']['version']}", (
        f"{override!r} does not match uv.lock's ray "
        f"{_locked()['ray']['version']!r}; bump both together."
    )


@pytest.mark.parametrize("package", ["nltk", "pillow", "urllib3"])
def test_cve_overrides_match_pyproject(package: str) -> None:
    """The restated CVE floors must not fall behind the parent's.

    These are the patched-version overrides from the root pyproject, repeated
    here because UV_NO_CONFIG=1 switches the originals off. Raising one without
    the other would leave the server on the vulnerable version.
    """
    assert _overrides()[package] == _pyproject_overrides()[package], (
        f"{package} differs between overrides.txt and pyproject.toml's "
        "override-dependencies; they must be raised together."
    )
