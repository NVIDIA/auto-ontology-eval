# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""run_gym_arm.sh rebuilds a server venv whose Auto Ontology commit is stale.

The script reuses the server venvs Gym built unless a checked package differs
from .venv. Auto Ontology is installed from a pinned Git commit and keeps the
same version number across commits, so a repin changed .venv but not the
server venv, and the arm silently ran the old commit. These run the script's
own ``dist_versions`` snippet against a fake installed package.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from ontology_sql_eval.paths import REPO_ROOT

SCRIPT = (REPO_ROOT / "scripts" / "run_gym_arm.sh").read_text()


def _dist_versions_snippet() -> str:
    match = re.search(r"dist_versions\(\) \{.*?<<'PY'\n(.*?)\nPY\n", SCRIPT, re.S)
    assert match, "dist_versions heredoc not found in run_gym_arm.sh"
    return match.group(1)


def _install_fake(site: Path, direct_url: dict | None) -> None:
    info = site / "auto_ontology_server-1.1.0.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: auto-ontology-server\nVersion: 1.1.0\n"
    )
    if direct_url is not None:
        (info / "direct_url.json").write_text(json.dumps(direct_url))


def _run(site: Path, *names: str) -> str:
    env = {**os.environ, "PYTHONPATH": str(site)}
    return subprocess.run(
        [sys.executable, "-", *names],
        input=_dist_versions_snippet(),
        capture_output=True,
        text=True,
        check=True,
        env=env,
    ).stdout


def test_auto_ontology_is_a_checked_package() -> None:
    match = re.search(r"^CHECKED_DISTS=\((.*)\)$", SCRIPT, re.M)
    assert match and "auto-ontology-server" in match.group(1).split()


def test_a_git_install_reports_its_commit(tmp_path: Path) -> None:
    _install_fake(
        tmp_path,
        {
            "url": "https://github.com/NVIDIA/auto-ontology.git",
            "vcs_info": {"vcs": "git", "commit_id": "abc123"},
        },
    )

    assert (
        _run(tmp_path, "auto-ontology-server") == "auto-ontology-server 1.1.0@abc123\n"
    )


def test_a_plain_install_reports_its_version(tmp_path: Path) -> None:
    _install_fake(tmp_path, None)

    assert _run(tmp_path, "auto-ontology-server") == "auto-ontology-server 1.1.0\n"


def test_a_missing_package_prints_nothing(tmp_path: Path) -> None:
    assert _run(tmp_path, "not-installed-anywhere") == ""
