# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seed BIRD-Interact lite from the official GitHub repository.

Sparse-clones https://github.com/bird-bench/BIRD-Interact into
third_party/BIRD-Interact/ and prepares datasets/bird_interact/ for use
with the GSF adapter and run_bird_interact_cinteract.py.

Layout after running this script::

    third_party/BIRD-Interact/
        BIRD-Interact-ADK/      ← orchestrator, user_simulator, db_environment
        bird-interact-lite/     ← public task JSONL
        env/                    ← Docker Compose for PostgreSQL task DBs

    datasets/bird_interact/
        bird_interact_data.jsonl            ← public tasks (no GT)
        bird_interact_data_with_gt.jsonl    ← combined file used by the runner
                                              (= GT copy if GT present, else public)
        manifest.json
        gt/
            README.md           ← instructions for obtaining GT files
            bird_interact_data_with_gt.jsonl  ← place emailed GT file here

Usage::

    uv run python scripts/seed_bird_interact.py
    uv run python scripts/seed_bird_interact.py --force

GT SQLs and test cases are NOT in the public repo. After running this script,
follow the instructions in datasets/bird_interact/gt/README.md (or the output
printed here) to obtain them by email and re-run.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

UPSTREAM_REPO = "https://github.com/bird-bench/BIRD-Interact.git"
DEFAULT_UPSTREAM_REF = "main"

# Cone-mode sparse-checkout paths: ADK code, lite data, Docker env.
# Database files live in PostgreSQL Docker containers (see env/), not in git.
_SPARSE_PATHS = (
    "BIRD-Interact-ADK",
    "bird-interact-lite",
    "env",
)

# Public JSONL names tried in order; first match wins.
_PUBLIC_JSONL_CANDIDATES = (
    "bird_interact_data.jsonl",
    "bird_interact_data_with_gt.jsonl",
)

# Filename the emailed GT package is expected to provide (placed in gt/).
GT_JSONL_NAME = "bird_interact_data_with_gt.jsonl"

_GT_README = """\
# BIRD-Interact GT & Test Cases

Ground-truth SQLs and test cases are NOT included in the public repository.

## How to obtain them

Send an email to:

    bird.bench25@gmail.com

Subject tag (copy exactly):

    [bird-interact-lite GT&Test Cases]

The files will be sent automatically within 30 minutes.

## Where to place them

Drop the received file directly into THIS directory
(datasets/bird_interact/gt/).  The seeder expects:

    datasets/bird_interact/gt/bird_interact_data_with_gt.jsonl

Once the file is present, re-run the seeder to regenerate the combined data
file and manifest:

    uv run python scripts/seed_bird_interact.py

The seeder copies it to:

    datasets/bird_interact/bird_interact_data_with_gt.jsonl

That path is what scripts/run_bird_interact_cinteract.py reads by default.

## Without GT

Until GT files arrive the seeder copies the public JSONL in their place so
the adapter and orchestrator can be exercised end-to-end — scoring will be
incomplete but the plumbing can be verified.
"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _datasets_dir() -> Path:
    return _repo_root() / "datasets" / "bird_interact"


def _gt_dir() -> Path:
    return _datasets_dir() / "gt"


def _upstream_root() -> Path:
    return _repo_root() / "third_party" / "BIRD-Interact"


def _lite_root() -> Path:
    return _upstream_root() / "bird-interact-lite"


def _adk_root() -> Path:
    return _upstream_root() / "BIRD-Interact-ADK"


def _env_root() -> Path:
    return _upstream_root() / "env"


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _current_ref(repo_dir: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _bootstrap_upstream(ref: str = DEFAULT_UPSTREAM_REF) -> str:
    """Clone or update the upstream BIRD-Interact repo (sparse checkout)."""
    upstream = _upstream_root()
    upstream.parent.mkdir(parents=True, exist_ok=True)

    if upstream.exists():
        logger.info("Updating existing checkout at %s", upstream)
        _run(["git", "sparse-checkout", "add", *_SPARSE_PATHS], cwd=upstream)
        _run(["git", "fetch", "--depth", "1", "origin", ref], cwd=upstream)
        _run(["git", "checkout", ref], cwd=upstream)
        _run(["git", "pull", "--ff-only", "origin", ref], cwd=upstream)
    else:
        logger.info("Cloning %s (sparse) into %s", UPSTREAM_REPO, upstream)
        _run([
            "git", "clone",
            "--depth", "1",
            "--filter=blob:none",
            "--sparse",
            "--branch", ref,
            UPSTREAM_REPO,
            str(upstream),
        ])
        _run(["git", "sparse-checkout", "set", *_SPARSE_PATHS], cwd=upstream)

    commit = _current_ref(upstream)
    logger.info("Upstream BIRD-Interact at commit %s", commit)

    missing = [p for p in (_lite_root(),) if not p.exists()]
    if missing:
        paths = "\n".join(f"  - {p}" for p in missing)
        raise SystemExit(f"Upstream checkout is missing required paths:\n{paths}")

    return commit


def _ensure_gt_readme() -> None:
    gt = _gt_dir()
    gt.mkdir(parents=True, exist_ok=True)
    readme = gt / "README.md"
    readme.write_text(_GT_README, encoding="utf-8")
    logger.info("Wrote GT placeholder: %s", readme)


def _find_public_jsonl() -> Path | None:
    lite = _lite_root()
    for name in _PUBLIC_JSONL_CANDIDATES:
        candidate = lite / name
        if candidate.exists():
            return candidate
    return None


def _detect_gt() -> Path | None:
    candidate = _gt_dir() / GT_JSONL_NAME
    return candidate if candidate.exists() else None


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}") from exc
    return rows


def _build_datasets(*, upstream_commit: str | None = None, force: bool = False) -> dict:
    dest = _datasets_dir()
    dest.mkdir(parents=True, exist_ok=True)
    _ensure_gt_readme()

    # --- public JSONL ---
    public_src = _find_public_jsonl()
    if public_src is None:
        raise SystemExit(
            f"No public JSONL found under {_lite_root()}. "
            "Expected one of: " + ", ".join(_PUBLIC_JSONL_CANDIDATES)
        )

    public_dest = dest / "bird_interact_data.jsonl"
    if not public_dest.exists() or force:
        shutil.copy2(public_src, public_dest)
        logger.info("Copied public JSONL: %s -> %s", public_src.name, public_dest)
    else:
        logger.info("Using existing public JSONL: %s", public_dest)

    rows = _load_jsonl(public_dest)
    task_count = len(rows)
    db_names: list[str] = sorted({
        str(row.get("db_name") or row.get("db_id") or row.get("database") or "")
        for row in rows
        if row.get("db_name") or row.get("db_id") or row.get("database")
    })

    # --- combined JSONL (GT copy if available, else public as development fallback) ---
    gt_src = _detect_gt()
    gt_available = gt_src is not None
    combined_dest = dest / GT_JSONL_NAME

    if gt_available:
        shutil.copy2(gt_src, combined_dest)
        logger.info("GT detected — copied to %s", combined_dest)
    else:
        shutil.copy2(public_dest, combined_dest)
        logger.warning(
            "GT file not found at %s. "
            "Copied public JSONL as development fallback; scoring will be incomplete. "
            "See datasets/bird_interact/gt/README.md to obtain GT files.",
            _gt_dir() / GT_JSONL_NAME,
        )

    # --- manifest ---
    manifest: dict = {
        "source": {
            "repository": UPSTREAM_REPO,
            "ref": DEFAULT_UPSTREAM_REF,
            "upstream_root": str(_upstream_root()),
            "upstream_commit": upstream_commit,
        },
        "scope": "bird-interact-lite",
        "task_count": task_count,
        "database_count": len(db_names),
        "databases": db_names,
        "gt_available": gt_available,
        "paths": {
            "public_jsonl": str(public_dest.relative_to(_repo_root())),
            "combined_jsonl": str(combined_dest.relative_to(_repo_root())),
            "gt_dir": str(_gt_dir().relative_to(_repo_root())),
            "adk_dir": str(_adk_root().relative_to(_repo_root())),
            "env_dir": str(_env_root().relative_to(_repo_root())),
        },
    }
    manifest_path = dest / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Manifest: %s (%d tasks, %d databases, gt_available=%s)",
        manifest_path, task_count, len(db_names), gt_available,
    )
    return manifest


def _print_summary(manifest: dict) -> None:
    combined = _repo_root() / manifest["paths"]["combined_jsonl"]
    env_dir = _repo_root() / manifest["paths"]["env_dir"]

    logger.info("=" * 60)
    logger.info("BIRD-Interact lite seed complete.")
    logger.info("  Tasks      : %d", manifest["task_count"])
    logger.info("  Databases  : %d", manifest["database_count"])
    for db in manifest["databases"]:
        logger.info("    - %s", db)
    logger.info("  GT present : %s", manifest["gt_available"])
    logger.info("=" * 60)

    print()
    if not manifest["gt_available"]:
        print("=" * 60)
        print("ACTION REQUIRED: GT files not found. Scoring will be incomplete.")
        print()
        print("  1. Email bird.bench25@gmail.com")
        print("     Subject: [bird-interact-lite GT&Test Cases]")
        print()
        print("  2. Place the received file at:")
        print(f"       {_gt_dir() / GT_JSONL_NAME}")
        print()
        print("  3. Re-run: uv run python scripts/seed_bird_interact.py")
        print("=" * 60)
        print()

    print("Start the BIRD-Interact PostgreSQL task databases (Docker, one-time):")
    print()
    print(f"  cd {env_dir}")
    print("  docker compose up -d bird_interact_postgresql")
    print()
    print("Start pg_wrappers shims (needed for :6002 db_environment scoring):")
    print()
    print("  uv run python scripts/setup_pg_wrappers.py")
    print()
    print("Then run the c-Interact evaluation:")
    print()
    print(f"  python scripts/run_bird_interact_cinteract.py --data {combined}")
    print()


def seed_bird_interact(
    *,
    ref: str = DEFAULT_UPSTREAM_REF,
    force: bool = False,
    skip_build: bool = False,
) -> dict:
    """Clone BIRD-Interact and prepare datasets/bird_interact/. Returns the manifest."""
    logger.info("=" * 60)
    logger.info("Seeding BIRD-Interact lite")
    logger.info("=" * 60)

    commit = _bootstrap_upstream(ref=ref)
    logger.info("Upstream BIRD-Interact commit: %s", commit)

    if skip_build:
        logger.info("Skipping dataset build (--skip-build).")
        return {}

    manifest = _build_datasets(upstream_commit=commit, force=force)
    _print_summary(manifest)
    return manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clone BIRD-Interact from GitHub and prepare datasets/bird_interact/ "
            "with the public task JSONL, GT placeholder, and manifest."
        ),
        epilog=(
            "GT SQLs and test cases require emailing bird.bench25@gmail.com "
            "with subject [bird-interact-lite GT&Test Cases]. "
            "Place received files in datasets/bird_interact/gt/ and re-run."
        ),
    )
    parser.add_argument(
        "--ref",
        default=DEFAULT_UPSTREAM_REF,
        help="Upstream git ref to check out (default: main)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-copy JSONL files even if already present in datasets/bird_interact/",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Only update the git clone; skip JSONL copy and manifest generation",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    seed_bird_interact(
        ref=args.ref,
        force=args.force,
        skip_build=args.skip_build,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
