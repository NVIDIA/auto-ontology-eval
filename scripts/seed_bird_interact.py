# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seed BIRD-Interact (lite or full) from the official sources.

Sparse-clones https://github.com/bird-bench/BIRD-Interact (ADK code + Docker
env only — the upstream repo no longer ships task data as git paths) into
third_party/BIRD-Interact/, downloads the requested variant's task data from
its HuggingFace dataset repo (birdsql/bird-interact-<variant>), and prepares
datasets/bird_interact[_full]/ for use with the Auto Ontology adapter and
run_bird_interact_cinteract.py.

Layout after running this script (example: --dataset lite)::

    third_party/BIRD-Interact/
        BIRD-Interact-ADK/      ← orchestrator, user_simulator, db_environment
        env/                    ← Docker Compose for PostgreSQL task DBs
        bird-interact-lite/     ← HF dataset snapshot (task JSONL, per-DB KB)

    datasets/bird_interact/
        bird_interact_data.jsonl            ← public tasks (no GT)
        bird_interact_data_with_gt.jsonl    ← combined file used by the runner
                                              (= GT copy if GT present, else public)
        manifest.json
        gt/
            README.md           ← instructions for obtaining GT files
            bird_interact_data_with_gt.jsonl  ← place emailed GT file here

--dataset full uses the same layout under bird-interact-full/ and
datasets/bird_interact_full/.

Usage::

    uv run python scripts/seed_bird_interact.py --dataset lite
    uv run python scripts/seed_bird_interact.py --dataset full
    uv run python scripts/seed_bird_interact.py --dataset lite --force

--dataset is required — there is no default variant.

GT SQLs and test cases are NOT in the public repo/dataset. After running this
script, follow the instructions in datasets/bird_interact[_full]/gt/README.md
(or the output printed here) to obtain them by email and re-run.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

UPSTREAM_REPO = "https://github.com/bird-bench/BIRD-Interact.git"
DEFAULT_UPSTREAM_REF = "main"

# Task data (JSONL + per-DB knowledge base) now lives in separate HuggingFace
# dataset repos per variant, not as git paths in the upstream repo.
HF_DATASET_REPO_TEMPLATE = "birdsql/bird-interact-{variant}"
DATASET_VARIANTS = ("lite", "full")

# Cone-mode sparse-checkout paths: ADK code + Docker env only.
# Database files live in PostgreSQL Docker containers (see env/), not in git.
_SPARSE_PATHS = (
    "BIRD-Interact-ADK",
    "env",
)

# Public JSONL names tried in order; first match wins.
_PUBLIC_JSONL_CANDIDATES = (
    "bird_interact_data.jsonl",
    "bird_interact_data_with_gt.jsonl",
)

# Filename the emailed GT package is expected to provide (placed in gt/).
GT_JSONL_NAME = "bird_interact_data_with_gt.jsonl"


def _gt_readme(variant: str) -> str:
    return f"""\
# BIRD-Interact GT & Test Cases

Ground-truth SQLs and test cases are NOT included in the public repository
or dataset.

## How to obtain them

Send an email to:

    bird.bench25@gmail.com

Subject tag (copy exactly):

    [bird-interact-{variant} GT&Test Cases]

The files will be sent automatically within 30 minutes.

Alternatively, get the file from the shared Google Drive.

## Where to place them

Drop the received file directly into THIS directory
(datasets/bird_interact{'_full' if variant == 'full' else ''}/gt/).  The seeder expects:

    datasets/bird_interact{'_full' if variant == 'full' else ''}/gt/bird_interact_data_with_gt.jsonl

Once the file is present, re-run the seeder to regenerate the combined data
file and manifest:

    uv run python scripts/seed_bird_interact.py --dataset {variant}

The seeder copies it to:

    datasets/bird_interact{'_full' if variant == 'full' else ''}/bird_interact_data_with_gt.jsonl

That path is what scripts/run_bird_interact_cinteract.py reads by default.

## Without GT

Until GT files arrive the seeder copies the public JSONL in their place so
the adapter and orchestrator can be exercised end-to-end — scoring will be
incomplete but the plumbing can be verified.
"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _datasets_dir(variant: str) -> Path:
    suffix = "_full" if variant == "full" else ""
    return _repo_root() / "datasets" / f"bird_interact{suffix}"


def _gt_dir(variant: str) -> Path:
    return _datasets_dir(variant) / "gt"


def _upstream_root() -> Path:
    return _repo_root() / "third_party" / "BIRD-Interact"


def _data_root(variant: str) -> Path:
    return _upstream_root() / f"bird-interact-{variant}"


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
    """Clone or update the upstream BIRD-Interact repo (sparse checkout: ADK + env only)."""
    upstream = _upstream_root()
    upstream.parent.mkdir(parents=True, exist_ok=True)

    if upstream.exists():
        if not (upstream / ".git").exists():
            # A directory here that isn't a git checkout (e.g. manually copied
            # in) is unsafe to touch: git would walk up to the nearest parent
            # .git (this repo's own) and run sparse-checkout/fetch/pull
            # against THAT instead, silently corrupting its state.
            raise SystemExit(
                f"{upstream} exists but is not a git checkout (no .git/ found). "
                "Remove or rename it, then re-run this script to get a fresh "
                "sparse clone."
            )
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

    missing = [p for p in (_adk_root(), _env_root()) if not p.exists()]
    if missing:
        paths = "\n".join(f"  - {p}" for p in missing)
        raise SystemExit(f"Upstream checkout is missing required paths:\n{paths}")

    return commit


def _bootstrap_dataset(variant: str, *, force: bool = False) -> None:
    """Download the variant's task data from its HuggingFace dataset repo."""
    repo_id = HF_DATASET_REPO_TEMPLATE.format(variant=variant)
    dest = _data_root(variant)
    dest.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s (HF dataset) into %s", repo_id, dest)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=dest,
        force_download=force,
    )

    if not any(dest.glob("*")):
        raise SystemExit(f"HF dataset download produced no files at {dest}")


def _ensure_gt_readme(variant: str) -> None:
    gt = _gt_dir(variant)
    gt.mkdir(parents=True, exist_ok=True)
    readme = gt / "README.md"
    readme.write_text(_gt_readme(variant), encoding="utf-8")
    logger.info("Wrote GT placeholder: %s", readme)


def _find_public_jsonl(variant: str) -> Path | None:
    root = _data_root(variant)
    for name in _PUBLIC_JSONL_CANDIDATES:
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def _detect_gt(variant: str) -> Path | None:
    candidate = _gt_dir(variant) / GT_JSONL_NAME
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


def _combine_public_with_gt(public_path: Path, gt_path: Path, output_path: Path) -> int:
    """Merge public task data with GT fields, keyed by instance_id.

    Ported verbatim (field list and all) from BIRD-Interact's own
    ``combine_public_with_gt.py`` (see the upstream repo's "Combine the
    Public Data with the Ground Truth and Test Cases" instructions) rather
    than reimplemented, so this matches upstream's official merge exactly:
    only sol_sql/external_knowledge/test_cases (and their follow_up
    equivalents) come from GT -- everything else (question text, db name,
    ambiguity structure, conditions, ...) is expected to already be present
    on the public row. Returns the number of instances actually merged.
    """
    gt_data: dict[str, dict] = {}
    for row in _load_jsonl(gt_path):
        instance_id = row.get("instance_id")
        if instance_id:
            gt_data[instance_id] = row

    combined = 0
    missing: list[str] = []
    with output_path.open("w", encoding="utf-8") as f_out:
        for public_entry in _load_jsonl(public_path):
            instance_id = public_entry.get("instance_id")
            if not instance_id:
                continue
            gt_entry = gt_data.get(instance_id)
            if gt_entry is not None:
                public_entry["sol_sql"] = gt_entry.get("sol_sql", [])
                public_entry["external_knowledge"] = gt_entry.get("external_knowledge", [])
                public_entry["test_cases"] = gt_entry.get("test_cases", [])
                if isinstance(gt_entry.get("follow_up"), dict):
                    follow_up = public_entry.setdefault("follow_up", {})
                    follow_up["sol_sql"] = gt_entry["follow_up"].get("sol_sql", [])
                    follow_up["external_knowledge"] = gt_entry["follow_up"].get("external_knowledge", [])
                    follow_up["test_cases"] = gt_entry["follow_up"].get("test_cases", [])
                combined += 1
            else:
                missing.append(instance_id)
            f_out.write(json.dumps(public_entry) + "\n")

    if missing:
        logger.warning(
            "%d instance(s) had no matching GT entry (first few: %s)",
            len(missing), missing[:5],
        )
    return combined


def _build_datasets(variant: str, *, upstream_commit: str | None = None, force: bool = False) -> dict:
    dest = _datasets_dir(variant)
    dest.mkdir(parents=True, exist_ok=True)
    _ensure_gt_readme(variant)

    # --- public JSONL ---
    public_src = _find_public_jsonl(variant)
    if public_src is None:
        raise SystemExit(
            f"No public JSONL found under {_data_root(variant)}. "
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
    # selected_database is what BIRD-Interact actually ships (and what every
    # runner reads); the rest are tolerated aliases from sibling BIRD datasets.
    _DB_KEYS = ("selected_database", "db_name", "db_id", "database")
    db_names: list[str] = sorted({
        name for row in rows
        if (name := next((str(row[k]) for k in _DB_KEYS if row.get(k)), ""))
    })
    if rows and not db_names:
        logger.warning(
            "No database name found on any of the %d rows (looked for: %s) — "
            "the manifest's databases list will be empty.",
            len(rows), ", ".join(_DB_KEYS),
        )

    # --- combined JSONL (public + GT merged if available, else public as development fallback) ---
    gt_src = _detect_gt(variant)
    gt_available = gt_src is not None
    combined_dest = dest / GT_JSONL_NAME

    if gt_available:
        n_combined = _combine_public_with_gt(public_dest, gt_src, combined_dest)
        logger.info(
            "GT detected — merged %d/%d instances into %s",
            n_combined, task_count, combined_dest,
        )
    else:
        shutil.copy2(public_dest, combined_dest)
        logger.warning(
            "GT file not found at %s. "
            "Copied public JSONL as development fallback; scoring will be incomplete. "
            "See %s/gt/README.md to obtain GT files.",
            _gt_dir(variant) / GT_JSONL_NAME,
            dest.relative_to(_repo_root()),
        )

    # --- manifest ---
    manifest: dict = {
        "source": {
            "repository": UPSTREAM_REPO,
            "ref": DEFAULT_UPSTREAM_REF,
            "upstream_root": str(_upstream_root()),
            "upstream_commit": upstream_commit,
            "dataset_repo": HF_DATASET_REPO_TEMPLATE.format(variant=variant),
        },
        "variant": variant,
        "scope": f"bird-interact-{variant}",
        "task_count": task_count,
        "database_count": len(db_names),
        "databases": db_names,
        "gt_available": gt_available,
        "paths": {
            "public_jsonl": str(public_dest.relative_to(_repo_root())),
            "combined_jsonl": str(combined_dest.relative_to(_repo_root())),
            "gt_dir": str(_gt_dir(variant).relative_to(_repo_root())),
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


def _print_summary(variant: str, manifest: dict) -> None:
    combined = _repo_root() / manifest["paths"]["combined_jsonl"]
    env_dir = _repo_root() / manifest["paths"]["env_dir"]
    docker_service = "bird_interact_postgresql_full" if variant == "full" else "bird_interact_postgresql"

    logger.info("=" * 60)
    logger.info("BIRD-Interact %s seed complete.", variant)
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
        print(f"     Subject: [bird-interact-{variant} GT&Test Cases]")
        print("     Or download the files from the shared drive")
        print()
        print("  2. Place the received file at:")
        print(f"       {_gt_dir(variant) / GT_JSONL_NAME}")
        print()
        print(f"  3. Re-run: uv run python scripts/seed_bird_interact.py --dataset {variant}")
        print("=" * 60)
        print()

    print("Start the BIRD-Interact PostgreSQL task databases (Docker, one-time):")
    print()
    print(f"  cd {env_dir}")
    print(f"  docker compose up -d {docker_service}")
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
    dataset: str,
    ref: str = DEFAULT_UPSTREAM_REF,
    force: bool = False,
    skip_build: bool = False,
) -> dict:
    """Clone BIRD-Interact ADK + fetch the requested dataset variant. Returns the manifest."""
    if dataset not in DATASET_VARIANTS:
        raise ValueError(f"dataset must be one of {DATASET_VARIANTS}, got {dataset!r}")

    logger.info("=" * 60)
    logger.info("Seeding BIRD-Interact %s", dataset)
    logger.info("=" * 60)

    commit = _bootstrap_upstream(ref=ref)
    logger.info("Upstream BIRD-Interact commit: %s", commit)

    _bootstrap_dataset(dataset, force=force)

    if skip_build:
        logger.info("Skipping dataset build (--skip-build).")
        return {}

    manifest = _build_datasets(dataset, upstream_commit=commit, force=force)
    _print_summary(dataset, manifest)
    return manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clone BIRD-Interact ADK code and fetch the requested dataset variant "
            "(from its HuggingFace dataset repo) into datasets/bird_interact[_full]/ "
            "with the public task JSONL, GT placeholder, and manifest."
        ),
        epilog=(
            "GT SQLs and test cases require emailing bird.bench25@gmail.com "
            "with subject [bird-interact-<variant> GT&Test Cases], or downloading "
            "them from the shared drive. "
            "Place received files in datasets/bird_interact[_full]/gt/ and re-run."
        ),
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASET_VARIANTS,
        help="Which BIRD-Interact variant to seed. Required — there is no default.",
    )
    parser.add_argument(
        "--ref",
        default=DEFAULT_UPSTREAM_REF,
        help="Upstream git ref to check out (default: main)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download/re-copy files even if already present",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Only update the git clone and HF dataset download; skip JSONL copy and manifest generation",
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
        dataset=args.dataset,
        ref=args.ref,
        force=args.force,
        skip_build=args.skip_build,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
