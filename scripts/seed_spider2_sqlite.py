# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seed Spider2-lite local SQLite databases and evaluations.

Usage::

    uv run python scripts/seed_spider2_sqlite.py
    uv run python scripts/seed_spider2_sqlite.py --force
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from spider2_seed_common import DEFAULT_UPSTREAM_REF, seed_spider2_sqlite


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Spider2-lite and populate datasets/spider2/ with SQLite "
            "databases, local evaluation.json, and per-DB metadata.json "
            "(from resource/databases/sqlite table JSON / sample_rows)."
        ),
        epilog=(
            "Manual fallback: place local_sqlite.zip at "
            "datasets/spider2/.local_sqlite.zip and re-run with --force."
        ),
    )
    parser.add_argument("--ref", default=DEFAULT_UPSTREAM_REF)
    parser.add_argument("--dest", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-archive", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
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
    seed_spider2_sqlite(
        ref=args.ref,
        dest=args.dest,
        force=args.force,
        keep_archive=args.keep_archive,
        skip_build=args.skip_build,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
