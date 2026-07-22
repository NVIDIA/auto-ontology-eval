# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seed Spider2-Snow evaluations and enrichment metadata.

This uses the full ``spider2-snow.jsonl`` (547 questions), Snowflake gold
artifacts, and table metadata. It never downloads or installs SQLite files.

Usage::

    uv run python scripts/seed_spider2_snow.py
"""

from __future__ import annotations

import argparse
import logging
import sys

from spider2_seed_common import DEFAULT_UPSTREAM_REF, seed_spider2_snow


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Populate datasets/spider2/ with Spider2-Snow evaluation.json "
            "and metadata.json files."
        )
    )
    parser.add_argument("--ref", default=DEFAULT_UPSTREAM_REF)
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
    manifest_path = seed_spider2_snow(
        ref=args.ref,
        skip_build=args.skip_build,
    )
    if manifest_path is not None:
        print(f"Spider2-Snow manifest: {manifest_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
