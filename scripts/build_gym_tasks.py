# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build NeMo Gym task JSONL from the eval datasets.

    python scripts/build_gym_tasks.py --dataset bird60 --arm schema_only

Writes to ``resources_servers/<arm-server>/data/<dataset>.jsonl``. Run it once
per (dataset, arm); the output is deterministic, so regenerating is safe.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ontology_sql_eval.gym.tasks import DATASETS, build_records, write_jsonl

SERVER_FOR_ARM = {
    "schema_only": "schema_only_sql",
    "auto_ontology": "auto_ontology_sql",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--arm", required=True, choices=sorted(SERVER_FOR_ARM))
    parser.add_argument(
        "--no-descriptions",
        action="store_true",
        help="omit BIRD's per-column description block from the schema arm",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    spec = DATASETS[args.dataset]
    # Resolve so a relative --output still reports cleanly below.
    out = (args.output.resolve() if args.output else None) or (
        root
        / "resources_servers"
        / SERVER_FOR_ARM[args.arm]
        / "data"
        / f"{args.dataset}.jsonl"
    )

    count = write_jsonl(
        build_records(
            spec,
            root,
            args.arm,  # type: ignore[arg-type]
            descriptions=not args.no_descriptions,
            limit=args.limit,
        ),
        out,
    )
    try:
        shown = out.relative_to(root)
    except ValueError:  # --output pointed outside the repo
        shown = out
    print(f"wrote {count} tasks -> {shown}")


if __name__ == "__main__":
    main()
