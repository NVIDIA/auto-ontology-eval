# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build NeMo Gym task JSONL from the eval datasets.

    python scripts/build_gym_tasks.py --dataset bird --arm schema_only

Writes to ``resources_servers/<arm-server>/data/tasks.jsonl``. Run it once
per (dataset, arm); the output is deterministic, so regenerating is safe.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ontology_sql_eval.gym.tasks import (
    DATASETS,
    EVIDENCE_OPTIONAL,
    SERVER_FOR_ARM,
    build_records,
    task_file,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--arm", required=True, choices=sorted(SERVER_FOR_ARM))
    parser.add_argument(
        "--no-descriptions",
        action="store_true",
        help="omit BIRD's per-column description block from the schema arm",
    )
    parser.add_argument(
        "--no-evidence",
        action="store_true",
        help="leave BEAVER's evidence (its domain-knowledge hints) out of both "
        "arms, as the BEAVER leaderboard's Setting 0 does; beaverbench only",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.no_evidence and args.dataset not in EVIDENCE_OPTIONAL:
        only = ", ".join(sorted(EVIDENCE_OPTIONAL))
        parser.error(f"--no-evidence is only for {only}, not {args.dataset}")

    root = Path(__file__).resolve().parent.parent
    spec = DATASETS[args.dataset]
    # Resolve so a relative --output still reports cleanly below.
    out = args.output.resolve() if args.output else task_file(root, args.arm)

    count = write_jsonl(
        build_records(
            spec,
            root,
            args.arm,
            descriptions=not args.no_descriptions,
            evidence=not args.no_evidence,
            limit=args.limit,
            start_index=args.start_index,
            end_index=args.end_index,
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
