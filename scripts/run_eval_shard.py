#!/usr/bin/env python3
"""Run a contiguous slice of the eval set into a dedicated CSV.

``run_evaluation`` is sequential, so wall-clock is cut by running several
disjoint slices as separate processes. Each shard owns its own output file to
keep the per-question ``writer.writerow``/``flush`` durable without processes
contending for one handle; ``merge_eval_shards.py`` reassembles them.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

# Auto Ontology resolves its model endpoint/key into module-level constants at import
# time, so the environment has to be populated before that import happens.
load_dotenv()

from ontology_sql_eval.retrieval.eval_chatbot import (  # noqa: E402
    _resolve_paths,
    run_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one eval shard.")
    parser.add_argument("--database-name", required=True)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument(
        "--end-index",
        type=int,
        required=True,
        help="Exclusive upper bound, matching Python slice semantics.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    input_path, _ = _resolve_paths(args.database_name, None, None)

    # A shard writes a fresh file, so run_evaluation emits a header even though
    # start_index > 0 (its append path only engages for an existing file).
    if args.output.exists():
        args.output.unlink()

    run_evaluation(
        input_path=input_path,
        output_path=args.output,
        start_index=args.start_index,
        end_index=args.end_index,
    )


if __name__ == "__main__":
    main()
