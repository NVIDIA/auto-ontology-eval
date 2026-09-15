# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare two arms' Gym rollouts and report the ontology delta.

    python scripts/compare_arms.py \
        --control  runs/schema_only/bird60_rollouts.jsonl \
        --treatment runs/gsf/bird60_rollouts.jsonl

Prints execution accuracy overall and by difficulty, the delta, and the
question-level flips -- which questions the ontology fixed and, just as
importantly, which it broke. The flip list is what makes a delta diagnosable
instead of merely quotable.

Refuses to report a headline number when either side's ``no_model_output`` rate
is above a threshold. A rate-limited run scores near zero and looks exactly like
a weak model; this is the guard that keeps an outage from being published as a
benchmark result.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

HEALTH_THRESHOLD = 0.05


def load(path: Path) -> dict[str, dict[str, Any]]:
    """Index a rollout JSONL by task_id, keeping the last repeat of each."""
    out: dict[str, dict[str, Any]] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = rec.get("task_id") or str(rec.get("id"))
            out[key] = rec
    return out


def correct(rec: dict[str, Any]) -> bool:
    return float(rec.get("reward", 0.0)) > 0


def no_output_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    n = sum(1 for r in rows if r.get("failure_reason") == "no_model_output")
    return n / len(rows)


def _rate(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "-"
    n = sum(1 for r in rows if correct(r))
    return f"{n}/{len(rows)} ({n / len(rows):.1%})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--control", required=True, type=Path, help="schema-only rollouts")
    ap.add_argument("--treatment", required=True, type=Path, help="GSF rollouts")
    ap.add_argument("--control-name", default="schema-only")
    ap.add_argument("--treatment-name", default="gsf-ontology")
    ap.add_argument(
        "--ignore-health",
        action="store_true",
        help="report even if a run looks rate-limited (do not use for published numbers)",
    )
    args = ap.parse_args()

    control, treatment = load(args.control), load(args.treatment)
    common = sorted(set(control) & set(treatment))
    if not common:
        print("no overlapping task_ids between the two files", file=sys.stderr)
        return 2

    print(
        f"{args.control_name}: {len(control)} tasks   "
        f"{args.treatment_name}: {len(treatment)} tasks   common: {len(common)}"
    )
    only = set(control) ^ set(treatment)
    if only:
        print(f"  !! {len(only)} task(s) present in only one arm, excluded")

    # Health gate first: a number computed over a rate-limited run is worse than
    # no number, because it looks publishable.
    unhealthy = False
    for name, data in ((args.control_name, control), (args.treatment_name, treatment)):
        rate = no_output_rate([data[k] for k in common])
        flag = ""
        if rate > HEALTH_THRESHOLD:
            unhealthy = True
            flag = "  <-- ABOVE THRESHOLD"
        print(f"  {name}: no_model_output {rate:.1%}{flag}")
    if unhealthy and not args.ignore_health:
        print(
            f"\nRefusing to report: >{HEALTH_THRESHOLD:.0%} of tasks produced no SQL, "
            "which usually means rate limiting rather than a low score.\n"
            "Re-run at lower concurrency, or pass --ignore-health to override.",
            file=sys.stderr,
        )
        return 1

    buckets = collections.OrderedDict()
    buckets["ALL"] = common
    for label in sorted({control[k].get("difficulty") or "" for k in common}):
        if label:
            buckets[label] = [
                k for k in common if control[k].get("difficulty") == label
            ]

    print(f"\n{'':<14}{args.control_name:>20}{args.treatment_name:>20}{'delta':>10}")
    for label, keys in buckets.items():
        c = [control[k] for k in keys]
        t = [treatment[k] for k in keys]
        if not c:
            continue
        ca = sum(1 for r in c if correct(r)) / len(c)
        ta = sum(1 for r in t if correct(r)) / len(t)
        print(f"{label:<14}{_rate(c):>20}{_rate(t):>20}{ta - ca:>+9.1%}")

    gained = [k for k in common if not correct(control[k]) and correct(treatment[k])]
    lost = [k for k in common if correct(control[k]) and not correct(treatment[k])]
    print(f"\nfixed by ontology: {len(gained)}   broken by ontology: {len(lost)}")
    for label, keys in (("FIXED", gained), ("BROKE", lost)):
        for k in keys:
            rec = control[k]
            print(
                f"  {label:<6}{k:<18}{(rec.get('difficulty') or ''):<12}"
                f"{str(rec.get('question', ''))[:56]}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
