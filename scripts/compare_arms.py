# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare two arms' Gym rollouts and report the ontology delta.

    python scripts/compare_arms.py \
        --control  runs/control/bird.jsonl \
        --treatment runs/treatment/bird.jsonl

Prints execution accuracy overall and by difficulty, the delta, and the
question-level flips -- which questions the ontology fixed and, just as
importantly, which it broke. The flip list is what makes a delta diagnosable
instead of merely quotable.

Refuses to report a headline number when either side's ``no_model_output`` rate
is above a threshold. A rate-limited run scores near zero and looks exactly like
a weak model; this is the guard that keeps an outage from being published as a
benchmark result.

Also refuses while any question Gym dispatched has no rollout row: it failed
(run_gym_arm.sh records failures in a sidecar and keeps going) or never ran.
Re-run that arm with ``--resume`` to finish them.
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


def unfinished(path: Path, done: dict[str, dict[str, Any]]) -> set[str]:
    """Task ids Gym dispatched for the run at *path* that have no rollout row.

    Read from Gym's ``<stem>_materialized_inputs.jsonl``. Such a question either
    failed -- its attempts are in ``<stem>_failures.jsonl`` -- or never ran.
    Comparing only the arms' rollouts would miss a question both arms lost.
    """
    materialized = path.with_name(path.stem + "_materialized_inputs.jsonl")
    if not materialized.exists():
        return set()
    dispatched = set()
    with materialized.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                dispatched.add(rec.get("task_id") or str(rec.get("id")))
    return dispatched - set(done)


# Mirrors ontology_sql_eval.gym.exec_match.UNUSABLE_FAILURES. A task whose gold
# query never ran says nothing about either arm, so it must not sit in the
# denominator -- the resources servers already exclude it from pass@k, and this
# report has to agree with them or the two disagree on the same run.
UNUSABLE_GOLD = ("gold_execution_error", "gold_execution_timeout")


def usable(rec: dict[str, Any]) -> bool:
    return rec.get("failure_reason") not in UNUSABLE_GOLD


def correct(rec: dict[str, Any]) -> bool:
    return float(rec.get("reward", 0.0)) > 0


# Failures that mean the run is broken rather than the model being weak. An arm
# that errors on every task would otherwise clear a gate that only watched
# no_model_output, and get published as "0% accuracy".
BROKEN_RUN_FAILURES = ("no_model_output", "unknown_error")


def broken_rate(rows: list[dict[str, Any]]) -> tuple[float, dict[str, int]]:
    """Fraction of rows whose reward reflects a broken run, with the breakdown."""
    if not rows:
        return 0.0, {}
    counts = {
        code: sum(1 for r in rows if r.get("failure_reason") == code)
        for code in BROKEN_RUN_FAILURES
    }
    return sum(counts.values()) / len(rows), {k: v for k, v in counts.items() if v}


def _rate(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "-"
    n = sum(1 for r in rows if correct(r))
    return f"{n}/{len(rows)} ({n / len(rows):.1%})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--control", required=True, type=Path, help="schema-only rollouts")
    ap.add_argument(
        "--treatment", required=True, type=Path, help="Auto Ontology rollouts"
    )
    ap.add_argument("--control-name", default="schema-only")
    ap.add_argument("--treatment-name", default="auto-ontology")
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
    for name, path, data in (
        (args.control_name, args.control, control),
        (args.treatment_name, args.treatment, treatment),
    ):
        missing = unfinished(path, data)
        if missing:
            unhealthy = True
            print(
                f"  {name}: {len(missing)} question(s) with no result (failed or never "
                f"ran; see {path.stem}_failures.jsonl)  <-- re-run with --resume"
            )
    for name, data in ((args.control_name, control), (args.treatment_name, treatment)):
        rate, breakdown = broken_rate([data[k] for k in common])
        flag = ""
        if rate > HEALTH_THRESHOLD:
            unhealthy = True
            flag = "  <-- ABOVE THRESHOLD"
        detail = ", ".join(f"{k}={v}" for k, v in breakdown.items()) or "none"
        print(f"  {name}: broken-run failures {rate:.1%} ({detail}){flag}")
    if unhealthy and not args.ignore_health:
        print(
            "\nRefusing to report: questions have no result, or more than "
            f"{HEALTH_THRESHOLD:.0%} failed in a way that reflects a broken run (no "
            "output, or an unknown error) rather than a low score.\n"
            "Re-run with --resume to finish missing questions, or at lower "
            "concurrency; pass --ignore-health to override.",
            file=sys.stderr,
        )
        return 1

    buckets = collections.OrderedDict()
    buckets["ALL"] = common
    # BEAVER's second headline number: accuracy without the questions whose gold
    # result is empty, which any empty or failed prediction matches. Gold is a
    # property of the question, so either arm's record will do.
    golds = {
        k: {control[k].get("gold_result"), treatment[k].get("gold_result")}
        for k in common
    }
    if any(g & {"empty_gold", "failed_gold"} for g in golds.values()):
        buckets["nonempty gold"] = [k for k in common if "nonempty_gold" in golds[k]]
    for label in sorted({control[k].get("difficulty") or "" for k in common}):
        if label:
            buckets[label] = [
                k for k in common if control[k].get("difficulty") == label
            ]

    print(f"\n{'':<14}{args.control_name:>20}{args.treatment_name:>20}{'delta':>10}")
    for label, keys in buckets.items():
        # A task is scored only when BOTH arms had a runnable gold query, so the
        # two columns always share a denominator.
        scored = [k for k in keys if usable(control[k]) and usable(treatment[k])]
        dropped = len(keys) - len(scored)
        c = [control[k] for k in scored]
        t = [treatment[k] for k in scored]
        if not c:
            continue
        ca = sum(1 for r in c if correct(r)) / len(c)
        ta = sum(1 for r in t if correct(r)) / len(t)
        note = f"  ({dropped} unusable gold)" if dropped else ""
        print(f"{label:<14}{_rate(c):>20}{_rate(t):>20}{ta - ca:>+9.1%}{note}")

    scorable = [k for k in common if usable(control[k]) and usable(treatment[k])]
    gained = [k for k in scorable if not correct(control[k]) and correct(treatment[k])]
    lost = [k for k in scorable if correct(control[k]) and not correct(treatment[k])]
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
