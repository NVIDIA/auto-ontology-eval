# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Put one of our runs next to cognee's published 100K figures, ability by ability.

Two things make this possible rather than approximate. Their report ships the
per-ability means for the run behind the 0.790 headline, and our judged rows
keep every nugget verdict even for the ability whose score is not made of them
-- so a finished run can be restated under either scoring rule without spending
a single judge call on it.

That matters because the two rules disagree. BEAM's ``compute_metrics`` scores
event ordering by rank correlation; cognee's ``beam_rubric.py`` scores it by
nuggets like everything else and reports rank correlation as a separate metric.
Read our tau against their nugget mean and a run looks ~0.06 worse than it is.
Run with ``--flavour beam`` to see both columns at once.

    uv run --no-sync python compare_cognee.py pathk4 --rounds 4

Their figures are vendored under ``upstream/cognee_report/`` so this works
offline; see that directory's entry in the benchmark README for provenance.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

from conversation.beam_benchmark.reference import cognee_columns

HERE = Path(__file__).parent
RESULTS = HERE / "conversation" / "beam_benchmark" / "results"
COGNEE_SUMMARY = (
    HERE
    / "conversation"
    / "beam_benchmark"
    / "upstream"
    / "cognee_report"
    / "hybrid_completion_20_20_qa_v1_cross_run_summary.json"
)

#: Theirs is one conversation of BEAM 100K, four runs. Anything compared
#: against it has to be the same conversation, or the numbers are unrelated.
COGNEE_CONVERSATION = "2"


def cognee_figures() -> tuple[dict[str, float], float, float]:
    """Their per-ability rubric means, their overall, and their tau."""
    report = json.loads(COGNEE_SUMMARY.read_text())
    by_ability = {
        ability: stats["beam_rubric"]["mean"]
        for ability, stats in report["by_question_type"].items()
    }
    return (
        by_ability,
        report["overall"]["beam_rubric"]["mean"],
        report["overall"]["kendall_tau"]["mean"],
    )


def _nugget_mean(row: dict[str, Any]) -> float:
    """What this question would score under cognee's rule.

    Their judge snaps each criterion to 0.0/0.5/1.0 and takes the mean, with no
    ability excepted. Ours already stored those verdicts, so this is arithmetic
    on a finished run rather than a re-judge.
    """
    verdicts = row.get("verdicts") or []
    return statistics.mean(v["score"] for v in verdicts) if verdicts else 0.0


def round_files(scale: str, tag: str, rounds: int) -> list[tuple[int, Path]]:
    """The judged files for *tag*, paired with the round each one is.

    A tag run once writes ``<tag>.jsonl``; run with ``--rounds`` it writes
    ``<tag>_r1.jsonl`` and so on. Both are accepted so an arm measured either
    way can be restated. The round index matters only for lining our rounds up
    with their four runs, and a single file is round 0.
    """
    directory = RESULTS / scale
    named = [(i - 1, directory / f"{tag}_r{i}.jsonl") for i in range(1, rounds + 1)]
    named.append((0, directory / f"{tag}.jsonl"))
    found = [(index, path) for index, path in named if path.exists()]
    if not found:
        raise SystemExit(f"no judged rows for {tag!r} under {directory}")
    return found


def read_rows(scale: str, tag: str, rounds: int) -> list[dict[str, Any]]:
    """Judged rows for *tag*, across its rounds if it has any."""
    rows: list[dict[str, Any]] = []
    for _, path in round_files(scale, tag, rounds):
        rows += [json.loads(line) for line in path.read_text().splitlines() if line]
    return rows


def backfill(scale: str, tag: str, rounds: int) -> None:
    """Add the cognee columns to a run judged before they existed.

    Judging writes them now, so this is only for the arms already measured. It
    rewrites each file whole rather than appending, because the columns are
    merged into every row; the new text is built in memory and the file is
    replaced in one write, so an interrupted run leaves the old file intact
    rather than half of each.
    """
    for index, path in round_files(scale, tag, rounds):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        added = 0
        for row in rows:
            columns = cognee_columns(
                conversation_id=row["conversation_id"],
                question=row["question"],
                run=index,
            )
            added += bool(columns)
            row.update(columns)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        print(f"  {path.name}: {added}/{len(rows)} rows now carry cognee's answer")


def _wrap(label: str, text: str, width: int = 96) -> str:
    """One field of an answer, indented under *label* and hard-wrapped."""
    body = textwrap.fill(
        " ".join((text or "(empty)").split()),
        width=width,
        initial_indent="      ",
        subsequent_indent="      ",
    )
    return f"    {label}\n{body}"


def _disagreements(rows: list[dict[str, Any]], flavour: str) -> None:
    """Print the questions where their score beats ours, with both answers.

    Only where they win: the cases where we win need no diagnosis, and printing
    all twenty buries the four that matter. Rows are collapsed by question so a
    four-round run does not print the same loss four times -- the answers
    differ between rounds, but the diagnosis does not.
    """
    seen: set[str] = set()
    printed = 0
    for row in rows:
        if "cognee_score" not in row or row["question"] in seen:
            continue
        ours = row["score"] if flavour == "beam" else _nugget_mean(row)
        if ours >= row["cognee_score_mean"] - 0.001:
            continue
        seen.add(row["question"])
        printed += 1
        print(f"\n{'=' * 100}")
        print(
            f"  {row['ability']}  ours {ours:.3f}  theirs {row['cognee_score_mean']:.3f}"
        )
        print(f"  Q: {row['question']}")
        print(_wrap("gold:", row.get("golden_answer", "")))
        print(_wrap(f"cognee (run {row['cognee_run']}):", row["cognee_answer"]))
        print(_wrap("ours:", row["response"]))
        for verdict in row.get("verdicts") or []:
            print(f"    [{verdict['score']}] {verdict['nugget']}")
    if not printed:
        print("\nNo question where their mean beats ours.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="the run to restate, e.g. pathk4")
    parser.add_argument("--scale", default="100K")
    parser.add_argument(
        "--rounds",
        type=int,
        default=4,
        help="how many <tag>_rN.jsonl files to pool (default 4)",
    )
    parser.add_argument(
        "--flavour",
        choices=("cognee", "beam"),
        default="cognee",
        help="which scoring rule to put in the 'ours' column (default cognee)",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="add cognee's answer and score to the judged rows, in place",
    )
    parser.add_argument(
        "--per-question",
        action="store_true",
        help="print every question where the two disagree, with both answers",
    )
    args = parser.parse_args()

    if args.backfill:
        print(f"Backfilling {args.tag}:")
        backfill(args.scale, args.tag, args.rounds)
        print()

    rows = read_rows(args.scale, args.tag, args.rounds)
    conversations = sorted({str(r["conversation_id"]) for r in rows})
    if conversations != [COGNEE_CONVERSATION]:
        print(
            f"warning: comparing conversation(s) {conversations} against their "
            f"conversation {COGNEE_CONVERSATION}; the gaps below are not "
            f"like-for-like.\n",
            file=sys.stderr,
        )

    as_judged: dict[str, list[float]] = defaultdict(list)
    as_cognee: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        as_judged[row["ability"]].append(row["score"])
        as_cognee[row["ability"]].append(_nugget_mean(row))
    ours = as_judged if args.flavour == "beam" else as_cognee

    theirs, their_overall, their_tau = cognee_figures()

    print(f"BEAM {args.scale} conversation {COGNEE_CONVERSATION}")
    print(f"  ours:   {args.tag}, {len(rows)} judged rows, scored as {args.flavour}")
    print("  theirs: hybrid_completion 20/20, 4 runs, published 0.794\n")
    print(f"{'ability':28}{'ours':>8}{'cognee':>9}{'gap':>8}")
    print("-" * 53)

    gaps: list[tuple[float, str]] = []
    for ability in sorted(ours):
        mine = statistics.mean(ours[ability])
        if ability not in theirs:
            print(f"{ability:28}{mine:8.3f}{'--':>9}{'--':>8}")
            continue
        gap = mine - theirs[ability]
        gaps.append((gap, ability))
        print(f"{ability:28}{mine:8.3f}{theirs[ability]:9.3f}{gap:+8.3f}")

    print("-" * 53)
    overall = statistics.mean(statistics.mean(v) for v in ours.values())
    print(
        f"{'OVERALL':28}{overall:8.3f}{their_overall:9.3f}{overall - their_overall:+8.3f}"
    )

    taus = [r["ordering"]["tau_norm"] for r in rows if r.get("ordering")]
    if taus:
        tau = statistics.mean(taus)
        print(
            f"{'kendall tau (ordering)':28}{tau:8.3f}{their_tau:9.3f}{tau - their_tau:+8.3f}"
        )

    if args.per_question:
        _disagreements(rows, args.flavour)

    behind = sorted(g for g in gaps if g[0] < -0.001)
    if behind:
        # Costs before credits: the abilities we lead on offset part of this,
        # so the figure here is larger than the overall gap and is meant to be.
        # It is what there is to win, not what is currently missing.
        print(
            f"\nBehind on {len(behind)} abilities, worth "
            f"{-sum(g for g, _ in behind) / len(ours):.3f} of overall if every "
            f"one were closed: " + ", ".join(f"{a} ({g:+.3f})" for g, a in behind)
        )
    print("\nTarget to beat: 0.790")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
