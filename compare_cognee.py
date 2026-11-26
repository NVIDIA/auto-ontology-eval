# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Put one of our runs next to cognee's published 100K figures, ability by ability.

Their report ships the per-ability means for the run behind the 0.794
headline, and our judged rows keep every nugget verdict, so a finished run can
be scored by cognee's rule -- the nugget mean for all ten abilities, event
ordering included -- without spending a judge call.

It cannot stand in for their judge, so it refuses rows that were not judged by
cognee's prompt and judge model, rounds that are missing questions, and a run
stored both as one file and as rounds. Each of those would put a number beside
0.794 that was not measured the way 0.794 was.

    uv run --no-sync python compare_cognee.py agent --rounds 4 \\
        --results-dir auto_ontology_eval/beam/results-<run>

Their figures are vendored under ``auto_ontology_eval/beam/cognee/vendor/results/``
so this works offline; see ``THIRD_PARTY_NOTICES.md`` for provenance.
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

from auto_ontology_eval.beam.reference import cognee_columns
from auto_ontology_eval.beam.cognee.config import BEAM

HERE = Path(__file__).parent
RESULTS = HERE / "results" / "beam"
COGNEE_SUMMARY = (
    HERE
    / "auto_ontology_eval"
    / "beam"
    / "cognee"
    / "vendor"
    / "results"
    / "hybrid_completion_20_20_qa_v1_cross_run_summary.json"
)

#: Theirs is one conversation of BEAM 100K, four runs. Anything compared
#: against it has to be the same conversation, or the numbers are unrelated.
COGNEE_CONVERSATION = "2"


def cognee_figures() -> tuple[dict[str, float], float, int]:
    """Their per-ability rubric means, overall, and questions per run."""
    report = json.loads(COGNEE_SUMMARY.read_text())
    by_ability = {
        ability: stats["beam_rubric"]["mean"]
        for ability, stats in report["by_question_type"].items()
    }
    return (
        by_ability,
        report["overall"]["beam_rubric"]["mean"],
        int(report["question_count"]),
    )


def _nugget_mean(row: dict[str, Any]) -> float:
    """What this question would score under cognee's rule.

    Their judge snaps each criterion to 0.0/0.5/1.0 and takes the mean, with no
    ability excepted. Ours already stored those verdicts, so this is arithmetic
    on a finished run rather than a re-judge.
    """
    verdicts = row.get("verdicts") or []
    return statistics.mean(v["score"] for v in verdicts) if verdicts else 0.0


def round_files(
    scale: str, tag: str, rounds: int, results: Path = RESULTS
) -> list[tuple[int, Path]]:
    """The judged files for *tag*, paired with the round each one is.

    A tag run once writes ``<tag>.jsonl``; run with ``--rounds`` it writes
    ``<tag>_r1.jsonl`` and so on. Either is accepted so an arm measured either
    way can be restated, but not both: pooling them counts the same answers
    twice. Every one of *rounds* has to be there, since a missing round leaves
    a mean over a different number of runs than the one it is compared with.
    The round index lines our rounds up with their four runs; a single file is
    round 0.
    """
    directory = results / scale
    single = directory / f"{tag}.jsonl"
    numbered = [(i - 1, directory / f"{tag}_r{i}.jsonl") for i in range(1, rounds + 1)]
    present = [(index, path) for index, path in numbered if path.exists()]
    if single.exists() and present:
        raise SystemExit(
            f"both {single.name} and {tag}_rN.jsonl exist under {directory}; "
            "pooling them would count answers twice. Move one set aside."
        )
    if single.exists():
        return [(0, single)]
    if not present:
        raise SystemExit(f"no judged rows for {tag!r} under {directory}")
    missing = [path.name for _, path in numbered if not path.exists()]
    if missing:
        raise SystemExit(f"missing round file(s) for {tag!r}: {', '.join(missing)}")
    return present


def _summary_for(path: Path) -> dict[str, Any]:
    """The summary judging wrote beside *path*, or an empty dict."""
    summary = path.with_name(f"{path.stem}_summary.json")
    return json.loads(summary.read_text()) if summary.exists() else {}


def read_rounds(
    scale: str, tag: str, rounds: int, results: Path = RESULTS
) -> list[list[dict[str, Any]]]:
    """Judged rows for *tag*, one list per round.

    Each row carries the judge that produced it. Rows judged before that was
    recorded take it from their round's summary.
    """
    out: list[list[dict[str, Any]]] = []
    for _, path in round_files(scale, tag, rounds, results):
        summary = _summary_for(path)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        for row in rows:
            row.setdefault("judge_prompt", summary.get("judge_prompt"))
            row.setdefault("judge_model", summary.get("judge_model"))
        out.append(rows)
    return out


def require_cognee_judge(rows: list[dict[str, Any]], judge_model: str) -> None:
    """Refuse verdicts that did not come from cognee's prompt and model.

    Restating a run under cognee's rule only re-does their arithmetic; it
    cannot re-do their judge. Verdicts from BEAM's prompt never saw the
    question, and ``gpt-4.1-mini`` is the more lenient model on the same
    answers, so either would be scored on a different instrument from 0.794.
    """
    judges = sorted(
        {(str(r.get("judge_prompt")), str(r.get("judge_model"))) for r in rows}
    )
    if judges != [("cognee", judge_model)]:
        raise SystemExit(
            f"rows were judged by {judges}, not cognee's prompt with "
            f"{judge_model}; re-judge them before comparing with 0.794."
        )


def require_complete_rounds(
    per_round: list[list[dict[str, Any]]], abilities: set[str], questions: int
) -> None:
    """Every round must hold all of cognee's questions, once each.

    A question that failed is missing from our rows but scored 0 on their side,
    and a round short of any ability changes how much each one weighs. Either
    makes the comparison unequal in a way the overall cannot show.
    """
    for index, rows in enumerate(per_round, start=1):
        keys = [(r["ability"], str(r["question_index"])) for r in rows]
        found = {ability for ability, _ in keys}
        problems = []
        if len(rows) != questions:
            problems.append(f"{len(rows)} rows, expected {questions}")
        if len(set(keys)) != len(keys):
            problems.append("duplicate questions")
        if found != abilities:
            problems.append(f"abilities differ: {sorted(found ^ abilities)}")
        if problems:
            raise SystemExit(f"round {index} is incomplete: {'; '.join(problems)}")


def cognee_overall(per_round: list[list[float]]) -> float:
    """Their overall: each run's flat mean over questions, averaged over runs.

    Not the mean of per-ability means. The two agree only when every ability
    has the same number of questions, which is a property of this question set
    rather than of the arithmetic.
    """
    return statistics.mean(statistics.mean(scores) for scores in per_round)


def judge_errors(rows: list[dict[str, Any]]) -> int:
    """Verdicts the cognee judge scored 0.0 because the call failed."""
    return sum(
        1
        for row in rows
        for verdict in row.get("verdicts") or []
        if str(verdict.get("reason", "")).startswith("ERROR:")
    )


def backfill(scale: str, tag: str, rounds: int, results: Path = RESULTS) -> None:
    """Add the cognee columns to a run judged before they existed.

    Judging writes them now, so this is only for the arms already measured. It
    rewrites each file whole rather than appending, because the columns are
    merged into every row; the new text is built in memory and the file is
    replaced in one write, so an interrupted run leaves the old file intact
    rather than half of each.
    """
    for index, path in round_files(scale, tag, rounds, results):
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


def _disagreements(rows: list[dict[str, Any]]) -> None:
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
        ours = _nugget_mean(row)
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
    parser.add_argument("tag", help="the run to restate, e.g. agent")
    parser.add_argument("--scale", default="100K")
    parser.add_argument(
        "--rounds",
        type=int,
        default=4,
        help="how many <tag>_rN.jsonl files to pool (default 4)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS,
        help="the run directory holding <scale>/<tag>*.jsonl (default: results/)",
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
        backfill(args.scale, args.tag, args.rounds, args.results_dir)
        print()

    per_round = read_rounds(args.scale, args.tag, args.rounds, args.results_dir)
    rows = [row for round_rows in per_round for row in round_rows]
    conversations = sorted({str(r["conversation_id"]) for r in rows})
    if conversations != [COGNEE_CONVERSATION]:
        raise SystemExit(
            f"rows cover conversation(s) {conversations}; cognee published "
            f"conversation {COGNEE_CONVERSATION} only."
        )

    theirs, their_overall, their_questions = cognee_figures()
    require_cognee_judge(rows, BEAM.judge_model)
    require_complete_rounds(per_round, set(theirs), their_questions)

    ours: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        ours[row["ability"]].append(_nugget_mean(row))

    print(f"BEAM {args.scale} conversation {COGNEE_CONVERSATION}")
    print(
        f"  ours:   {args.tag}, {len(per_round)} round(s) of {their_questions}, "
        f"judged by {rows[0]['judge_model']} on cognee's prompt"
    )
    print("  theirs: hybrid_completion 20/20, 4 runs, published 0.794\n")
    print(f"{'ability':28}{'ours':>8}{'cognee':>9}{'gap':>8}")
    print("-" * 53)

    gaps: list[tuple[float, str]] = []
    for ability in sorted(ours):
        mine = statistics.mean(ours[ability])
        gap = mine - theirs[ability]
        gaps.append((gap, ability))
        print(f"{ability:28}{mine:8.3f}{theirs[ability]:9.3f}{gap:+8.3f}")

    print("-" * 53)
    overall = cognee_overall(
        [[_nugget_mean(row) for row in round_rows] for round_rows in per_round]
    )
    print(
        f"{'OVERALL':28}{overall:8.3f}{their_overall:9.3f}{overall - their_overall:+8.3f}"
    )

    errors = judge_errors(rows)
    if errors:
        print(
            f"\nwarning: {errors} verdict(s) are judge failures scored 0.0 "
            "(reason starts with ERROR:); the score above counts them as zeros.",
            file=sys.stderr,
        )

    if args.per_question:
        _disagreements(rows)

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
