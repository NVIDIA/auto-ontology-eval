# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ingest BEAM conversations and run its questions against them.

Three steps, each writing a file the next one reads:

``ingest_scale``
    One memory per conversation. Each gets its own ``database_name`` scope
    (``beam_100K_7``), which is how a question about conversation 7 is kept from
    retrieving conversation 3 -- the benchmark exists to catch exactly that kind
    of pooling, so the isolation is structural rather than a filter someone has
    to remember to pass.

``answer_questions``
    Retrieve and answer. Writes ``<tag>_answers.jsonl``: the question, what
    retrieval returned, and what the reader said. No scoring.

``judge_answers``
    Scores that file. Writes ``<tag>.jsonl`` and ``<tag>_summary.json``.

Answering and judging are separate because they fail and change for unrelated
reasons, and each is expensive enough that redoing both to fix one is the main
cost of an experiment. A judge prompt fix should not cost 400 reader calls, and
a retrieval change should not cost 1,051 judge calls. It also means an answers
file is reviewable on its own: when a score looks wrong the question is almost
always whether retrieval found the turn, and that is answerable from
``<tag>_answers.jsonl`` without rerunning anything.

Upstream splits them the same way, into ``answer_generation.py`` and
``run_evaluation.py``.

All three are resumable in the sense that matters: ingest upserts, so
re-running converges rather than duplicating, and both of the others write each
row as it finishes rather than at the end.
"""

from __future__ import annotations

import csv
import json
import logging
import statistics
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from typing import Any

from .evaluator import ORDERING_ABILITY, ability_means, overall, score_question
from .evaluator.scoring import mean
from .evaluator.judge import DEFAULT_JUDGE_MODEL, make_ask
from .reader import answer as read_answer
from .reference import (
    COGNEE_OVERALL,
    COGNEE_TAU,
    COVERED_CONVERSATION,
    cognee_ability_means,
    cognee_columns,
)
from .reader import cognee_system_prompts, make_reader, resolve_model

logger = logging.getLogger(__name__)

#: Where ``download_data.py`` puts things.
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "beam"

#: Where runs land. Git-ignored; a results file committed by hand is a result
#: nobody can reproduce.
RESULTS_DIR = Path(__file__).resolve().parent / "results"

#: The paper's retrieval depth for the RAG baseline that scored 0.323.
DEFAULT_TOP_K = 5


def scope_for(scale: str, conversation_id: str) -> str:
    """The memory scope one conversation is ingested under."""
    return f"beam_{scale}_{conversation_id}"


def conversation_ids(scale: str) -> list[str]:
    """Every conversation id on disk for *scale*, in numeric order."""
    directory = DATA_DIR / scale / "conversations"
    if not directory.is_dir():
        raise SystemExit(
            f"No BEAM data at {directory}. Fetch it with:\n\n"
            f"    uv run python -m conversation.beam_benchmark.download_data "
            f"--scale {scale}"
        )
    return [p.stem for p in sorted(directory.glob("*.json"), key=lambda p: int(p.stem))]


def load_questions(
    scale: str, ids: Sequence[str] | None = None
) -> list[dict[str, Any]]:
    """BEAM's questions for *scale*, rubrics already parsed.

    Returned in file order, which is conversation then ability then index, so a
    truncated run covers whole conversations rather than a slice of every one.
    """
    path = DATA_DIR / scale / "questions.csv"
    if not path.is_file():
        raise SystemExit(
            f"No questions at {path}; run download_data.py --scale {scale}"
        )

    wanted = set(ids) if ids is not None else None
    with path.open(encoding="utf-8") as handle:
        rows = [
            {**row, "rubric": json.loads(row["rubric"])}
            for row in csv.DictReader(handle)
            if wanted is None or row["conversation_id"] in wanted
        ]
    return rows


def ingest_scale(
    scale: str, *, ids: Sequence[str] | None = None, reset: bool = True
) -> int:
    """Build one memory per conversation. Returns the turns written.

    *reset* clears a scope's vectors before reloading it, which is what makes a
    re-ingest after an ingest-side change honest -- without it, turns that no
    longer exist keep answering searches.
    """
    from auto_ontology.conversations import Conversation, turns_from_sessions
    from auto_ontology.conversations.ingest import run_conversation_ingest

    targets = list(ids) if ids is not None else conversation_ids(scale)
    total = 0
    for position, conversation_id in enumerate(targets, start=1):
        path = DATA_DIR / scale / "conversations" / f"{conversation_id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        conversation = Conversation(
            conversation_id=str(payload["conversation_id"]),
            turns=tuple(turns_from_sessions(payload["chat"])),
        )
        scope = scope_for(scale, conversation_id)
        started = time.monotonic()
        written = run_conversation_ingest(
            conversation, database_name=scope, reset=reset
        )
        total += written
        logger.info(
            "[%d/%d] %s: %d turns in %.1fs",
            position,
            len(targets),
            scope,
            written,
            time.monotonic() - started,
        )
    logger.info("Ingested %d turn(s) across %d conversation(s)", total, len(targets))
    return total


def answers_path(scale: str, tag: str) -> Path:
    """Where ``answer_questions`` writes and ``judge_answers`` reads."""
    return RESULTS_DIR / scale / f"{tag}_answers.jsonl"


def _run_concurrently(
    rows: Sequence[dict[str, Any]],
    work: Any,
    *,
    out: Path,
    workers: int,
    verb: str,
) -> tuple[list[dict[str, Any]], int, float]:
    """Apply *work* to every row, streaming results to *out*.

    Shared by both steps because the only thing that differs between answering
    and judging is the function in the middle. Rows are written as they finish,
    so a run killed partway leaves its completed work on disk; a row that raises
    is logged with the question it came from and skipped, because one bad
    question should not cost the other 399.

    Returns ``(results, failures, seconds)``.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    failures = 0
    write_lock = Lock()
    started = time.monotonic()

    with out.open("w", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(work, row): row for row in rows}
            for done, future in enumerate(as_completed(futures), start=1):
                row = futures[future]
                try:
                    payload = future.result()
                except Exception:
                    failures += 1
                    logger.exception(
                        "conversation %s / %s failed",
                        row.get("conversation_id"),
                        row.get("ability"),
                    )
                    continue
                results.append(payload)
                with write_lock:
                    handle.write(json.dumps(payload) + "\n")
                    handle.flush()
                if done % 20 == 0 or done == len(rows):
                    logger.info(
                        "  %d/%d %s (%.0fs elapsed)",
                        done,
                        len(rows),
                        verb,
                        time.monotonic() - started,
                    )

    return results, failures, time.monotonic() - started


def answer_questions(
    *,
    scale: str,
    ids: Sequence[str] | None = None,
    top_k: int = DEFAULT_TOP_K,
    widen: int = 0,
    node_k: int = 8,
    path_k: int = 10,
    node_turns: int = 10,
    reader_model: str | None = None,
    workers: int = 4,
    tag: str = "ontology",
    limit: int | None = None,
    reader_prompt: str = "beam",
    retrieval: str = "vector",
) -> int:
    """Retrieve and answer, without scoring. Returns the answers written.

    The rubric is carried into the output so the judging step needs nothing but
    this file -- a scored result should be reproducible from one artifact, not
    from one artifact plus whatever ``questions.csv`` happens to say later.

    *node_k* and *path_k* only apply to the semantic arms and are ignored by
    ``--retrieval vector``.
    """
    from auto_ontology.retrieval.conversation import search_semantic, search_turns

    rows = load_questions(scale, ids)
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise SystemExit(f"No questions selected for scale {scale}")

    read = make_reader(model=reader_model)
    resolved_reader = resolve_model(reader_model)
    prompts = cognee_system_prompts() if reader_prompt == "cognee" else {}
    logger.info(
        "Answering %d question(s) over %d conversation(s): "
        "retrieval=%s top_k=%d widen=%d%s workers=%d reader=%s prompts=%s",
        len(rows),
        len({r["conversation_id"] for r in rows}),
        retrieval,
        top_k,
        widen,
        f" node_k={node_k} path_k={path_k}" if retrieval != "vector" else "",
        workers,
        resolved_reader,
        f"cognee, per-ability ({len(prompts)})" if prompts else "beam, single",
    )

    def work(row: dict[str, Any]) -> dict[str, Any]:
        scope = scope_for(scale, row["conversation_id"])
        started = time.monotonic()
        note = ""
        nodes = edges = via_paths = 0
        if retrieval == "vector":
            hits = search_turns(
                row["question"], database_name=scope, top_k=top_k, widen=widen
            )
        elif retrieval in ("semantic", "hybrid", "graph"):
            found = search_semantic(
                row["question"],
                database_name=scope,
                top_k=top_k,
                widen=widen,
                node_k=node_k,
                path_k=path_k,
                node_turns=node_turns,
                hybrid=retrieval == "hybrid",
            )
            hits = found.turns
            # 'graph' differs from 'semantic' in how the ontology is written
            # down and in nothing else -- same nodes, same edges, same turns.
            # Dropping the turns as well was measured and cost 0.19, and not
            # because the graph is missing the facts: HTML5, CSS3 and ES2021
            # were all extracted with correct built_with edges, and went
            # unretrieved because fetch_edges keeps an edge only when both ends
            # matched. Summarisation is the real loss -- its nuggets are
            # narrative, and the graph has no prose.
            note = found.graph if retrieval == "graph" else found.note
            nodes, edges = len(found.nodes), len(found.edges)
            via_paths = len(found.via_paths)
        else:
            raise ValueError(f"unknown retrieval {retrieval!r}")
        system = (
            prompts.get(row["ability"], prompts.get("default")) if prompts else None
        )
        response, context = read_answer(
            row["question"], hits, read, system=system, style=reader_prompt, note=note
        )
        return {
            "conversation_id": row["conversation_id"],
            "ability": row["ability"],
            "question_index": row["question_index"],
            "question": row["question"],
            "difficulty": row["difficulty"],
            "rubric": row["rubric"],
            "response": response,
            "retrieved": [
                {
                    "turn_index": h.turn_index,
                    "session_index": h.session_index,
                    "distance": h.distance,
                }
                for h in hits
            ],
            "matched": sum(1 for h in hits if h.matched),
            "context_chars": len(context),
            "reader_model": resolved_reader,
            "reader_prompt": reader_prompt,
            "retrieval": retrieval,
            "top_k": top_k,
            "widen": widen,
            # What the ontology contributed, recorded per row so an arm's gain
            # can be attributed to the nodes it actually matched rather than to
            # the flag being on. *via_paths* is the subset of those nodes that
            # only arrived because a path they sit on matched, which is the
            # whole case for embedding paths and is worth isolating.
            "nodes": nodes,
            "edges": edges,
            "via_paths": via_paths,
            "note_chars": len(note),
            "seconds": round(time.monotonic() - started, 1),
        }

    out = answers_path(scale, tag)
    results, failures, seconds = _run_concurrently(
        rows, work, out=out, workers=workers, verb="answered"
    )

    empty = sum(1 for r in results if not r["context_chars"])
    blank = sum(1 for r in results if not r["response"].strip())
    logger.info("Wrote %d answer(s) in %.0fs -> %s", len(results), seconds, out)
    if empty:
        logger.warning(
            "%d question(s) retrieved no context at all -- those are retrieval "
            "failures, not reader failures.",
            empty,
        )
    if blank:
        logger.warning(
            "%d answer(s) came back empty with context present -- a reasoning "
            "reader out of token budget. They will score 0 on every nugget.",
            blank,
        )
    if failures:
        logger.warning("%d question(s) failed and are not in the file.", failures)
    return len(results)


def load_answers(scale: str, tag: str) -> list[dict[str, Any]]:
    """Read back what ``answer_questions`` wrote."""
    path = answers_path(scale, tag)
    if not path.is_file():
        raise SystemExit(
            f"No answers at {path}. Produce them first with:\n\n"
            f"    uv run python main.py --database-name beam --scale {scale} "
            f"--tag {tag} --stage answer"
        )
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _require_one_question_set(per_round: Sequence[dict[str, Any]]) -> None:
    """Refuse to pool rounds that answered different numbers of questions.

    A round whose reader call failed writes no row for that question, and its
    ``overall`` is then the mean over the questions that survived. Pooling that
    with a complete round compares two different question sets and reports the
    difference as run-to-run noise -- which is how an endpoint outage became a
    0.520 round that looked like a real regression.

    Raising rather than warning. A warning in a log that is already thousands
    of lines of httpx chatter is not a result anybody sees, and the number this
    produces goes into a README next to a competitor's.
    """
    counts = {int(summary.get("answered") or 0) for summary in per_round}
    if len(counts) > 1:
        detail = ", ".join(
            f"r{i}={summary.get('answered')}" for i, summary in enumerate(per_round, 1)
        )
        raise RuntimeError(
            f"rounds answered different question counts ({detail}); the means "
            "are over different question sets and must not be pooled. The "
            "per-round files are on disk -- re-run the short rounds."
        )


def _retrieval_settings(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """How *rows* were produced, for the summary.

    A list rather than a value per key, because a mixed answers file is a real
    possibility -- appending a re-run of a few questions at a different ``top_k``
    is the obvious way to patch a partial run, and a summary claiming the whole
    file used one setting would hide that. Older files predate these keys and
    report ``unrecorded``.
    """
    settings: dict[str, Any] = {}
    for key in ("reader_model", "reader_prompt", "retrieval", "top_k", "widen"):
        seen = sorted({str(r.get(key, "unrecorded")) for r in rows})
        settings[key] = seen[0] if len(seen) == 1 else seen
    return settings


def judge_answers(
    *,
    scale: str,
    tag: str = "ontology",
    judge_model: str | None = None,
    workers: int = 4,
    limit: int | None = None,
    flavour: str = "beam",
    cognee_run: int = 0,
) -> dict[str, Any]:
    """Score an answers file. Returns the summary.

    The summary lands only at the end, unlike the per-question rows. A partial
    summary read as a score is worse than no summary: the headline number gives
    each of the ten abilities a tenth of the weight regardless of how many of
    its questions actually ran.

    *cognee_run* is which of their four published runs each row shows the
    answer from; ``run_rounds`` pairs it with ours so four rounds show four of
    theirs rather than one of theirs four times.
    """
    rows = load_answers(scale, tag)
    if limit is not None:
        rows = rows[:limit]

    ask = make_ask(model=judge_model)
    logger.info(
        "Judging %d answer(s) with %s (%s prompt), workers=%d",
        len(rows),
        judge_model or DEFAULT_JUDGE_MODEL,
        flavour,
        workers,
    )

    def work(row: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        result = score_question(
            conversation_id=row["conversation_id"],
            ability=row["ability"],
            question=row["question"],
            response=row["response"],
            rubric=row["rubric"],
            ask=ask,
            flavour=flavour,
        )
        return {
            **{k: v for k, v in asdict(result).items() if k != "verdicts"},
            "question_index": row["question_index"],
            "difficulty": row["difficulty"],
            "retrieved": row["retrieved"],
            "context_chars": row["context_chars"],
            # Empty for every question they did not publish, which is all of
            # them outside conversation 2. A lost question is then readable
            # without opening a second file: our answer, theirs, and the gold.
            **cognee_columns(
                conversation_id=row["conversation_id"],
                question=row["question"],
                run=cognee_run,
            ),
            "verdicts": [asdict(v) for v in result.verdicts],
            "seconds": round(time.monotonic() - started, 1),
        }

    out_dir = RESULTS_DIR / scale
    results, failures, seconds = _run_concurrently(
        rows, work, out=out_dir / f"{tag}.jsonl", workers=workers, verb="judged"
    )

    scored = [(r["conversation_id"], r["ability"], r["score"]) for r in results]
    summary = {
        "scale": scale,
        "tag": tag,
        "answers_read": len(rows),
        "questions_scored": len(results),
        "questions_failed": failures,
        "conversations": sorted({r["conversation_id"] for r in results}, key=int),
        # Read off the answers rather than taken as an argument: judging does
        # not choose the reader, and a summary that named a reader the answers
        # were not produced by would be worse than one that named none.
        **_retrieval_settings(rows),
        "judge_model": judge_model or DEFAULT_JUDGE_MODEL,
        "judge_prompt": flavour,
        "judge_errors": sum(
            1
            for r in results
            for v in r["verdicts"]
            if str(v["reason"]).startswith("ERROR:")
        ),
        "ability_means": ability_means(scored),
        "overall": overall(scored),
        # Reported beside the headline figure rather than inside it, which is
        # what cognee's registry does: their cross-run summary carries
        # kendall_tau as its own metric, null for the nine abilities it does
        # not apply to. Under flavour="beam" this is also the event-ordering
        # score; under flavour="cognee" it is a second opinion on it.
        "kendall_tau": _tau_mean(results),
        "seconds": round(seconds, 1),
    }
    (out_dir / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))

    _report(summary)
    return summary


def _comparable(summary: dict[str, Any]) -> bool:
    """Whether this run can honestly be put beside cognee's published figures.

    Their 0.794 is conversation 2 under their own scoring rule. A run over other
    conversations is a different question set, and a run judged under
    ``flavour="beam"`` scores event ordering by rank correlation where they
    score it by nuggets -- so in both cases the gap column would be comparing
    unlike things, and is left off rather than printed with a caveat nobody
    reads.
    """
    return summary.get("judge_prompt") == "cognee" and [
        str(c) for c in summary.get("conversations") or []
    ] == [COVERED_CONVERSATION]


def _round_taus(per_round: Sequence[dict[str, Any]]) -> list[float]:
    """Every round's ``kendall_tau``, skipping rounds that have none."""
    return [s["kendall_tau"] for s in per_round if s.get("kendall_tau") is not None]


def _tau_mean(results: Sequence[dict[str, Any]]) -> float | None:
    """Mean ``tau_norm`` over the event-ordering questions, or None if none ran.

    Pooled over that ability alone, not over all twenty questions. A mean that
    counted the other nine as zero would be a different and much smaller
    number, and not the one cognee publishes.
    """
    taus = [
        row["ordering"]["tau_norm"]
        for row in results
        if row["ability"] == ORDERING_ABILITY and row.get("ordering")
    ]
    return mean(taus) if taus else None


def _report(summary: dict[str, Any]) -> None:
    """Print the summary next to the number it has to be compared against.

    Cognee's column is printed inline rather than left to ``compare_cognee.py``,
    because the question a finished run raises is always which abilities it lost
    on, and reading that off two files is how a 0.19 metric mismatch went
    unnoticed for a week.
    """
    means: dict[str, float] = summary["ability_means"]
    theirs = cognee_ability_means() if _comparable(summary) else {}

    logger.info("%s", "=" * 62)
    logger.info("BEAM %s / %s", summary["scale"], summary["tag"])
    logger.info("%s", "=" * 62)
    if theirs:
        logger.info("  %-26s %6s %8s %8s", "", "ours", "cognee", "gap")
    for ability in sorted(means):
        if ability in theirs:
            logger.info(
                "  %-26s %6.3f %8.3f %+8.3f",
                ability,
                means[ability],
                theirs[ability],
                means[ability] - theirs[ability],
            )
        else:
            logger.info("  %-26s %6.3f", ability, means[ability])
    if theirs:
        logger.info(
            "  %-26s %6.3f %8.3f %+8.3f",
            "OVERALL",
            summary["overall"],
            COGNEE_OVERALL,
            summary["overall"] - COGNEE_OVERALL,
        )
    else:
        logger.info("  %-26s %.3f", "OVERALL", summary["overall"])
    if summary.get("kendall_tau") is not None:
        note = (
            "the score above for this ability"
            if summary["judge_prompt"] == "beam"
            else "reported beside it, as cognee does"
        )
        # Theirs alongside ours on the same terms as OVERALL. Their tau is
        # published for the same conversation and the same questions, so the
        # comparison holds wherever the headline comparison does -- and without
        # it the figure reads as a bare 0.155 with nothing to be 0.155 against.
        if theirs:
            logger.info(
                "  %-26s %6.3f %8.3f %+8.3f  (event ordering only, %s)",
                "kendall tau",
                summary["kendall_tau"],
                COGNEE_TAU,
                summary["kendall_tau"] - COGNEE_TAU,
                note,
            )
        else:
            logger.info(
                "  %-26s %6.3f  (event ordering only, %s)",
                "kendall tau",
                summary["kendall_tau"],
                note,
            )
    logger.info("%s", "-" * 62)
    logger.info(
        "  %-26s %.3f  (Llama-4-Maverick + RAG, all 400 questions)",
        "published best baseline",
        0.323,
    )
    if summary["judge_prompt"] == "cognee":
        logger.info(
            "  %-26s %.3f  (conversation 2, 4 runs, tau %.3f)",
            "cognee hybrid 20/20",
            0.794,
            0.515,
        )
    if len(means) < 10:
        logger.warning(
            "Only %d of 10 abilities have questions -- this number is not "
            "comparable to the published one.",
            len(means),
        )
    if summary["questions_failed"]:
        logger.warning(
            "%d question(s) failed and are excluded.", summary["questions_failed"]
        )


def run_rounds(
    *,
    scale: str,
    rounds: int,
    tag: str,
    ids: Sequence[str] | None = None,
    top_k: int = DEFAULT_TOP_K,
    widen: int = 0,
    node_k: int = 8,
    path_k: int = 10,
    node_turns: int = 10,
    reader_model: str | None = None,
    judge_model: str | None = None,
    workers: int = 4,
    limit: int | None = None,
    reader_prompt: str = "beam",
    retrieval: str = "vector",
    flavour: str = "beam",
) -> dict[str, Any]:
    """Answer and judge *rounds* times, then average, writing a pooled summary.

    Each round re-answers as well as re-judges. Re-judging fixed answers would
    measure only judge variance and would understate the spread a reader
    contributes at the same temperature, which is the larger of the two.
    Cognee's four 100K rounds repeat "the full QA-and-scoring round", so this
    does the same.

    Rounds land under ``<tag>_r<N>`` so each is readable on its own, and the
    pooled file is ``<tag>_rounds.json``. The round standard deviation it
    reports is run-to-run noise on a fixed question set; it is not a confidence
    interval over questions, and on a single conversation it will look far
    smaller than the between-conversation spread that actually limits the
    comparison. Reported because the number it is being compared against is of
    the same kind.
    """
    per_round: list[dict[str, Any]] = []
    for index in range(1, rounds + 1):
        round_tag = f"{tag}_r{index}"
        logger.info("%s", "=" * 62)
        logger.info("Round %d of %d (%s)", index, rounds, round_tag)
        logger.info("%s", "=" * 62)
        answered = answer_questions(
            scale=scale,
            ids=ids,
            top_k=top_k,
            widen=widen,
            node_k=node_k,
            path_k=path_k,
            node_turns=node_turns,
            reader_model=reader_model,
            workers=workers,
            tag=round_tag,
            limit=limit,
            reader_prompt=reader_prompt,
            retrieval=retrieval,
        )
        per_round.append(
            judge_answers(
                scale=scale,
                tag=round_tag,
                judge_model=judge_model,
                workers=workers,
                flavour=flavour,
                cognee_run=index - 1,
            )
        )
        per_round[-1]["answered"] = answered

    _require_one_question_set(per_round)

    overalls = [s["overall"] for s in per_round]
    abilities = sorted({a for s in per_round for a in s["ability_means"]})
    pooled = {
        "scale": scale,
        "tag": tag,
        "rounds": rounds,
        "round_tags": [f"{tag}_r{i}" for i in range(1, rounds + 1)],
        "round_overalls": [round(v, 4) for v in overalls],
        "questions_failed": sum(s["questions_failed"] for s in per_round),
        **{k: v for k, v in per_round[0].items() if k in _PINNED},
        "judge_model": per_round[0]["judge_model"],
        "judge_prompt": per_round[0]["judge_prompt"],
        "judge_errors": sum(s["judge_errors"] for s in per_round),
        "ability_means": {
            a: mean(
                [s["ability_means"][a] for s in per_round if a in s["ability_means"]]
            )
            for a in abilities
        },
        "overall": mean(overalls),
        "overall_sd": statistics.stdev(overalls) if len(overalls) > 1 else 0.0,
        "overall_min": min(overalls),
        "overall_max": max(overalls),
        # Averaged over the rounds that have one, so a run with no
        # event-ordering questions reports None rather than zero.
        "kendall_tau": mean(taus) if (taus := _round_taus(per_round)) else None,
    }
    (RESULTS_DIR / scale / f"{tag}_rounds.json").write_text(
        json.dumps(pooled, indent=2)
    )

    _report(pooled)
    logger.info(
        "  %-26s %.3f +/- %.3f  (%d rounds, %.3f to %.3f)",
        "pooled",
        pooled["overall"],
        pooled["overall_sd"],
        rounds,
        pooled["overall_min"],
        pooled["overall_max"],
    )
    return pooled


#: Settings carried from a round's summary into the pooled one unchanged.
_PINNED = (
    "reader_model",
    "reader_prompt",
    "retrieval",
    "top_k",
    "widen",
    "conversations",
)


def iter_summaries(scale: str) -> Iterable[dict[str, Any]]:
    """Every summary written for *scale*, for comparing arms after the fact."""
    for path in sorted((RESULTS_DIR / scale).glob("*_summary.json")):
        yield json.loads(path.read_text(encoding="utf-8"))
