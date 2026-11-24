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

Ingest upserts, so re-running it converges rather than duplicating. The other
two are not resumable: each writes rows as they finish, so a killed run leaves
its completed work on disk to read, but a re-run starts its file over.
"""

from __future__ import annotations

import csv
import html
import json
import logging
import re
import statistics
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from typing import Any

from langchain_core.callbacks import get_usage_metadata_callback

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


def dataset_source(scale: str) -> dict[str, Any] | None:
    """The repo and commit *scale* was downloaded from, if recorded.

    ``None`` for data fetched before the revision was pinned; re-download it
    with ``--force`` to get a run whose questions can be traced.
    """
    path = DATA_DIR / scale / "source.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


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


def token_counts(usage: dict[str, Any]) -> dict[str, Any]:
    """Flatten a usage callback's per-model totals into row fields.

    Summed across models rather than kept per model: a question may touch more
    than one endpoint, and what is being compared is the cost of answering it,
    not of any one call. The model names are kept alongside so a total can be
    read back to the endpoints that produced it, since a cached-heavy total
    from one provider is not comparable with an uncached total from another.

    ``cached_input_tokens`` is reported separately because it is already part
    of ``input_tokens`` -- the providers count it that way, and adding it again
    would roughly double a loop that re-sends its whole history every step.
    """

    def across(field: str, details: str | None = None) -> int:
        if details is None:
            return sum(int(m.get(field) or 0) for m in usage.values())
        return sum(int((m.get(details) or {}).get(field) or 0) for m in usage.values())

    return {
        "input_tokens": across("input_tokens"),
        "cached_input_tokens": across("cache_read", "input_token_details"),
        "output_tokens": across("output_tokens"),
        "total_tokens": across("total_tokens"),
        "token_models": sorted(usage),
    }


def agent_limits() -> dict[str, int]:
    """The agent's budget for this run, read from the library that enforces it.

    Recorded per row because it is not ours to set. The ReAct loop takes these
    from the environment at call time, so they are neither in the benchmark
    spec nor in the payload, and a score saved without them cannot be
    reproduced or compared: raising the observation cap from its default moved
    this arm by more than the gap between the two arms being compared, and
    nothing in the output said which cap had been used.

    Empty for the arms that never reach the agent, so a vector run is not
    annotated with budgets that did not apply to it.
    """
    from auto_ontology.retrieval.text_to_sql.react import config

    return {
        # The conversation ceiling is the one this arm answers under; the
        # general one is recorded beside it because it is what the arm would
        # fall back to, and the two differing is the normal case.
        "react_max_conversation_observation_chars": (
            config.max_conversation_observation_chars()
        ),
        "react_max_observation_chars": config.max_observation_chars(),
        "react_max_tool_calls": config.max_tool_calls(),
    }


def answers_path(scale: str, tag: str, results_dir: Path = RESULTS_DIR) -> Path:
    """Where ``answer_questions`` writes and ``judge_answers`` reads."""
    return results_dir / scale / f"{tag}_answers.jsonl"


def agent_answer(question: str, scope: str) -> tuple[str, list[dict[str, Any]]]:
    """Answer *question* about the *scope* transcript through the ReAct agent.

    The other arms retrieve once and hand the result to a reader. This hands
    the question to the agent instead, with conversation search on and nothing
    else: the agent chooses what to search for, reads what comes back, and may
    search again before it answers. That loop is the whole point of the arm --
    two of the losses that survived every prompt change need a second look at
    the transcript, and no amount of tuning a single retrieval can give them
    one.

    A conversation memory is a ``catalog_database`` carrying transcripts and no
    tables, so ``calculation`` is off: there is no warehouse to connect to and
    no query to write. ``information`` is off too, because the catalog of a
    database with no tables describes nothing.

    Returns ``(response, trajectory)``. The trajectory is the agent's whole
    path -- every tool call with its arguments and every observation that came
    back -- which is the only record of what it chose to do, since unlike the
    one-shot arms nothing here decided the retrieval in advance. It is also
    why no context string is returned: the agent assembles its own and does
    not report it, so a row from this arm has ``context_chars`` of 0 and is
    excluded from the empty-context warning below rather than counted as a
    retrieval failure.
    """
    from auto_ontology.retrieval.text_to_sql.main import (
        get_agent_response_with_state,
    )
    from auto_ontology.retrieval.text_to_sql.state import AgentRequest

    answer = get_agent_response_with_state(
        AgentRequest(
            question=question,
            calculation=False,
            information=False,
            conversation=True,
            # The agent takes the database it works on from here, and it is
            # what scopes the conversation search.
            path_state={"target_db": scope},
            connectors=[],
            prediction=False,
        )
    )
    path_state = answer.get("path_state") or {}
    return (answer.get("response") or "").strip(), path_state.get(
        "react_trajectory"
    ) or []


#: The transcript a retrieved turn came from, as the agent's search tool
#: labels it in every observation.
_TURN_CONVERSATION = re.compile(r'<turn\b[^>]*\bconversation="([^"]*)"')


def require_own_conversation(
    trajectory: Sequence[dict[str, Any]], conversation_id: str
) -> None:
    """Fail if the agent saw a turn from any transcript but *conversation_id*.

    Isolation rests on the library scoping each search to ``target_db``, and
    nothing else would notice if it stopped: a question answered from another
    conversation's turns scores as if it had been answered from its own, which
    is the pooling the benchmark exists to catch.
    """
    seen = {
        html.unescape(found)
        for step in trajectory
        if step.get("role") == "tool"
        for found in _TURN_CONVERSATION.findall(str(step.get("content") or ""))
    }
    foreign = sorted(seen - {conversation_id})
    if foreign:
        raise RuntimeError(
            f"the agent answering a question about conversation "
            f"{conversation_id} retrieved turns from {foreign}"
        )


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
    so a run killed partway leaves its completed work on disk. A row that
    raises is logged with the question it came from and the others carry on,
    but the run then fails: a mean over the questions that survived is higher
    than the score, and nothing downstream could tell it from a complete one.

    Returns ``(results, seconds)``.
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

    if failures:
        raise RuntimeError(
            f"{failures} of {len(rows)} question(s) failed while {verb}; "
            f"the other {len(results)} are in {out}. A score over the rest "
            "would leave the failures out instead of counting them."
        )
    return results, time.monotonic() - started


def answer_questions(
    *,
    scale: str,
    ids: Sequence[str] | None = None,
    top_k: int = DEFAULT_TOP_K,
    widen: int = 0,
    reader_model: str | None = None,
    workers: int = 4,
    tag: str = "ontology",
    limit: int | None = None,
    reader_prompt: str = "beam",
    retrieval: str = "vector",
    results_dir: Path = RESULTS_DIR,
) -> int:
    """Retrieve and answer, without scoring. Returns the answers written.

    The rubric is carried into the output so the judging step needs nothing but
    this file -- a scored result should be reproducible from one artifact, not
    from one artifact plus whatever ``questions.csv`` happens to say later.

    Two arms. ``vector`` searches turn text once and reads the result;
    ``agent`` hands the question to the ReAct loop, which searches as often as
    it decides to. The one-shot ontology arms were removed: the agent reaches
    the same ontology through its own tool, so keeping a second path to it
    meant two ways of asking one question and only one of them maintained.
    """
    from auto_ontology.retrieval.conversation import search_turns

    rows = load_questions(scale, ids)
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise SystemExit(f"No questions selected for scale {scale}")

    # What actually governed this run, per arm. The agent is its own reader
    # and does its own searching, so building a reader for it, or recording
    # the reader and the turn budget beside its answers, would describe a
    # configuration that never touched them -- the same failure as omitting
    # the budget that did.
    read: Any = None
    prompts: dict[str, str] = {}
    if retrieval == "agent":
        applied: dict[str, Any] = agent_limits()
    else:
        read = make_reader(model=reader_model)
        prompts = cognee_system_prompts() if reader_prompt == "cognee" else {}
        applied = {
            "reader_model": resolve_model(reader_model),
            "reader_prompt": reader_prompt,
            "top_k": top_k,
            "widen": widen,
        }
    logger.info(
        "Answering %d question(s) over %d conversation(s): retrieval=%s workers=%d %s",
        len(rows),
        len({r["conversation_id"] for r in rows}),
        retrieval,
        workers,
        " ".join(f"{k}={v}" for k, v in applied.items()),
    )

    def work(row: dict[str, Any]) -> dict[str, Any]:
        scope = scope_for(scale, row["conversation_id"])
        started = time.monotonic()
        # Entered inside the worker, because the handler travels on a
        # contextvar and a ThreadPoolExecutor thread does not inherit one
        # entered outside. Verified to capture both arms: the reader's single
        # call and the agent's whole loop, several frames down in the library.
        trajectory: list[dict[str, Any]] = []
        with get_usage_metadata_callback() as usage:
            if retrieval == "agent":
                # The agent reads for itself, so the reader built above is unused
                # and the retrieval counters stay at zero: what it searched for and
                # how often is its own decision and is not knowable from here.
                # Deliberately not comparable with the other arms -- they get one
                # search and one answer, which is the published setup.
                hits = []
                context = ""
                response, trajectory = agent_answer(row["question"], scope)
                require_own_conversation(trajectory, row["conversation_id"])
            elif retrieval == "vector":
                hits = search_turns(
                    row["question"], database_name=scope, top_k=top_k, widen=widen
                )
            else:
                raise ValueError(f"unknown retrieval {retrieval!r}")
            if retrieval != "agent":
                system = (
                    prompts.get(row["ability"], prompts.get("default"))
                    if prompts
                    else None
                )
                response, context = read_answer(
                    row["question"],
                    hits,
                    read,
                    system=system,
                    style=reader_prompt,
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
            # The reader's whole input, not just its length. Without it a
            # wrong answer cannot be told apart from a wrong retrieval, which
            # is the first question asked of every loss. Empty on the agent
            # arm, which assembles its own and reports a trajectory instead.
            "context": context,
            "context_chars": len(context),
            "retrieval": retrieval,
            **applied,
            "seconds": round(time.monotonic() - started, 1),
            # What the answer cost, summed over every model call the question
            # caused. The one-shot arms spend one call and the agent spends
            # one per step, so this is the only number that puts them on the
            # same footing: a higher score bought with four times the tokens
            # is a different trade from the same score bought with one call,
            # and the score alone cannot tell them apart.
            **token_counts(usage.usage_metadata),
            # Only the agent arm has one. For the others the retrieval is
            # settled before the question is seen, so "retrieved" above
            # already says everything there is to say about what was looked
            # at; here it is the only record of what the agent chose to do.
            **({"trajectory": trajectory} if trajectory else {}),
        }

    out = answers_path(scale, tag, results_dir)
    results, seconds = _run_concurrently(
        rows, work, out=out, workers=workers, verb="answered"
    )

    # The agent arm reports no context by construction, so counting it here
    # would report every question as a retrieval failure.
    empty = (
        0 if retrieval == "agent" else sum(1 for r in results if not r["context_chars"])
    )
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
    return len(results)


def load_answers(
    scale: str, tag: str, results_dir: Path = RESULTS_DIR
) -> list[dict[str, Any]]:
    """Read back what ``answer_questions`` wrote."""
    path = answers_path(scale, tag, results_dir)
    if not path.is_file():
        raise SystemExit(
            f"No answers at {path}. Produce them first with:\n\n"
            f"    uv run python main.py --database-name beam --scale {scale} "
            f"--retrieval {tag} --stage answer"
        )
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def question_key(row: dict[str, Any]) -> str:
    """One question's identity across files: conversation, ability, index."""
    return f"{row['conversation_id']}/{row['ability']}/{row['question_index']}"


def _require_one_question_set(
    per_round: Sequence[dict[str, Any]], expected: Sequence[str]
) -> None:
    """Refuse to pool rounds unless every one scored exactly *expected*.

    A round that lost a question has an ``overall`` over the questions that
    survived, which is higher than the score. Pooling it with a complete round
    also compares two question sets and reports the difference as run-to-run
    noise -- which is how an endpoint outage became a 0.520 round that looked
    like a real regression. Counts alone are not enough: two rounds that each
    lost a different question have the same count.

    Raising rather than warning. A warning in a log that is already thousands
    of lines of httpx chatter is not a result anybody sees.
    """
    want = sorted(expected)
    short = [
        f"r{i} (missing {len(set(want) - set(s.get('question_keys') or []))}, "
        f"extra {len(set(s.get('question_keys') or []) - set(want))})"
        for i, s in enumerate(per_round, 1)
        if sorted(s.get("question_keys") or []) != want
    ]
    if short:
        raise RuntimeError(
            f"rounds did not all score the {len(want)} expected questions: "
            f"{', '.join(short)}. Their means are over different question sets "
            "and must not be pooled; the per-round files are on disk -- re-run "
            "those rounds."
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
    for key in (
        "reader_model",
        "reader_prompt",
        "retrieval",
        "top_k",
        "widen",
        "react_max_conversation_observation_chars",
        "react_max_observation_chars",
        "react_max_tool_calls",
    ):
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
    results_dir: Path = RESULTS_DIR,
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
    rows = load_answers(scale, tag, results_dir)
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
            # On every row, so a verdict read back later can be checked against
            # the judge it is being compared as.
            "judge_prompt": flavour,
            "judge_model": judge_model or DEFAULT_JUDGE_MODEL,
            "seconds": round(time.monotonic() - started, 1),
        }

    out_dir = results_dir / scale
    results, seconds = _run_concurrently(
        rows, work, out=out_dir / f"{tag}.jsonl", workers=workers, verb="judged"
    )

    scored = [(r["conversation_id"], r["ability"], r["score"]) for r in results]
    summary = {
        "scale": scale,
        "tag": tag,
        "answers_read": len(rows),
        "questions_scored": len(results),
        "questions_failed": 0,
        # Which questions, not just how many: two rounds that each lost a
        # different question have equal counts and different question sets.
        "question_keys": sorted(question_key(r) for r in results),
        "conversations": sorted({r["conversation_id"] for r in results}, key=int),
        "dataset": dataset_source(scale),
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


def run_rounds(
    *,
    scale: str,
    rounds: int,
    tag: str,
    ids: Sequence[str] | None = None,
    top_k: int = DEFAULT_TOP_K,
    widen: int = 0,
    reader_model: str | None = None,
    judge_model: str | None = None,
    workers: int = 4,
    limit: int | None = None,
    results_dir: Path = RESULTS_DIR,
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
            reader_model=reader_model,
            workers=workers,
            tag=round_tag,
            limit=limit,
            results_dir=results_dir,
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
                results_dir=results_dir,
            )
        )
        per_round[-1]["answered"] = answered

    expected = load_questions(scale, ids)
    if limit is not None:
        expected = expected[:limit]
    _require_one_question_set(per_round, [question_key(q) for q in expected])

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
    (results_dir / scale / f"{tag}_rounds.json").write_text(
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
    "react_max_conversation_observation_chars",
    "react_max_observation_chars",
    "react_max_tool_calls",
    "conversations",
)
