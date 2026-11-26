# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rebuild one transcript's semantic layer from scratch.

The sibling of :mod:`~auto_ontology_eval.ingestion.semantic`, which compiles a
SQL database's taxonomy, and the second half of the conversation path: run it
after :mod:`~auto_ontology_eval.ingestion.conversation_ingest` has written the
turns and embedded them.

Shared by every corpus, because nothing in a rebuild is per corpus. The source
that parsed the transcript is already behind it -- what this reads is turns in
the store, and a turn is a turn whether two roles or forty people said it. All
a caller names is which corpus, which revision, and which conversation, and the
scope is spelled by the same function the ingest spells it with.

Kept as a command rather than a stage in ``main.py`` so a rebuild is an
explicit act: it is twenty minutes of model calls and it replaces rows the eval
is measured against.

Everything is removed first, including the vectors. A partial rebuild is worse
than none: a run that dies between writing nodes and embedding them leaves the
previous graph's vectors pointing at deleted ids, and retrieval then matches
rows that are gone. That happened once and cost a measurement.

Usage::

    uv run --no-sync python -m auto_ontology_eval.ingestion.conversation_semantic \\
        beam 2 --per-turn
    uv run --no-sync python -m auto_ontology_eval.ingestion.conversation_semantic \\
        slack ct_lss_qa --revision v1
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import TYPE_CHECKING

from auto_ontology_eval.ingestion.conversation_ingest import (
    CORPORA,
    DEFAULT_REVISION,
    scope_for,
)

if TYPE_CHECKING:
    from auto_ontology.conversations.semantic.pipeline import GraphResult

logger = logging.getLogger("conversation_semantic")

__all__ = ["run_conversation_semantic"]

#: Turns per extraction call, as a character budget. Cognee chunks one turn at
#: a time, so ``--per-turn`` reproduces their granularity; the default packs
#: turns up to the library's window instead.
_PER_TURN_CHARS = 1
_WINDOW_CHARS = 12000


def run_conversation_semantic(
    corpus: str,
    conversation_id: str,
    revision: str,
    *,
    per_turn: bool = False,
    relate: bool = True,
    verify: bool | None = None,
    subject_agent: bool = False,
) -> GraphResult:
    """Wipe and rebuild *conversation_id*'s graph. Returns what was written."""
    from auto_ontology.conversations.semantic.pipeline import (
        populate_conversation_graph,
    )

    if corpus not in CORPORA:
        known = ", ".join(sorted(CORPORA))
        raise SystemExit(f"Unknown corpus {corpus!r}. Known: {known}.")

    database_name = scope_for(corpus, revision, conversation_id)
    _wipe(database_name=database_name)

    return populate_conversation_graph(
        database_name=database_name,
        conversation_id=conversation_id,
        relate=relate,
        # Reset here as well as in ``_wipe``: the collection may hold rows this
        # command could not enumerate, and the first write of a fresh build is
        # the one moment a reset is safe.
        reset_vectors=True,
        window_chars=_PER_TURN_CHARS if per_turn else _WINDOW_CHARS,
        verify=verify,
        subject_agent=subject_agent,
    )


def repair(corpus: str, conversation_id: str, revision: str, *, apply: bool = False):
    """Judge a stored graph's edges instead of rebuilding it.

    Here only to spell the scope: the work is
    :func:`auto_ontology.conversations.semantic.repair.repair_stored_graph`,
    which is the library's because nothing about reading a graph back is
    evaluation.

    Separate from a rebuild rather than a flag on it, because the two are not
    interchangeable at the cost: this reads the stored graph in about five
    minutes, while a rebuild is half an hour and hands back a *different*
    graph than the one the verdicts were read for.
    """
    from auto_ontology.conversations.semantic.repair import repair_stored_graph

    if corpus not in CORPORA:
        known = ", ".join(sorted(CORPORA))
        raise SystemExit(f"Unknown corpus {corpus!r}. Known: {known}.")

    return repair_stored_graph(
        database_name=scope_for(corpus, revision, conversation_id),
        conversation_id=conversation_id,
        apply=apply,
    )


def _wipe(*, database_name: str) -> None:
    """Clear this transcript's vectors.

    The rows are not touched here: ``replace_graph`` opens the build by
    deleting the transcript's nodes, and ``conversation_edge`` cascades from
    its endpoints, so doing it twice would only be a second chance to get the
    predicate wrong.

    The vectors are a different store with no such transaction, which is the
    whole reason this runs first: a build that dies midway must not leave
    vectors pointing at ids the rebuild has already deleted. For the same
    reason a failure here stops the build rather than carrying on with the old
    vectors in place.
    """
    from auto_ontology.vdb import get_conversation_semantic_vdb, get_one_hop_path_vdb

    for name, factory in (
        ("node", get_conversation_semantic_vdb),
        ("path", get_one_hop_path_vdb),
    ):
        factory(database_name=database_name).delete_by_database(database_name)
        logger.info("Cleared the %s vectors.", name)


def _report(result: GraphResult) -> None:
    print(
        f"\nconversation {result.conversation_id}: {result.nodes} nodes "
        f"({result.things} things, {result.statements} statements), "
        f"{result.knowledge_edges} knowledge edges, "
        f"{result.semantic_edges} semantic edges "
        f"({result.derived_edges} derived), {result.vectors} node vectors, "
        f"{result.paths} path vectors, {result.seconds:.0f}s"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", choices=sorted(CORPORA), help="Which corpus.")
    parser.add_argument(
        "conversation", help="Conversation id: 2 for BEAM, a channel for Slack."
    )
    parser.add_argument(
        "--revision",
        help="Which seeded revision: a scale for BEAM, an export for Slack. "
        "Defaults per corpus, see DEFAULT_REVISION.",
    )
    parser.add_argument(
        "--per-turn",
        action="store_true",
        help="One extraction call per turn, as cognee chunks. Otherwise turns "
        "are packed to the default window size.",
    )
    parser.add_argument(
        "--no-relate",
        action="store_true",
        help="Derive the semantic edges but skip adjudication.",
    )
    parser.add_argument(
        "--verify",
        # Three states, not two. Omitted leaves the decision to the
        # ``conversation_verify_enabled`` flag; ``--verify`` and ``--no-verify``
        # pin it, which is what a run being measured wants -- the graph behind a
        # published number must not depend on a row someone toggled.
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Read the extracted graph back against the turns it came from, "
        "and change it. Edges their own turns contradict are dropped. A "
        "statement citing a turn that denies it is kept -- the fix there is a "
        "node that is missing, not one to remove -- but the denying turn "
        "yields a retraction node, and the denied citation is detached once "
        "another remains to carry the statement. Defaults to the "
        "conversation_verify_enabled feature flag.",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Judge the stored graph's edges instead of rebuilding it: about "
        "five minutes, read-only unless --apply. The rebuild flags are ignored.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="With --repair, delete the contradicted edges and re-embed the "
        "paths. Without it the repair pass only reports.",
    )
    parser.add_argument(
        "--subject-agent",
        action="store_true",
        help="Let the subject pass read turns it was not shown before "
        "answering, rather than deciding from its batch alone. For statements "
        "whose subject is a thing introduced outside that batch, which arrives "
        "as a description rather than as the turns stating it.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    from auto_ontology_eval.env import load_env

    load_env()

    revision = args.revision or DEFAULT_REVISION[args.corpus]

    if args.repair:
        repair(args.corpus, args.conversation, revision, apply=args.apply)
        return 0

    if args.apply:
        raise SystemExit("--apply only means something with --repair.")

    _report(
        run_conversation_semantic(
            args.corpus,
            args.conversation,
            revision,
            per_turn=args.per_turn,
            relate=not args.no_relate,
            verify=args.verify,
            subject_agent=args.subject_agent,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
