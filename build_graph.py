"""Rebuild one BEAM transcript's semantic layer from scratch.

Lives here rather than in ``main.py`` because BEAM has no semantic stage there:
the eval reads a graph someone else built, and this is the someone else. Kept
as a script so a rebuild is an explicit act -- it is twenty minutes of model
calls and it replaces rows the eval is measured against.

Everything is removed first, including the vectors. A partial rebuild is worse
than none: a run that dies between writing nodes and embedding them leaves the
previous graph's vectors pointing at deleted ids, and retrieval then matches
rows that are gone. That happened once and cost a measurement.

Usage::

    uv run --no-sync python build_graph.py 2 --per-turn
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("conversation", help="Conversation id, e.g. 2")
    parser.add_argument("--scale", default="100K")
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
        "--cache-draft",
        action="store_true",
        help="Reuse the extraction draft from .scratch/drafts when the turns, "
        "the prompt and the schema are unchanged, and write it there otherwise. "
        "For retrying the stages after extraction without paying for it again. "
        "Leave off for a measured run.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Read the extracted graph back against the turns it came from. "
        "Edges their own turns contradict are dropped; statements citing a "
        "turn that denies them are reported, since the fix there is a node "
        "that is missing rather than one to remove.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    from ontology_sql_eval.env import load_env

    from conversation.beam_benchmark.run import scope_for

    load_env()

    from auto_ontology.conversations.semantic.pipeline import (
        populate_conversation_graph,
    )

    database_name = scope_for(args.scale, args.conversation)
    _wipe(database_name=database_name, conversation_id=args.conversation)

    result = populate_conversation_graph(
        database_name=database_name,
        conversation_id=args.conversation,
        relate=not args.no_relate,
        # Reset here as well as in _wipe: the collection may hold rows this
        # script could not enumerate, and the first write of a fresh build is
        # the one moment a reset is safe.
        reset_vectors=True,
        window_chars=1 if args.per_turn else 12000,
        draft_cache=_draft_path(args) if args.cache_draft else None,
        verify=args.verify,
    )
    print(
        f"\nconversation {result.conversation_id}: {result.nodes} nodes "
        f"({result.things} things, {result.statements} statements), "
        f"{result.knowledge_edges} knowledge edges, "
        f"{result.semantic_edges} semantic edges "
        f"({result.derived_edges} derived), {result.vectors} node vectors, "
        f"{result.paths} path vectors, {result.seconds:.0f}s"
    )
    return 0


def _draft_path(args) -> "Path":
    """Where this transcript's cached draft lives.

    The window size is in the name as well as the conversation: ``--per-turn``
    and the packed default are two different extractions of the same transcript,
    and both are run. The fingerprint would catch a mix-up, but only by throwing
    the other one away every time the flag is flipped.
    """
    from pathlib import Path

    window = "per-turn" if args.per_turn else "packed"
    return Path(".scratch/drafts") / f"{args.scale}-{args.conversation}-{window}.json"


def _wipe(*, database_name: str, conversation_id: str) -> None:
    """Clear this transcript's vectors.

    The rows are not touched here: ``replace_graph`` opens the build by
    deleting the transcript's nodes, and ``conversation_edge`` cascades from
    its endpoints, so doing it twice would only be a second chance to get the
    predicate wrong.

    The vectors are a different store with no such transaction, which is the
    whole reason this runs first: a build that dies midway must not leave
    vectors pointing at ids the rebuild has already deleted.
    """
    from auto_ontology.vdb import get_conversation_semantic_vdb, get_one_hop_path_vdb

    logger = logging.getLogger("build_graph")
    for name, factory in (
        ("node", get_conversation_semantic_vdb),
        ("path", get_one_hop_path_vdb),
    ):
        try:
            factory(database_name=database_name).delete_by_database(database_name)
            logger.info("Cleared the %s vectors.", name)
        except Exception:
            logger.exception("Could not clear the %s vectors.", name)


if __name__ == "__main__":
    sys.exit(main())
