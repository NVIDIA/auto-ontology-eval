# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The conversation write path, shared by every conversation corpus.

The sibling of :mod:`~auto_ontology_eval.ingestion.ingest`, which is the one
place a SQL source becomes catalog rows and vectors no matter which benchmark
asked. This is that place for transcripts, and it exists for the same reason:
BEAM and Slack were each carrying their own copy of the loop, identical but
for the format name and how a scope is spelled.

The structured path's shape, kept deliberately:

- ``ingest`` is mutual. Nothing below is per corpus.
- The read is the connector's. A SQL connector translates a dialect; a
  conversation source parses a format, including where its files sit under the
  root -- so a caller passes a name and a root, never a layout.
- What comes back is mutual. ``Turn`` objects reach Postgres and pgvector
  through the library's ``run_conversation_ingest``, which has no idea whether
  the speakers were two roles or forty people.

What stays per corpus is a row in :data:`CORPORA`: which source reads it, and
nothing else. That table is ingest configuration and is kept apart from
``auto_ontology_eval.beam.cognee.config``, which holds scoring instruments -- a judge model
is a claim about a published number, a parser name is not.

Run after the Postgres catalog and pgvector are up, and after the corpus has
been seeded::

    uv run python -m auto_ontology_eval.ingestion.conversation_ingest slack
    uv run python -m auto_ontology_eval.ingestion.conversation_ingest beam \\
        --revision 100K --conversation 2
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "CORPORA",
    "DEFAULT_REVISION",
    "ingest_conversations",
    "ingest_corpus",
    "scope_for",
]

#: Where the seed scripts put corpora, one directory per corpus and one
#: subdirectory per revision.
DATASETS_DIR = Path(__file__).resolve().parents[2] / "datasets"

#: Which source format reads each corpus. The two differ: BEAM ships the
#: two-party ``chat`` format, where a user asks and an assistant answers, and
#: is named for the benchmark rather than the format because a second
#: ``chat`` corpus would sit beside it.
#:
#: Named rather than sniffed. A connection string carries its scheme, so the
#: SQL side can dispatch on it; a directory of JSON says nothing about whether
#: its speakers are roles or people, and guessing wrong is not an error, it is
#: a corpus parsed into plausible nonsense.
CORPORA: dict[str, str] = {
    "beam": "chat",
    "slack": "slack",
}

#: The revision each corpus is read at unless one is named. Per corpus rather
#: than one default for all, because the revisions are not comparable: BEAM's
#: is a transcript scale and Slack's is an export date. Both commands read it
#: from here -- a default that differed between ingest and rebuild would put
#: the turns in one scope and the graph in another, and neither would complain.
DEFAULT_REVISION: dict[str, str] = {
    "beam": "100K",
    "slack": "v1",
}

#: How each corpus says it has not been seeded. The message names the script
#: rather than the directory, because an empty corpus reads as a clean run of
#: nothing: without this, ingesting an unseeded corpus reports zero turns and
#: exits successfully.
_SEED_HINT: dict[str, str] = {
    "beam": "uv run python scripts/seed/seed_beam.py --scale {revision}",
    "slack": "uv run python scripts/seed/seed_slack.py --export ~/Downloads/slack",
}


def root_for(corpus: str, revision: str) -> Path:
    """Where *revision* of *corpus* was seeded to."""
    return DATASETS_DIR / corpus / revision


def scope_for(corpus: str, revision: str, conversation_id: str) -> str:
    """The memory scope one conversation is ingested under.

    One scope per conversation, not one per corpus. The scope is what a vector
    search filters on, so sharing it would let a question about one transcript
    be answered from another -- a leak rather than a feature, and the thing
    that would make a per-conversation score meaningless.

    The revision is in the name because turn indices are only meaningful
    against the export that produced them, so re-pinning a corpus must land in
    a different scope rather than mixing with what is already there.
    """
    return f"{corpus}_{revision}_{conversation_id}"


def ingest_corpus(
    corpus: str,
    revision: str,
    *,
    ids: Sequence[str] | None = None,
    reset: bool = True,
) -> int:
    """Ingest *revision* of *corpus*. Returns the turns written.

    Pass *ids* to ingest a subset, or when the corpus has an index that is
    narrower than its directory -- BEAM's question set covers twenty
    conversations and the scale may hold more, and embedding one nobody asks
    about is spend without a measurement.
    """
    from auto_ontology.conversations.sources import create_source

    source = CORPORA.get(corpus)
    if source is None:
        known = ", ".join(sorted(CORPORA))
        raise SystemExit(f"Unknown corpus {corpus!r}. Known: {known}.")

    root = root_for(corpus, revision)
    targets = (
        list(ids)
        if ids is not None
        else create_source(source, root).get_conversations()
    )
    if not targets:
        hint = _SEED_HINT[corpus].format(revision=revision)
        raise SystemExit(f"No {corpus} data at {root}. Seed it with:\n\n    {hint}")

    return ingest_conversations(
        source,
        root,
        scope=lambda conversation_id: scope_for(corpus, revision, conversation_id),
        ids=targets,
        reset=reset,
    )


def ingest_conversations(
    source: str,
    root: str | Path,
    *,
    scope: Callable[[str], str],
    ids: Sequence[str] | None = None,
    reset: bool = True,
) -> int:
    """Write, embed and index each conversation. Returns the turns written.

    The loop itself, for callers that have already resolved where a corpus
    lives -- :func:`ingest_corpus` is the entry point that reads
    :data:`CORPORA`.

    *reset* clears a scope's vectors before reloading it, which is what makes
    a re-ingest after a parser change honest. Without it, turns that no longer
    exist keep answering searches.
    """
    from auto_ontology.conversations.ingest import run_conversation_ingest
    from auto_ontology.conversations.sources import create_source

    reader = create_source(source, root)
    targets = list(ids) if ids is not None else reader.get_conversations()

    total = 0
    for position, conversation_id in enumerate(targets, start=1):
        conversation = reader.read(conversation_id)
        database_name = scope(conversation_id)
        started = time.monotonic()
        written = run_conversation_ingest(
            conversation, database_name=database_name, reset=reset
        )
        total += written
        logger.info(
            "[%d/%d] %s: %d turns in %.1fs",
            position,
            len(targets),
            database_name,
            written,
            time.monotonic() - started,
        )
    logger.info("Ingested %d turn(s) across %d conversation(s)", total, len(targets))
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", choices=sorted(CORPORA), help="Which corpus.")
    parser.add_argument(
        "--revision",
        help="Which seeded revision: a scale for BEAM, an export for Slack. "
        "Defaults per corpus, see DEFAULT_REVISION.",
    )
    parser.add_argument(
        "--conversation",
        action="append",
        help="Only this conversation, repeatable. Default is all of them.",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Keep existing vectors instead of clearing each scope first.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    ingest_corpus(
        args.corpus,
        args.revision or DEFAULT_REVISION[args.corpus],
        ids=args.conversation,
        reset=not args.no_reset,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
