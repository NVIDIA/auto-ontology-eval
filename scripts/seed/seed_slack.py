# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin a Slack export so the benchmark has something fixed to run against.

The conversation counterpart of ``scripts/seed/seed_bird.py``, and the same
job: put the source where the rest of the system expects it, record exactly
which copy it is, and interpret nothing. Reading the export is
:class:`~auto_ontology.conversations.sources.SlackSource`'s work, and it
happens at ingest time rather than here -- nothing derived is written, in the
same way that seeding BIRD leaves SQLite files and no extracted schema.

**Why a copy at all, when the export already exists on disk.** BIRD is pinned
by a published URL, so its seed can re-download the identical bytes a year
later. A Slack export has no such address: it is one person's download, and
once that directory is cleared there is nothing to go back to. The copy is
the only pin available, and ``source.json`` records the hash so a later run
can tell whether it is reading the corpus a score came from.

Usage::

    uv run python scripts/seed/seed_slack.py --export ~/Downloads/slack
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Where the pinned export lands. Named for the source rather than for a
#: revision: there is one export, and a second cut of it would be a second
#: directory beside this one rather than a silent overwrite.
DATA_DIR = Path(__file__).resolve().parents[2] / "datasets" / "slack"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export",
        default="~/Downloads/slack",
        help="The export directory: one subdirectory of one JSON per channel.",
    )
    parser.add_argument("--revision", default="v1", help="Output subdirectory.")
    parser.add_argument(
        "--channel",
        action="append",
        help="Only this channel, repeatable. Default is all of them.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    export = Path(args.export).expanduser()
    files = sorted(export.glob("*/*.json"))
    if args.channel:
        wanted = set(args.channel)
        files = [path for path in files if path.stem in wanted]
    if not files:
        raise SystemExit(f"No channel files under {export}")

    target = DATA_DIR / args.revision / "export"
    target.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256()
    channels: list[dict[str, Any]] = []

    for path in files:
        raw = path.read_bytes()
        digest.update(raw)

        # The source's own layout, kept exactly: ``<channel>/<channel>.json``
        # is what ``SlackSource`` globs for, so the pinned copy and the
        # original are interchangeable and a run can point at either.
        destination = target / path.parent.name / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)

        payload = json.loads(raw.decode("utf-8"))
        records = payload.get("records") or []
        channel = payload.get("collection") or path.stem
        channels.append(
            {
                "conversation_id": channel,
                "threads_in_export": len(records),
                "bytes": len(raw),
            }
        )
        logger.info("%s: %d thread(s), %d bytes", channel, len(records), len(raw))

    (DATA_DIR / args.revision / "source.json").write_text(
        json.dumps(
            {
                "export": str(export),
                "sha256": digest.hexdigest(),
                "seeded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "revision": args.revision,
                "channels": channels,
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    logger.info(
        "Pinned %d channel(s), %d thread(s) to %s",
        len(channels),
        sum(row["threads_in_export"] for row in channels),
        target,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
