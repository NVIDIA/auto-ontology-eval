"""
Remove ingested documents from LanceDB.

Usage:
    python remove.py --list                          # show what is in the table
    python remove.py Get_Started_With_Smallpdf.pdf   # remove one document
    python remove.py a.pdf data/b.pdf                # remove several
    python remove.py --all                           # drop the whole table
"""

import argparse
from collections import Counter

import config


def show(counts: Counter) -> None:
    print(f"\n{config.TABLE_NAME} @ {config.LANCEDB_URI}")
    for raw, n in sorted(counts.items(), key=lambda kv: config.source_labels(kv[0])[0]):
        name, path = config.source_labels(raw)
        print(f"  {name}  —  {n} chunk(s)")
        print(f"      {path}")
    print(f"\n{len(counts)} document(s), {sum(counts.values())} chunk(s) total.\n")


def remove_all(db, counts: Counter, assume_yes: bool) -> None:
    total = sum(counts.values())
    if not assume_yes:
        reply = input(f"Delete all {len(counts)} document(s) / {total} chunk(s)? [y/N] ")
        if reply.strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return
    db.drop_table(config.TABLE_NAME)
    print(f"Dropped table '{config.TABLE_NAME}'. Run `python ingest.py` to rebuild it.")


def remove_targets(table, counts: Counter, targets: list[str]) -> None:
    doomed: dict[str, int] = {}
    unmatched: list[str] = []

    for target in targets:
        hits = {raw: n for raw, n in counts.items() if config.matches(target, raw)}
        if hits:
            doomed.update(hits)
        else:
            unmatched.append(target)

    if unmatched:
        print("Not found in the table: " + ", ".join(unmatched))
        print("Run `python remove.py --list` to see the available documents.")
    if not doomed:
        return

    for raw, n in doomed.items():
        print(f"Removing {config.source_labels(raw)[0]} ({n} chunk(s)) ...")

    table.delete(config.source_predicate(doomed))

    print(f"Removed {sum(doomed.values())} chunk(s). {table.count_rows()} chunk(s) left.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="*", help="File names or paths as they were ingested")
    parser.add_argument("--list", action="store_true", help="List the ingested documents and exit")
    parser.add_argument("--all", action="store_true", help="Drop the whole table")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt for --all")
    args = parser.parse_args()

    if not config.has_ingested_data():
        raise SystemExit(f"No table '{config.TABLE_NAME}' at {config.LANCEDB_URI}. Nothing to remove.")

    db = config.connect()
    table = db.open_table(config.TABLE_NAME)
    counts = config.chunk_counts(table)

    if args.list or not (args.all or args.targets):
        show(counts)
        return

    if args.all:
        remove_all(db, counts, args.yes)
        return

    remove_targets(table, counts, args.targets)


if __name__ == "__main__":
    main()
