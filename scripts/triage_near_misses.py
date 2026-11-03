"""Execute gold vs agent SQL for failed BIRD-Interact tasks and classify the diff.

Usage:
    .venv/bin/python scripts/triage_near_misses.py RESULTS.jsonl [--ids a,b,c] [--out out.json]

Needs the task Postgres on 127.0.0.1:5433 (root/123123). Read-only, with a statement timeout.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import LiteralString, cast

import psycopg

GOLD_PATH = Path(__file__).resolve().parents[1] / "datasets/bird_interact_full/bird_interact_data_with_gt.jsonl"
DSN = "postgresql://root:123123@127.0.0.1:5433/{db}"
TIMEOUT_MS = 60_000


def norm(v: object) -> object:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float, Decimal)):
        return round(float(v), 2)
    if isinstance(v, str):
        s = v.strip()
        try:
            return round(float(s.rstrip("%")), 2)
        except ValueError:
            return s.lower()
    return str(v)


def run(db: str, sql: str) -> tuple[list[str], list[tuple]] | str:
    try:
        with psycopg.connect(DSN.format(db=db), connect_timeout=10, autocommit=False) as conn:
            conn.read_only = True
            conn.execute(cast(LiteralString, f"SET statement_timeout = {TIMEOUT_MS}"))
            cur = conn.execute(cast(LiteralString, sql.strip().rstrip(";")))
            cols = [d.name for d in cur.description or ()]
            rows = [tuple(norm(x) for x in r) for r in cur.fetchall()]
            conn.rollback()
            return cols, rows
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"


def best_col_map(g_rows: list[tuple], a_rows: list[tuple]) -> dict[int, int]:
    """For each gold column find the agent column with the largest multiset overlap."""
    if not g_rows or not a_rows:
        return {}
    gc, ac = len(g_rows[0]), len(a_rows[0])
    mapping: dict[int, int] = {}
    for i in range(gc):
        gset = Counter(r[i] for r in g_rows)
        best, best_j = 0.0, -1
        for j in range(ac):
            aset = Counter(r[j] for r in a_rows)
            inter = sum((gset & aset).values())
            score = inter / max(len(g_rows), 1)
            if score > best:
                best, best_j = score, j
        if best_j >= 0 and best > 0.5:
            mapping[i] = best_j
    return mapping


def classify(g: tuple[list[str], list[tuple]], a: tuple[list[str], list[tuple]]) -> dict:
    gcols, grows = g
    acols, arows = a
    info: dict = {
        "gold_rows": len(grows),
        "agent_rows": len(arows),
        "gold_cols": gcols,
        "agent_cols": acols,
    }
    tags: list[str] = []
    if len(gcols) != len(acols):
        tags.append("column_count_differs")
    if len(grows) != len(arows):
        tags.append("row_count_differs")
    # positional full-row comparison
    gset, aset = Counter(grows), Counter(arows)
    if gset == aset:
        tags.append("identical")
    else:
        inter = sum((gset & aset).values())
        info["row_recall"] = round(inter / max(len(grows), 1), 2)
        info["row_precision"] = round(inter / max(len(arows), 1), 2)
    # column permutation / mapping
    cmap = best_col_map(grows, arows)
    info["col_map_gold_to_agent"] = cmap
    if cmap and len(cmap) == len(gcols) and any(i != j for i, j in cmap.items()):
        tags.append("column_order_or_position_differs")
    unmatched = [gcols[i] for i in range(len(gcols)) if i not in cmap]
    if unmatched:
        info["gold_cols_with_no_matching_agent_col"] = unmatched
        tags.append("value_mismatch_in_some_columns")
    # all-null agent columns
    for j, c in enumerate(acols):
        if arows and all(r[j] is None for r in arows):
            tags.append(f"agent_col_all_null:{c}")
    # key overlap on first matched column
    if grows and arows and 0 in cmap:
        gk = {r[0] for r in grows}
        ak = {r[cmap[0]] for r in arows}
        info["key_overlap"] = round(len(gk & ak) / max(len(gk), 1), 2)
        if gk == ak:
            tags.append("same_keys")
    info["tags"] = tags
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--ids", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    gold = {}
    for line in GOLD_PATH.open():
        d = json.loads(line)
        gold[d["instance_id"]] = d
    res = {}
    for line in Path(args.results).open():
        d = json.loads(line)
        res[d["instance_id"]] = d

    ids = [i for i in args.ids.split(",") if i] or [k for k, v in res.items() if not v.get("phase1_passed")]
    out: dict[str, dict] = {}
    for iid in ids:
        r, g = res[iid], gold[iid]
        db = g["selected_database"]
        gres = run(db, g["sol_sql"][0])
        agent_sql = r.get("p1_debug_sql") or r.get("p1_sql")
        ares = run(db, agent_sql) if agent_sql else "no sql"
        if isinstance(gres, str):
            entry = {"status": "gold_error", "error": gres}
        elif isinstance(ares, str):
            entry = {"status": "agent_error", "error": ares}
        else:
            entry = {"status": "compared", **classify(gres, ares)}
        out[iid] = entry
        print(iid, entry["status"], entry.get("tags", entry.get("error")), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
