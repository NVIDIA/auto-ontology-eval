"""Why does a lenient-identical result still fail the official scorer?

Replays the official ex_base preprocessing (remove_round/remove_distinct on SQL,
2-decimal rounding, exact tuple/row-order comparison) and names the first strict
difference: column_count, row_count, case/whitespace, order, decimals, or value.
Usage: .venv/bin/python scripts/strict_diff.py RESULTS.jsonl id1,id2,...
"""

import json
import sys
from decimal import Decimal
from pathlib import Path

import psycopg

from types import ModuleType, SimpleNamespace  # noqa: E402

# db_utils imports psycopg2 + shared.config at module level; only the pure helpers are needed here.
for _name in ("psycopg2", "psycopg2.pool"):
    _m = ModuleType(_name)
    _m.OperationalError = Exception  # type: ignore[attr-defined]
    _m.ThreadedConnectionPool = object  # type: ignore[attr-defined]
    sys.modules[_name] = _m
sys.modules["psycopg2"].pool = sys.modules["psycopg2.pool"]  # type: ignore[attr-defined]
_cfg = ModuleType("shared.config")
_cfg.settings = SimpleNamespace()  # type: ignore[attr-defined]
sys.modules["shared.config"] = _cfg
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/BIRD-Interact/BIRD-Interact-ADK"))
from shared.db_utils import preprocess_results, remove_comments, remove_distinct, remove_round  # noqa: E402

GOLD = Path(__file__).resolve().parents[1] / "datasets/bird_interact_full/bird_interact_data_with_gt.jsonl"
DSN = "postgresql://root:123123@127.0.0.1:5433/{db}"


def run(db, sql):
    sql = remove_round(remove_distinct(remove_comments([sql])))[0]
    with psycopg.connect(DSN.format(db=db), connect_timeout=10) as c:
        c.read_only = True
        c.execute("SET statement_timeout = 60000")
        return preprocess_results(c.execute(sql.strip().rstrip(";")).fetchall())


gold = {json.loads(line)["instance_id"]: json.loads(line) for line in GOLD.open()}
res = {json.loads(line)["instance_id"]: json.loads(line) for line in open(sys.argv[1])}
for iid in sys.argv[2].split(","):
    r, g = res[iid], gold[iid]
    try:
        gr = run(g["selected_database"], g["sol_sql"][0])
        ar = run(g["selected_database"], r.get("p1_debug_sql") or r["p1_sql"])
    except Exception as e:  # noqa: BLE001
        print(iid, "ERR", str(e)[:100])
        continue
    order = g["conditions"].get("order")
    why = []

    def tnorm(rows, lower=False, perm=False):
        out = []
        for t in rows:
            u = []
            for x in t:
                if isinstance(x, (int, float, Decimal)) and not isinstance(x, bool):
                    x = round(float(x), 2)
                elif lower and isinstance(x, str):
                    x = x.strip().lower()
                u.append(x)
            out.append(tuple(sorted(map(str, u))) if perm else tuple(u))
        return out

    same_shape = bool(gr) and bool(ar) and len(gr) == len(ar) and len(gr[0]) == len(ar[0])
    if gr == ar:
        why.append("EXACT MATCH")
    elif same_shape and tnorm(gr) == tnorm(ar):
        why.append("TYPE: float/Decimal mismatch only")
    elif same_shape and set(gr) == set(ar):
        why.append("ORDER: same rows, different order (ties?)")
    elif same_shape and tnorm(gr, lower=True) == tnorm(ar, lower=True):
        why.append("CASE: values differ only in case/whitespace")
    elif same_shape and sorted(tnorm(gr, lower=True), key=str) == sorted(tnorm(ar, lower=True), key=str):
        why.append("CASE+ORDER")
    elif same_shape and sorted(tnorm(gr, True, True)) == sorted(tnorm(ar, True, True)):
        why.append("COLUMNS: same values, columns permuted")
    elif gr and ar and len(gr[0]) != len(ar[0]):
        why.append(f"SHAPE: column_count gold={len(gr[0])} agent={len(ar[0])}")
    elif len(gr) != len(ar):
        why.append(f"ROWS: row_count gold={len(gr)} agent={len(ar)}")
    else:
        why.append("VALUES differ")
    print(f"{iid:36s} order_required={order} :: {'; '.join(why)}"[:330])
