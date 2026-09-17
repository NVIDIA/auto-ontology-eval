"""Find BIRD train examples structurally matching the regressed dev questions.

The regressions have a shape problem, not a vocabulary problem: the generator
picks the wrong join key, projects extra columns, or answers from one table
where gold bridges two. Train examples that share the *skeleton* of a
regressed question therefore demonstrate the convention it got wrong, and can
serve as few-shot exemplars -- the databases differ, which is a feature here,
because it forces the example to teach form rather than content.

A skeleton erases every identifier, literal and alias, keeping only keywords,
functions and the slot kind (T for table, C for column, L/N for literals). Two
queries with the same skeleton answer differently-worded questions with the
same SQL construction.
"""

import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

KEYWORDS = {
    "select", "distinct", "from", "where", "group", "by", "order", "having",
    "limit", "join", "inner", "left", "outer", "on", "and", "or", "not", "in",
    "is", "null", "like", "between", "as", "asc", "desc", "case", "when",
    "then", "else", "end", "cast", "real", "integer", "union", "all", "exists",
    "count", "sum", "avg", "max", "min", "iif", "strftime", "substr", "round",
    "julianday", "date", "datetime", "length", "abs", "coalesce", "div",
}
ALIAS_PREFIX = re.compile(r"\b[Tt]\d+\s*\.|\b[a-z]{1,3}\d?\s*\.", re.ASCII)
AS_ALIAS = re.compile(r"\bas\s+[`\"\[]?\w+[`\"\]]?", re.IGNORECASE)
STRINGS = re.compile(r"'[^']*'")
BACKTICKED = re.compile(r"[`\"\[][^`\"\]]+[`\"\]]")
NUMBERS = re.compile(r"\b\d+(\.\d+)?\b")
WORD = re.compile(r"[A-Za-z_]\w*")


def skeleton(sql: object) -> str:
    s = " ".join(str(sql).split())
    s = STRINGS.sub(" L ", s)
    s = BACKTICKED.sub(" C ", s)
    s = NUMBERS.sub(" N ", s)
    # Aliases carry no structure; drop the declaration and the qualifier.
    s = AS_ALIAS.sub(" ", s)
    s = ALIAS_PREFIX.sub(" ", s)
    out: list[str] = []
    prev = ""
    for tok in re.findall(r"[A-Za-z_]\w*|[(),*=<>!+/-]|N|L|C", s):
        low = tok.lower()
        if low in KEYWORDS:
            out.append(low)
        elif tok in {"N", "L", "C"}:
            out.append(tok)
        elif WORD.fullmatch(tok):
            out.append("T" if prev in {"from", "join"} else "C")
        else:
            out.append(tok)
        if WORD.fullmatch(tok):
            prev = low
        elif tok not in {"(", ")"}:
            prev = ""
    # "as" only ever introduced an alias or a cast type; keep casts readable.
    return " ".join(t for t in out if t != "as")


def main() -> None:
    old = pd.read_csv(
        ROOT / "output/bird_bedrock-claude-opus-4-8_scores.csv"
    ).set_index("question_id")
    new = pd.read_csv(
        ROOT / "output/bird_bedrock-claude-opus-4-8_fix2_scores.csv"
    ).set_index("question_id")
    patterns = pd.read_csv(ROOT / "output/regression_rule_patterns.csv").set_index(
        "question_id"
    )

    common = old.index.intersection(new.index)
    o = old.loc[common, "bird_ex_match"].astype(int)
    n = new.loc[common, "bird_ex_match"].astype(int)
    regressed = list(common[(o == 1) & (n == 0)])

    train = json.loads((ROOT / "datasets/bird/train/train.json").read_text())
    train_skel: dict[str, list[dict]] = defaultdict(list)
    for ex in train:
        train_skel[skeleton(ex["SQL"])].append(ex)
    print(f"train examples: {len(train)}   distinct skeletons: {len(train_skel)}")

    keys = list(train_skel)
    rows = []
    for qid in regressed:
        skel = skeleton(old.loc[qid, "expected_sql"])
        exact = train_skel.get(skel, [])
        best_key, best_ratio = None, 0.0
        if not exact:
            # Only compare against skeletons of similar length; SequenceMatcher
            # over 6k candidates per question is otherwise the whole runtime.
            target_len = len(skel)
            for k in keys:
                if abs(len(k) - target_len) > target_len * 0.25:
                    continue
                r = SequenceMatcher(None, skel, k).ratio()
                if r > best_ratio:
                    best_key, best_ratio = k, r
        rows.append(
            {
                "question_id": qid,
                "pattern": patterns.loc[qid, "pattern"] if qid in patterns.index else "",
                "skeleton": skel,
                "exact_twins": len(exact),
                "best_ratio": 1.0 if exact else round(best_ratio, 3),
                "best_key": skel if exact else best_key,
            }
        )

    df = pd.DataFrame(rows)
    print(f"\nregressed questions: {len(df)}")
    print(f"  with an EXACT structural twin in train : {(df.exact_twins > 0).sum()}")
    print(f"  closest twin >= 0.90 similarity        : {(df.best_ratio >= 0.90).sum()}")
    print(f"  closest twin >= 0.80 similarity        : {(df.best_ratio >= 0.80).sum()}")
    print(f"  median exact twins (where any)         : "
          f"{int(df.loc[df.exact_twins > 0, 'exact_twins'].median() or 0)}")

    print("\n=== exact-twin coverage by regression pattern ===")
    g = df.groupby("pattern").agg(
        questions=("question_id", "size"),
        with_exact_twin=("exact_twins", lambda s: int((s > 0).sum())),
        total_twins=("exact_twins", "sum"),
    )
    g["coverage %"] = (100 * g.with_exact_twin / g.questions).round(0)
    print(g.sort_values("questions", ascending=False).to_string())

    df.to_csv(ROOT / "output/train_structural_twins.csv", index=False)
    print("\nwrote output/train_structural_twins.csv")

    print("\n" + "=" * 100)
    print("EXEMPLARS: train examples sharing a regressed question's skeleton")
    print("=" * 100)
    shown = 0
    for _, r in df[df.exact_twins > 0].sort_values("exact_twins", ascending=False).iterrows():
        if shown >= 5:
            break
        qid = r.question_id
        twins = train_skel[r.skeleton]
        print(f"\n--- q{qid} [{r.pattern}] — {r.exact_twins} train twins")
        print(f"    dev question : {str(old.loc[qid, 'question']).strip()[:110]}")
        print(f"    dev gold     : {' '.join(str(old.loc[qid, 'expected_sql']).split())[:135]}")
        print(f"    now produces : {' '.join(str(new.loc[qid, 'returned_sql']).split())[:135]}")
        print(f"    skeleton     : {r.skeleton[:135]}")
        for ex in twins[:2]:
            print(f"      train [{ex['db_id']}] {ex['question'].strip()[:100]}")
            print(f"        SQL: {' '.join(ex['SQL'].split())[:130]}")
        shown += 1


if __name__ == "__main__":
    main()
