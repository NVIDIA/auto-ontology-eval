"""Reverse index from stored values to the columns that hold them.

The pipeline can already check a literal against the column the model chose --
``db_probe/literal_check.py`` probes that column's distinct values, and
``empty_result_value_repair`` retries when a query comes back empty. Neither
answers the question that actually goes wrong most often: the model filtered
``MailCity = 'San Joaquin'`` when the value also sits in ``City``, the query
returned rows, and nothing objected. Answering "which columns contain this
phrase?" needs the opposite direction of lookup, which is what this builds.

``build`` walks every text column of every database and records its distinct
values with the columns holding them. Columns that are effectively free text or
unique keys are skipped: a value seen once in a 60k-row description column is
never the target of a question filter, and indexing it only adds noise.

``validate`` asks what the index would be worth before any of it is wired into
a prompt. Two numbers matter and they measure different things. *Coverage* is
whether the index knows gold's value at all -- an upper bound set by the build
filters. *Recall from the question* is whether phrases taken from the question
text alone surface gold's column, which is what the prompt would actually get,
and is bounded by how the question words relate to the stored value.
"""

import argparse
import csv
import json
import re
import sqlite3
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_ROOT = ROOT / "datasets/bird/dev"
INDEX = ROOT / "output/value_index.sqlite"

# A column holding more distinct values than this is a key or free text, not a
# vocabulary a question filters on by name. Set above the obvious cut-off
# because timestamps and short codes are high-cardinality yet routinely
# filtered on by name ("the Q2 time 1:40").
MAX_DISTINCT = 60_000
# Longer than this is prose; questions quote labels, not paragraphs.
MAX_VALUE_LEN = 80

STOPWORDS = {
    "a", "all", "an", "and", "any", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "give", "had", "has", "have", "how", "in", "is", "it", "its", "list",
    "many", "me", "much", "name", "of", "on", "or", "please", "show", "state", "tell",
    "that", "the", "their", "there", "they", "this", "to", "was", "were", "what", "when",
    "which", "who", "whose", "with", "you", "your",
}


def norm(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def variants(text: str) -> set[str]:
    """Extra lookup keys for a value, covering the two ways a question names a
    stored value without matching it character for character.

    *Format*: "0:01:40" in the question against a stored "1:40", or "40"
    against a zero-padded "0040". Leading zeros carry no meaning in either
    direction, so both sides collapse to the same canonical form.

    *Qualifier*: a stored "High Schools (Public)" is named in questions as
    "high schools"; the parenthetical is a schema-side distinction the asker
    has no reason to know about.
    """
    keys = set()
    value = norm(text)
    if re.fullmatch(r"[\d:]+", value) and value:
        parts = [p.lstrip("0") or "0" for p in value.split(":")]
        while len(parts) > 1 and parts[0] == "0":
            parts.pop(0)
        canonical = ":".join(parts)
        if canonical != value:
            keys.add(canonical)
    if "(" in value:
        stripped = value.split("(")[0].strip()
        if len(stripped) > 2:
            keys.add(stripped)
    return keys - {value}


def singular(phrase: str) -> str:
    """Crude de-pluralisation, enough for "continuation schools" -> the stored
    "Continuation School". Questions name a category in the plural far more
    often than a column stores it that way."""
    words = phrase.split()
    if words and len(words[-1]) > 3 and words[-1].endswith("s") and not words[-1].endswith("ss"):
        words[-1] = words[-1][:-1]
    return " ".join(words)


def databases() -> list[str]:
    return sorted(p.name for p in DB_ROOT.iterdir() if (p / f"{p.name}.sqlite").is_file())


def build(verbose: bool = True) -> None:
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    if INDEX.exists():
        INDEX.unlink()
    out = sqlite3.connect(INDEX)
    out.executescript(
        """
        CREATE TABLE value_column (
            db TEXT, tbl TEXT, col TEXT,
            value_norm TEXT, value_raw TEXT, n_rows INTEGER
        );
        CREATE TABLE skipped (db TEXT, tbl TEXT, col TEXT, reason TEXT, detail INTEGER);
        -- Word-level postings, because gold filters are often
        -- `LIKE '%Riverside%'` against a stored "Riverside Unified": the
        -- question names a word inside the value, never the value itself.
        CREATE TABLE word_column (
            db TEXT, word TEXT, tbl TEXT, col TEXT, sample_value TEXT, n_values INTEGER
        );
        -- Alternate spellings of the same stored value, kept apart from
        -- value_column so an exact match always outranks a normalised one.
        CREATE TABLE alias_key (
            db TEXT, key TEXT, tbl TEXT, col TEXT, value_raw TEXT, n_rows INTEGER
        );
        """
    )
    totals = Counter()
    for db in databases():
        path = DB_ROOT / db / f"{db}.sqlite"
        src = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=60)
        src.text_factory = lambda b: b.decode("utf-8", "replace")
        tables = [r[0] for r in src.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        rows: list[tuple] = []
        word_rows: list[tuple] = []
        alias_rows: list[tuple] = []
        for tbl in tables:
            if tbl.startswith("sqlite_"):
                continue
            for info in src.execute(f'PRAGMA table_info("{tbl}")').fetchall():
                col = info[1]
                try:
                    n = src.execute(
                        f'SELECT COUNT(DISTINCT "{col}") FROM "{tbl}" '
                        f'WHERE "{col}" IS NOT NULL AND LENGTH("{col}") <= {MAX_VALUE_LEN}'
                    ).fetchone()[0]
                except Exception:
                    continue
                if n == 0:
                    out.execute("INSERT INTO skipped VALUES (?,?,?,?,?)", (db, tbl, col, "empty", 0))
                    continue
                if n > MAX_DISTINCT:
                    out.execute(
                        "INSERT INTO skipped VALUES (?,?,?,?,?)", (db, tbl, col, "high-cardinality", n)
                    )
                    totals["columns skipped"] += 1
                    continue
                try:
                    pairs = src.execute(
                        f'SELECT "{col}", COUNT(*) FROM "{tbl}" '
                        f'WHERE "{col}" IS NOT NULL AND LENGTH("{col}") <= {MAX_VALUE_LEN} '
                        f'GROUP BY "{col}"'
                    ).fetchall()
                except Exception:
                    continue
                totals["columns indexed"] += 1
                words: dict[str, tuple[str, int]] = {}
                for raw, count in pairs:
                    text = str(raw)
                    if not text.strip():
                        continue
                    rows.append((db, tbl, col, norm(text), text, count))
                    alias_rows.extend(
                        (db, key, tbl, col, text, count) for key in variants(text)
                    )
                    tokens = re.findall(r"[\w'&/.-]+", norm(text))
                    # Only multi-word values need postings; for a single-word
                    # value the exact index already answers the query.
                    if 1 < len(tokens) <= 8:
                        for w in set(tokens):
                            if len(w) < 2 or w in STOPWORDS:
                                continue
                            sample, n = words.get(w, (text, 0))
                            words[w] = (sample, n + 1)
                word_rows.extend(
                    (db, w, tbl, col, sample, n) for w, (sample, n) in words.items()
                )
        out.executemany("INSERT INTO value_column VALUES (?,?,?,?,?,?)", rows)
        out.executemany("INSERT INTO word_column VALUES (?,?,?,?,?,?)", word_rows)
        out.executemany("INSERT INTO alias_key VALUES (?,?,?,?,?,?)", alias_rows)
        out.commit()
        totals["values"] += len(rows)
        totals["postings"] += len(word_rows)
        totals["aliases"] += len(alias_rows)
        if verbose:
            print(f"  {db:<28} {len(rows):>8,} values")
        src.close()
    out.executescript(
        "CREATE INDEX idx_value ON value_column(db, value_norm);"
        "CREATE INDEX idx_col ON value_column(db, tbl, col);"
        "CREATE INDEX idx_word ON word_column(db, word);"
        "CREATE INDEX idx_alias ON alias_key(db, key);"
    )
    out.commit()
    out.close()
    size = INDEX.stat().st_size / 1e6
    print(
        f"\nindexed {totals['values']:,} values over {totals['columns indexed']} columns "
        f"({totals['columns skipped']} skipped as high-cardinality) -> {INDEX.name}, {size:.1f} MB"
    )


def ngrams(text: str, n_max: int = 5) -> list[str]:
    """Candidate phrases a question might be naming a stored value with."""
    # Punctuation is kept inside a token ("St.", "5-17", "AT&T") but stripped
    # at its edges, or a phrase ending a sentence never matches its value.
    words = [w.strip(".,;:!?'\"()") for w in re.findall(r"[\w'&/.-]+", text)]
    words = [w for w in words if w]
    out = []
    for size in range(1, n_max + 1):
        for i in range(len(words) - size + 1):
            phrase = " ".join(words[i : i + size])
            if all(w.lower() in STOPWORDS for w in words[i : i + size]):
                continue
            if len(phrase) < 2:
                continue
            out.append(norm(phrase))
    return out


def lookup(con: sqlite3.Connection, db: str, phrases: list[str], limit: int = 8) -> list[tuple]:
    """Anchors for a question: (phrase, table, column, stored value, rows, kind).

    ``kind`` is "value" when the phrase names the stored value outright and
    "contains" when the phrase is only a word inside it -- the difference
    between an equality filter and a LIKE, and between a statement that is true
    and one that is not.

    Longer phrases first -- 'Continuation School' is evidence, 'school' is not.
    Within a phrase, commoner values first: a value spanning many rows is a
    category, which is what questions filter on, while a single-row match is
    usually a coincidence in unrelated data (the surname 'Free' for the phrase
    "free meals"). Single-row matches are dropped outright when the same phrase
    has a substantial match elsewhere.
    """
    exact: list[tuple] = []
    partial: list[tuple] = []
    seen = set()

    def add(bucket, phrase, tbl, col, raw, n, kind):
        key = (tbl.lower(), col.lower(), norm(raw))
        if key not in seen:
            seen.add(key)
            bucket.append((phrase, tbl, col, raw, n, kind))

    ordered = sorted(set(phrases), key=len, reverse=True)
    for phrase in ordered:
        keys = {phrase, singular(phrase)}
        # The question's own wording gets normalised the same way stored values
        # were, so "0:01:40" reaches a stored "1:40" from either side.
        keys |= variants(phrase) | variants(singular(phrase))
        for key in keys:
            for tbl, col, raw, n in con.execute(
                "SELECT tbl, col, value_raw, n_rows FROM value_column "
                "WHERE db = ? AND value_norm = ? ORDER BY n_rows DESC LIMIT 6",
                (db, key),
            ):
                add(exact, phrase, tbl, col, raw, n, "value")
            for tbl, col, raw, n in con.execute(
                "SELECT tbl, col, value_raw, n_rows FROM alias_key "
                "WHERE db = ? AND key = ? ORDER BY n_rows DESC LIMIT 4",
                (db, key),
            ):
                add(exact, phrase, tbl, col, raw, n, "value")
    # Word postings only for phrases that found no value of their own, so a
    # phrase that names a value exactly is never diluted by the values that
    # merely contain one of its words.
    if len(exact) < limit:
        matched = {p for p, *_ in exact}
        for phrase in ordered:
            if phrase in matched or " " in phrase or len(phrase) < 4:
                continue
            for tbl, col, raw, n in con.execute(
                "SELECT tbl, col, sample_value, n_values FROM word_column "
                "WHERE db = ? AND word = ? ORDER BY n_values DESC LIMIT 4",
                (db, singular(phrase)),
            ):
                add(partial, phrase, tbl, col, raw, n, "contains")
    # A one-row hit beside a substantial one for the same phrase is noise.
    strong = {h[0] for h in exact if h[4] >= 5}
    exact = [h for h in exact if h[4] > 1 or h[0] not in strong]
    return (exact + partial)[:limit]


def validate(targets_path: Path, limit: int) -> None:
    con = sqlite3.connect(f"file:{INDEX}?mode=ro", uri=True)
    targets = json.loads(targets_path.read_text())
    cov = Counter()
    noise = []
    misses = []
    for t in targets:
        db, want_col, want_val = t["db"], t["gold_column"].lower(), t["gold_value"]
        # Gold's filter may be `LIKE '%Riverside%'`, so a column whose values
        # merely contain the phrase counts as holding it.
        in_index = con.execute(
            "SELECT tbl, col FROM value_column WHERE db = ? AND value_norm = ?",
            (db, norm(want_val)),
        ).fetchall() + con.execute(
            "SELECT tbl, col FROM word_column WHERE db = ? AND word = ?",
            (db, norm(want_val)),
        ).fetchall()
        if not in_index:
            cov["value not in index at all"] += 1
            misses.append((t, "value absent"))
            continue
        if not any(c.lower() == want_col for _, c in in_index):
            cov["value indexed, but not under gold's column"] += 1
            misses.append((t, "column absent"))
            continue
        cov["gold column+value present in index"] += 1

        anchors = lookup(con, db, ngrams(t["question"] + " " + t.get("evidence", "")), limit)
        noise.append(len(anchors))
        # A word posting counts as a hit too: pointing at gold's column with a
        # value containing the phrase is what a LIKE filter needs.
        if any(
            c.lower() == want_col and (norm(v) == norm(want_val) or norm(want_val) in norm(v))
            for _, _, c, v, _, _ in anchors
        ):
            cov["  ...and reachable from the question text"] += 1
        else:
            cov["  ...but question text does not surface it"] += 1
            misses.append((t, "not surfaced"))

    n = len(targets)
    print(f"validation over {n} filter-linking failures\n")
    for label in ("gold column+value present in index", "value indexed, but not under gold's column",
                  "value not in index at all", "  ...and reachable from the question text",
                  "  ...but question text does not surface it"):
        if label in cov:
            print(f"  {cov[label]:>3} ({100*cov[label]/n:4.1f}%)  {label}")
    if noise:
        print(f"\n  anchors offered per question: mean {sum(noise)/len(noise):.1f}, max {max(noise)}")
    print("\n  sample misses:")
    for t, why in misses[:6]:
        print(f"    q{t['question_id']} [{t['db']}] {why}: gold {t['gold_column']} = {t['gold_value']!r}")
        print(f"      question: {t['question'][:110]}")


def label_phrases(text: str) -> list[str]:
    """Phrases a question presents as a stored label: quoted spans, and runs of
    capitalised words. Used only for negative evidence, where a wrong guess
    costs more than a missing one, so the bar is deliberately high.

    Only two forms qualify: a quoted span, and a run of two or more capitalised
    words. Single capitalised words are excluded because English capitalises
    them for reasons of its own -- after a full stop, at the start of a
    sentence -- and treating "Please" or "Math" as a label the database fails
    to store is worse than saying nothing.
    """
    out = [m.group(1) for m in re.finditer(r"['\"]([^'\"]{2,60})['\"]", text)]
    out += [
        m.group(0)
        for m in re.finditer(r"\b[A-Z][\w.-]*(?:\s+[A-Z][\w.-]*)+\b", text)
        if len(m.group(0)) > 5
    ]
    return out


@lru_cache(maxsize=None)
def _schema_words(con: sqlite3.Connection, db: str) -> frozenset[str]:
    """Table and column names of a database, as normalised words.

    A question naming a table ("the FRPM table", "the Street column") is naming
    schema, not a stored value. Absent from the value index it certainly is,
    but reporting it as stored nowhere would be false where it matters most.
    """
    words: set[str] = set()
    for (name,) in con.execute(
        "SELECT DISTINCT tbl FROM value_column WHERE db = ? "
        "UNION SELECT DISTINCT col FROM value_column WHERE db = ?", (db, db)
    ):
        words.add(norm(name))
        words.update(re.findall(r"[a-z0-9]+", norm(name)))
    return frozenset(words)


def anchors(out_path: Path, limit: int, max_contains: int, max_absent: int) -> None:
    """Write the anchors each dev question would receive, for prompt injection."""
    con = sqlite3.connect(f"file:{INDEX}?mode=ro", uri=True)
    questions = json.loads((ROOT / "datasets/bird/evaluation.json").read_text())
    rows, stats = [], Counter()
    for q in questions:
        db = q["db_id"]
        text = q["question"] + " " + (q.get("evidence") or "")
        hits = lookup(con, db, ngrams(text), limit)
        kept, n_contains = [], 0
        for hit in hits:
            if hit[5] == "contains":
                n_contains += 1
                if n_contains > max_contains:
                    continue
            kept.append(hit)
        hits = kept
        for rank, (phrase, tbl, col, raw, n, kind) in enumerate(hits, start=1):
            rows.append(
                {
                    "question_id": q["question_id"], "rank": rank, "kind": kind,
                    "phrase": phrase, "tbl": tbl, "col": col,
                    "stored_value": raw, "n_rows": n,
                }
            )
        stats["anchors"] += len(hits)
        # Negative evidence -- a label the question states that the database
        # stores nowhere -- is off by default (--max-absent). It would stop an
        # invented filter value, but extracting labels from question text
        # yields mostly noun-phrase fragments ("National Center" out of
        # "National Center for Educational Statistics"), and it reached only 2%
        # of questions, so the section would assert falsehoods more often than
        # it prevented one.
        if max_absent <= 0:
            continue
        found = {norm(p) for p, *_ in hits}
        schema = _schema_words(con, db)
        absent = []
        for phrase in label_phrases(q["question"]):
            key = norm(phrase)
            if key in found or any(key in f or f in key for f in found):
                continue
            if key in schema or all(w in schema for w in re.findall(r"[a-z0-9]+", key)):
                continue
            if con.execute(
                "SELECT 1 FROM value_column WHERE db = ? AND value_norm = ? LIMIT 1", (db, key)
            ).fetchone():
                continue
            absent.append(phrase)
        for rank, phrase in enumerate(dict.fromkeys(absent), start=1):
            if rank > max_absent:
                break
            rows.append(
                {
                    "question_id": q["question_id"], "rank": rank, "kind": "absent",
                    "phrase": phrase, "tbl": "", "col": "", "stored_value": "", "n_rows": 0,
                }
            )
            stats["absent notes"] += 1
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(
        f"{stats['anchors']:,} anchors and {stats['absent notes']:,} 'not stored' notes "
        f"over {len(questions)} questions -> {out_path.relative_to(ROOT)}\n"
        f"  mean {stats['anchors']/len(questions):.1f} anchors per question"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    v = sub.add_parser("validate")
    v.add_argument("--targets", type=Path, default=ROOT / "output/value_link_targets.json")
    v.add_argument("--limit", type=int, default=8)
    a = sub.add_parser("anchors")
    a.add_argument("--out", type=Path, default=ROOT / "output/value_anchors.csv")
    a.add_argument("--limit", type=int, default=8)
    a.add_argument("--max-contains", type=int, default=2)
    a.add_argument("--max-absent", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "build":
        build()
    elif args.cmd == "validate":
        validate(args.targets, args.limit)
    else:
        anchors(args.out, args.limit, args.max_contains, args.max_absent)


if __name__ == "__main__":
    main()
