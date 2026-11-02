"""Find train exemplars by the question with its names and values masked.

The AskData way: an LLM rewrites every question with each database name,
entity, value, number and date replaced by a placeholder, so what is left is
what the question asks for ("the [ENTITY] with the highest [ATTRIBUTE]").
Masked train and dev questions are embedded, and each dev question gets the
train examples whose masked question is nearest. The dev gold SQL is never
read.

Writes a CSV in the format ``--sql-examples-csv`` reads::

    python scripts/extra_prompts/masked_exemplars.py --k 8
    python main.py ... --sql-examples-csv inputs/extra_prompts/bird/masked_exemplars.csv --sql-examples-k 8

Masks and vectors are cached under ``output/masked_exemplars/``, so a re-run
only pays for what it has not already asked.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

TRAIN = ROOT / "datasets/bird/train/train.json"
DEV = ROOT / "datasets/bird/evaluation.json"
CACHE = ROOT / "output/masked_exemplars"
OUT = ROOT / "inputs/extra_prompts/bird/masked_exemplars.csv"

SYSTEM = """You mask questions asked of a database so that only what they ask for is left.

Replace every name, entity, literal value, number, date, code and domain noun with a placeholder:
[TABLE] for the kind of thing the question is about (schools, players, cards),
[COLUMN] for an attribute (height, salary, rating),
[VALUE] for a literal (Alameda, 'Male', Coldsnap),
[NUMBER] for a number, [DATE] for a date or year.

Keep every word that says what is asked and how: how many, list, which, highest, lowest,
most, average, percentage, ratio, difference, each, per, between, more than, at least,
both, and, or, not, distinct, first, top 3, in descending order.

Return one masked question per input, with the same id."""


class Masked(BaseModel):
    id: int
    masked: str


class MaskedBatch(BaseModel):
    questions: list[Masked]


def _key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:20]


class JsonCache:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.data: dict = json.loads(path.read_text()) if path.exists() else {}

    def save(self) -> None:
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data))


def mask_all(questions: list[str], batch: int, workers: int) -> list[str]:
    from auto_ontology.utils.llm_invoke import (
        get_llm_client,
        safe_invoke_with_structured_output,
    )
    from langchain_core.messages import HumanMessage, SystemMessage

    cache = JsonCache(CACHE / "masks.json")
    todo = sorted({q for q in questions if _key(q) not in cache.data})
    print(f"masking {len(todo)} questions ({len(questions) - len(todo)} cached)")
    llm = get_llm_client(temperature=0.0)

    def ask(chunk: list[str]) -> None:
        listing = "\n".join(f"{i}. {q}" for i, q in enumerate(chunk))
        result = safe_invoke_with_structured_output(
            llm,
            [SystemMessage(content=SYSTEM), HumanMessage(content=listing)],
            MaskedBatch,
        )
        for item in result.questions if result else []:
            if 0 <= item.id < len(chunk) and item.masked.strip():
                with cache.lock:
                    cache.data[_key(chunk[item.id])] = item.masked.strip()

    chunks = [todo[i : i + batch] for i in range(0, len(todo), batch)]
    with ThreadPoolExecutor(workers) as pool:
        for done, future in enumerate(as_completed(pool.submit(ask, c) for c in chunks), 1):
            future.result()
            if done % 20 == 0 or done == len(chunks):
                cache.save()
                print(f"  {done}/{len(chunks)} batches")
    cache.save()
    return [cache.data.get(_key(q), q) for q in questions]


def embed_all(texts: list[str], input_type: str) -> list[list[float]]:
    from auto_ontology.utils.embedding import embed_texts, get_embed_params

    cache = JsonCache(CACHE / f"vectors_{input_type}.json")
    todo = sorted({t for t in texts if _key(t) not in cache.data})
    print(f"embedding {len(todo)} {input_type} texts ({len(texts) - len(todo)} cached)")
    params = get_embed_params()
    for start in range(0, len(todo), 500):
        chunk = todo[start : start + 500]
        for text, vector in zip(chunk, embed_texts(chunk, params, input_type)):
            if vector:
                cache.data[_key(text)] = vector
        cache.save()
    return [cache.data.get(_key(t)) for t in texts]


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--batch", type=int, default=25)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    train = json.loads(TRAIN.read_text())
    dev = json.loads(DEV.read_text())
    masks = mask_all(
        [t["question"] for t in train] + [d["question"] for d in dev],
        args.batch,
        args.workers,
    )
    train_masked, dev_masked = masks[: len(train)], masks[len(train) :]

    import numpy as np

    train_vectors = embed_all(train_masked, "passage")
    dev_vectors = embed_all(dev_masked, "query")
    keep = [i for i, v in enumerate(train_vectors) if v]
    matrix = np.array([_unit(train_vectors[i]) for i in keep])

    rows = []
    for d, masked, vector in zip(dev, dev_masked, dev_vectors):
        if not vector:
            continue
        scores = matrix @ np.array(_unit(vector))
        for rank, j in enumerate(np.argsort(-scores)[: args.k], 1):
            t = train[keep[j]]
            rows.append(
                {
                    "question_id": d["question_id"],
                    "db_id": d["db_id"],
                    "question": d["question"],
                    "masked_question": masked,
                    "rank": rank,
                    "score": round(float(scores[j]), 4),
                    "train_db": t["db_id"],
                    "train_question": t["question"],
                    "train_masked_question": train_masked[keep[j]],
                    "train_evidence": t.get("evidence", ""),
                    "train_sql": t["SQL"],
                }
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows for {len({r['question_id'] for r in rows})} questions -> {args.out}")


if __name__ == "__main__":
    main()
