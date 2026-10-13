"""
Ask a question against documents already ingested by ingest.py:
  1. SEARCH — retrieve the closest chunks from LanceDB
  2. ANSWER — generate a grounded answer via NVIDIA NIM LLM

Usage:
    source setup.sh
    python ingest.py                      # once, whenever the corpus changes
    python main.py "your question"
    python main.py "your question" --top-k 8 --no-answer
    python main.py "your question" --doc report.pdf      # search one document only
"""

import argparse

import config

# Used when no question is passed on the command line.
QUESTION = "What is the main topic of these documents?"


# ── 1: SEARCH — retrieve closest chunks ──────────────────────────────────────

def search(question: str, top_k: int, where: str | None = None) -> list[dict]:
    from nemo_retriever.graph.retriever import Retriever

    retriever = Retriever(
        top_k=top_k,
        vdb_kwargs={"uri": config.LANCEDB_URI, "table_name": config.TABLE_NAME},
    )
    hits = retriever.query(question, vdb_kwargs={"where": where} if where else None)

    print(f"\n{'─'*60}")
    print(f"Query: {question}")
    print(f"{'─'*60}")
    for i, hit in enumerate(hits, 1):
        score = hit.get("_relevance_score") or hit.get("_score") or hit.get("_distance", "—")
        source = hit.get("filename") or hit.get("source", "unknown")
        page = hit.get("page_number")
        ctype = hit.get("content_type", "text")
        location = f"p.{page}" if page is not None else ""
        print(f"\n  [{i}] {source} {location}  |  type={ctype}  score={score}")
        print(f"      {hit.get('text', '')[:300]}")
    print(f"{'─'*60}\n")

    return hits


# ── 2: ANSWER — grounded generation via NIM LLM ──────────────────────────────

def generate_answer(question: str, hits: list[dict]) -> str:
    from openai import OpenAI

    context = "\n\n".join(
        f"[Source {i+1} — {h.get('filename', 'doc')} p.{h.get('page_number', '?')}]:\n{h.get('text', '')}"
        for i, h in enumerate(hits)
    )

    prompt = f"""You are a helpful assistant. Answer the question below using ONLY the provided sources.
If the answer is not in the sources, say "I could not find this in the provided documents."

Question: {question}

Sources:
{context}

Answer:"""

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=config.NVIDIA_API_KEY,
    )

    print("Generating answer ...\n")
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    answer_parts = []
    for chunk in response:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)
        answer_parts.append(delta)
    print("\n")

    return "".join(answer_parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default=QUESTION)
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--no-answer", action="store_true", help="Only show retrieved chunks")
    parser.add_argument(
        "--doc",
        help="Restrict the search to one ingested document (file name or path)",
    )
    args = parser.parse_args()

    config.require_api_key()

    if not config.has_ingested_data():
        raise SystemExit(f"No ingested documents at {config.LANCEDB_URI}. Run `python ingest.py` first.")

    where = None
    if args.doc:
        sources = config.resolve_documents(args.doc)
        if not sources:
            raise SystemExit(
                f"No ingested document matches {args.doc!r}. "
                "Run `python remove.py --list` to see what is stored."
            )
        for raw in sources:
            print(f"Scoped to {config.source_labels(raw)[1]}")
        where = config.source_predicate(sources)

    hits = search(args.question, args.top_k, where)
    if not hits:
        print("No results found.")
        return
    if not args.no_answer:
        generate_answer(args.question, hits)


if __name__ == "__main__":
    main()
