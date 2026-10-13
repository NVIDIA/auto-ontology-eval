"""
Ingest documents into LanceDB: extract text/tables/charts, embed, store.

Run this when the corpus changes — not before every question. main.py reads the
table this script writes.

Usage:
    python ingest.py                             # everything in ./data
    python ingest.py file.pdf folder/ https://example.com/doc.pdf
    python ingest.py --append extra.pdf          # add to the table instead of replacing it
"""

import argparse
import shutil
from pathlib import Path

import config


def collect_inputs(args: list[str]) -> tuple[list[str], list[str]]:
    """Split CLI arguments into local file paths and URLs, expanding directories."""
    files: list[str] = []
    urls: list[str] = []

    for arg in args:
        if arg.startswith(("http://", "https://")):
            urls.append(arg)
            continue
        path = Path(arg)
        if path.is_dir():
            files.extend(str(f) for f in sorted(path.rglob("*")) if f.is_file())
        elif path.is_file():
            files.append(str(path))
        else:
            raise SystemExit(f"No such file or directory: {arg}")

    return files, urls


def split_supported(files: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    """Separate ingestable files from the rest, with a reason for each skip."""
    from nemo_retriever.common.input_files import input_type_for_path

    # DOCX/PPTX are converted to PDF by a `libreoffice` binary on $PATH.
    have_libreoffice = shutil.which("libreoffice") is not None

    supported: list[str] = []
    skipped: list[tuple[str, str]] = []
    for f in files:
        input_type = input_type_for_path(f)
        if input_type is None:
            skipped.append((f, "unsupported file type"))
        elif input_type == "doc" and not have_libreoffice:
            skipped.append((f, "DOCX/PPTX needs LibreOffice on $PATH"))
        else:
            supported.append(f)
    return supported, skipped


def ingest(files: list[str], urls: list[str], append: bool = False) -> int:
    from nemo_retriever import create_ingestor

    ingestor = create_ingestor(run_mode="batch")
    if files:
        ingestor = ingestor.files(files)
    if urls:
        ingestor = ingestor.urls(urls)

    ingestor = (
        ingestor.extract(
            extract_text=True,
            extract_tables=True,
            extract_charts=True,
            extract_infographics=True,
        )
        .embed(config.embed_params())
        .vdb_upload(
            vdb_op="lancedb",
            vdb_kwargs={
                "uri": config.LANCEDB_URI,
                "table_name": config.TABLE_NAME,
                "overwrite": not append,
                # Stamp the table with the model that produced its vectors.
                # Querying refuses to run without it: .env says what we embed
                # the query with now, and only the table can say what the stored
                # vectors came from. A mismatch between the two returns almost
                # nothing instead of failing, so NeMo verifies rather than trusts.
                "embedding_model_name": config.EMBED_MODEL_NAME,
            },
        )
    )

    chunks = ingestor.ingest()
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", help="Files, directories, or URLs (default: ./data)")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Add to the existing table instead of replacing its contents",
    )
    args = parser.parse_args()

    config.require_api_key()

    files, urls = collect_inputs(args.inputs or [str(config.DATA_DIR)])
    files, skipped = split_supported(files)

    for f, reason in skipped:
        print(f"  skipping {Path(f).name} — {reason}")

    if not files and not urls:
        raise SystemExit(f"Nothing to ingest. Put PDFs or text files in {config.DATA_DIR} and rerun.")

    print(f"Ingesting {len(files)} file(s) and {len(urls)} URL(s) ...")
    for f in files:
        print(f"  • {Path(f).name}")
    for u in urls:
        print(f"  • {u}")

    n_chunks = ingest(files, urls, append=args.append)

    verb = "Added" if args.append else "Stored"
    print(f"\n{verb} {n_chunks} chunks in {config.LANCEDB_URI} (table '{config.TABLE_NAME}').")
    print('Ask a question with:  python main.py "your question"')


if __name__ == "__main__":
    main()
