"""
LanceDB settings and helpers for ingest.py, main.py, and remove.py.

Scoped to this folder's own store. The eval's shared settings — models,
reasoning, pacing, budgets, corpus paths — are in unstructured/pipeline.py.

Import this before nemo_retriever so the Ray resource override is in place
before Ray starts.
"""

import json
import os
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).parent

# The repo root, two levels up, is where the .env with NVIDIA_API_KEY lives.
load_dotenv(PROJECT_DIR.parents[1] / ".env")

# Ray advertises Apple Silicon as a GPU. NeMo Retriever then picks the CUDA-only
# operator variants (which import torch) instead of the hosted NIM endpoints.
# The CPU count is raised past the physical cores because every stage here waits
# on hosted NIM calls, and Ray refuses to plan a pipeline whose actors request
# more CPUs than it believes exist.
os.environ.setdefault(
    "RAY_OVERRIDE_RESOURCES",
    json.dumps({"GPU": 0, "CPU": max(16, os.cpu_count() or 8)}),
)

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")

# The ViDoRe v3 HR corpus that ships with this repo; ingest.py with no
# arguments walks it, so only the PDFs are pointed at, not the csv alongside.
DATASET_DIR = PROJECT_DIR.parent / "data" / "vidore_v3_hr"
DATA_DIR = DATASET_DIR / "pdfs"

# Absolute so both scripts hit the same table regardless of working directory.
LANCEDB_URI = str(PROJECT_DIR / "lancedb")
TABLE_NAME = "nv-ingest"

TOP_K = 5

# The ad-hoc question tool in main.py only. The eval's models and every other
# shared setting live in unstructured/pipeline.py, which is the authority for
# anything the two pipelines are compared on; this one is outside that.
LLM_MODEL = "nvidia/nemotron-3-super-120b-a12b"


def require_api_key() -> None:
    if not NVIDIA_API_KEY:
        raise SystemExit(
            "NVIDIA_API_KEY is not set. Add it to .env — get a free key at https://build.nvidia.com"
        )


def connect():
    import lancedb

    return lancedb.connect(LANCEDB_URI)


def table_names(db) -> list[str]:
    listing = db.list_tables()
    return list(getattr(listing, "tables", listing))


def has_ingested_data() -> bool:
    return Path(LANCEDB_URI).exists() and TABLE_NAME in table_names(connect())


def open_table():
    return connect().open_table(TABLE_NAME)


def chunk_counts(table) -> Counter:
    """Count chunks per raw `source` value (a JSON blob written by the ingestor)."""
    rows = table.search().select(["source"]).limit(0).to_arrow()
    return Counter(rows.column("source").to_pylist())


def source_labels(raw: str) -> tuple[str, str]:
    """Return (name, path) for a raw `source` value."""
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return raw, raw
    return parsed.get("source_name") or raw, parsed.get("source_id") or raw


def matches(target: str, raw: str) -> bool:
    """Whether a user-supplied name or path refers to the document in `raw`."""
    name, path = source_labels(raw)
    if target in (name, path):
        return True
    return str(Path(target).expanduser().resolve()) == path


def source_predicate(raws: Iterable[str]) -> str:
    """SQL predicate selecting exactly the given raw `source` values."""
    literals = ", ".join("'" + raw.replace("'", "''") + "'" for raw in raws)
    return f"source IN ({literals})"


def resolve_documents(target: str) -> list[str]:
    """Raw `source` values of the ingested documents a name or path refers to."""
    return [raw for raw in chunk_counts(open_table()) if matches(target, raw)]
