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

# The embedder, from the same EMBED_* triplet the attribute pipeline uses. The
# two pipelines are compared against each other, and vectors from two different
# models are not comparable: whichever side embeds differently is being measured
# on a different space, so the comparison stops being about retrieval strategy.
# One triplet in .env is what keeps them honest.
#
# NeMo's own default (nemotron-3-embed-1b) is deliberately not used here. It is
# text-only, and half of what the extract stage yields from this corpus is
# charts, tables and infographics; the VL model in EMBED_MODEL reads those as
# images, which is also why ingest asks for the text_image modality.
#
# Both stages must agree. Ingest embeds the chunks and the retriever embeds the
# query, and a mismatch between them puts the two in different spaces, where
# search silently returns near-nothing rather than failing.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "")
EMBED_ENDPOINT = os.environ.get("EMBED_ENDPOINT", "")
EMBED_API_KEY = os.environ.get("EMBED_API_KEY", "")


def _split_provider_prefix(model: str) -> tuple[str, str | None]:
    """Split a wire model name into the registered name and a provider prefix.

    This endpoint wants the vendor segment twice —
    ``nvidia/nvidia/llama-nemotron-embed-vl-1b-v2`` — and the single-segment form
    is rejected outright. NeMo wants the opposite: it looks the model up in its
    registry to enforce the right tokenizer, prefix and input limit, and refuses
    a name it cannot find, which the doubled form is. It prepends the provider
    prefix itself when the request goes out, so splitting here satisfies both.
    """
    head, _, rest = model.partition("/")
    if rest.startswith(f"{head}/"):
        return rest, head
    return model, None


#: The registry name NeMo resolves, and the prefix it prepends on the wire.
EMBED_MODEL_NAME, EMBED_PROVIDER_PREFIX = _split_provider_prefix(EMBED_MODEL)


def require_embed_config() -> None:
    missing = [
        name
        for name, value in (
            ("EMBED_MODEL", EMBED_MODEL),
            ("EMBED_ENDPOINT", EMBED_ENDPOINT),
            ("EMBED_API_KEY", EMBED_API_KEY),
        )
        if not value
    ]
    if missing:
        raise SystemExit(
            f"{', '.join(missing)} not set in .env. The benchmark and the "
            "attribute pipeline must embed with the same model to be comparable."
        )


def embed_params():
    """``EmbedParams`` for the ingest-side chunk embeddings."""
    from nemo_retriever.common.params import EmbedParams

    require_embed_config()
    return EmbedParams(
        embed_invoke_url=EMBED_ENDPOINT,
        model_name=EMBED_MODEL_NAME,
        embed_model_name=EMBED_MODEL_NAME,
        embed_model_provider_prefix=EMBED_PROVIDER_PREFIX,
        api_key=EMBED_API_KEY,
        embed_modality="text_image",
    )


def embed_kwargs() -> dict:
    """Query-side embedding overrides for ``Retriever``.

    Without these it falls back to NeMo's default model, which would embed the
    query in a different space from the chunks ingest wrote.
    """
    require_embed_config()
    return {
        "model_name": EMBED_MODEL_NAME,
        "embed_model_name": EMBED_MODEL_NAME,
        "embed_model_provider_prefix": EMBED_PROVIDER_PREFIX,
        "embed_invoke_url": EMBED_ENDPOINT,
        "api_key": EMBED_API_KEY,
    }

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
