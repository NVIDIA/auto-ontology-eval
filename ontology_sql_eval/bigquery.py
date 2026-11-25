# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BigQuery plumbing for the verifier and the schema-only arm.

Spider2-lite's ``bq*``/``ga*`` questions run against public BigQuery datasets.
One Spider2 database (``ga4``, ``new_york_plus``) is one or more BigQuery
datasets, usually all in ``bigquery-public-data``, queried from a *billing*
project of the user's own, which pays for the bytes scanned. A connection
string names all three::

    bigquery://<billing_project>/<db_id>?dataset=<project>.<dataset>[&dataset=...]

Optional parameters match Auto Ontology's connector:
``maximum_bytes_billed`` (see :func:`parse_byte_size`), ``location``,
``schema``, ``credentials`` and ``credentials_file``. Several datasets are
given by repeating ``dataset`` rather than with a list, because
``CONNECTION_STRINGS`` is split on commas.

The treatment arm reads the same URLs through Auto Ontology's own connector
(``auto_ontology.connectors.bigquery``). Two rules are kept identical to it:

* **Every query is capped in bytes billed.** ``LIMIT`` does not reduce what
  BigQuery scans: ``SELECT body FROM stackoverflow.posts_questions LIMIT 1``
  bills 36 GB. The cap is ``maximum_bytes_billed`` on the URL, else
  :data:`DEFAULT_MAXIMUM_BYTES_BILLED`. The seeder resolves
  ``BIGQUERY_MAXIMUM_BYTES_BILLED`` and writes it into every URL.
* **A dataset named without its project resolves to its own project**, not to
  the billing project, the way a Trino session's catalog fills in a two-part
  name (:func:`qualify_datasets`). Auto Ontology's catalog keys tables by
  dataset, so the agent writes ``stackoverflow.users``; BigQuery alone would
  look for that dataset in the billing project.

Nothing here imports Auto Ontology or pandas at module level: the schema-only
arm's server imports it and has neither as a dependency of its own.
"""

from __future__ import annotations

import base64
import binascii
import concurrent.futures
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional
from urllib.parse import parse_qs, quote, unquote, urlparse

if TYPE_CHECKING:  # pragma: no cover - typing only
    from google.cloud import bigquery

__all__ = [
    "DEFAULT_MAXIMUM_BYTES_BILLED",
    "BigQueryTarget",
    "BytesCapExceeded",
    "NotReadOnly",
    "QueryTimeout",
    "bigquery_url",
    "is_bytes_cap_error",
    "is_infrastructure_error",
    "make_client",
    "parse_bigquery_url",
    "parse_byte_size",
    "qualify_datasets",
    "run_query",
    "targets_from_connection_strings",
]

SCHEME = "bigquery"

# 40 GiB, about $0.25 at on-demand pricing, and the default of Auto Ontology's
# BigQuery connector, so both arms are capped alike. Most Spider2 questions scan
# far less; the few gold queries over multi-terabyte tables (gnomAD, NOAA
# ICOADS) need it raised, and every query above it fails rather than bills.
DEFAULT_MAXIMUM_BYTES_BILLED = 40 * 1024**3

MAXIMUM_BYTES_BILLED_ENV = "BIGQUERY_MAXIMUM_BYTES_BILLED"

_GOOGLE_TOKEN_URI = "https://oauth2.googleapis.com/token"
_GOOGLE_TOKEN_URIS = frozenset(
    {_GOOGLE_TOKEN_URI, "https://accounts.google.com/o/oauth2/token"}
)
_JOB_LABELS = {"source": "auto-ontology"}

# Emitted on every ``to_dataframe`` call; the suggested pandas-gbq would pull in
# grpcio and the Storage Read API, which needs a role beyond Job User.
warnings.filterwarnings(
    "ignore",
    message="Retrieving DataFrames via google-cloud-bigquery is deprecated",
    category=PendingDeprecationWarning,
)

_BYTE_UNITS = {
    "": 1,
    "b": 1,
    "kb": 10**3,
    "mb": 10**6,
    "gb": 10**9,
    "tb": 10**12,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
    "tib": 1024**4,
}


class BytesCapExceeded(RuntimeError):
    """A query would bill more than the configured ``maximum_bytes_billed``."""


class QueryTimeout(TimeoutError):
    """A query outlived its time limit and was cancelled."""


class NotReadOnly(ValueError):
    """A statement that is not a ``SELECT`` was sent where only reads may run."""


def parse_byte_size(text: str | int | None) -> Optional[int]:
    """``"50GiB"`` -> bytes; ``0``, ``"0"``, ``"none"`` and ``"unlimited"`` -> no cap.

    Plain integers are bytes. Units are decimal (``GB``) or binary (``GiB``).
    """
    if text is None:
        return None
    if isinstance(text, int):
        if text < 0:
            raise ValueError(f"Byte size must not be negative: {text!r}")
        return text or None
    value = text.strip().lower().replace(" ", "")
    if value in ("", "0", "none", "unlimited"):
        return None
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([a-z]*)", value)
    if not match or match.group(2) not in _BYTE_UNITS:
        raise ValueError(
            f"Not a byte size: {text!r} (use bytes, or e.g. 500MB, 20GiB, 1TB)"
        )
    size = int(float(match.group(1)) * _BYTE_UNITS[match.group(2)])
    if size < 1:
        raise ValueError(f"Byte size must be at least one byte: {text!r}")
    return size


def _first(query: dict[str, list[str]], key: str) -> Optional[str]:
    value = query.get(key, [None])[0]
    return unquote(value).strip() or None if value else None


def _service_account_key(text: str) -> dict[str, Any]:
    try:
        info = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("BigQuery credentials are not a valid JSON key") from exc
    if not isinstance(info, dict) or info.get("type") != "service_account":
        raise ValueError("BigQuery credentials must be a service-account JSON key")
    token_uri = str(info.get("token_uri") or _GOOGLE_TOKEN_URI).strip()
    if token_uri not in _GOOGLE_TOKEN_URIS:
        raise ValueError(
            "BigQuery credentials must use Google's token endpoint "
            f"({_GOOGLE_TOKEN_URI})"
        )
    return {**info, "token_uri": token_uri}


def _decode_credentials(encoded: str) -> dict[str, Any]:
    """Decode the URL's raw or URL-safe-base64 service-account JSON key."""
    text = encoded.strip()
    if not text.startswith("{"):
        try:
            text = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4)).decode()
        except (binascii.Error, UnicodeDecodeError) as exc:
            raise ValueError(
                "BigQuery credentials are not valid base64-encoded JSON"
            ) from exc
    return _service_account_key(text)


def _by_dataset(items: Mapping[str, Any], dataset: str) -> Any:
    """Exact dataset match, else the only match differing only in case."""
    if dataset in items:
        return items[dataset]
    folded = [
        value for key, value in items.items() if key.casefold() == dataset.casefold()
    ]
    return folded[0] if len(folded) == 1 else None


@dataclass(frozen=True)
class BigQueryTarget:
    """One Spider2 database: where it lives and who pays for querying it."""

    db_id: str
    billing_project: str
    #: ``(project, dataset)`` pairs, in the order the URL gives them.
    datasets: tuple[tuple[str, str], ...]
    maximum_bytes_billed: Optional[int] = DEFAULT_MAXIMUM_BYTES_BILLED
    location: Optional[str] = None
    schema: Optional[str] = None
    credentials: Optional[Mapping[str, Any]] = None
    credentials_file: Optional[str] = None

    @property
    def scoped_datasets(self) -> tuple[tuple[str, str], ...]:
        """Datasets visible after applying the URL's optional ``schema=``."""
        if not self.schema:
            return self.datasets
        if not self.datasets:
            return ((self.billing_project, self.schema),)
        by_name = {dataset: (project, dataset) for project, dataset in self.datasets}
        selected = _by_dataset(by_name, self.schema)
        return (selected,) if selected is not None else ()

    @property
    def default_project(self) -> str:
        """The project an unknown dataset is assumed to belong to."""
        return self.datasets[0][0] if self.datasets else self.billing_project

    @property
    def dataset_projects(self) -> dict[str, str]:
        """Dataset name -> its owning project."""
        return {dataset: project for project, dataset in self.datasets}

    @property
    def default_dataset(self) -> Optional[str]:
        """``project.dataset`` when the database is a single dataset.

        Set as the job's default dataset so a bare table name resolves the way
        it does in a single-schema database.
        """
        if len(self.scoped_datasets) != 1:
            return None
        project, dataset = self.scoped_datasets[0]
        return f"{project}.{dataset}"


def parse_bigquery_url(connection_string: str) -> BigQueryTarget:
    """Parse the same ``bigquery://`` contract as Auto Ontology."""
    parsed = urlparse(connection_string)
    if parsed.scheme.split("+", 1)[0].lower() != SCHEME:
        raise ValueError(f"Not a BigQuery URL: {connection_string}")
    billing_project = unquote(parsed.netloc or "").strip()
    if not billing_project:
        raise ValueError(
            "BigQuery connection string requires a project, e.g. bigquery://my-project"
        )
    db_id = unquote(parsed.path.strip("/")).strip() or billing_project
    query = parse_qs(parsed.query)

    datasets: list[tuple[str, str]] = []
    seen: dict[str, str] = {}
    for raw in query.get("dataset", []):
        project, _, dataset = unquote(raw).strip().rpartition(".")
        project = project or billing_project
        if not dataset:
            raise ValueError(f"BigQuery dataset {raw!r} must be [project.]dataset")
        other = seen.setdefault(dataset, project)
        if other != project:
            raise ValueError(
                f"Dataset {dataset!r} appears in both {other!r} and {project!r}; "
                "one database cannot hold two datasets of the same name."
            )
        datasets.append((project, dataset))

    raw_cap = _first(query, "maximum_bytes_billed")
    cap = parse_byte_size(raw_cap) if raw_cap else DEFAULT_MAXIMUM_BYTES_BILLED
    raw_credentials = _first(query, "credentials")
    return BigQueryTarget(
        db_id=db_id,
        billing_project=billing_project,
        datasets=tuple(datasets),
        maximum_bytes_billed=cap,
        location=_first(query, "location"),
        schema=_first(query, "schema"),
        credentials=_decode_credentials(raw_credentials) if raw_credentials else None,
        credentials_file=_first(query, "credentials_file"),
    )


def bigquery_url(
    billing_project: str,
    db_id: str | None,
    datasets: Iterable[str] = (),
    *,
    maximum_bytes_billed: str | None = None,
    location: str | None = None,
    schema: str | None = None,
    credentials: str | None = None,
    credentials_file: str | None = None,
) -> str:
    """Build the connection string :func:`parse_bigquery_url` reads."""
    params = [f"dataset={quote(d, safe='.-_')}" for d in datasets]
    if maximum_bytes_billed is not None:
        params.append(f"maximum_bytes_billed={quote(maximum_bytes_billed)}")
    for key, value in (
        ("location", location),
        ("schema", schema),
        ("credentials", credentials),
        ("credentials_file", credentials_file),
    ):
        if value is not None:
            params.append(f"{key}={quote(value, safe='')}")
    path = f"/{quote(db_id)}" if db_id else ""
    query = f"?{'&'.join(params)}" if params else ""
    return f"{SCHEME}://{billing_project}{path}{query}"


def targets_from_connection_strings(raw: str) -> dict[str, BigQueryTarget]:
    """``db_id`` -> target for every ``bigquery://`` entry in a CONNECTION_STRINGS value."""
    out: dict[str, BigQueryTarget] = {}
    for entry in re.split(r"[,\r\n]+", raw or ""):
        entry = entry.strip()
        if entry.lower().startswith(f"{SCHEME}://"):
            target = parse_bigquery_url(entry)
            out[target.db_id] = target
    return out


# --- column types ------------------------------------------------------------

# Spelled-out STRUCT types run to tens of kilobytes on some genomics tables
# (gnomAD's ``vep``). They are kept whole up to this length, which still shows
# every field of GA4's and GA360's nested records.
MAX_TYPE_CHARS = 4000


def clip_type(text: str) -> str:
    """A column type as the catalog stores it, cut at :data:`MAX_TYPE_CHARS`."""
    return text if len(text) <= MAX_TYPE_CHARS else text[: MAX_TYPE_CHARS - 1] + "…"


# --- SQL ---------------------------------------------------------------------


def qualify_datasets(sql: str, dataset_projects: Mapping[str, str]) -> str:
    """Prefix ``dataset.table`` references with the project that owns *dataset*.

    Only two-part names whose dataset is one of *dataset_projects* change. The
    query is regenerated through sqlglot only when something does, so SQL that
    already names its projects -- all of Spider2's gold, and most of what the
    schema-only arm writes -- reaches BigQuery byte for byte. SQL sqlglot
    cannot parse is passed through for BigQuery to judge.
    """
    try:
        import sqlglot
        from sqlglot import exp

        statements = sqlglot.parse(sql, read="bigquery")
    except Exception:
        return sql

    changed = False
    for statement in statements:
        if statement is None:
            continue
        for table in statement.find_all(exp.Table):
            if table.catalog or not table.db:
                continue
            project = _by_dataset(dataset_projects, table.db)
            if project is None:
                continue
            table.set("catalog", exp.to_identifier(project))
            changed = True
    if not changed:
        return sql
    return ";\n".join(s.sql(dialect="bigquery") for s in statements if s is not None)


def make_client(target: BigQueryTarget) -> "bigquery.Client":
    """Build the same explicitly-authenticated-or-ADC client as Auto Ontology."""
    from google.cloud import bigquery
    from google.oauth2 import service_account

    info = target.credentials
    if info is None and target.credentials_file:
        path = Path(target.credentials_file)
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ValueError(
                f"BigQuery credentials file {path} cannot be read"
            ) from exc
        info = _service_account_key(text)
    credentials = (
        service_account.Credentials.from_service_account_info(
            dict(info), scopes=bigquery.Client.SCOPE
        )
        if info is not None
        else None
    )

    return bigquery.Client(
        project=target.billing_project,
        credentials=credentials,
        location=target.location,
    )


def _job_config(
    target: BigQueryTarget, *, dry_run: bool = False, timeout_s: float | None = None
) -> "bigquery.QueryJobConfig":
    from google.cloud import bigquery

    config = bigquery.QueryJobConfig(dry_run=dry_run, labels=dict(_JOB_LABELS))
    if target.maximum_bytes_billed:
        config.maximum_bytes_billed = target.maximum_bytes_billed
    if target.default_dataset:
        config.default_dataset = target.default_dataset
    if timeout_s is not None and not dry_run:
        config.job_timeout_ms = max(1, int(timeout_s * 1000))
    return config


def _reasons(exc: BaseException) -> set[str]:
    return {
        str(error.get("reason") or "")
        for error in getattr(exc, "errors", None) or []
        if isinstance(error, dict)
    }


def is_bytes_cap_error(exc: BaseException) -> bool:
    """True for BigQuery's ``bytesBilledLimitExceeded`` (and our own pre-check)."""
    if isinstance(exc, BytesCapExceeded):
        return True
    return "bytesBilledLimitExceeded" in _reasons(exc) or (
        "exceeded limit for bytes billed" in str(exc).lower()
    )


def is_infrastructure_error(exc: BaseException) -> bool:
    """True when the failure is the setup's, not the SQL's.

    Missing credentials, a billing project the caller cannot run jobs in, and
    quota exhaustion fail every query alike; scoring them as wrong SQL would let
    a broken setup read as a weak model.
    """
    try:
        from google.auth.exceptions import GoogleAuthError
    except ImportError:  # pragma: no cover - google-auth ships with the client
        GoogleAuthError = ()  # type: ignore[assignment,misc]
    if GoogleAuthError and isinstance(exc, GoogleAuthError):
        return True
    text = str(exc)
    return "bigquery.jobs.create" in text or bool(
        _reasons(exc) & {"quotaExceeded", "rateLimitExceeded", "billingNotEnabled"}
    )


# Statements a read-only script may not contain. EXECUTE IMMEDIATE is listed
# because the statement it runs is only known at run time.
_SCRIPT_WRITE_KEYWORDS = frozenset(
    {
        "ALTER",
        "CALL",
        "DELETE",
        "DROP",
        "EXECUTE",
        "EXPORT",
        "GRANT",
        "INSERT",
        "LOAD",
        "MERGE",
        "REVOKE",
        "TRUNCATE",
        "UPDATE",
    }
)


def script_write_keywords(sql: str) -> list[str]:
    """Write statements in a BigQuery script, in order of first use.

    A dry run types a multi-statement script only as ``SCRIPT``, so its
    statements are checked by keyword. ``CREATE`` is allowed only for
    ``TEMP``/``TEMPORARY`` objects, which last for the script alone.
    """
    from sqlglot.tokens import TokenType
    from sqlglot.dialects.bigquery import BigQuery

    tokens = [
        token
        for token in BigQuery().tokenize(sql)
        if token.token_type not in (TokenType.STRING, TokenType.IDENTIFIER)
    ]
    found: list[str] = []
    for i, token in enumerate(tokens):
        word = token.text.upper()
        if word == "CREATE":
            following = [t.text.upper() for t in tokens[i + 1 : i + 4]]
            if "TEMP" in following or "TEMPORARY" in following:
                continue
        elif word not in _SCRIPT_WRITE_KEYWORDS:
            continue
        # An unquoted column named like one of these is refused too.
        if word not in found:
            found.append(word)
    return found


def run_query(
    client: "bigquery.Client",
    target: BigQueryTarget,
    sql: str,
    *,
    timeout_s: float | None = None,
    read_only: bool = False,
) -> tuple[Any, Any]:
    """Run *sql* for *target*; return ``(row_iterator, job)``.

    With *read_only*, a free dry run first refuses anything but a ``SELECT``
    and any query it can already tell is over the byte cap. Without it the cap
    is still enforced, by BigQuery, when the job runs.

    Raises :class:`BytesCapExceeded`, :class:`QueryTimeout` or
    :class:`NotReadOnly`; BigQuery's own errors propagate unchanged.
    """
    sql = qualify_datasets(sql, target.dataset_projects)
    if read_only:
        dry = client.query(
            sql, job_config=_job_config(target, dry_run=True), location=target.location
        )
        statement_type = (dry.statement_type or "").upper()
        if statement_type == "SCRIPT":
            writes = script_write_keywords(sql)
            if writes:
                raise NotReadOnly(
                    "Scripts may only declare variables, temporary functions and "
                    f"temporary tables, and select; this one uses {', '.join(writes)}."
                )
        elif statement_type != "SELECT":
            raise NotReadOnly(
                f"Only SELECT queries may run here, not {dry.statement_type}."
            )
        cap = target.maximum_bytes_billed
        estimate = dry.total_bytes_processed or 0
        if cap and estimate > cap:
            raise BytesCapExceeded(
                f"Query would process {estimate / 1024**3:.1f} GiB, over the "
                f"{cap / 1024**3:.1f} GiB maximum_bytes_billed cap. Read fewer "
                "columns or shards, or raise BIGQUERY_MAXIMUM_BYTES_BILLED."
            )

    job = client.query(
        sql,
        job_config=_job_config(target, timeout_s=timeout_s),
        location=target.location,
    )
    try:
        rows = job.result(timeout=timeout_s)
    except concurrent.futures.TimeoutError as exc:
        try:
            job.cancel()
        except Exception:
            pass
        raise QueryTimeout(f"Query ran longer than {timeout_s:g}s") from exc
    except Exception as exc:
        if is_bytes_cap_error(exc):
            raise BytesCapExceeded(str(exc).splitlines()[0]) from exc
        if "timeout" in _reasons(exc) or "job timed out" in str(exc).lower():
            raise QueryTimeout(str(exc).splitlines()[0]) from exc
        raise
    return rows, job
