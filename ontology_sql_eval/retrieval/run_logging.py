# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run-scoped logging and timing instrumentation for the retrieval eval.

The eval can run many questions concurrently, which makes a single interleaved
stdout stream useless for finding where the time goes. This module gives every
run its own directory under ``logs/`` containing:

``run.log``
    Every log record from every thread at DEBUG, tagged with the worker thread
    and the question it belongs to. This is where third-party timing signal
    lands too (``httpx`` logs one line per LLM / embedding HTTP call).
``console.log``
    The INFO-level subset, i.e. what was printed to the terminal.
``questions/q<id>.log``
    One file per question holding only that question's records -- read a single
    question's life end to end without untangling the interleaving.
``events.jsonl``
    A timeline. One record per phase boundary with monotonic offsets from the
    run start, so a concurrency / Gantt view can be reconstructed and stalls
    (serialised sections, lock convoys) become visible.
``questions.jsonl``
    One record per finished question: total / agent / scoring seconds, per-node
    agent timings, HTTP call counts, error, and the worker that ran it.
``summary.json``
    Aggregates: wall clock, throughput, per-node and per-phase statistics
    (count / total / mean / median / p95 / max), and the slowest questions.

Nothing here is dataset-specific; the eval driver decides what to record.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import resource
import statistics
import subprocess
import threading
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "RunLogger",
    "question_context",
    "current_question",
    "setup_run_logging",
    "slug",
    "http_calls",
    "peak_rss_mb",
    "system_memory",
    "ResourceSampler",
]

_UNSET = "-"

# Per-question context. A ContextVar is per-thread here (each pool worker starts
# from an empty context), which is exactly the scoping we want: records emitted
# anywhere down the call stack -- including inside GSF and its dependencies --
# carry the question that triggered them.
_question_ctx: ContextVar[dict[str, Any] | None] = ContextVar(
    "eval_question_ctx", default=None
)


def slug(value: Any) -> str:
    """Return *value* reduced to a filesystem-safe token."""
    text = str(value) if value is not None else _UNSET
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return text[:80] or _UNSET


def current_question() -> dict[str, Any]:
    """Return the calling thread's question context (empty dict when unset)."""
    return _question_ctx.get() or {}


class question_context:  # noqa: N801 - context manager, not a type
    """Bind ``qid`` / ``row_index`` / ``worker`` to every log record in this block."""

    def __init__(self, qid: Any, row_index: Any = None, **fields: Any) -> None:
        self._fields: dict[str, Any] = {
            "qid": qid,
            "row_index": row_index,
            "qslug": f"q{slug(qid)}",
            **fields,
        }
        self._token = None

    def __enter__(self) -> dict[str, Any]:
        self._token = _question_ctx.set(self._fields)
        return self._fields

    def __exit__(self, *_exc: Any) -> None:
        if self._token is not None:
            _question_ctx.reset(self._token)


def http_calls() -> int:
    """Model round-trips made *on this thread* for the current question.

    Every LLM and embedding call goes out over ``httpx``, which logs one line
    per request; :class:`_HttpCallCounter` tallies them into the question
    context. Diffing this across a phase boundary attributes calls to the agent
    node that made them, which is what separates "the model is slow" from "we
    are calling the model too many times".

    Caveat worth knowing when reading the numbers: only calls issued on the
    question's own thread can be attributed. ``nemo_retriever`` dispatches
    embedding requests to its own thread pool, and a pool thread starts from an
    empty context, so those land in the run-level ``http_endpoints`` tally
    instead of on a question. In practice the per-question figure is the *chat
    completion* count, which is the one that maps to agent nodes.
    """
    return int(current_question().get("http_calls", 0))


# Endpoint buckets for the run-level tally. Substring match on the request URL,
# first hit wins; anything unmatched is counted as "other".
_ENDPOINT_KINDS = (
    ("chat_completions", "/chat/completions"),
    ("embeddings", "/embeddings"),
)


# httpx logs `HTTP Request: POST <url> "HTTP/1.1 429 Too Many Requests"`. The
# status is the whole point when measuring concurrency: a run that slows down
# because the endpoint started returning 429s is throttled, not merely loaded,
# and only the status code tells those apart.
_STATUS_RE = re.compile(r'"HTTP/[\d.]+ (\d{3})')


class _HttpCallCounter(logging.Handler):
    """Tally ``httpx`` request lines per question, endpoint, and status code."""

    def __init__(self, level: int = logging.INFO) -> None:
        super().__init__(level)
        self.endpoints: dict[str, int] = {}
        self.statuses: dict[str, int] = {}
        self.non_ok: dict[str, int] = {}
        self.unattributed = 0
        self._tally_lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if not message.startswith("HTTP Request:"):
            return

        kind = "other"
        for name, marker in _ENDPOINT_KINDS:
            if marker in message:
                kind = name
                break

        status_match = _STATUS_RE.search(message)
        status = status_match.group(1) if status_match else "unknown"

        ctx = _question_ctx.get()
        with self._tally_lock:
            self.endpoints[kind] = self.endpoints.get(kind, 0) + 1
            self.statuses[status] = self.statuses.get(status, 0) + 1
            if status != "200":
                key = f"{kind}:{status}"
                self.non_ok[key] = self.non_ok.get(key, 0) + 1
            if ctx is None:
                self.unattributed += 1
        if ctx is not None:
            # Only ever mutated by the thread that owns this context.
            ctx["http_calls"] = ctx.get("http_calls", 0) + 1
            if status != "200":
                ctx["http_non_ok"] = ctx.get("http_non_ok", 0) + 1


def _resolved_model_config() -> dict[str, Any]:
    """The endpoint/model/key-gateway GSF actually resolved for each triplet.

    Resolution is ``<PREFIX>_<FIELD>`` -> ``DEFAULT_MODELS_<FIELD>`` -> a
    built-in keyed off the triplet's API-key prefix, so the effective model is
    frequently not written down anywhere in ``.env``. Recording the resolved
    values makes a run self-describing: an upstream default change, or a local
    pin that overrides one, both show up here instead of having to be inferred
    later from which vectors look wrong.
    """
    try:
        from gsf.utils.model_config import resolve
    except Exception:  # pragma: no cover - provenance is best-effort
        return {}

    resolved: dict[str, Any] = {}
    for prefix in ("REASONING", "NON_REASONING", "EMBED", "RERANK"):
        try:
            key = resolve(prefix, "API_KEY")
            resolved[prefix] = {
                "endpoint": resolve(prefix, "ENDPOINT"),
                "model": resolve(prefix, "MODEL"),
                # The prefix decides which built-in defaults apply, so it is
                # part of the configuration; the key itself never is.
                "key_prefix": key.split("-", 1)[0] + "-" if "-" in key else "",
                "pinned_by_env": bool(os.environ.get(f"{prefix}_MODEL")),
            }
        except Exception:  # pragma: no cover
            continue
    return resolved


class ResourceSampler:
    """Sample CPU and memory on a background thread for the life of a run.

    Answers "what did this level cost the machine?" — which wall clock alone
    cannot, since a run that is twice as fast because it used four times the CPU
    is a different result from one that simply waited on the network less.

    Two scopes are recorded because they answer different questions. *Process*
    CPU and RSS are what the eval itself consumed; *system* CPU and memory
    include everything else on the box, which is what says whether a level was
    competing for the machine. Process CPU is a percentage of one core, so it
    exceeds 100% whenever threads run in parallel — ``*_per_core`` divides by the
    core count to give the saturation fraction.
    """

    def __init__(self, path: Path, interval: float = 1.0) -> None:
        self.path = path
        self.interval = interval
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc: Any = None
        self._cpu_count = os.cpu_count() or 1
        self._started_at = 0.0

    def start(self) -> None:
        try:
            import psutil
        except ImportError:  # pragma: no cover - sampling is optional
            logging.getLogger(__name__).warning(
                "psutil unavailable — CPU/RAM sampling disabled"
            )
            return
        self._proc = psutil.Process(os.getpid())
        # cpu_percent is a delta since the previous call, so the first reading
        # is meaningless. Prime both scopes now and discard.
        self._proc.cpu_percent(None)
        psutil.cpu_percent(None)
        self._started_at = time.perf_counter()
        self._thread = threading.Thread(
            target=self._loop, name="resource-sampler", daemon=True
        )
        self._thread.start()

    def _loop(self) -> None:
        import psutil

        while not self._stop.wait(self.interval):
            try:
                mem = psutil.virtual_memory()
                sample = {
                    "t": round(time.perf_counter() - self._started_at, 2),
                    "proc_cpu_percent": round(self._proc.cpu_percent(None), 1),
                    "proc_rss_mb": round(self._proc.memory_info().rss / 1024**2, 1),
                    "proc_threads": self._proc.num_threads(),
                    "sys_cpu_percent": round(psutil.cpu_percent(None), 1),
                    "sys_mem_used_gb": round((mem.total - mem.available) / 1024**3, 2),
                    "sys_mem_percent": mem.percent,
                }
                self.samples.append(sample)
            except Exception:  # pragma: no cover - never break a run to sample
                continue

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 3)
        if self.samples:
            with self.path.open("w", encoding="utf-8") as handle:
                for sample in self.samples:
                    handle.write(json.dumps(sample) + "\n")

    def stats(self) -> dict[str, Any]:
        """Average / max (and median / p95) per metric across the run."""
        if not self.samples:
            return {}

        def summarise(field: str) -> dict[str, float]:
            data = sorted(s[field] for s in self.samples if field in s)
            if not data:
                return {}
            return {
                "avg": round(statistics.fmean(data), 1),
                "max": round(data[-1], 1),
                "median": round(statistics.median(data), 1),
                "p95": round(data[min(len(data) - 1, int(0.95 * len(data)))], 1),
            }

        out: dict[str, Any] = {
            "samples": len(self.samples),
            "interval_seconds": self.interval,
            "cpu_count": self._cpu_count,
        }
        for field in (
            "proc_cpu_percent",
            "proc_rss_mb",
            "proc_threads",
            "sys_cpu_percent",
            "sys_mem_used_gb",
            "sys_mem_percent",
        ):
            if summary := summarise(field):
                out[field] = summary
        # Saturation as a fraction of the whole machine, so numbers stay
        # comparable across boxes with different core counts.
        if "proc_cpu_percent" in out:
            out["proc_cpu_per_core"] = {
                k: round(v / self._cpu_count, 3)
                for k, v in out["proc_cpu_percent"].items()
            }
        return out


def peak_rss_mb() -> float:
    """Peak resident set size of this process, in MB.

    ``ru_maxrss`` is bytes on macOS and kilobytes on Linux; normalise so the
    number means the same thing on both. Used by the concurrency sweep, where a
    level that slows down because the *machine* ran out of memory looks exactly
    like a level that slowed down because the endpoint throttled it — and only a
    memory figure distinguishes them.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if platform.system() == "Darwin" else 1024
    return round(raw / divisor, 1)


def system_memory() -> dict[str, Any]:
    """Best-effort host memory pressure snapshot (macOS ``vm_stat``).

    Returns ``{}`` when the numbers cannot be read; this is diagnostic context,
    never something a run should fail on.
    """
    if platform.system() != "Darwin":
        return {}
    try:
        out = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=5
        ).stdout
    except Exception:  # pragma: no cover - diagnostics only
        return {}

    page_size = 16384
    if match := re.search(r"page size of (\d+) bytes", out):
        page_size = int(match.group(1))

    def pages(label: str) -> int:
        match = re.search(rf"{label}:\s+(\d+)", out)
        return int(match.group(1)) if match else 0

    to_gb = page_size / 1024**3
    return {
        "free_gb": round(pages("Pages free") * to_gb, 2),
        "active_gb": round(pages("Pages active") * to_gb, 2),
        "inactive_gb": round(pages("Pages inactive") * to_gb, 2),
        "wired_gb": round(pages(r"Pages wired down") * to_gb, 2),
        "compressed_gb": round(pages(r"Pages occupied by compressor") * to_gb, 2),
        "swapins": pages("Swapins"),
        "swapouts": pages("Swapouts"),
    }


def _pin_logger_level(logger: logging.Logger, level: int) -> None:
    """Hold *logger* at *level* against third-party reconfiguration.

    ``nemo_retriever.common.api.internal.transform.embed_text`` runs
    ``logging.getLogger("httpx").setLevel(logging.ERROR)`` at import time, and
    that module is imported lazily on the first embedding call — i.e. partway
    through the first question, not at startup. Setting the level once up front
    is therefore not enough: httpx goes quiet after a request or two and every
    per-node model-call count silently reads zero.
    """
    logging.Logger.setLevel(logger, level)
    # Shadow the instance's bound setLevel so later callers cannot lower it.
    # The requested level is discarded on purpose -- that is the whole point --
    # and the class method is called directly to avoid recursing into this
    # replacement. Instance-level only: other loggers are untouched, and
    # _unpin_logger_level puts the original method back.
    logger.setLevel = (  # type: ignore[method-assign]
        lambda _ignored, _logger=logger, _level=level: logging.Logger.setLevel(
            _logger, _level
        )
    )


def _unpin_logger_level(logger: logging.Logger) -> None:
    """Undo :func:`_pin_logger_level`, restoring the class-level ``setLevel``."""
    # Deleting the instance attribute is what restores the real method: the
    # lambda only ever shadowed it in the instance __dict__, so the class
    # implementation was never modified and re-emerges via normal lookup.
    logger.__dict__.pop("setLevel", None)


class _QuestionContextFilter(logging.Filter):
    """Stamp the current question onto every record so formats can use it."""

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = current_question()
        record.qid = ctx.get("qid", _UNSET)
        record.row_index = ctx.get("row_index", _UNSET)
        record.qslug = ctx.get("qslug") or _UNSET
        return True


class _PerQuestionFileRouter(logging.Handler):
    """Fan records out to one log file per question, opened lazily.

    Files are keyed by the question slug on the record (set by
    :class:`_QuestionContextFilter`); records emitted outside any question
    context are dropped, since they already land in ``run.log``.
    """

    def __init__(self, directory: Path, level: int = logging.DEBUG) -> None:
        super().__init__(level)
        self._dir = directory
        self._dir.mkdir(parents=True, exist_ok=True)
        self._handlers: dict[str, logging.FileHandler] = {}
        self._router_lock = threading.Lock()

    def _handler_for(self, key: str) -> logging.FileHandler:
        handler = self._handlers.get(key)
        if handler is None:
            with self._router_lock:
                handler = self._handlers.get(key)
                if handler is None:
                    handler = logging.FileHandler(
                        self._dir / f"{key}.log", encoding="utf-8"
                    )
                    handler.setLevel(self.level)
                    if self.formatter is not None:
                        handler.setFormatter(self.formatter)
                    self._handlers[key] = handler
        return handler

    def emit(self, record: logging.LogRecord) -> None:
        key = getattr(record, "qslug", _UNSET)
        if not key or key == _UNSET:
            return
        try:
            self._handler_for(key).emit(record)
        except Exception:  # pragma: no cover - logging must never break the run
            self.handleError(record)

    def close(self) -> None:
        with self._router_lock:
            for handler in self._handlers.values():
                try:
                    handler.close()
                except Exception:  # pragma: no cover
                    pass
            self._handlers.clear()
        super().close()


# Chatty third-party loggers that would otherwise bury the run at DEBUG. They
# stay at INFO, which keeps the useful "HTTP Request: ..." lines (one per LLM /
# embedding call -- the cheapest latency probe there is) while dropping
# per-frame connection noise.
_NOISY_LOGGERS = (
    "httpcore",
    "urllib3",
    "asyncio",
    "neo4j",
    "openai._base_client",
    "sqlalchemy.engine",
    "matplotlib",
    "PIL",
)

# Loggers whose INFO output is worth keeping in the files but is pure noise on
# a live console -- SQLAlchemy's engine echo alone prints every catalog query,
# several per node, times every worker.
_CONSOLE_QUIET_LOGGERS = ("sqlalchemy.engine", "neo4j")


class _ConsoleQuietFilter(logging.Filter):
    """Keep verbose-but-useful loggers out of the console, not out of the files."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        return not record.name.startswith(_CONSOLE_QUIET_LOGGERS)


_FILE_FORMAT = (
    "%(asctime)s %(levelname)-8s [%(threadName)-12s] [q=%(qid)s] "
    "%(name)s:%(lineno)d: %(message)s"
)
_CONSOLE_FORMAT = "%(asctime)s %(levelname)-7s [q=%(qid)s] %(name)s: %(message)s"


class RunLogger:
    """Owns one run's log directory, timeline, and per-question records."""

    def __init__(self, log_dir: Path, run_id: str) -> None:
        self.run_id = run_id
        self.dir = log_dir / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.questions_dir = self.dir / "questions"
        self.questions_dir.mkdir(parents=True, exist_ok=True)

        self.events_path = self.dir / "events.jsonl"
        self.questions_path = self.dir / "questions.jsonl"
        self.summary_path = self.dir / "summary.json"
        self.run_log_path = self.dir / "run.log"
        self.console_log_path = self.dir / "console.log"

        self._t0 = time.perf_counter()
        self._started_at = datetime.now(timezone.utc)
        self._write_lock = threading.Lock()
        # Truncate rather than append. A run id identifies one run, and
        # summary.json is rewritten wholesale from this process's records; if
        # these two files appended instead, re-running an id after a crash
        # would leave the partial attempt's rows in front of the real ones and
        # every downstream mean, percentile and count would silently be
        # computed over both.
        self._events = self.events_path.open("w", encoding="utf-8")
        self._questions = self.questions_path.open("w", encoding="utf-8")
        self._question_records: list[dict[str, Any]] = []
        self._handlers: list[logging.Handler] = []
        self._httpx_logger: logging.Logger | None = None
        self._http_counter: _HttpCallCounter | None = None
        self._memory_at_start: dict[str, Any] = system_memory()
        self.resources = ResourceSampler(self.dir / "resources.jsonl")
        self.resources.start()
        # Handlers displaced by attach_logging, restored on close so a caller
        # that keeps running after the eval (the pipeline's judge stage) does
        # not end up with a silent root logger.
        self._displaced: list[logging.Handler] = []
        self._prior_level: int | None = None

    # -- timeline -------------------------------------------------------

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._t0

    def event(self, kind: str, **fields: Any) -> None:
        """Append one timeline record (monotonic offset from the run start)."""
        ctx = current_question()
        record = {
            "t": round(self.elapsed, 4),
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "thread": threading.current_thread().name,
            "qid": ctx.get("qid"),
            "row_index": ctx.get("row_index"),
            **fields,
        }
        with self._write_lock:
            self._events.write(json.dumps(record, default=str) + "\n")
            self._events.flush()

    def question(self, record: dict[str, Any]) -> None:
        """Append one finished-question record."""
        with self._write_lock:
            self._question_records.append(record)
            self._questions.write(json.dumps(record, default=str) + "\n")
            self._questions.flush()

    # -- logging wiring -------------------------------------------------

    def attach_logging(self, console_level: int = logging.INFO) -> None:
        """Route the root logger into this run's files (and the console)."""
        root = logging.getLogger()
        self._prior_level = root.level
        self._displaced = list(root.handlers)
        root.setLevel(logging.DEBUG)
        for handler in self._displaced:
            root.removeHandler(handler)

        ctx_filter = _QuestionContextFilter()
        file_formatter = logging.Formatter(_FILE_FORMAT)

        console = logging.StreamHandler()
        console.setLevel(console_level)
        console.setFormatter(logging.Formatter(_CONSOLE_FORMAT))
        console.addFilter(ctx_filter)
        console.addFilter(_ConsoleQuietFilter())

        console_file = logging.FileHandler(self.console_log_path, encoding="utf-8")
        console_file.setLevel(console_level)
        console_file.setFormatter(file_formatter)
        console_file.addFilter(ctx_filter)

        run_file = logging.FileHandler(self.run_log_path, encoding="utf-8")
        run_file.setLevel(logging.DEBUG)
        run_file.setFormatter(file_formatter)
        run_file.addFilter(ctx_filter)

        per_question = _PerQuestionFileRouter(self.questions_dir)
        per_question.setFormatter(file_formatter)
        per_question.addFilter(ctx_filter)

        for handler in (console, console_file, run_file, per_question):
            root.addHandler(handler)
            self._handlers.append(handler)

        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.INFO)
        httpx_logger = logging.getLogger("httpx")
        _pin_logger_level(httpx_logger, logging.INFO)
        counter = _HttpCallCounter(logging.INFO)
        httpx_logger.addHandler(counter)
        self._handlers.append(counter)
        self._httpx_logger = httpx_logger
        self._http_counter = counter

    def log_environment(self, **extra: Any) -> None:
        """Record the run's inputs and machine so timings stay interpretable."""
        env_keys = (
            "DEFAULT_MODELS_MODEL",
            "DEFAULT_MODELS_ENDPOINT",
            "REASONING_MODEL",
            "REASONING_ENDPOINT",
            "EMBED_MODEL",
            "EMBED_ENDPOINT",
            "MODEL_NAME",
            "POSTGRES_HOST",
            "POSTGRES_PORT",
            "POSTGRES_DATABASE",
            "NEO4J_URI",
        )
        info = {
            "run_id": self.run_id,
            "started_at": self._started_at.isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "env": {k: os.environ.get(k, "") for k in env_keys},
            # What GSF *resolved*, not what .env says. Those differ whenever a
            # var is unset and a built-in default applies -- and, more
            # dangerously, whenever a pinned var silently overrides a default
            # that changed upstream. A run whose provenance is only its raw env
            # cannot answer "which embedding model produced these vectors".
            "resolved_models": _resolved_model_config(),
            **extra,
        }
        (self.dir / "run_config.json").write_text(
            json.dumps(info, indent=2, default=str), encoding="utf-8"
        )
        self.event("run_config", **{k: v for k, v in info.items() if k != "env"})

    # -- summary --------------------------------------------------------

    def write_summary(self, **extra: Any) -> dict[str, Any]:
        """Aggregate the per-question records into ``summary.json``; return it."""
        records = list(self._question_records)
        wall = self.elapsed

        def stats(values: Iterable[float]) -> dict[str, float] | None:
            data = sorted(float(v) for v in values)
            if not data:
                return None
            return {
                "count": len(data),
                "total": round(sum(data), 3),
                "mean": round(statistics.fmean(data), 3),
                "median": round(statistics.median(data), 3),
                "p95": round(data[min(len(data) - 1, int(0.95 * len(data)))], 3),
                "max": round(data[-1], 3),
            }

        def numeric(field: str) -> list[float]:
            return [r[field] for r in records if isinstance(r.get(field), (int, float))]

        phase_stats = {
            phase: stats(values)
            for phase in ("total_seconds", "agent_seconds", "scoring_seconds")
            if (values := numeric(phase))
        }

        node_times: dict[str, list[float]] = {}
        for record in records:
            for node in record.get("node_timings") or []:
                node_times.setdefault(node["node"], []).append(node["seconds"])
        node_stats = {
            name: stats(values)
            for name, values in sorted(node_times.items(), key=lambda kv: -sum(kv[1]))
        }

        total_agent = sum(numeric("agent_seconds"))
        busy = sum(numeric("total_seconds"))

        summary = {
            "run_id": self.run_id,
            "started_at": self._started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "wall_clock_seconds": round(wall, 2),
            "questions": len(records),
            "errors": sum(1 for r in records if r.get("error")),
            "questions_per_minute": round(len(records) / wall * 60, 2)
            if wall
            else None,
            # Sum of per-question durations over wall clock: how much parallelism
            # the run actually achieved, versus the --workers we asked for.
            "effective_parallelism": round(busy / wall, 2) if wall else None,
            "agent_share_of_busy_time": round(total_agent / busy, 3) if busy else None,
            "phases": phase_stats,
            "agent_nodes": node_stats,
            "http_calls": stats(numeric("http_calls")),
            # Run-wide totals by endpoint, including the embedding calls that
            # nemo_retriever issues off-thread and that no question can claim.
            "http_endpoints": dict(
                sorted(
                    (
                        self._http_counter.endpoints if self._http_counter else {}
                    ).items(),
                    key=lambda kv: -kv[1],
                )
            ),
            # Local resource ceiling. If peak RSS climbs with worker count and
            # the host starts swapping, the slowdown is this machine, not the
            # endpoint -- the two are indistinguishable from wall clock alone.
            "peak_rss_mb": peak_rss_mb(),
            # Average and max CPU / RAM for this run, process and system-wide.
            "resources": self.resources.stats(),
            # Repeated from run_config.json so each level's summary is
            # self-describing: comparing levels across a sweep is meaningless if
            # one of them silently ran a different embedding model.
            "resolved_models": _resolved_model_config(),
            "system_memory_start": self._memory_at_start,
            "system_memory_end": system_memory(),
            "http_calls_unattributed": (
                self._http_counter.unattributed if self._http_counter else 0
            ),
            # Throttling evidence. Any 429 here means the concurrency level
            # outran the endpoint's quota, which is the difference between
            # "slower under load" and "rate limited".
            "http_statuses": dict(
                sorted(
                    (self._http_counter.statuses if self._http_counter else {}).items()
                )
            ),
            "http_non_ok": dict(
                sorted(
                    (self._http_counter.non_ok if self._http_counter else {}).items(),
                    key=lambda kv: -kv[1],
                )
            ),
            "http_calls_per_question": (
                round(
                    sum(
                        (
                            self._http_counter.endpoints if self._http_counter else {}
                        ).values()
                    )
                    / len(records),
                    1,
                )
                if records
                else None
            ),
            "slowest_questions": [
                {
                    "qid": r.get("qid"),
                    "row_index": r.get("row_index"),
                    "total_seconds": r.get("total_seconds"),
                    "agent_seconds": r.get("agent_seconds"),
                    "slowest_node": r.get("slowest_node"),
                    "slowest_node_seconds": r.get("slowest_node_seconds"),
                }
                for r in sorted(
                    records, key=lambda r: -(r.get("total_seconds") or 0.0)
                )[:10]
            ],
            **extra,
        }
        self.summary_path.write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        return summary

    def close(self) -> None:
        self.resources.stop()
        root = logging.getLogger()
        if self._httpx_logger is not None:
            _unpin_logger_level(self._httpx_logger)
        for handler in self._handlers:
            try:
                root.removeHandler(handler)
                if self._httpx_logger is not None:
                    self._httpx_logger.removeHandler(handler)
                handler.close()
            except Exception:  # pragma: no cover
                pass
        self._handlers.clear()
        for handler in self._displaced:
            root.addHandler(handler)
        self._displaced.clear()
        if self._prior_level is not None:
            root.setLevel(self._prior_level)
        with self._write_lock:
            for stream in (self._events, self._questions):
                try:
                    stream.close()
                except Exception:  # pragma: no cover
                    pass


def setup_run_logging(
    log_dir: Path,
    run_id: str | None = None,
    console_level: int = logging.INFO,
) -> RunLogger:
    """Create a :class:`RunLogger` under *log_dir* and wire up the root logger."""
    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    runner = RunLogger(log_dir, run_id)
    runner.attach_logging(console_level=console_level)
    return runner
