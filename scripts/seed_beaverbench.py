# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download BEAVER and populate ``datasets/beaverbench/``.

Fetches questions from HuggingFace ``beaverbench/beaver-query``, optional table
metadata from ``beaverbench/beaver-table``, and the gated MySQL dump zip from
``beaverbench/beaver-table`` (``beaver_db.zip``). Writes::

    datasets/beaverbench/evaluation.json
    datasets/beaverbench/<db_id>/metadata.json   # when --with-metadata
    datasets/beaverbench/dumps/<db_id>.sql

Optionally imports the SQL dumps into a local MySQL (default: Docker container
``beaver-mysql`` on port 3306) and writes ``CONNECTION_STRINGS`` into ``.env``.

Usage::

    # Questions + dumps only (default: dw, 100-question sample, seed 77)
    uv run python scripts/seed_beaverbench.py

    # Full dw split, import into MySQL, update .env
    uv run python scripts/seed_beaverbench.py --sample 0 --domains dw --import-mysql

    # Tasks only (no dump download / import)
    uv run python scripts/seed_beaverbench.py --tasks-only

After seeding::

    uv run python main.py --database-name beaverbench
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

logger = logging.getLogger(__name__)

DOMAINS = ("dw", "nova", "neutron", "dw_real")
# dw_real questions execute against the same physical ``dw`` database.
DOMAIN_DB: dict[str, str] = {
    "dw": "dw",
    "nova": "nova",
    "neutron": "neutron",
    "dw_real": "dw",
}
DUMP_MEMBERS = {
    "dw": "beaver_db/dw.sql",
    "nova": "beaver_db/nova.sql",
    "neutron": "beaver_db/neutron.sql",
}
HF_QUERY_DATASET = "beaverbench/beaver-query"
HF_TABLE_DATASET = "beaverbench/beaver-table"
HF_DUMP_REPO = "beaverbench/beaver-table"
HF_DUMP_FILE = "beaver_db.zip"
DEFAULT_SAMPLE = 100
DEFAULT_SAMPLE_SEED = 77
DEFAULT_MYSQL_CONTAINER = "beaver-mysql"
DEFAULT_MYSQL_IMAGE = "mysql:8.0"
DEFAULT_MYSQL_PORT = 3306
DEFAULT_MYSQL_USER = "root"
DEFAULT_MYSQL_PASSWORD = "beaver-benchmark"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_dest() -> Path:
    return _repo_root() / "datasets" / "beaverbench"


def _parse_if_string(val: Any, default_type: type = list) -> Any:
    if isinstance(val, str):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return default_type()
    if val is None:
        return default_type()
    return val


def _evidence_from_domain_knowledge(domain_knowledge: Any) -> str:
    items = _parse_if_string(domain_knowledge, list)
    if not isinstance(items, list):
        return ""
    lines = [str(item).strip() for item in items if str(item).strip()]
    return "\n".join(lines)


def _row_from_entry(entry: dict[str, Any], *, domain: str) -> dict[str, Any]:
    db_id = str(entry.get("db") or DOMAIN_DB[domain]).strip() or DOMAIN_DB[domain]
    return {
        "question_id": entry.get("id"),
        "db_id": db_id,
        "question": entry.get("question") or "",
        "evidence": _evidence_from_domain_knowledge(entry.get("domain_knowledge")),
        "SQL": entry.get("sql") or "",
        "difficulty": entry.get("detailed_category") or entry.get("category") or "",
        "category": entry.get("category") or "",
        "contains_domain_knowledge": entry.get("contains_domain_knowledge"),
        "answer_raw": "",
        "answer": "",
    }


def _load_query_rows(domains: Iterable[str]) -> list[dict[str, Any]]:
    from datasets import load_dataset

    rows: list[dict[str, Any]] = []
    for domain in domains:
        logger.info("Loading beaver-query split %r ...", domain)
        split = load_dataset(HF_QUERY_DATASET, split=domain)
        for entry in split:
            rows.append(_row_from_entry(dict(entry), domain=domain))
        logger.info("  %s: %d question(s)", domain, len(split))
    return rows


def _sample_rows(
    rows: list[dict[str, Any]], *, sample: int, seed: int
) -> list[dict[str, Any]]:
    if sample <= 0 or sample >= len(rows):
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, sample)


def _write_evaluation_json(dest: Path, rows: list[dict[str, Any]]) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "evaluation.json"
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    logger.info("Wrote %d question(s) to %s", len(rows), path)
    return path


def _write_metadata_from_tables(dest: Path, db_ids: set[str]) -> None:
    """Write per-db ``metadata.json`` from beaver-table example columns."""
    from datasets import load_dataset

    table_ds = load_dataset(HF_TABLE_DATASET)
    for db_id in sorted(db_ids):
        split_name = db_id if db_id in table_ds else None
        if split_name is None:
            logger.warning("No beaver-table split for db_id=%s; skip metadata", db_id)
            continue
        metadata: dict[str, Any] = {}
        for entry in table_ds[split_name]:
            table_name = str(entry.get("table_name") or "").strip()
            if not table_name:
                continue
            column_names = _parse_if_string(entry.get("column_names"), list)
            example_columns = _parse_if_string(entry.get("example_columns"), dict)
            columns = []
            for name in column_names:
                examples = (
                    example_columns.get(name)
                    if isinstance(example_columns, dict)
                    else None
                )
                if isinstance(examples, list):
                    value_examples = [str(v) for v in examples[:5]]
                elif examples is None:
                    value_examples = None
                else:
                    value_examples = [str(examples)]
                columns.append(
                    {
                        "name": str(name),
                        "description": "",
                        "value_examples": value_examples,
                    }
                )
            metadata[table_name] = {"description": "", "columns": columns}
        db_dir = dest / db_id
        db_dir.mkdir(parents=True, exist_ok=True)
        out = db_dir / "metadata.json"
        out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
        logger.info(
            "Wrote metadata for %s (%d table(s)) -> %s", db_id, len(metadata), out
        )


def _download_dump_zip(cache_dir: Path, *, force: bool = False) -> Path:
    """Download ``beaver_db.zip`` via huggingface_hub into *cache_dir*."""
    from huggingface_hub import hf_hub_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / HF_DUMP_FILE
    if dest.exists() and not force:
        logger.info("Using cached dump zip at %s", dest)
        return dest

    logger.info("Downloading %s from %s ...", HF_DUMP_FILE, HF_DUMP_REPO)
    path = hf_hub_download(
        repo_id=HF_DUMP_REPO,
        filename=HF_DUMP_FILE,
        repo_type="dataset",
        local_dir=str(cache_dir),
        force_download=force,
    )
    return Path(path)


def _extract_dumps(
    zip_path: Path, dest: Path, db_ids: set[str], *, force: bool = False
) -> list[Path]:
    dumps_dir = dest / "dumps"
    dumps_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        for db_id in sorted(db_ids):
            member = DUMP_MEMBERS.get(db_id)
            if member is None:
                logger.warning("No SQL dump member for db_id=%s", db_id)
                continue
            target = dumps_dir / f"{db_id}.sql"
            if target.exists() and not force:
                logger.info("Keeping existing dump %s", target)
                extracted.append(target)
                continue
            logger.info("Extracting %s -> %s", member, target)
            with zf.open(member) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
            extracted.append(target)
            logger.info("  %s (%d bytes)", target.name, target.stat().st_size)
    return extracted


def _mysql_uri(*, user: str, password: str, host: str, port: int, database: str) -> str:
    return (
        f"mysql://{quote(user, safe='')}:{quote(password, safe='')}"
        f"@{host}:{port}/{quote(database, safe='')}"
    )


def _update_env_file(env_path: Path, connection_strings_value: str) -> bool:
    if not env_path.exists():
        return False
    new_line = f"CONNECTION_STRINGS={connection_strings_value}"
    lines = env_path.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.startswith("CONNECTION_STRINGS="):
            lines[i] = new_line
            break
    else:
        if lines and lines[-1] != "":
            lines.append("")
        lines.append(new_line)
    env_path.write_text("\n".join(lines) + "\n")
    return True


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def _ensure_mysql_container(
    *,
    container: str,
    image: str,
    port: int,
    password: str,
) -> None:
    if not _docker_available():
        raise RuntimeError("docker is required for --import-mysql")

    inspect = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container],
        capture_output=True,
        text=True,
        check=False,
    )
    if inspect.returncode == 0 and inspect.stdout.strip() == "true":
        logger.info("MySQL container %s already running", container)
        return

    if inspect.returncode == 0:
        logger.info("Starting existing MySQL container %s", container)
        subprocess.run(["docker", "start", container], check=True)
    else:
        logger.info(
            "Creating MySQL container %s (%s) on port %d", container, image, port
        )
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                container,
                "-e",
                f"MYSQL_ROOT_PASSWORD={password}",
                "-p",
                f"{port}:3306",
                image,
                "--character-set-server=utf8mb4",
                "--collation-server=utf8mb4_unicode_ci",
            ],
            check=True,
        )

    logger.info("Waiting for MySQL to accept connections ...")
    deadline = time.time() + 120
    while time.time() < deadline:
        ping = subprocess.run(
            [
                "docker",
                "exec",
                container,
                "mysqladmin",
                "ping",
                "-h",
                "127.0.0.1",
                "-uroot",
                f"-p{password}",
                "--silent",
            ],
            capture_output=True,
            check=False,
        )
        if ping.returncode == 0:
            logger.info("MySQL is ready")
            return
        time.sleep(2)
    raise RuntimeError(f"MySQL container {container} did not become ready in time")


def _import_dump(
    *,
    container: str,
    password: str,
    database: str,
    dump_path: Path,
) -> None:
    logger.info("Importing %s into MySQL database %s ...", dump_path.name, database)
    # Dumps start with ``CREATE DATABASE <db>; USE <db>;`` — drop only, then let
    # the dump recreate the schema.
    subprocess.run(
        [
            "docker",
            "exec",
            "-i",
            container,
            "mysql",
            "-uroot",
            f"-p{password}",
            "-e",
            f"DROP DATABASE IF EXISTS `{database}`;",
        ],
        check=True,
    )
    with dump_path.open("rb") as dump_file:
        proc = subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                container,
                "mysql",
                "-uroot",
                f"-p{password}",
            ],
            stdin=dump_file,
            capture_output=True,
            check=False,
        )
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace")[-2000:]
        raise RuntimeError(f"Failed to import {dump_path.name}: {stderr}")
    logger.info("  imported %s", database)


def seed_beaverbench(
    *,
    dest: Path | None = None,
    domains: list[str] | None = None,
    sample: int = DEFAULT_SAMPLE,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    tasks_only: bool = False,
    with_metadata: bool = True,
    import_mysql: bool = False,
    force: bool = False,
    write_env: bool = True,
    mysql_host: str = "localhost",
    mysql_port: int = DEFAULT_MYSQL_PORT,
    mysql_user: str = DEFAULT_MYSQL_USER,
    mysql_password: str = DEFAULT_MYSQL_PASSWORD,
    mysql_container: str = DEFAULT_MYSQL_CONTAINER,
    mysql_image: str = DEFAULT_MYSQL_IMAGE,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Download BEAVER questions/dumps and optionally import into MySQL."""
    target = dest or _default_dest()
    selected_domains = list(domains or ["dw"])
    for domain in selected_domains:
        if domain not in DOMAINS:
            raise ValueError(f"Unknown domain {domain!r}; choose from {DOMAINS}")

    logger.info("=" * 60)
    logger.info("Seeding BEAVER into %s", target)
    logger.info("  Domains: %s", ", ".join(selected_domains))
    logger.info("  Sample : %s", "all" if sample <= 0 else sample)
    logger.info("=" * 60)

    rows = _load_query_rows(selected_domains)
    rows = _sample_rows(rows, sample=sample, seed=sample_seed)
    _write_evaluation_json(target, rows)

    db_ids = sorted({str(row["db_id"]) for row in rows})
    if with_metadata:
        try:
            _write_metadata_from_tables(target, set(db_ids))
        except Exception:
            logger.exception("Failed to write metadata; continuing without it")

    if tasks_only:
        logger.info("--tasks-only: skipping dump download / MySQL import")
        return rows, db_ids

    cache_dir = target / ".cache"
    zip_path = _download_dump_zip(cache_dir, force=force)
    dumps = _extract_dumps(zip_path, target, set(db_ids), force=force)

    if import_mysql:
        _ensure_mysql_container(
            container=mysql_container,
            image=mysql_image,
            port=mysql_port,
            password=mysql_password,
        )
        for dump_path in dumps:
            _import_dump(
                container=mysql_container,
                password=mysql_password,
                database=dump_path.stem,
                dump_path=dump_path,
            )

        conn_strings = [
            _mysql_uri(
                user=mysql_user,
                password=mysql_password,
                host=mysql_host,
                port=mysql_port,
                database=db_id,
            )
            for db_id in db_ids
        ]
        connection_strings_value = ",".join(conn_strings)
        env_path = _repo_root() / ".env"
        if write_env and _update_env_file(env_path, connection_strings_value):
            logger.info("Wrote CONNECTION_STRINGS to %s", env_path)
        else:
            logger.info("Add this to your .env:")
            logger.info("CONNECTION_STRINGS=%s", connection_strings_value)

    logger.info("=" * 60)
    logger.info("BEAVER seed complete.")
    logger.info("  Questions : %d", len(rows))
    logger.info("  Databases : %s", ", ".join(db_ids))
    logger.info("  Dumps     : %s", target / "dumps")
    if not import_mysql:
        logger.info(
            "Next: re-run with --import-mysql (requires Docker), then:\n"
            "  uv run python main.py --database-name beaverbench"
        )
    else:
        logger.info("Next:\n  uv run python main.py --database-name beaverbench")
    logger.info("=" * 60)
    return rows, db_ids


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download BEAVER (beaverbench) questions + MySQL dumps into "
            "datasets/beaverbench/ for ontology-sql-eval."
        )
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["dw"],
        choices=list(DOMAINS),
        help="Query splits to include (default: dw). dw_real uses the dw database.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=DEFAULT_SAMPLE,
        help=(
            f"Sample size across selected domains (default: {DEFAULT_SAMPLE}; "
            "0 = all questions)."
        ),
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=DEFAULT_SAMPLE_SEED,
        help=f"RNG seed for --sample (default: {DEFAULT_SAMPLE_SEED}).",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination directory (default: datasets/beaverbench/).",
    )
    parser.add_argument(
        "--tasks-only",
        action="store_true",
        help="Write evaluation.json (+ metadata) only; skip dump download/import.",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Skip writing per-db metadata.json from beaver-table.",
    )
    parser.add_argument(
        "--import-mysql",
        action="store_true",
        help="Start/use a Docker MySQL and import extracted dumps.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download / re-extract dumps even if present.",
    )
    parser.add_argument(
        "--no-write-env",
        action="store_true",
        help="Don't write CONNECTION_STRINGS into .env.",
    )
    parser.add_argument(
        "--mysql-host", default=os.environ.get("MYSQL_HOST", "localhost")
    )
    parser.add_argument(
        "--mysql-port",
        type=int,
        default=int(os.environ.get("MYSQL_PORT", str(DEFAULT_MYSQL_PORT))),
    )
    parser.add_argument(
        "--mysql-user", default=os.environ.get("MYSQL_USER", DEFAULT_MYSQL_USER)
    )
    parser.add_argument(
        "--mysql-password",
        default=os.environ.get("MYSQL_PASSWORD", DEFAULT_MYSQL_PASSWORD),
    )
    parser.add_argument("--mysql-container", default=DEFAULT_MYSQL_CONTAINER)
    parser.add_argument("--mysql-image", default=DEFAULT_MYSQL_IMAGE)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    seed_beaverbench(
        dest=args.dest,
        domains=args.domains,
        sample=args.sample,
        sample_seed=args.sample_seed,
        tasks_only=args.tasks_only,
        with_metadata=not args.no_metadata,
        import_mysql=args.import_mysql,
        force=args.force,
        write_env=not args.no_write_env,
        mysql_host=args.mysql_host,
        mysql_port=args.mysql_port,
        mysql_user=args.mysql_user,
        mysql_password=args.mysql_password,
        mysql_container=args.mysql_container,
        mysql_image=args.mysql_image,
    )


if __name__ == "__main__":
    main()
