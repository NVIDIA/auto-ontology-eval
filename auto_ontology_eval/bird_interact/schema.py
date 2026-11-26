"""BIRD-Interact schema normalization for Auto Ontology clarification."""

from __future__ import annotations

import re

_CREATE_TABLE_RE = re.compile(
    r"""^\s*["']?CREATE["']?\s+TABLE\s+
        (?:(?:["']?[\w]+["']?)\.)?
        (?P<table>"(?:[^"]|"")*"|'(?:[^']|'')*'|[\w]+)\s*\(
    """,
    re.IGNORECASE | re.VERBOSE,
)
_LEADING_IDENTIFIER_RE = re.compile(
    r"""^(?P<identifier>"(?:[^"]|"")*"|'(?:[^']|'')*'|[^\s,]+)"""
)
_TABLE_CONSTRAINTS = frozenset(
    {"PRIMARY", "FOREIGN", "CONSTRAINT", "UNIQUE", "CHECK", "EXCLUDE"}
)


def _identifier(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        quote = value[0]
        return value[1:-1].replace(quote * 2, quote)
    return value


def compact_bird_schema(db_schema: str) -> str:
    """Return compact table/column lines from benchmark DDL plus sample rows.

    Six Full databases quote SQL keywords (for example ``"CREATE" TABLE`` and
    ``"PRIMARY" KEY``). They are not executable PostgreSQL, but they are the
    official schema text returned by the DB environment, so parsing accepts
    both normal and quoted forms.
    """
    result: list[str] = []
    current_table: str | None = None
    columns: list[str] = []
    in_ddl = False

    def finish_table() -> None:
        if current_table and columns:
            result.append(f"{current_table}: {', '.join(columns)}")

    for line in db_schema.splitlines():
        create = _CREATE_TABLE_RE.match(line)
        if create:
            finish_table()
            current_table = _identifier(create.group("table"))
            columns = []
            in_ddl = True
            continue
        if not in_ddl:
            continue

        stripped = line.strip()
        if stripped.startswith(")"):
            in_ddl = False
            continue
        if not stripped:
            continue
        match = _LEADING_IDENTIFIER_RE.match(stripped)
        if not match:
            continue
        column = _identifier(match.group("identifier"))
        if column.upper() in _TABLE_CONSTRAINTS:
            continue
        columns.append(column)

    finish_table()
    return "\n".join(result) or db_schema[:3000]


def install_compact_schema_parser() -> None:
    """Replace both imported Auto Ontology references to its DDL compactor."""
    try:
        import auto_ontology.retrieval.interactive.clarify as clarify
        import auto_ontology.retrieval.interactive.kb_coverage as kb_coverage
    except (ImportError, ModuleNotFoundError):
        return

    kb_coverage._compact_schema = compact_bird_schema
    clarify._compact_schema = compact_bird_schema
