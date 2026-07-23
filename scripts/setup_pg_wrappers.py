#!/usr/bin/env python3
"""
Create Python shim scripts for psql, createdb, dropdb in /tmp/pg_wrappers.

Required because the Bird ADK db_environment service calls these PostgreSQL
client tools as subprocesses, but they are not installed on this machine.
The shims implement the same interface using psycopg2.

Run once before starting Bird ADK services:
    python scripts/setup_pg_wrappers.py
"""
import os
import stat
import sys
import textwrap
from pathlib import Path

WRAPPERS_DIR = Path("/tmp/pg_wrappers")
PYTHON = sys.executable  # use same venv python that has psycopg2


def make_wrapper(name: str, body: str) -> None:
    path = WRAPPERS_DIR / name
    path.write_text(f"#!{PYTHON}\n{textwrap.dedent(body)}")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"  Created {path}")


def main() -> None:
    WRAPPERS_DIR.mkdir(parents=True, exist_ok=True)

    make_wrapper("psql", """
        import sys, os, psycopg2
        args = sys.argv[1:]
        host, port, user, dbname, cmd = "127.0.0.1", "5432", "root", "postgres", ""
        i = 0
        while i < len(args):
            if args[i] == "-h": host = args[i+1]; i += 2
            elif args[i] == "-p": port = args[i+1]; i += 2
            elif args[i] == "-U": user = args[i+1]; i += 2
            elif args[i] == "-d": dbname = args[i+1]; i += 2
            elif args[i] == "-c": cmd = args[i+1]; i += 2
            else: i += 1
        pw = os.environ.get("PGPASSWORD", "123123")
        conn = psycopg2.connect(dbname=dbname, user=user, password=pw, host=host, port=int(port))
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute(cmd)
        try: print(cur.fetchall())
        except Exception: pass
        conn.close()
    """)

    make_wrapper("createdb", """
        import sys, os, psycopg2
        args = sys.argv[1:]
        host, port, user, template, dbname = "127.0.0.1", "5432", "root", None, None
        i = 0
        while i < len(args):
            if args[i] == "-h": host = args[i+1]; i += 2
            elif args[i] == "-p": port = args[i+1]; i += 2
            elif args[i] == "-U": user = args[i+1]; i += 2
            elif args[i] == "--template": template = args[i+1]; i += 2
            elif not args[i].startswith("-"): dbname = args[i]; i += 1
            else: i += 1
        pw = os.environ.get("PGPASSWORD", "123123")
        conn = psycopg2.connect(dbname="postgres", user=user, password=pw, host=host, port=int(port))
        conn.autocommit = True
        cur = conn.cursor()
        sql = f'CREATE DATABASE "{dbname}"'
        if template:
            sql += f' TEMPLATE "{template}"'
        cur.execute(sql)
        conn.close()
    """)

    make_wrapper("dropdb", """
        import sys, os, psycopg2
        args = sys.argv[1:]
        host, port, user, dbname, if_exists = "127.0.0.1", "5432", "root", None, False
        i = 0
        while i < len(args):
            if args[i] == "-h": host = args[i+1]; i += 2
            elif args[i] == "-p": port = args[i+1]; i += 2
            elif args[i] == "-U": user = args[i+1]; i += 2
            elif args[i] == "--if-exists": if_exists = True; i += 1
            elif not args[i].startswith("-"): dbname = args[i]; i += 1
            else: i += 1
        pw = os.environ.get("PGPASSWORD", "123123")
        conn = psycopg2.connect(dbname="postgres", user=user, password=pw, host=host, port=int(port))
        conn.autocommit = True
        cur = conn.cursor()
        ie = "IF EXISTS " if if_exists else ""
        cur.execute(f'DROP DATABASE {ie}"{dbname}"')
        conn.close()
    """)

    print(f"\nDone. Add to PATH before starting Bird services:")
    print(f"  export PATH={WRAPPERS_DIR}:$PATH")


if __name__ == "__main__":
    main()
