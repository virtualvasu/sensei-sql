"""Run model-written SQL safely against a SQLite database.

Layers of protection:
  1. ``sqlglot`` validation: exactly one statement, and it must be a query (SELECT / WITH /
     set operation). DDL, DML, PRAGMA, ATTACH, etc. are rejected before touching the database.
  2. Read-only connection (``mode=ro`` URI) with ``PRAGMA query_only``.
  3. Wall-clock timeout via the SQLite progress handler, and a cap on returned rows.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from sensei_sql.core.schema import connect_read_only

DEFAULT_TIMEOUT_S = 5.0
DEFAULT_MAX_ROWS = 10_000
_PROGRESS_EVERY = 1_000  # SQLite VM instructions between timeout checks


@dataclass
class ExecResult:
    ok: bool
    rows: list[tuple] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    error: str | None = None
    elapsed_ms: float = 0.0
    truncated: bool = False  # more than max_rows rows were produced


class UnsafeSQLError(ValueError):
    pass


def validate_sql(sql: str) -> exp.Query:
    """Parse ``sql`` and return its single query expression, or raise :class:`UnsafeSQLError`."""
    try:
        statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    except SqlglotError as e:
        raise UnsafeSQLError(f"could not parse SQL: {e}") from e
    if len(statements) != 1:
        raise UnsafeSQLError(f"expected exactly one statement, got {len(statements)}")
    (stmt,) = statements
    if not isinstance(stmt, exp.Query):
        raise UnsafeSQLError(f"only SELECT/WITH queries are allowed, got {stmt.key.upper()}")
    # e.g. a CTE wrapping DML, which some dialect quirks can produce
    for node in stmt.walk():
        if isinstance(node, (exp.DML, exp.DDL, exp.Drop, exp.Command, exp.Pragma, exp.Attach, exp.Detach)):
            raise UnsafeSQLError(f"disallowed {node.key.upper()} inside query")
    return stmt


def run_sql(
    db_path: str | Path,
    sql: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> ExecResult:
    start = time.perf_counter()

    def done(**kw) -> ExecResult:
        return ExecResult(elapsed_ms=(time.perf_counter() - start) * 1000, **kw)

    try:
        validate_sql(sql)
    except UnsafeSQLError as e:
        return done(ok=False, error=f"rejected: {e}")

    try:
        con = connect_read_only(db_path)
    except sqlite3.Error as e:
        return done(ok=False, error=f"cannot open database: {e}")
    deadline = time.monotonic() + timeout_s
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, _PROGRESS_EVERY)
    try:
        con.execute("PRAGMA query_only = ON")
        cur = con.execute(sql)
        columns = [d[0] for d in cur.description or ()]
        rows = cur.fetchmany(max_rows + 1)
        truncated = len(rows) > max_rows
        return done(ok=True, rows=rows[:max_rows], columns=columns, truncated=truncated)
    except sqlite3.OperationalError as e:
        if time.monotonic() > deadline:
            return done(ok=False, error=f"timeout: query exceeded {timeout_s:g}s")
        return done(ok=False, error=str(e))
    except sqlite3.Error as e:
        return done(ok=False, error=str(e))
    finally:
        con.close()
