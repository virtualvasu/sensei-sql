"""Turn a SQLite database into the schema text used in every prompt.

The output is deterministic for a given database file: tables are sorted by name, columns keep
their declared order, and sample rows come from a fixed ``LIMIT`` query. Example::

    CREATE TABLE singer (
      Singer_ID int,
      Name text,
      Country text,
      PRIMARY KEY (Singer_ID)
    );
    /* 3 example rows:
    Singer_ID | Name | Country
    1 | Joe Sharp | Netherlands
    ...
    */
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SAMPLE_ROWS = 3
DEFAULT_MAX_VALUE_CHARS = 40

_SIMPLE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def approx_token_count(text: str) -> int:
    """Cheap tokenizer-free estimate (~4 chars/token for code-ish text)."""
    return len(text) // 4 + 1


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    pk_index: int  # 1-based position in the primary key, 0 if not part of it


@dataclass(frozen=True)
class ForeignKey:
    columns: tuple[str, ...]
    ref_table: str
    ref_columns: tuple[str, ...]


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]
    foreign_keys: tuple[ForeignKey, ...]
    sample_rows: tuple[tuple, ...]


def connect_read_only(db_path: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    # Some Spider databases contain invalid UTF-8; never fail on decoding.
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")
    return con


def quote_ident(name: str) -> str:
    if _SIMPLE_IDENT.match(name):
        return name
    return '"' + name.replace('"', '""') + '"'


def read_tables(db_path: str | Path, sample_rows: int = DEFAULT_SAMPLE_ROWS) -> list[Table]:
    con = connect_read_only(db_path)
    try:
        names = [
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        tables = []
        for name in sorted(names, key=str.lower):
            q = quote_ident(name)
            columns = tuple(
                Column(name=r[1], type=(r[2] or "").strip(), pk_index=r[5])
                for r in con.execute(f"PRAGMA table_info({q})")
            )
            tables.append(
                Table(
                    name=name,
                    columns=columns,
                    foreign_keys=_read_foreign_keys(con, q, columns),
                    sample_rows=tuple(
                        con.execute(f"SELECT * FROM {q} LIMIT {int(sample_rows)}").fetchall()
                    )
                    if sample_rows > 0
                    else (),
                )
            )
        return tables
    finally:
        con.close()


def _read_foreign_keys(con: sqlite3.Connection, q: str, columns: tuple[Column, ...]) -> tuple[ForeignKey, ...]:
    # PRAGMA rows: (id, seq, table, from, to, on_update, on_delete, match). Group by id.
    groups: dict[int, list[tuple]] = {}
    for r in con.execute(f"PRAGMA foreign_key_list({q})"):
        groups.setdefault(r[0], []).append(r)
    fks = []
    for rows in groups.values():
        rows.sort(key=lambda r: r[1])
        ref_table = rows[0][2]
        # `to` is NULL when the FK references the parent's primary key implicitly.
        ref_cols = tuple(r[4] for r in rows if r[4] is not None)
        if len(ref_cols) != len(rows):
            ref_cols = _primary_key(con, ref_table) or ref_cols
        fks.append(ForeignKey(tuple(r[3] for r in rows), ref_table, ref_cols))
    return tuple(sorted(fks, key=lambda fk: (fk.columns, fk.ref_table.lower(), fk.ref_columns)))


def _primary_key(con: sqlite3.Connection, table: str) -> tuple[str, ...]:
    cols = [r for r in con.execute(f"PRAGMA table_info({quote_ident(table)})") if r[5]]
    return tuple(r[1] for r in sorted(cols, key=lambda r: r[5]))


def _format_value(v, max_chars: int) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bytes):
        return f"<blob {len(v)} bytes>"
    s = " ".join(str(v).split())
    return s if len(s) <= max_chars else s[: max_chars - 3] + "..."


def render_table(table: Table, sample_rows: int, max_value_chars: int = DEFAULT_MAX_VALUE_CHARS) -> str:
    lines = [f"CREATE TABLE {quote_ident(table.name)} ("]
    body = [f"  {quote_ident(c.name)} {c.type}".rstrip() for c in table.columns]
    pk = [c.name for c in sorted((c for c in table.columns if c.pk_index), key=lambda c: c.pk_index)]
    if pk:
        body.append(f"  PRIMARY KEY ({', '.join(map(quote_ident, pk))})")
    for fk in table.foreign_keys:
        body.append(
            f"  FOREIGN KEY ({', '.join(map(quote_ident, fk.columns))}) "
            f"REFERENCES {quote_ident(fk.ref_table)}({', '.join(map(quote_ident, fk.ref_columns))})"
        )
    lines.append(",\n".join(body))
    lines.append(");")

    rows = table.sample_rows[:sample_rows]
    if rows:
        lines.append(f"/* {len(rows)} example row{'s' if len(rows) != 1 else ''}:")
        lines.append(" | ".join(c.name for c in table.columns))
        lines.extend(" | ".join(_format_value(v, max_value_chars) for v in row) for row in rows)
        lines.append("*/")
    return "\n".join(lines)


def render_schema(tables: list[Table], sample_rows: int, max_value_chars: int = DEFAULT_MAX_VALUE_CHARS) -> str:
    return "\n\n".join(render_table(t, sample_rows, max_value_chars) for t in tables)


def format_schema(
    db_path: str | Path,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
    max_value_chars: int = DEFAULT_MAX_VALUE_CHARS,
    max_tokens: int | None = None,
    count_tokens: Callable[[str], int] = approx_token_count,
) -> str:
    """Schema text for ``db_path``: CREATE TABLEs with PK/FK plus example rows.

    If ``max_tokens`` is given and the text is over budget, sample rows are reduced one at a
    time (down to none) before anything else. Pass ``count_tokens`` (e.g.
    ``lambda s: len(tokenizer.encode(s))``) for exact counts; the default is an estimate.
    """
    tables = read_tables(db_path, sample_rows)
    for n in range(sample_rows, -1, -1):
        text = render_schema(tables, n, max_value_chars)
        if max_tokens is None or count_tokens(text) <= max_tokens:
            return text
    return text  # still over budget with no rows; structure is never dropped
