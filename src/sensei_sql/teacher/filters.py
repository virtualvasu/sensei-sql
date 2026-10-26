"""CPU-side verification of teacher pairs: parse, execute, emptiness, normalisation, dedup."""

from __future__ import annotations

import random
import re
from collections import defaultdict
from pathlib import Path

import sqlglot
from sqlglot.errors import SqlglotError

from sensei_sql.core.compare import has_order_by, results_match
from sensei_sql.core.executor import ExecResult, UnsafeSQLError, run_sql, validate_sql

# Stage names, in pipeline order; a candidate's "failed" field holds the first one it failed.
PARSE, EXECUTE, EMPTY, CONSISTENCY, GOLD, DUPLICATE = (
    "parse", "execute", "empty", "self_consistency", "gold_check", "duplicate"
)


def parses(sql: str) -> bool:
    """SQL parses with sqlglot (SQLite dialect) as a single read-only query."""
    if not sql:
        return False
    try:
        validate_sql(sql)
    except UnsafeSQLError:
        return False
    return True


def execute(db_path: Path, sql: str, timeout_s: float, max_rows: int) -> ExecResult:
    return run_sql(db_path, sql, timeout_s=timeout_s, max_rows=max_rows)


def is_empty(result: ExecResult) -> bool:
    """No rows, or every value NULL (e.g. an aggregate over no rows)."""
    return not result.rows or all(v is None for row in result.rows for v in row)


def same_result(reference_sql: str, reference: ExecResult, other: ExecResult) -> bool:
    return (
        other.ok
        and other.truncated == reference.truncated
        and results_match(reference.rows, other.rows, ordered=has_order_by(reference_sql))
    )


def cap_empty(candidates: list[dict], max_fraction: float, seed: int) -> None:
    """Among passing candidates flagged ``empty``, keep at most ``max_fraction`` of the
    non-empty count (chosen with ``seed``); mark the rest failed at the EMPTY stage."""
    empty = [c for c in candidates if c["failed"] is None and c.get("empty")]
    nonempty = sum(1 for c in candidates if c["failed"] is None and not c.get("empty"))
    allowed = int(max_fraction * nonempty)
    random.Random(seed).shuffle(empty)
    for c in empty[allowed:]:
        c["failed"] = EMPTY


_PUNCT = re.compile(r"[^\w\s]")


def normalise_question(q: str) -> str:
    return " ".join(_PUNCT.sub(" ", q.lower()).split())


def normalise_sql(sql: str) -> str:
    try:
        return sqlglot.parse_one(sql, read="sqlite").sql(dialect="sqlite", normalize=True).lower()
    except SqlglotError:
        return " ".join(sql.lower().split())


def deduplicate(pairs: list[dict], max_questions_per_sql: int) -> None:
    """Within each database, mark repeated questions and the overflow beyond
    ``max_questions_per_sql`` per normalised SQL as DUPLICATE. Earlier pairs win, so put
    the preferred source first."""
    seen_q: set[tuple[str, str]] = set()
    per_sql: dict[tuple[str, str], int] = defaultdict(int)
    for p in pairs:
        if p["failed"] is not None:
            continue
        qk = (p["db_id"], normalise_question(p["question"]))
        sk = (p["db_id"], normalise_sql(p["sql"]))
        if qk in seen_q or per_sql[sk] >= max_questions_per_sql:
            p["failed"] = DUPLICATE
            continue
        seen_q.add(qk)
        per_sql[sk] += 1
