"""Execution-match check between two result sets.

Rows are compared as a multiset (duplicates count) unless the reference query has a top-level
``ORDER BY``, in which case row order must match too. Column order matters. Numbers compare
with a tolerance, so ``1``, ``1.0`` and ``0.99999999`` are equal; everything else compares
exactly. (Official Spider accuracy uses test-suite-sql-eval, see #3; this check is for
filtering and self-correction.)
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from sensei_sql.core.executor import DEFAULT_TIMEOUT_S, run_sql

DEFAULT_REL_TOL = 1e-6
DEFAULT_ABS_TOL = 1e-6

Row = Sequence


def has_order_by(sql: str) -> bool:
    """True if the outermost query is ordered (``ORDER BY`` in a subquery doesn't count)."""
    try:
        stmt = sqlglot.parse_one(sql, read="sqlite")
    except SqlglotError:
        return "order by" in " ".join(sql.lower().split())
    return isinstance(stmt, exp.Query) and stmt.args.get("order") is not None


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def values_equal(a, b, rel_tol: float = DEFAULT_REL_TOL, abs_tol: float = DEFAULT_ABS_TOL) -> bool:
    if _is_number(a) and _is_number(b):
        if isinstance(a, float) and math.isnan(a):
            return isinstance(b, float) and math.isnan(b)
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    return type(a) is type(b) and a == b


def rows_equal(a: Row, b: Row, **tol) -> bool:
    return len(a) == len(b) and all(values_equal(x, y, **tol) for x, y in zip(a, b))


def _sort_key(row: Row):
    # Total order over mixed SQLite values: NULL < numbers < text < blobs.
    def key(v):
        if v is None:
            return (0, 0)
        if _is_number(v):
            return (1, float(v))
        if isinstance(v, str):
            return (2, v)
        return (3, bytes(v))

    return tuple(key(v) for v in row)


def results_match(
    gold: Sequence[Row],
    pred: Sequence[Row],
    ordered: bool = False,
    rel_tol: float = DEFAULT_REL_TOL,
    abs_tol: float = DEFAULT_ABS_TOL,
) -> bool:
    """Do two result sets match? ``ordered`` makes row order significant."""
    if len(gold) != len(pred):
        return False
    if not ordered:
        gold, pred = sorted(gold, key=_sort_key), sorted(pred, key=_sort_key)
    tol = {"rel_tol": rel_tol, "abs_tol": abs_tol}
    return all(rows_equal(g, p, **tol) for g, p in zip(gold, pred))


def execution_match(
    db_path: str | Path, gold_sql: str, pred_sql: str, timeout_s: float = DEFAULT_TIMEOUT_S
) -> bool:
    """Run both queries and compare; order matters only if ``gold_sql`` has ``ORDER BY``.
    A failing prediction never matches; a failing gold query raises ``ValueError``."""
    gold = run_sql(db_path, gold_sql, timeout_s=timeout_s)
    if not gold.ok:
        raise ValueError(f"gold SQL failed: {gold.error}")
    pred = run_sql(db_path, pred_sql, timeout_s=timeout_s)
    if not pred.ok or pred.truncated != gold.truncated:
        return False
    return results_match(gold.rows, pred.rows, ordered=has_order_by(gold_sql))
