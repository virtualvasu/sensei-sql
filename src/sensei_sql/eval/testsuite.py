"""Thin wrapper around third_party/test-suite-sql-eval (taoyds), the official Spider scorer.

``evaluation.evaluate()`` only prints aggregate tables, so we call the same per-example pieces
it uses (``eval_hardness`` and ``eval_exec_match``) to get identical scores plus a per-example
verdict. Before any prediction reaches the scorer it goes through our safe runner (#2): the
scorer executes SQL on a writable connection and its timeout cannot interrupt SQLite, so a
predicted ``DROP``/``UPDATE`` or a runaway query could otherwise damage or hang the run.
Predictions that are rejected, error, or time out score 0, exactly as the scorer treats
execution exceptions.
"""

from __future__ import annotations

import contextlib
import io
import sys
from functools import lru_cache
from pathlib import Path

from sensei_sql.core.executor import run_sql
from sensei_sql.paths import TEST_SUITE

HARDNESS_LEVELS = ("easy", "medium", "hard", "extra")


@lru_cache(maxsize=1)
def _modules():
    if not (TEST_SUITE / "evaluation.py").exists():
        raise FileNotFoundError(f"{TEST_SUITE} missing; run scripts/download.py")
    sys.path.insert(0, str(TEST_SUITE))
    import evaluation
    import exec_eval
    import process_sql

    return evaluation, exec_eval, process_sql


@lru_cache(maxsize=None)
def _schema(db_path: str):
    _, _, process_sql = _modules()
    return process_sql.Schema(process_sql.get_schema(db_path))


def hardness(db_path: str | Path, gold_sql: str) -> str:
    """Spider hardness of ``gold_sql``: easy / medium / hard / extra."""
    evaluation, _, process_sql = _modules()
    return evaluation.Evaluator().eval_hardness(process_sql.get_sql(_schema(str(db_path)), gold_sql))


def exec_match(db_path: str | Path, gold_sql: str, pred_sql: str, timeout_s: float) -> bool:
    """Official execution-accuracy verdict for one prediction (1 = correct)."""
    _, exec_eval, _ = _modules()
    pred_sql = pred_sql.replace("value", "1")  # as evaluation.evaluate() does
    if not pred_sql.strip() or not run_sql(db_path, pred_sql, timeout_s=timeout_s, max_rows=10**9).ok:
        return False
    with contextlib.redirect_stdout(io.StringIO()):
        return bool(
            exec_eval.eval_exec_match(
                db=str(db_path),
                p_str=pred_sql,
                g_str=gold_sql,
                plug_value=False,
                keep_distinct=False,
                progress_bar_for_each_datapoint=False,
            )
        )


def revision() -> str:
    rev = TEST_SUITE / "REVISION"
    return rev.read_text().strip() if rev.exists() else "unknown"


