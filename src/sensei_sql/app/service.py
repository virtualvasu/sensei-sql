"""App logic, independent of the UI: databases, schema cache, generate -> execute -> self-correct."""

from __future__ import annotations

import re
import shutil
import sqlite3
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from sensei_sql.app.backends import Backend
from sensei_sql.core.executor import ExecResult, run_sql
from sensei_sql.core.prompts import correction_messages, student_messages
from sensei_sql.core.schema import connect_read_only, format_schema
from sensei_sql.paths import DATA, SPIDER

UPLOADS = DATA / "uploads"


@dataclass
class Answer:
    question: str
    db: str
    sql: str
    result: ExecResult
    gen_ms: float = 0.0
    attempts: list[dict] = field(default_factory=list)  # [{"sql", "error"}] for failed tries

    @property
    def status(self) -> str:
        if not self.sql:
            return "no_sql"
        if not self.result.ok:
            if self.result.error.startswith("rejected"):
                return "rejected"
            if self.result.error.startswith("timeout"):
                return "timeout"
            return "error"
        return "empty" if not self.result.rows else "ok"


def spider_databases() -> dict[str, Path]:
    dbs = {}
    for split_dir in ("database", "test_database"):
        for p in sorted((SPIDER / split_dir).glob("*/*.sqlite")):
            dbs.setdefault(p.stem, p)
    return dbs


def check_sqlite(path: Path) -> None:
    """Raise ValueError unless ``path`` is a readable SQLite database with at least one table."""
    try:
        con = connect_read_only(path)
        n = con.execute("SELECT count(*) FROM sqlite_master WHERE type = 'table'").fetchone()[0]
        con.close()
    except sqlite3.Error as e:
        raise ValueError(f"not a SQLite database: {e}") from e
    if n == 0:
        raise ValueError("database has no tables")


class SQLAssistant:
    def __init__(self, backend: Backend, timeout_s: float = 10.0, max_rows: int = 1000, self_correct: bool = True):
        self.backend = backend
        self.timeout_s = timeout_s
        self.max_rows = max_rows
        self.self_correct = self_correct
        self.databases: dict[str, Path] = spider_databases()
        if UPLOADS.exists():
            self.databases.update({p.stem: p for p in sorted(UPLOADS.glob("*.sqlite"))})

    # --- databases ------------------------------------------------------------------------

    def add_upload(self, src: str | Path, name: str | None = None) -> str:
        """Copy an uploaded SQLite file into data/uploads/ and register it; returns its name."""
        src = Path(src)
        check_sqlite(src)
        base = re.sub(r"[^A-Za-z0-9_]+", "_", name or src.stem).strip("_") or "upload"
        name, i = f"upload_{base}", 1
        while name in self.databases and self.databases[name].resolve() != src.resolve():
            i += 1
            name = f"upload_{base}_{i}"
        UPLOADS.mkdir(parents=True, exist_ok=True)
        dest = UPLOADS / f"{name}.sqlite"
        if src.resolve() != dest.resolve():
            shutil.copyfile(src, dest)
        self.databases[name] = dest
        return name

    def schema(self, db: str) -> str:
        return _cached_schema(str(self.databases[db]), self.databases[db].stat().st_mtime)

    # --- answering ------------------------------------------------------------------------

    def run(self, db: str, sql: str) -> ExecResult:
        return run_sql(self.databases[db], sql, timeout_s=self.timeout_s, max_rows=self.max_rows)

    def ask(self, db: str, question: str) -> Answer:
        messages = student_messages(self.schema(db), question.strip())
        gen = self.backend.generate_sql(messages)
        answer = Answer(question=question, db=db, sql=gen.sql, result=self._run_or_empty(db, gen.sql), gen_ms=gen.gen_ms)
        if self.self_correct and not answer.result.ok and not answer.result.error.startswith("timeout"):
            answer.attempts.append({"sql": gen.sql, "error": answer.result.error})
            retry = self.backend.generate_sql(correction_messages(messages, gen.sql, answer.result.error))
            answer.sql = retry.sql
            answer.gen_ms += retry.gen_ms
            answer.result = self._run_or_empty(db, retry.sql)
        return answer

    def _run_or_empty(self, db: str, sql: str) -> ExecResult:
        if not sql:
            return ExecResult(ok=False, error="the model did not produce a SQL query")
        return self.run(db, sql)


@lru_cache(maxsize=256)
def _cached_schema(path: str, mtime: float) -> str:
    return format_schema(path)


def timing_text(gen_ms: float | None, exec_ms: float) -> str:
    parts = [f"generation {gen_ms:.0f} ms"] if gen_ms is not None else []
    parts.append(f"execution {exec_ms:.0f} ms")
    return " · ".join(parts)


def now() -> str:
    return time.strftime("%H:%M:%S")
