"""HTTP API and static file server for the web app (``frontend/``).

    GET  /api/health      backend name, title
    GET  /api/databases   database names (+ which are uploads)
    GET  /api/schema      ?db=<name>  -> schema text shown to the model
    GET  /api/examples    ?db=<name>  -> suggested questions for that database
    POST /api/ask         {db, question} -> generated SQL, status, result table, timings
    POST /api/run         {db, sql}      -> run edited SQL through the same safe runner
    POST /api/upload      multipart file -> register a SQLite file as a database

Everything else is served from ``frontend/dist`` (the built React app), with ``index.html`` as the
fallback so client-side routes work on reload.
"""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sensei_sql.app.service import Answer, SQLAssistant
from sensei_sql.core.executor import ExecResult
from sensei_sql.paths import ROOT

DIST = ROOT / "frontend" / "dist"
MAX_UPLOAD_BYTES = 200 * 2**20


class AskRequest(BaseModel):
    db: str
    question: str


class RunRequest(BaseModel):
    db: str
    sql: str


def json_value(v):
    """Make a SQLite value JSON-safe (blobs become a short placeholder)."""
    if isinstance(v, (bytes, bytearray, memoryview)):
        return f"<{len(v)} bytes>"
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return str(v)


def unique_columns(columns: list[str]) -> list[str]:
    out: list[str] = []
    for i, c in enumerate(columns):
        out.append(c if c not in columns[:i] else f"{c}_{i}")
    return out


def result_payload(result: ExecResult) -> dict:
    ok = result.ok
    return {
        "columns": unique_columns(result.columns) if ok else [],
        "rows": [[json_value(v) for v in row] for row in result.rows] if ok else [],
        "row_count": len(result.rows) if ok else 0,
        "truncated": result.truncated,
        "error": result.error,
        "exec_ms": round(result.elapsed_ms, 1),
    }


def create_app(assistant: SQLAssistant, title: str = "sensei-sql", dist: Path = DIST) -> FastAPI:
    app = FastAPI(title=title, docs_url=None, redoc_url=None, openapi_url=None)
    generate_lock = threading.Lock()  # the model backends are not thread-safe

    def known(db: str) -> str:
        if db not in assistant.databases:
            raise HTTPException(404, f"unknown database {db!r}")
        return db

    @app.get("/api/health")
    def health():
        return {"title": title, "backend": assistant.backend.name, "self_correct": assistant.self_correct}

    @app.get("/api/databases")
    def databases():
        names = sorted(assistant.databases)
        default = "concert_singer" if "concert_singer" in names else (names[0] if names else None)
        return {
            "databases": [{"name": n, "upload": n.startswith("upload_")} for n in names],
            "default": default,
        }

    @app.get("/api/schema")
    def schema(db: str):
        return {"db": known(db), "schema": assistant.schema(db)}

    @app.get("/api/examples")
    def examples(db: str):
        return {"db": known(db), "examples": assistant.examples(db)}

    @app.post("/api/ask")
    def ask(req: AskRequest):
        known(req.db)
        if not req.question.strip():
            raise HTTPException(400, "empty question")
        with generate_lock:
            ans: Answer = assistant.ask(req.db, req.question)
        return {
            "sql": ans.sql,
            "status": ans.status,
            "gen_ms": round(ans.gen_ms, 1),
            "attempts": ans.attempts,
            **result_payload(ans.result),
        }

    @app.post("/api/run")
    def run(req: RunRequest):
        known(req.db)
        if not req.sql.strip():
            raise HTTPException(400, "empty SQL")
        res = assistant.run(req.db, req.sql)
        return {"sql": req.sql, "status": Answer("", req.db, req.sql, res).status, "gen_ms": None, "attempts": [], **result_payload(res)}

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / Path(file.filename or "upload.sqlite").name
            size = 0
            with dest.open("wb") as f:
                while chunk := await file.read(1 << 20):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "file too large (limit 200 MB)")
                    f.write(chunk)
            try:
                name = assistant.add_upload(dest)
            except ValueError as e:
                raise HTTPException(400, str(e)) from e
        return {"name": name}

    @app.get("/api/{path:path}")
    def unknown_api(path: str):
        raise HTTPException(404, "not found")

    if (dist / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}")
        def spa(path: str):
            target = (dist / path).resolve()
            if path and target.is_file() and dist.resolve() in target.parents:
                return FileResponse(target)
            return FileResponse(dist / "index.html")
    else:

        @app.get("/")
        def missing_frontend():
            return {"error": "frontend not built: run `make frontend` (or `cd frontend && npm ci && npm run build`)"}

    return app
