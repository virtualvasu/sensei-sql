import sqlite3

import pytest
from conftest import requires_spider

from sensei_sql.app import service
from sensei_sql.app.backends import Backend
from sensei_sql.app.service import SQLAssistant, check_sqlite


class FakeBackend(Backend):
    """Returns scripted completions and records the prompts it saw."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.seen = []

    def _complete(self, messages):
        self.seen.append(messages)
        return self.outputs.pop(0)


@pytest.fixture
def assistant(tiny_db, tmp_path, monkeypatch):
    monkeypatch.setattr(service, "UPLOADS", tmp_path / "uploads")
    monkeypatch.setattr(service, "spider_databases", lambda: {"tiny": tiny_db})
    return lambda outputs, **kw: SQLAssistant(FakeBackend(outputs), **kw)


def test_ask_ok(assistant):
    a = assistant(["```sql\nSELECT count(*) FROM animal;\n```"])
    ans = a.ask("tiny", "How many animals?")
    assert ans.sql == "SELECT count(*) FROM animal"
    assert ans.status == "ok" and ans.result.rows == [(20,)]
    assert ans.gen_ms >= 0 and ans.result.elapsed_ms >= 0
    prompt = a.backend.seen[0]
    assert prompt[-1]["content"].endswith("Question: How many animals?")
    assert "CREATE TABLE animal" in prompt[-1]["content"]


def test_self_correction_retries_once_with_error(assistant):
    a = assistant(["SELECT nope FROM animal", "SELECT count(*) FROM animal"])
    ans = a.ask("tiny", "How many animals?")
    assert ans.status == "ok" and ans.result.rows == [(20,)]
    assert ans.attempts == [{"sql": "SELECT nope FROM animal", "error": "no such column: nope"}]
    retry_prompt = a.backend.seen[1]
    assert retry_prompt[-2] == {"role": "assistant", "content": "SELECT nope FROM animal"}
    assert "no such column: nope" in retry_prompt[-1]["content"]


def test_self_correction_gives_up_after_one_retry(assistant):
    a = assistant(["SELECT nope FROM animal", "SELECT still_nope FROM animal"])
    ans = a.ask("tiny", "q")
    assert ans.status == "error" and len(a.backend.seen) == 2


def test_unsafe_sql_never_reaches_db(assistant, tiny_db):
    from sensei_sql.app.service import Answer

    # model output: extract_sql only ever pulls out a SELECT/WITH statement
    a = assistant(["DROP TABLE animal", "DELETE FROM animal"])
    assert a.ask("tiny", "delete everything").status == "no_sql"
    # user-edited SQL goes through the safe runner
    for sql in ["DROP TABLE animal", "SELECT 1; DELETE FROM animal", "UPDATE animal SET weight = 0"]:
        res = a.run("tiny", sql)
        assert Answer("", "tiny", sql, res).status == "rejected"
    assert sqlite3.connect(tiny_db).execute("SELECT count(*), max(weight) FROM animal").fetchone() == (20, 210.0)


def test_empty_and_no_sql(assistant):
    assert assistant(["SELECT * FROM animal WHERE weight < 0"]).ask("tiny", "q").status == "empty"
    a = assistant(["I don't know.", "Still no idea."])
    assert a.ask("tiny", "q").status == "no_sql"


def test_upload(assistant, tmp_path):
    a = assistant([])
    good = tmp_path / "my shop.db"
    con = sqlite3.connect(good)
    con.execute("CREATE TABLE item (id INT, price REAL)")
    con.commit()
    con.close()
    name = a.add_upload(good)
    assert name == "upload_my_shop" and "CREATE TABLE item" in a.schema(name)
    bad = tmp_path / "bad.sqlite"
    bad.write_text("not a database")
    with pytest.raises(ValueError):
        a.add_upload(bad)
    with pytest.raises(ValueError):
        check_sqlite(tmp_path / "missing.sqlite")


def test_ui_builds(assistant):
    pytest.importorskip("gradio")
    from sensei_sql.app.ui import build_ui, status_markdown

    demo = build_ui(assistant([]))
    assert demo is not None
    from sensei_sql.core.executor import ExecResult

    assert "no rows" in status_markdown("empty", ExecResult(ok=True))


@requires_spider
def test_spider_databases_listed():
    dbs = service.spider_databases()
    assert "concert_singer" in dbs and len(dbs) >= 166
