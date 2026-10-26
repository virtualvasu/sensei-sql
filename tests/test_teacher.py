import pytest
from conftest import requires_spider

from sensei_sql.core.executor import ExecResult
from sensei_sql.teacher import filters as F
from sensei_sql.teacher.pipeline import assert_no_dev_leakage, stable_seed
from sensei_sql.teacher.report import stage_counts


def cand(db="db", q="q", sql="SELECT 1", failed=None, empty=False):
    return {"db_id": db, "question": q, "sql": sql, "failed": failed, "empty": empty}


def test_parses():
    assert F.parses("SELECT a FROM t")
    assert not F.parses("")
    assert not F.parses("DROP TABLE t")
    assert not F.parses("SELECT FROM (")


def test_is_empty():
    assert F.is_empty(ExecResult(ok=True, rows=[]))
    assert F.is_empty(ExecResult(ok=True, rows=[(None,), (None,)]))
    assert not F.is_empty(ExecResult(ok=True, rows=[(0,)]))


def test_same_result_respects_order_by():
    a = ExecResult(ok=True, rows=[(1,), (2,)])
    b = ExecResult(ok=True, rows=[(2,), (1,)])
    assert F.same_result("SELECT x FROM t", a, b)
    assert not F.same_result("SELECT x FROM t ORDER BY x", a, b)
    assert not F.same_result("SELECT x FROM t", a, ExecResult(ok=False, error="boom"))


def test_cap_empty():
    cands = [cand(q=f"n{i}") for i in range(40)] + [cand(q=f"e{i}", empty=True) for i in range(5)]
    F.cap_empty(cands, max_fraction=0.05, seed=0)  # 5% of 40 -> 2 empties allowed
    assert sum(c["failed"] == F.EMPTY for c in cands) == 3
    assert all(c["failed"] is None for c in cands[:40])


def test_normalise():
    assert F.normalise_question("How many  Singers?") == F.normalise_question("how many singers")
    assert F.normalise_sql("select  count(*) from Singer") == F.normalise_sql("SELECT COUNT(*) FROM singer")


def test_deduplicate():
    pairs = [
        cand(q="How many singers?", sql="SELECT count(*) FROM singer"),
        cand(q="how many singers", sql="SELECT COUNT(*) FROM singer"),  # same question
        cand(q="Count the singers.", sql="select count(*) from singer"),  # 2nd paraphrase: kept
        cand(q="Number of singers?", sql="SELECT count(*) FROM singer"),  # 3rd: over the cap
        cand(db="other", q="How many singers?", sql="SELECT count(*) FROM singer"),  # other db
        cand(q="x", failed=F.EXECUTE),
    ]
    F.deduplicate(pairs, max_questions_per_sql=2)
    assert [p["failed"] for p in pairs] == [None, F.DUPLICATE, None, F.DUPLICATE, None, F.EXECUTE]


def test_stage_counts():
    cands = [cand(failed=f) for f in [None, None, F.PARSE, F.EXECUTE, F.EXECUTE, F.CONSISTENCY]]
    rows = stage_counts(cands, [F.PARSE, F.EXECUTE, F.EMPTY, F.CONSISTENCY, F.DUPLICATE])
    assert [(r["in"], r["passed"]) for r in rows] == [(6, 5), (5, 3), (3, 3), (3, 2), (2, 2)]


@requires_spider
def test_leakage_guard():
    assert_no_dev_leakage(["academic", "activity_1"])
    with pytest.raises(AssertionError):
        assert_no_dev_leakage(["academic", "concert_singer"])


def test_stable_seed():
    assert stable_seed(0, "synth", "db", "join", 1) == stable_seed(0, "synth", "db", "join", 1)
    assert stable_seed(0, "synth", "db", "join", 1) != stable_seed(0, "synth", "db", "join", 2)
