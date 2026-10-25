import pytest
from conftest import requires_spider, spider_db, spider_examples

from sensei_sql.core.executor import UnsafeSQLError, run_sql, validate_sql


def test_select_ok(tiny_db):
    r = run_sql(tiny_db, "SELECT animal_id, weight FROM animal WHERE animal_id <= 2 ORDER BY animal_id")
    assert r.ok and r.error is None
    assert r.columns == ["animal_id", "weight"]
    assert r.rows == [(1, 10.5), (2, 21.0)]
    assert r.elapsed_ms >= 0 and not r.truncated


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE animal",
        "DELETE FROM animal",
        "UPDATE animal SET weight = 0",
        "INSERT INTO animal VALUES (9, 9, 1, 1.0, NULL)",
        "CREATE TABLE x (a INT)",
        "SELECT 1; DROP TABLE animal",
        "SELECT 1; SELECT 2",
        "PRAGMA table_info(animal)",
        "ATTACH DATABASE 'other.db' AS other",
        "WITH c AS (SELECT 1) DELETE FROM animal",
        "VACUUM",
        "",
    ],
)
def test_rejects_non_select(tiny_db, sql):
    r = run_sql(tiny_db, sql)
    assert not r.ok and r.error.startswith("rejected:")
    assert run_sql(tiny_db, "SELECT count(*) FROM animal").rows == [(20,)]  # untouched


def test_with_and_set_ops_allowed():
    for sql in [
        "WITH c AS (SELECT 1 AS x) SELECT x FROM c",
        "SELECT 1 UNION SELECT 2",
        "SELECT 1 INTERSECT SELECT 1",
        "SELECT 1 EXCEPT SELECT 2",
    ]:
        assert validate_sql(sql)
    with pytest.raises(UnsafeSQLError):
        validate_sql("SELECT FROM WHERE (")


def test_read_only_even_if_validation_bypassed(tiny_db):
    from sensei_sql.core.schema import connect_read_only

    con = connect_read_only(tiny_db)
    with pytest.raises(Exception):
        con.execute("DELETE FROM animal")
    con.close()


def test_timeout_fires(tiny_db):
    infinite = "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT count(*) FROM n"
    r = run_sql(tiny_db, infinite, timeout_s=0.2)
    assert not r.ok and r.error.startswith("timeout")
    assert r.elapsed_ms < 2000


def test_row_cap(tiny_db):
    r = run_sql(tiny_db, "SELECT * FROM animal", max_rows=5)
    assert r.ok and len(r.rows) == 5 and r.truncated


def test_sql_errors_reported(tiny_db):
    r = run_sql(tiny_db, "SELECT nope FROM animal")
    assert not r.ok and "no such column" in r.error
    r = run_sql(tiny_db.parent / "missing.sqlite", "SELECT 1")
    assert not r.ok and "cannot open database" in r.error


@requires_spider
def test_spider_dev_gold_executes():
    for ex in spider_examples("dev"):
        r = run_sql(spider_db(ex["db_id"]), ex["query"])
        assert r.ok, (ex["query"], r.error)


@requires_spider
def test_spider_train_gold_validates():
    for split in ("train_spider", "train_others"):
        for ex in spider_examples(split):
            validate_sql(ex["query"])
