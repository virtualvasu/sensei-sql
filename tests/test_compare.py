from conftest import requires_spider, spider_db, spider_examples

from sensei_sql.core.compare import execution_match, has_order_by, results_match


def test_order_insensitive_by_default():
    assert results_match([(1, "a"), (2, "b")], [(2, "b"), (1, "a")])
    assert not results_match([(1, "a"), (2, "b")], [(2, "b"), (1, "a")], ordered=True)


def test_duplicates_count():
    assert not results_match([(1,), (1,), (2,)], [(1,), (2,), (2,)])
    assert not results_match([(1,), (1,)], [(1,)])
    assert results_match([(1,), (1,), (2,)], [(2,), (1,), (1,)])


def test_float_tolerance():
    assert results_match([(1,)], [(1.0,)])
    assert results_match([(0.1 + 0.2,)], [(0.3,)])
    assert results_match([(1 / 3,)], [(0.33333333,)])
    assert not results_match([(1.0,)], [(1.01,)])
    assert results_match([(float("nan"),)], [(float("nan"),)])


def test_types_and_nulls():
    assert results_match([(None, "x")], [(None, "x")])
    assert not results_match([(None,)], [(0,)])
    assert not results_match([("1",)], [(1,)])
    assert not results_match([(1, 2)], [(1,)])
    assert not results_match([(1, 2)], [(2, 1)])  # column order matters
    assert results_match([(None,), (2,), ("a",), (b"x",)], [(b"x",), ("a",), (None,), (2.0,)])
    assert results_match([], [])


def test_has_order_by():
    assert has_order_by("SELECT a FROM t ORDER BY a")
    assert has_order_by("SELECT a FROM t UNION SELECT b FROM u ORDER BY 1")
    assert not has_order_by("SELECT a FROM (SELECT a FROM t ORDER BY a LIMIT 3)")
    assert not has_order_by("SELECT a FROM t")


def test_execution_match(tiny_db):
    gold = "SELECT keeper_id, count(*) FROM animal GROUP BY keeper_id"
    assert execution_match(tiny_db, gold, "SELECT keeper_id, COUNT(animal_id) FROM animal GROUP BY 1 ORDER BY 2 DESC")
    assert not execution_match(tiny_db, gold, "SELECT keeper_id, count(*) FROM animal WHERE weight > 50 GROUP BY keeper_id")
    assert not execution_match(tiny_db, gold, "DROP TABLE animal")
    ordered = "SELECT animal_id FROM animal ORDER BY weight DESC LIMIT 3"
    assert not execution_match(tiny_db, ordered, "SELECT animal_id FROM animal ORDER BY weight ASC LIMIT 3")


@requires_spider
def test_spider_dev_gold_matches_itself():
    for ex in spider_examples("dev")[:200]:
        assert execution_match(spider_db(ex["db_id"]), ex["query"], ex["query"]), ex["query"]
