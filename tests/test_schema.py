import glob

from conftest import requires_spider, spider_db

from sensei_sql.core.schema import approx_token_count, format_schema
from sensei_sql.paths import SPIDER_DB


def test_structure(tiny_db):
    text = format_schema(tiny_db)
    assert 'CREATE TABLE "Zoo Keeper" (' in text
    assert "PRIMARY KEY (zoo_id, animal_id)" in text
    assert 'FOREIGN KEY (keeper_id) REFERENCES "Zoo Keeper"(keeper_id)' in text
    assert "  notes," in text  # untyped column: no type printed
    # tables sorted by name, case-insensitively
    assert text.index("CREATE TABLE animal") < text.index('CREATE TABLE "Zoo Keeper"')


def test_sample_rows_and_truncation(tiny_db):
    text = format_schema(tiny_db, sample_rows=2, max_value_chars=20)
    assert text.count("/* 2 example rows:") == 2
    assert "x" * 17 + "..." in text and "x" * 21 not in text
    assert "<blob 2 bytes>" in text
    assert "NULL" in text
    assert "example row" not in format_schema(tiny_db, sample_rows=0)


def test_deterministic(tiny_db):
    assert format_schema(tiny_db) == format_schema(tiny_db)


def test_token_budget_drops_sample_rows_first(tiny_db):
    full = format_schema(tiny_db, sample_rows=3)
    bare = format_schema(tiny_db, sample_rows=0)
    budget = approx_token_count(bare) + 5
    trimmed = format_schema(tiny_db, sample_rows=3, max_tokens=budget)
    assert approx_token_count(trimmed) <= budget < approx_token_count(full)
    assert trimmed.startswith(bare.split("\n")[0])
    # structure is kept even when the budget is impossible
    assert format_schema(tiny_db, max_tokens=1) == bare


@requires_spider
def test_concert_singer():
    text = format_schema(spider_db("concert_singer"))
    assert "FOREIGN KEY (Stadium_ID) REFERENCES stadium(Stadium_ID)" in text
    assert "PRIMARY KEY (concert_ID, Singer_ID)" in text


@requires_spider
def test_all_spider_databases():
    paths = sorted(glob.glob(str(SPIDER_DB / "*" / "*.sqlite")))
    assert len(paths) >= 166
    for p in paths:
        text = format_schema(p)
        assert "CREATE TABLE" in text, p
        assert text == format_schema(p), p
