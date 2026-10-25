import json
import sqlite3

import pytest

from sensei_sql.paths import SPIDER, SPIDER_DB

pytestmark = pytest.mark.skipif(
    not (SPIDER / "dev.json").exists(), reason="Spider not downloaded (run scripts/download.py)"
)


def test_dev_split_shape():
    dev = json.loads((SPIDER / "dev.json").read_text())
    assert len(dev) == 1034
    assert len({ex["db_id"] for ex in dev}) == 20


def test_dev_databases_open_read_only():
    dev = json.loads((SPIDER / "dev.json").read_text())
    for db_id in {ex["db_id"] for ex in dev}:
        con = sqlite3.connect(f"file:{SPIDER_DB / db_id / db_id}.sqlite?mode=ro", uri=True)
        assert con.execute("SELECT count(*) FROM sqlite_master").fetchone()[0] > 0
        con.close()
