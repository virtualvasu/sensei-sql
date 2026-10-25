import json
import sqlite3

import pytest

from sensei_sql.paths import SPIDER, SPIDER_DB

requires_spider = pytest.mark.skipif(
    not (SPIDER / "dev.json").exists(), reason="Spider not downloaded (run scripts/download.py)"
)


def spider_db(db_id: str):
    return SPIDER_DB / db_id / f"{db_id}.sqlite"


def spider_examples(split: str) -> list[dict]:
    return json.loads((SPIDER / f"{split}.json").read_text())


@pytest.fixture
def tiny_db(tmp_path):
    """Two tables with a PK, a composite PK, an FK, a long value and a blob."""
    path = tmp_path / "tiny.sqlite"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE "Zoo Keeper" (keeper_id INTEGER PRIMARY KEY, name TEXT, bio TEXT, photo BLOB);
        CREATE TABLE animal (
            zoo_id INT, animal_id INT, keeper_id INT, weight REAL, notes,
            PRIMARY KEY (zoo_id, animal_id),
            FOREIGN KEY (keeper_id) REFERENCES "Zoo Keeper"(keeper_id)
        );
        """
    )
    con.executemany(
        'INSERT INTO "Zoo Keeper" VALUES (?, ?, ?, ?)',
        [(i, f"keeper {i}", "x" * 200, b"\x00\x01") for i in range(1, 6)],
    )
    con.executemany(
        "INSERT INTO animal VALUES (?, ?, ?, ?, ?)",
        [(1, i, 1 + i % 5, 10.5 * i, None) for i in range(1, 21)],
    )
    con.commit()
    con.close()
    return path
