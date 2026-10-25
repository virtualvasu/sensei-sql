"""Evaluation splits and few-shot example selection."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from sensei_sql.core.schema import approx_token_count, format_schema
from sensei_sql.paths import SPIDER

# split -> (questions file, database dir)
SPLITS = {
    "dev": ("dev.json", "database"),
    "test": ("test.json", "test_database"),
}


@dataclass(frozen=True)
class Example:
    db_id: str
    question: str
    gold_sql: str
    db_path: Path


def db_path(db_id: str, split: str = "dev") -> Path:
    return SPIDER / SPLITS[split][1] / db_id / f"{db_id}.sqlite"


def load_split(split: str, limit: int | None = None) -> list[Example]:
    """Questions of a split in file order; ``limit`` takes the first N (deterministic)."""
    fname, _ = SPLITS[split]
    raw = json.loads((SPIDER / fname).read_text())
    examples = [Example(r["db_id"], r["question"], r["query"], db_path(r["db_id"], split)) for r in raw]
    return examples[:limit] if limit else examples


@lru_cache(maxsize=None)
def schema_text(path: Path) -> str:
    return format_schema(path)


def select_few_shot(
    hardness_levels: list[str],
    max_schema_tokens: int,
    seed: int,
    exclude_db_ids: set[str],
) -> list[Example]:
    """One train_spider example per requested hardness, each from a different database whose
    schema fits ``max_schema_tokens``. Never uses a database in ``exclude_db_ids`` (the
    evaluation split's databases)."""
    from sensei_sql.eval.testsuite import hardness

    raw = json.loads((SPIDER / "train_spider.json").read_text())
    rng = random.Random(seed)
    rng.shuffle(raw)
    chosen: list[Example] = []
    used_dbs = set(exclude_db_ids)
    for level in hardness_levels:
        for r in raw:
            if r["db_id"] in used_dbs:
                continue
            path = db_path(r["db_id"])
            if approx_token_count(schema_text(path)) > max_schema_tokens:
                continue
            if hardness(path, r["query"]) != level:
                continue
            chosen.append(Example(r["db_id"], r["question"], r["query"], path))
            used_dbs.add(r["db_id"])
            break
        else:
            raise ValueError(f"no train example found for hardness {level!r}")
    return chosen
