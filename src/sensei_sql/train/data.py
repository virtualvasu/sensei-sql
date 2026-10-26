"""Build the SFT dataset: canonical student prompt (schema + question) -> SQL completion."""

from __future__ import annotations

import json
import random
from pathlib import Path

from sensei_sql.core.prompts import extract_sql, render_prompt, student_messages
from sensei_sql.core.schema import format_schema
from sensei_sql.paths import ROOT, SPIDER, SPIDER_DB
from sensei_sql.teacher.pipeline import TRAIN_FILES, assert_no_dev_leakage


def human_pairs() -> list[dict]:
    """Spider's original training pairs (train_spider + train_others)."""
    pairs = []
    for f in TRAIN_FILES:
        for e in json.loads((SPIDER / f).read_text()):
            pairs.append(
                {
                    "db_id": e["db_id"],
                    "question": " ".join(e["question"].split()),
                    # Spider gold has irregular spacing ("age  >  56"); normalise like teacher SQL.
                    "sql": extract_sql(e["query"]),
                    "source": "human",
                }
            )
    return pairs


def teacher_pairs(path: Path) -> list[dict]:
    path = path if path.is_absolute() else ROOT / path
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return [{"db_id": r["db_id"], "question": r["question"], "sql": r["sql"], "source": f"teacher-{r['source']}"} for r in rows]


def load_pairs(cfg: dict) -> list[dict]:
    pairs = []
    for name in cfg["data"]:
        if name == "human":
            pairs += human_pairs()
        elif name == "teacher":
            pairs += teacher_pairs(Path(cfg["teacher_pairs"]))
        else:
            raise ValueError(f"unknown data source {name!r}")
    assert_no_dev_leakage({p["db_id"] for p in pairs})
    return pairs


def to_example(pair: dict) -> dict:
    """Prompt-completion conversational format (TRL computes loss on the completion only)."""
    schema = format_schema(SPIDER_DB / pair["db_id"] / f"{pair['db_id']}.sqlite")
    return {
        "prompt": student_messages(schema, pair["question"]),
        "completion": [{"role": "assistant", "content": pair["sql"]}],
        "db_id": pair["db_id"],
        "source": pair["source"],
    }


def split(pairs: list[dict], cfg: dict) -> tuple[list[dict], list[dict]]:
    """Shuffle (seeded) and hold out ``val_fraction`` for validation, then keep only
    ``teacher_fraction`` of the teacher pairs in the training part (dataset-size ablation;
    human pairs are always kept). The validation set is identical across teacher fractions."""
    rng = random.Random(cfg["seed"])
    pairs = pairs[:]
    rng.shuffle(pairs)
    n_val = max(1, round(len(pairs) * cfg["val_fraction"]))
    val, train = pairs[:n_val], pairs[n_val:]
    frac = cfg.get("teacher_fraction", 1.0)
    if frac < 1.0:
        teacher = [p for p in train if p["source"].startswith("teacher")]
        keep = set(map(id, teacher[: round(len(teacher) * frac)]))
        train = [p for p in train if not p["source"].startswith("teacher") or id(p) in keep]
    return train, val


def drop_too_long(examples: list[dict], tokenizer, max_length: int) -> tuple[list[dict], int]:
    """Drop examples whose prompt + completion exceeds ``max_length`` tokens (TRL would
    truncate from the end, cutting off the SQL)."""
    kept, dropped = [], 0
    for ex in examples:
        n = len(tokenizer(render_prompt(tokenizer, ex["prompt"]) + ex["completion"][0]["content"])["input_ids"]) + 2
        if n <= max_length:
            kept.append(ex)
        else:
            dropped += 1
    return kept, dropped
