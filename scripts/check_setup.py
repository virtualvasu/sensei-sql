"""Verify the Part 1 acceptance criteria.

Usage:
    python scripts/check_setup.py              # data checks + both models
    python scripts/check_setup.py --data-only
    python scripts/check_setup.py --model student

Each model is loaded with vLLM in its own subprocess (so GPU memory is freed between
them) with HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE set, and generates one completion.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sensei_sql.paths import CONFIGS, MODELS, SPIDER, SPIDER_DB  # noqa: E402

OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "VLLM_NO_USAGE_STATS": "1",
    "DO_NOT_TRACK": "1",
}

PROMPT = "Write a SQLite query that counts the rows in a table named singer. Reply with SQL only."


def check_data() -> None:
    dev = json.loads((SPIDER / "dev.json").read_text())
    dev_dbs = {ex["db_id"] for ex in dev}
    print(f"[data] dev.json: {len(dev)} examples across {len(dev_dbs)} databases")
    assert len(dev) == 1034, f"expected 1034 dev examples, got {len(dev)}"
    assert len(dev_dbs) == 20, f"expected 20 dev databases, got {len(dev_dbs)}"

    referenced = set(dev_dbs)
    for split in ("train_spider.json", "train_others.json"):
        exs = json.loads((SPIDER / split).read_text())
        referenced |= {ex["db_id"] for ex in exs}
        print(f"[data] {split}: {len(exs)} examples")

    for db_id in sorted(referenced):
        path = SPIDER_DB / db_id / f"{db_id}.sqlite"
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        con.execute("SELECT name FROM sqlite_master LIMIT 1").fetchall()
        con.close()
    print(f"[data] all {len(referenced)} referenced .sqlite files open read-only")


def run_model(role: str) -> None:
    """Runs in a child process with offline env vars already set."""
    from vllm import LLM, SamplingParams

    spec = yaml.safe_load((CONFIGS / "models.yaml").read_text())[role]
    llm = LLM(model=str(MODELS / spec["local_dir"]), seed=0, **spec.get("vllm", {}))
    messages = [{"role": "user", "content": PROMPT}]
    chat_kwargs = {"chat_template_kwargs": {"enable_thinking": False}} if role == "teacher" else {}
    out = llm.chat(messages, SamplingParams(temperature=0.0, max_tokens=64), **chat_kwargs)
    text = out[0].outputs[0].text.strip()
    print(f"[{role}] completion: {text!r}")
    assert text, f"{role} produced an empty completion"


def check_model(role: str) -> None:
    env = {**os.environ, **OFFLINE_ENV}
    print(f"[{role}] loading offline ...")
    subprocess.run([sys.executable, __file__, "--_child", role], env=env, check=True)
    print(f"[{role}] OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-only", action="store_true")
    ap.add_argument("--model", choices=["teacher", "student", "all"], default="all")
    ap.add_argument("--_child", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._child:
        run_model(args._child)
        return

    check_data()
    if args.data_only:
        return
    for role in (["student", "teacher"] if args.model == "all" else [args.model]):
        check_model(role)
    print("setup OK")


if __name__ == "__main__":
    main()
