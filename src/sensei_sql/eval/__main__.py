"""Evaluate a model (optionally + LoRA adapter) on a Spider split.

    python -m sensei_sql.eval --model student --split dev
    python -m sensei_sql.eval --model teacher --split dev --few-shot
    python -m sensei_sql.eval --model student --adapter outputs/train/<run>/adapter --limit 200

Writes outputs/eval/<run>/predictions.jsonl and metrics.json. Summarise all runs with
``python -m sensei_sql.eval.aggregate``.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import yaml

from sensei_sql.eval.data import load_split, select_few_shot
from sensei_sql.eval.runner import generate, relpath, resolve_model
from sensei_sql.eval.score import METRICS_FILE, PREDICTIONS_FILE, score_rows, write_jsonl
from sensei_sql.paths import CONFIGS, OUTPUTS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="teacher | student | path to a local model dir")
    ap.add_argument("--adapter", help="path to a LoRA adapter dir")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--few-shot", action="store_true", help="3 fixed train_spider examples (see configs/eval.yaml)")
    ap.add_argument("--limit", type=int, help="evaluate only the first N questions")
    ap.add_argument("--no-throughput", action="store_true", help="skip the batched throughput pass")
    ap.add_argument("--run", help="run name (default: derived from model/adapter/split)")
    ap.add_argument("--config", type=Path, default=CONFIGS / "eval.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    spec = resolve_model(args.model, args.adapter)
    examples = load_split(args.split, args.limit)

    few_shot = []
    if args.few_shot:
        fs = cfg["few_shot"]
        all_split_dbs = {e.db_id for e in load_split(args.split)}
        few_shot = select_few_shot(fs["hardness"], fs["max_schema_tokens"], fs["seed"], all_split_dbs)

    run = args.run or "-".join(
        filter(
            None,
            [
                spec.name,
                spec.adapter.name if spec.adapter else None,
                "fewshot" if args.few_shot else "zeroshot",
                args.split,
                f"n{args.limit}" if args.limit else None,
            ],
        )
    )
    out_dir = OUTPUTS / "eval" / run
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval] {run}: {len(examples)} questions -> {out_dir}")

    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    rows, efficiency = generate(spec, examples, few_shot, cfg, throughput=not args.no_throughput)
    accuracy = score_rows(rows, args.split, cfg["exec_timeout_s"])

    import vllm

    metrics = {
        "run": run,
        "model": spec.name,
        "model_path": relpath(spec.path),
        "adapter": relpath(spec.adapter),
        "split": args.split,
        "num_questions": len(rows),
        "few_shot": [{"db_id": e.db_id, "question": e.question, "sql": e.gold_sql} for e in few_shot],
        "accuracy": accuracy,
        "efficiency": efficiency,
        "config": cfg,
        "env": {"vllm": vllm.__version__, "python": platform.python_version()},
        "started": started,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_jsonl(out_dir / PREDICTIONS_FILE, rows)
    (out_dir / METRICS_FILE).write_text(json.dumps(metrics, indent=2) + "\n")

    acc = accuracy["by_hardness"]
    print(
        f"[eval] {run}: EX {acc['all']['exec_accuracy']:.3f} "
        + " ".join(f"{h} {acc[h]['exec_accuracy']}" for h in ("easy", "medium", "hard", "extra"))
        + f" | p50 {efficiency['latency_ms_p50']} ms, peak VRAM {efficiency['peak_gpu_mem_gib']} GiB"
    )


if __name__ == "__main__":
    main()
