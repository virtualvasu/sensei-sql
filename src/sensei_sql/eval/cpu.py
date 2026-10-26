"""CPU efficiency: per-query latency (batch size 1) of a model on CPU, plus accuracy on the subset.

    python -m sensei_sql.eval.cpu --backend llama-server --gguf models/student-gguf/student-Q4_K_M.gguf --limit 100
    python -m sensei_sql.eval.cpu --backend transformers --model-dir models/student-merged --limit 100

Uses the app backends (#7), the canonical prompt and the same scoring as the GPU harness, and
writes outputs/eval/<run>/{predictions.jsonl,metrics.json} so the aggregator picks it up.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import yaml

from sensei_sql.eval.data import load_split
from sensei_sql.eval.runner import build_messages, dir_size_gib, percentile, relpath
from sensei_sql.eval.score import METRICS_FILE, PREDICTIONS_FILE, score_rows, write_jsonl
from sensei_sql.paths import CONFIGS, OUTPUTS


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["llama-server", "transformers"], required=True)
    ap.add_argument("--gguf", type=Path)
    ap.add_argument("--model-dir", type=Path)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--run")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # make sure nothing lands on the GPU
    from sensei_sql.app.backends import LlamaServerBackend, TransformersBackend

    import torch

    torch.set_num_threads(args.threads)
    if args.backend == "llama-server":
        backend = LlamaServerBackend.launch(args.gguf, threads=args.threads)
        model_name, size = args.gguf.stem, args.gguf.stat().st_size / 2**30
    else:
        backend = TransformersBackend(str(args.model_dir), device="cpu")
        model_name, size = f"{args.model_dir.name}-fp32", dir_size_gib(args.model_dir)

    cfg = yaml.safe_load((CONFIGS / "eval.yaml").read_text())
    examples = load_split(args.split, args.limit)
    run = args.run or f"cpu-{model_name}-{args.split}-n{args.limit}"
    out_dir = OUTPUTS / "eval" / run
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[cpu] {run}: {len(examples)} questions on {cpu_name()} ({args.threads} threads)", flush=True)

    backend.generate_sql(build_messages(examples[0], []))  # warm-up
    rows, latencies = [], []
    try:
        for i, ex in enumerate(examples, start=1):
            gen = backend.generate_sql(build_messages(ex, []))
            latencies.append(gen.gen_ms)
            rows.append(
                {
                    "db_id": ex.db_id,
                    "question": ex.question,
                    "gold_sql": ex.gold_sql,
                    "pred_sql": gen.sql,
                    "raw_output": gen.raw,
                    "latency_ms": round(gen.gen_ms, 2),
                    "tokens_out": None,
                }
            )
            if i % 10 == 0 or i == len(examples):
                print(f"[cpu] {i}/{len(examples)}, mean {statistics.fmean(latencies):.0f} ms", flush=True)
    finally:
        backend.close()

    accuracy = score_rows(rows, args.split, cfg["exec_timeout_s"])
    metrics = {
        "run": run,
        "model": model_name,
        "model_path": relpath(args.gguf or args.model_dir),
        "adapter": None,
        "split": args.split,
        "num_questions": len(rows),
        "few_shot": [],
        "accuracy": accuracy,
        "efficiency": {
            "device": f"CPU: {cpu_name()}, {args.threads} threads",
            "backend": args.backend,
            "latency_ms_mean": round(statistics.fmean(latencies), 2),
            "latency_ms_p50": round(percentile(latencies, 50), 2),
            "latency_ms_p95": round(percentile(latencies, 95), 2),
            "peak_gpu_mem_gib": 0.0,
            "model_size_gib": round(size, 2),
        },
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_jsonl(out_dir / PREDICTIONS_FILE, rows)
    (out_dir / METRICS_FILE).write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"[cpu] {run}: EX {accuracy['exec_accuracy']} | p50 {metrics['efficiency']['latency_ms_p50']} ms", flush=True)


if __name__ == "__main__":
    main()
