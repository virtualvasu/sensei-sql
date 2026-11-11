"""Collect every number shown on the web app's Results page into one JSON file.

    python scripts/export_results.py     # -> frontend/src/content/results.json

Reads outputs/eval/<run>/{metrics,predictions}, outputs/train/<run>/{notes,log_history}.json,
data/distill/stats.json and outputs/results/error_categories.csv. The JSON is committed, so the
frontend builds without those (git-ignored) inputs.
"""

from __future__ import annotations

import csv
import json
from math import sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL, TRAIN = ROOT / "outputs" / "eval", ROOT / "outputs" / "train"
OUT = ROOT / "frontend" / "src" / "content" / "results.json"
HARDNESS = ("easy", "medium", "hard", "extra")

MAIN = [  # (run, id, label, short label, kind)
    ("teacher-zeroshot-dev", "teacher", "Teacher Qwen3.5-9B (FP8), zero-shot", "Teacher", "teacher"),
    ("teacher-fewshot-dev", "teacher-3shot", "Teacher, 3-shot", "Teacher 3-shot", "teacher"),
    ("student-zeroshot-dev", "base", "Student 1.5B, no fine-tuning", "Student (untuned)", "base"),
    ("student-human-zeroshot-dev", "human", "Student + human data (LoRA)", "Human only", "student"),
    ("student-teacher-zeroshot-dev", "teacher-data", "Student + teacher data (LoRA)", "Teacher data only", "student"),
    ("student-combined-zeroshot-dev", "combined", "Student + human + teacher data (LoRA)", "Human + teacher", "student"),
    ("student-merged-zeroshot-dev", "merged", "Student + human + teacher data, merged fp16", "Final student", "final"),
]
CPU = [  # (run, label, model, format, runtime)
    ("cpu-student-merged-fp32-dev-n100", "Student fp32 (transformers)", "student", "fp32", "transformers"),
    ("cpu-student-Q8_0-dev-n100", "Student Q8_0 (llama.cpp)", "student", "Q8_0", "llama.cpp"),
    ("cpu-student-Q4_K_M-dev-n100", "Student Q4_K_M (llama.cpp)", "student", "Q4_K_M", "llama.cpp"),
    ("cpu-teacher-Q8_0-dev-n100", "Teacher Q8_0 (llama.cpp)", "teacher", "Q8_0", "llama.cpp"),
    ("cpu-teacher-Q4_K_M-dev-n100", "Teacher Q4_K_M (llama.cpp)", "teacher", "Q4_K_M", "llama.cpp"),
]
SIZE_ABLATION = {  # variant -> {percent: run}
    "teacher-only": {25: "student-teacher-teacher25pct-zeroshot-dev", 50: "student-teacher-teacher50pct-zeroshot-dev", 100: "student-teacher-zeroshot-dev"},
    "human + teacher": {25: "student-combined-teacher25pct-zeroshot-dev", 50: "student-combined-teacher50pct-zeroshot-dev", 100: "student-combined-zeroshot-dev"},
}
TRAIN_RUNS = [
    ("human", "Human only"), ("teacher", "Teacher only"), ("combined", "Human + teacher"),
    ("teacher-teacher25pct", "Teacher only, 25%"), ("teacher-teacher50pct", "Teacher only, 50%"),
    ("combined-teacher25pct", "Human + teacher, 25%"), ("combined-teacher50pct", "Human + teacher, 50%"),
]


def metrics(run: str) -> dict:
    return json.loads((EVAL / run / "metrics.json").read_text())


def summary(run: str) -> dict:
    m = metrics(run)
    a, e = m["accuracy"], m["efficiency"]
    return {
        "run": run,
        "n": m["num_questions"],
        "ex": a["exec_accuracy"],
        "hardness": {h: a["by_hardness"][h]["exec_accuracy"] for h in HARDNESS},
        "p50_ms": e["latency_ms_p50"],
        "p95_ms": e["latency_ms_p95"],
        "mean_ms": e["latency_ms_mean"],
        "qps": e.get("throughput_qps"),
        "vram_gib": e.get("peak_gpu_mem_gib") or None,
        "size_gib": e.get("model_size_gib"),
    }


def downsample(points: list[dict], n: int = 140) -> list[dict]:
    if len(points) <= n:
        return points
    step = len(points) / n
    return [points[int(i * step)] for i in range(n)] + [points[-1]]


def loss_curves(run: str) -> dict:
    hist = json.loads((TRAIN / run / "log_history.json").read_text())
    train = [{"step": r["step"], "loss": round(r["loss"], 4)} for r in hist if "loss" in r]
    val = [{"step": r["step"], "loss": round(r["eval_loss"], 4)} for r in hist if "eval_loss" in r]
    return {"train": downsample(train), "val": val}


def main() -> None:
    models = [{"id": i, "label": label, "short": short, "kind": kind, **summary(run)} for run, i, label, short, kind in MAIN]
    by_id = {m["id"]: m for m in models}
    teacher, base, final = by_id["teacher"], by_id["base"], by_id["merged"]
    se = sqrt(final["ex"] * (1 - final["ex"]) / final["n"])
    headline = {
        "teacher_ex": teacher["ex"], "student_ex": final["ex"], "base_ex": base["ex"],
        "retained_pct": round(100 * final["ex"] / teacher["ex"], 1),
        "gap_closed_pct": round(100 * (final["ex"] - base["ex"]) / (teacher["ex"] - base["ex"]), 1),
        "latency_speedup": round(teacher["p50_ms"] / final["p50_ms"], 1),
        "throughput_speedup": round(final["qps"] / teacher["qps"], 1),
        "vram_reduction": round(teacher["vram_gib"] / final["vram_gib"], 1),
        "size_reduction": round(teacher["size_gib"] / final["size_gib"], 1),
        "std_error": round(se, 4),
    }

    cpu = [{"label": label, "model": model, "format": fmt, "runtime": rt, **summary(run)} for run, label, model, fmt, rt in CPU]
    cpu_by = {(c["model"], c["format"]): c for c in cpu}
    headline["cpu_speedup_q4"] = round(cpu_by[("teacher", "Q4_K_M")]["p50_ms"] / cpu_by[("student", "Q4_K_M")]["p50_ms"], 1)
    headline["cpu_speedup_q8"] = round(cpu_by[("teacher", "Q8_0")]["p50_ms"] / cpu_by[("student", "Q8_0")]["p50_ms"], 1)
    headline["cpu_size_reduction_q4"] = round(cpu_by[("teacher", "Q4_K_M")]["size_gib"] / cpu_by[("student", "Q4_K_M")]["size_gib"], 1)

    size_ablation = []
    for variant, runs in SIZE_ABLATION.items():
        for pct, run in runs.items():
            s = summary(run)
            notes = json.loads((TRAIN / (run.replace("student-", "").replace("-zeroshot-dev", "")) / "notes.json").read_text())
            size_ablation.append({"variant": variant, "percent": pct, "ex": s["ex"], "hardness": s["hardness"], "train_examples": notes["train_examples"]})

    training = []
    for run, label in TRAIN_RUNS:
        n = json.loads((TRAIN / run / "notes.json").read_text())
        training.append({
            "run": run, "label": label, "train_examples": n["train_examples"], "steps": n["steps"], "minutes": n["train_minutes"],
            "best_val_loss": n["best_eval_loss"], "peak_gpu_gib": n["peak_gpu_mem_gib"], "by_source": n["train_by_source"],
            **loss_curves(run),
        })

    stats = json.loads((ROOT / "data" / "distill" / "stats.json").read_text())
    stages = {
        src: [{"stage": s["stage"], "in": s["in"], "passed": s["passed"]} for s in stats["stages"][src]]
        if isinstance(stats["stages"][src], list) else stats["stages"][src]
        for src in stats["stages"]
    }

    with (ROOT / "outputs" / "results" / "error_categories.csv").open() as f:
        errors = [{"category": r["category"], "count": int(float(r["count"]))} for r in csv.DictReader(f)]

    # student (model c, adapter) vs teacher, question by question
    s_pred = [json.loads(l) for l in (EVAL / "student-combined-zeroshot-dev" / "predictions.jsonl").read_text().splitlines()]
    t_pred = [json.loads(l) for l in (EVAL / "teacher-zeroshot-dev" / "predictions.jsonl").read_text().splitlines()]
    pair = {"both_correct": 0, "student_only": 0, "teacher_only": 0, "both_wrong": 0}
    for s, t in zip(s_pred, t_pred):
        key = ("both_correct" if t["exec_match"] else "student_only") if s["exec_match"] else ("teacher_only" if t["exec_match"] else "both_wrong")
        pair[key] += 1

    out = {
        "benchmark": {"name": "Spider dev", "questions": teacher["n"], "databases": 20, "metric": "execution accuracy (test-suite scorer)"},
        "headline": headline,
        "models": models,
        "cpu": cpu,
        "ablation_size": size_ablation,
        "training": training,
        "teacher_data": {
            "kept": stats["kept"], "kept_by_source": stats["kept_by_source"], "kept_by_pattern": stats["kept_by_pattern"],
            "kept_by_hardness": stats["kept_by_hardness"], "databases": stats["databases"], "stages": stages,
        },
        "errors": errors,
        "student_vs_teacher": pair,
        "export_check": json.loads((ROOT / "outputs" / "export" / "verify.json").read_text()),
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
