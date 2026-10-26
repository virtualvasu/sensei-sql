"""Final results (#6): tables, plots, ablations, error analysis, side-by-side examples.

    python -m sensei_sql.eval.analysis      # reads outputs/eval/*, writes outputs/results/

Every number comes from a saved outputs/eval/<run>/{predictions.jsonl,metrics.json}; the
selection of runs is configs/results.yaml.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

from sensei_sql.eval.data import db_path
from sensei_sql.eval.errors import CATEGORIES, categorise
from sensei_sql.eval.score import METRICS_FILE, PREDICTIONS_FILE, read_jsonl
from sensei_sql.paths import CONFIGS, OUTPUTS

HARDNESS = ("easy", "medium", "hard", "extra")


def load_run(eval_dir: Path, run: str) -> dict | None:
    path = eval_dir / run / METRICS_FILE
    if not path.exists():
        return None
    m = json.loads(path.read_text())
    m["predictions"] = read_jsonl(eval_dir / run / PREDICTIONS_FILE)
    return m


def md_table(df: pd.DataFrame, floatfmt: dict[str, str] | None = None) -> str:
    floatfmt = floatfmt or {}

    def fmt(col, v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "–"
        if isinstance(v, float):
            return format(v, floatfmt.get(col, ".2f"))
        return str(v)

    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(c, v) for c, v in zip(df.columns, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def main_table(runs: list[tuple[dict, dict]], ref: dict | None) -> pd.DataFrame:
    rows = []
    for entry, m in runs:
        acc, eff = m["accuracy"]["by_hardness"], m["efficiency"]
        row = {
            "model": entry["label"],
            "n": m["num_questions"],
            "EX": acc["all"]["exec_accuracy"],
            **{h: acc[h]["exec_accuracy"] for h in HARDNESS},
            "p50 ms": eff["latency_ms_p50"],
            "p95 ms": eff["latency_ms_p95"],
            "q/s (batch)": eff.get("throughput_qps"),
            "VRAM GiB": eff.get("peak_gpu_mem_gib"),
            "size GiB": eff.get("model_size_gib"),
            "device": eff.get("device", "GPU (RTX 4000 Ada)"),
        }
        if ref:
            ref_acc = ref["accuracy"]["exec_accuracy"]
            row["% of teacher EX"] = 100 * row["EX"] / ref_acc if ref_acc else None
            row["speed-up vs teacher (p50)"] = ref["efficiency"]["latency_ms_p50"] / row["p50 ms"]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_accuracy_vs_latency(df: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for _, r in df.iterrows():
        cpu = str(r["device"]).startswith("CPU")
        ax.scatter(r["p50 ms"], 100 * r["EX"], s=60, marker="s" if cpu else "o")
        ax.annotate(r["model"], (r["p50 ms"], 100 * r["EX"]), textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.set_xscale("log")
    ax.margins(x=0.35, y=0.08)  # room for the point labels
    ax.set(xlabel="median latency per query, batch size 1 (ms, log scale)", ylabel="execution accuracy (%)",
           title="Spider dev: accuracy vs latency (squares = CPU)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_by_hardness(df: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gpu = df[~df["device"].astype(str).str.startswith("CPU")]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / max(1, len(gpu))
    for i, (_, r) in enumerate(gpu.iterrows()):
        xs = [k + i * width for k in range(len(HARDNESS))]
        ax.bar(xs, [100 * r[h] for h in HARDNESS], width=width, label=r["model"])
    ax.set_xticks([k + width * (len(gpu) - 1) / 2 for k in range(len(HARDNESS))], HARDNESS)
    ax.set(ylabel="execution accuracy (%)", title="Spider dev: accuracy by hardness", ylim=(0, 100))
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def size_ablation(eval_dir: Path, cfg: dict) -> pd.DataFrame:
    rows = []
    for variant, by_frac in cfg.items():
        for frac, run in sorted(by_frac.items()):
            m = load_run(eval_dir, run)
            if m:
                rows.append({"variant": variant, "teacher pairs used (%)": int(frac), "EX": m["accuracy"]["exec_accuracy"], "run": run})
    return pd.DataFrame(rows)


def plot_size_ablation(df: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for variant, g in df.groupby("variant"):
        ax.plot(g["teacher pairs used (%)"], 100 * g["EX"], marker="o", label=variant)
    ax.set(xlabel="teacher pairs used for training (%)", ylabel="execution accuracy (%)", title="Effect of teacher-data size")
    ax.set_xticks([25, 50, 100])
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def error_analysis(best: dict, n: int, seed: int) -> tuple[pd.DataFrame, list[dict]]:
    fails = [r for r in best["predictions"] if not r["exec_match"]]
    rng = random.Random(seed)
    sample = rng.sample(fails, min(n, len(fails)))
    for r in sample:
        r["category"] = categorise(db_path(r["db_id"]), r["gold_sql"], r["pred_sql"])
    counts = Counter(r["category"] for r in sample)
    df = pd.DataFrame(
        [{"category": c, "count": counts.get(c, 0), "share (%)": 100 * counts.get(c, 0) / len(sample)} for c in CATEGORIES]
    )
    return df, sample


def side_by_side(student: dict, teacher: dict, k: int, seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    pairs = list(zip(student["predictions"], teacher["predictions"]))
    assert all(s["question"] == t["question"] for s, t in pairs), "runs must cover the same questions"
    groups = {
        "student matches teacher (both correct)": [p for p in pairs if p[0]["exec_match"] and p[1]["exec_match"] and p[1]["hardness"] in ("hard", "extra")],
        "student beats teacher": [p for p in pairs if p[0]["exec_match"] and not p[1]["exec_match"]],
        "student fails, teacher correct": [p for p in pairs if not p[0]["exec_match"] and p[1]["exec_match"]],
    }
    return {name: rng.sample(g, min(k, len(g))) for name, g in groups.items()} | {
        "_counts": {name: len(g) for name, g in groups.items()}
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=CONFIGS / "results.yaml")
    ap.add_argument("--eval-dir", type=Path, default=OUTPUTS / "eval")
    ap.add_argument("--out", type=Path, default=OUTPUTS / "results")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    ref = load_run(args.eval_dir, cfg["reference_teacher"])
    runs = [(e, m) for e in cfg["main"] if (m := load_run(args.eval_dir, e["run"]))]
    for p in sorted(args.eval_dir.glob(cfg["cpu"]["glob"])):
        m = load_run(args.eval_dir, p.name)
        if m:
            runs.append(({"label": f"{m['model']} on CPU ({m['efficiency'].get('backend')}, n={m['num_questions']})", "run": p.name, "kind": "student"}, m))

    main_df = main_table(runs, ref)
    main_df.to_csv(out / "main_results.csv", index=False)
    pct = {c: ".3f" for c in ("EX", *HARDNESS)} | {"% of teacher EX": ".1f", "speed-up vs teacher (p50)": ".1f", "p50 ms": ".0f", "p95 ms": ".0f"}
    sections = ["# Results on Spider dev\n", "## Main results\n", md_table(main_df, pct)]
    plot_accuracy_vs_latency(main_df, out / "accuracy_vs_latency.png")
    plot_by_hardness(main_df, out / "accuracy_by_hardness.png")
    sections.append("\n![accuracy vs latency](accuracy_vs_latency.png)\n\n![accuracy by hardness](accuracy_by_hardness.png)\n")

    # Ablation: data source (a vs b vs c)
    variants = [(e, m) for e, m in runs if e["run"] in {"student-zeroshot-dev", "student-human-zeroshot-dev", "student-teacher-zeroshot-dev", "student-combined-zeroshot-dev"}]
    if variants:
        abl = main_df[main_df["model"].isin([e["label"] for e, _ in variants])][["model", "EX", *HARDNESS]]
        abl.to_csv(out / "ablation_data_source.csv", index=False)
        sections += ["\n## Ablation: training data source\n", md_table(abl, pct)]

    size_df = size_ablation(args.eval_dir, cfg["ablation_size"])
    if not size_df.empty:
        size_df.to_csv(out / "ablation_data_size.csv", index=False)
        plot_size_ablation(size_df, out / "ablation_data_size.png")
        sections += ["\n## Ablation: amount of teacher data\n", md_table(size_df, {"EX": ".3f"}), "\n![data size](ablation_data_size.png)\n"]

    # Error analysis + examples for the best fine-tuned student
    students = [(e, m) for e, m in runs if e["kind"] == "student" and m.get("adapter") and m["num_questions"] == 1034]
    if students:
        best_entry, best = max(students, key=lambda em: em[1]["accuracy"]["exec_accuracy"])
        ea = cfg["error_analysis"]
        err_df, sample = error_analysis(best, ea["num_failures"], ea["seed"])
        err_df.to_csv(out / "error_categories.csv", index=False)
        (out / "error_samples.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sample))
        sections += [f"\n## Error analysis: {len(sample)} random failures of the best student ({best_entry['label']})\n",
                     md_table(err_df, {"share (%)": ".0f"})]
        for cat in CATEGORIES:
            ex = [r for r in sample if r["category"] == cat][:2]
            for r in ex:
                sections.append(f"- **{cat}** ({r['db_id']}, {r['hardness']}): {r['question']}  \n  gold: `{r['gold_sql']}`  \n  pred: `{r['pred_sql']}`\n")
        if ref:
            sbs = side_by_side(best, ref, k=3, seed=ea["seed"])
            counts = sbs.pop("_counts")
            sections.append("\n## Student vs teacher, side by side\n")
            sections.append(", ".join(f"{k}: {v}" for k, v in counts.items()) + " (out of 1034)\n")
            for name, items in sbs.items():
                sections.append(f"\n### {name}\n")
                for s, t in items:
                    sections.append(
                        f"- ({s['db_id']}, {s['hardness']}) {s['question']}  \n  gold: `{s['gold_sql']}`  \n"
                        f"  student: `{s['pred_sql']}` {'✅' if s['exec_match'] else '❌'}  \n"
                        f"  teacher: `{t['pred_sql']}` {'✅' if t['exec_match'] else '❌'}\n"
                    )
    (out / "results.md").write_text("\n".join(sections))
    print(f"[analysis] wrote {out / 'results.md'}")


if __name__ == "__main__":
    main()
