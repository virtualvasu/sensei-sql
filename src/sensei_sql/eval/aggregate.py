"""Collect every outputs/eval/*/metrics.json into one comparison table.

    python -m sensei_sql.eval.aggregate     # -> outputs/eval/summary.md and summary.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sensei_sql.eval.score import METRICS_FILE
from sensei_sql.paths import OUTPUTS

COLUMNS = {
    "run": "run",
    "model": "model",
    "few_shot": "few-shot",
    "n": "n",
    "ex_all": "EX",
    "ex_easy": "easy",
    "ex_medium": "medium",
    "ex_hard": "hard",
    "ex_extra": "extra",
    "latency_ms_mean": "lat mean (ms)",
    "latency_ms_p50": "lat p50 (ms)",
    "latency_ms_p95": "lat p95 (ms)",
    "throughput_qps": "q/s",
    "throughput_tokens_per_s": "tok/s",
    "peak_gpu_mem_gib": "peak VRAM (GiB)",
    "model_size_gib": "size (GiB)",
}


ACCURACY_KEYS = [k for k in COLUMNS if k.startswith("ex_")]
EFFICIENCY_KEYS = list(COLUMNS)[list(COLUMNS).index("latency_ms_mean") :]


def collect(eval_dir: Path) -> pd.DataFrame:
    records = []
    for path in sorted(eval_dir.glob(f"*/{METRICS_FILE}")):
        m = json.loads(path.read_text())
        acc, eff = m["accuracy"]["by_hardness"], m["efficiency"]
        records.append(
            {
                "run": m["run"],
                "model": m["model"] + (f" + {Path(m['adapter']).name}" if m.get("adapter") else ""),
                "few_shot": len(m["few_shot"]),
                "n": m["num_questions"],
                **{f"ex_{h}": acc[h]["exec_accuracy"] for h in ("all", "easy", "medium", "hard", "extra")},
                **{k: eff.get(k) for k in EFFICIENCY_KEYS},
            }
        )
    return pd.DataFrame.from_records(records, columns=list(COLUMNS))


def to_markdown(df: pd.DataFrame) -> str:
    header = list(COLUMNS.values())

    def fmt(key, v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "–"
        return f"{v:.3f}" if key in ACCURACY_KEYS else str(v)

    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += [
        "| " + " | ".join(fmt(k, v) for k, v in zip(COLUMNS, row)) + " |"
        for row in df.itertuples(index=False)
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", type=Path, default=OUTPUTS / "eval")
    args = ap.parse_args()
    df = collect(args.eval_dir)
    df.rename(columns=COLUMNS).to_csv(args.eval_dir / "summary.csv", index=False)
    md = to_markdown(df)
    (args.eval_dir / "summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
