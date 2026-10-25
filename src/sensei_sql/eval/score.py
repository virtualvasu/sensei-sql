"""Score a predictions.jsonl with the Spider test-suite scorer and write metrics.

    python -m sensei_sql.eval.score outputs/eval/<run>     # re-score an existing run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from sensei_sql.eval import testsuite
from sensei_sql.eval.data import db_path

METRICS_FILE = "metrics.json"
PREDICTIONS_FILE = "predictions.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def score_rows(rows: list[dict], split: str, timeout_s: float) -> dict:
    """Adds ``hardness`` and ``exec_match`` to each row in place; returns accuracy metrics."""
    for r in tqdm(rows, desc="scoring", leave=False):
        path = db_path(r["db_id"], split)
        r["hardness"] = testsuite.hardness(path, r["gold_sql"])
        r["exec_match"] = testsuite.exec_match(path, r["gold_sql"], r["pred_sql"], timeout_s)

    def summary(subset: list[dict]) -> dict:
        n = len(subset)
        correct = sum(r["exec_match"] for r in subset)
        return {"count": n, "correct": correct, "exec_accuracy": round(correct / n, 4) if n else None}

    return {
        "exec_accuracy": summary(rows)["exec_accuracy"],
        "by_hardness": {
            "all": summary(rows),
            **{h: summary([r for r in rows if r["hardness"] == h]) for h in testsuite.HARDNESS_LEVELS},
        },
        "scorer": f"taoyds/test-suite-sql-eval@{testsuite.revision()[:10]} (etype=exec)",
    }


def rescore(run_dir: Path, timeout_s: float | None = None) -> dict:
    metrics = json.loads((run_dir / METRICS_FILE).read_text())
    rows = read_jsonl(run_dir / PREDICTIONS_FILE)
    timeout_s = timeout_s or metrics["config"]["exec_timeout_s"]
    metrics["accuracy"] = score_rows(rows, metrics["split"], timeout_s)
    write_jsonl(run_dir / PREDICTIONS_FILE, rows)
    (run_dir / METRICS_FILE).write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()
    m = rescore(args.run_dir)
    print(json.dumps(m["accuracy"], indent=2))


if __name__ == "__main__":
    main()
