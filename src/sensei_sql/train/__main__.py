"""Train a student variant with QLoRA, then sanity-check it on dev.

    python -m sensei_sql.train configs/train/human.yaml
    python -m sensei_sql.train configs/train/teacher.yaml --teacher-fraction 0.25

Adapter -> models/adapters/<run>/ ; logs, loss curve and run notes -> outputs/train/<run>/.
The sanity check runs the eval harness (#3) on the first N dev questions in a fresh process
(so training memory is released first).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from sensei_sql.paths import MODELS, OUTPUTS
from sensei_sql.train.qlora import load_config, train


def release_gpu() -> None:
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()
    print(f"[train] GPU memory still allocated before sanity eval: {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--teacher-fraction", type=float, help="override teacher_fraction")
    ap.add_argument("--run", help="run name (default: config name [+ fraction])")
    ap.add_argument("--skip-sanity", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.teacher_fraction is not None:
        cfg["teacher_fraction"] = args.teacher_fraction
    frac = cfg["teacher_fraction"]
    run = args.run or args.config.stem + (f"-teacher{round(frac * 100)}pct" if frac < 1 else "")
    run_dir = OUTPUTS / "train" / run
    adapter_dir = MODELS / "adapters" / run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"[train] {run}: data={cfg['data']} teacher_fraction={frac}", flush=True)

    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    notes = {"run": run, "config": str(args.config), "started": started, **train(cfg, run_dir, adapter_dir)}
    notes["adapter"] = str(adapter_dir)
    notes["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (run_dir / "notes.json").write_text(json.dumps(notes, indent=2) + "\n")
    print(f"[train] {run}: done in {notes['train_minutes']} min, best val loss {notes['best_eval_loss']}", flush=True)

    if not args.skip_sanity:
        release_gpu()  # vLLM in the child process needs the memory the trainer held
        limit = cfg["sanity_check"]["limit"]
        cmd = [
            sys.executable, "-m", "sensei_sql.eval", "--model", cfg["base_model"], "--adapter", str(adapter_dir),
            "--limit", str(limit), "--no-throughput", "--run", f"sanity-{run}-dev-n{limit}",
        ]
        subprocess.run(cmd, check=True)
        metrics = json.loads((OUTPUTS / "eval" / f"sanity-{run}-dev-n{limit}" / "metrics.json").read_text())
        notes["sanity_dev_exec_accuracy"] = {"limit": limit, "exec_accuracy": metrics["accuracy"]["exec_accuracy"]}
        (run_dir / "notes.json").write_text(json.dumps(notes, indent=2) + "\n")
        print(f"[train] {run}: sanity EX on first {limit} dev questions = {metrics['accuracy']['exec_accuracy']}", flush=True)


if __name__ == "__main__":
    main()
