"""QLoRA fine-tuning of the student with TRL's SFTTrainer."""

from __future__ import annotations

import json
import time
from pathlib import Path

import yaml

from sensei_sql.paths import CONFIGS, MODELS
from sensei_sql.train.data import drop_too_long, load_pairs, split, to_example


def load_config(path: Path) -> dict:
    """Load a variant config, merging it over the file named in its ``defaults`` key."""
    cfg = yaml.safe_load(path.read_text())
    if "defaults" in cfg:
        base = load_config(path.parent / cfg.pop("defaults"))
        for k, v in cfg.items():
            base[k] = {**base[k], **v} if isinstance(v, dict) and isinstance(base.get(k), dict) else v
        cfg = base
    return cfg


def base_model_path(cfg: dict) -> Path:
    spec = yaml.safe_load((CONFIGS / "models.yaml").read_text())[cfg["base_model"]]
    return MODELS / spec["local_dir"]


def train(cfg: dict, run_dir: Path, adapter_dir: Path) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from trl import SFTConfig, SFTTrainer

    set_seed(cfg["seed"])
    model_path = base_model_path(cfg)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    pairs = load_pairs(cfg)
    train_pairs, val_pairs = split(pairs, cfg)
    train_ex, dropped_train = drop_too_long([to_example(p) for p in train_pairs], tokenizer, cfg["max_length"])
    val_ex, dropped_val = drop_too_long([to_example(p) for p in val_pairs], tokenizer, cfg["max_length"])
    print(f"[train] {len(train_ex)} train / {len(val_ex)} val examples "
          f"(dropped {dropped_train}/{dropped_val} longer than {cfg['max_length']} tokens)", flush=True)

    q = dict(cfg["quantization"])
    q["bnb_4bit_compute_dtype"] = getattr(torch, q["bnb_4bit_compute_dtype"])
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=BitsAndBytesConfig(**q),
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )

    t = cfg["training"]
    args = SFTConfig(
        output_dir=str(run_dir / "checkpoints"),
        seed=cfg["seed"],
        data_seed=cfg["seed"],
        max_length=cfg["max_length"],
        completion_only_loss=True,
        bf16=True,
        num_train_epochs=t["num_train_epochs"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_steps=t["warmup_steps"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        weight_decay=t["weight_decay"],
        gradient_checkpointing=t["gradient_checkpointing"],
        train_sampling_strategy=t["train_sampling_strategy"],
        logging_steps=t["logging_steps"],
        eval_strategy="steps",
        eval_steps=t["eval_steps"],
        save_strategy="steps",
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
    )
    lora = LoraConfig(task_type="CAUSAL_LM", **cfg["lora"])
    columns = ["prompt", "completion"]
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list([{k: e[k] for k in columns} for e in train_ex]),
        eval_dataset=Dataset.from_list([{k: e[k] for k in columns} for e in val_ex]),
        processing_class=tokenizer,
        peft_config=lora,
    )
    t0 = time.time()
    result = trainer.train()
    final_eval = trainer.evaluate()
    trainer.save_model(str(adapter_dir))  # best checkpoint (lowest val loss), adapter only
    tokenizer.save_pretrained(str(adapter_dir))

    log_history = trainer.state.log_history
    (run_dir / "log_history.json").write_text(json.dumps(log_history, indent=1) + "\n")
    plot_curves(log_history, run_dir / "loss.png", title=run_dir.name)
    return {
        "train_examples": len(train_ex),
        "val_examples": len(val_ex),
        "dropped_too_long": {"train": dropped_train, "val": dropped_val},
        "train_by_source": _count(train_ex),
        "train_loss": round(result.training_loss, 4),
        "best_eval_loss": round(trainer.state.best_metric, 4) if trainer.state.best_metric else None,
        "final_eval_loss": round(final_eval["eval_loss"], 4),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "steps": trainer.state.global_step,
        "train_minutes": round((time.time() - t0) / 60, 1),
        "peak_gpu_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }


def _count(examples: list[dict]) -> dict:
    out: dict[str, int] = {}
    for e in examples:
        out[e["source"]] = out.get(e["source"], 0) + 1
    return out


def plot_curves(log_history: list[dict], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train = [(r["step"], r["loss"]) for r in log_history if "loss" in r]
    val = [(r["step"], r["eval_loss"]) for r in log_history if "eval_loss" in r]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    if train:
        ax.plot(*zip(*train), label="train", linewidth=1.5)
    if val:
        ax.plot(*zip(*val), label="validation", marker="o", markersize=3, linewidth=1.5)
    ax.set(xlabel="step", ylabel="loss", title=title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
