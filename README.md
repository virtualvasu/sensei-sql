# sensei-sql

Knowledge distillation for text-to-SQL: a **Qwen3.5-9B** teacher generates execution-verified
question/SQL pairs over Spider's training databases, and a **Qwen2.5-Coder-1.5B-Instruct** student
is fine-tuned on them with QLoRA. The distilled student powers a fully offline
natural-language-to-SQL app.

Target hardware: a single NVIDIA RTX 4000 Ada (20 GB VRAM) with 32 GB system RAM.

## Layout

```
configs/      YAML configs (models.yaml pins checkpoints; generation/training/eval configs)
data/         Spider + generated data            (git-ignored)
models/       weights, adapters, merged/GGUF      (git-ignored)
outputs/      eval runs, results                  (git-ignored)
third_party/  test-suite-sql-eval                 (git-ignored, fetched by download script)
src/sensei_sql/
  core/       schema formatter, prompts, safe SQL runner, result comparison
  teacher/    teacher data generation
  train/      QLoRA fine-tuning
  eval/       evaluation harness
  app/        user-facing interface
scripts/      CLI entry points (download.py, check_setup.py)
tests/
```

## Setup

Requires Python ≥ 3.10 and a CUDA GPU for the `gpu` extras.

```bash
# 1. Environment (.venv in the repo root)
make install            # creates .venv, pip install -e ".[gpu,app,download,dev]"
                        # (torch 2.10.0 cu128 + vllm 0.19.1: matches the NVIDIA 570 / CUDA 12.8 driver)
                        # app-only machines: make install-app

# 2. Data + weights (the only step that needs the network)
make download           # = python scripts/download.py
```

`scripts/download.py` fetches:

| What | Where |
|---|---|
| Spider 1.0 (`spider_data.zip`): `database/<db_id>/<db_id>.sqlite`, `tables.json`, `train_spider.json`, `train_others.json`, `dev.json` | `data/spider/` |
| Teacher: [`RedHatAI/Qwen3.5-9B-FP8-dynamic`](https://huggingface.co/RedHatAI/Qwen3.5-9B-FP8-dynamic) @ `790f057` | `models/teacher-qwen3.5-9b-fp8/` |
| Student: [`Qwen/Qwen2.5-Coder-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) @ `2e1fd39` | `models/student-qwen2.5-coder-1.5b/` |
| [`taoyds/test-suite-sql-eval`](https://github.com/taoyds/test-suite-sql-eval) (commit recorded in `REVISION`) | `third_party/test-suite-sql-eval/` |
| NLTK `punkt` / `punkt_tab` (needed by the scorer; NLTK refuses group-writable data dirs) | `~/nltk_data/` |

Use `--skip-spider`, `--skip-models` or `--skip-eval` to fetch only part.

**Why FP8 for the teacher:** Qwen3.5-9B in bf16 is ~19 GB, which leaves no room for the KV cache on
a 20 GB card. The FP8-dynamic checkpoint runs natively on Ada (sm_89): loaded text-only
(`language_model_only`) its weights take ~11.9 GiB, leaving ~3.6 GiB of KV cache (~29k tokens) with
`max_num_seqs=64` / `max_num_batched_tokens=4096` (see `configs/models.yaml`). If memory is ever
tight, vLLM's `cpu_offload_gb` can spill weights into system RAM.

## Offline operation

After `make download`, nothing needs the network. `.env` sets `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1` and disables vLLM usage stats. The Makefile exports
these for every target; when running scripts directly, use `set -a; source .env; set +a`.

## Verify

```bash
make check    # Spider sanity checks + each model generates one completion offline
make test
```

`check_setup.py` asserts that `dev.json` has 1,034 examples across 20 databases, that every
referenced `.sqlite` file opens read-only, and that both models load from `models/` and generate
with the offline flags set.

## Evaluation

```bash
set -a; source .env; set +a
python -m sensei_sql.eval --model student --split dev            # base student, zero-shot
python -m sensei_sql.eval --model teacher --split dev            # teacher, zero-shot
python -m sensei_sql.eval --model teacher --split dev --few-shot # teacher, 3 train examples
python -m sensei_sql.eval --model student --adapter <lora_dir> --limit 200   # quick check
python -m sensei_sql.eval.aggregate     # -> outputs/eval/summary.md + summary.csv
python -m sensei_sql.eval.score outputs/eval/<run>   # re-score saved predictions
```

Each run writes `outputs/eval/<run>/predictions.jsonl` (`db_id`, `question`, `gold_sql`,
`pred_sql`, `raw_output`, `latency_ms`, `tokens_out`, `hardness`, `exec_match`) and
`metrics.json`. Settings are in `configs/eval.yaml`.

- **Accuracy:** execution accuracy from the official
  [test-suite-sql-eval](https://github.com/taoyds/test-suite-sql-eval) scorer (`etype=exec`,
  original Spider databases), overall and per Spider hardness. Each prediction first goes through
  the safe runner (`core/executor.py`), because the scorer executes SQL on a writable connection
  with no effective timeout: rejected, failing or timed-out predictions score 0.
- **Latency:** mean/p50/p95 at batch size 1, from the same pass that produces the predictions.
- **Throughput:** queries/s and output tokens/s at `throughput_batch_size` (a second pass).
- **Peak VRAM:** NVML device memory above the pre-load level. Every model gets the same fixed
  KV cache (`kv_cache_gib`) so the numbers compare models rather than vLLM's pre-allocation.
- **Size on disk:** model directory (+ adapter). The teacher checkpoint includes its unused
  vision encoder.
- Decoding is greedy with a fixed seed, so re-running a model gives the same predictions.
