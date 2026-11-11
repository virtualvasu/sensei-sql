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

## Teacher data generation

```bash
python -m sensei_sql.teacher --limit-dbs 5 --out data/distill/pilot   # pilot
python -m sensei_sql.teacher                                          # all 146 train DBs (make teacher)
```

Training databases only (`train_spider` + `train_others`, 146 DBs); an assertion fails if any dev
database appears. Settings are in `configs/teacher.yaml`. Two sources:

- **Synthesis:** per database and SQL pattern (single-table filters, aggregation/GROUP BY/HAVING,
  joins, nested subqueries, ORDER BY…LIMIT, set operations) the teacher writes 5 × 10
  question/SQL pairs (temperature 0.9, seeded per request).
- **Relabelling:** the teacher answers Spider's own 8,659 training questions (greedy).

Every pair must pass: parses (sqlglot) → executes read-only within 5 s → non-empty result (at most
5% empty kept) → **self-consistency** (synth: ≥3 of 5 re-samples at T=0.7 reproduce the result) or
**gold check** (relabel: same result as Spider's gold SQL) → dedup (no repeated question per DB, ≤2
questions per normalised SQL). Runs are resumable (`data/distill/work/<source>.{jsonl,done}`).
Prefix caching is turned on for the teacher here, which speeds up the 5× re-sampling ~1.7×.

Result (full run, ~7.5 h on the RTX 4000 Ada): **23,703 verified pairs** in
`data/distill/teacher_pairs.jsonl` (18,346 synthesised + 5,357 relabelled; 16–245 per database,
median 170). Pass rates per stage, pattern/hardness distributions and 50 random samples are in
`data/distill/stats.md`:

| stage | synth in → passed | relabel in → passed |
|---|---|---|
| parse | 41,055 → 40,797 | 8,659 → 8,656 |
| execute | → 39,041 | → 8,388 |
| non-empty | → 33,236 | → 6,839 |
| self-consistency / gold check | → 18,986 (57%) | → 5,410 (79%) |
| dedup | → 18,346 | → 5,357 |

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

## Fine-tuning (QLoRA) and export

```bash
python -m sensei_sql.train configs/train/human.yaml      # Spider train only
python -m sensei_sql.train configs/train/teacher.yaml    # teacher pairs only
python -m sensei_sql.train configs/train/combined.yaml   # both          (make train = all three)
python -m sensei_sql.eval --model student --adapter models/adapters/combined
python -m sensei_sql.train.export merge  --adapter models/adapters/combined  # -> models/student-merged/
python -m sensei_sql.train.export verify --adapter models/adapters/combined  # merged == adapter?
python -m sensei_sql.train.export gguf   # -> models/student-gguf/ (needs third_party/llama.cpp built)
```

The 4-bit (NF4, double quant, bf16 compute) student gets a LoRA adapter (r=16, α=32, dropout 0.05,
all linear layers). It is trained with TRL's `SFTTrainer` on prompt/completion pairs, with loss on the
SQL only. The prompts are the same ones the eval harness uses. Settings: 2 epochs, lr 2e-4 with a
cosine schedule and 3% warmup, effective batch 16, gradient checkpointing, and batches grouped by
length. That last setting cuts step time from ~12 s to ~5 s, because prompts range from ~300 to 4,096
tokens. Examples longer than 4,096 tokens are dropped (<1%), not truncated. 2% of the training pairs
are held out for validation, and the adapter with the lowest validation loss is kept. Settings live in
`configs/train/base.yaml`; the variant files only set `data`. Each run writes
`outputs/train/<run>/{notes.json,log_history.json,loss.png}`.

| variant | train examples | steps | time | best val loss | peak VRAM (train) |
|---|---|---|---|---|---|
| human | 8,406 | 1,052 | 94 min | 0.065 | 8.3 GiB |
| teacher | 23,149 | 2,894 | 250 min | 0.106 | 8.4 GiB |
| combined | 31,552 | 3,944 | 350 min | 0.114 | 8.4 GiB |

The validation losses are not comparable across variants: each is measured on that variant's own
held-out data, and the teacher's synthetic questions are more varied than Spider's. Spider dev, zero-shot (official
test-suite scorer, execution accuracy):

| model | EX | easy | medium | hard | extra | p50 latency | throughput (bs 32) | peak VRAM | size |
|---|---|---|---|---|---|---|---|---|---|
| teacher Qwen3.5-9B FP8 | **0.782** | 0.940 | 0.834 | 0.707 | 0.488 | 945 ms | 4.2 q/s | 15.5 GiB | 13.1 GiB |
| student base | 0.672 | 0.867 | 0.729 | 0.512 | 0.398 | 214 ms | 42.7 q/s | 6.5 GiB | 2.9 GiB |
| student + human | 0.755 | 0.907 | 0.812 | 0.632 | 0.506 | 224 ms | 38.7 q/s | 7.2 GiB | 3.0 GiB |
| student + teacher | 0.769 | 0.911 | 0.834 | 0.713 | 0.440 | 244 ms | 42.4 q/s | 7.2 GiB | 3.0 GiB |
| student + combined | 0.772 | 0.923 | 0.807 | 0.707 | 0.518 | 235 ms | 39.0 q/s | 7.2 GiB | 3.0 GiB |
| **student merged (combined)** | **0.773** | 0.919 | 0.812 | 0.713 | 0.512 | 206 ms | 44.1 q/s | 6.5 GiB | 2.9 GiB |

Training on teacher pairs alone beats training on Spider's human labels (0.769 vs 0.755), and
combining both is best: the 1.5B student comes within 1 point of the 9B teacher, while answering
~4.5× faster and ~10× more queries/s in under half the VRAM. The adapter rows run the
LoRA through vLLM, which adds latency and memory; merging removes it.

**Export.** `merge` folds the combined adapter into fp16 weights (`models/student-merged/`, 2.9 GiB),
and the app and later evaluations use that model. `verify` regenerates the first 50 dev questions. 47 of 50
merged outputs are identical to adapter+base; the 3 that differ only in identifier case or
spacing and give the same execution result (fp16 merge vs bf16 LoRA rounding). Results are in
`outputs/export/verify.json`. `gguf` converts the merged model to f16/Q8_0/Q4_K_M GGUF for llama.cpp or
Ollama, using `third_party/llama.cpp`, which must be cloned and built first:
`cd third_party/llama.cpp && cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF && cmake --build build -j --target llama-quantize llama-server llama-cli`.
`verify --gguf` also compares each GGUF with the adapter model; the outputs often differ only in case or quote style, so
judge them by accuracy (`python -m sensei_sql.eval.cpu --backend llama-server --gguf models/student-gguf/student-Q4_K_M.gguf`).

## Results

```bash
make results     # aggregate + analysis -> outputs/results/ (tables, plots, error analysis)
```

On Spider dev (1,034 questions, execution accuracy) the distilled student reaches **0.773 EX against the
teacher's 0.782, 98.8% of its accuracy**. It is 4.6× faster at batch size 1 (206 vs 945 ms p50) with 10.6×
the throughput, needs 6.5 instead of 15.5 GiB of VRAM and is 4.5× smaller on disk (2.9 vs 13.1 GiB).
Without fine-tuning the student scores 0.672.

| training data | EX | | teacher pairs used | teacher-only | human + teacher |
|---|---|---|---|---|---|
| none | 0.672 | | 25% | 0.750 | 0.767 |
| human only | 0.755 | | 50% | 0.760 | 0.759 |
| teacher only | 0.769 | | 100% | 0.769 | 0.772 |
| human + teacher | 0.772 | | | | |

Teacher-generated data matches or beats human labels; more of it keeps helping the teacher-only variant, while
human + teacher is flat within noise (±1.3 points standard error). 99% of the best student's failures are
valid SQL that answers a slightly different question; the most common are wrong table/column (34%), wrong
join (19%) and wrong condition (17%). On a CPU (i9-14900K, 100 questions, llama.cpp, Q4_K_M) the student answers in 0.67 s p50 at 0.92 GiB; the
teacher needs 9.9 s at 5.4 GiB, so the student is 14.6× faster and 5.8× smaller on CPU too. Accuracy on that
subset: student 0.77, teacher 0.84 (fp32 student: 0.83, 6.0 s).

The full tables, plots, error analysis and per-question examples are in [docs/results/](docs/results/results.md);
the write-up with caveats is [docs/results_draft.md](docs/results_draft.md).

## App: offline natural-language-to-SQL

```bash
make app                                          # backend from configs/app.yaml (default: vLLM, GPU); builds the frontend on first use
python -m sensei_sql.app --backend transformers   # GPU or CPU, no extra deps
python -m sensei_sql.app --backend llama-server   # CPU, GGUF (needs the GGUF export)
ollama create sensei-sql -f Modelfile             # once: import the Q4_K_M GGUF into a local Ollama
python -m sensei_sql.app --backend ollama         # then use it
```

A React app (Vite, TypeScript, Tailwind, shadcn/ui, Recharts; source in [frontend/](frontend/README.md)) served by a
small FastAPI server (`src/sensei_sql/app/api.py`) on http://127.0.0.1:7860 (local only; `host` in `configs/app.yaml`).
Building the frontend needs node ≥ 20 once (`make frontend`); after that everything is bundled and runs offline.

- **Assistant** (`/`): pick one of the Spider databases or upload your own `.sqlite` file, ask a question, and get the
  generated SQL (editable, with a Run button), the result table, the schema, timings and a session history. Queries run
  through the same safe runner as the evaluation: read-only connection, `SELECT`/`WITH` only (an edited `DELETE` is
  rejected), a timeout and a row cap. If a generated query fails to execute, the error is shown to the model once for a
  corrected attempt (`self_correct` in `configs/app.yaml`).
- **Results** (`/results`): the project's numbers with charts: headline figures, the model comparison table,
  accuracy vs latency, accuracy by hardness, both ablations, CPU results, training curves, the teacher-data filter chain
  and the error analysis. Tabs are linkable (`/results?tab=cpu`). The data comes from `frontend/src/content/results.json`
  (`make frontend-data` regenerates it from the evaluation outputs).

Checked with the final student: vLLM on the GPU (~0.2–0.7 s per question), the transformers backend on CPU, llama-server
with the Q4_K_M GGUF, and Ollama with the same GGUF (4 questions, all correct and executable); the web page was driven
end to end in headless Chrome. Stopping the app (Ctrl-C or SIGTERM) also stops the vLLM engine process.

Frontend development with hot reload: run the python app, then `cd frontend && npm install && npm run dev`
(http://127.0.0.1:5173, proxies `/api` to the python server).

## Report material

[report/](report/README.md) collects everything needed to write up the project: overview, methodology,
all results and ablation tables, error analysis, the problems met and design decisions, timeline and compute,
reproduction commands, limitations and references, plus the data tables, figures and samples behind them.
