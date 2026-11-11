# 2. Methodology: how each stage works

## 2.1 Shared core (`src/sensei_sql/core/`)
Every component (teacher, student training, evaluation, app) uses the same code for the three things
that must never differ between training and testing.

**Schema text.** `format_schema` turns a SQLite file into prompt text: tables sorted by name, columns in
declared order, `CREATE TABLE` statements with primary/foreign keys, plus three example rows per table
(values cut to 40 characters). It is deterministic for a given database.

**Prompt.** One system message ("You are an expert SQLite developer... Reply with the SQL query only")
and one user message ("Database schema: ... Question: ..."). The training target is the bare SQL string.
The student is always used zero-shot; the teacher gets 3 train-database examples only in its few-shot run.

**Safe execution.** Model-written SQL is never run unchecked: (1) `sqlglot` parses it and requires exactly
one SELECT/WITH/set-operation statement, so DDL/DML/PRAGMA/ATTACH are rejected before touching the
database; (2) the connection is read-only (`mode=ro` plus `PRAGMA query_only`); (3) a wall-clock timeout
(progress handler) and a row cap apply. The same runner is used for data filtering, scoring and the app.

**Result comparison.** Two results match if they contain the same rows, ignoring row order unless the query
has an `ORDER BY`, and ignoring column names.

## 2.2 Evaluation harness (`src/sensei_sql/eval/`)
- **Accuracy:** the official `taoyds/test-suite-sql-eval` scorer (commit e97acc5), `etype=exec`, on the
  original Spider databases. Because that scorer executes SQL on a writable connection without a real
  timeout, each prediction first goes through the safe runner; rejected, failing or timed-out
  predictions score 0. Reported overall and per Spider hardness (easy/medium/hard/extra).
- **Decoding:** greedy (temperature 0), max 256 new tokens, fixed seed, so re-running gives the same output.
- **Latency:** one question at a time (batch size 1), mean/p50/p95, from the same pass that produces the
  predictions.
- **Throughput:** a second pass at batch size 32 (queries per second, output tokens per second).
- **Peak VRAM:** device memory above the pre-load level, read from NVML. Every model gets the same fixed
  2 GiB KV cache so the numbers compare models, not vLLM's pre-allocation.
- **Size on disk:** the model directory (+ adapter).
- All GPU runs use vLLM 0.19.1. The CPU runs (`eval/cpu.py`) use the app backends on the first 100 dev
  questions, batch size 1, 32 threads.

## 2.3 Teacher data generation (`src/sensei_sql/teacher/`)
Training databases only (`train_spider` + `train_others`, 146 databases); an assertion fails if any dev
database appears, which guarantees there is no leakage into the test benchmark.

**Two sources**
1. *Synthesis.* For each database and each of six SQL patterns (single-table filters; aggregation with
   GROUP BY/HAVING; joins; nested subqueries; ORDER BY ... LIMIT; UNION/INTERSECT/EXCEPT) the teacher is asked
   for 10 question/SQL pairs, 5 times per pattern (temperature 0.9, seeded per request). 146 × 6 × 5 requests (up to 43,800 pairs)
   yielded 41,055 parseable candidate pairs.
2. *Relabelling.* The teacher answers Spider's own 8,659 training questions (greedy). This gives
   teacher-written SQL for human-written questions.

**Filter chain** (every pair must pass all stages, in order)

| stage | what it checks | synth in → out | relabel in → out |
|---|---|---|---|
| parse | sqlglot parses it as one SELECT | 41,055 → 40,797 | 8,659 → 8,656 |
| execute | runs read-only within 5 s | → 39,041 | → 8,388 |
| non-empty | result is not empty/NULL-only; at most 5% empty kept per database and source | → 33,236 | → 6,839 |
| verify | *synth:* self-consistency: re-ask the question 5 times at T=0.7, at least 3 must reproduce the result; *relabel:* result equals Spider's gold SQL result | → 18,986 (57%) | → 5,410 (79%) |
| dedup | no repeated question per database; at most 2 questions per normalised SQL | → 18,346 | → 5,357 |

Final: **23,703 pairs** (18,346 synthesised + 5,357 relabelled), 16–245 per database, median 170. The
empty-result cap exists because "return nothing" queries are trivially easy to get right and teach little.
Self-consistency is the key quality gate for synthesised data: there is no gold answer, so agreement between
independent samples is used as the proxy for correctness. Runs are resumable (work files appended after each
chunk of 8 databases). The full run took about 7.5 h. Prefix caching was enabled for the teacher in this stage,
which sped up the 5× re-sampling about 1.7×.

**Teacher settings.** `RedHatAI/Qwen3.5-9B-FP8-dynamic` @ 790f057, loaded text-only (`language_model_only`),
`max_num_seqs` 64, `max_num_batched_tokens` 4096, thinking disabled (`enable_thinking: false`) so it answers
directly.

Distribution of the kept pairs: single_table 3,672; join 3,382; nested 3,220; set_ops 2,961; aggregation
2,578; order_limit 2,533; relabel 5,357. By hardness (re-derived with Spider's rules): easy 3,957, medium
6,483, hard 6,506, extra 4,033, unparsed 2,724. 200 random samples are in `samples/`.

## 2.4 QLoRA fine-tuning (`src/sensei_sql/train/`)
- **Student:** `Qwen/Qwen2.5-Coder-1.5B-Instruct` @ 2e1fd39.
- **Quantisation:** 4-bit NF4 with double quantisation, bf16 compute (bitsandbytes).
- **LoRA:** rank 16, alpha 32, dropout 0.05, on all linear layers.
- **Trainer:** TRL `SFTTrainer`, prompt/completion format, loss only on the completion (the SQL).
- **Hyper-parameters:** 2 epochs, learning rate 2e-4, cosine schedule, 3% warm-up, per-device batch 8 ×
  gradient accumulation 2 (effective 16), gradient checkpointing, weight decay 0, seed 0.
- **Sequence length:** max 4,096 tokens; longer examples are dropped, not truncated (truncation would cut
  the SQL). This removed under 1% (80 of the human set, 163 of the combined set).
- **Batching:** `group_by_length` (similar-length examples together). Prompts range from about 300 to 4,096
  tokens, and random batches wasted most compute on padding: it cut step time from ~12 s to ~5 s.
- **Validation:** 2% of the training pairs are held out (training databases only); the adapter with the
  lowest validation loss is kept (`load_best_model_at_end`).
- **Three variants** share all settings and differ only in data: (a) human (Spider train, 8,406 examples),
  (b) teacher (23,149), (c) combined (31,552).
- **Ablation runs** use `teacher_fraction` 0.25/0.5 to keep that share of the teacher pairs (human data is
  always kept whole in the combined variant).

## 2.5 Export (`train/export.py`)
- **merge:** folds the LoRA adapter into the fp16 base weights (`models/student-merged/`, 2.9 GiB).
- **verify:** greedy SQL on the first 50 dev questions must match between adapter+base and the merged model.
- **gguf:** `convert_hf_to_gguf.py` to f16, then `llama-quantize` to Q8_0 and Q4_K_M (llama.cpp built CPU-only).

## 2.6 The app (`src/sensei_sql/app/`)
A Gradio page bound to `127.0.0.1`. The user picks a Spider database or uploads a `.sqlite` file, asks a
question, and sees the generated SQL (editable, with a Run button), the result table, the schema and timings.
Four interchangeable backends: vLLM (GPU), transformers (GPU/CPU), llama-server (CPU, GGUF) and Ollama. The
safe runner blocks anything but a single SELECT/WITH; if the generated query fails, the error is shown to the
model once for a corrected attempt. Offline flags (`HF_HUB_OFFLINE`, Gradio analytics off) are set.
