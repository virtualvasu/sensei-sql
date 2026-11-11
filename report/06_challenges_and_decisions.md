# 6. Challenges, fixes and design decisions

## 6.1 Problems met and how they were solved

| Area | Problem | Fix |
|---|---|---|
| GPU stack | The default install pulled CUDA 13 wheels, which the machine's NVIDIA 570 / CUDA 12.8 driver cannot run. The owner required keeping the driver. | Pinned torch 2.10.0+cu128 and vLLM 0.19.1 (`pyproject.toml`, cu128 extra index). |
| Teacher memory | vLLM ran out of memory profiling CUDA graphs for the 9B model on 20 GB. | Load text-only (`language_model_only`, no vision encoder), `max_num_seqs` 64, `max_num_batched_tokens` 4096. |
| Teacher size | Qwen3.5-9B in bf16 (~19 GB) leaves no KV cache on a 20 GB card. | Used the FP8-dynamic checkpoint, native on Ada GPUs; weights ~11.9 GiB. |
| Scoring safety | The official scorer runs SQL on a writable connection with no effective timeout, and model SQL can be anything. | Every prediction first goes through the safe runner (parse check, read-only connection, timeout); failures score 0. |
| Scoring setup | NLTK refused its data in a group-writable directory. | Moved the punkt data to the default `~/nltk_data`. |
| Tokeniser API | In transformers 5, `apply_chat_template(tokenize=True)` returns a dict, not token ids. | Tokenise the rendered prompt text explicitly. |
| Hardness labels | Many generated SQL showed "unparsed" because Spider's hardness rules cannot read some styles (implicit aliases). | Added a rewrite to Spider style (explicit `AS`, inlined aliases) with a fall back to the gold SQL's hardness. |
| Error categories | First categoriser was too coarse. | Compare per-clause column sets; "no such column/table" maps to wrong table/column. |
| Training speed | 12 s/step, mostly padding (prompts 300–4,096 tokens). | `group_by_length` batching: ~5 s/step. |
| Training config | `SFTConfig` rejected `warmup_ratio` in the new transformers. | `warmup_steps: 0.03` (a fraction of total steps). |
| Chained sanity eval | The post-training dev check needs ~17.6 GB free for vLLM, but the trainer still held ~9.5 GB. | Added `release_gpu()` (garbage-collect + empty cache) before launching the check. The first two variants' checks failed and were replaced by the full evals. |
| Long jobs | A watcher script matched its own command line and missed the end of the 7.5 h teacher run (about 5 h of idle GPU). | Wait on the process ID instead of pattern-matching. |
| Process control | `pkill -f` in the same shell killed the shell. | Filter `ps` output and kill by PID. |
| App shutdown | Stopping the app left a `VLLM::EngineCore` process holding ~7 GB of GPU memory. | `VLLMBackend.close()` shuts the engine down; SIGTERM now exits normally so cleanup runs. |
| Analysis report | The "both correct" count in the student-vs-teacher section counted only hard/extra questions. | Count over all questions; sample the shown examples from hard/extra. |
| CPU comparison | CPU rows (100 questions, different device) showed "% of teacher" and "speed-up" against the full GPU run. | Blanked those columns for subset runs. |

## 6.2 Key design decisions and the reason for each

- **Train only on training databases; assert no dev database.** Spider dev measures generalisation to unseen
  schemas, so any leak would invalidate the headline result.
- **Verify teacher data by execution, not by trust.** An unverified teacher would pass its own errors to the
  student. Self-consistency (≥3 of 5) is a standard no-labels check; for relabelled Spider questions the gold
  SQL is available and used directly (stricter).
- **Cap empty results at 5%.** Otherwise "return nothing" queries are over-represented and teach little.
- **Dedup per database and per SQL.** Stops the generator's repeats from skewing the mix and keeps Spider-like
  paraphrase counts (≤2 questions per SQL).
- **Generate both new questions and relabelled Spider questions.** Synthesis broadens pattern coverage; relabelling
  tests whether teacher SQL for human-phrased questions is as good as human SQL (it is, see ablation 1).
- **Thinking disabled for the teacher.** Direct answers are what the student has to imitate, and thinking would
  multiply latency; it is also fairer to compare at equal output format.
- **Same prompt everywhere.** One `answer_messages` function is used by the teacher, training, evaluation and
  the app, so there is no train/test prompt mismatch.
- **Greedy decoding and a fixed seed.** Makes every evaluation reproducible.
- **Fixed 2 GiB KV cache for all models.** Without it vLLM reserves a fraction of the whole card for any model,
  and peak memory would be meaningless.
- **QLoRA with loss on the completion only.** The prompt is a long schema that is identical in nature
  across examples; training on it would waste capacity and dilute the SQL signal.
- **Drop, not truncate, over-long examples.** Truncation would cut the SQL off.
- **Pick the best variant by full-dev accuracy, not validation loss.** Losses are measured on different held-out
  sets and are not comparable across variants. Variant (c) won on dev EX (0.772 vs 0.769 vs 0.755).
- **Merge into fp16 for deployment.** Removes adapter overhead (no accuracy change: 0.773 vs 0.772).
- **Judge quantised models by accuracy, not by text identity.** They often differ only in case or quote style.
- **Safe SQL runner in the app too.** An edited `DELETE` is rejected; the app is read-only by construction.
- **Local commits only, owner pushes.** There were no GitHub credentials in the work environment.

## 6.3 Things that did not work or were abandoned
- Few-shot prompting for the teacher: no gain (0.775 vs 0.782), slower.
- Random-order batching for training: too slow (see above).
- Interrupted ablation run (combined, 25% teacher) was stopped at step 1,370 and rerun from scratch, because it
  had saved no adapter (only checkpoints).
- Teacher on CPU and multiple training seeds were considered and not done (see 09).
