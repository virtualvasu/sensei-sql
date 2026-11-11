# Report material: sensei-sql

Everything needed to write the final report on this project, in one place. Nothing here is new
analysis: every number comes from the saved evaluation and training outputs, and the files in `data/`
are copies of them. If a number in a text file disagrees with `data/`, trust `data/`.

**One-paragraph summary.** sensei-sql distils a 9B-parameter language model (Qwen3.5-9B, FP8) into a
1.5B-parameter model (Qwen2.5-Coder-1.5B-Instruct) for text-to-SQL. The large "teacher" writes and
answers ~50k question/SQL pairs over Spider's training databases; only pairs whose SQL runs and is
confirmed correct (self-consistency or agreement with Spider's gold SQL) are kept: 23,703 pairs. The small
"student" is fine-tuned on them with QLoRA, alone and mixed with Spider's own human-written training data.
On the Spider dev set (1,034 questions, official execution-accuracy scorer) the best student scores
**0.773 vs the teacher's 0.782** (98.8% of its accuracy; the untuned student scores 0.672), while being
4.6× faster, 10.6× higher throughput, 2.4× less GPU memory and 4.5× smaller. A 4-bit GGUF version
(0.9 GiB) answers in 0.67 s median on a CPU, 14.6× faster than the teacher in the same format (9.9 s), and the
whole thing runs as an offline web app.

## Contents

| File | What it is |
|---|---|
| [01_overview.md](01_overview.md) | The problem, goal, approach, hardware and the seven work packages |
| [02_methodology.md](02_methodology.md) | How each stage works: prompts, safe execution, teacher data pipeline and filters, training, export, evaluation, app |
| [03_results.md](03_results.md) | Every result table (accuracy, hardness, latency, memory, CPU) with the figures |
| [04_ablations.md](04_ablations.md) | Effect of the data source and of the amount of teacher data |
| [05_error_analysis.md](05_error_analysis.md) | What the best student gets wrong, and student-vs-teacher comparison |
| [06_challenges_and_decisions.md](06_challenges_and_decisions.md) | Problems hit, how they were solved, and why key choices were made |
| [07_timeline_and_compute.md](07_timeline_and_compute.md) | Dates, commits, wall-clock and GPU time per stage |
| [08_reproducibility.md](08_reproducibility.md) | Pinned versions, commands and where each artefact lives |
| [09_limitations_and_future_work.md](09_limitations_and_future_work.md) | What the results do and do not show; next steps |
| [10_references.md](10_references.md) | Papers, datasets, models and tools to cite |
| [suggested_outline.md](suggested_outline.md) | A section-by-section outline for the final report, mapped to these files |
| `data/` | CSV/JSON/Markdown tables behind every number (see below) |
| `figures/` | PNG plots: loss curves per run, accuracy vs latency, accuracy by hardness, data-size ablation |
| `samples/` | 200 random teacher-generated pairs, and the 100 categorised failures of the best student |

### `data/`

| File | Content |
|---|---|
| `eval_all_runs.csv` | One row per evaluated model: accuracy overall and per hardness, latency p50/p95/mean, throughput, VRAM, size (11 GPU runs + 5 CPU runs) |
| `main_results.csv` | The main results table as generated for the repository |
| `ablation_data_source.csv`, `ablation_data_size.csv` | The two ablation tables |
| `error_categories.csv` | Failure categories for the 100 sampled errors |
| `training_runs.csv` | Per training run: examples, steps, minutes, losses, peak memory, data mix |
| `teacher_data_stats.json` / `.md` | Teacher-data filter pass rates, distribution by pattern/hardness/database |
| `export_verify.json` | Merged-model vs adapter check (49 of 50 identical on the final run, 47 of 50 on an earlier run) |
| `results_generated.md` | The full generated results report, including side-by-side examples |

A note on rigour: with 1,034 questions the standard error of an accuracy figure near 0.77 is about
±1.3 points, and each variant was trained once. Treat differences smaller than about 2.5 points as
unproven. Section 09 spells this out.
