# 7. Timeline and compute

## 7.1 Commit history (local repository)
| Date | Commit | Content |
|---|---|---|
| 2026-10-25 | `8b94b33` | #1 Setup: environment, datasets and model weights |
| 2026-10-25 | `8a9ba1e` | #2 Shared core: schema formatter, prompts, safe SQL runner |
| 2026-10-26 | `e5bdec9` | #3 Evaluation harness and pre-training baselines |
| 2026-10-26 | `2e536c3` | #4 Teacher data generation with execution-verified filtering |
| 2026-10-26 | `30b4d77`, `0d674f5`, `dd03a03` | #5 training pipeline, #7 app, #6 analysis tooling (in progress, written while the GPU was busy) |
| 2026-10-27 | `a0f6ef7` | #5 Three QLoRA variants, merge, export verification |
| 2026-11-11 | `89c75d7` | #6 Final evaluation, ablations, error analysis |
| 2026-11-11 | `d3bcb48` | #7 App verified on GPU and CPU |
| 2026-11-11 | `a7bd689` | GGUF export, llama.cpp CPU evaluation |
| 2026-11-11 | `8a12ad2` | Ollama Modelfile and backend verified |

Work was done in the order of the issues, with the long GPU jobs run unattended (overnight) and the next
issue's code written while they ran.

## 7.2 Wall-clock per stage (RTX 4000 Ada)
| Stage | Time |
|---|---|
| Teacher data generation (146 databases, all filters) | ~7.5 h |
| Baseline evaluations (full dev incl. latency and throughput passes) | teacher zero-shot 25 min, teacher few-shot 35 min, untuned student 5.5 min |
| Training: human | 94 min |
| Training: teacher | 250 min |
| Training: combined | 350 min |
| Training: teacher 25% / 50% | 64 / 125 min |
| Training: combined 25% / 50% teacher | 161 / 225 min |
| Full dev evaluation of one student (accuracy + latency + throughput) | ~5–6 min |
| Merge + verify + merged eval | ~10 min |
| GGUF conversion + quantisation | ~1 min |
| CPU evaluations (100 questions): student | 2–10 min each |
| Teacher GGUF conversion + quantisation (llama.cpp) | ~5 min |
| CPU evaluations (100 questions): teacher Q4_K_M / Q8_0 | 18 / 25 min |

Training time totals about 21 h for the seven runs. All in all roughly 30 GPU-hours, of which data
generation and training are almost all.

Training throughput was about 5 s per optimiser step (effective batch 16) with peak memory 8.3–8.4 GiB;
the combined-25% run was stopped part-way and redone from scratch, which cost about 2 h more that is not included above.

## 7.3 Size of the deliverable
- Code: about 3,100 lines in `src/sensei_sql/` (packages `core`, `teacher`, `eval`, `train`, `app`), 66 tests.
- Data produced: `data/distill/teacher_pairs.jsonl` (23,703 pairs), per-run predictions in `outputs/eval/`.
- Models produced: 7 LoRA adapters (~82 MB each for the combined variant), the merged model (2.9 GiB), three
  GGUF files (0.9 / 1.5 / 2.9 GiB), an Ollama model `sensei-sql`.
