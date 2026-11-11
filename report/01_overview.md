# 1. Overview

## The problem
Text-to-SQL means turning a question in plain English ("Which singers are older than the average
age?") into a SQL query over a given database. Large language models do this well but are expensive
to run: the teacher used here is a 9B-parameter model that needs about 15 GB of GPU memory and about 1 s per
question. The goal of this project is to get most of that accuracy from a model small enough to run
quickly on modest hardware, ideally with no network at all.

## The approach: knowledge distillation
1. A **teacher** (Qwen3.5-9B, FP8) produces training data: new question/SQL pairs for each training database,
   and its own answers to Spider's existing training questions.
2. The data is **verified by execution**: SQL that does not run, returns nothing, or is not confirmed by
   independent checks is discarded. What remains is 23,703 pairs.
3. A **student** (Qwen2.5-Coder-1.5B-Instruct) is fine-tuned on the data with **QLoRA** (a 4-bit base
   model plus small trainable LoRA adapters), in three variants: human data only (Spider's own labels),
   teacher data only, and both.
4. Everything is measured on **Spider dev** (1,034 questions over 20 databases the models never trained on)
   with the official test-suite execution-accuracy scorer, against the teacher and the untuned student.
5. The best student is merged into plain weights, converted to GGUF for CPU use, and wrapped in a local
   web app.

## Why these choices
- **Spider 1.0** is the standard cross-domain text-to-SQL benchmark: dev databases are disjoint from
  training databases, so it measures generalisation to unseen schemas.
- **Execution accuracy** (does the predicted SQL return the same result as the gold SQL) is used because many
  different SQL strings are correct. The test-suite variant checks the query on several database copies to
  reduce false positives.
- **FP8 teacher.** Qwen3.5-9B in bf16 needs ~19 GB, which does not fit with a KV cache on a 20 GB card; the
  FP8 checkpoint runs natively on the Ada GPU.
- **QLoRA** fits fine-tuning of the student on the same 20 GB card (peak ~8.4 GB).

## Hardware and software
- One NVIDIA RTX 4000 Ada (20 GB), driver 570 / CUDA 12.8, 31 GB RAM, Intel i9-14900K (32 threads).
- Constraint set by the owner: stay on the existing NVIDIA 570 / CUDA 12.8 driver. This fixed the stack at
  torch 2.10.0+cu128 and vLLM 0.19.1.
- Python 3.10.12, venv in the project root.

## The seven work packages (GitHub issues)
| # | Work package | Outcome |
|---|---|---|
| 1 | Setup: environment, data, models | Spider, both models and the scorer downloaded and checked offline |
| 2 | Shared core | Schema formatter, prompt templates, safe SQL runner, result comparison |
| 3 | Evaluation harness and baselines | Teacher 0.782 (zero-shot), 0.775 (3-shot), untuned student 0.672 |
| 4 | Teacher data generation | 23,703 execution-verified pairs over 146 training databases |
| 5 | QLoRA fine-tuning and export | Three variants, best merged to 0.773 EX; GGUF export |
| 6 | Final evaluation, ablations, error analysis | Results tables, plots, two ablations, error analysis |
| 7 | Offline app | Gradio app with four backends (vLLM, transformers, llama.cpp, Ollama) |

The code is in `src/sensei_sql/` (about 3,100 lines in five packages: `core`, `teacher`, `train`, `eval`,
`app`), with 66 unit tests.
