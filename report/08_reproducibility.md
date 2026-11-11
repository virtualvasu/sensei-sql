# 8. Reproducibility

## 8.1 Pinned versions
| Item | Version |
|---|---|
| Python | 3.10.12 (venv `.venv` in the repo root) |
| torch | 2.10.0+cu128 |
| vLLM | 0.19.1 |
| transformers / TRL / PEFT | 5.20.2 / 1.14.2 / 0.21.2 |
| bitsandbytes | 0.50.2 |
| Gradio | 6.30.0 |
| sqlglot | 30.24.0 |
| Teacher | `RedHatAI/Qwen3.5-9B-FP8-dynamic` @ `790f0576d2d77dd5322aa0603a470bd9e3a3d1f6` |
| Student | `Qwen/Qwen2.5-Coder-1.5B-Instruct` @ `2e1fd397ee46e1388853d2af2c993145b0f1098a` |
| Scorer | `taoyds/test-suite-sql-eval` @ `e97acc5` |
| llama.cpp | the clone in `third_party/llama.cpp` @ `620d222`, built CPU-only |
| Hardware | RTX 4000 Ada 20 GB, NVIDIA driver 570.133 (CUDA 12.8), 31 GB RAM, i9-14900K |

All seeds are 0; evaluation decoding is greedy.

## 8.2 Commands (from the repository root)
```bash
make install                 # venv + dependencies (cu128 wheels)
make download                # Spider, both models, the scorer, NLTK data (needs network once)
make check && make test      # setup check, 66 unit tests
make teacher                 # teacher data -> data/distill/teacher_pairs.jsonl (~7.5 h)
python -m sensei_sql.eval --model teacher --split dev            # baselines
python -m sensei_sql.eval --model teacher --split dev --few-shot
python -m sensei_sql.eval --model student --split dev
python -m sensei_sql.train configs/train/{human,teacher,combined}.yaml     # make train
python -m sensei_sql.train configs/train/teacher.yaml  --teacher-fraction 0.25   # ablations (also 0.5, and combined)
python -m sensei_sql.eval --model student --adapter models/adapters/<run>
python -m sensei_sql.train.export merge --adapter models/adapters/combined
python -m sensei_sql.train.export verify --adapter models/adapters/combined [--gguf]
python -m sensei_sql.eval --model models/student-merged
python -m sensei_sql.train.export gguf       # after building llama.cpp (see README)
python -m sensei_sql.eval.cpu --backend llama-server --gguf models/student-gguf/student-Q4_K_M.gguf --limit 100
make results                 # tables, plots, error analysis -> outputs/results/
make app                     # offline web app
```

## 8.3 Where things are
| What | Where | In git? |
|---|---|---|
| Code, configs, tests, README | `src/`, `configs/`, `tests/`, repo root | yes |
| Results tables and plots | `docs/results/`, `docs/results_draft.md`, this `report/` folder | yes |
| Per-run predictions and metrics | `outputs/eval/<run>/{predictions.jsonl,metrics.json}` | no (git-ignored) |
| Training logs, loss curves, notes | `outputs/train/<run>/` | no |
| Teacher data | `data/distill/teacher_pairs.jsonl`, `stats.md` | no |
| Spider | `data/spider/` | no |
| Models, adapters, merged, GGUF | `models/` | no |
| Ollama import | `Modelfile` (in git), model `sensei-sql` in the local Ollama store | file only |

Every number in the results tables can be reproduced from the saved `predictions.jsonl` + `metrics.json` of
the run (`python -m sensei_sql.eval.score outputs/eval/<run>` re-scores a run; `python -m
sensei_sql.eval.aggregate` rebuilds the summary table). Configs are saved with each training run in
`outputs/train/<run>/config.json`.

## 8.4 Caveat on exact reproduction
Greedy decoding with a fixed seed is deterministic on the same hardware and software, but batched GPU kernels
can produce small numeric differences between machines or batch sizes. The measured "same SQL under batching"
rate was 85% (teacher) to 100% (merged student) depending on the model (`batched_same_sql_rate` in each `metrics.json`), so expect
accuracy to reproduce to within a few questions rather than bit-for-bit. Training is not bit-reproducible
across hardware (4-bit kernels, gradient checkpointing).
