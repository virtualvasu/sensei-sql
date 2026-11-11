# 9. Limitations and future work

## 9.1 What the results do and do not show
- **One benchmark, one split.** Spider dev (1,034 questions, 20 databases). Spider is SQLite and mostly
  single-domain schemas of moderate size; results may not carry over to other dialects, much larger schemas
  or messier real-world questions. Other benchmarks (BIRD, Spider 2.0, domain data) were not tried.
- **Statistical power.** The standard error of an accuracy figure near 0.77 is about ±1.3 points. Gaps below
  about 2.5 points (teacher-only vs human-only, combined vs teacher-only, most of the data-size curve) are not
  established. Only larger effects are solid: fine-tuning helps by ~8–10 points, and the student ends within 1
  point of the teacher.
- **One seed per setting.** No variance across training seeds was measured.
- **Execution accuracy only.** It credits queries that happen to return the right result by chance and
  penalises correct queries on ambiguous questions. Exact-match accuracy was not reported (it penalises
  correct queries written differently).
- **Teacher is not a ceiling or a gold standard.** Its own EX is 0.782; its mistakes are also in the
  distilled data unless the filters catch them (the filters check consistency and executability, not truth for
  synthesised questions).
- **Data-source confound.** The teacher set is 2.8× larger than the human set, so ablation 1 cannot separate
  "teacher data is better" from "more data is better". Ablation 2 (size) partly addresses this.
- **Training/test overlap in style.** Teacher questions for training databases may resemble dev questions in
  style (same benchmark family). No database or question from dev was used.
- **CPU numbers** use 100 questions (easier than average) and one machine (a desktop-class CPU); there is no
  laptop measurement and no teacher on CPU.
- **Quantisation cost** is measured on 100 questions only (fp32 0.83, Q8_0 0.80, Q4_K_M 0.77).
- **Prompt includes sample rows of the database**; accuracy may depend on that. Not ablated.

## 9.2 Possible next steps
1. Repeat every training setting with 3 seeds and report mean ± standard deviation (about 12 h of GPU per
   extra seed of the best variant).
2. Full-dev evaluation of the quantised GGUF models, and the teacher on CPU (a quantised build) for a CPU
   speed-up figure.
3. Generate more teacher data (the teacher-only curve had not saturated) and test a harder filter.
4. Self-correction in training: the app already retries once on execution errors; training on
   error-and-fix pairs could help.
5. Schema linking, the main error class: add column descriptions or retrieved values to the prompt.
6. Evaluate on BIRD or a held-out in-domain set.
7. Preference-style distillation (teacher-chosen vs student-failed outputs) instead of plain imitation.
