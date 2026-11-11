# 3. Results

All on Spider dev (1,034 questions, 20 databases not used in training), zero-shot, greedy decoding,
execution accuracy (EX) from the official test-suite scorer. Raw numbers: `data/eval_all_runs.csv`.
Standard error of an EX near 0.77 is about ±1.3 points.

## 3.1 Main table (GPU: RTX 4000 Ada, vLLM)

| model | EX | easy | medium | hard | extra | p50 ms | p95 ms | q/s (bs 32) | peak VRAM GiB | size GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| Teacher Qwen3.5-9B FP8, zero-shot | **0.782** | 0.940 | 0.834 | 0.707 | 0.488 | 945 | 2,169 | 4.2 | 15.5 | 13.1 |
| Teacher, 3-shot | 0.775 | 0.944 | 0.848 | 0.667 | 0.440 | 1,252 | 2,480 | 1.7 | 16.0 | 13.1 |
| Student, no fine-tuning | 0.672 | 0.867 | 0.729 | 0.511 | 0.398 | 214 | 598 | 42.7 | 6.5 | 2.9 |
| Student (a) human-only (adapter) | 0.755 | 0.907 | 0.812 | 0.632 | 0.506 | 224 | 638 | 38.7 | 7.2 | 3.0 |
| Student (b) teacher-only (adapter) | 0.769 | 0.911 | 0.834 | 0.713 | 0.440 | 244 | 583 | 42.4 | 7.2 | 3.0 |
| Student (c) human + teacher (adapter) | 0.772 | 0.923 | 0.807 | 0.707 | 0.518 | 235 | 623 | 39.0 | 7.2 | 3.0 |
| **Student (c) merged fp16 (final)** | **0.773** | 0.919 | 0.812 | 0.713 | 0.512 | **206** | 550 | **44.1** | **6.5** | **2.9** |

The adapter rows run the LoRA through vLLM, which costs some latency and memory; merging removes that.
Few-shot prompting did not help the teacher (0.775 vs 0.782).

## 3.2 The answer to the project question
Fraction of the teacher's accuracy retained and the cost saved, final merged student vs teacher:

| | teacher | student | ratio |
|---|---|---|---|
| execution accuracy | 0.782 | 0.773 | **98.8%** retained |
| median latency (batch 1) | 945 ms | 206 ms | 4.6× faster |
| throughput (batch 32) | 4.2 q/s | 44.1 q/s | 10.6× |
| peak GPU memory | 15.5 GiB | 6.5 GiB | 2.4× less |
| size on disk | 13.1 GiB | 2.9 GiB | 4.5× smaller |

Distillation closed about 90% of the gap between the untuned student (0.672) and the teacher:
(0.773 − 0.672) / (0.782 − 0.672) ≈ 0.92.

## 3.3 Where the gain is
Compared with the untuned student, the merged model improves hard questions from 0.51 to 0.71 and
extra-hard from 0.40 to 0.51; on easy questions (0.87 → 0.92) it already was close. On extra-hard it
matches the teacher (0.512 vs 0.488, within noise).

## 3.4 CPU (Intel i9-14900K, 32 threads, batch size 1, first 100 dev questions)
The first 100 questions are easier than average, so these accuracies are comparable with each other only,
not with the full-dev numbers above.

| model | format | runtime | size GiB | p50 | p95 | mean | EX (n=100) |
|---|---|---|---|---|---|---|---|
| student | fp32 | transformers | 2.89 | 5.98 s | 10.6 s | 5.95 s | 0.83 |
| student | GGUF Q8_0 | llama.cpp | 1.53 | 1.24 s | 5.09 s | 1.84 s | 0.80 |
| student | GGUF Q4_K_M | llama.cpp | 0.92 | 0.67 s | 4.86 s | 1.18 s | 0.77 |
| teacher | GGUF Q8_0 | llama.cpp | 9.11 | 13.7 s | 23.5 s | 14.9 s | 0.84 |
| teacher | GGUF Q4_K_M | llama.cpp | 5.38 | 9.86 s | 18.8 s | 10.6 s | 0.84 |

Same machine, same runtime, same quantisation: the student is **14.6× faster than the teacher at Q4_K_M and 11×
faster at Q8_0**, and 5.8× smaller. The student's Q4_K_M is also 8.9× faster than its own fp32 version, and 3×
smaller than fp16 weights.

Accuracy on these 100 questions: the teacher scores 0.84 in both formats; the student 0.80 (Q8_0) and 0.77
(Q4_K_M), which is 95% and 92% of the teacher. That is a larger gap than on the full dev set (0.773 vs 0.782 on
the GPU). The standard error at n=100 is about ±4 points, so the difference is 1.5 to 2 standard errors; it may be
partly noise and partly the cost of 4/8-bit quantisation for the small model, which this experiment does not
separate. Per-hardness figures at n=100 (about 20 questions for extra-hard) are too small to interpret.
The teacher GGUF was converted by llama.cpp from the FP8 checkpoint (dequantised, then quantised again), so it is
not exactly the model used in the GPU runs. The teacher was run with thinking disabled, as on the GPU.

Quantised models often write the same SQL with different case or quote style, so text identity is a poor test:
against the fp32 student only 47 (Q8_0) and 29 (Q4_K_M) of 100 outputs are textually identical, while correctness
agreed on 91 and 86 questions.

## 3.5 Export check
Merged fp16 vs adapter+base, first 50 dev questions, greedy: 49 of 50 identical SQL (47 of 50 in an earlier
run; the remaining differences were only identifier case or spacing and gave the same execution result).

## 3.6 Training cost per run (`data/training_runs.csv`)
| run | train examples | steps | minutes | best val loss |
|---|---|---|---|---|
| human | 8,406 | 1,052 | 94 | 0.065 |
| teacher | 23,149 | 2,894 | 250 | 0.106 |
| combined | 31,552 | 3,944 | 350 | 0.114 |
| teacher 25% | 5,786 | 724 | 64 | 0.131 |
| teacher 50% | 11,566 | 1,446 | 125 | 0.119 |
| combined 25% teacher | 14,185 | 1,774 | 161 | 0.136 |
| combined 50% teacher | 19,973 | 2,498 | 225 | 0.123 |

Validation losses are not comparable across variants: each is measured on that variant's own held-out data
(human labels are more predictable than the varied synthetic questions). Peak training memory was 8.3–8.4 GiB
for every run.

## 3.7 Figures (`figures/`)
- `accuracy_vs_latency.png`: accuracy against median latency for every model (the trade-off picture).
- `accuracy_by_hardness.png`: accuracy per hardness level per model.
- `ablation_data_size.png`: accuracy against share of teacher data used.
- `loss_<run>.png`: training and validation loss for each of the seven runs.
