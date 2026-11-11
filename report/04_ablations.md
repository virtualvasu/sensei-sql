# 4. Ablation studies

Two questions: (1) does teacher-generated data help, compared with Spider's human labels? (2) how much
teacher data is needed? Same training settings throughout (see 02_methodology.md); only the data changes.
Data: `data/ablation_data_source.csv`, `data/ablation_data_size.csv`, figure `figures/ablation_data_size.png`.

## 4.1 Ablation 1: training data source

| training data | examples | EX | easy | medium | hard | extra | gain over untuned |
|---|---|---|---|---|---|---|---|
| none (untuned student) | 0 | 0.672 | 0.867 | 0.729 | 0.511 | 0.398 | – |
| (a) human only (Spider train) | 8,406 | 0.755 | 0.907 | 0.812 | 0.632 | 0.506 | +8.3 |
| (b) teacher only | 23,149 | 0.769 | 0.911 | 0.834 | 0.713 | 0.440 | +9.7 |
| (c) human + teacher | 31,552 | 0.772 | 0.923 | 0.807 | 0.707 | 0.518 | +10.0 |

Findings
- Every kind of fine-tuning helps a lot (+8 to +10 points) over the untuned student.
- Teacher data alone matches or beats the human labels (0.769 vs 0.755), even though no human wrote it.
  The 1.4-point gap is inside the noise (about one standard error), so the honest claim is "at least as good",
  not "better". The teacher set is also 2.8× larger, so size and source are confounded here.
- Adding human data to the teacher data gives a further +0.3 points, which is not significant overall.
- The hardness breakdown is more informative than the headline: the teacher data helps most on hard
  questions (0.713 vs 0.632 for human-only), while the human data helps most on extra-hard (0.506 vs 0.440 for
  teacher-only). The combination gets both (0.707 hard, 0.518 extra), which is the best reason to prefer (c).
- Likely explanation (a hypothesis, not tested): the synthetic data is designed to cover patterns such as nested
  queries and set operations evenly, while Spider's human questions contain more of the long, awkward
  multi-table phrasings that matter for extra-hard questions.

## 4.2 Ablation 2: amount of teacher data

Share of the 23,703 teacher pairs used for training (the combined variant always keeps all 8,406 human examples).

| teacher pairs used | teacher-only: examples | EX | combined: examples | EX |
|---|---|---|---|---|
| 25% | 5,786 | 0.750 | 14,185 | 0.767 |
| 50% | 11,566 | 0.760 | 19,973 | 0.759 |
| 100% | 23,149 | 0.769 | 31,552 | 0.772 |

Findings
- **Teacher-only improves steadily** with data: +1 point per doubling (0.750, 0.760, 0.769), with no sign of
  saturation at the full set. More verified teacher data would probably help further.
- **Combined is flat within noise** (0.759 to 0.772). With the human data already in the mix, 25% of the
  teacher data (about 5,800 pairs) gets 0.767, within about 0.5 points of using all of it. The dip at 50% is
  about one standard error and should not be read as a real effect.
- Practical reading: if data generation is expensive, human data plus a quarter of the teacher pairs recovers
  almost everything. Teacher generation took about 7.5 h on this GPU.
- **Caveat:** one training run per setting (seed 0). Differences of one to two points between neighbouring
  settings are not conclusive. Repeating each setting with a few seeds is the obvious way to tighten this.

## 4.3 Other comparisons that act as ablations
- **Few-shot vs zero-shot (teacher):** 3 examples from training databases did not help (0.775 vs 0.782) and
  made it 1.3× slower, so zero-shot was used for all later work.
- **Merged vs adapter (same weights):** 0.773 vs 0.772, so merging loses nothing, and removes the adapter's
  latency and memory overhead.
- **Quantisation (CPU subset):** fp32 0.83, Q8_0 0.80, Q4_K_M 0.77 on 100 questions. See 03_results.md.
