# Suggested outline for the final report

| Section | Content | Source files |
|---|---|---|
| Abstract | Question, method in one sentence, headline: 98.8% of teacher accuracy, 4.6× faster, 4.5× smaller | README.md (summary), 03 §3.2 |
| 1. Introduction | Why small text-to-SQL models; distillation; contributions | 01 |
| 2. Background and related work | Spider, execution accuracy, QLoRA, distillation | 10 |
| 3. Method | Prompting and safe execution; teacher data pipeline and filters; fine-tuning; export | 02 |
| 4. Experimental setup | Data, models, hardware, metrics, decoding, baselines | 02 §2.2, 01, 08 |
| 5. Results | Main table, accuracy vs latency, hardness, CPU | 03, figures/ |
| 6. Ablations | Data source; amount of teacher data | 04, figures/ablation_data_size.png |
| 7. Error analysis | Categories, student vs teacher examples | 05, samples/ |
| 8. Deployment | Merge, GGUF, the offline app, four backends | 02 §2.5–2.6, 03 §3.4 |
| 9. Discussion and limitations | What is and is not established | 09 |
| 10. Conclusion and future work | 98.8% retained, next steps | 03 §3.2, 09 §9.2 |
| Appendix | Hyper-parameters, filter pass rates, challenges, timeline, reproduction commands | 02, 06, 07, 08, data/ |

Figures worth including: `accuracy_vs_latency.png` (the central trade-off), `accuracy_by_hardness.png`,
`ablation_data_size.png`, one loss curve (`loss_combined.png`), and the filter pass-rate table from 02 §2.3.

Examples worth quoting: `samples/teacher_pairs_sample_200.jsonl` (what the distilled data looks like) and the
side-by-side cases in `data/results_generated.md`.
