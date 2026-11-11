# 10. References and sources

**Dataset and metric**
- Yu, T. et al. (2018). *Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic
  Parsing and Text-to-SQL Task.* EMNLP 2018.
- Zhong, R., Yu, T., Klein, D. (2020). *Semantic Evaluation for Text-to-SQL with Distilled Test Suites.* EMNLP
  2020. Code: https://github.com/taoyds/test-suite-sql-eval (used at commit e97acc5).

**Methods**
- Hu, E. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.*
- Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.*
- Hinton, G., Vinyals, O., Dean, J. (2015). *Distilling the Knowledge in a Neural Network.*
- Wang, X. et al. (2023). *Self-Consistency Improves Chain of Thought Reasoning in Language Models* (the idea
  behind the self-consistency filter, here applied to execution results).

**Models**
- Teacher: Qwen3.5-9B, FP8-dynamic quantisation by RedHatAI (`RedHatAI/Qwen3.5-9B-FP8-dynamic`, revision 790f057).
- Student: Qwen2.5-Coder-1.5B-Instruct (`Qwen/Qwen2.5-Coder-1.5B-Instruct`, revision 2e1fd39); see the Qwen2.5-Coder
  technical report (Hui, B. et al., 2024).

**Software**
- vLLM (Kwon, W. et al., 2023, *Efficient Memory Management for Large Language Model Serving with
  PagedAttention*), Hugging Face transformers, TRL, PEFT, bitsandbytes, Gradio, sqlglot, llama.cpp (GGUF,
  `llama-quantize`, `llama-server`), Ollama.

Check the exact titles, years and author lists against the original sources before citing; they are given here
from memory of the standard references.
