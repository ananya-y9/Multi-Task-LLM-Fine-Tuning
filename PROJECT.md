# Multi-Task LLM Fine-Tuning

## Objective

Fine-tune a base large language model to perform well on three tasks simultaneously:

1. **Instruction Following** — evaluated on [IFEval](https://arxiv.org/abs/2311.07911)
2. **Math Reasoning** — evaluated on [GSM8K](https://arxiv.org/abs/2110.14168)
3. **Code Generation** — evaluated on [HumanEval](https://arxiv.org/abs/2107.03374)

All training can be done via LoRA fine-tuning on the [Tinker](https://github.com/thinking-machines-lab) platform, which provides API-based access to models and training infrastructure.

### Task Details

| Task | Benchmark | Samples | What It Measures | Metric |
|------|-----------|---------|------------------|--------|
| Instruction Following | IFEval | 541 | Whether the model follows verifiable constraints in prompts (e.g., "write exactly 3 paragraphs", "include the keyword X", "respond in all caps"). | Average of prompt-level and instruction-level accuracy (strict + loose) |
| Math Reasoning | GSM8K | 1,319 | Multi-step grade-school math word problems requiring arithmetic reasoning. The model must show its work and produce a final numeric answer. Evaluated zero-shot (no examples provided). | Exact-match on the final numeric answer |
| Code Generation | HumanEval | 164 | Python function completion given a docstring specification. The generated code is executed against hidden unit tests. | pass@1 (fraction of problems where the generated code passes all tests) |

---

## Model Constraints

- Available models:
  - `meta-llama/Llama-3.2-1B` 
  - `meta-llama/Llama-3.2-3B` 
  - `meta-llama/Llama-3.1-8B` 

## Passing Baseline

| Task | Baseline Score |
|------|---------------|
| IFEval | 45.0% |
| GSM8K | 50.0% |
| HumanEval | 30.0% |

---

## Suggested Datasets

| Dataset | Task | Size |
|---------|------|------|
| [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) (train split) | Math | 7,473 |
| [`allenai/tulu-3-sft-mixture`](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) | IF | ~939,000 |
| [`nvidia/OpenCodeInstruct`](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) | Code | ~5M |

> **Do NOT train on test data.** Only use training splits for training. Do not train on the IFEval prompts, GSM8K test split, or HumanEval problems. 

---

## Explore

- **Data mixing strategies** — ratios, curriculum ordering, sampling weights
- **Hyperparameter tuning** — learning rate, LoRA rank, batch size, number of steps
- **Data selection and filtering** — quality filtering, deduplication, difficulty-based selection
- **Data augmentation** — generate additional training data or use alternative datasets
- **Reinforcement learning** — GRPO or other RL methods after SFT. Be careful: RL can improve one task at the cost of degrading others. Always evaluate all three tasks after any RL stage to ensure it does not harm overall performance.

## Tips

- **Start small.** Use Llama-3.2-1B or 3B for rapid experimentation. Training is much faster on smaller models, and trends often transfer to the 8B model.
- **Evaluate intermediate checkpoints.** The final checkpoint is often not the best. Save checkpoints every N steps and evaluate each one. Overtraining is the main risk.
- **Watch for catastrophic forgetting.** Training on one task can hurt performance on others. Multi-task training helps prevent this.
