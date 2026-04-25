# RAPM Baseline Replication with GPT-4.1

Replication of the RAPM (Retrieval-Augmented Protein Modeling) baseline from
Wu et al. 2025 (arXiv:2505.20354), evaluated on the Prot-Inst-OOD dataset.

## Setup
- **Inference LLM**: GPT-4.1 (`gpt-4.1-2025-04-14`)
- **Judge LLM**: Claude Sonnet 4.6 (`claude-sonnet-4-6`)
- **Retrieval**: ESM-2 (1280-dim) + MMseqs2, hybrid α=0.5, top-K=10
- **Test set**: Full Prot-Inst-OOD (14,503 samples across 4 tasks) for lexical metrics; 500 samples/task subset for LLM-as-Judge
- **Hardware**: MIT ORCD Engaging cluster, NVIDIA L40S (embedding stage only)
- **Inference cost**: $67.19 (full test set, GPT-4.1)
- **Judge cost**: $18.86 (2,000 samples, Claude Sonnet 4.6)

## Tasks
| Task | Test samples |
|---|---|
| catalytic_activity_OOD | 1,987 |
| domain_motif_OOD | 2,732 |
| general_function_OOD | 4,297 |
| protein_function_OOD | 5,487 |

## Headline Results

### RAPM rescore metrics — full test set, Llama-3.2-1B tokenizer

| Task | BLEU-2 | BLEU-4 | Meta-BLEU-2 | Meta-BLEU-4 | METEOR | Meta-METEOR | ROUGE-L |
|---|---|---|---|---|---|---|---|
| catalytic_activity_OOD | 36.82 | 27.56 | 26.96 | 22.49 | 48.69 | 39.37 | 46.69 |
| domain_motif_OOD | 17.10 | 11.62 | 19.23 | 15.23 | 38.45 | 30.16 | 30.94 |
| general_function_OOD | 24.07 | 15.91 | 9.54 | 6.61 | 33.08 | 28.77 | 30.64 |
| protein_function_OOD | 24.02 | 15.23 | 33.51 | 25.71 | 42.59 | 43.39 | 28.36 |

### Entity-BLEU / Entity-F1 — full test set

| Task | E-BLEU-2 | E-BLEU-4 | E-Precision | E-Recall | E-F1 |
|---|---|---|---|---|---|
| catalytic_activity_OOD | 10.03 | 1.88 | 31.49 | 36.73 | 32.05 |
| domain_motif_OOD | 0.77 | 0.05 | 7.52 | 14.26 | 9.20 |
| general_function_OOD | 0.07 | 0.01 | 2.20 | 1.96 | 1.99 |
| protein_function_OOD | 1.03 | 0.04 | 9.05 | 18.69 | 11.34 |
| **Average** | **2.97** | **0.49** | **12.57** | **17.91** | **13.64** |

### LLM-as-Judge — Claude Sonnet 4.6, 500 samples/task, 0–10 scale

| Task | Recall | Precision | Specificity | Plausibility | Final | n_scored |
|---|---|---|---|---|---|---|
| catalytic_activity_OOD | 3.87 | 5.31 | 9.54 | 5.98 | **6.28** | 497 |
| domain_motif_OOD | 4.18 | 6.19 | 8.65 | 6.81 | **6.62** | 492 |
| general_function_OOD | 4.40 | 5.22 | 6.97 | 5.83 | **5.74** | 489 |
| protein_function_OOD | 5.60 | 5.81 | 8.15 | 6.44 | **6.66** | 500 |
| **Macro mean** | **4.51** | **5.63** | **8.33** | **6.26** | **6.32** | 1,978 |

## Pipeline

1. **Stage 1 — Retrieval-augmented prompt construction** (RAPM repo): ESM-2 embeddings, hybrid FAISS HNSW + MMseqs2 retrieval, build prompts with combined-task retrieval pool and task-specific few-shot examples.
2. **Stage 2 — GPT-4.1 inference** (`scripts/gpt41_inference.py`): async parallelism (50 concurrent), exponential backoff, resumable checkpointing.
3. **Stage 3 — Lexical metric scoring** (saper-clip-benchmarks): `rapm_rescore.py` and `fix_entity_bleu.py`.
4. **Stage 4 — LLM-as-Judge scoring** (`scripts/llm_judge_score_anthropic.py`): Claude Sonnet 4.6 scoring on 4 axes (recall, precision, specificity, plausibility) with rubric from saper-clip-benchmarks.

## How to reproduce

```bash
# Stage 2: inference (after Stage 1 prompts are built)
python scripts/gpt41_inference.py --task all --concurrency 50

# Stage 3: lexical metrics (requires saper-clip-benchmarks repo)
python rapm_rescore.py --results_dir results/predictions --dataset_dir <DATASET>
python fix_entity_bleu.py --results_dir results/predictions --dataset_dir <DATASET>

# Stage 4: LLM-as-Judge with Claude Sonnet 4.6
export ANTHROPIC_API_KEY="sk-ant-..."
python scripts/llm_judge_score_anthropic.py \
  --predictions_dir results/predictions \
  --dataset_dir <DATASET> \
  --num_samples 500 \
  --max_concurrent 5
```

## Notes
- Retrieval uses a **combined training pool across all 4 tasks** (~204k proteins), but few-shot examples are task-specific.
- Llama-3.2-1B tokenizer used for Meta-BLEU (matches RAPM paper methodology).
- Judge script adapted from `saper-clip-benchmarks/llm_judge_score.py` to use Anthropic API; rubric and prompt unchanged.
- ~1.5% of judge samples failed to parse; these are excluded from per-axis means.

## References
- RAPM: Wu et al. 2025, arXiv:2505.20354
- Dataset: TimeRune/Prot-Inst-OOD (HuggingFace)
- Benchmarks repo: github.com/ywliang1/saper-clip-benchmarks
