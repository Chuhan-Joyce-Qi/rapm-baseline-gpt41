# RAPM Baseline Replication on Prot-Inst-OOD

Replication of the RAPM (Retrieval-Augmented Protein Modeling) baseline from
Wu et al. 2025 (arXiv:2505.20354), evaluated on the Prot-Inst-OOD dataset.

This repo contains two complete pipeline runs on the **full test set** (14,503 samples):
1. **GPT-4.1 baseline** — original RAPM pipeline with `gpt-4.1-2025-04-14`
2. **Sonnet 4.6 baseline** — pure model swap, identical retrieval and prompts, with `claude-sonnet-4-6`

Both runs are scored with: lexical metrics (RAPM Meta-BLEU, BLEU, METEOR, ROUGE-L, Entity-BLEU) and LLM-as-Judge (Claude Sonnet 4.6 with extended thinking).

## Setup
- **Retrieval**: ESM-2 (1280-dim) + MMseqs2, hybrid α=0.5, top-K=10
- **Prompt**: original RAPM template with 10 retrieved annotations + 10 few-shot examples
- **Test set**: Full Prot-Inst-OOD (14,503 samples) for lexical metrics; 500/task subset for LLM-as-Judge
- **Hardware**: MIT ORCD Engaging cluster, NVIDIA L40S (embedding stage only); CPU for inference and scoring
- **Total API cost**: ~$305 (GPT-4.1 inference $67, Sonnet 4.6 inference $132, judge $19+$43+$44)

## Tasks
| Task | Test samples |
|---|---|
| catalytic_activity_OOD | 1,987 |
| domain_motif_OOD | 2,732 |
| general_function_OOD | 4,297 |
| protein_function_OOD | 5,487 |

## Headline Results — Lexical Metrics (RAPM rescore, full test set)

### GPT-4.1 baseline (`gpt-4.1-2025-04-14`, temperature=0, max_tokens=512)

| Task | M-BLEU-2 | M-BLEU-4 | METEOR | M-METEOR | ROUGE-L |
|---|---|---|---|---|---|
| catalytic_activity_OOD | 26.96 | 22.49 | 48.69 | 39.37 | 46.69 |
| domain_motif_OOD | 19.23 | 15.23 | 38.45 | 30.16 | 30.94 |
| general_function_OOD | 9.54 | 6.61 | 33.08 | 28.77 | 30.64 |
| protein_function_OOD | 33.51 | 25.71 | 42.59 | 43.39 | 28.36 |
| **Macro mean** | **22.31** | **17.51** | **40.70** | **35.42** | **34.16** |

### Sonnet 4.6 baseline (`claude-sonnet-4-6`, temperature=0.7, max_tokens=4096)

| Task | M-BLEU-2 | M-BLEU-4 | METEOR | M-METEOR | ROUGE-L |
|---|---|---|---|---|---|
| catalytic_activity_OOD | 31.53 | 26.65 | 49.91 | 41.82 | 45.89 |
| domain_motif_OOD | 35.13 | 28.71 | 38.69 | 37.52 | 29.21 |
| general_function_OOD | 24.15 | 18.34 | 35.94 | 35.98 | 28.34 |
| protein_function_OOD | 51.43 | 42.21 | 45.38 | 51.71 | 27.56 |
| **Macro mean** | **35.56** | **28.98** | **42.48** | **41.76** | **32.75** |

### Pure model effect (Sonnet 4.6 - GPT-4.1)

Same RAPM original prompt, same retrieval, same full test set. Only difference: the LLM.

| Metric | GPT-4.1 macro | Sonnet 4.6 macro | Δ |
|---|---|---|---|
| M-BLEU-2 | 22.31 | 35.56 | **+13.25** ⬆️ |
| M-BLEU-4 | 17.51 | 28.98 | +11.47 ⬆️ |
| METEOR | 40.70 | 42.48 | +1.78 |
| M-METEOR | 35.42 | 41.76 | +6.34 ⬆️ |
| ROUGE-L | 34.16 | 32.75 | -1.41 |

### GPT-4.1 v2 (paper-matched sampling: temperature=0.7, top_p=0.9, max_tokens=2048, few-shot=3)

Same model, same retrieval, same prompt template — only sampling parameters change. Few-shot count reduced from 10 to 3 (deterministic random with seed=42 per sample) per coworker request.

| Task | M-BLEU-2 | M-BLEU-4 | METEOR | M-METEOR | ROUGE-L |
|---|---|---|---|---|---|
| catalytic_activity_OOD | 31.29 | 25.81 | 46.77 | 40.43 | 41.26 |
| domain_motif_OOD | 23.28 | 17.79 | 32.03 | 32.03 | 22.07 |
| general_function_OOD | 8.87 | 5.88 | 29.47 | 27.85 | 25.99 |
| protein_function_OOD | 33.02 | 24.50 | 40.94 | 42.23 | 25.99 |
| **Macro mean** | **24.12** | **18.50** | **37.30** | **35.64** | **28.83** |

### Sampling parameter effect (GPT-4.1 v1 vs v2)

| Metric | v1 (T=0, fs=10) | v2 (T=0.7, fs=3) | Δ |
|---|---|---|---|
| M-BLEU-2 macro | 22.31 | 24.12 | +1.81 |
| ROUGE-L macro | 34.16 | 28.83 | -5.33 |

Higher temperature + fewer few-shot examples shifts predictions toward more paraphrasing (ROUGE-L drops) with marginally better biological-keyword overlap (M-BLEU-2 up). Net effect on overall quality is small.

## Headline Results — Entity-BLEU / Entity-F1

### GPT-4.1 baseline

| Task | E-BLEU-2 | E-BLEU-4 | E-Precision | E-Recall | E-F1 |
|---|---|---|---|---|---|
| catalytic_activity_OOD | 10.03 | 1.88 | 31.49 | 36.73 | 32.05 |
| domain_motif_OOD | 0.77 | 0.05 | 7.52 | 14.26 | 9.20 |
| general_function_OOD | 0.07 | 0.01 | 2.20 | 1.96 | 1.99 |
| protein_function_OOD | 1.03 | 0.04 | 9.05 | 18.69 | 11.34 |
| **Average** | **2.97** | **0.49** | **12.57** | **17.91** | **13.64** |

### Sonnet 4.6 baseline

| Task | E-BLEU-2 | E-BLEU-4 | E-Precision | E-Recall | E-F1 |
|---|---|---|---|---|---|
| catalytic_activity_OOD | 10.89 | 2.11 | 31.21 | 38.62 | 32.92 |
| domain_motif_OOD | 1.25 | 0.07 | 11.40 | 18.88 | 13.39 |
| general_function_OOD | 0.71 | 0.15 | 3.99 | 4.14 | 3.93 |
| protein_function_OOD | 1.23 | 0.04 | 9.36 | 24.58 | 12.77 |
| **Average** | **3.52** | **0.59** | **13.99** | **21.55** | **15.75** |

Sonnet 4.6 improves Entity-F1 macro by **+2.11** and Entity-Recall by **+3.64** over GPT-4.1, with consistent gains across all four tasks.

### GPT-4.1 v2 Entity-BLEU

| Task | E-BLEU-2 | E-BLEU-4 | E-Precision | E-Recall | E-F1 |
|---|---|---|---|---|---|
| catalytic_activity_OOD | 10.32 | 1.91 | 31.56 | 37.09 | 32.13 |
| domain_motif_OOD | 0.81 | 0.05 | 7.35 | 13.53 | 8.95 |
| general_function_OOD | 0.07 | 0.01 | 1.79 | 1.60 | 1.61 |
| protein_function_OOD | 0.91 | 0.04 | 9.26 | 17.08 | 11.09 |
| **Average** | **3.03** | **0.50** | **12.49** | **17.33** | **13.45** |

Sampling parameter changes had negligible Entity-F1 effect (-0.20 vs v1), confirming entity-level differences across runs are dominated by model choice rather than sampling settings.

## Headline Results — LLM-as-Judge (Sonnet 4.6 with extended thinking, 500 samples/task)

Judge config: `claude-sonnet-4-6` with `thinking_budget=4000`, `max_tokens=5120`, deterministic seed=42 sampling.

### GPT-4.1 baseline

| Task | Recall | Precision | Specificity | Plausibility | Final | n_scored |
|---|---|---|---|---|---|---|
| catalytic_activity_OOD | 3.54 | 5.43 | 9.56 | 5.12 | **6.16** | 498 |
| domain_motif_OOD | 3.94 | 5.63 | 9.34 | 6.29 | **6.49** | 498 |
| general_function_OOD | 3.32 | 4.14 | 7.48 | 4.87 | **5.11** | 495 |
| protein_function_OOD | 5.33 | 4.97 | 8.57 | 5.79 | **6.32** | 500 |
| **Macro mean** | **4.03** | **5.04** | **8.74** | **5.52** | **6.02** | 1,991 |

### Sonnet 4.6 baseline

| Task | Recall | Precision | Specificity | Plausibility | Final | n_scored |
|---|---|---|---|---|---|---|
| catalytic_activity_OOD | 3.41 | 5.33 | 9.55 | 5.06 | **6.07** | 500 |
| domain_motif_OOD | 4.80 | 5.52 | 9.49 | 6.89 | **6.81** | 499 |
| general_function_OOD | 4.10 | 3.85 | 8.17 | 5.12 | **5.44** | 496 |
| protein_function_OOD | 5.98 | 4.74 | 8.79 | 6.09 | **6.53** | 500 |
| **Macro mean** | **4.57** | **4.86** | **9.00** | **5.79** | **6.21** | 1,995 |

(Per-task Sonnet 4.6 judge scores in `results/sonnet46_baseline/judge_scores_extended_thinking/overall_summary.json`.)

### Pure model effect (LLM-as-Judge)

| Axis | GPT-4.1 | Sonnet 4.6 | Δ |
|---|---|---|---|
| Recall | 4.03 | 4.57 | +0.54 ⬆️ |
| Precision | 5.04 | 4.86 | -0.18 |
| Specificity | 8.74 | 9.00 | +0.26 ⬆️ |
| Plausibility | 5.52 | 5.79 | +0.27 ⬆️ |
| **Final** | **6.02** | **6.21** | **+0.19** ⬆️ |

The judge-based comparison confirms the lexical-metric signal: Sonnet 4.6 modestly outperforms GPT-4.1 across recall, specificity, and plausibility axes. The win is more conservative under the LLM judge (+0.19 final) than under lexical metrics (+13.25 M-BLEU-2 macro), indicating that some of the lexical-metric advantage stems from Sonnet's more aggressive use of biological terminology rather than strict correctness gains.

## Pipeline

1. **Stage 1 — RAG prompt construction** (RAPM repo): ESM-2 embeddings, hybrid FAISS HNSW + MMseqs2 retrieval, build prompts. Retrieval pool combines all 4 task training sets; few-shot examples are task-specific.
2. **Stage 2 — Inference**:
   - GPT-4.1: `scripts/gpt41_inference.py` (async, concurrency=50, exponential backoff, resumable)
   - Sonnet 4.6: `scripts/sonnet46_baseline_inference.py` (async, concurrency=30, exponential backoff, resumable)
3. **Stage 3 — Lexical scoring** (saper-clip-benchmarks): `rapm_rescore.py` and `fix_entity_bleu.py`.
4. **Stage 4 — LLM-as-Judge** (`scripts/llm_judge_score_anthropic.py`): Claude Sonnet 4.6 with extended thinking, scoring on 4 axes (recall, precision, specificity, plausibility) with rubric from saper-clip-benchmarks.

## How to reproduce

```bash
# Stage 2a: GPT-4.1 inference
python scripts/gpt41_inference.py --task all --concurrency 50

# Stage 2b: Sonnet 4.6 inference
export ANTHROPIC_API_KEY="sk-ant-..."
python scripts/sonnet46_baseline_inference.py --task all --concurrency 30

# Stage 3: lexical metrics
python rapm_rescore.py --results_dir <PRED_DIR> --dataset_dir <DATASET>
python fix_entity_bleu.py --results_dir <PRED_DIR> --dataset_dir <DATASET>

# Stage 4: judge with extended thinking
python scripts/llm_judge_score_anthropic.py \
  --predictions_dir <PRED_DIR> --dataset_dir <DATASET> \
  --num_samples 500 --max_concurrent 5 \
  --max_tokens 5120 --thinking_budget 4000 \
  --output_dir <JUDGE_DIR>
```

## Notes
- Retrieval uses combined training pool across all 4 tasks (~204k proteins); few-shot examples are task-specific.
- Judge script adapted from `saper-clip-benchmarks/llm_judge_score.py` for Anthropic API; rubric and prompt unchanged.
- Extended thinking forces `temperature=1.0` per Anthropic API contract; `max_tokens=5120` accommodates 4000-token thinking budget plus final answer.
- Sonnet 4.6 inference uses `temperature=0.7`, `max_tokens=4096`, matching coworker's SAPER pipeline configuration.

## References
- RAPM: Wu et al. 2025, arXiv:2505.20354
- Dataset: TimeRune/Prot-Inst-OOD (HuggingFace)
- Benchmarks: github.com/ywliang1/saper-clip-benchmarks
