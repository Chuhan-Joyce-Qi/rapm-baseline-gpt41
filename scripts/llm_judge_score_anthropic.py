#!/usr/bin/env python3
"""LLM-as-Judge scoring for the SAPER protein-function benchmark — Anthropic version.

Adaptation of llm_judge_score.py to use the Anthropic API (Claude Sonnet 4.6)
instead of OpenAI. The judge prompt, rubric, parsing, and output format are
unchanged — only the SDK, API call shape, error types, env var, and price
defaults differ.
"""

import argparse
import asyncio
import json
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path

try:
    from tqdm.asyncio import tqdm_asyncio
    from tqdm import tqdm
except ImportError:
    print("ERROR: tqdm not installed. Run: pip install tqdm", file=sys.stderr)
    sys.exit(1)

try:
    from anthropic import AsyncAnthropic, APIError, APITimeoutError, RateLimitError
except ImportError:
    print("ERROR: anthropic SDK not installed. Run: pip install anthropic", file=sys.stderr)
    sys.exit(1)


DEFAULT_DATASETS = [
    "catalytic_activity_OOD",
    "domain_motif_OOD",
    "general_function_OOD",
    "protein_function_OOD",
]

VALID_AXIS_SCORES = {0, 3, 5, 7, 10, -1}


SYSTEM_PROMPT = (
    "You are an expert biocurator evaluating protein function predictions "
    "against curated UniProt ground truth. Apply the rubric strictly and "
    "consistently. Biological knowledge matters — recognize synonyms, "
    "equivalent rephrasings, and EC-class relationships. Identical inputs "
    "must yield identical scores. Scores are chosen from a fixed 5-point "
    "scale: {0, 3, 5, 7, 10}. Do not output other values for any axis."
)

USER_PROMPT_TEMPLATE = """OUTPUT FORMAT IS CRITICAL: Produce ONLY the 8 labeled lines specified at the end. No preamble, no markdown, no step-by-step reasoning, no bullet points. Internal analysis must remain internal — write only the 8 final lines.

Evaluate a PREDICTION against a curated GROUND TRUTH for a protein function prediction task.

---
INPUT

TASK INSTRUCTION:
{instruction}

GROUND TRUTH DESCRIPTION (curated from UniProt):
{description}

KEY BIO-ENTITIES (atomic entities pre-extracted from UniProt annotations; top-level separators are "|" and ";", while "+" and "=" within a single entity denote reaction components):
{metadata}

PREDICTION:
{prediction}

---
STEP 1: Parse reference entities.

Split KEY BIO-ENTITIES into atomic entities using "|" and ";" as top-level separators. Keep reaction components joined: "(R)-prunasin + H2O = D-glucose + mandelonitrile" is ONE catalytic-reaction entity. For concept lists like "lipid binding" or "nucleus, cytoplasm", each comma-separated item is one entity unless the comma is inside a reaction.

Let N_ref = count of atomic reference entities.

STEP 2: Match reference entities — 5 match types.

For each reference entity, determine if it is expressed in PREDICTION using these match types (priority order):

  TYPE 1 — EXACT: case-insensitive substring match of entity or core term.
  TYPE 2 — LEXICAL VARIANT: morphological or formatting variant.
  TYPE 3 — BIOCHEMICAL SYNONYM: different terms for the same biological entity.
  TYPE 4 — REACTION EQUIVALENCE: catalytic reaction described with different syntax.
  TYPE 5 — NOT A MATCH (superclass-only, different family member, opposite direction, related but distinct).

Let N_matched = count of reference entities matched via Types 1-4.

STEP 3: Count contradictions.

A contradiction is a prediction claim mutually exclusive with the reference (e.g., reference "kinase" vs prediction "phosphatase"). Reference silent or general with compatible specifics from prediction is NOT a contradiction.

Let N_contradicted = count of distinct contradictions.

STEP 4: Score four axes. Each axis score MUST be one of {0, 3, 5, 7, 10}.

AXIS 1 — RECALL: fraction of reference entities recovered via Types 1-4.
  Let r = N_matched / N_ref (score -1 if N_ref = 0).
    10: r = 1.00 — all reference entities recovered
     7: r ≥ 0.60 — majority recovered, few gaps
     5: r ≥ 0.40 — about half recovered
     3: r ≥ 0.15 — minority recovered
     0: r < 0.15 — essentially nothing recovered

AXIS 2 — PRECISION: prediction stays on-topic to reference biology.
  Let U = count of biological claims in PREDICTION unsupported by reference or reasonable inference.
    10: U = 0     7: U ≤ 2     5: U ≤ 5     3: U ≤ 10     0: U > 10 OR different protein

AXIS 3 — SPECIFICITY: granularity of biological content, INDEPENDENT of correctness.
    10: Names concrete entities — specific substrates, products, EC numbers, named domains, named localizations, named reactions, specific residues
     7: Names specific biology at comparable granularity but 1 step coarser
     5: Family or class level only
     3: Superfamily or very general biology
     0: Generic platitudes with no specific biology

AXIS 4 — PLAUSIBILITY: free of contradictions and fabrications.
    10: N_contradicted = 0, all claims plausible and grounded
     7: N_contradicted = 0, some speculative content
     5: N_contradicted = 1, otherwise plausible
     3: N_contradicted = 2
     0: N_contradicted >= 3 OR describes different protein entirely

STEP 5: Final score.
FINAL_SCORE = round((Axis1 + Axis2 + Axis3 + Axis4) / N_active_axes), excluding any axis scored -1.

---
OUTPUT FORMAT — return EXACTLY these 8 lines, no preamble, no commentary:

N_ref: <integer>
N_matched: <integer>
Match_types: <comma-separated type integers per matched entity, e.g., "1,3,2,4">
N_contradicted: <integer>
N_unsupported: <integer U from Axis 2>
Axis_scores: <recall>,<precision>,<specificity>,<plausibility>
Critique: <one sentence: what is the key mismatch, synonym match, or strength>
Final_score: <integer 0-10>
"""

STRICT_REMINDER = (
    "\n\nREMINDER: OUTPUT EXACTLY THE 8 LINES DEFINED ABOVE. "
    "NO MARKDOWN, NO CODE FENCES, NO BULLETS, NO PREAMBLE. "
    "Each line must begin with the exact label shown (e.g., 'N_ref:')."
)


def build_user_prompt(sample, strict=False):
    prompt = USER_PROMPT_TEMPLATE
    for key in ("instruction", "description", "metadata", "prediction"):
        prompt = prompt.replace("{" + key + "}", (sample.get(key) or "").strip())
    if strict:
        prompt += STRICT_REMINDER
    return prompt


def parse_judge_output(text):
    if not text:
        raise ValueError("empty response")
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    fields = {}
    for ln in lines:
        m = re.match(r"^\s*([A-Za-z_]+)\s*:\s*(.*)$", ln)
        if not m:
            continue
        key = m.group(1).strip()
        val = m.group(2).strip()
        if key not in fields:
            fields[key] = val
    required = ["N_ref", "N_matched", "Match_types", "N_contradicted",
                "N_unsupported", "Axis_scores", "Critique", "Final_score"]
    missing = [k for k in required if k not in fields]
    if missing:
        raise ValueError(f"missing fields: {missing}")

    def as_int(s, name):
        m = re.search(r"-?\d+", s)
        if not m:
            raise ValueError(f"{name} not an integer: {s!r}")
        return int(m.group(0))

    n_ref = as_int(fields["N_ref"], "N_ref")
    n_matched = as_int(fields["N_matched"], "N_matched")
    n_contradicted = as_int(fields["N_contradicted"], "N_contradicted")
    n_unsupported = as_int(fields["N_unsupported"], "N_unsupported")
    for name, v in [("N_ref", n_ref), ("N_matched", n_matched),
                    ("N_contradicted", n_contradicted), ("N_unsupported", n_unsupported)]:
        if v < 0:
            raise ValueError(f"{name} must be >= 0, got {v}")

    match_types_raw = fields["Match_types"]
    match_types = []
    if match_types_raw and match_types_raw.lower() not in {"none", "n/a", "-"}:
        for tok in re.split(r"[,\s]+", match_types_raw):
            if not tok:
                continue
            mi = re.search(r"\d+", tok)
            if not mi:
                continue
            t = int(mi.group(0))
            if t == 5:
                continue  # type 5 means "not a match" per rubric; skip
            if t not in {1, 2, 3, 4}:
                raise ValueError(f"invalid match type {t}")
            match_types.append(t)

    axis_tokens = [t.strip() for t in re.split(r"[,\s]+", fields["Axis_scores"]) if t.strip()]
    if len(axis_tokens) != 4:
        raise ValueError(f"Axis_scores needs 4 values, got {len(axis_tokens)}")
    axis_scores = []
    for tok in axis_tokens:
        m = re.search(r"-?\d+", tok)
        if not m:
            raise ValueError(f"axis token not numeric: {tok!r}")
        v = int(m.group(0))
        if v not in VALID_AXIS_SCORES:
            raise ValueError(f"axis score {v} not in {sorted(VALID_AXIS_SCORES)}")
        axis_scores.append(v)

    final_score = as_int(fields["Final_score"], "Final_score")
    if not (0 <= final_score <= 10):
        raise ValueError(f"Final_score out of range: {final_score}")

    return {
        "n_ref": n_ref,
        "n_matched": n_matched,
        "match_types": match_types,
        "n_contradicted": n_contradicted,
        "n_unsupported": n_unsupported,
        "axis_scores": axis_scores,
        "critique": fields["Critique"],
        "final_score": final_score,
    }


def load_dataset_instructions(dataset_dir, dataset):
    path = Path(dataset_dir) / f"{dataset}.json"
    with open(path) as f:
        data = json.load(f)
    return [x for x in data if x.get("split") == "test"]


def load_predictions(preds_dir, dataset):
    path = Path(preds_dir) / f"{dataset}_predictions.json"
    with open(path) as f:
        return json.load(f)


def build_samples(preds, test_entries):
    n = min(len(preds), len(test_entries))
    samples = []
    for i in range(n):
        p = preds[i]
        t = test_entries[i]
        samples.append({
            "sample_index": i,
            "instruction": t.get("instruction", ""),
            "sequence": t.get("sequence", ""),
            "description": p.get("label", t.get("description", "")),
            "metadata": p.get("metadata", t.get("metadata", "")),
            "prediction": p.get("prediction", ""),
        })
    return samples


def select_samples(samples, num_samples, seed):
    if num_samples <= 0 or num_samples >= len(samples):
        return samples
    rng = random.Random(seed)
    idxs = sorted(rng.sample(range(len(samples)), num_samples))
    return [samples[i] for i in idxs]


def read_done_indices(jsonl_path):
    done = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "sample_index" in rec:
                    done.add(rec["sample_index"])
            except json.JSONDecodeError:
                continue
    return done


# === ANTHROPIC API CALL (replaces OpenAI version) ===
async def call_judge(client, model, system_prompt, user_prompt, max_tokens, temperature):
    resp = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,                            # Anthropic: system as separate kwarg
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = resp.content[0].text if resp.content else ""
    usage = resp.usage
    return text, {
        "prompt_tokens": getattr(usage, "input_tokens", 0),       # Anthropic naming
        "completion_tokens": getattr(usage, "output_tokens", 0),
    }


async def score_sample(client, semaphore, model, sample, max_tokens, temperature, max_retries=3):
    async with semaphore:
        last_error = None
        raw_text = ""
        cumulative_tokens = {"prompt_tokens": 0, "completion_tokens": 0}

        for parse_attempt in range(2):
            user_prompt = build_user_prompt(sample, strict=(parse_attempt > 0))
            for retry in range(max_retries):
                try:
                    text, tokens = await call_judge(
                        client, model, SYSTEM_PROMPT, user_prompt, max_tokens, temperature
                    )
                    raw_text = text
                    cumulative_tokens["prompt_tokens"] += tokens["prompt_tokens"]
                    cumulative_tokens["completion_tokens"] += tokens["completion_tokens"]
                    try:
                        parsed = parse_judge_output(text)
                        return parsed, text, cumulative_tokens, None
                    except ValueError as pe:
                        last_error = f"parse error: {pe}"
                        break
                except (RateLimitError, APITimeoutError, APIError) as e:
                    last_error = f"{type(e).__name__}: {e}"
                    await asyncio.sleep(2 ** retry)
                except Exception as e:
                    last_error = f"{type(e).__name__}: {e}"
                    await asyncio.sleep(2 ** retry)
            else:
                return None, raw_text, cumulative_tokens, last_error
        return None, raw_text, cumulative_tokens, last_error


def compute_cost(tokens, input_price, output_price):
    return (tokens.get("prompt_tokens", 0) * input_price +
            tokens.get("completion_tokens", 0) * output_price) / 1_000_000.0


def summarize_dataset(records, cost_usd):
    n_scored = len(records)
    if n_scored == 0:
        return {"n_scored": 0, "cost_usd": cost_usd}
    axes = list(zip(*[r["axis_scores"] for r in records]))
    def mean_active(vals):
        v = [x for x in vals if x != -1]
        return (sum(v) / len(v)) if v else float("nan")
    finals = [r["final_score"] for r in records]
    mt_hist = {"1": 0, "2": 0, "3": 0, "4": 0}
    for r in records:
        for t in r["match_types"]:
            key = str(t)
            if key in mt_hist:
                mt_hist[key] += 1
    n_refs = [r["n_ref"] for r in records]
    n_matches = [r["n_matched"] for r in records]
    n_contras = [r["n_contradicted"] for r in records]
    return {
        "n_scored": n_scored,
        "cost_usd": round(cost_usd, 6),
        "mean_recall": round(mean_active(axes[0]), 4),
        "mean_precision": round(mean_active(axes[1]), 4),
        "mean_specificity": round(mean_active(axes[2]), 4),
        "mean_plausibility": round(mean_active(axes[3]), 4),
        "mean_final_score": round(statistics.mean(finals), 4),
        "std_final_score": round(statistics.pstdev(finals), 4) if len(finals) > 1 else 0.0,
        "match_type_histogram": mt_hist,
        "n_ref_stats": {"mean": round(statistics.mean(n_refs), 4), "median": statistics.median(n_refs)},
        "n_matched_stats": {"mean": round(statistics.mean(n_matches), 4), "median": statistics.median(n_matches)},
        "n_contradicted_stats": {"mean": round(statistics.mean(n_contras), 4), "median": statistics.median(n_contras)},
    }


async def score_dataset(client, args, dataset, output_dir):
    jsonl_path = output_dir / f"{dataset}_judge.jsonl"
    summary_path = output_dir / f"{dataset}_summary.json"
    test_entries = load_dataset_instructions(args.dataset_dir, dataset)
    preds = load_predictions(args.predictions_dir, dataset)
    samples = build_samples(preds, test_entries)
    chosen = select_samples(samples, args.num_samples, args.seed)

    if args.dry_run:
        print(f"\n=== DRY RUN: {dataset} ({len(chosen)} samples; showing up to 2) ===")
        for s in chosen[:2]:
            print(f"\n--- sample_index={s['sample_index']} ---")
            print(build_user_prompt(s))
        return None

    done = read_done_indices(jsonl_path) if args.resume else set()
    pending = [s for s in chosen if s["sample_index"] not in done]
    print(f"[{dataset}] chosen={len(chosen)} already_done={len(done)} pending={len(pending)}")

    semaphore = asyncio.Semaphore(args.max_concurrent)
    async def _worker(sample):
        parsed, raw, tokens, err = await score_sample(
            client, semaphore, args.judge_model, sample, args.max_tokens, args.temperature)
        return sample, parsed, raw, tokens, err

    n_failed_new = 0
    n_scored_new = 0
    cost_new = 0.0
    f = open(jsonl_path, "a", buffering=1)
    try:
        tasks = [asyncio.create_task(_worker(s)) for s in pending]
        pbar = tqdm(total=len(tasks), desc=f"{dataset}", unit="samp")
        processed = 0
        for coro in asyncio.as_completed(tasks):
            sample, parsed, raw, tokens, err = await coro
            cost = compute_cost(tokens, args.input_price, args.output_price)
            cost_new += cost
            record = {
                "sample_index": sample["sample_index"],
                "dataset": dataset,
                "timestamp": time.time(),
                "judge_model": args.judge_model,
                "prompt_tokens": tokens.get("prompt_tokens", 0),
                "completion_tokens": tokens.get("completion_tokens", 0),
                "cost_usd": round(cost, 8),
                "raw_response": raw,
                "error": err,
            }
            if parsed is not None:
                record.update(parsed)
                n_scored_new += 1
            else:
                n_failed_new += 1
            f.write(json.dumps(record) + "\n")
            f.flush()
            processed += 1
            pbar.update(1)
            pbar.set_postfix({"$": f"{cost_new:.3f}", "fail": n_failed_new})
            if processed % 50 == 0:
                print(f"[{dataset}] processed={processed} cost_this_run=${cost_new:.4f} failed={n_failed_new}")
        pbar.close()
    finally:
        f.close()

    all_records = []
    total_cost = 0.0
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            total_cost += rec.get("cost_usd", 0.0)
            if "axis_scores" in rec and "final_score" in rec:
                all_records.append(rec)
    total_attempts = 0
    with open(jsonl_path) as fh:
        for line in fh:
            if line.strip():
                total_attempts += 1
    n_failed_total = total_attempts - len(all_records)

    summary = summarize_dataset(all_records, total_cost)
    summary["n_failed"] = n_failed_total
    summary["dataset"] = dataset
    summary["judge_model"] = args.judge_model
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[{dataset}] done. n_scored={len(all_records)} n_failed={n_failed_total} total_cost=${total_cost:.4f}")
    return summary


async def main_async(args):
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.predictions_dir) / "judge_scores"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        client = None
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()    # Anthropic env var
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY is not set", file=sys.stderr)
            sys.exit(2)
        client = AsyncAnthropic(api_key=api_key)

    datasets = args.datasets.split(",") if args.datasets else DEFAULT_DATASETS
    datasets = [d.strip() for d in datasets if d.strip()]

    per_dataset_summaries = {}
    for ds in datasets:
        try:
            summary = await score_dataset(client, args, ds, output_dir)
        except FileNotFoundError as e:
            print(f"[{ds}] SKIPPED: {e}", file=sys.stderr)
            continue
        if summary is not None:
            per_dataset_summaries[ds] = summary

    if args.dry_run:
        print("\nDry run complete — no API calls made.")
        return

    def macro(key):
        vals = [s[key] for s in per_dataset_summaries.values() if key in s and isinstance(s[key], (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else float("nan")

    overall = {
        "judge_model": args.judge_model,
        "datasets": list(per_dataset_summaries.keys()),
        "macro_mean_recall": macro("mean_recall"),
        "macro_mean_precision": macro("mean_precision"),
        "macro_mean_specificity": macro("mean_specificity"),
        "macro_mean_plausibility": macro("mean_plausibility"),
        "macro_mean_final_score": macro("mean_final_score"),
        "total_cost_usd": round(sum(s.get("cost_usd", 0.0) for s in per_dataset_summaries.values()), 4),
        "total_scored": sum(s.get("n_scored", 0) for s in per_dataset_summaries.values()),
        "total_failed": sum(s.get("n_failed", 0) for s in per_dataset_summaries.values()),
        "per_dataset": per_dataset_summaries,
    }
    with open(output_dir / "overall_summary.json", "w") as fh:
        json.dump(overall, fh, indent=2)
    print("\n==== OVERALL ====")
    for k in ["macro_mean_recall", "macro_mean_precision", "macro_mean_specificity",
              "macro_mean_plausibility", "macro_mean_final_score",
              "total_cost_usd", "total_scored", "total_failed"]:
        print(f"  {k}: {overall[k]}")


def build_argparser():
    p = argparse.ArgumentParser(description="LLM-as-Judge scoring (Anthropic version) for SAPER benchmark predictions.")
    p.add_argument("--predictions_dir", required=True, type=str)
    p.add_argument("--dataset_dir", type=str,
                   default=os.environ.get("SAPER_DATA_ROOT",
                                          os.path.expanduser("~/work/SAPER/dataset")))
    p.add_argument("--output_dir", type=str, default=None,
                   help="Defaults to <predictions_dir>/judge_scores")
    p.add_argument("--num_samples", type=int, default=500, help="Samples per dataset (0 = all).")
    p.add_argument("--max_concurrent", type=int, default=3)
    p.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    p.add_argument("--judge_model", type=str, default="claude-sonnet-4-6")   # Anthropic default
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--input_price", type=float, default=3.00,                # Sonnet 4.6 pricing
                   help="USD per 1M input tokens (Sonnet 4.6 = 3.00)")
    p.add_argument("--output_price", type=float, default=15.00,
                   help="USD per 1M output tokens (Sonnet 4.6 = 15.00)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=1024)
    return p


def main():
    args = build_argparser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
