"""
GPT-4.1 inference for RAPM baseline replication.
- Reads {task}_RAP_Top_10_normal.json
- Writes {task}_gpt41_results.json with key "prediction" (matches benchmarks repo)
- Resumable: saves checkpoint every 50 samples, skips already-completed indices
- Async with bounded concurrency
"""
import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import tiktoken
from openai import AsyncOpenAI, RateLimitError, APIError, APITimeoutError
from tqdm.asyncio import tqdm_asyncio

TASKS = [
    "catalytic_activity_OOD",
    "domain_motif_OOD",
    "general_function_OOD",
    "protein_function_OOD",
]

MODEL = "gpt-4.1"
INPUT_PRICE  = 2.00 / 1_000_000   # $/token
OUTPUT_PRICE = 8.00 / 1_000_000

enc = tiktoken.get_encoding("cl100k_base")


async def call_gpt41(client: AsyncOpenAI, prompt: str, idx: int,
                     max_retries: int = 6) -> tuple[int, str, dict]:
    """One GPT-4.1 call with exponential backoff."""
    delay = 2.0
    last_err = None
    for attempt in range(max_retries):
        try:
            r = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.0,
                timeout=120,
            )
            usage = {
                "input_tokens": r.usage.prompt_tokens,
                "output_tokens": r.usage.completion_tokens,
            }
            return idx, r.choices[0].message.content or "", usage
        except (RateLimitError, APIError, APITimeoutError) as e:
            last_err = e
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)
        except Exception as e:
            last_err = e
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)
    return idx, f"[ERROR after {max_retries} retries: {last_err}]", \
           {"input_tokens": 0, "output_tokens": 0}


def parse_prediction(raw: str) -> str:
    """Extract the description string from the model's JSON answer."""
    raw = raw.strip()
    # Try direct JSON parse
    try:
        d = json.loads(raw)
        if isinstance(d, dict) and "description" in d:
            return d["description"]
    except json.JSONDecodeError:
        pass
    # Fallback: find description in raw text
    import re
    m = re.search(r'"description"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    if m:
        return bytes(m.group(1), "utf-8").decode("unicode_escape", errors="replace")
    return raw


async def run_task(task: str, limit: int | None, concurrency: int):
    in_path  = Path(f"{task}_RAP_Top_10_normal.json")
    out_path = Path(f"{task}_gpt41_results.json")
    ckpt_path = Path(f"{task}_gpt41_results.ckpt.json")

    with open(in_path) as f:
        prompts = json.load(f)
    if limit is not None:
        prompts = prompts[:limit]
    n = len(prompts)

    # Resume from checkpoint if exists
    completed = {}
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            completed = {item["idx"]: item for item in json.load(f)}
        print(f"  [{task}] Resuming: {len(completed)}/{n} already done")

    todo_indices = [i for i in range(n) if i not in completed]
    if not todo_indices:
        print(f"  [{task}] All {n} samples already processed.")
    else:
        print(f"  [{task}] Processing {len(todo_indices)} samples (concurrency={concurrency})")

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(concurrency)
    total_in = sum(c.get("input_tokens", 0) for c in completed.values())
    total_out = sum(c.get("output_tokens", 0) for c in completed.values())
    last_save = time.time()

    async def worker(idx: int):
        async with sem:
            return await call_gpt41(client, prompts[idx]["RAG_prompt"], idx)

    # Run with progress bar
    coros = [worker(i) for i in todo_indices]
    for fut in tqdm_asyncio.as_completed(coros, total=len(coros), desc=task):
        idx, raw, usage = await fut
        prediction = parse_prediction(raw)
        completed[idx] = {
            "idx": idx,
            "prediction": prediction,
            "raw_response": raw,
            "label": prompts[idx]["labels"],
            "meta_label": prompts[idx]["meta_label"],
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
        }
        total_in  += usage["input_tokens"]
        total_out += usage["output_tokens"]

        # Periodic checkpoint
        if time.time() - last_save > 30:  # save every 30s
            with open(ckpt_path, "w") as f:
                json.dump(list(completed.values()), f)
            last_save = time.time()

    # Final write
    sorted_results = [completed[i] for i in range(n)]
    with open(out_path, "w") as f:
        json.dump(sorted_results, f, indent=2)
    if ckpt_path.exists():
        ckpt_path.unlink()

    cost = total_in * INPUT_PRICE + total_out * OUTPUT_PRICE
    print(f"  [{task}] Done. Input={total_in/1e6:.2f}M  Output={total_out/1e6:.2f}M  Cost=${cost:.2f}")
    return n, total_in, total_out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASKS + ["all"], default="all")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only first N samples (for testing)")
    ap.add_argument("--concurrency", type=int, default=50,
                    help="Concurrent API calls (raise if Tier 3+)")
    args = ap.parse_args()

    tasks_to_run = TASKS if args.task == "all" else [args.task]
    grand_in = grand_out = 0
    for t in tasks_to_run:
        n, ti, to = await run_task(t, args.limit, args.concurrency)
        grand_in  += ti
        grand_out += to

    cost = grand_in * INPUT_PRICE + grand_out * OUTPUT_PRICE
    print(f"\nGRAND TOTAL: input={grand_in/1e6:.2f}M  output={grand_out/1e6:.2f}M  cost=${cost:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
