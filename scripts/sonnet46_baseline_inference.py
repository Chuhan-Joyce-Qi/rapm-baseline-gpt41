#!/usr/bin/env python3
"""
Sonnet 4.6 inference on the original RAPM prompt files.
Pure model-swap ablation vs the GPT-4.1 baseline.
"""
import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path

from anthropic import AsyncAnthropic, APIError, APITimeoutError, RateLimitError
from tqdm.asyncio import tqdm_asyncio

TASKS = [
    "catalytic_activity_OOD",
    "domain_motif_OOD",
    "general_function_OOD",
    "protein_function_OOD",
]

MODEL = "claude-sonnet-4-6"
TEMPERATURE = 0.7
MAX_TOKENS = 4096
CONCURRENCY = 30
INPUT_PRICE = 3.00 / 1_000_000
OUTPUT_PRICE = 15.00 / 1_000_000


def parse_description(raw):
    if not raw:
        return ""
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.MULTILINE)
    s = re.sub(r"\s*```\s*$", "", s, flags=re.MULTILINE)
    s = s.strip()
    try:
        d = json.loads(s)
        if isinstance(d, dict) and "description" in d:
            return d["description"]
    except json.JSONDecodeError:
        pass
    m = re.search(r'"description"\s*:\s*"((?:[^"\\]|\\.)*)"', s)
    if m:
        return bytes(m.group(1), "utf-8").decode("unicode_escape", errors="replace")
    return s


async def call_sonnet(client, prompt, max_retries=5):
    delay = 2.0
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = await client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
            )
            text = ""
            for block in (resp.content or []):
                if getattr(block, "type", None) == "text":
                    text = block.text
                    break
            usage = resp.usage
            return text, {
                "input_tokens": getattr(usage, "input_tokens", 0),
                "output_tokens": getattr(usage, "output_tokens", 0),
            }
        except (RateLimitError, APIError, APITimeoutError) as e:
            last_err = e
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)
        except Exception as e:
            last_err = e
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)
    return f"[ERROR after {max_retries} retries: {last_err}]", {"input_tokens": 0, "output_tokens": 0}


async def run_task(task, concurrency, limit=None):
    in_path = Path(f"{task}_RAP_Top_10_normal.json")
    out_path = Path(f"{task}_sonnet46_results.json")
    ckpt_path = Path(f"{task}_sonnet46_results.ckpt.json")

    with open(in_path) as f:
        prompts = json.load(f)
    if limit is not None:
        prompts = prompts[:limit]
    n = len(prompts)

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

    client = AsyncAnthropic()
    sem = asyncio.Semaphore(concurrency)
    total_in = sum(c.get("input_tokens", 0) for c in completed.values())
    total_out = sum(c.get("output_tokens", 0) for c in completed.values())
    last_save = time.time()

    async def worker(idx):
        async with sem:
            text, usage = await call_sonnet(client, prompts[idx]["RAG_prompt"])
            return idx, text, usage

    coros = [worker(i) for i in todo_indices]
    for fut in tqdm_asyncio.as_completed(coros, total=len(coros), desc=task):
        idx, raw, usage = await fut
        prediction = parse_description(raw)
        completed[idx] = {
            "idx": idx,
            "prediction": prediction,
            "raw_response": raw,
            "label": prompts[idx].get("labels", ""),
            "meta_label": prompts[idx].get("meta_label", ""),
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
        }
        total_in += usage["input_tokens"]
        total_out += usage["output_tokens"]
        if time.time() - last_save > 30:
            with open(ckpt_path, "w") as f:
                json.dump(list(completed.values()), f)
            last_save = time.time()

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
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = ap.parse_args()

    tasks = TASKS if args.task == "all" else [args.task]
    grand_in = grand_out = 0
    for t in tasks:
        n, ti, to = await run_task(t, args.concurrency, args.limit)
        grand_in += ti
        grand_out += to

    cost = grand_in * INPUT_PRICE + grand_out * OUTPUT_PRICE
    print(f"\nGRAND TOTAL: input={grand_in/1e6:.2f}M  output={grand_out/1e6:.2f}M  cost=${cost:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
