#!/usr/bin/env python3
"""
Rebuild RAPM prompts with 3 few-shot examples instead of 10.

- Reads existing *_RAP_Top_10_normal.json (which has 10 retrieved + 10 few-shot)
- Keeps the 10 retrieved annotations unchanged
- Subsamples 3 random few-shot examples from the existing 10, deterministic seed=42 per sample
- Reformats the prompt in the same RAPM style
- Outputs: {task}_RAP_Top_10_3fewshot.json
"""
import ast
import json
import random
import re
from pathlib import Path

TASKS = [
    "catalytic_activity_OOD",
    "domain_motif_OOD",
    "general_function_OOD",
    "protein_function_OOD",
]
SEED = 42
NUM_FEWSHOT = 3


def parse_prompt_components(rag_prompt_str):
    """Extract: instruction, sequence, retrieved list, few-shot list, suffix from old prompt."""
    # Instruction
    m_inst = re.search(r"Instruction: (.*?)\nProtein sequence:", rag_prompt_str, re.DOTALL)
    instruction = m_inst.group(1).strip() if m_inst else ""

    # Sequence
    m_seq = re.search(r"Protein sequence: (.*?)\nRetrieved proteins annotations", rag_prompt_str, re.DOTALL)
    sequence = m_seq.group(1).strip() if m_seq else ""

    # Retrieved annotations (Python-stringified list of dicts)
    m_ret = re.search(
        r"Retrieved proteins annotations by weighted Faiss/MMSeqs2:\s*(\[.*?\])\s*\nHere are some example",
        rag_prompt_str, re.DOTALL,
    )
    retrieved_str = m_ret.group(1) if m_ret else "[]"
    try:
        retrieved = ast.literal_eval(retrieved_str)
    except Exception:
        retrieved = []

    # Few-shot examples
    m_fs = re.search(
        r"Here are some example input-output pairs for this task:\s*(\[.*?\])\s*\nBased on the instruction",
        rag_prompt_str, re.DOTALL,
    )
    fewshot_str = m_fs.group(1) if m_fs else "[]"
    try:
        fewshot = ast.literal_eval(fewshot_str)
    except Exception:
        fewshot = []

    return instruction, sequence, retrieved, fewshot


def build_prompt(instruction, sequence, retrieved, fewshot):
    return (
        f"You are given a protein sequence and a list of related proteins retrieved from a database.\n"
        f"Instruction: {instruction}\n"
        f"Protein sequence: {sequence}\n"
        f"Retrieved proteins annotations by weighted Faiss/MMSeqs2: {retrieved}\n"
        f"Here are some example input-output pairs for this task:\n"
        f"{fewshot}\n"
        "Based on the instruction, the protein sequence, the retrieved information, and the examples, "
        "output ONLY the functional description of this protein in the following JSON format:\n"
        '{"description": "..."}'
        "\nDo not output any other text or explanation. Only output the JSON answer."
    )


def main():
    for task in TASKS:
        in_path = Path(f"{task}_RAP_Top_10_normal.json")
        out_path = Path(f"{task}_RAP_Top_10_3fewshot.json")

        if not in_path.exists():
            print(f"[{task}] missing {in_path} -- skip")
            continue

        with open(in_path) as f:
            items = json.load(f)
        print(f"[{task}] loaded {len(items)} prompts")

        out = []
        for idx, item in enumerate(items):
            inst, seq, retrieved, fewshot = parse_prompt_components(item.get("RAG_prompt", ""))

            # Deterministic random sample of 3 from the 10 few-shot, seed unique per (task, sample idx)
            local_rng = random.Random(SEED * 100000 + idx)
            if len(fewshot) > NUM_FEWSHOT:
                fewshot_3 = local_rng.sample(fewshot, NUM_FEWSHOT)
            else:
                fewshot_3 = fewshot

            new_prompt = build_prompt(inst, seq, retrieved, fewshot_3)

            out.append({
                "instructions": item.get("instructions", inst),
                "sequence": item.get("sequence", seq),
                "labels": item.get("labels", ""),
                "meta_label": item.get("meta_label", ""),
                "RAG_prompt": new_prompt,
                "context_mode": item.get("context_mode", "normal_3fewshot"),
            })

        with open(out_path, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[{task}] wrote {len(out)} prompts to {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
