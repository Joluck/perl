#!/usr/bin/env python
"""Download common reasoning benchmarks and convert to nanoeval jsonl format."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset


def save_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} records to {path}")


# OPENR1_MATH_PROMPT = """Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.

# {problem}

# Remember to put your answer on its own line after "Answer:".""".strip()
OPENR1_MATH_PROMPT = """Solve the following math problem step by step. The last line of your response should be of the form Answer: \\boxed{{$Answer}} where $Answer is the answer to the problem.

{problem}

Remember to put your answer on its own line after "Answer:".""".strip()

def prepare_aime(year: int, output_dir: Path) -> None:
    """AIME from HuggingFace datasets."""
    # Try primary source first
    ds_name = f"HuggingFaceH4/aime_{year}"
    try:
        ds = load_dataset(ds_name, split="train")
    except Exception:
        # Fallback to alternative source
        if year == 2025:
            ds_name = "MathArena/aime_2025"
            try:
                ds = load_dataset(ds_name, split="train")
            except Exception as e:
                print(f"[WARN] Failed to load {ds_name}: {e}")
                return
        else:
            print(f"[WARN] Failed to load {ds_name}")
            return

    records = []
    for i, item in enumerate(ds):
        problem = str(item.get("problem", ""))
        records.append({
            "id": str(i),
            "prompt": OPENR1_MATH_PROMPT.format(problem=problem),
            "label": str(item.get("answer", "")),
        })
    save_jsonl(output_dir / f"aime{year}.jsonl", records)


def prepare_amc2023(output_dir: Path) -> None:
    """AMC 2023 from math-ai/amc23."""
    ds_name = "math-ai/amc23"
    try:
        ds = load_dataset(ds_name, split="test")
    except Exception as e:
        print(f"[WARN] Failed to load {ds_name}: {e}")
        return

    records = []
    for i, item in enumerate(ds):
        problem = str(item.get("question", ""))
        records.append({
            "id": str(i),
            "prompt": OPENR1_MATH_PROMPT.format(problem=problem),
            "label": str(item.get("answer", "")),
        })
    save_jsonl(output_dir / "amc2023.jsonl", records)


def prepare_math500(output_dir: Path) -> None:
    try:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    except Exception as e:
        print(f"[WARN] Failed to load MATH-500: {e}")
        return

    records = []
    for i, item in enumerate(ds):
        problem = str(item.get("problem", ""))
        records.append({
            "id": str(i),
            "prompt": OPENR1_MATH_PROMPT.format(problem=problem),
            "label": str(item.get("answer", "")),
        })
    save_jsonl(output_dir / "math500.jsonl", records)


def prepare_hmmt2025(output_dir: Path) -> None:
    """HMMT 2025 from MathArena (feb + nov merged)."""
    records = []
    for ds_name in ["MathArena/hmmt_feb_2025", "MathArena/hmmt_nov_2025"]:
        try:
            ds = load_dataset(ds_name, split="train")
        except Exception as e:
            print(f"[WARN] Failed to load {ds_name}: {e}")
            continue
        for item in ds:
            problem = str(item.get("problem", ""))
            records.append({
                "id": str(len(records)),
                "prompt": OPENR1_MATH_PROMPT.format(problem=problem),
                "label": str(item.get("answer", "")),
            })
    if records:
        save_jsonl(output_dir / "hmmt2025.jsonl", records)


def prepare_minerva(output_dir: Path) -> None:
    """Minerva Math from math-ai/minervamath."""
    ds_name = "math-ai/minervamath"
    try:
        ds = load_dataset(ds_name, split="test")
    except Exception as e:
        print(f"[WARN] Failed to load {ds_name}: {e}")
        return

    records = []
    for i, item in enumerate(ds):
        question = str(item.get("question", ""))
        records.append({
            "id": str(i),
            "prompt": OPENR1_MATH_PROMPT.format(problem=question),
            "label": str(item.get("answer", "")),
        })
    save_jsonl(output_dir / "minerva.jsonl", records)


def prepare_gpqa_diamond(output_dir: Path) -> None:
    try:
        ds = load_dataset("google/gpqa", "gpqa_diamond", split="train", trust_remote_code=True)
    except Exception as e:
        print(f"[WARN] Failed to load gpqa_diamond: {e}")
        return

    records = []
    for i, item in enumerate(ds):
        question = str(item.get("Question", ""))
        records.append({
            "id": str(i),
            "prompt": OPENR1_MATH_PROMPT.format(problem=question),
            "label": str(item.get("Correct Answer", "")),
        })
    save_jsonl(output_dir / "gpqa_diamond.jsonl", records)


def prepare_ifeval(output_dir: Path) -> None:
    try:
        ds = load_dataset("google/ifeval", split="train")
    except Exception as e:
        print(f"[WARN] Failed to load ifeval: {e}")
        return

    records = []
    for i, item in enumerate(ds):
        records.append({
            "id": str(i),
            "prompt": str(item.get("prompt", "")),
            "label": "",  # IFEval uses instruction-based evaluation
        })
    save_jsonl(output_dir / "ifeval.jsonl", records)


def main() -> None:
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/nano_eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set HF mirror if needed
    # os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    prepare_aime(2024, output_dir)
    prepare_aime(2025, output_dir)
    prepare_amc2023(output_dir)
    prepare_hmmt2025(output_dir)
    prepare_minerva(output_dir)
    prepare_math500(output_dir)
    prepare_gpqa_diamond(output_dir)
    prepare_ifeval(output_dir)

    print(f"\nDone. Task files are in {output_dir}")


if __name__ == "__main__":
    main()
