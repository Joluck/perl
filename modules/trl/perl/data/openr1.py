import re
from contextlib import contextmanager

from typing import Optional
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from datasets import load_dataset

from .system_prompts import make_conversation

def accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution, strict=True):
        gold_parsed = parse(sol)
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(units=True),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Compute binary rewards if verifiable, `None` otherwise to skip this example
            reward = float(verify(gold_parsed, answer_parsed))
        else:
            # If the gold solution is not parseable, we assign `None` to skip this example
            reward = float(content.strip().lower() == sol.strip().lower())
        rewards.append(reward)

    return rewards

def format_reward(completions, **kwargs):
    pattern = r"</think>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, content) for content in completion_contents]
    
    # output if mismatch (truncated)
    for content, match in zip(completion_contents, matches, strict=True):
        if not match:
            truncated = content[:200] + "..." if len(content) > 200 else content
            print(f"Mismatch: {truncated}")
            print("-" * 100)
    return [1.0 if match else 0.0 for match in matches]

def load_openr1_dataset(dataset_name_or_path: str, example_numbers: int = None):
    dataset = load_dataset(
        dataset_name_or_path, split="train"
    )

    if "messages" in dataset.column_names:
        # 数据集已经是 conversation 格式（如 zhuzilin/dapo-math-17k）
        # 为了避免 GRPOTrainer 自动应用 chat_template 产生特殊 token，
        # 把 messages 列表转成纯文本字符串
        def _convert(example):
            msgs = example["messages"]
            if isinstance(msgs, list):
                text = "\n\n".join(
                    m.get("content", "") for m in msgs if m.get("content")
                )
            else:
                text = str(msgs)
            result = {"prompt": text}
            for key in ["answer", "solution", "ground_truth", "label"]:
                if key in example:
                    result["solution"] = example[key]
                    break
            else:
                result["solution"] = ""
            return result
        dataset = dataset.map(_convert, remove_columns=dataset.column_names, load_from_cache_file=False)
    else:
        dataset = dataset.map(make_conversation, load_from_cache_file=False)

    if example_numbers is not None and len(dataset) > example_numbers:
        dataset = dataset.select(range(example_numbers))

    return {
        "train_dataset": dataset,
        "test_dataset": None,
        "reward_functions": [accuracy_reward],
        "reward_weights": [1.0],
    }