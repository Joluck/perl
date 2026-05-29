SYSTEM_PROMPT = """
You are a helpful AI Assistant that provides well-reasoned and detailed responses.
You first think about the reasoning process as an internal monologue and then provide the user with the answer.
Respond in the following format: <think>\n...\n</think>\n, then answer.
"""

def make_conversation(example):
    # 尝试各种常见的字段名组合
    problem = None
    answer = ""

    if "problem" in example:
        problem = example["problem"]
        answer = example.get("answer", "")
    elif "prompt" in example:
        prompt_val = example["prompt"]
        if isinstance(prompt_val, list):
            # prompt 已经是 conversation messages 格式（如 zhuzilin/dapo-math-17k）
            # 为了避免 GRPOTrainer 自动应用 chat_template 产生特殊 token，
            # 把 messages 转成纯文本字符串
            text = "\n\n".join(
                m.get("content", "") for m in prompt_val if m.get("content")
            )
            return {
                "prompt": text,
                "solution": example.get("label", example.get("answer", example.get("solution", "")))
            }
        problem = prompt_val
        answer = example.get("solution", example.get("answer", example.get("label", "")))
    elif "question" in example:
        problem = example["question"]
        answer = example.get("answer", example.get("solution", example.get("label", "")))
    elif "input" in example:
        problem = example["input"]
        answer = example.get("output", example.get("answer", example.get("solution", example.get("label", ""))))
    elif "text" in example:
        problem = example["text"]
        answer = example.get("label", example.get("answer", example.get("solution", "")))
    elif "content" in example:
        problem = example["content"]
        answer = example.get("label", example.get("answer", example.get("solution", "")))

    if problem is None:
        raise ValueError(
            f"Unknown dataset format. Available keys: {list(example.keys())}. "
            f"Expected one of: problem/answer, prompt/solution, question/answer, input/output, text/label, content/label"
        )

    prompt = [
        # {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem}
    ]
    return {"prompt": prompt, "solution": answer}