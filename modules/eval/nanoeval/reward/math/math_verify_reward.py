from typing import Tuple

import re
from fractions import Fraction

try:
    from math_verify import parse, verify
except Exception:
    parse = None
    verify = None

def extract_answer(text: str) -> str:
    """Extract answer from model response using regex (boxed or last value)."""
    if not text:
        return ""

    # NOTE:
    # - 一些 jsonl 里可能错误地写成 "\boxed{...}"（单反斜杠）。
    #   json.loads 会把 "\b" 解析成退格符 \x08，导致后续正则匹配不到。
    #   这里把退格符还原为字面量 "\b"（两字符：反斜杠 + b）。
    if "\x08" in text:
        text = text.replace("\x08", "\\b")

    # 1) 优先提取 \boxed{...}
    # 不能用简单正则去找 "第一个 }" 结束，因为 boxed 内容里常见嵌套花括号：
    #   \boxed{9.0 \times 10^{11}}
    # 这里用括号配对解析，确保提取完整 boxed 内容；若有多个，取最后一个。
    results = []
    for m in re.finditer(r"\\boxed\b", text):
        i = m.end()
        # 跳过 \boxed 后面的空白，找到第一个 '{'
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text) or text[i] != "{":
            continue

        i += 1  # skip '{'
        depth = 1
        start = i
        while i < len(text) and depth > 0:
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            i += 1

        if depth == 0:
            # i 已经指向匹配到的 '}' 之后
            results.append(text[start : i - 1].strip())

    if results:
        return results[-1]

    # 2) Fallback: extract "Answer: ..." or "Final Answer: ..."
    # Use a non-greedy match up to the end of line/paragraph.
    fallback_match = re.search(
        r"(?:\bfinal\s+answer|answer)\s*[:：]\s*(.+?)(?:\n|$)",
        text,
        flags=re.IGNORECASE,
    )
    if fallback_match:
        ans = fallback_match.group(1).strip()
        # 去掉 markdown 加粗符号 **
        ans = ans.strip("*")
        # 去掉模型从 prompt 抄来的提示后缀，如 (without quotes)
        ans = re.sub(r"\s*\(without[^)]*\)$", "", ans, flags=re.IGNORECASE)
        return ans

    # 3) 最终 fallback: 提取最后几行中的最后一个数字
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in reversed(lines[-5:]):
        line = line.rstrip(".。,，!！?？;；")
        nums = re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", line)
        if nums:
            return nums[-1].replace(",", "")

    return None

def grade_answer(solution_str: str, ground_truth: str) -> Tuple[float, float]:
    if parse is None or verify is None:
        pred_norm = _normalize_answer(solution_str)
        gold_norm = _normalize_answer(ground_truth)
        return (1.0, 1.0) if pred_norm == gold_norm else (0.0, 1.0)

    try: 
        ground_truth = parse(ground_truth)
        solution = parse(solution_str)
        if verify(ground_truth, solution):
            return 1.0, 1.0
        else:
            return 0.0, 1.0
    except Exception as e:
        print(f"Error: {e}")
        return 0.0, 0.0


def _normalize_answer(text: str) -> str:
    normalized = text.strip()
    if normalized.startswith("$") and normalized.endswith("$") and len(normalized) >= 2:
        normalized = normalized[1:-1].strip()
    normalized = normalized.replace(" ", "")
    try:
        return str(Fraction(normalized))
    except Exception:
        return normalized

def math_judge(
    response: str,
    label: str = "",
    **kwargs
) -> dict:
    raw_eval_res = response
    pred_ans = extract_answer(raw_eval_res)
    
    if not pred_ans:
        return {
            "pred": pred_ans,
            "pass": False
        }
    
    if pred_ans == label:
        return {
            "pred": pred_ans,
            "pass": True
        }

    if label in pred_ans:
        return {
            "pred": pred_ans,
            "pass": True
        }

    score, _ = grade_answer(f"${pred_ans}$", f"${label}$")
    return {
        "pred": pred_ans,
        "pass": True if score == 1.0 else False
    }