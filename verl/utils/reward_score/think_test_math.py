import re
from math_verify import verify, parse

def format_verify_and_extract(solution_str: str) -> tuple[float, str]:
    """
    要求：
     1. 以 <think> 开头；
     2. </think> 和 <answer> 之间只能有空白字符（或直接相连）；
     3. </answer> 必须是最后一个字符；
     4. 不再强制任何换行或其它空白。
    """
    pattern = r"(?s)^<think>(.*?)</think>\s*<answer>(.*?)</answer>$"
    m = re.match(pattern,solution_str)
    if not m:
        return 0.0, ""
    # m.group(1) <think>…</think> 之间的内容
    # m.group(2) <answer>…</answer> 之间的内容
    answer = m.group(2).strip()
    return 1.0, answer

def compute_score(solution_str, ground_truth):
    acc = 0
    pred = ""
    format_verify = 0.0
    try:
        format_verify,answer_str = format_verify_and_extract(solution_str)
        pred=parse(answer_str)
        acc=int(verify(parse(ground_truth), pred))
    except Exception as e:
        print(e)
    
    reward = 1.0 if acc else -1.0

    return {
        "score": reward,
        "acc": acc,
        "answer": answer_str,
        "pred": str(pred),
        "format_verify": format_verify
    }