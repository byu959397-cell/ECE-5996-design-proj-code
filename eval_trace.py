from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from .data_trace import TASK_METRICS

def normalize_text(text):
    if text is None:
        return ""
    return " ".join(str(text).strip().split()).lower()


def parse_explicit_choice(text, valid_choices=("A", "B", "C")):
    if text is None:
        return None

    raw = str(text).strip()
    if not raw:
        return None

    choices = "".join(valid_choices)
    choices_lower = choices.lower()

    m = re.search(
        rf"(?:the\s+correct\s+answer|correct\s+answer|the\s+answer|answer|答案|选项|choice|option|label|result|态度|stance)"
        rf"\s*(?:is|是|为|=|:|：)?\s*"
        rf"([{choices}{choices_lower}])"
        rf"(?:\s|$|[\.\)\]）】、,，。:：;；])",
        raw,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()

    m = re.match(
        rf"^\s*[\(\[（【]?\s*([{choices}])\s*[\)\]）】]?"
        rf"(?:\s|$|[\.\、,，。:：;；])",
        raw,
    )
    if m:
        return m.group(1).upper()

    m = re.match(
        rf"^\s*[\(\[（【]?\s*([{choices_lower}])\s*[\)\]）】]?"
        rf"(?:$|[\.\、,，。:：;；])",
        raw,
    )
    if m:
        return m.group(1).upper()

    m = re.search(
        rf"(?:choose|select|selected|chooses|choosing|option|choice|选)\s*"
        rf"([{choices}{choices_lower}])"
        rf"(?:\s|$|[\.\)\]）】、,，。:：;；])",
        raw,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()

    return None


def parse_cstance_answer(text):
    label = parse_explicit_choice(text, valid_choices=("A", "B", "C"))
    if label is not None:
        return label

    if text is None:
        return None

    t = normalize_text(text)

    m = re.search(
        r"(?:答案|态度|选项|answer|label|stance)"
        r"\s*(?:是|为|=|:|：)?\s*"
        r"(支持|反对|中立)",
        t,
        flags=re.IGNORECASE,
    )
    if m:
        return {"支持": "A", "反对": "B", "中立": "C"}[m.group(1)]

    m = re.match(r"^\s*(支持|反对|中立)(?:\s|$|。|，|,|\.|！|!)", t)
    if m:
        return {"支持": "A", "反对": "B", "中立": "C"}[m.group(1)]

    found = set()

    if re.search(r"(反对|不支持|否定)", t):
        found.add("B")

    if re.search(r"(?<!不)支持|赞成|肯定", t):
        found.add("A")

    if re.search(r"(中立|无明显态度|无法判断)", t):
        found.add("C")

    if len(found) == 1:
        return next(iter(found))

    return None


def parse_fomc_answer(text):
    label = parse_explicit_choice(text, valid_choices=("A", "B", "C"))
    if label is not None:
        return label

    if text is None:
        return None

    t = normalize_text(text)
    found = set()

    if re.search(r"\bdovish\b", t, flags=re.IGNORECASE):
        found.add("A")

    if re.search(r"\bhawkish\b", t, flags=re.IGNORECASE):
        found.add("B")

    if re.search(r"\bneutral\b", t, flags=re.IGNORECASE):
        found.add("C")

    if len(found) == 1:
        return next(iter(found))

    return None


def parse_scienceqa_answer(text):
    return parse_explicit_choice(
        text,
        valid_choices=("A", "B", "C", "D", "E"),
    )


def parse_answer_for_task(task_name, text):
    if task_name == "C-STANCE":
        return parse_cstance_answer(text)

    if task_name == "FOMC":
        return parse_fomc_answer(text)

    if task_name == "ScienceQA":
        return parse_scienceqa_answer(text)

    return normalize_text(text)


MONTHS = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}


def parse_decimal(text):
    if text is None:
        return None

    s = str(text).strip().replace(",", "")

    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def extract_explicit_number(text):
    if text is None:
        return None

    raw = str(text).replace(",", "")

    m = re.search(
        r"(?:the\s+answer|answer|final\s+answer|result|答案)"
        r"\s*(?:is|是|为|=|:|：)?\s*"
        r"([-+]?\d+(?:\.\d+)?)",
        raw,
        flags=re.IGNORECASE,
    )

    if m:
        return parse_decimal(m.group(1))

    return None


def extract_leading_number(text):
    if text is None:
        return None

    raw = str(text).strip().replace(",", "")

    m = re.match(
        r"^\s*([-+]?\d+(?:\.\d+)?)(?:\s|$|[%\.\,\;\:\)\]）】])",
        raw,
    )

    if m:
        return parse_decimal(m.group(1))

    return None


def extract_last_number(text):
    if text is None:
        return None

    raw = str(text).replace(",", "")
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", raw)

    if not nums:
        return None

    return parse_decimal(nums[-1])


def extract_number_answer(text):
    num = extract_explicit_number(text)
    if num is not None:
        return num

    num = extract_leading_number(text)
    if num is not None:
        return num

    return extract_last_number(text)


def month_pattern():
    return (
        r"(?:january|february|march|april|may|june|july|august|"
        r"september|october|november|december)"
    )


def extract_explicit_month(text):
    if text is None:
        return None

    raw = str(text).strip()

    m = re.search(
        rf"(?:the\s+answer|answer|final\s+answer|result|答案)"
        rf"\s*(?:is|是|为|=|:|：)?\s*"
        rf"({month_pattern()})"
        rf"(?:\s|$|[\.\)\]）】、,，。:：;；])",
        raw,
        flags=re.IGNORECASE,
    )

    if m:
        return normalize_text(m.group(1))

    return None


def extract_leading_month(text):
    if text is None:
        return None

    raw = str(text).strip()

    m = re.match(
        rf"^\s*({month_pattern()})(?:\s|$|[\.\)\]）】、,，。:：;；])",
        raw,
        flags=re.IGNORECASE,
    )

    if m:
        return normalize_text(m.group(1))

    return None


def extract_last_month(text):
    if text is None:
        return None

    t = normalize_text(text)
    hits = []

    for month in MONTHS:
        for m in re.finditer(rf"\b{month}\b", t):
            hits.append((m.start(), month))

    if not hits:
        return None

    hits.sort(key=lambda x: x[0])
    return hits[-1][1]


def extract_month_answer(text):
    month = extract_explicit_month(text)
    if month is not None:
        return month

    month = extract_leading_month(text)
    if month is not None:
        return month

    return extract_last_month(text)


def clean_final_text_answer(text):
    if text is None:
        return ""

    raw = str(text).strip()

    m = re.search(
        r"(?:the\s+answer|answer|final\s+answer|result|答案)"
        r"\s*(?:is|是|为|=|:|：)?\s*"
        r"([a-zA-Z]+)",
        raw,
        flags=re.IGNORECASE,
    )
    if m:
        return normalize_text(m.group(1)).strip(".,;:!?")

    return normalize_text(raw).strip(".,;:!?")


def exact_match_score(preds, refs):
    correct = 0
    total = len(preds)

    for pred, ref in zip(preds, refs):
        ref_num = parse_decimal(ref)

        if ref_num is not None:
            pred_num = extract_number_answer(pred)

            if pred_num is not None and pred_num == ref_num:
                correct += 1

        else:
            ref_month = extract_month_answer(ref)

            if ref_month is not None:
                pred_month = extract_month_answer(pred)

                if pred_month is not None and pred_month == ref_month:
                    correct += 1

            else:
                pred_clean = clean_final_text_answer(pred)
                ref_clean = clean_final_text_answer(ref)

                if pred_clean == ref_clean:
                    correct += 1

    return correct / max(1, total)


def accuracy_score(preds, refs, task_name):
    correct = 0
    total = len(preds)

    for pred, ref in zip(preds, refs):
        if task_name in {"C-STANCE", "FOMC", "ScienceQA"}:
            parsed_pred = parse_answer_for_task(task_name, pred)
            parsed_ref = parse_answer_for_task(task_name, ref)

            if (
                parsed_pred is not None
                and parsed_ref is not None
                and parsed_pred == parsed_ref
            ):
                correct += 1

        else:
            pred_clean = normalize_text(pred)
            ref_clean = normalize_text(ref)

            if pred_clean == ref_clean:
                correct += 1

    return correct / max(1, total)


@dataclass
class TaskEvalResult:
    task_name: str
    metric_name: str
    metric_value: float
    num_examples: int


def score_task(task_name, preds, refs, sources=None):
    metric_name = TASK_METRICS.get(task_name, "exact_match")

    if metric_name == "accuracy":
        value = accuracy_score(preds, refs, task_name)

    elif metric_name == "exact_match":
        value = exact_match_score(preds, refs)

    else:
        raise ValueError(f"Unsupported metric: {metric_name}")

    return TaskEvalResult(
        task_name=task_name,
        metric_name=metric_name,
        metric_value=value,
        num_examples=len(preds),
    )


def parse_for_debug(task_name, pred, ref):
    metric_name = TASK_METRICS.get(task_name, "exact_match")

    if metric_name == "accuracy":
        parsed_pred = parse_answer_for_task(task_name, pred)
        parsed_ref = parse_answer_for_task(task_name, ref)
        correct = (
            parsed_pred is not None
            and parsed_ref is not None
            and parsed_pred == parsed_ref
        )
        return parsed_pred, parsed_ref, correct

    if metric_name == "exact_match":
        ref_num = parse_decimal(ref)

        if ref_num is not None:
            parsed_pred = extract_number_answer(pred)
            parsed_ref = ref_num
            correct = (
                parsed_pred is not None
                and parsed_ref is not None
                and parsed_pred == parsed_ref
            )
            return parsed_pred, parsed_ref, correct

        ref_month = extract_month_answer(ref)

        if ref_month is not None:
            parsed_pred = extract_month_answer(pred)
            parsed_ref = ref_month
            correct = (
                parsed_pred is not None
                and parsed_ref is not None
                and parsed_pred == parsed_ref
            )
            return parsed_pred, parsed_ref, correct

        parsed_pred = clean_final_text_answer(pred)
        parsed_ref = clean_final_text_answer(ref)
        correct = (
            parsed_pred is not None
            and parsed_ref is not None
            and parsed_pred == parsed_ref
        )
        return parsed_pred, parsed_ref, correct

    return None, None, False


def debug_parse_examples(task_name, preds, refs, n=20):
    for i, (pred, ref) in enumerate(zip(preds[:n], refs[:n])):
        parsed_pred, parsed_ref, correct = parse_for_debug(task_name, pred, ref)

        print("=" * 80)
        print(f"idx: {i}")
        print(f"task: {task_name}")
        print(f"raw pred: {repr(pred)}")
        print(f"raw ref : {repr(ref)}")
        print(f"parsed pred: {parsed_pred}")
        print(f"parsed ref : {parsed_ref}")
        print(f"correct: {correct}")


def build_debug_rows(task_name, preds, refs, prompts):
    rows = []

    for prompt, pred, ref in zip(prompts, preds, refs):
        parsed_pred, parsed_ref, correct = parse_for_debug(task_name, pred, ref)

        rows.append({
            "task": task_name,
            "prompt": prompt,
            "ref": ref,
            "pred": pred,
            "parsed_ref": str(parsed_ref) if parsed_ref is not None else None,
            "parsed_pred": str(parsed_pred) if parsed_pred is not None else None,
            "correct": correct,
        })

    return rows


def average_accuracy(results):
    return sum(results.values()) / len(results) if results else 0.0


def backward_transfer(score_matrix, trained_tasks):
    if len(trained_tasks) <= 1:
        return 0.0

    current = trained_tasks[-1]
    current_row = score_matrix.get(current, {})

    diffs = []
    for task in trained_tasks[:-1]:
        if task in current_row and task in score_matrix and task in score_matrix[task]:
            diffs.append(current_row[task] - score_matrix[task][task])

    return sum(diffs) / len(diffs) if diffs else 0.0


def forward_transfer(zero_shot_scores, score_matrix, trained_tasks):
    # If zero-shot scores are not explicitly computed, FWT is undefined in this
    # experiment code path. Return 0.0 to keep the existing table format stable.
    if len(trained_tasks) <= 1 or not zero_shot_scores:
        return 0.0

    total, count = 0.0, 0

    for i, task in enumerate(trained_tasks[1:], 1):
        prev = trained_tasks[i - 1]
        prev_row = score_matrix.get(prev, {})

        if task in prev_row and task in zero_shot_scores:
            total += prev_row[task] - zero_shot_scores[task]
            count += 1

    return total / count if count else 0.0