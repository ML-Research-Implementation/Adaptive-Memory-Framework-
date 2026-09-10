"""Shared SQuAD decoding and task-preservation metrics."""

import collections
import re
import string
from typing import Dict, Iterable, List, Sequence, Tuple

import torch


def normalize_answer(text: str) -> str:
    def remove_articles(value):
        return re.sub(r"\b(a|an|the)\b", " ", value)
    value = text.lower()
    value = "".join(ch for ch in value if ch not in set(string.punctuation))
    return " ".join(remove_articles(value).split())


def squad_exact(gold: str, prediction: str) -> int:
    return int(normalize_answer(gold) == normalize_answer(prediction))


def squad_f1(gold: str, prediction: str) -> float:
    gold_tokens = normalize_answer(gold).split()
    prediction_tokens = normalize_answer(prediction).split()
    if not gold_tokens or not prediction_tokens:
        return float(gold_tokens == prediction_tokens)
    overlap = sum((collections.Counter(gold_tokens) & collections.Counter(prediction_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def decode_feature_span(tokenizer, feature: Dict, start: int, end: int) -> str:
    if end < start or start < 0 or end >= len(feature["input_ids"]):
        return ""
    offsets = feature.get("offset_mapping")
    if offsets and offsets[start] is not None and offsets[end] is not None:
        context_ids = feature["input_ids"]
        return tokenizer.decode(context_ids[start:end + 1], skip_special_tokens=True)
    return tokenizer.decode(feature["input_ids"][start:end + 1], skip_special_tokens=True)


def evaluate_squad_predictions(logits_by_feature: Sequence[Tuple[torch.Tensor, torch.Tensor]],
                               features: Sequence[Dict], raw_examples: Iterable[Dict],
                               tokenizer) -> Tuple[float, float, List[Dict]]:
    """Aggregate feature predictions by example and score decoded answer text."""
    examples = {example["id"]: example for example in raw_examples}
    grouped: Dict[str, List[Tuple[float, str]]] = collections.defaultdict(list)
    for (start_logits, end_logits), feature in zip(logits_by_feature, features):
        start = int(torch.argmax(start_logits).item())
        end = int(torch.argmax(end_logits).item())
        prediction = decode_feature_span(tokenizer, feature, start, end)
        grouped[feature["example_id"]].append((float(start_logits[start] + end_logits[end]), prediction))

    exact_scores, f1_scores, details = [], [], []
    for example_id, example in examples.items():
        candidates = grouped.get(example_id, [])
        prediction = max(candidates, key=lambda item: item[0])[1] if candidates else ""
        gold_answers = example.get("answers", {}).get("text", [])
        if not gold_answers:
            continue
        exact = max(squad_exact(answer, prediction) for answer in gold_answers)
        f1 = max(squad_f1(answer, prediction) for answer in gold_answers)
        exact_scores.append(exact)
        f1_scores.append(f1)
        details.append({"example_id": example_id, "prediction": prediction, "exact": exact, "f1": f1})
    count = max(len(exact_scores), 1)
    return 100.0 * sum(exact_scores) / count, 100.0 * sum(f1_scores) / count, details
