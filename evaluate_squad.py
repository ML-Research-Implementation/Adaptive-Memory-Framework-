import argparse
import gc
import hashlib
import json
import os
import time
import collections
import string
import re

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from config import MODEL_NAME, DEVICE
from src.squad_data import get_squad_dataloaders
from src.baseline import BaselineQAModel
from src.models_adaptive import AdaptiveDistilBertQA
from src.utils import print_header

DEFAULT_BEST_CHECKPOINT = "/content/drive/MyDrive/ML-Research/AMMR_FULL_SQUAD_TRAINING_20260911_161221/squad_best_checkpoint.pt"
DEFAULT_FINAL_CHECKPOINT = "/content/drive/MyDrive/ML-Research/AMMR_FULL_SQUAD_TRAINING_20260911_161221/squad_final_checkpoint.pt"
EXPECTED_TRAIN_EXAMPLES = 87599
EXPECTED_VALIDATION_EXAMPLES = 10570


def unpack_student_outputs(outputs):
    """Normalize the adaptive student's three-output forward contract."""
    if not isinstance(outputs, tuple) or len(outputs) != 3:
        raise RuntimeError("Adaptive student must return (start_logits, end_logits, layer_metrics)")
    return outputs


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def get_tokens(s):
    return normalize_answer(s).split() if s else []


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if not gold_toks or not pred_toks:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def _state_digest(parameters):
    digest = hashlib.sha256()
    for parameter in parameters:
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _state_difference(model_a, model_b):
    a = dict(model_a.get_retention_scorers().named_parameters())
    b = dict(model_b.get_retention_scorers().named_parameters())
    differences = [
        (left - b[name]).detach().abs()
        for name, left in a.items()
    ]
    nonzero = sum(int(torch.any(value > 0).item()) for value in differences)
    total = sum(float(value.sum()) for value in differences)
    maximum = max((float(value.max()) for value in differences), default=0.0)
    return len(differences), nonzero, total, maximum


def load_ammr_checkpoint(model, checkpoint_path):
    """Strictly load an explicit AMMR checkpoint; never fall back to fresh scorers."""
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise RuntimeError(
            "AMMR trained checkpoint not found. Refusing to evaluate with initial scorer weights. "
            f"Requested path: {checkpoint_path}"
        )

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    checkpoint_format = "direct state_dict"
    metadata = {}
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        checkpoint_format = "full training checkpoint"
        metadata = checkpoint
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise RuntimeError(f"Unsupported AMMR checkpoint format: {checkpoint_path}")

    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Checkpoint state_dict is not a mapping: {checkpoint_path}")

    scorer = model.get_retention_scorers()
    scorer_keys = set(scorer.state_dict())
    full_keys = set(model.state_dict())
    state_keys = set(state_dict)

    if state_keys == scorer_keys:
        scorer.load_state_dict(state_dict, strict=True)
        structure = "retention_scorers ModuleList"
    elif state_keys == full_keys:
        model.load_state_dict(state_dict, strict=True)
        structure = "full AdaptiveDistilBertQA"
    elif state_keys and all(key.startswith("retention_scorers.") for key in state_keys):
        stripped = {key[len("retention_scorers."):]: value for key, value in state_dict.items()}
        if set(stripped) != scorer_keys:
            raise RuntimeError(f"Checkpoint scorer keys do not match model: {checkpoint_path}")
        scorer.load_state_dict(stripped, strict=True)
        structure = "retention_scorers-prefixed"
    else:
        missing = sorted((scorer_keys | full_keys) - state_keys)
        unexpected = sorted(state_keys - scorer_keys - full_keys)
        raise RuntimeError(
            f"AMMR checkpoint state mismatch: {checkpoint_path}; "
            f"missing_or_unmatched={missing[:10]}, unexpected={unexpected[:10]}"
        )

    print("=" * 48)
    print("AMMR CHECKPOINT LOAD")
    print("=" * 48)
    print(f"Checkpoint: {os.path.abspath(checkpoint_path)}")
    print(f"Checkpoint format: {checkpoint_format}")
    print(f"Loaded structure: {structure}")
    print(f"Epoch: {metadata.get('epoch', 'N/A')}")
    print(f"Step: {metadata.get('step', 'N/A')}")
    print(f"Target ratio: {metadata.get('target_ratio', 'N/A')}")
    print(f"Lambda: {metadata.get('lagrangian_multiplier', 'N/A')}")
    print("Missing keys: 0")
    print("Unexpected keys: 0")
    print(f"Loaded keys: {len(state_keys)}")
    print("Status: SUCCESS")
    print("=" * 48)
    return checkpoint


def verify_loaded_model(fresh, loaded, dataloader, batches=2):
    tensor_count, different_count, total_difference, max_difference = _state_difference(fresh, loaded)
    print("AMMR scorer verification:")
    print(f"  scorer tensors: {tensor_count}")
    print(f"  differing tensors: {different_count}")
    print(f"  total absolute difference: {total_difference:.6g}")
    print(f"  maximum absolute difference: {max_difference:.6g}")
    if different_count == 0:
        print("WARNING: Loaded scorer tensors match fresh initialization.")

    fresh.eval()
    loaded.eval()
    differences = 0
    total = 0
    fresh_ratios = []
    loaded_ratios = []
    fresh_originals = []
    loaded_originals = []
    fresh_selecteds = []
    loaded_selecteds = []
    with torch.no_grad():
        for batch in dataloader:
            if total >= batches:
                break
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            fresh_out = unpack_student_outputs(fresh(ids, mask, return_layer_metrics=True, training=False))
            loaded_out = unpack_student_outputs(loaded(ids, mask, return_layer_metrics=True, training=False))
            fresh_results = fresh_out[2].get("selection_results", [])
            loaded_results = loaded_out[2].get("selection_results", [])
            for fresh_result, loaded_result in zip(fresh_results, loaded_results):
                if fresh_result is None or loaded_result is None:
                    continue
                fresh_indices = fresh_result.selected_indices
                loaded_indices = loaded_result.selected_indices

                fresh_mask = torch.zeros(
                    fresh_indices.size(0), fresh_result.num_original,
                    dtype=torch.bool, device=fresh_indices.device
                )
                loaded_mask = torch.zeros(
                    loaded_indices.size(0), loaded_result.num_original,
                    dtype=torch.bool, device=loaded_indices.device
                )

                fresh_mask.scatter_(1, fresh_indices, True)
                loaded_mask.scatter_(1, loaded_indices, True)

                if fresh_mask.size() == loaded_mask.size():
                    differences += int(torch.any(fresh_mask != loaded_mask).item())
                else:
                    differences += 1
                total += 1
                fresh_ratios.append(fresh_result.retention_ratio)
                loaded_ratios.append(loaded_result.retention_ratio)
                fresh_originals.append(float(fresh_result.num_original))
                loaded_originals.append(float(loaded_result.num_original))
                fresh_selecteds.append(float(fresh_result.num_selected))
                loaded_selecteds.append(float(loaded_result.num_selected))

    print(f"Fresh original length: {sum(fresh_originals) / max(1, len(fresh_originals)):.4f}")
    print(f"Loaded original length: {sum(loaded_originals) / max(1, len(loaded_originals)):.4f}")
    print(f"Fresh selected count: {sum(fresh_selecteds) / max(1, len(fresh_selecteds)):.4f}")
    print(f"Loaded selected count: {sum(loaded_selecteds) / max(1, len(loaded_selecteds)):.4f}")
    print(f"Fresh retention ratio: {sum(fresh_ratios) / max(1, len(fresh_ratios)):.4f}")
    print(f"Loaded retention ratio: {sum(loaded_ratios) / max(1, len(loaded_ratios)):.4f}")
    print(f"Different mask batches: {differences} / {total}")
    if different_count and differences == 0:
        print("WARNING: TRAINED SCORER PARAMETERS DIFFER, BUT RETENTION DECISIONS MATCH INITIAL SCORER.")
    return {
        "scorer_tensors": tensor_count,
        "differing_scorer_tensors": different_count,
        "total_parameter_difference": total_difference,
        "max_parameter_difference": max_difference,
        "different_masks": differences,
        "mask_comparisons": total,
    }


def evaluate_model(model, dataloader, dataset_features, raw_val_data, tokenizer, is_baseline=False, threshold_bias=0.0):
    if hasattr(model, "eval"):
        model.eval()
    elif hasattr(model, "qa_model"):
        model.qa_model.eval()

    all_start_logits, all_end_logits = [], []
    total_latency = 0.0
    num_batches = 0
    total_retained_tokens = 0
    total_original_tokens = 0
    layer_span_survival = [0] * 6
    layer_span_total = [0] * 6
    all_retention_scores = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(DEVICE)

    for batch in tqdm(dataloader, leave=False, desc="Evaluating"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        start_time = time.perf_counter()
        with torch.no_grad():
            if is_baseline:
                outputs = model.qa_model(input_ids, attention_mask) if hasattr(model, "qa_model") else model(input_ids, attention_mask)
                start_logits, end_logits, layer_metrics = outputs.start_logits, outputs.end_logits, None
            else:
                start_logits, end_logits, layer_metrics = unpack_student_outputs(model(
                    input_ids, attention_mask, return_layer_metrics=True,
                    training=False, threshold_bias=threshold_bias
                ))
        total_latency += time.perf_counter() - start_time
        num_batches += 1
        all_start_logits.append(start_logits.cpu())
        all_end_logits.append(end_logits.cpu())

        if layer_metrics and layer_metrics.get("selection_results"):
            start_pos = batch.get("start_positions", torch.zeros_like(input_ids[:, 0]))
            end_pos = batch.get("end_positions", torch.zeros_like(input_ids[:, 0]))
            batch_tokens = batch_original = 0
            for layer_idx, result in enumerate(layer_metrics["selection_results"]):
                if result is None:
                    continue
                batch_tokens += result.num_selected
                batch_original += result.num_original
                for batch_idx in range(input_ids.size(0)):
                    start_idx = int(start_pos[batch_idx])
                    end_idx = int(end_pos[batch_idx])
                    if start_idx == 0 and end_idx == 0:
                        continue
                    span = torch.arange(start_idx, end_idx + 1, device=DEVICE)
                    survived = torch.all(torch.isin(span, result.selected_indices[batch_idx])).item()
                    layer_span_survival[layer_idx] += survived
                    layer_span_total[layer_idx] += 1
            total_retained_tokens += batch_tokens
            total_original_tokens += batch_original
            if threshold_bias == 0.0:
                all_retention_scores.extend(
                    result.retention_scores.detach().cpu() for result in layer_metrics["selection_results"] if result is not None
                )

    if not all_start_logits:
        raise RuntimeError("Evaluation produced no batches.")
    all_start_logits = torch.cat(all_start_logits)
    all_end_logits = torch.cat(all_end_logits)
    raw_by_id = {example["id"]: example for example in raw_val_data}
    exact_scores, f1_scores = [], []
    for index, feature in enumerate(dataset_features):
        example = raw_by_id.get(feature["example_id"])
        if example is None or not example["answers"]["text"]:
            continue
        start_idx = torch.argmax(all_start_logits[index]).item()
        end_idx = torch.argmax(all_end_logits[index]).item()
        prediction = "" if end_idx < start_idx else tokenizer.decode(
            feature["input_ids"][start_idx:end_idx + 1], skip_special_tokens=True
        )
        answers = example["answers"]["text"]
        exact_scores.append(max(compute_exact(answer, prediction) for answer in answers))
        f1_scores.append(max(compute_f1(answer, prediction) for answer in answers))

    retention = 100.0 * total_retained_tokens / total_original_tokens if total_original_tokens else 100.0
    spans = [100.0 if total == 0 else 100.0 * kept / total for kept, total in zip(layer_span_survival, layer_span_total)]
    scores = torch.cat([value.reshape(-1) for value in all_retention_scores]) if all_retention_scores else None
    return {
        "em": 100.0 * sum(exact_scores) / max(1, len(exact_scores)),
        "f1": 100.0 * sum(f1_scores) / max(1, len(f1_scores)),
        "latency_ms": 1000.0 * total_latency / max(1, num_batches),
        "retention": retention,
        "attention_cost": (retention / 100.0) ** 2 * 100.0,
        "compute_reduction": 100.0 - (retention / 100.0) ** 2 * 100.0,
        "answer_survival": sum(spans) / len(spans),
        "span_survival_rates": spans,
        "all_scores": scores,
    }


def find_best_threshold(scores, target_retentions):
    return -torch.quantile(scores.float(), 1.0 - target_retentions).item()


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate AMMR on SQuAD")
    parser.add_argument("--ammr_best_checkpoint", default=DEFAULT_BEST_CHECKPOINT)
    parser.add_argument("--ammr_final_checkpoint", default=DEFAULT_FINAL_CHECKPOINT)
    parser.add_argument("--max_val_samples", type=int, default=-1, help="-1 means all validation examples")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_json", default="ammr_final_benchmark.json")
    return parser


def evaluate_checkpoint(label, checkpoint_path, val_dl, val_features, val_data, tokenizer, reference_model):
    model = AdaptiveDistilBertQA(model_name=MODEL_NAME, device=DEVICE).to(DEVICE)
    load_ammr_checkpoint(model, checkpoint_path)
    verify_loaded_model(reference_model, model, val_dl, batches=2)
    result = evaluate_model(model, val_dl, val_features, val_data, tokenizer, is_baseline=False, threshold_bias=0.0)
    result.pop("all_scores", None)
    result.update({"model": label, "checkpoint": os.path.abspath(checkpoint_path), "setting": "bias=0.0"})
    return result


def main(argv=None):
    args = build_parser().parse_args(argv)
    max_val_samples = None if args.max_val_samples < 0 else args.max_val_samples
    _, val_dl, train_data, val_data, val_features = get_squad_dataloaders(
        batch_size=args.batch_size,
        max_train_samples=1,
        max_val_samples=max_val_samples,
    )
    if args.max_val_samples < 0 and len(val_data) != EXPECTED_VALIDATION_EXAMPLES:
        raise RuntimeError("Final benchmark requires all 10,570 SQuAD validation examples.")
    print(f"Training examples loaded for evaluation: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    if args.max_val_samples < 0:
        assert len(val_data) == EXPECTED_VALIDATION_EXAMPLES

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    baseline = BaselineQAModel(freeze_parameters=True)
    baseline_result = evaluate_model(baseline, val_dl, val_features, val_data, tokenizer, is_baseline=True)
    del baseline
    gc.collect()

    fresh = AdaptiveDistilBertQA(model_name=MODEL_NAME, device=DEVICE).to(DEVICE)
    results = [baseline_result | {
        "model": "Baseline", "checkpoint": None, "retention": 100.0,
        "attention_cost": 100.0, "answer_survival": 100.0,
    }]
    for label, path in (("AMMR Best", args.ammr_best_checkpoint), ("AMMR Final", args.ammr_final_checkpoint)):
        results.append(evaluate_checkpoint(label, path, val_dl, val_features, val_data, tokenizer, fresh))

    output = {
        "dataset": "SQuAD",
        "train_examples": len(train_data),
        "validation_examples": len(val_data),
        "baseline_checkpoint": None,
        "ammr_best_checkpoint": os.path.abspath(args.ammr_best_checkpoint),
        "ammr_final_checkpoint": os.path.abspath(args.ammr_final_checkpoint),
        "initial_scorer_fallback": False,
        "results": results,
    }
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print_header("FINAL AMMR BENCHMARK")
    print(f"Dataset: SQuAD\nTraining examples: {len(train_data):,}\nValidation examples: {len(val_data):,}")
    print(f"AMMR Best checkpoint: {os.path.abspath(args.ammr_best_checkpoint)}")
    print(f"AMMR Final checkpoint: {os.path.abspath(args.ammr_final_checkpoint)}")
    print("Checkpoint loading: VERIFIED\nInitial scorer fallback: DISABLED")
    print_header("ACCURACY-EFFICIENCY TRADE-OFF")
    print("Model / Setting | Exact Match | F1 | Tokens Retained | Attn Cost | Latency")
    for result in results:
        print(f"{result['model']:<16} | {result['em']:.2f} | {result['f1']:.2f} | {result['retention']:.2f}% | {result['attention_cost']:.2f}% | {result['latency_ms']:.2f} ms")
    return output


if __name__ == "__main__":
    main()
