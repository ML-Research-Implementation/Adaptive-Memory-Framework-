"""Diagnostic-only comparison of AMMR soft and hard retention.

This script loads a trained checkpoint, runs normal inference on real SQuAD
validation examples, and reports the exact production soft-retention values
alongside hard gate decisions. It never trains, updates parameters, or writes
checkpoints.
"""

import argparse
import os
from collections import defaultdict

import torch

from config import DEVICE, MODEL_NAME
from evaluate_squad import load_ammr_checkpoint
from src.models_adaptive import AdaptiveDistilBertQA
from src.squad_data import get_squad_dataloaders


def build_parser():
    parser = argparse.ArgumentParser(description="Diagnose AMMR soft versus hard retention")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-examples", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threshold-biases", nargs="+", type=float,
                        default=[0.0, -0.5, -1.0, -2.0, -3.0])
    return parser


def _stats(values):
    values = values.detach().float().reshape(-1)
    return {
        "min": float(values.min().item()),
        "mean": float(values.mean().item()),
        "max": float(values.max().item()),
        "std": float(values.std(unbiased=False).item()),
    }


def _safe_mean(values):
    return sum(values) / max(len(values), 1)


def _layer_record(result, threshold_biases):
    scores = result.retention_scores.detach().float()
    probabilities = result.retention_probs.detach().float()
    valid_counts = result.actual_valid_counts.detach().long()
    protected_counts = result.raw_protected_counts.detach().long()
    valid_mask = result.selected_valid_mask.detach().bool()

    hard_mask_at_zero = scores > 0.0
    valid_score_values = []
    valid_probability_values = []
    hard_zero_count = 0
    valid_total = 0
    protected_total = int(protected_counts.sum().item())
    soft_ratios = []
    threshold_counts = {bias: 0 for bias in threshold_biases}
    for row, prob_row, valid_row, count in zip(scores, probabilities, valid_mask, valid_counts):
        valid_positions = valid_row.nonzero(as_tuple=False).reshape(-1)
        count_value = int(count.item())
        # selected_valid_mask contains the compacted valid positions in the
        # inference result. For the diagnostic's direct hard-vs-soft comparison,
        # use the original scorer tensor and the attention-valid prefix/count.
        if count_value < row.numel():
            valid_positions = torch.arange(count_value, device=row.device)
        valid_scores = row[valid_positions]
        valid_probs = prob_row[valid_positions]
        valid_score_values.append(valid_scores)
        valid_probability_values.append(valid_probs)
        valid_total += int(valid_scores.numel())
        hard_zero_count += int((valid_scores > 0.0).sum().item())
        soft_ratios.append(float(valid_probs.mean().item()) if valid_probs.numel() else 0.0)
        for bias in threshold_biases:
            threshold_counts[bias] += int((valid_scores + bias > 0.0).sum().item())

    score_values = torch.cat(valid_score_values) if valid_score_values else torch.zeros(1)
    probability_values = torch.cat(valid_probability_values) if valid_probability_values else torch.zeros(1)
    soft_retention = _safe_mean(soft_ratios)
    hard_retention = hard_zero_count / max(valid_total, 1)
    return {
        "score": _stats(score_values),
        "probability": _stats(probability_values),
        "soft_retention": soft_retention,
        "hard_retention": hard_retention,
        "hard_minus_soft": hard_retention - soft_retention,
        "valid_tokens": valid_total,
        "protected_tokens": protected_total,
        "hard_zero_count": hard_zero_count,
        "threshold_counts": threshold_counts,
        "threshold_fractions": {
            bias: threshold_counts[bias] / max(valid_total, 1)
            for bias in threshold_biases
        },
        "probability_ge_half": float((probability_values >= 0.5).float().mean().item()),
        "probability_lt_half": float((probability_values < 0.5).float().mean().item()),
    }


def run_diagnostic(args):
    if not os.path.isfile(args.checkpoint):
        raise RuntimeError(f"Checkpoint does not exist: {args.checkpoint}")
    if args.num_examples < 1:
        raise ValueError("--num-examples must be positive")

    _, val_dl, _, _, _ = get_squad_dataloaders(
        batch_size=args.batch_size,
        max_train_samples=1,
        max_val_samples=args.num_examples,
    )
    model = AdaptiveDistilBertQA(
        model_name=MODEL_NAME, device=DEVICE, freeze_transformer=True
    ).to(DEVICE)
    checkpoint = load_ammr_checkpoint(model, args.checkpoint)
    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    target_ratio = float(metadata.get("target_ratio", model.retention_schedule[0]))
    model.eval()

    layer_records = defaultdict(list)
    examples_seen = 0
    with torch.no_grad():
        for batch in val_dl:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            start_logits, end_logits, metrics = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_layer_metrics=True,
                training=False,
                threshold_bias=0.0,
                minimum_retention_ratio=target_ratio,
                answer_span_mask=None,
                return_original_selection=True,
            )
            if start_logits.shape != end_logits.shape or start_logits.shape[1] != input_ids.shape[1]:
                raise RuntimeError("QA logits were not scattered to original input length")
            for layer_idx, result in enumerate(metrics["selection_results"]):
                if result is not None:
                    layer_records[layer_idx].append(_layer_record(result, args.threshold_biases))
            examples_seen += input_ids.shape[0]
            if examples_seen >= args.num_examples:
                break

    print("AMMR SOFT VS HARD RETENTION DIAGNOSTIC")
    print(f"Checkpoint: {os.path.abspath(args.checkpoint)}")
    print(f"Epoch: {metadata.get('epoch', 'N/A')}; Step: {metadata.get('step', 'N/A')}")
    print(f"Target ratio metadata: {target_ratio}; Lambda: {metadata.get('lagrangian_multiplier', 'N/A')}")
    print(f"Examples: {examples_seen}; normal inference: training=False, answer_span_mask=None")
    print("Production hard gate: z = (retention_logit + threshold_bias > 0).float()")
    print("Production HardConcreteGate inference: no noise, no sigmoid threshold, threshold bias applied to logits")
    print("Production training soft retention: mean over valid sigmoid(logit) per example, then mean across batch")
    print("Protected tokens are counted separately; padding is excluded from all retention fractions.")

    print("\nPER-LAYER AGGREGATES")
    print("layer | score min/mean/max/std | prob min/mean/max | soft | hard@0 | hard-soft | valid | protected | logit>0 | logit<=0")
    aggregate = {}
    for layer_idx in sorted(layer_records):
        records = layer_records[layer_idx]
        score_min = min(row["score"]["min"] for row in records)
        score_mean = _safe_mean([row["score"]["mean"] for row in records])
        score_max = max(row["score"]["max"] for row in records)
        score_std = _safe_mean([row["score"]["std"] for row in records])
        prob_min = min(row["probability"]["min"] for row in records)
        prob_mean = _safe_mean([row["probability"]["mean"] for row in records])
        prob_max = max(row["probability"]["max"] for row in records)
        soft = _safe_mean([row["soft_retention"] for row in records])
        hard = _safe_mean([row["hard_retention"] for row in records])
        valid = sum(row["valid_tokens"] for row in records)
        protected = sum(row["protected_tokens"] for row in records)
        positive = sum(row["hard_zero_count"] for row in records)
        aggregate[layer_idx] = {"soft": soft, "hard": hard, "gap": hard - soft}
        print(f"{layer_idx} | {score_min:.5f}/{score_mean:.5f}/{score_max:.5f}/{score_std:.5f} | "
              f"{prob_min:.5f}/{prob_mean:.5f}/{prob_max:.5f} | {soft:.5f} | {hard:.5f} | "
              f"{hard-soft:.5f} | {valid} | {protected} | {positive / max(valid, 1):.5f} | {1 - positive / max(valid, 1):.5f}")

    print("\nHYPOTHETICAL THRESHOLD SENSITIVITY")
    print("layer | " + " | ".join(f"bias={bias:g}" for bias in args.threshold_biases))
    for layer_idx in sorted(layer_records):
        values = []
        records = layer_records[layer_idx]
        for bias in args.threshold_biases:
            total = sum(row["threshold_counts"][bias] for row in records)
            valid = sum(row["valid_tokens"] for row in records)
            values.append(f"{total / max(valid, 1):.5f}")
        print(f"{layer_idx} | " + " | ".join(values))

    print("\nGATE SEMANTICS")
    print("training gate: stochastic hard-concrete noise, temperature=0.5, stretch=[-0.1, 1.1], clamp=[0,1]")
    print("inference gate: direct logit comparison; sigmoid probability is returned for budget/diagnostic use")
    print("sigmoid(logit) >= 0.5 is mathematically equivalent to logit >= 0, but production uses strict logit > 0")
    print("Conclusion: compare soft and hard values to classify expected threshold calibration versus implementation mismatch.")
    return aggregate


if __name__ == "__main__":
    run_diagnostic(build_parser().parse_args())
