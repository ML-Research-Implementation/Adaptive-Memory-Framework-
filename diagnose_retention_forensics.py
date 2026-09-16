"""Diagnostic-only inspection of learned AMMR retention behavior.

This script never trains, updates parameters, or writes checkpoints/results.
It uses the production model, selector, SQuAD loader, and strict checkpoint
loader. Run it against the actual trained checkpoint, for example:

python diagnose_retention_forensics.py \
  --checkpoint /content/AMMR_CLEAN_RUN/squad_final_checkpoint.pt \
  --split validation --num-examples 1 --threshold 0.0
"""

import argparse
import os
from typing import Dict, List

import torch

from config import DEVICE, MODEL_NAME
from evaluate_squad import load_ammr_checkpoint
from src.models_adaptive import AdaptiveDistilBertQA
from src.squad_data import get_squad_dataloaders


def build_parser():
    parser = argparse.ArgumentParser(description="Diagnose learned AMMR retention without training")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("validation",), default="validation")
    parser.add_argument("--num-examples", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="HardConcrete logit threshold bias; 0.0 is production default")
    parser.add_argument("--target-ratio", type=float, default=None,
                        help="Optional inference floor; defaults to checkpoint target_ratio")
    parser.add_argument("--batch-size", type=int, default=1)
    return parser


def _stat(tensor):
    flat = tensor.detach().float().reshape(-1).cpu()
    if flat.numel() == 0:
        return {"min": 0.0, "mean": 0.0, "max": 0.0, "std": 0.0, "median": 0.0}
    return {
        "min": float(flat.min()),
        "mean": float(flat.mean()),
        "max": float(flat.max()),
        "std": float(flat.std(unbiased=False)),
        "median": float(flat.median()),
    }


def _parameter_stats(model):
    rows = []
    for layer, scorer in enumerate(model.retention_scorers):
        values = torch.cat([p.detach().float().reshape(-1).cpu() for p in scorer.parameters()])
        final = scorer.network[-1]
        rows.append({
            "layer": layer,
            "parameter_norm": float(values.norm()),
            "parameter_mean": float(values.mean()),
            "parameter_std": float(values.std(unbiased=False)),
            "final_bias": float(final.bias.detach().float().mean()),
            "gradient_norm": None,
        })
    return rows


def _layer_row(layer_idx, result, input_ids, target_ratio, threshold):
    valid = result.actual_valid_counts.detach().cpu().reshape(-1)
    raw = result.raw_retained_counts.detach().cpu().reshape(-1)
    final = result.actual_retained_counts.detach().cpu().reshape(-1)
    added = result.floor_added_counts.detach().cpu().reshape(-1)
    scores = _stat(result.retention_scores)
    probs = _stat(result.retention_probs)
    valid_total = int(valid.sum())
    raw_total = int(raw.sum())
    final_total = int(final.sum())
    floor_required = int(torch.ceil(valid.float() * float(target_ratio)).sum())
    hard_mask = result.selected_valid_mask.detach().cpu()
    protected = ((input_ids == 101) | (input_ids == 102))
    protected_count = int(protected.sum())
    prob_flat = result.retention_probs.detach().float().reshape(-1)
    valid_flat = torch.ones_like(prob_flat, dtype=torch.bool)
    pct_above = float((prob_flat[valid_flat] >= 0.5).float().mean() * 100.0) if prob_flat.numel() else 0.0
    pct_below = 100.0 - pct_above
    return {
        "layer": layer_idx,
        "valid_tokens": valid_total,
        "protected_tokens_observed": protected_count,
        "score": scores,
        "probability": probs,
        "hard_threshold_convention": f"logit + {threshold} > 0",
        "probability_ge_0.5_percent": pct_above,
        "probability_lt_0.5_percent": pct_below,
        "raw_selected_before_floor": raw_total,
        "raw_retention_ratio": raw_total / max(valid_total, 1),
        "floor_required": floor_required,
        "floor_added_tokens": int(added.sum()),
        "topk_repair_activated": bool(result.topk_repair_activated),
        "final_retained_tokens": final_total,
        "final_retention_ratio": final_total / max(valid_total, 1),
        "input_sequence_length": result.num_original,
        "output_sequence_length": result.num_selected,
        "selected_valid_mask_count": int(hard_mask.sum()),
    }


def run_diagnostic(args):
    if args.num_examples < 1:
        raise ValueError("--num-examples must be positive")
    if not os.path.isfile(args.checkpoint):
        raise RuntimeError(f"Checkpoint does not exist: {args.checkpoint}")

    _, val_dl, _, val_data, _ = get_squad_dataloaders(
        batch_size=args.batch_size,
        max_train_samples=1,
        max_val_samples=args.num_examples,
    )
    model = AdaptiveDistilBertQA(model_name=MODEL_NAME, device=DEVICE, freeze_transformer=True).to(DEVICE)
    checkpoint = load_ammr_checkpoint(model, args.checkpoint)
    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    target_ratio = args.target_ratio
    if target_ratio is None:
        target_ratio = float(metadata.get("target_ratio", model.retention_schedule[0]))

    model.eval()
    layer_rows: List[Dict] = []
    total_before = 0
    total_after = 0
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
                threshold_bias=args.threshold,
                minimum_retention_ratio=target_ratio,
                answer_span_mask=None,
                return_original_selection=True,
            )
            assert start_logits.shape == end_logits.shape
            assert start_logits.shape[1] == input_ids.shape[1], "QA logits were not scattered to original length"
            for layer_idx, result in enumerate(metrics["selection_results"]):
                if result is None:
                    continue
                row = _layer_row(layer_idx, result, input_ids, target_ratio, args.threshold)
                layer_rows.append(row)
                total_before += row["input_sequence_length"] * input_ids.shape[0]
                total_after += row["output_sequence_length"] * input_ids.shape[0]
            examples_seen += input_ids.shape[0]
            if examples_seen >= args.num_examples:
                break

    print("AMMR RETENTION FORENSICS")
    print(f"Checkpoint: {os.path.abspath(args.checkpoint)}")
    print(f"Split: {args.split}; examples: {examples_seen}")
    print(f"Epoch: {metadata.get('epoch', 'N/A')}")
    print(f"Step: {metadata.get('step', 'N/A')}")
    print(f"Target ratio: {metadata.get('target_ratio', target_ratio)}")
    print(f"Lambda: {metadata.get('lagrangian_multiplier', 'N/A')}")
    print("Normal inference: training=False, answer_span_mask=None")
    print("Hard decision: z = (retention_logit + threshold_bias > 0).float()")
    print(f"Threshold bias: {args.threshold}; probability >= 0.5 is diagnostic only")
    print("\nRAW SELECTION / FLOOR ADDITIONS / TOP-K REPAIR / FINAL SELECTION")
    for row in layer_rows:
        print(row)
    print(f"Physical token positions before compaction: {total_before}")
    print(f"Physical token positions after compaction: {total_after}")
    print(f"Physical compaction changed length: {total_after < total_before}")
    print("QA start/end logits scattered to original coordinate length: verified")
    return layer_rows


if __name__ == "__main__":
    run_diagnostic(build_parser().parse_args())
