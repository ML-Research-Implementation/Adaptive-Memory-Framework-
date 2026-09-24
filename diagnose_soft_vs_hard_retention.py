"""Diagnostic-only comparison of AMMR soft and hard retention."""
import argparse
import os
import json
from collections import defaultdict
import torch
from config import DEVICE, MODEL_NAME
from evaluate_squad import load_ammr_checkpoint
from src.models_adaptive import AdaptiveDistilBertQA
from src.squad_data import get_squad_dataloaders

def build_parser():
    parser = argparse.ArgumentParser(description="Diagnose AMMR soft versus hard retention")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-examples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threshold-biases", nargs='+', type=float,
                        default=[0.00, -0.05, -0.10, -0.15, -0.20, -0.25, -0.30, -0.35, -0.40, -0.45, -0.50])
    return parser

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

    bias_results = {bias: {"layers": defaultdict(list), "examples": 0, "span_survived": 0, "span_total": 0} for bias in args.threshold_biases}

    examples_seen = 0
    with torch.no_grad():
        for batch in val_dl:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            start_target = batch.get("start_positions", torch.zeros_like(input_ids[:, 0])).to(DEVICE)
            end_target = batch.get("end_positions", torch.zeros_like(input_ids[:, 0])).to(DEVICE)
            for bias in args.threshold_biases:
                start_logits, end_logits, metrics = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_layer_metrics=True,
                    training=False,
                    threshold_bias=bias,
                    minimum_retention_ratio=target_ratio,
                    answer_span_mask=None,
                    return_original_selection=True,
                )
                bias_results[bias]["examples"] += input_ids.shape[0]
                final_selection = metrics["selection_results"][-1] if metrics["selection_results"] else None
                if final_selection is not None and final_selection.selected_original_indices is not None:
                    final_indices = final_selection.selected_original_indices
                    start_kept = (final_indices == start_target.unsqueeze(1)).any(dim=1)
                    end_kept = (final_indices == end_target.unsqueeze(1)).any(dim=1)
                    survived = (start_kept & end_kept).sum().item()
                    bias_results[bias]["span_survived"] += survived
                    bias_results[bias]["span_total"] += input_ids.shape[0]

                for layer_idx, result in enumerate(metrics["selection_results"]):
                    if result is None:
                        continue
                    scores = result.retention_scores.detach().float()
                    probs = result.retention_probs.detach().float()
                    avc = getattr(result, "actual_valid_counts", None)
                    if avc is not None:
                        valid_counts = avc.detach().long()
                    else:
                        valid_counts = torch.full((scores.shape[0],), scores.shape[1], dtype=torch.long, device=scores.device)
                    svm = getattr(result, "selected_valid_mask", None)
                    if svm is not None:
                        valid_mask = svm.detach().bool()
                    else:
                        valid_mask = torch.ones_like(scores, dtype=torch.bool)
                    batch_valid_total = 0
                    batch_raw_hard_selected = 0
                    batch_soft_sum = 0.0
                    batch_items = 0
                    for row, prob_row, valid_row, count in zip(scores, probs, valid_mask, valid_counts):
                        count_value = int(count.item())
                        if count_value < row.numel():
                            valid_positions = torch.arange(count_value, device=row.device)
                        else:
                            valid_positions = valid_row.nonzero(as_tuple=False).reshape(-1)
                        valid_scores = row[valid_positions]
                        valid_probs = prob_row[valid_positions]
                        batch_valid_total += valid_scores.numel()
                        batch_raw_hard_selected += int((valid_scores + bias > 0.0).sum().item())
                        if valid_probs.numel() > 0:
                            batch_soft_sum += float(valid_probs.mean().item())
                            batch_items += 1
                    final_hard_selected = int(result.actual_retained_counts.sum().item()) if hasattr(result, "actual_retained_counts") else batch_valid_total
                    soft_retention = batch_soft_sum / max(batch_items, 1)
                    bias_results[bias]["layers"][layer_idx].append({
                        "valid_total": batch_valid_total,
                        "raw_hard_selected": batch_raw_hard_selected,
                        "final_hard_selected": final_hard_selected,
                        "soft_retention": soft_retention
                    })
            examples_seen += input_ids.shape[0]
            if examples_seen >= args.num_examples:
                break
    summary = []
    print("\nAMMR THRESHOLD SWEEP DIAGNOSTIC")
    print(f"Checkpoint: {os.path.abspath(args.checkpoint)}")
    print(f"Target ratio: {target_ratio}")
    print(f"Examples: {examples_seen}\n")
    print(f"{'bias':>6} | {'soft_ret':>8} | {'raw_hard_ret':>12} | {'final_hard_ret':>14} | {'gap':>8} | {'answer_survival':>15}")
    print("-" * 75)
    for bias in args.threshold_biases:
        layer_0 = bias_results[bias]["layers"][0]
        soft_ret = sum(l["soft_retention"] for l in layer_0) / max(len(layer_0), 1)
        valid_total = sum(l["valid_total"] for l in layer_0)
        raw_hard = sum(l["raw_hard_selected"] for l in layer_0) / max(valid_total, 1)
        final_hard = sum(l["final_hard_selected"] for l in layer_0) / max(valid_total, 1)
        gap = final_hard - soft_ret
        span_surv = bias_results[bias]["span_survived"] / max(bias_results[bias]["span_total"], 1)
        print(f"{bias:>6.2f} | {soft_ret:>8.5f} | {raw_hard:>12.5f} | {final_hard:>14.5f} | {gap:>8.5f} | {span_surv:>14.2%}")
        summary.append({
            "bias": bias,
            "soft_retention": soft_ret,
            "raw_hard_retention": raw_hard,
            "final_hard_retention": final_hard,
            "gap": gap,
            "answer_survival": span_surv
        })
    print("\nPER-LAYER FINAL HARD RETENTION")
    layer_indices = sorted(bias_results[args.threshold_biases[0]]["layers"].keys())
    header = "bias   | " + " | ".join(f"L{l}" for l in layer_indices)
    print(header)
    print("-" * len(header))
    for bias in args.threshold_biases:
        row = f"{bias:>6.2f} | "
        vals = []
        for l in layer_indices:
            valid = sum(x["valid_total"] for x in bias_results[bias]["layers"][l])
            final = sum(x["final_hard_selected"] for x in bias_results[bias]["layers"][l])
            vals.append(f"{final/max(valid, 1):.4f}")
        print(row + " | ".join(vals))
    with open("threshold_sweep_results.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    run_diagnostic(build_parser().parse_args())
