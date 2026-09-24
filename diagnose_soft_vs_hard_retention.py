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
                    if layer_idx == 0:
                        valid_mask = attention_mask >= 0.5
                    else:
                        prev_result = metrics["selection_results"][layer_idx - 1]
                        valid_mask = prev_result.new_attention_mask >= 0.5
                    
                    valid_mask_aligned = valid_mask.to(device=scores.device, dtype=torch.bool)
                        
                    batch_valid_total = int(valid_mask.sum().item())
                    raw_gate = (scores + bias > 0).float()
                    batch_raw_hard_selected = int(raw_gate[valid_mask_aligned].sum().item())
                    
                    floor_added = getattr(result, "floor_added_counts", torch.tensor(0)).sum().item()
                    
                    if probs.numel() > 0:
                        batch_soft_sum = float(probs[valid_mask_aligned].mean().item())
                        batch_items = 1
                    else:
                        batch_soft_sum = 0.0
                        batch_items = 1

                    final_hard_selected = int(result.actual_retained_counts.sum().item()) if hasattr(result, "actual_retained_counts") else batch_valid_total
                    soft_retention = batch_soft_sum
                    
                    floor_required = int(torch.ceil(torch.tensor(batch_valid_total * target_ratio)).item())
                    original_tokens = result.num_original

                    bias_results[bias]["layers"][layer_idx].append({
                        "valid_total": batch_valid_total,
                        "raw_hard_selected": batch_raw_hard_selected,
                        "floor_required": floor_required,
                        "floor_added": floor_added,
                        "final_hard_selected": final_hard_selected,
                        "soft_retention": soft_retention,
                        "original_tokens": original_tokens * scores.shape[0]
                    })
            examples_seen += input_ids.shape[0]
            if examples_seen >= args.num_examples:
                break
    summary = []
    print("\nAMMR THRESHOLD SWEEP DIAGNOSTIC")
    print(f"Checkpoint: {os.path.abspath(args.checkpoint)}")
    print(f"Target ratio: {target_ratio}")
    print(f"Examples: {examples_seen}\n")
    print(f"{'bias':>6} | {'soft_ret':>8} | {'raw_hard_ret':>12} | {'final_hard_ret':>14} | {'gap':>8} | {'answer_survival':>15} | {'floor_added':>11}")
    print("-" * 105)
    for bias in args.threshold_biases:
        layer_metrics = bias_results[bias]["layers"]
        # Averages across all layers for the overall summary
        all_layers = [l for layer_idx in layer_metrics for l in layer_metrics[layer_idx]]
        soft_ret = sum(l["soft_retention"] for l in all_layers) / max(len(all_layers), 1)
        valid_total = sum(l["valid_total"] for l in all_layers)
        raw_hard = sum(l["raw_hard_selected"] for l in all_layers) / max(valid_total, 1)
        final_hard = sum(l["final_hard_selected"] for l in all_layers) / max(valid_total, 1)
        gap = final_hard - soft_ret
        span_surv = bias_results[bias]["span_survived"] / max(bias_results[bias]["span_total"], 1)
        floor_added = sum(l["floor_added"] for l in all_layers)
        
        print(f"{bias:>6.2f} | {soft_ret:>8.5f} | {raw_hard:>12.5f} | {final_hard:>14.5f} | {gap:>8.5f} | {span_surv:>14.2%} | {floor_added:>11}")
        summary.append({
            "bias": bias,
            "soft_retention": soft_ret,
            "raw_hard_retention": raw_hard,
            "final_hard_retention": final_hard,
            "gap": gap,
            "answer_survival": span_surv,
            "floor_added": floor_added
        })

    print("\nPER-LAYER ACCOUNTING AUDIT")
    layer_indices = sorted(bias_results[args.threshold_biases[0]]["layers"].keys())
    for bias in args.threshold_biases:
        print(f"\n--- Threshold Bias: {bias:.2f} ---")
        print(f"{'Layer':>5} | {'pre_rep':>8} | {'floor_req':>9} | {'floor_add':>9} | {'final_sel':>9} | {'valid_toks':>10} | {'orig_toks':>9} | {'pre_ret%':>8} | {'final_ret%':>10}")
        for l in layer_indices:
            data = bias_results[bias]["layers"][l]
            floor_req = sum(x["floor_required"] for x in data)
            floor_add = sum(x["floor_added"] for x in data)
            final_sel = sum(x["final_hard_selected"] for x in data)
            # Pre-repair selection is the final selection minus whatever the floor top-k repair added
            pre_rep = final_sel - floor_add
            valid_toks = sum(x["valid_total"] for x in data)
            orig_toks = sum(x["original_tokens"] for x in data)
            
            pre_ret = pre_rep / max(valid_toks, 1)
            final_ret = final_sel / max(valid_toks, 1)
            
            print(f"{l:>5} | {pre_rep:>8} | {floor_req:>9} | {floor_add:>9} | {final_sel:>9} | {valid_toks:>10} | {orig_toks:>9} | {pre_ret:>8.1%} | {final_ret:>10.1%}")

    with open("threshold_sweep_results.json", "w") as f:
        json.dump(summary, f, indent=2)

def build_parser():
    parser = argparse.ArgumentParser(description="Diagnose AMMR soft versus hard retention")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-examples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threshold-biases", nargs='+', type=float,
                        default=[0.00, -0.05, -0.10, -0.15, -0.20, -0.25, -0.30, -0.35, -0.40, -0.45, -0.50, -0.55, -0.60, -0.65, -0.70, -0.75, -0.80, -0.85, -0.90, -0.95, -1.00])
    return parser

if __name__ == "__main__":
    run_diagnostic(build_parser().parse_args())
