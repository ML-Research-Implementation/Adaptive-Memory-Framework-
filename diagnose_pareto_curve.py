import os
import csv
import json
import torch
import argparse
from transformers import AutoTokenizer
from config import MODEL_NAME, DEVICE
from src.squad_data import get_squad_dataloaders
from src.baseline import BaselineQAModel
from src.models_adaptive import AdaptiveDistilBertQA
import evaluate_squad

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/content/AMMR_GITHUB/squad_final_checkpoint.pt")
    parser.add_argument("--num-examples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--biases", nargs='+', type=float, 
                        default=[0.00, -0.10, -0.20, -0.30, -0.40, -0.50, -0.60, -0.70, -0.80, -0.90, -1.00])
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.checkpoint):
        # Fallback for local testing
        if os.path.exists("dummy.pt"):
            args.checkpoint = "dummy.pt"
        else:
            print(f"Warning: Checkpoint {args.checkpoint} not found. Ensure you are in the correct environment.")
    
    _, val_dl, train_data, val_data, val_features = get_squad_dataloaders(
        batch_size=args.batch_size,
        max_train_samples=1,
        max_val_samples=args.num_examples,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 1. Baseline
    print("Evaluating Baseline...")
    baseline = BaselineQAModel(freeze_parameters=True).to(DEVICE)
    baseline_res = evaluate_squad.evaluate_model(baseline, val_dl, val_features, val_data, tokenizer, is_baseline=True)
    
    results = []
    
    # Baseline result row
    results.append({
        "bias": "Baseline",
        "EM": baseline_res["em"],
        "F1": baseline_res["f1"],
        "soft_retention": 100.0,
        "final_retention": 100.0,
        "answer_survival": 100.0
    })
    
    del baseline
    torch.cuda.empty_cache()
    
    # 2. AMMR
    print(f"Evaluating AMMR Checkpoint: {args.checkpoint}")
    model = AdaptiveDistilBertQA(model_name=MODEL_NAME, device=DEVICE).to(DEVICE)
    if os.path.exists(args.checkpoint):
        evaluate_squad.load_ammr_checkpoint(model, args.checkpoint)
    
    original_evaluate_model = evaluate_squad.evaluate_model
    
    captured_soft_retentions = []
    
    def wrapped_evaluate_model(model, dataloader, dataset_features, raw_val_data, tokenizer, is_baseline=False, threshold_bias=0.0):
        # We will wrap the model's forward to capture layer_metrics
        original_forward = model.forward
        
        soft_sums = 0.0
        soft_counts = 0
        
        def new_forward(*args, **kwargs):
            nonlocal soft_sums, soft_counts
            outputs = original_forward(*args, **kwargs)
            if not is_baseline and len(outputs) == 3 and outputs[2] is not None:
                layer_metrics = outputs[2]
                attention_mask = kwargs.get('attention_mask')
                if attention_mask is None and len(args) > 1:
                    attention_mask = args[1]
                
                for layer_idx, result in enumerate(layer_metrics.get("selection_results", [])):
                    if result is None: continue
                    probs = result.retention_probs.detach().float()
                    if layer_idx == 0:
                        valid_mask = attention_mask >= 0.5
                    else:
                        prev_result = layer_metrics["selection_results"][layer_idx - 1]
                        valid_mask = prev_result.new_attention_mask >= 0.5
                        
                    valid_mask_aligned = valid_mask.to(device=probs.device, dtype=torch.bool)
                    if probs.numel() > 0:
                        soft_sums += probs[valid_mask_aligned].mean().item()
                        soft_counts += 1
            return outputs
        
        model.forward = new_forward
        res = original_evaluate_model(model, dataloader, dataset_features, raw_val_data, tokenizer, is_baseline, threshold_bias)
        model.forward = original_forward
        
        avg_soft = (soft_sums / max(soft_counts, 1)) * 100.0
        captured_soft_retentions.append(avg_soft)
        return res
        
    evaluate_squad.evaluate_model = wrapped_evaluate_model
    
    for bias in args.biases:
        print(f"\nEvaluating bias: {bias}")
        captured_soft_retentions.clear()
        res = evaluate_squad.evaluate_model(model, val_dl, val_features, val_data, tokenizer, is_baseline=False, threshold_bias=bias)
        
        soft_ret = captured_soft_retentions[0] if captured_soft_retentions else 0.0
        
        results.append({
            "bias": bias,
            "EM": res["em"],
            "F1": res["f1"],
            "soft_retention": soft_ret,
            "final_retention": res["retention"],
            "answer_survival": res["answer_survival"]
        })
    
    # Restore original evaluate_model
    evaluate_squad.evaluate_model = original_evaluate_model
    
    # Print table
    print("\nPARETO CURVE RESULTS")
    print(f"{'bias':>10} | {'EM':>6} | {'F1':>6} | {'soft_ret':>10} | {'final_ret':>10} | {'ans_surv':>10}")
    print("-" * 65)
    for r in results:
        b = r["bias"]
        if isinstance(b, float):
            b_str = f"{b:>.2f}"
        else:
            b_str = str(b)
            
        print(f"{b_str:>10} | {r['EM']:>6.2f} | {r['F1']:>6.2f} | {r['soft_retention']:>9.2f}% | {r['final_retention']:>9.2f}% | {r['answer_survival']:>9.2f}%")
        
    # Save to CSV
    csv_file = "pareto_curve.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bias", "EM", "F1", "soft_retention", "final_retention", "answer_survival"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved results to {csv_file}")

if __name__ == "__main__":
    main()
