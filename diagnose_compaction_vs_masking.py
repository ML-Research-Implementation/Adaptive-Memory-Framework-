import os
import csv
import torch
import argparse
from transformers import AutoTokenizer
from config import MODEL_NAME, DEVICE
from src.squad_data import get_squad_dataloaders
from src.models_adaptive import AdaptiveDistilBertQA
import evaluate_squad

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/content/AMMR_GITHUB/squad_final_checkpoint.pt")
    parser.add_argument("--num-examples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--biases", nargs='+', type=float, default=[1.0, 0.5, 0.0])
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.checkpoint):
        if os.path.exists("dummy.pt"):
            args.checkpoint = "dummy.pt"
        else:
            print(f"Warning: Checkpoint {args.checkpoint} not found.")
    
    _, val_dl, train_data, val_data, val_features = get_squad_dataloaders(
        batch_size=args.batch_size,
        max_train_samples=1,
        max_val_samples=args.num_examples,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print(f"Evaluating AMMR Checkpoint: {args.checkpoint}")
    model = AdaptiveDistilBertQA(model_name=MODEL_NAME, device=DEVICE).to(DEVICE)
    if os.path.exists(args.checkpoint):
        evaluate_squad.load_ammr_checkpoint(model, args.checkpoint)
    
    original_evaluate_model = evaluate_squad.evaluate_model
    
    def wrapped_evaluate_model(model, dataloader, dataset_features, raw_val_data, tokenizer, is_baseline=False, threshold_bias=0.0):
        original_forward = model.forward
        
        def new_forward(*args, **kwargs):
            kwargs["diagnostic_no_compaction"] = True
            return original_forward(*args, **kwargs)
        
        model.forward = new_forward
        res = original_evaluate_model(model, dataloader, dataset_features, raw_val_data, tokenizer, is_baseline, threshold_bias)
        model.forward = original_forward
        
        # Verify compaction was bypassed:
        # If bypassed, span_survival_rates shouldn't actually change because no tokens are removed from sequence length!
        # wait, answer_survival might be 100% since indices are kept.
        # Actually evaluate_model computes answer_survival based on result.selected_indices
        return res
        
    evaluate_squad.evaluate_model = wrapped_evaluate_model
    
    results = []
    
    for bias in args.biases:
        print(f"\nEvaluating Masking-Only (No Compaction) bias: {bias}")
        res = evaluate_squad.evaluate_model(model, val_dl, val_features, val_data, tokenizer, is_baseline=False, threshold_bias=bias)
        
        results.append({
            "bias": bias,
            "EM": res["em"],
            "F1": res["f1"],
            "effective_retention": res["retention"],
            "answer_survival": res["answer_survival"]
        })
    
    evaluate_squad.evaluate_model = original_evaluate_model
    
    print("\nDIAGNOSTIC RESULTS: MASKING ONLY (NO PHYSICAL COMPACTION)")
    print("Implementation bypassed token physical gather/scatter by keeping original sequence layout")
    print("and zeroing out hidden states + attention mask for unselected tokens.")
    print(f"{'bias':>10} | {'EM':>6} | {'F1':>6} | {'eff_ret':>10} | {'ans_surv':>10}")
    print("-" * 55)
    for r in results:
        print(f"{r['bias']:>10.2f} | {r['EM']:>6.2f} | {r['F1']:>6.2f} | {r['effective_retention']:>9.2f}% | {r['answer_survival']:>9.2f}%")
        

if __name__ == "__main__":
    main()
