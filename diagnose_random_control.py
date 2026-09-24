import os
import argparse
import torch
import numpy as np
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
    return parser.parse_args()

def evaluate_with_flag(model, val_dl, val_features, val_data, tokenizer, flag_name, bias=0.0, seed_val=None):
    original_forward = model.forward
    
    total_valid = 0
    total_selected = 0
    
    if flag_name is None:
        model._diagnostic_target_counts_write = []
        model._diagnostic_target_counts_read = None
    elif flag_name == "diagnostic_random_seed":
        model._diagnostic_target_counts_read = list(model._diagnostic_target_counts_write) if hasattr(model, "_diagnostic_target_counts_write") else []
        model._diagnostic_target_counts_write = None
    else:
        model._diagnostic_target_counts_write = None
        model._diagnostic_target_counts_read = None
        
    def new_forward(*args, **kwargs):
        nonlocal total_valid, total_selected
        if flag_name:
            if flag_name == "diagnostic_random_seed":
                kwargs[flag_name] = seed_val
            else:
                kwargs[flag_name] = True
        res = original_forward(*args, **kwargs)
        if len(res) == 3 and res[2] is not None:
            for result in res[2].get("selection_results", []):
                if result is not None:
                    total_valid += int(result.actual_valid_counts.sum().item())
                    total_selected += int(result.actual_retained_counts.sum().item())
        return res
        
    model.forward = new_forward
    res = evaluate_squad.evaluate_model(model, val_dl, val_features, val_data, tokenizer, is_baseline=False, threshold_bias=bias)
    model.forward = original_forward
    
    res["total_valid"] = total_valid
    res["total_selected"] = total_selected
    res["effective_retention"] = (total_selected / total_valid) * 100.0 if total_valid > 0 else 100.0
    
    return res

def main():
    args = parse_args()
    
    if not os.path.exists(args.checkpoint):
        if os.path.exists("dummy.pt"):
            args.checkpoint = "dummy.pt"
            
    _, val_dl, train_data, val_data, val_features = get_squad_dataloaders(
        batch_size=args.batch_size,
        max_train_samples=1,
        max_val_samples=args.num_examples,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print("Evaluating Baseline...")
    baseline = BaselineQAModel(freeze_parameters=True).to(DEVICE)
    baseline_res = evaluate_squad.evaluate_model(baseline, val_dl, val_features, val_data, tokenizer, is_baseline=True)
    del baseline
    torch.cuda.empty_cache()
    
    model = AdaptiveDistilBertQA(model_name=MODEL_NAME, device=DEVICE).to(DEVICE)
    if os.path.exists(args.checkpoint):
        evaluate_squad.load_ammr_checkpoint(model, args.checkpoint)
        
    print("\nEvaluating AMMR Learned Selection (bias=0.0)...")
    res_ammr = evaluate_with_flag(model, val_dl, val_features, val_data, tokenizer, None, bias=0.0)
    
    print("\nEvaluating Forced All-Retain (diagnostic_force_all_retain=True)...")
    res_forced = evaluate_with_flag(model, val_dl, val_features, val_data, tokenizer, "diagnostic_force_all_retain", bias=0.0)
    
    seeds = [42, 123, 2026]
    random_results = []
    
    for seed in seeds:
        print(f"\nEvaluating Random Matched Selection (seed={seed})...")
        res_rand = evaluate_with_flag(model, val_dl, val_features, val_data, tokenizer, "diagnostic_random_seed", bias=0.0, seed_val=seed)
        
        # Verify matched retention
        print(f"  [Invariant] target_selected={res_ammr['total_selected']} random_selected={res_rand['total_selected']}")
        assert res_rand['total_selected'] == res_ammr['total_selected'], \
            f"Random selection didn't match count! {res_rand['total_selected']} vs {res_ammr['total_selected']}"
            
        random_results.append(res_rand)
        
    rand_ems = [r['em'] for r in random_results]
    rand_f1s = [r['f1'] for r in random_results]
    
    print("\n" + "="*85)
    print("INVARIANT CHECK:")
    print(f"Production total selected: {res_ammr['total_selected']}")
    print(f"Random total selected (seed {seeds[0]}): {random_results[0]['total_selected']}")
    print(f"Mismatch count: {abs(res_ammr['total_selected'] - random_results[0]['total_selected'])}")
    print("="*85)
    
    print("\nDIAGNOSTIC RESULTS: MATCHED-RETENTION RANDOM CONTROL")
    print("-" * 85)
    print(f"{'Condition':>25} | {'EM':>6} | {'F1':>6} | {'eff_ret':>8} | {'ans_surv':>9}")
    print("-" * 85)
    print(f"{'Baseline':>25} | {baseline_res['em']:>6.2f} | {baseline_res['f1']:>6.2f} | {'100.00%':>8} | {'100.00%':>9}")
    print(f"{'Forced All-Retain':>25} | {res_forced['em']:>6.2f} | {res_forced['f1']:>6.2f} | {res_forced['effective_retention']:>7.2f}% | {res_forced['answer_survival']:>8.2f}%")
    print(f"{'AMMR Learned (bias=0.0)':>25} | {res_ammr['em']:>6.2f} | {res_ammr['f1']:>6.2f} | {res_ammr['effective_retention']:>7.2f}% | {res_ammr['answer_survival']:>8.2f}%")
    print(f"{'Random Matched (mean)':>25} | {np.mean(rand_ems):>6.2f} | {np.mean(rand_f1s):>6.2f} | {random_results[0]['effective_retention']:>7.2f}% | {np.mean([r['answer_survival'] for r in random_results]):>8.2f}%")
    print(f"{'Random (std dev)':>25} | {np.std(rand_ems):>6.2f} | {np.std(rand_f1s):>6.2f} | {'-':>8} | {'-':>9}")

if __name__ == "__main__":
    main()
