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

def evaluate_with_fresh_model(checkpoint, num_examples, batch_size, flag_name, bias=0.0, seed_val=None, target_counts_write=None):
    # Fresh dataloaders
    _, val_dl, _, val_data, val_features = get_squad_dataloaders(
        batch_size=batch_size,
        max_train_samples=1,
        max_val_samples=num_examples,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Fresh model
    model = AdaptiveDistilBertQA(model_name=MODEL_NAME, device=DEVICE).to(DEVICE)
    if os.path.exists(checkpoint):
        evaluate_squad.load_ammr_checkpoint(model, checkpoint)
    model.eval()
    
    total_valid = 0
    total_selected = 0
    layer_valid = [0] * 6
    layer_selected = [0] * 6
    
    if flag_name is None:
        model._diagnostic_target_counts_write = {}
        model._diagnostic_target_counts_read = None
        model._diagnostic_target_counts_consumed = None
    elif flag_name == "diagnostic_random_seed":
        model._diagnostic_target_counts_write = dict(target_counts_write)
        model._diagnostic_target_counts_read = dict(target_counts_write)
        model._diagnostic_target_counts_consumed = set()
    else:
        model._diagnostic_target_counts_read = None
        model._diagnostic_target_counts_consumed = None
        
    forward_count = 0
    original_forward = model.forward
    
    def new_forward(*args, **kwargs):
        nonlocal total_valid, total_selected, forward_count
        kwargs["diagnostic_batch_id"] = forward_count
        if flag_name:
            if flag_name == "diagnostic_random_seed":
                kwargs[flag_name] = seed_val
            else:
                kwargs[flag_name] = True
        res = original_forward(*args, **kwargs)
        if len(res) == 3 and res[2] is not None:
            results = res[2].get("selection_results", [])
            if len(results) > 0:
                # Add first layer valid tokens to total
                if results[0] is not None:
                    total_valid += int(results[0].actual_valid_counts.sum().item())
                # Add last layer selected tokens to total
                if results[-1] is not None:
                    total_selected += int(results[-1].actual_retained_counts.sum().item())
                
                # Per layer counts
                for i, result in enumerate(results):
                    if result is not None:
                        layer_valid[i] += int(result.actual_valid_counts.sum().item())
                        layer_selected[i] += int(result.actual_retained_counts.sum().item())
        forward_count += 1
        return res
        
    model.forward = new_forward
    res = evaluate_squad.evaluate_model(model, val_dl, val_features, val_data, tokenizer, is_baseline=False, threshold_bias=bias)
    model.forward = original_forward
    
    res["total_valid"] = total_valid
    res["total_selected"] = total_selected
    res["effective_retention"] = (total_selected / total_valid) * 100.0 if total_valid > 0 else 100.0
    res["layer_valid"] = layer_valid
    res["layer_selected"] = layer_selected
    
    if flag_name is None:
        res["target_counts"] = dict(model._diagnostic_target_counts_write)
        res["target_count_records_captured"] = len(model._diagnostic_target_counts_write)
    elif flag_name == "diagnostic_random_seed":
        initial = len(model._diagnostic_target_counts_write)
        consumed = len(model._diagnostic_target_counts_consumed)
        
        captured_keys = set(model._diagnostic_target_counts_write.keys())
        consumed_keys = model._diagnostic_target_counts_consumed
        missing_keys = captured_keys - consumed_keys
        extra_keys = consumed_keys - captured_keys
        
        res["target_count_records_consumed"] = consumed
        res["target_count_records_remaining"] = initial - consumed
        res["missing_records"] = len(missing_keys)
        res["extra_records"] = len(extra_keys)
        
        assert len(missing_keys) == 0, f"Random control did not consume all target counts! Remaining: {len(missing_keys)}"
        assert len(extra_keys) == 0, f"Random control consumed unknown target counts! Extra: {len(extra_keys)}"
        
    return res

import csv

def main():
    args = parse_args()
    
    if not os.path.exists(args.checkpoint):
        if os.path.exists("dummy.pt"):
            args.checkpoint = "dummy.pt"
            
    print("Evaluating Baseline...")
    _, val_dl, _, val_data, val_features = get_squad_dataloaders(
        batch_size=args.batch_size,
        max_train_samples=1,
        max_val_samples=args.num_examples,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    baseline = BaselineQAModel(freeze_parameters=True).to(DEVICE)
    baseline_res = evaluate_squad.evaluate_model(baseline, val_dl, val_features, val_data, tokenizer, is_baseline=True)
    del baseline
    torch.cuda.empty_cache()
    
    biases = [1.0, 0.5, 0.2, 0.0, -0.2, -0.4, -0.6]
    seeds = [42, 123, 2026]
    
    csv_filename = "multi_budget_random_control.csv"
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "bias",
            "AMMR_EM", "AMMR_F1", "AMMR_eff_ret", "AMMR_ans_surv",
            "Rand_mean_EM", "Rand_std_EM", "Rand_mean_F1", "Rand_std_F1", "Rand_eff_ret"
        ])
        
        for bias in biases:
            print(f"\n======================================")
            print(f"Evaluating AMMR Learned Selection (bias={bias})...")
            res_ammr = evaluate_with_fresh_model(args.checkpoint, args.num_examples, args.batch_size, None, bias=bias)
            
            assert res_ammr.get("target_counts") is not None, "AMMR target_counts is None. Target-count capture failed."
            assert len(res_ammr["target_counts"]) > 0, f"Captured 0 target-count records!"
            
            random_results = []
            target_counts = res_ammr["target_counts"]
            
            for seed in seeds:
                print(f"  Evaluating Random Matched Selection (seed={seed})...")
                res_rand = evaluate_with_fresh_model(args.checkpoint, args.num_examples, args.batch_size, "diagnostic_random_seed", bias=bias, seed_val=seed, target_counts_write=target_counts)
                
                # Verify matched retention
                assert res_rand['total_selected'] == res_ammr['total_selected'], \
                    f"Random selection didn't match count! {res_rand['total_selected']} vs {res_ammr['total_selected']}"
                    
                random_results.append(res_rand)
                
            rand_ems = [r['em'] for r in random_results]
            rand_f1s = [r['f1'] for r in random_results]
            
            print(f"\nINVARIANT CHECK for bias={bias}:")
            print(f"  target_count_records_captured: {res_ammr.get('target_count_records_captured', 'N/A')}")
            print(f"  target_count_records_consumed: {random_results[0].get('target_count_records_consumed', 'N/A')}")
            print(f"  missing_records: {random_results[0].get('missing_records', 'N/A')}")
            print(f"  extra_records: {random_results[0].get('extra_records', 'N/A')}")
            print(f"  Mismatch count (tokens): {abs(res_ammr['total_selected'] - random_results[0]['total_selected'])}")
            
            print(f"\nRESULTS for bias={bias}:")
            print(f"  AMMR EM={res_ammr['em']:.2f}, F1={res_ammr['f1']:.2f}, EffRet={res_ammr['effective_retention']:.2f}%")
            print(f"  Rand mean EM={np.mean(rand_ems):.2f}, std EM={np.std(rand_ems):.2f}")
            print(f"  Rand mean F1={np.mean(rand_f1s):.2f}, std F1={np.std(rand_f1s):.2f}")
            
            writer.writerow([
                f"{bias:.1f}",
                f"{res_ammr['em']:.2f}",
                f"{res_ammr['f1']:.2f}",
                f"{res_ammr['effective_retention']:.2f}",
                f"{res_ammr['answer_survival']:.2f}",
                f"{np.mean(rand_ems):.2f}",
                f"{np.std(rand_ems):.2f}",
                f"{np.mean(rand_f1s):.2f}",
                f"{np.std(rand_f1s):.2f}",
                f"{random_results[0]['effective_retention']:.2f}"
            ])
            
    print(f"\nFinished. Results exported to {csv_filename}")

if __name__ == "__main__":
    main()
