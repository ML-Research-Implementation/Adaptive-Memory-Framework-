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
    seeds = [42, 123, 2026, 7, 19, 37, 101, 256, 512, 999]
    
    all_results = []
    
    for bias in biases:
        print(f"\n======================================")
        print(f"Evaluating AMMR Learned Selection (bias={bias})...")
        try:
            res_ammr = evaluate_with_fresh_model(args.checkpoint, args.num_examples, args.batch_size, None, bias=bias)
            
            assert res_ammr.get("target_counts") is not None, "AMMR target_counts is None. Target-count capture failed."
            assert len(res_ammr["target_counts"]) > 0, f"Captured 0 target-count records!"
            
            random_results = []
            target_counts = res_ammr["target_counts"]
            captured = res_ammr.get('target_count_records_captured', 0)
            
            for seed in seeds:
                print(f"  Evaluating Random Matched Selection (seed={seed})...")
                res_rand = evaluate_with_fresh_model(args.checkpoint, args.num_examples, args.batch_size, "diagnostic_random_seed", bias=bias, seed_val=seed, target_counts_write=target_counts)
                
                # Verify matched retention
                if res_rand['total_selected'] != res_ammr['total_selected']:
                    raise AssertionError(f"Count Mismatch! Random selected {res_rand['total_selected']} != AMMR selected {res_ammr['total_selected']}")
                if res_rand['missing_records'] > 0:
                    raise AssertionError(f"Mismatch! missing_records={res_rand['missing_records']}")
                if res_rand['extra_records'] > 0:
                    raise AssertionError(f"Mismatch! extra_records={res_rand['extra_records']}")
                    
                random_results.append(res_rand)
                
            rand_ems = [r['em'] for r in random_results]
            rand_f1s = [r['f1'] for r in random_results]
            rand_ans_survs = [r['answer_survival'] for r in random_results]
            
            mean_em = float(np.mean(rand_ems))
            std_em = float(np.std(rand_ems))
            mean_f1 = float(np.mean(rand_f1s))
            std_f1 = float(np.std(rand_f1s))
            mean_ans_surv = float(np.mean(rand_ans_survs))
            em_diff = float(res_ammr['em']) - mean_em
            f1_diff = float(res_ammr['f1']) - mean_f1
            
            print(f"\nINVARIANT CHECK for bias={bias}:")
            print(f"  target_count_records_captured: {captured}")
            print(f"  target_count_records_consumed: {random_results[0].get('target_count_records_consumed', 'N/A')}")
            print(f"  missing_records: 0")
            print(f"  extra_records: 0")
            print(f"  Mismatch count (tokens): 0")
            
            print(f"\nRESULTS for bias={bias}:")
            print(f"  AMMR EM={res_ammr['em']:.2f}, F1={res_ammr['f1']:.2f}, EffRet={res_ammr['effective_retention']:.2f}%")
            print(f"  Rand mean EM={mean_em:.2f}, std EM={std_em:.2f}")
            print(f"  Rand mean F1={mean_f1:.2f}, std F1={std_f1:.2f}")
            print(f"  EM diff={em_diff:.2f}, F1 diff={f1_diff:.2f}")
            
            row_dict = {
                "bias": bias,
                "AMMR_EM": res_ammr['em'],
                "AMMR_F1": res_ammr['f1'],
                "AMMR_eff_ret": res_ammr['effective_retention'],
                "AMMR_ans_surv": res_ammr['answer_survival'],
                "Rand_mean_EM": mean_em,
                "Rand_std_EM": std_em,
                "Rand_mean_F1": mean_f1,
                "Rand_std_F1": std_f1,
                "Rand_eff_ret": random_results[0]['effective_retention'],
                "Rand_ans_surv": mean_ans_surv,
                "EM_diff": em_diff,
                "F1_diff": f1_diff
            }
            for i, seed in enumerate(seeds):
                row_dict[f"Rand_EM_seed_{seed}"] = rand_ems[i]
                row_dict[f"Rand_F1_seed_{seed}"] = rand_f1s[i]
            all_results.append(row_dict)
            
        except Exception as e:
            consumed = 0
            if 'res_rand' in locals() and res_rand is not None:
                consumed = res_rand.get('target_count_records_consumed', 0)
            raise RuntimeError(
                f"Evaluation failed for bias={bias}.\n"
                f"Stage: AMMR/Random execution\n"
                f"Exception: {e}\n"
                f"Target records captured: {captured if 'captured' in locals() else 'Unknown'}\n"
                f"Target records consumed: {consumed}"
            ) from e

    # Assert exactly 7 results
    assert len(all_results) == 7, f"Expected 7 results, got {len(all_results)}"
    
    # Assert exact bias set
    produced = {round(float(row["bias"]), 1) for row in all_results}
    expected = {1.0, 0.5, 0.2, 0.0, -0.2, -0.4, -0.6}
    assert produced == expected, f"Produced biases {produced} do not match expected {expected}"
    
    # Sort descending
    all_results.sort(key=lambda x: x["bias"], reverse=True)
    
    csv_filename = "multi_budget_random_control.csv"
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        headers = [
            "bias", "AMMR_EM", "AMMR_F1", "AMMR_eff_ret", "AMMR_ans_surv"
        ]
        for seed in seeds:
            headers.append(f"Rand_EM_seed_{seed}")
        for seed in seeds:
            headers.append(f"Rand_F1_seed_{seed}")
        headers.extend([
            "Rand_mean_EM", "Rand_std_EM", "Rand_mean_F1", "Rand_std_F1", 
            "Rand_eff_ret", "Rand_ans_surv", "EM_diff", "F1_diff"
        ])
        writer.writerow(headers)
        
        for row in all_results:
            row_vals = [
                f"{row['bias']:.1f}",
                f"{row['AMMR_EM']:.2f}",
                f"{row['AMMR_F1']:.2f}",
                f"{row['AMMR_eff_ret']:.2f}",
                f"{row['AMMR_ans_surv']:.2f}"
            ]
            for seed in seeds:
                row_vals.append(f"{row[f'Rand_EM_seed_{seed}']:.2f}")
            for seed in seeds:
                row_vals.append(f"{row[f'Rand_F1_seed_{seed}']:.2f}")
            row_vals.extend([
                f"{row['Rand_mean_EM']:.2f}",
                f"{row['Rand_std_EM']:.2f}",
                f"{row['Rand_mean_F1']:.2f}",
                f"{row['Rand_std_F1']:.2f}",
                f"{row['Rand_eff_ret']:.2f}",
                f"{row['Rand_ans_surv']:.2f}",
                f"{row['EM_diff']:.2f}",
                f"{row['F1_diff']:.2f}"
            ])
            writer.writerow(row_vals)
            
    print(f"\n======================================")
    print("COMPLETE MULTI-BUDGET SWEEP")
    print(f"rows: {len(all_results)}")
    print(f"biases: {[round(r['bias'], 1) for r in all_results]}")
    print("missing: []")
    print("duplicate_biases: []")
    print(f"CSV: {csv_filename}")

if __name__ == "__main__":
    main()
