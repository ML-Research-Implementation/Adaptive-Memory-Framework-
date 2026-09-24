import os
import argparse
import torch
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

def evaluate_with_flag(model, val_dl, val_features, val_data, tokenizer, flag_name, bias=0.0):
    original_forward = model.forward
    
    total_valid = 0
    total_selected = 0
    
    def new_forward(*args, **kwargs):
        nonlocal total_valid, total_selected
        if flag_name:
            kwargs[flag_name] = True
        res = original_forward(*args, **kwargs)
        if len(res) == 3 and res[2] is not None:
            for result in res[2].get("selection_results", []):
                if result is not None:
                    # Assert forced-all-retain selected_count == valid_token_count
                    assert int(result.actual_retained_counts.sum().item()) == int(result.actual_valid_counts.sum().item()), \
                        f"Mismatch at layer: {result.actual_retained_counts.sum().item()} vs {result.actual_valid_counts.sum().item()}"
                    
                    total_valid += int(result.actual_valid_counts.sum().item())
                    total_selected += int(result.actual_retained_counts.sum().item())
                    
        return res
        
    model.forward = new_forward
    res = evaluate_squad.evaluate_model(model, val_dl, val_features, val_data, tokenizer, is_baseline=False, threshold_bias=bias)
    model.forward = original_forward
    
    res["total_valid"] = total_valid
    res["total_selected"] = total_selected
    res["final_retention_ratio"] = (total_selected / total_valid) * 100.0 if total_valid > 0 else 100.0
    
    return res


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
    
    # Baseline
    print("Evaluating Baseline...")
    baseline = BaselineQAModel(freeze_parameters=True).to(DEVICE)
    baseline_res = evaluate_squad.evaluate_model(baseline, val_dl, val_features, val_data, tokenizer, is_baseline=True)
    del baseline
    torch.cuda.empty_cache()
    
    # AMMR Checkpoint
    model = AdaptiveDistilBertQA(model_name=MODEL_NAME, device=DEVICE).to(DEVICE)
    if os.path.exists(args.checkpoint):
        evaluate_squad.load_ammr_checkpoint(model, args.checkpoint)
        
    print("\nEvaluating Forced All-Retain (diagnostic_force_all_retain=True)...")
    res_forced = evaluate_with_flag(model, val_dl, val_features, val_data, tokenizer, "diagnostic_force_all_retain", bias=0.0)
    
    print("\nEvaluating Permissive Bias (bias=+100.0)...")
    res_permissive = evaluate_with_flag(model, val_dl, val_features, val_data, tokenizer, None, bias=100.0)

    print("\nDIAGNOSTIC RESULTS: FULL-RETENTION CONTROL")
    print("-" * 75)
    print(f"{'Condition':>20} | {'EM':>6} | {'F1':>6} | {'valid_toks':>10} | {'sel_toks':>10} | {'ans_surv':>9}")
    print("-" * 75)
    
    print(f"{'Baseline':>20} | {baseline_res['em']:>6.2f} | {baseline_res['f1']:>6.2f} | {'-':>10} | {'-':>10} | {'100.00%':>9}")
    
    print(f"{'Forced All-Retain':>20} | {res_forced['em']:>6.2f} | {res_forced['f1']:>6.2f} | {res_forced['total_valid']:>10} | {res_forced['total_selected']:>10} | {res_forced['answer_survival']:>8.2f}%")
    
    print(f"{'Permissive (+100)':>20} | {res_permissive['em']:>6.2f} | {res_permissive['f1']:>6.2f} | {res_permissive['total_valid']:>10} | {res_permissive['total_selected']:>10} | {res_permissive['answer_survival']:>8.2f}%")


if __name__ == "__main__":
    main()
