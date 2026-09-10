import os
import argparse
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

from config import MODEL_NAME, DEVICE
from src.squad_data import get_squad_dataloaders
from src.models_adaptive import AdaptiveDistilBertQA
from src.baseline import BaselineQAModel
from src.utils import set_seed, print_header, save_checkpoint, load_checkpoint
from evaluate_squad import evaluate_model

# We will modify train_phase5 logic slightly here to support ablations
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from src.losses import calculate_distillation_loss, calculate_lagrangian_budget_loss, calculate_hidden_state_distillation_loss

def train_ablation(name, lambda_kd, lambda_h, max_train_samples=5000, epochs=5):
    set_seed(42)
    print_header(f"TRAINING ABLATION: {name}")
    
    batch_size = 2
    train_dl, _, _, _, _ = get_squad_dataloaders(batch_size=batch_size, max_train_samples=max_train_samples, max_val_samples=10)
    
    teacher = BaselineQAModel(freeze_parameters=True)
    teacher.qa_model.eval()
    
    student = AdaptiveDistilBertQA(model_name=MODEL_NAME, device=DEVICE, freeze_transformer=True)
    student.unfreeze_scorers()
    student.train()
    
    # Ablations use the same conservative scorer optimization as the main
    # training path; aggressive gate updates destabilize QA/KD objectives.
    optimizer = AdamW(student.retention_scorers.parameters(), lr=3e-4)
    total_steps = len(train_dl) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
    
    lagrangian_multiplier = 0.0
    lagrangian_lr = 0.05
    curriculum = [0.95, 0.90, 0.80, 0.70, 0.60]
    
    for epoch in range(epochs):
        target_ratio = curriculum[min(epoch, len(curriculum)-1)]
        progress_bar = tqdm(train_dl, desc=f"Epoch {epoch+1} [{name}]")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            start_target = batch['start_positions'].to(DEVICE)
            end_target = batch['end_positions'].to(DEVICE)
            
            target_budget_per_seq = 6 * input_ids.size(1) * target_ratio
            answer_span_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for batch_idx in range(input_ids.size(0)):
                start_idx = int(start_target[batch_idx].item())
                end_idx = int(end_target[batch_idx].item())
                if 0 <= start_idx < input_ids.size(1) and 0 <= end_idx < input_ids.size(1) and start_idx <= end_idx:
                    answer_span_mask[batch_idx, start_idx:end_idx + 1] = True
            
            with torch.no_grad():
                teacher_outputs = teacher.model(input_ids, attention_mask, output_hidden_states=True)
                t_start = teacher_outputs.start_logits
                t_end = teacher_outputs.end_logits
                t_hidden = teacher_outputs.hidden_states
                
            s_start, s_end, layer_metrics = student(
                input_ids,
                attention_mask,
                return_layer_metrics=True,
                training=True,
                minimum_retention_ratio=target_ratio,
                answer_span_mask=answer_span_mask
            )
            expected_retained = layer_metrics['expected_retained_tokens']
            
            qa_loss = (F.cross_entropy(s_start, start_target) + F.cross_entropy(s_end, end_target)) / 2
            
            logit_kd_loss = torch.tensor(0.0, device=DEVICE)
            if lambda_kd > 0:
                logit_kd_loss = calculate_distillation_loss(s_start, s_end, t_start, t_end)
                
            hidden_kd_loss = torch.tensor(0.0, device=DEVICE)
            if lambda_h > 0:
                for l_idx in range(6):
                    if layer_metrics['selection_results'][l_idx] is not None:
                        h_loss = calculate_hidden_state_distillation_loss(
                            layer_metrics['hidden_states'][l_idx], t_hidden[l_idx+1],
                            layer_metrics['selection_results'][l_idx].selected_indices,
                            layer_metrics['selection_results'][l_idx].new_attention_mask
                        )
                        hidden_kd_loss += h_loss
                        
            # Minimum-retention constraint: positive violation means the
            # student retained too few tokens.
            violation = target_budget_per_seq - expected_retained
            budget_loss = calculate_lagrangian_budget_loss(expected_retained, target_budget_per_seq, lagrangian_multiplier)
            total_loss = qa_loss + lambda_kd * logit_kd_loss + lambda_h * hidden_kd_loss + 1.0 * budget_loss

            losses_finite = all(torch.isfinite(value).item() for value in (qa_loss, logit_kd_loss, hidden_kd_loss, total_loss))
            losses_stable = all(value.detach().abs().item() <= 100.0 for value in (qa_loss, logit_kd_loss, hidden_kd_loss, total_loss))
            optimizer.zero_grad(set_to_none=True)
            if losses_finite and losses_stable and epoch > 0:
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(student.retention_scorers.parameters(), 0.5)
                if torch.isfinite(grad_norm):
                    optimizer.step()
                    # Optimizer update always precedes scheduler update.
                    scheduler.step()
                else:
                    optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                if losses_finite and losses_stable:
                    lagrangian_multiplier = max(0.0, lagrangian_multiplier + lagrangian_lr * violation.item())
                
    save_checkpoint(student, optimizer=optimizer, step=total_steps, checkpoint_path=f"ablation_{name}.pt", config={"name": name, "lambda_kd": lambda_kd, "lambda_h": lambda_h}, epochs=epochs)
    del teacher, student, optimizer, scheduler
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def measure_retention(model, calibration_dl, bias):
    total_ret, total_orig = 0, 0
    for batch in calibration_dl:
        with torch.no_grad():
            _, _, metrics = model(
                batch['input_ids'].to(DEVICE), 
                batch['attention_mask'].to(DEVICE), 
                return_layer_metrics=True, training=False, threshold_bias=bias
            )
            for res in metrics['selection_results']:
                if res is not None:
                    total_ret += res.num_selected
                    total_orig += res.num_original
    return total_ret / max(1, total_orig)

def calibrate_threshold(model, calibration_dl, target_ratio: float, tolerance: float = 0.005) -> float:
    """
    Robust global threshold calibration procedure.
    Uses actual gate probabilities to form an initial estimate, then performs a monotonic search.
    """
    print(f"Calibrating threshold for target retention {target_ratio*100:.1f}%...")
    model.eval()
    
    # 1. Collect all raw logits
    all_logits = []
    total_tokens = 0
    total_protected = 0
    
    for batch in calibration_dl:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        with torch.no_grad():
            _, _, metrics = model(
                input_ids, attention_mask, return_layer_metrics=True, training=False, threshold_bias=15.0
            )
            special_tokens_mask = (input_ids == 101) | (input_ids == 102)
            padding_mask = attention_mask < 0.5
            
            # This valid mask starts at length 512
            current_valid_mask = (~padding_mask) & (~special_tokens_mask)
            
            for res in metrics['selection_results']:
                if res is not None:
                    # res.retention_scores shape matches current_valid_mask exactly at each layer
                    all_logits.append(res.retention_scores[current_valid_mask].cpu())
                    total_tokens += current_valid_mask.sum().item()
                    total_protected += (special_tokens_mask & ~padding_mask).sum().item()
                    
                    # Update masks for the next layer (physically compacted)
                    current_valid_mask = torch.gather(current_valid_mask, 1, res.selected_indices)
                    special_tokens_mask = torch.gather(special_tokens_mask, 1, res.selected_indices)
                    padding_mask = torch.gather(padding_mask, 1, res.selected_indices)
                    
    # 2. Compute initial analytical estimate
    all_logits = torch.cat(all_logits)
    target_retained = int(target_ratio * total_tokens)
    tokens_to_keep_from_valid = target_retained - total_protected
    
    if tokens_to_keep_from_valid <= 0:
        initial_bias = -15.0
    elif tokens_to_keep_from_valid >= len(all_logits):
        initial_bias = 15.0
    else:
        adjusted_ratio = tokens_to_keep_from_valid / len(all_logits)
        # Find the threshold logit
        l_th = torch.quantile(all_logits.float(), 1.0 - adjusted_ratio).item()
        # threshold_bias = -2.397895 - L_th
        initial_bias = -2.397895 - l_th
        
    print(f"  Initial analytical estimate: bias = {initial_bias:.3f}")
    
    # 3. Monotonic search around the estimate
    bias = initial_bias
    current_ratio = measure_retention(model, calibration_dl, bias)
    step = 0.5
    direction = 1 if current_ratio < target_ratio else -1
    
    for i in range(10):
        if abs(current_ratio - target_ratio) <= tolerance:
            break
            
        # If we crossed the target, halve step size and reverse
        if (current_ratio < target_ratio and direction == -1) or (current_ratio > target_ratio and direction == 1):
            step /= 2.0
            direction *= -1
            
        bias += direction * step
        current_ratio = measure_retention(model, calibration_dl, bias)
        
    print(f"  Final bias = {bias:.3f} (Achieved {current_ratio*100:.2f}%)")
    return bias


def main(test_mode=False):
    max_train = 50 if test_mode else 5000
    epochs = 1 if test_mode else 5
    max_val = 20 if test_mode else 1000  # Evaluate on same large subset!
    
    # Train ablations
    train_ablation("QA_Only", lambda_kd=0.0, lambda_h=0.0, max_train_samples=max_train, epochs=epochs)
    train_ablation("Logit_KD", lambda_kd=1.0, lambda_h=0.0, max_train_samples=max_train, epochs=epochs)
    train_ablation("Full_KD", lambda_kd=1.0, lambda_h=1.0, max_train_samples=max_train, epochs=epochs)
    
    # Evaluation
    print_header("PHASE 5 ABLATION EVALUATION")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _, val_dl, _, val_data, val_features = get_squad_dataloaders(batch_size=4, max_train_samples=10, max_val_samples=max_val)
    
    # Calibration set (use a small subset of val)
    _, cal_dl, _, _, _ = get_squad_dataloaders(batch_size=4, max_train_samples=10, max_val_samples=50)
    
    results = []
    
    # 1. Baseline
    baseline = BaselineQAModel(freeze_parameters=True)
    b_em, b_f1, b_lat, _, _, _, b_mem, _, _ = evaluate_model(baseline, val_dl, val_features, val_data, tokenizer, is_baseline=True)
    results.append(["Baseline", "100%", 100.0, b_em, b_f1, 0.0, 100.0, 0.0, b_mem, b_lat, [100.0]*6])
    del baseline
    
    # 2. AMMR Models
    models_to_eval = ["QA_Only", "Logit_KD", "Full_KD"]
    budgets = [0.90, 0.80, 0.70, 0.60, 0.50]
    
    for name in models_to_eval:
        ammr = AdaptiveDistilBertQA(model_name=MODEL_NAME, device=DEVICE)
        load_checkpoint(ammr, optimizer=None, checkpoint_path=f"ablation_{name}.pt")
        
        for b in budgets:
            print(f"\nEvaluating {name} at {b*100:.0f}% Budget")
            bias = calibrate_threshold(ammr, cal_dl, target_ratio=b)
            em, f1, lat, ret, cost, comp_red, mem, span_surv, _ = evaluate_model(ammr, val_dl, val_features, val_data, tokenizer, is_baseline=False, threshold_bias=bias)
            drop = b_f1 - f1
            results.append([name, f"{b*100:.0f}%", ret, em, f1, drop, cost, comp_red, mem, lat, span_surv])
            
        del ammr
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    # Print Table
    print_header("FINAL PARETO CURVE: ACCURACY-EFFICIENCY ABLATION")
    print(f"{'Model':<12} | {'Budget':<7} | {'Retained %':<10} | {'EM':<6} | {'F1':<6} | {'F1 Drop':<7} | {'Attn Cost':<10} | {'Attn Reduc':<10} | {'Peak Mem (MB)':<13} | {'Lat (ms)':<9} | {'Span Survival (L1-L6)':<40}")
    print("-" * 150)
    for res in results:
        m, b, ret, em, f1, drop, cost, red, mem, lat, surv = res
        surv_str = ", ".join([f"{s:.0f}%" for s in surv])
        print(f"{m:<12} | {b:<7} | {ret:<10.1f} | {em:<6.1f} | {f1:<6.1f} | {drop:<7.1f} | {cost:<10.1f} | {red:<10.1f} | {mem:<13.1f} | {lat:<9.1f} | [{surv_str}]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run in fast test mode (small data)")
    args = parser.parse_args()
    main(args.test)
