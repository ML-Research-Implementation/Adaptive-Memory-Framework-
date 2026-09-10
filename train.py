import os
import time
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from config import MODEL_NAME, DEVICE
from src.models_adaptive import AdaptiveDistilBertQA
from src.baseline import BaselineQAModel
from src.squad_data import get_squad_dataloaders
from src.losses import (
    calculate_distillation_loss,
    calculate_lagrangian_budget_loss,
    calculate_hidden_state_distillation_loss
)
from src.utils import set_seed, print_header, save_checkpoint

def evaluate(student, val_dl):
    student.eval()
    total_loss = 0
    total_em = 0
    total_f1 = 0
    count = 0
    
    with torch.no_grad():
        for batch in tqdm(val_dl, desc="Validating"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            start_target = batch['start_positions'].to(DEVICE)
            end_target = batch['end_positions'].to(DEVICE)
            
            s_start, s_end = student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_layer_metrics=False,
                training=False
            )
            
            loss_start = F.cross_entropy(s_start, start_target)
            loss_end = F.cross_entropy(s_end, end_target)
            qa_loss = (loss_start + loss_end) / 2
            total_loss += qa_loss.item()
            
            pred_start = torch.argmax(s_start, dim=1)
            pred_end = torch.argmax(s_end, dim=1)
            
            for i in range(input_ids.size(0)):
                s_p, e_p = pred_start[i].item(), pred_end[i].item()
                s_t, e_t = start_target[i].item(), end_target[i].item()
                
                em = int(s_p == s_t and e_p == e_t)
                total_em += em
                
                if s_p <= e_p and s_t <= e_t:
                    overlap = max(0, min(e_p, e_t) - max(s_p, s_t) + 1)
                    pred_len = e_p - s_p + 1
                    true_len = e_t - s_t + 1
                    if overlap == 0:
                        f1 = 0.0
                    else:
                        prec = overlap / pred_len
                        rec = overlap / true_len
                        f1 = 2 * prec * rec / (prec + rec)
                else:
                    f1 = 0.0
                total_f1 += f1
                count += 1
                
    student.train()
    if count == 0:
        return 0, 0, 0
    return total_loss / len(val_dl), (total_em / count) * 100, (total_f1 / count) * 100

def train(args):
    set_seed(42)
    print_header("STABLE TASK-PRESERVING AMMR TRAINING")
    
    train_dl, val_dl, train_data, val_data, val_features = get_squad_dataloaders(
        batch_size=args.batch_size, 
        max_train_samples=args.max_train_samples, 
        max_val_samples=args.max_val_samples
    )
    
    print("\nInitializing Teacher (Frozen DistilBERT) and Student (AMMR)...")
    teacher = BaselineQAModel(freeze_parameters=True)
    teacher.qa_model.eval()
    
    student = AdaptiveDistilBertQA(
        model_name=MODEL_NAME, 
        device=DEVICE,
        freeze_transformer=True
    )
    student.unfreeze_scorers()
    student.train()
    
    optimizer = AdamW(student.retention_scorers.parameters(), lr=args.learning_rate)
    
    total_steps = len(train_dl) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    lagrangian_multiplier = 0.0
    lagrangian_lr = 0.05
    
    start_epoch = 0
    global_step = 0
    
    use_amp = (DEVICE.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    
    if args.resume_from and os.path.exists(args.resume_from):
        from src.utils import load_checkpoint
        print(f"Loading checkpoint from {args.resume_from}")
        checkpoint = load_checkpoint(student.retention_scorers, optimizer, args.resume_from, scheduler=scheduler)
        global_step = checkpoint.get('step', 0)
        start_epoch = checkpoint.get('epoch', 0)
        lagrangian_multiplier = checkpoint.get('lagrangian_multiplier', 0.0)
        print(f"Resuming training from epoch {start_epoch}, step {global_step}")

    curriculum = [0.95, 0.90, 0.80, 0.70, 0.60]
    
    lambda_kd = 1.0       
    lambda_h = 1.0        
    lambda_b = 1.0        
    
    print(f"Starting training for {args.epochs} epochs ({total_steps} steps).")
    
    best_val_score = 0.0
    
    for epoch in range(start_epoch, args.epochs):
        target_ratio = curriculum[min(epoch, len(curriculum)-1)]
        print(f"\n[Epoch {epoch+1}/{args.epochs}] Curriculum Target: {target_ratio*100:.1f}%")
        
        progress_bar = tqdm(train_dl, desc=f"Epoch {epoch+1}")
        
        epoch_retention = 0.0
        epoch_target = 0.0
        epoch_violation = 0.0
        num_batches = 0
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            start_target = batch['start_positions'].to(DEVICE)
            end_target = batch['end_positions'].to(DEVICE)
            
            # target_budget must be in the same units as expected_retained_tokens.
            # expected_retained_tokens accumulates mean-per-sequence values, one per
            # active layer (6 total).  So the target must be:
            #   avg_valid_per_seq  (mean valid tokens/seq in this batch)
            #   × target_ratio     (fraction to retain)
            #   × num_active_layers (6)
            batch_size_cur = input_ids.size(0)
            avg_valid_per_seq = attention_mask.sum().item() / batch_size_cur
            # Number of layers that apply retention (all 6 by default).
            n_active_layers = sum(student.apply_retention_per_layer)
            target_budget = avg_valid_per_seq * target_ratio * n_active_layers
            
            with torch.amp.autocast(device_type=DEVICE.type, enabled=use_amp):
                with torch.no_grad():
                    teacher_outputs = teacher.model(
                        input_ids, 
                        attention_mask, 
                        output_hidden_states=True
                    )
                    t_start = teacher_outputs.start_logits
                    t_end = teacher_outputs.end_logits
                    teacher_hidden_states = teacher_outputs.hidden_states
                    
                s_start, s_end, layer_metrics = student(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    return_layer_metrics=True,
                    training=True
                )
                
                expected_retained = layer_metrics['expected_retained_tokens']
                
                final_selection = layer_metrics['selection_results'][-1]
                if final_selection is not None:
                    final_indices = final_selection.selected_indices
                    start_kept = (final_indices == start_target.unsqueeze(1)).any(dim=1)
                    end_kept = (final_indices == end_target.unsqueeze(1)).any(dim=1)
                    span_survived = (start_kept & end_kept).float().mean().item()
                else:
                    span_survived = 1.0
                    
                loss_start = F.cross_entropy(s_start, start_target)
                loss_end = F.cross_entropy(s_end, end_target)
                qa_loss = (loss_start + loss_end) / 2
                
                logit_kd_loss = calculate_distillation_loss(s_start, s_end, t_start, t_end, temperature=2.0)
                
                hidden_kd_loss = 0.0
                for layer_idx in range(6):
                    if layer_metrics['selection_results'][layer_idx] is not None:
                        s_hidden = layer_metrics['hidden_states'][layer_idx]
                        t_hidden = teacher_hidden_states[layer_idx + 1]
                        sel_indices = layer_metrics['selection_results'][layer_idx].selected_indices
                        att_mask = layer_metrics['selection_results'][layer_idx].new_attention_mask
                        
                        h_loss = calculate_hidden_state_distillation_loss(
                            student_hidden=s_hidden,
                            teacher_hidden=t_hidden,
                            selected_indices=sel_indices,
                            attention_mask=att_mask,
                            mse_weight=1.0,
                            cos_weight=1.0
                        )
                        hidden_kd_loss += h_loss
                
                budget_loss = calculate_lagrangian_budget_loss(
                    expected_retained,
                    target_budget,
                    lagrangian_multiplier
                )
                
                total_loss = qa_loss + lambda_kd * logit_kd_loss + lambda_h * hidden_kd_loss + lambda_b * budget_loss
            
            if use_amp:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.retention_scorers.parameters(), max_norm=1.0)
                # scaler.step() returns None when it skips the step (overflow);
                # we must not advance the scheduler in that case.
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # Only step scheduler when the optimizer actually updated.
                if scaler.get_scale() == scale_before:
                    scheduler.step()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(student.retention_scorers.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
            
            with torch.no_grad():
                # violation > 0 means actual retention > target → increase λ.
                # violation < 0 means under-budget → λ stays at / near 0.
                violation = (expected_retained - target_budget).item()
                lagrangian_multiplier = max(0.0, lagrangian_multiplier + lagrangian_lr * violation)

                epoch_retention += expected_retained.item()
                epoch_target += target_budget
                epoch_violation += violation
                num_batches += 1
            
            global_step += 1
            
            if global_step % 10 == 0:
                progress_bar.set_postfix({
                    'L': f"{total_loss.item():.1f}",
                    'QA': f"{qa_loss.item():.1f}",
                    'LogKD': f"{logit_kd_loss.item():.1f}",
                    'HidKD': f"{hidden_kd_loss.item():.1f}",
                    'Ret': f"{expected_retained.item():.1f}/{target_budget:.1f}",
                    'Lam': f"{lagrangian_multiplier:.3f}",
                    'AnsSurv': f"{span_survived*100:.0f}%"
                })
        
        avg_retention = epoch_retention / num_batches
        avg_target = epoch_target / num_batches
        avg_violation = epoch_violation / num_batches
        
        print(f"Epoch {epoch+1} Budget Stats:")
        print(f"  Target Retention: {avg_target:.1f} tokens")
        print(f"  Actual Retention: {avg_retention:.1f} tokens")
        print(f"  Avg Violation: {avg_violation:.1f} tokens")
        print(f"  Final Lambda: {lagrangian_multiplier:.4f}")
        
        val_loss, val_em, val_f1 = evaluate(student, val_dl)
        print(f"Validation - Epoch {epoch+1}: Loss = {val_loss:.4f}, EM = {val_em:.2f}%, F1 = {val_f1:.2f}%")
        
        score = val_f1
        is_best = score > best_val_score
        if is_best:
            best_val_score = score
            save_checkpoint(
                student.retention_scorers, 
                optimizer=optimizer, 
                step=global_step, 
                checkpoint_path="squad_best_checkpoint.pt",
                scheduler_state_dict=scheduler.state_dict(),
                epoch=epoch+1,
                lagrangian_multiplier=lagrangian_multiplier,
                target_ratio=target_ratio
            )
            print(f"Saved new best checkpoint (EM: {score:.2f}%)")
            
    # Save final checkpoint
    save_checkpoint(
        student.retention_scorers, 
        optimizer=optimizer, 
        step=global_step, 
        checkpoint_path="squad_final_checkpoint.pt",
        scheduler_state_dict=scheduler.state_dict(),
        epoch=args.epochs,
        lagrangian_multiplier=lagrangian_multiplier,
        target_ratio=target_ratio
    )
    print(f"\nTraining complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AMMR Model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--max_train_samples", type=int, default=5000, help="Max training samples")
    parser.add_argument("--max_val_samples", type=int, default=500, help="Max validation samples")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-3, help="Learning rate")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    train(args)
