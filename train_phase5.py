import os
import time
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

def train_phase5():
    set_seed(42)
    print_header("PHASE 5: STABLE TASK-PRESERVING AMMR TRAINING")
    
    # 1. Load Data
    batch_size = 16
    max_train_samples = 5000  # Substantially larger subset
    train_dl, val_dl, train_data, val_data, val_features = get_squad_dataloaders(
        batch_size=batch_size, 
        max_train_samples=max_train_samples, 
        max_val_samples=200
    )
    
    # 2. Initialize Models
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
    
    # 3. Setup Optimizers
    learning_rate = 3e-3
    optimizer = AdamW(student.retention_scorers.parameters(), lr=learning_rate)
    
    epochs = 5
    total_steps = len(train_dl) * epochs
    warmup_steps = int(0.1 * total_steps)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    # Lagrangian Multiplier
    lagrangian_multiplier = 0.0
    lagrangian_lr = 0.05
    
    # Curriculum: Target global retention ratio over epochs
    # Epochs 0->4: 95% -> 90% -> 80% -> 70% -> 60%
    curriculum = [0.95, 0.90, 0.80, 0.70, 0.60]
    
    # Loss Weights
    lambda_kd = 1.0       # Logit KD
    lambda_h = 1.0        # Hidden State KD
    lambda_b = 1.0        # Budget
    
    print(f"Starting training for {epochs} epochs ({total_steps} steps).")
    
    global_step = 0
    
    for epoch in range(epochs):
        # Set curriculum target for this epoch
        target_ratio = curriculum[min(epoch, len(curriculum)-1)]
        print(f"\n[Epoch {epoch+1}/{epochs}] Curriculum Target: {target_ratio*100:.1f}%")
        
        progress_bar = tqdm(train_dl, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            start_target = batch['start_positions'].to(DEVICE)
            end_target = batch['end_positions'].to(DEVICE)
            
            seq_len = input_ids.size(1)
            # Global budget per sequence (across 6 layers)
            target_budget_per_seq = 6 * seq_len * target_ratio
            
            # --- Forward Teacher ---
            with torch.no_grad():
                teacher_outputs = teacher.model(
                    input_ids, 
                    attention_mask, 
                    output_hidden_states=True
                )
                t_start = teacher_outputs.start_logits
                t_end = teacher_outputs.end_logits
                # teacher_outputs.hidden_states contains 7 elements: [embedding, layer1, layer2, ..., layer6]
                teacher_hidden_states = teacher_outputs.hidden_states
                
            # --- Forward Student ---
            s_start, s_end, layer_metrics = student(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                return_layer_metrics=True,
                training=True
            )
            
            expected_retained = layer_metrics['expected_retained_tokens']
            
            # --- Answer Span Tracking (Diagnostic) ---
            # We check if the start/end tokens were retained at the final layer.
            # final layer selection results:
            final_selection = layer_metrics['selection_results'][-1]
            if final_selection is not None:
                final_indices = final_selection.selected_indices
                # Check how many batch elements kept both start and end
                batch_range = torch.arange(input_ids.size(0), device=DEVICE).unsqueeze(1)
                start_kept = (final_indices == start_target.unsqueeze(1)).any(dim=1)
                end_kept = (final_indices == end_target.unsqueeze(1)).any(dim=1)
                span_survived = (start_kept & end_kept).float().mean().item()
            else:
                span_survived = 1.0
                
            # --- Calculate Losses ---
            # 1. QA Loss
            loss_start = F.cross_entropy(s_start, start_target)
            loss_end = F.cross_entropy(s_end, end_target)
            qa_loss = (loss_start + loss_end) / 2
            
            # 2. Logit Distillation Loss
            logit_kd_loss = calculate_distillation_loss(s_start, s_end, t_start, t_end, temperature=2.0)
            
            # 3. Hidden State Distillation Loss
            hidden_kd_loss = 0.0
            for layer_idx in range(6):
                if layer_metrics['selection_results'][layer_idx] is not None:
                    # student hidden states at end of layer_idx
                    s_hidden = layer_metrics['hidden_states'][layer_idx]
                    # teacher hidden states at end of layer_idx
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
            
            # 4. Lagrangian Budget Loss
            budget_loss = calculate_lagrangian_budget_loss(
                expected_retained, 
                target_budget_per_seq, 
                lagrangian_multiplier
            )
            
            # 5. Total Loss
            total_loss = qa_loss + lambda_kd * logit_kd_loss + lambda_h * hidden_kd_loss + lambda_b * budget_loss
            
            # --- Backward ---
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(student.retention_scorers.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # --- Dual Gradient Ascent for Lambda ---
            with torch.no_grad():
                violation = (expected_retained - target_budget_per_seq).item()
                lagrangian_multiplier = max(0.0, lagrangian_multiplier + lagrangian_lr * violation)
            
            # Logging
            global_step += 1
            
            if global_step % 10 == 0:
                progress_bar.set_postfix({
                    'L': f"{total_loss.item():.1f}",
                    'QA': f"{qa_loss.item():.1f}",
                    'LogKD': f"{logit_kd_loss.item():.1f}",
                    'HidKD': f"{hidden_kd_loss.item():.1f}",
                    'Ret': f"{expected_retained.item():.0f}/{target_budget_per_seq:.0f}",
                    'Lam': f"{lagrangian_multiplier:.3f}",
                    'AnsSurv': f"{span_survived*100:.0f}%"
                })
                
    # Save model
    save_checkpoint(student, optimizer=optimizer, step=global_step, checkpoint_path="squad_phase5_checkpoint.pt")
    print(f"\nTraining complete. Model saved to squad_phase5_checkpoint.pt")

if __name__ == "__main__":
    train_phase5()
