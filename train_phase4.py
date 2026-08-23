import os
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import AdamW

from config import MODEL_NAME, DEVICE
from src.models_adaptive import AdaptiveDistilBertQA
from src.baseline import BaselineQAModel
from src.squad_data import get_squad_dataloaders
from src.losses import calculate_distillation_loss, calculate_lagrangian_budget_loss
from src.utils import set_seed, print_header, save_checkpoint

def train_phase4():
    set_seed(42)
    print_header("PHASE 4: SQuAD TRAINING (DISTILLATION + LAGRANGIAN BUDGET)")
    
    # 1. Load Data
    batch_size = 2
    max_train_samples = 50
    train_dl, val_dl, train_data, val_data, val_features = get_squad_dataloaders(
        batch_size=batch_size, 
        max_train_samples=max_train_samples, 
        max_val_samples=100
    )
    
    # 2. Initialize Models
    print("\nInitializing Teacher (Frozen DistilBERT) and Student (AMMR)...")
    teacher = BaselineQAModel(freeze_parameters=True)
    teacher.qa_model.eval()
    
    # AMMR model (Transformer frozen, only retention scorers trainable)
    student = AdaptiveDistilBertQA(
        model_name=MODEL_NAME, 
        device=DEVICE,
        freeze_transformer=True
    )
    student.unfreeze_scorers()
    student.train()
    
    # 3. Setup Optimizers
    learning_rate = 5e-3
    optimizer = AdamW(student.retention_scorers.parameters(), lr=learning_rate)
    
    # Lagrangian Multiplier (lambda) setup
    lagrangian_multiplier = 0.0
    lagrangian_lr = 0.01
    
    # 4. Training Configuration
    epochs = 1
    global_budget_ratio = 0.50 # Target retaining 50% of all tokens across all layers
    # Total tokens = batch_size * num_layers * seq_len
    # Instead of calculating per-batch, we can calculate per-sequence-layer:
    # budget per sequence per layer = seq_len * target_ratio
    
    total_steps = len(train_dl) * epochs
    print(f"Starting training for {epochs} epochs ({total_steps} steps).")
    print(f"Target Budget Ratio: {global_budget_ratio}")
    
    global_step = 0
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_qa = 0
        epoch_distill = 0
        epoch_budget = 0
        
        progress_bar = tqdm(train_dl, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            start_target = batch['start_positions'].to(DEVICE)
            end_target = batch['end_positions'].to(DEVICE)
            
            seq_len = input_ids.size(1)
            # We have 6 layers. Budget per sequence is 6 * seq_len * ratio
            target_budget_per_seq = 6 * seq_len * global_budget_ratio
            
            # --- Forward Teacher ---
            with torch.no_grad():
                teacher_outputs = teacher.qa_model(input_ids, attention_mask)
                t_start = teacher_outputs.start_logits
                t_end = teacher_outputs.end_logits
                
            # --- Forward Student ---
            # Training=True passes Gumbel-Softmax noise and returns continuous expected L0 penalty
            s_start, s_end, layer_metrics = student(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                return_layer_metrics=True,
                training=True
            )
            
            expected_retained_tokens_per_seq = layer_metrics['expected_retained_tokens']
            
            # --- Calculate Losses ---
            # 1. QA Loss (Cross Entropy with ground truth)
            loss_start = F.cross_entropy(s_start, start_target)
            loss_end = F.cross_entropy(s_end, end_target)
            qa_loss = (loss_start + loss_end) / 2
            
            # 2. Distillation Loss (KL Div with teacher)
            distill_loss = calculate_distillation_loss(s_start, s_end, t_start, t_end, temperature=2.0)
            
            # 3. Lagrangian Budget Loss
            budget_loss = calculate_lagrangian_budget_loss(
                expected_retained_tokens_per_seq, 
                target_budget_per_seq, 
                lagrangian_multiplier
            )
            
            # 4. Total Loss
            total_loss = qa_loss + distill_loss + budget_loss
            
            # --- Backward Model ---
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # --- Dual Gradient Ascent for Lambda ---
            # lambda = max(0, lambda + lr_lambda * (E[tokens] - Budget))
            with torch.no_grad():
                violation = (expected_retained_tokens_per_seq - target_budget_per_seq).item()
                lagrangian_multiplier = max(0.0, lagrangian_multiplier + lagrangian_lr * violation)
            
            # Logging
            global_step += 1
            epoch_loss += total_loss.item()
            epoch_qa += qa_loss.item()
            epoch_distill += distill_loss.item()
            
            progress_bar.set_postfix({
                'Loss': f"{total_loss.item():.2f}",
                'QA': f"{qa_loss.item():.2f}",
                'Distill': f"{distill_loss.item():.2f}",
                'Retained': f"{expected_retained_tokens_per_seq.item():.1f}/{target_budget_per_seq:.1f}",
                'Lambda': f"{lagrangian_multiplier:.4f}"
            })
            
    # Save model
    save_checkpoint(student, optimizer=optimizer, step=global_step, checkpoint_path="squad_phase4_checkpoint.pt")
    print(f"Training complete. Model saved to squad_phase4_checkpoint.pt")

if __name__ == "__main__":
    train_phase4()
