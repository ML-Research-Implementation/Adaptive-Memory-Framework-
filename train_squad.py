import torch
from tqdm import tqdm
from config import MODEL_NAME, DEVICE
from src.squad_data import get_squad_dataloaders
from src.models_adaptive import AdaptiveDistilBertQA
from src.training_layerwise import LayerwiseAdaptiveTrainer
from src.utils import print_header, save_checkpoint

def train():
    print_header("PHASE 3: TRAINING AMMR ON SQuAD")
    
    # Configuration
    batch_size = 4
    max_train_samples = 500  # Train on a small subset for this phase
    max_val_samples = 10
    epochs = 1
    
    # 1. Load Data
    train_dl, _, _, _, _ = get_squad_dataloaders(
        batch_size=batch_size,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples
    )
    
    # 2. Model
    schedule = [0.90, 0.85, 0.80, 0.75, 0.70, 0.70]
    model = AdaptiveDistilBertQA(
        model_name=MODEL_NAME,
        device=DEVICE,
        freeze_transformer=True,
        retention_schedule=schedule
    )
    
    # 3. Trainer
    trainer = LayerwiseAdaptiveTrainer(
        model=model,
        learning_rate=3e-4,
        budget_lambda=0.5,
        entropy_lambda=0.01,
        device=DEVICE
    )
    
    # 4. Training Loop
    print(f"Starting training on {max_train_samples} examples for {epochs} epochs...")
    model.train()
    
    global_step = 0
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        pbar = tqdm(train_dl, desc="Training")
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            start_target = batch['start_positions'].to(DEVICE)
            end_target = batch['end_positions'].to(DEVICE)
            
            # Step
            metrics = trainer.train_step(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_target=start_target,
                end_target=end_target
            )
            
            global_step += 1
            
            if global_step % 10 == 0:
                pbar.set_postfix({
                    'Loss': f"{metrics['total']:.3f}",
                    'QA': f"{metrics['qa']:.3f}",
                    'Budg': f"{metrics['budget']:.3f}",
                    'Entr': f"{metrics['entropy']:.3f}"
                })
                
    # 5. Save model
    save_checkpoint(model, optimizer=trainer.optimizer, step=global_step, checkpoint_path="squad_phase3_checkpoint.pt")
    print(f"Training complete. Model saved to squad_phase3_checkpoint.pt")

if __name__ == "__main__":
    train()
