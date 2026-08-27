"""
End-to-end SQuAD training script for Adaptive DistilBERT with layer-wise stochastic gating.
Includes memory management and periodic auto-checkpointing.
"""

import os
import gc
import torch
from config import (
    MODEL_NAME,
    DEVICE,
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_SEQUENCE_LENGTH,
    BUDGET_LAMBDA,
    ENTROPY_LAMBDA
)
from src.models_adaptive import AdaptiveDistilBertQA
from src.training_layerwise import LayerwiseAdaptiveTrainer
from src.squad_data import get_squad_dataloaders


def main():
    print(f"Using device: {DEVICE}")

    # 1. Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    checkpoint_path = "models/layerwise_scorers_phase4.pt"

    # 2. Load SQuAD DataLoaders
    train_loader, val_loader, _, _, _ = get_squad_dataloaders(
        batch_size=BATCH_SIZE,
        max_train_samples=None,
        max_length=MAX_SEQUENCE_LENGTH
    )
    print(f"Loaded training batches: {len(train_loader)}")

    # 3. Initialize Model
    """ model = AdaptiveDistilBertQA(
        model_name=MODEL_NAME,
        device=DEVICE,
        freeze_transformer=True
    ).to(DEVICE)

    # 4. Initialize Trainer
    trainer = LayerwiseAdaptiveTrainer(
        model=model,
        learning_rate=LEARNING_RATE,
        device=DEVICE,
        budget_lambda=BUDGET_LAMBDA,
        entropy_lambda=ENTROPY_LAMBDA
    ) """

    # 3. Initialize Model
    model = AdaptiveDistilBertQA(
        model_name=MODEL_NAME,
        device=DEVICE,
        freeze_transformer=True
    ).to(DEVICE)

    checkpoint_path = "models/layerwise_scorers_phase4.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading existing checkpoint from {checkpoint_path}...")
        model.get_retention_scorers().load_state_dict(
            torch.load(checkpoint_path, map_location=DEVICE)
        )
        print("Scorer weights loaded successfully! Continuing from previous progress.")

    # 4. Initialize Trainer
    trainer = LayerwiseAdaptiveTrainer(
        model=model,
        learning_rate=LEARNING_RATE,
        device=DEVICE,
        budget_lambda=BUDGET_LAMBDA,
        entropy_lambda=ENTROPY_LAMBDA
    )

    # 5. Training Loop (1 Epoch = 5,533 steps is sufficient for scorer convergence)
    start_epoch = 2
    num_epochs = 3
    print("\n--- Starting Phase 4 Training ---")
    
    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            start_target = batch['start_positions'].to(DEVICE)
            end_target = batch['end_positions'].to(DEVICE)

            result = trainer.train_step(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_target=start_target,
                end_target=end_target
            )

            if trainer.should_log():
                print(trainer.format_result(result, step=trainer.current_step))

            # Auto-save and clear memory every 500 steps
            if trainer.current_step % 500 == 0:
                torch.save(model.get_retention_scorers().state_dict(), checkpoint_path)
                print(f"--> [Checkpoint Saved at Step {trainer.current_step}]")

            # Routine memory cleanup every 200 steps
            if step % 200 == 0:
                if DEVICE.type == "mps":
                    torch.mps.empty_cache()
                gc.collect()

    # 6. Final Save
    torch.save(model.get_retention_scorers().state_dict(), checkpoint_path)
    print(f"\nTraining complete. Saved final scorers to {checkpoint_path}")


if __name__ == "__main__":
    main()