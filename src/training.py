"""
Training loop and trainer for retention scorer.
Handles training, validation, and checkpoint management.
"""

from typing import Dict, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    LEARNING_RATE,
    OPTIMIZER_WEIGHT_DECAY,
    GRADIENT_CLIP,
    BUDGET_LAMBDA,
    ENTROPY_LAMBDA,
    LOG_INTERVAL,
    DEVICE
)
from src.utils import print_header, save_checkpoint, load_checkpoint, format_number
from src.models import RetentionScorer
from src.losses import (
    calculate_qa_loss,
    calculate_budget_loss,
    calculate_entropy_loss,
    calculate_combined_loss
)


class RetentionScorerTrainer:
    """
    Trainer for learning retention probabilities.
    
    This trainer:
    - Manages optimizer and training loop
    - Computes multi-objective loss (QA + budget + entropy)
    - Handles checkpointing and logging
    - Supports gradient accumulation and clipping
    """
    
    def __init__(
        self,
        scorer: RetentionScorer,
        qa_model: nn.Module,
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = OPTIMIZER_WEIGHT_DECAY,
        gradient_clip: float = GRADIENT_CLIP,
        device: Optional[torch.device] = None,
        budget_lambda: float = BUDGET_LAMBDA,
        entropy_lambda: float = ENTROPY_LAMBDA,
    ):
        """
        Initialize trainer.
        
        Args:
            scorer: RetentionScorer model to train.
            qa_model: Pretrained QA model (frozen).
            learning_rate: Optimizer learning rate.
            weight_decay: L2 regularization weight.
            gradient_clip: Maximum gradient norm.
            device: Training device.
            budget_lambda: Weight for budget loss.
            entropy_lambda: Weight for entropy loss.
        """
        self.scorer = scorer
        self.qa_model = qa_model
        self.device = device or DEVICE
        self.gradient_clip = gradient_clip
        self.budget_lambda = budget_lambda
        self.entropy_lambda = entropy_lambda
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            scorer.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Training state
        self.current_step = 0
        self.training_history = {
            'total_loss': [],
            'qa_loss': [],
            'budget_loss': [],
            'entropy_loss': [],
            'expected_tokens': [],
            'gradient_norm': [],
        }
    
    def train_step(
        self,
        hidden_states: torch.Tensor,
        protected_mask: torch.Tensor,
        valid_mask: torch.Tensor,
        start_target: torch.Tensor,
        end_target: torch.Tensor,
        target_budget: int,
        temperature: float = 1.0
    ) -> Dict[str, float]:
        """
        Perform single training step.
        
        Args:
            hidden_states: Token hidden states (batch, seq_len, hidden_dim).
            protected_mask: Protected token mask (seq_len,).
            valid_mask: Valid (adaptive) token mask (seq_len,).
            start_target: Ground truth start indices (batch,).
            end_target: Ground truth end indices (batch,).
            target_budget: Target number of tokens to retain.
            temperature: Temperature for probability sharpness.
            
        Returns:
            Dictionary with loss components and metrics.
        """
        # Forward pass: compute retention probabilities
        scores, probabilities = self.scorer(hidden_states, temperature)
        
        # Protect special tokens
        protected_mask = protected_mask.to(self.device)
        probabilities = torch.where(
            protected_mask.unsqueeze(0),
            torch.ones_like(probabilities),
            probabilities
        )
        
        # Apply soft gate
        gated_hidden_states = hidden_states * probabilities.unsqueeze(-1)
        
        # Compute QA loss
        qa_loss, start_logits, end_logits = calculate_qa_loss(
            self.qa_model,
            gated_hidden_states,
            start_target,
            end_target
        )
        
        # Compute budget loss
        budget_loss, expected_tokens = calculate_budget_loss(
            probabilities,
            valid_mask,
            target_budget,
            penalty_mode='excess'
        )
        
        # Compute entropy loss
        entropy_loss = calculate_entropy_loss(probabilities, valid_mask)
        
        # Combine losses
        total_loss, loss_dict = calculate_combined_loss(
            qa_loss,
            budget_loss,
            entropy_loss,
            budget_weight=self.budget_lambda,
            entropy_weight=self.entropy_lambda
        )
        
        # Backward pass
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        
        # Gradient clipping
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.scorer.parameters(),
            self.gradient_clip
        )
        
        # Optimizer step
        self.optimizer.step()
        
        # Update step counter
        self.current_step += 1
        
        # Prepare output
        result = loss_dict.copy()
        result['expected_tokens'] = expected_tokens.item()
        result['gradient_norm'] = float(gradient_norm)
        
        # Log to history
        self._log_to_history(result)
        
        return result
    
    def _log_to_history(self, result: Dict) -> None:
        """Log metrics to training history."""
        self.training_history['total_loss'].append(result['total'])
        self.training_history['qa_loss'].append(result['qa'])
        self.training_history['budget_loss'].append(result['budget'])
        self.training_history['entropy_loss'].append(result['entropy'])
        self.training_history['expected_tokens'].append(result['expected_tokens'])
        self.training_history['gradient_norm'].append(result['gradient_norm'])
    
    def should_log(self, log_interval: int = LOG_INTERVAL) -> bool:
        """
        Check if current step should be logged.
        
        Args:
            log_interval: Logging frequency in steps.
            
        Returns:
            True if should log this step.
        """
        return (
            self.current_step == 1 or
            self.current_step % log_interval == 0
        )
    
    def format_result(
        self,
        result: Dict,
        step: Optional[int] = None
    ) -> str:
        """
        Format training result for display.
        
        Args:
            result: Result dictionary from train_step.
            step: Step number (uses current_step if None).
            
        Returns:
            Formatted string for printing.
        """
        if step is None:
            step = self.current_step
        
        return (
            f"Step {step:4d} | "
            f"Total={format_number(result['total'], 5)} | "
            f"QA={format_number(result['qa'], 5)} | "
            f"Budget={format_number(result['budget'], 5)} | "
            f"Entropy={format_number(result['entropy'], 5)} | "
            f"Expected={format_number(result['expected_tokens'], 3)} | "
            f"Grad={format_number(result['gradient_norm'], 4)}"
        )
    
    def save_checkpoint(self, path: str) -> None:
        """Save trainer state."""
        save_checkpoint(self.scorer, self.optimizer, self.current_step, path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load trainer state."""
        self.current_step = load_checkpoint(self.scorer, self.optimizer, path)
    
    def get_history(self) -> Dict[str, List]:
        """Get training history."""
        return self.training_history.copy()


def train_retention_scorer(
    scorer: RetentionScorer,
    qa_model: nn.Module,
    hidden_states: torch.Tensor,
    protected_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    start_target: torch.Tensor,
    end_target: torch.Tensor,
    target_budget: int,
    num_steps: int,
    learning_rate: float = LEARNING_RATE,
    budget_lambda: float = BUDGET_LAMBDA,
    entropy_lambda: float = ENTROPY_LAMBDA,
    temperature: float = 1.0,
    log_interval: int = LOG_INTERVAL,
    verbose: bool = True
) -> Tuple[RetentionScorerTrainer, Dict]:
    """
    Train retention scorer with simple interface.
    
    This is a convenience function for straightforward training.
    For more control, use RetentionScorerTrainer directly.
    
    Args:
        scorer: RetentionScorer model.
        qa_model: Frozen QA model.
        hidden_states: Token hidden states.
        protected_mask: Protected token mask.
        valid_mask: Valid (adaptive) token mask.
        start_target: Ground truth start indices.
        end_target: Ground truth end indices.
        target_budget: Target number of tokens to retain.
        num_steps: Number of training steps.
        learning_rate: Optimizer learning rate.
        budget_lambda: Budget loss weight.
        entropy_lambda: Entropy loss weight.
        temperature: Probability temperature.
        log_interval: Logging frequency.
        verbose: Whether to print progress.
        
    Returns:
        Tuple of (trainer, final_result_dict).
    """
    # Create trainer
    trainer = RetentionScorerTrainer(
        scorer,
        qa_model,
        learning_rate=learning_rate,
        budget_lambda=budget_lambda,
        entropy_lambda=entropy_lambda
    )
    
    if verbose:
        print_header("TRAINING RETENTION SCORER")
        print(f"Steps: {num_steps}")
        print(f"Learning Rate: {learning_rate}")
        print(f"Budget Lambda: {budget_lambda}")
        print(f"Entropy Lambda: {entropy_lambda}")
        print()
    
    # Training loop
    scorer.train()
    
    for step in range(1, num_steps + 1):
        result = trainer.train_step(
            hidden_states,
            protected_mask,
            valid_mask,
            start_target,
            end_target,
            target_budget,
            temperature=temperature
        )
        
        # Log results
        if trainer.should_log(log_interval) or step == num_steps:
            if verbose:
                print(trainer.format_result(result, step))
    
    scorer.eval()
    
    if verbose:
        print()
    
    return trainer, result
