"""
Training loop and trainer for layer-wise adaptive memory models.
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
from src.utils import format_number, save_checkpoint, load_checkpoint
from src.models_adaptive import AdaptiveDistilBertQA
from src.losses import (
    calculate_qa_loss,
    calculate_budget_loss,
    calculate_entropy_loss
)


class LayerwiseAdaptiveTrainer:
    """
    Trainer for learning layer-wise retention probabilities.
    
    This trainer handles the complex multi-objective optimization across
    all Transformer layers simultaneously.
    """
    
    def __init__(
        self,
        model: AdaptiveDistilBertQA,
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = OPTIMIZER_WEIGHT_DECAY,
        gradient_clip: float = GRADIENT_CLIP,
        device: Optional[torch.device] = None,
        budget_lambda: float = BUDGET_LAMBDA,
        entropy_lambda: float = ENTROPY_LAMBDA,
    ):
        self.model = model
        self.device = device or DEVICE
        self.gradient_clip = gradient_clip
        self.budget_lambda = budget_lambda
        self.entropy_lambda = entropy_lambda
        
        # Unfreeze scorers for training
        self.model.unfreeze_scorers()
        
        # Create optimizer for the retention scorers only
        self.optimizer = optim.AdamW(
            self.model.get_retention_scorers().parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.current_step = 0
        self.training_history = {
            'total_loss': [],
            'qa_loss': [],
            'budget_loss': [],
            'entropy_loss': [],
            'gradient_norm': [],
        }
        
    def train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        start_target: torch.Tensor,
        end_target: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform single training step.
        
        Args:
            input_ids: Token IDs (batch, seq_len).
            attention_mask: Attention mask (batch, seq_len).
            start_target: Ground truth start indices (batch,).
            end_target: Ground truth end indices (batch,).
            
        Returns:
            Dictionary with loss components.
        """
        # Forward pass
        start_logits, end_logits, layer_metrics = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_layer_metrics=True
        )
        
        # Compute QA Loss
        # We can't directly use calculate_qa_loss because it expects a qa_model and hidden states.
        # But we already have start_logits and end_logits, so we compute cross entropy directly.
        start_loss = torch.nn.functional.cross_entropy(start_logits, start_target)
        end_loss = torch.nn.functional.cross_entropy(end_logits, end_target)
        qa_loss = (start_loss + end_loss) / 2
        
        # Compute Layer-wise Losses
        total_budget_loss = 0.0
        total_entropy_loss = 0.0
        
        for layer_idx, selection_result in enumerate(layer_metrics['selection_results']):
            if selection_result is None:
                continue
                
            probs = selection_result.retention_probs  # (batch, seq_len)
            
            # Reconstruct a valid mask (we can just use all 1s of the same shape)
            valid_mask = torch.ones_like(probs, dtype=torch.bool)
            
            target_ratio = self.model.retention_schedule[layer_idx]
            # Average sequence length for the batch
            target_budget = max(3, int(probs.shape[1] * target_ratio))
            
            b_loss, _ = calculate_budget_loss(
                probs,
                valid_mask,
                target_budget,
                penalty_mode='excess'
            )
            
            e_loss = calculate_entropy_loss(probs, valid_mask)
            
            total_budget_loss += b_loss
            total_entropy_loss += e_loss
            
        # Combine losses
        total_loss = (
            qa_loss +
            self.budget_lambda * total_budget_loss +
            self.entropy_lambda * total_entropy_loss
        )
        
        # Backward pass
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        
        # Gradient clipping
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.get_retention_scorers().parameters(),
            self.gradient_clip
        )
        
        # Optimizer step
        self.optimizer.step()
        self.current_step += 1
        
        # Result dict
        result = {
            'total': total_loss.item(),
            'qa': qa_loss.item(),
            'budget': total_budget_loss.item(),
            'entropy': total_entropy_loss.item(),
            'gradient_norm': float(gradient_norm)
        }
        
        self.training_history['total_loss'].append(result['total'])
        self.training_history['qa_loss'].append(result['qa'])
        self.training_history['budget_loss'].append(result['budget'])
        self.training_history['entropy_loss'].append(result['entropy'])
        self.training_history['gradient_norm'].append(result['gradient_norm'])
        
        return result
        
    def should_log(self, log_interval: int = LOG_INTERVAL) -> bool:
        return self.current_step == 1 or self.current_step % log_interval == 0
        
    def format_result(self, result: Dict, step: Optional[int] = None) -> str:
        step = step or self.current_step
        return (
            f"Step {step:4d} | "
            f"Total={format_number(result['total'], 4)} | "
            f"QA={format_number(result['qa'], 4)} | "
            f"Budget={format_number(result['budget'], 4)} | "
            f"Entropy={format_number(result['entropy'], 4)} | "
            f"Grad={format_number(result['gradient_norm'], 4)}"
        )
