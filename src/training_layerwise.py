"""
Training loop and trainer for layer-wise adaptive memory models.
Handles multi-layer training with differentiable token gating.
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
from src.losses import calculate_budget_loss, calculate_entropy_loss


class LayerwiseAdaptiveTrainer:
    """
    Trainer for learning layer-wise retention probabilities across all Transformer layers.
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
        
        # Unfreeze retention scorers for optimization
        self.model.unfreeze_scorers()
        
        # Optimize retention scorers only
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
        Perform a single training step.
        """
        # Ensure tensors are on the target device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        start_target = start_target.to(self.device)
        end_target = end_target.to(self.device)

        # Forward pass with stochastic relaxation active
        start_logits, end_logits, layer_metrics = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_layer_metrics=True,
            training=True
        )
        
        # 1. QA Cross-Entropy Loss
        start_loss = torch.nn.functional.cross_entropy(start_logits, start_target, ignore_index=-100)
        end_loss = torch.nn.functional.cross_entropy(end_logits, end_target, ignore_index=-100)
        qa_loss = (start_loss + end_loss) / 2.0
        
        # 2. Layer-wise Budget Loss
        total_budget_loss = torch.tensor(0.0, device=self.device)
        total_entropy_loss = torch.tensor(0.0, device=self.device)
        
        if layer_metrics and 'selection_results' in layer_metrics:
            for layer_idx, selection_result in enumerate(layer_metrics['selection_results']):
                if selection_result is None:
                    continue
                    
                probs = selection_result.retention_probs  # (batch, seq_len)
                valid_mask = (probs > 0.0)
                
                target_ratio = self.model.retention_schedule[layer_idx]
                target_budget = max(2, int(probs.shape[1] * target_ratio))
                
                b_loss, _ = calculate_budget_loss(
                    probs,
                    valid_mask,
                    target_budget,
                    penalty_mode='excess'
                )
                total_budget_loss = total_budget_loss + b_loss
                
                if self.entropy_lambda > 0.0:
                    e_loss = calculate_entropy_loss(probs, valid_mask)
                    total_entropy_loss = total_entropy_loss + e_loss

        # 3. Combined Total Loss
        total_loss = (
            qa_loss +
            (self.budget_lambda * total_budget_loss) +
            (self.entropy_lambda * total_entropy_loss)
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
        
        # Log results
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