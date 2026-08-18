"""
Loss functions for AMMR framework.
Includes QA loss, memory budget loss, and entropy regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def calculate_qa_loss(
    qa_model: nn.Module,
    gated_hidden_states: torch.Tensor,
    start_target: torch.Tensor,
    end_target: torch.Tensor,
    reduction: str = 'mean'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate question-answering loss (combined start and end position losses).
    
    Args:
        qa_model: DistilBert QA model with qa_outputs head.
        gated_hidden_states: Hidden states after retention gating (batch, seq_len, hidden_dim).
        start_target: Ground-truth start positions (batch,).
        end_target: Ground-truth end positions (batch,).
        reduction: Loss reduction method ('mean', 'sum', 'none').
        
    Returns:
        Tuple of:
            - qa_loss: Combined start+end loss.
            - start_logits: Predicted start position logits.
            - end_logits: Predicted end position logits.
    """
    # Compute QA head logits using qa_outputs
    logits = qa_model.qa_outputs(gated_hidden_states)
    
    start_logits = logits[..., 0]  # (batch, seq_len)
    end_logits = logits[..., 1]    # (batch, seq_len)
    
    # Compute cross-entropy loss for start and end positions
    start_loss = F.cross_entropy(start_logits, start_target, reduction=reduction)
    end_loss = F.cross_entropy(end_logits, end_target, reduction=reduction)
    
    # Average start and end loss
    if reduction == 'none':
        qa_loss = (start_loss + end_loss) / 2
    else:
        qa_loss = (start_loss + end_loss) / 2
    
    return qa_loss, start_logits, end_logits


def calculate_budget_loss(
    probabilities: torch.Tensor,
    valid_mask: torch.Tensor,
    target_budget: int,
    penalty_mode: str = 'excess'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate memory budget loss (encourages retention within budget).
    
    The budget loss penalizes when expected retained tokens exceeds target budget:
    
        Expected retained = sum(p_t * valid_mask_t)
        Excess = max(0, Expected - Budget)
        Loss = Excess^2
    
    Args:
        probabilities: Retention probabilities (batch, seq_len).
        valid_mask: Mask for adaptively-retained tokens (seq_len,).
        target_budget: Target number of tokens to retain.
        penalty_mode: How to compute penalty ('excess', 'mse', 'l1').
        
    Returns:
        Tuple of:
            - budget_loss: Scalar loss value.
            - expected_tokens: Expected number of retained tokens.
    """
    # Move valid_mask to same device as probabilities
    valid_mask = valid_mask.to(probabilities.device)
    
    # Compute expected number of retained tokens
    # Only count valid (adaptive) tokens, not protected tokens
    expected_tokens = (probabilities * valid_mask.unsqueeze(0)).sum(dim=-1)
    
    # Compute penalty based on mode
    if penalty_mode == 'excess':
        # Only penalize when exceeding budget
        excess = F.relu(expected_tokens - target_budget)
        loss = (excess ** 2).mean()
    
    elif penalty_mode == 'mse':
        # MSE from target budget
        loss = ((expected_tokens - target_budget) ** 2).mean()
    
    elif penalty_mode == 'l1':
        # L1 distance from target budget
        loss = torch.abs(expected_tokens - target_budget).mean()
    
    else:
        raise ValueError(f"Unknown penalty_mode: {penalty_mode}")
    
    return loss, expected_tokens


def calculate_entropy_loss(
    probabilities: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Calculate entropy regularization loss.
    
    Entropy encourages sharp decisions (probabilities close to 0 or 1)
    rather than ambiguous middle values (0.5). We compute binary entropy
    and sum over all adaptive tokens.
    
    Binary entropy: H(p) = -[p*log(p) + (1-p)*log(1-p)]
    
    Minimizing entropy means reducing uncertainty, pushing decisions to extremes.
    
    Args:
        probabilities: Retention probabilities (batch, seq_len).
        valid_mask: Mask for adaptive tokens (seq_len,).
        eps: Small value to avoid log(0).
        
    Returns:
        Mean entropy over adaptive tokens.
    """
    # Clamp probabilities to avoid log(0)
    p = probabilities.clamp(min=eps, max=1 - eps)
    
    # Binary entropy: H = -[p*log(p) + (1-p)*log(1-p)]
    entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
    
    # Move valid_mask to same device
    valid_mask = valid_mask.to(entropy.device)
    
    # Apply mask - only count adaptive tokens
    masked_entropy = entropy * valid_mask.unsqueeze(0)
    
    # Average over valid tokens
    mean_entropy = masked_entropy.sum() / (valid_mask.sum() + eps)
    
    return mean_entropy


def calculate_combined_loss(
    qa_loss: torch.Tensor,
    budget_loss: torch.Tensor,
    entropy_loss: torch.Tensor,
    budget_weight: float = 0.10,
    entropy_weight: float = 0.001
) -> Tuple[torch.Tensor, dict]:
    """
    Calculate combined loss with multiple objectives.
    
    Total loss = QA_loss + budget_weight * budget_loss + entropy_weight * entropy_loss
    
    Args:
        qa_loss: Task performance loss.
        budget_loss: Memory budget constraint loss.
        entropy_loss: Decision sharpness regularization.
        budget_weight: Weight for budget objective.
        entropy_weight: Weight for entropy objective.
        
    Returns:
        Tuple of:
            - total_loss: Weighted sum of all losses.
            - loss_dict: Dictionary with individual loss components.
    """
    total_loss = (
        qa_loss +
        budget_weight * budget_loss +
        entropy_weight * entropy_loss
    )
    
    loss_dict = {
        'total': total_loss.item(),
        'qa': qa_loss.item(),
        'budget': budget_loss.item(),
        'entropy': entropy_loss.item(),
    }
    
    return total_loss, loss_dict


class QALossFunction(nn.Module):
    """
    Wrapper for QA loss calculation as a module.
    Useful for integration into training pipelines.
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Initialize QA loss function.
        
        Args:
            reduction: Loss reduction method.
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        qa_model: nn.Module,
        gated_hidden_states: torch.Tensor,
        start_target: torch.Tensor,
        end_target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute QA loss.
        
        Args:
            qa_model: DistilBERT QA model.
            gated_hidden_states: Hidden states after retention.
            start_target: Ground-truth start indices.
            end_target: Ground-truth end indices.
            
        Returns:
            Tuple of (loss, start_logits, end_logits).
        """
        return calculate_qa_loss(
            qa_model,
            gated_hidden_states,
            start_target,
            end_target,
            reduction=self.reduction
        )
