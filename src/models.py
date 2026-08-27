"""
Neural network models for AMMR framework.
Includes the learnable retention scorer, stochastic gates, and layerwise components.
"""

import torch
import torch.nn as nn
from typing import Tuple
from config import (
    HIDDEN_DIMENSION,
    TEMPERATURE,
    RETENTION_SCORER_DROPOUT,
    RETENTION_SCORER_INTERMEDIATE_DIM_RATIO
)


class RetentionScorer(nn.Module):
    """
    Learnable token retention scoring network.
    Outputs a single logit per token for HardConcreteGate.
    """
    def __init__(
        self,
        hidden_dimension: int = HIDDEN_DIMENSION,
        dropout: float = RETENTION_SCORER_DROPOUT,
        intermediate_dim_ratio: int = RETENTION_SCORER_INTERMEDIATE_DIM_RATIO
    ):
        super().__init__()
        self.hidden_dimension = hidden_dimension
        self.intermediate_dimension = max(64, hidden_dimension // intermediate_dim_ratio)
        
        # MLP ending in 1 logit output
        self.network = nn.Sequential(
            nn.Linear(hidden_dimension, self.intermediate_dimension),
            nn.LayerNorm(self.intermediate_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.intermediate_dimension, 1)
        )
        
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        temperature: float = TEMPERATURE
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # hidden_states: (batch, seq_len, hidden_dim)
        scores = self.network(hidden_states).squeeze(-1)  # shape: (batch, seq_len)
        probabilities = torch.sigmoid(scores / temperature)
        return scores, probabilities

    def get_config(self) -> dict:
        return {
            'hidden_dimension': self.hidden_dimension,
            'intermediate_dimension': self.intermediate_dimension,
            'model_type': 'RetentionScorer',
        }


class SoftRetentionGate(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        probabilities: torch.Tensor
    ) -> torch.Tensor:
        return hidden_states * probabilities.unsqueeze(-1)


class AdaptiveMemoryRetention(nn.Module):
    def __init__(
        self,
        hidden_dimension: int = HIDDEN_DIMENSION,
        dropout: float = RETENTION_SCORER_DROPOUT
    ):
        super().__init__()
        self.scorer = RetentionScorer(hidden_dimension, dropout)
        self.gate = SoftRetentionGate()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        protected_mask: torch.Tensor,
        temperature: float = TEMPERATURE
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores, probabilities = self.scorer(hidden_states, temperature)
        protected_mask_device = protected_mask.to(probabilities.device)
        
        probabilities = torch.where(
            protected_mask_device.unsqueeze(0),
            torch.ones_like(probabilities),
            probabilities
        )
        gated_hidden_states = self.gate(hidden_states, probabilities)
        return gated_hidden_states, probabilities, scores


def physical_compaction(
    hidden_states: torch.Tensor,
    probabilities: torch.Tensor,
    attention_mask: torch.Tensor,
    original_indices: torch.Tensor,
    target_budget: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert hidden_states.shape[0] == 1, "Physical compaction only supports batch_size=1."
    
    seq_len = hidden_states.shape[1]
    if seq_len <= target_budget:
        return hidden_states, attention_mask, original_indices
        
    _, topk_indices_unordered = torch.topk(probabilities[0], target_budget)
    topk_indices, _ = torch.sort(topk_indices_unordered)
    
    topk_indices_hidden = topk_indices.unsqueeze(0).unsqueeze(-1).expand(1, target_budget, hidden_states.shape[-1])
    compacted_hidden = torch.gather(hidden_states, 1, topk_indices_hidden)
    compacted_mask = torch.gather(attention_mask, 1, topk_indices.unsqueeze(0))
    compacted_indices = torch.gather(original_indices, 0, topk_indices)
    
    return compacted_hidden, compacted_mask, compacted_indices


class LayerwiseAdaptiveMemory(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        hidden_dimension: int = HIDDEN_DIMENSION,
        dropout: float = RETENTION_SCORER_DROPOUT
    ):
        super().__init__()
        self.num_layers = num_layers
        self.scorers = nn.ModuleList([
            RetentionScorer(hidden_dimension, dropout) for _ in range(num_layers)
        ])
    
    def forward(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.scorers[layer_idx](hidden_states, temperature)