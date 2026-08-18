"""
Neural network models for AMMR framework.
Includes the learnable retention scorer and other model components.
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
    
    Architecture:
        h_t (hidden_dimension)
          ↓
        Linear + LayerNorm + GELU + Dropout
          ↓
        Linear (output: 1)
          ↓
        Sigmoid (with temperature)
          ↓
        p_t (retention probability in [0, 1])
    
    The retention probability p_t indicates how strongly each token should
    be retained. p_t close to 1 means keep the token, p_t close to 0 means
    discard the token.
    
    Attributes:
        network: Sequential MLP for scoring.
        hidden_dimension: Size of input hidden states.
        intermediate_dimension: Size of hidden layer.
    """
    
    def __init__(
        self,
        hidden_dimension: int = HIDDEN_DIMENSION,
        dropout: float = RETENTION_SCORER_DROPOUT,
        intermediate_dim_ratio: int = RETENTION_SCORER_INTERMEDIATE_DIM_RATIO
    ):
        """
        Initialize retention scorer.
        
        Args:
            hidden_dimension: Size of input hidden states (default 768 for DistilBERT).
            dropout: Dropout rate in the network.
            intermediate_dim_ratio: Intermediate layer is hidden_dimension / this ratio.
        """
        super().__init__()
        
        self.hidden_dimension = hidden_dimension
        
        # Compute intermediate dimension
        self.intermediate_dimension = max(
            64,
            hidden_dimension // intermediate_dim_ratio
        )
        
        # Build MLP
        self.network = nn.Sequential(
            nn.Linear(hidden_dimension, self.intermediate_dimension),
            nn.LayerNorm(self.intermediate_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.intermediate_dimension, 1)
        )
        
        # Initialize final layer close to 0 for ~0.5 initial probability
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        temperature: float = TEMPERATURE
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute retention scores and probabilities.
        
        Args:
            hidden_states: Token hidden states of shape (batch, seq_len, hidden_dim).
            temperature: Temperature for probability scaling. Lower = sharper decisions.
            
        Returns:
            Tuple of:
                - scores: Raw scores from MLP, shape (batch, seq_len).
                - probabilities: Retention probabilities after sigmoid, shape (batch, seq_len).
        """
        # Compute raw scores
        scores = self.network(hidden_states)  # (batch, seq_len, 1)
        scores = scores.squeeze(-1)  # (batch, seq_len)
        
        # Convert to probabilities with temperature scaling
        # Temperature controls probability sharpness:
        # - T < 1.0: sharper (closer to 0 or 1)
        # - T = 1.0: default
        # - T > 1.0: softer (closer to 0.5)
        probabilities = torch.sigmoid(scores / temperature)  # (batch, seq_len)
        
        return scores, probabilities
    
    def get_config(self) -> dict:
        """
        Return configuration dictionary for logging/reproduction.
        
        Returns:
            Dictionary with model configuration.
        """
        return {
            'hidden_dimension': self.hidden_dimension,
            'intermediate_dimension': self.intermediate_dimension,
            'model_type': 'RetentionScorer',
        }


class SoftRetentionGate(nn.Module):
    """
    Applies soft retention gate to hidden states.
    
    Instead of hard selection (keep/discard), this applies probabilistic
    gating: h'_t = p_t * h_t, where p_t is the retention probability.
    
    This is differentiable and allows end-to-end training.
    """
    
    def __init__(self):
        """Initialize soft retention gate (no parameters)."""
        super().__init__()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        probabilities: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply soft retention gate.
        
        Args:
            hidden_states: Token representations of shape (batch, seq_len, hidden_dim).
            probabilities: Retention probabilities of shape (batch, seq_len).
            
        Returns:
            Gated hidden states of same shape, with reduced magnitude for low-probability tokens.
        """
        # Add dimension for broadcasting: (batch, seq_len) -> (batch, seq_len, 1)
        prob_expanded = probabilities.unsqueeze(-1)
        
        # Element-wise multiplication
        gated = hidden_states * prob_expanded
        
        return gated


class AdaptiveMemoryRetention(nn.Module):
    """
    Complete adaptive memory retention module combining scorer and gate.
    
    This module:
    1. Takes token hidden states from Transformer
    2. Scores each token using RetentionScorer
    3. Converts scores to retention probabilities
    4. Protects special tokens
    5. Applies soft retention gate
    
    Attributes:
        scorer: RetentionScorer network.
        gate: SoftRetentionGate module.
    """
    
    def __init__(
        self,
        hidden_dimension: int = HIDDEN_DIMENSION,
        dropout: float = RETENTION_SCORER_DROPOUT
    ):
        """
        Initialize adaptive memory retention module.
        
        Args:
            hidden_dimension: Size of hidden states.
            dropout: Dropout rate for scorer.
        """
        super().__init__()
        self.scorer = RetentionScorer(hidden_dimension, dropout)
        self.gate = SoftRetentionGate()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        protected_mask: torch.Tensor,
        temperature: float = TEMPERATURE
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through retention module.
        
        Args:
            hidden_states: Token hidden states (batch, seq_len, hidden_dim).
            protected_mask: Mask for protected tokens (batch, seq_len).
                            True for special tokens that must be retained.
            temperature: Temperature for probability scaling.
            
        Returns:
            Tuple of:
                - gated_hidden_states: Soft-gated representations.
                - probabilities: Retention probabilities after protection.
                - scores: Raw scores from scorer.
        """
        # Score tokens
        scores, probabilities = self.scorer(hidden_states, temperature)
        
        # Protect special tokens - force probability to 1.0
        # Ensure protected_mask is on same device
        protected_mask_device = protected_mask.to(probabilities.device)
        
        probabilities = torch.where(
            protected_mask_device.unsqueeze(0),
            torch.ones_like(probabilities),
            probabilities
        )
        
        # Apply soft gate
        gated_hidden_states = self.gate(hidden_states, probabilities)
        
        return gated_hidden_states, probabilities, scores
