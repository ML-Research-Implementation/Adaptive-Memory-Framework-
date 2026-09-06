import torch
import torch.nn as nn
import torch.nn.functional as F

class StochasticGumbelGate(nn.Module):
    """
    Computes a keep/drop decision for each token.
    Uses Gumbel-Softmax so gradients flow backward into the scorer.
    """
    def __init__(self, hidden_dim=768):
        super().__init__()
        # A lightweight 2-layer MLP that outputs 2 scores per token: [drop_score, keep_score]
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 2)
        )

    def forward(self, hidden_states, temperature=1.0, hard=True):
        """
        hidden_states: [batch_size, seq_len, 768]
        temperature: controls softness (higher = smooth/exploratory, lower = sharp/binary)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # 1. Compute logits: shape [batch_size, seq_len, 2]
        logits = self.scorer(hidden_states)
        
        if self.training:
            # 2. Gumbel-Softmax trick:
            # - Forward pass gives discrete 0s and 1s (if hard=True)
            # - Backward pass uses continuous probabilities for gradients
            gate_probs = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)
            keep_mask = gate_probs[:, :, 1]  # index 1 represents 'KEEP'
        else:
            # Inference: standard deterministic choice
            keep_mask = (logits[:, :, 1] > logits[:, :, 0]).float()
            
        # 3. CRITICAL: Never drop the [CLS] token (token 0)
        keep_mask[:, 0] = 1.0
        
        # 4. Apply the gate to zero out dropped token representations
        # shape of keep_mask: [batch_size, seq_len, 1]
        gated_hidden_states = hidden_states * keep_mask.unsqueeze(-1)
        
        return gated_hidden_states, keep_mask