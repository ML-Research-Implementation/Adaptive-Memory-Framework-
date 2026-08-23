"""
Adaptive DistilBERT model with layer-wise retention mechanism.

This module implements a custom forward pass through DistilBERT where:
- After each Transformer layer, retention scores are computed
- Top-K tokens are selected based on retention probability
- The reduced token sequence is passed to the next layer
- Special tokens ([CLS], [SEP]) are always protected

This enables in-pipeline adaptive computation rather than end-of-pipeline retention.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from transformers import DistilBertForQuestionAnswering
from config import MODEL_NAME, DEVICE, HIDDEN_DIMENSION
from src.models import RetentionScorer

class HardConcreteGate(nn.Module):
    """
    Hard-Concrete (or Gumbel-Softmax) gate for differentiable binary decisions.
    Outputs continuous z in [0, 1] during training, and discrete z in {0, 1} at inference.
    """
    def __init__(self, temperature=0.5, stretch_min=-0.1, stretch_max=1.1):
        super().__init__()
        self.temp = temperature
        self.stretch_min = stretch_min
        self.stretch_max = stretch_max
        
    def forward(self, logits: torch.Tensor, training: bool = True, threshold_bias: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            logits: Unnormalized log probabilities (batch, seq_len)
            training: If True, adds Gumbel noise.
            threshold_bias: Optional bias to adjust retention rate during inference.
            
        Returns:
            z: Gate values (batch, seq_len) in [0, 1]
            l0_penalty: Expected probability of keeping the token, for budget constraint
        """
        if training:
            u = torch.rand_like(logits)
            # Logistic noise
            noise = torch.log(u + 1e-8) - torch.log(1 - u + 1e-8)
            s = torch.sigmoid((logits + noise) / self.temp)
        else:
            # Deterministic at inference, apply threshold bias
            s = torch.sigmoid(logits + threshold_bias)
            
        # Stretch
        s_stretched = s * (self.stretch_max - self.stretch_min) + self.stretch_min
        # Hard clamp
        z = torch.clamp(s_stretched, 0.0, 1.0)
        
        # Exact expected L0 penalty (P(z > 0))
        shift = -self.stretch_min / (self.stretch_max - self.stretch_min)
        l0_penalty = torch.sigmoid(logits - self.temp * torch.log(torch.tensor(shift / (1 - shift), device=logits.device)))
        
        return z, l0_penalty



class TokenSelectionResult:
    """Container for token selection outputs."""
    
    def __init__(
        self,
        selected_indices: torch.Tensor,
        selected_hidden_states: torch.Tensor,
        new_attention_mask: torch.Tensor,
        retention_scores: torch.Tensor,
        retention_probs: torch.Tensor,
        num_selected: int,
        num_original: int
    ):
        """
        Initialize token selection result.
        
        Args:
            selected_indices: Indices of selected tokens in original sequence (seq_len_selected,)
            selected_hidden_states: Hidden states of selected tokens (batch, seq_len_selected, hidden_dim)
            new_attention_mask: Updated attention mask (batch, seq_len_selected)
            retention_scores: Raw retention scores for all tokens (batch, seq_len_original)
            retention_probs: Retention probabilities for all tokens (batch, seq_len_original)
            num_selected: Number of tokens selected
            num_original: Original number of tokens
        """
        self.selected_indices = selected_indices
        self.selected_hidden_states = selected_hidden_states
        self.new_attention_mask = new_attention_mask
        self.retention_scores = retention_scores
        self.retention_probs = retention_probs
        self.num_selected = num_selected
        self.num_original = num_original
        self.retention_ratio = num_selected / num_original if num_original > 0 else 1.0


class TokenSelector:
    """
    Handles adaptive token selection using Hard-Concrete gates.
    """
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or DEVICE
        self.gate = HardConcreteGate()
    
    def select_adaptive(
        self,
        hidden_states: torch.Tensor,
        retention_scores: torch.Tensor,
        protected_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        training: bool = True,
        threshold_bias: float = 0.0
    ) -> TokenSelectionResult:
        """
        Dynamically selects tokens based on Hard-Concrete gates.
        Physically compacts the tensor, padding only to the maximum retained length in the batch.
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # 1. Compute differentiable gate and L0 penalty
        z, l0_penalty = self.gate(retention_scores, training=training, threshold_bias=threshold_bias)
        
        # 2. Force protected tokens to be kept (z=1)
        z = torch.where(protected_mask, torch.ones_like(z), z)
        
        # Also force padding tokens to 0 so they don't contribute to budget or get selected
        padding_mask = attention_mask < 0.5
        z = torch.where(padding_mask, torch.zeros_like(z), z)
        l0_penalty = torch.where(padding_mask, torch.zeros_like(l0_penalty), l0_penalty)
        
        # 3. Determine binary keep mask based on z > 0
        keep_mask = z > 0
        
        # Calculate how many tokens are retained per example
        retained_counts = keep_mask.sum(dim=1)
        max_retained = retained_counts.max().item()
        
        # If nothing is retained (shouldn't happen due to protected tokens), fallback
        if max_retained == 0:
            max_retained = 1
            
        # 4. Multiply hidden states by continuous z for gradient flow
        gated_hidden = hidden_states * z.unsqueeze(-1)
        
        # 5. Extract selected indices while preserving temporal order
        # We assign a high penalty to dropped tokens so they sort to the end
        indices = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        sort_keys = indices + (~keep_mask).long() * 10000
        
        _, sorted_indices = torch.sort(sort_keys, dim=1)
        
        # Take only up to max_retained
        selected_indices = sorted_indices[:, :max_retained]
        
        # 6. Gather the physically compacted tensors
        expanded_indices = selected_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
        selected_hidden_states = torch.gather(gated_hidden, 1, expanded_indices)
        
        # 7. Update Attention Mask
        # We gathered tokens up to max_retained. Some might be dropped tokens (padding for the batch).
        # We need to set their attention mask to 0.
        new_attention_mask = torch.gather(attention_mask, 1, selected_indices)
        is_retained = torch.gather(keep_mask, 1, selected_indices)
        new_attention_mask = new_attention_mask * is_retained.float()
        
        return TokenSelectionResult(
            selected_indices=selected_indices,
            selected_hidden_states=selected_hidden_states,
            new_attention_mask=new_attention_mask,
            retention_scores=retention_scores,
            retention_probs=l0_penalty,  # Store expected penalty here for convenience
            num_selected=max_retained,
            num_original=seq_len
        )


class AdaptiveDistilBertQA(nn.Module):
    """
    DistilBERT QA model with layer-wise retention mechanism.
    
    Architecture:
        Input (31 tokens)
          ↓
        Embedding layer
          ↓
        Layer 1 → RetentionScorer → Top-K Selection → reduced tokens (e.g., 23)
          ↓
        Layer 2 → RetentionScorer → Top-K Selection → reduced tokens (e.g., 17)
          ↓
        ...
        Layer 6 → final hidden states
          ↓
        QA Head (start/end predictions)
    
    All Transformer parameters are frozen; only RetentionScorer is trained.
    """
    
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[torch.device] = None,
        freeze_transformer: bool = True,
        hidden_dimension: int = HIDDEN_DIMENSION,
        apply_retention_per_layer: Optional[List[bool]] = None,
        retention_schedule: Optional[List[float]] = None
    ):
        """
        Initialize adaptive DistilBERT QA model.
        
        Args:
            model_name: Pretrained model identifier
            device: Device for model
            freeze_transformer: Whether to freeze Transformer parameters
            hidden_dimension: Hidden dimension (768 for DistilBERT)
            apply_retention_per_layer: List of bools indicating which layers to apply retention to.
                                      If None, apply to all layers.
            retention_schedule: List of target retention ratios for each layer (0.0 to 1.0)
        """
        super().__init__()
        
        self.model_name = model_name
        self.device = device or DEVICE
        self.hidden_dimension = hidden_dimension
        self.retention_schedule = retention_schedule or [0.90, 0.85, 0.80, 0.75, 0.70, 0.70]
        self.num_layers = 6
        
        # Load pretrained model
        self.model = DistilBertForQuestionAnswering.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Extract components
        self.distilbert = self.model.distilbert
        self.qa_outputs = self.model.qa_outputs
        
        # Freeze Transformer if requested
        if freeze_transformer:
            for param in self.distilbert.parameters():
                param.requires_grad = False
        
        # Initialize retention scorers for each layer
        self.retention_scorers = nn.ModuleList([
            RetentionScorer(hidden_dimension)
            for _ in range(self.num_layers)
        ])
        
        # Send scorers to device
        for scorer in self.retention_scorers:
            scorer.to(self.device)
        
        # Configure which layers have retention
        if apply_retention_per_layer is None:
            self.apply_retention_per_layer = [True] * self.num_layers
        else:
            self.apply_retention_per_layer = apply_retention_per_layer
        
        # Token selector
        self.token_selector = TokenSelector(device=self.device)
    
    def create_protected_mask(
        self,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Create mask for tokens that must always be retained.
        
        Protected tokens: [CLS] (token_id=101), [SEP] (token_id=102)
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            
        Returns:
            Boolean mask (seq_len,) where True = protected
        """
        # Mark [CLS] (101) and [SEP] (102) as protected
        protected_mask = (input_ids == 101) | (input_ids == 102)
        
        return protected_mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_layer_metrics: bool = True,
        training: bool = False,
        threshold_bias: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Forward pass with layer-wise retention.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            return_layer_metrics: Whether to return per-layer metrics
            
        Returns:
            Tuple of:
                - start_logits: Start position logits (batch, seq_len_original)
                - end_logits: End position logits (batch, seq_len_original)
                - layer_metrics: Dictionary with per-layer metrics (if return_layer_metrics=True)
        """
        batch_size = input_ids.shape[0]
        original_seq_len = input_ids.shape[1]
        
        # Initialize layer metrics
        layer_metrics = {
            'tokens_per_layer': [],
            'retention_ratios': [],
            'selection_results': [],
            'expected_retained_tokens': 0.0  # Accumulate global budget here
        }
        
        # Create protected mask
        protected_mask = self.create_protected_mask(input_ids)
        
        # Get embeddings
        embedding_output = self.distilbert.embeddings(input_ids)
        
        # Process through layers with retention
        hidden_states = embedding_output
        current_input_ids = input_ids
        current_attention_mask = attention_mask
        
        # Track token indices for reconstruction
        token_index_mapping = torch.arange(original_seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        
        for layer_idx, layer in enumerate(self.distilbert.transformer.layer):
            # Apply Transformer layer
            # Note: DistilBERT TransformerBlock uses attn_mask (not attention_mask)
            # Convert 2D attention mask (batch, seq_len) to bias format if needed
            attn_bias = None
            if current_attention_mask is not None:
                # Create attention bias from mask: 1 -> 0 (attend), 0 -> -1e9 (ignore)
                attn_bias = (1.0 - current_attention_mask[:, None, None, :]) * -1e9
            
            # Pass hidden_states as positional argument to support different transformers versions
            layer_output = layer(
                hidden_states,
                attn_mask=attn_bias
            )
            
            if isinstance(layer_output, tuple):
                hidden_states = layer_output[0]
            else:
                hidden_states = layer_output
            
            # Record tokens before retention
            tokens_before = hidden_states.shape[1]
            layer_metrics['tokens_per_layer'].append(tokens_before)
            
            # Apply retention if configured for this layer
            if self.apply_retention_per_layer[layer_idx]:
                # Compute retention scores using the linear layer
                # We ignore the probs returned by RetentionScorer because HardConcreteGate handles it
                scores, _ = self.retention_scorers[layer_idx](
                    hidden_states,
                    temperature=1.0
                )
                
                # Select tokens adaptively
                selection_result = self.token_selector.select_adaptive(
                    hidden_states=hidden_states,
                    retention_scores=scores,
                    protected_mask=protected_mask,
                    attention_mask=current_attention_mask,
                    training=training,
                    threshold_bias=threshold_bias
                )
                
                # Update hidden states and attention mask
                hidden_states = selection_result.selected_hidden_states
                current_attention_mask = selection_result.new_attention_mask
                
                # Update token mapping for later reconstruction
                protected_mask = torch.gather(protected_mask, 1, selection_result.selected_indices)
                token_index_mapping = torch.gather(token_index_mapping, 1, selection_result.selected_indices)
                
                # Record metrics and expected tokens
                # We sum the expected kept tokens per batch element, and mean over the batch
                expected_kept = selection_result.retention_probs.sum(dim=1).mean()
                layer_metrics['expected_retained_tokens'] += expected_kept
                
                layer_metrics['retention_ratios'].append(selection_result.retention_ratio)
                layer_metrics['selection_results'].append(selection_result)
                
                # Save the hidden states for distillation
                if 'hidden_states' not in layer_metrics:
                    layer_metrics['hidden_states'] = []
                layer_metrics['hidden_states'].append(hidden_states)
            else:
                layer_metrics['retention_ratios'].append(1.0)
                layer_metrics['selection_results'].append(None)
                
                if 'hidden_states' not in layer_metrics:
                    layer_metrics['hidden_states'] = []
                layer_metrics['hidden_states'].append(hidden_states)
        
        # Get final QA logits
        # QA head returns tensor of shape (batch, seq_len, 2)
        # where last dimension is [start_logits, end_logits] for each token
        qa_logits_output = self.qa_outputs(hidden_states)  # (batch, seq_len_final, 2)
        start_logits_final = qa_logits_output[:, :, 0]  # (batch, seq_len_final)
        end_logits_final = qa_logits_output[:, :, 1]
        
        # Pad/reconstruct logits to original sequence length
        start_logits_padded = torch.full(
            (batch_size, original_seq_len),
            -1e4,
            device=self.device
        )
        end_logits_padded = torch.full(
            (batch_size, original_seq_len),
            -1e4,
            device=self.device
        )
        
        # Scatter the logits back to their original positions
        start_logits_padded.scatter_(1, token_index_mapping, start_logits_final)
        end_logits_padded.scatter_(1, token_index_mapping, end_logits_final)
        
        if return_layer_metrics:
            return start_logits_padded, end_logits_padded, layer_metrics
        else:
            return start_logits_padded, end_logits_padded, None
    
    def get_retention_scorers(self) -> nn.ModuleList:
        """Get the retention scorer modules for training."""
        return self.retention_scorers
    
    def freeze_scorers(self):
        """Freeze all retention scorers."""
        for scorer in self.retention_scorers:
            for param in scorer.parameters():
                param.requires_grad = False
    
    def unfreeze_scorers(self):
        """Unfreeze all retention scorers for training."""
        for scorer in self.retention_scorers:
            for param in scorer.parameters():
                param.requires_grad = True


class AdaptiveQAInference:
    """
    High-level interface for running adaptive QA inference and comparison.
    
    Supports:
    - Running adaptive forward pass with different retention ratios
    - Comparing with baseline
    - Extracting layer-wise metrics
    """
    
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[torch.device] = None,
        retention_schedule: Optional[List[float]] = None
    ):
        """
        Initialize adaptive QA inference wrapper.
        
        Args:
            model_name: Pretrained model identifier
            device: Device for computation
            retention_schedule: Target retention ratio for each layer
        """
        self.device = device or DEVICE
        self.retention_schedule = retention_schedule or [0.90, 0.85, 0.80, 0.75, 0.70, 0.70]
        
        # Create adaptive model
        self.adaptive_model = AdaptiveDistilBertQA(
            model_name=model_name,
            device=self.device,
            freeze_transformer=True,
            retention_schedule=self.retention_schedule
        )
        self.adaptive_model.eval()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[int, int, Dict]:
        """
        Run adaptive forward pass and extract answer predictions.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            
        Returns:
            Tuple of:
                - start_idx: Predicted start position
                - end_idx: Predicted end position
                - metrics_dict: Dictionary with layer-wise metrics
        """
        with torch.no_grad():
            start_logits, end_logits, layer_metrics = self.adaptive_model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                return_layer_metrics=True
            )
        
        # Extract predictions
        start_idx = torch.argmax(start_logits, dim=-1).item()
        end_idx = torch.argmax(end_logits, dim=-1).item()
        
        # Ensure valid span
        if end_idx < start_idx:
            start_idx, end_idx = end_idx, start_idx
        
        return start_idx, end_idx, layer_metrics
    
    def set_retention_schedule(self, retention_schedule: List[float]):
        """Update retention schedule for all layers."""
        self.retention_schedule = retention_schedule
        self.adaptive_model.retention_schedule = retention_schedule
