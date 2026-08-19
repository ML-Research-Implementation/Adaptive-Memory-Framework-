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
    Handles deterministic Top-K token selection with protection for special tokens.
    """
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        min_tokens_to_keep: int = 3
    ):
        """
        Initialize token selector.
        
        Args:
            device: Device for tensor operations.
            min_tokens_to_keep: Minimum number of tokens to retain (including protected).
        """
        self.device = device or DEVICE
        self.min_tokens_to_keep = min_tokens_to_keep
    
    def select_top_k(
        self,
        hidden_states: torch.Tensor,
        retention_probs: torch.Tensor,
        retention_scores: torch.Tensor,
        protected_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        retention_ratio: float
    ) -> TokenSelectionResult:
        """
        Select top-K tokens based on retention probability, protecting special tokens.
        
        Args:
            hidden_states: Token hidden states (batch, seq_len, hidden_dim)
            retention_probs: Retention probabilities (batch, seq_len)
            retention_scores: Raw retention scores (batch, seq_len)
            protected_mask: Boolean mask for protected tokens (seq_len,) - True if protected
            attention_mask: Original attention mask (batch, seq_len) - 1 if valid, 0 if padding
            retention_ratio: Target retention ratio (0.0 to 1.0)
            
        Returns:
            TokenSelectionResult with selected tokens and metadata
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # For now, assume batch_size = 1 (single example)
        assert batch_size == 1, "Batch processing not yet supported"
        
        # Get original sequence
        hidden_states_sq = hidden_states[0]  # (seq_len, hidden_dim)
        probs_sq = retention_probs[0]  # (seq_len,)
        scores_sq = retention_scores[0]  # (seq_len,)
        attention_sq = attention_mask[0]  # (seq_len,)
        
        # Count valid (non-padding) tokens
        valid_mask = attention_sq > 0.5  # (seq_len,)
        num_valid = valid_mask.sum().item()
        
        # Count protected tokens
        protected_mask_device = protected_mask.to(self.device)
        num_protected = protected_mask_device.sum().item()
        
        # Calculate target number of tokens to select
        target_count = max(
            self.min_tokens_to_keep,
            int(num_valid * retention_ratio)
        )
        
        # Ensure we don't select more than available
        target_count = min(target_count, num_valid)
        
        # Number of tokens to select from non-protected set
        tokens_to_select_from_adaptive = target_count - num_protected
        tokens_to_select_from_adaptive = max(0, tokens_to_select_from_adaptive)
        
        # Create selection mask
        selection_mask = torch.zeros(seq_len, dtype=torch.bool, device=self.device)
        
        # Always select protected tokens
        selection_mask = selection_mask | protected_mask_device
        
        # Select top-K from non-protected, valid tokens
        non_protected_valid = (~protected_mask_device) & valid_mask
        
        if tokens_to_select_from_adaptive > 0 and non_protected_valid.sum() > 0:
            # Get scores for non-protected valid tokens
            scores_adaptive = scores_sq.clone()
            scores_adaptive[~non_protected_valid] = float('-inf')
            
            # Select top-k
            _, top_indices = torch.topk(
                scores_adaptive,
                k=min(tokens_to_select_from_adaptive, non_protected_valid.sum().item()),
                dim=0
            )
            selection_mask[top_indices] = True
        
        # Get selected indices and hidden states
        selected_indices = torch.where(selection_mask)[0]  # (num_selected,)
        selected_hidden_states = hidden_states_sq[selected_indices]  # (num_selected, hidden_dim)
        
        # Create new attention mask for selected tokens
        new_attention_mask = torch.ones(
            1,
            len(selected_indices),
            dtype=attention_mask.dtype,
            device=self.device
        )
        
        num_selected = len(selected_indices)
        
        return TokenSelectionResult(
            selected_indices=selected_indices,
            selected_hidden_states=selected_hidden_states.unsqueeze(0),  # (1, num_selected, hidden_dim)
            new_attention_mask=new_attention_mask,
            retention_scores=scores_sq,
            retention_probs=probs_sq,
            num_selected=num_selected,
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
        retention_ratio: float = 0.75
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
            retention_ratio: Target retention ratio for each layer (0.0 to 1.0)
        """
        super().__init__()
        
        self.model_name = model_name
        self.device = device or DEVICE
        self.hidden_dimension = hidden_dimension
        self.retention_ratio = retention_ratio
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
        seq_len = input_ids.shape[1]
        protected_mask = torch.zeros(seq_len, dtype=torch.bool, device=self.device)
        
        # For single example (batch_size=1), check first row
        input_ids_sq = input_ids[0]
        
        # Mark [CLS] (101) and [SEP] (102) as protected
        protected_mask = (input_ids_sq == 101) | (input_ids_sq == 102)
        
        return protected_mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_layer_metrics: bool = True
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
            'selection_results': []
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
        token_index_mapping = torch.arange(original_seq_len, device=self.device)
        
        for layer_idx, layer in enumerate(self.distilbert.transformer.layer):
            # Apply Transformer layer
            # Note: DistilBERT TransformerBlock uses attn_mask (not attention_mask)
            # Convert 2D attention mask (batch, seq_len) to bias format if needed
            attn_bias = None
            if current_attention_mask is not None:
                # Create attention bias from mask: 1 -> 0 (attend), 0 -> -1e9 (ignore)
                attn_bias = (1.0 - current_attention_mask[:, None, None, :]) * -1e9
            
            layer_output = layer(
                x=hidden_states,
                attn_mask=attn_bias
            )
            hidden_states = layer_output[0]  # (batch, seq_len, hidden_dim)
            
            # Record tokens before retention
            tokens_before = hidden_states.shape[1]
            layer_metrics['tokens_per_layer'].append(tokens_before)
            
            # Apply retention if configured for this layer
            if self.apply_retention_per_layer[layer_idx]:
                # Compute retention scores
                scores, probs = self.retention_scorers[layer_idx](
                    hidden_states,
                    temperature=1.0
                )
                
                # Select top-K tokens
                selection_result = self.token_selector.select_top_k(
                    hidden_states=hidden_states,
                    retention_probs=probs,
                    retention_scores=scores,
                    protected_mask=protected_mask,
                    attention_mask=current_attention_mask,
                    retention_ratio=self.retention_ratio
                )
                
                # Update hidden states and attention mask
                hidden_states = selection_result.selected_hidden_states
                current_attention_mask = selection_result.new_attention_mask
                
                # Update token mapping for later reconstruction
                protected_mask = protected_mask[selection_result.selected_indices]
                token_index_mapping = token_index_mapping[selection_result.selected_indices]
                
                # Record metrics
                layer_metrics['retention_ratios'].append(selection_result.retention_ratio)
                layer_metrics['selection_results'].append(selection_result)
            else:
                layer_metrics['retention_ratios'].append(1.0)
                layer_metrics['selection_results'].append(None)
        
        # Get final QA logits
        # QA head returns tensor of shape (batch, seq_len, 2)
        # where last dimension is [start_logits, end_logits] for each token
        qa_logits_output = self.qa_outputs(hidden_states)  # (batch, seq_len_final, 2)
        start_logits_final = qa_logits_output[:, :, 0]  # (batch, seq_len_final)
        end_logits_final = qa_logits_output[:, :, 1]
        
        # Pad/reconstruct logits to original sequence length
        # For now, create zero-padded versions
        start_logits_padded = torch.zeros(
            batch_size,
            original_seq_len,
            device=self.device
        )
        end_logits_padded = torch.zeros(
            batch_size,
            original_seq_len,
            device=self.device
        )
        
        # Fill in the values at the positions of selected tokens
        for i, idx in enumerate(token_index_mapping):
            start_logits_padded[0, idx] = start_logits_final[0, i]
            end_logits_padded[0, idx] = end_logits_final[0, i]
        
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
        retention_ratio: float = 0.75
    ):
        """
        Initialize adaptive QA inference wrapper.
        
        Args:
            model_name: Pretrained model identifier
            device: Device for computation
            retention_ratio: Target retention ratio for each layer
        """
        self.device = device or DEVICE
        self.retention_ratio = retention_ratio
        
        # Create adaptive model
        self.adaptive_model = AdaptiveDistilBertQA(
            model_name=model_name,
            device=self.device,
            freeze_transformer=True,
            retention_ratio=retention_ratio
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
    
    def set_retention_ratio(self, retention_ratio: float):
        """Update retention ratio for all layers."""
        self.retention_ratio = retention_ratio
        self.adaptive_model.retention_ratio = retention_ratio
