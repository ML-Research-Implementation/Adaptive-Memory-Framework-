"""
Baseline DistilBERT QA model setup and evaluation.
Handles loading pretrained model, freezing parameters, and baseline predictions.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from transformers import DistilBertForQuestionAnswering
from config import MODEL_NAME, DEVICE
from src.utils import count_parameters, freeze_model, print_header
from src.data import QADataLoader


class BaselineQAModel:
    """
    Wrapper for baseline DistilBERT QA model.
    
    This class handles:
    - Loading pretrained DistilBERT
    - Freezing model parameters
    - Computing baseline predictions
    - Extracting hidden states for retention
    """
    
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[torch.device] = None,
        freeze_parameters: bool = True
    ):
        """
        Initialize baseline QA model.
        
        Args:
            model_name: Pretrained model identifier.
            device: Device for model (defaults to config.DEVICE).
            freeze_parameters: Whether to freeze all parameters (True for frozen baseline).
        """
        self.model_name = model_name
        self.device = device or DEVICE
        
        # Load pretrained model
        self.model = DistilBertForQuestionAnswering.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Extract encoder for hidden state access
        self.encoder = self.model.distilbert
        
        # Freeze parameters if requested
        if freeze_parameters:
            freeze_model(self.model)
        
        self.num_parameters = count_parameters(self.model)
        self.num_layers = 6  # DistilBERT has 6 layers
        self.hidden_dim = 768
    
    @property
    def qa_model(self) -> nn.Module:
        """Get the QA model for use in training (alias for self.model)."""
        return self.model
    
    def get_baseline_prediction(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[int, int, str, Dict]:
        """
        Get baseline QA prediction without any retention mechanism.
        
        Args:
            input_ids: Token IDs (batch, seq_len).
            attention_mask: Attention mask (batch, seq_len).
            
        Returns:
            Tuple of:
                - start_index: Predicted start position.
                - end_index: Predicted end position.
                - answer_text: Decoded answer text (requires tokens).
                - info_dict: Dictionary with logits and confidence.
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        start_logits = outputs.start_logits  # (batch, seq_len)
        end_logits = outputs.end_logits      # (batch, seq_len)
        
        # Get argmax predictions
        start_idx = torch.argmax(start_logits, dim=-1).item()
        end_idx = torch.argmax(end_logits, dim=-1).item()
        
        # Compute confidence scores
        start_prob = torch.softmax(start_logits, dim=-1)[0, start_idx].item()
        end_prob = torch.softmax(end_logits, dim=-1)[0, end_idx].item()
        
        info_dict = {
            'start_logits': start_logits.cpu(),
            'end_logits': end_logits.cpu(),
            'start_prob': start_prob,
            'end_prob': end_prob,
            'start_idx': start_idx,
            'end_idx': end_idx,
        }
        
        return start_idx, end_idx, None, info_dict
    
    def get_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_all_layers: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get hidden states from DistilBERT encoder.
        
        Args:
            input_ids: Token IDs (batch, seq_len).
            attention_mask: Attention mask (batch, seq_len).
            return_all_layers: If True, return all layer outputs; if False, return only final.
            
        Returns:
            Tuple of:
                - final_hidden_states: Last layer output (batch, seq_len, hidden_dim).
                - all_hidden_states: All layer outputs if requested, else None.
        """
        with torch.no_grad():
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        
        final_hidden_states = outputs.last_hidden_state
        
        all_hidden_states = None
        if return_all_layers:
            all_hidden_states = outputs.hidden_states
        
        return final_hidden_states, all_hidden_states
    
    def get_qa_head(self) -> nn.Module:
        """
        Get the QA classification head (for computing logits on gated representations).
        
        Returns:
            The qa_outputs module.
        """
        return self.model.qa_outputs
    
    def print_info(self):
        """Print model information."""
        print_header("BASELINE MODEL INFO")
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Total Parameters: {self.num_parameters:,}")
        print(f"Layers: {self.num_layers}")
        print(f"Hidden Dimension: {self.hidden_dim}")
        print(f"Parameters Frozen: Yes")


def compute_baseline_metrics(
    predictions: Dict,
    ground_truth_start: int,
    ground_truth_end: int,
    tokens: list
) -> Dict[str, float]:
    """
    Compute metrics comparing baseline prediction with ground truth.
    
    Args:
        predictions: Dictionary from get_baseline_prediction.
        ground_truth_start: Ground truth start index.
        ground_truth_end: Ground truth end index.
        tokens: List of token strings for decoding.
        
    Returns:
        Dictionary with metrics (EM, F1, confidence, etc.).
    """
    pred_start = predictions['start_idx']
    pred_end = predictions['end_idx']
    
    # Exact Match
    exact_match = (pred_start == ground_truth_start and pred_end == ground_truth_end)
    
    # Token overlap
    pred_tokens = set(range(pred_start, pred_end + 1))
    gt_tokens = set(range(ground_truth_start, ground_truth_end + 1))
    
    intersection = len(pred_tokens & gt_tokens)
    union = len(pred_tokens | gt_tokens)
    
    # Precision, Recall, F1
    if len(pred_tokens) > 0:
        precision = intersection / len(pred_tokens)
    else:
        precision = 0.0
    
    if len(gt_tokens) > 0:
        recall = intersection / len(gt_tokens)
    else:
        recall = 0.0
    
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    metrics = {
        'exact_match': float(exact_match),
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'start_confidence': predictions['start_prob'],
        'end_confidence': predictions['end_prob'],
        'mean_confidence': (predictions['start_prob'] + predictions['end_prob']) / 2,
    }
    
    return metrics
