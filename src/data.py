"""
Data loading and tokenization utilities for AMMR framework.
Handles tokenization, sequence processing, and answer span location.
"""

from typing import Dict, List, Tuple, Optional
import torch
from transformers import AutoTokenizer
from config import MODEL_NAME, MAX_SEQUENCE_LENGTH, TRUNCATION_ENABLED


class QADataLoader:
    """
    Handles tokenization and data preparation for QA tasks.
    """
    
    def __init__(self, model_name: str = MODEL_NAME):
        """
        Initialize QA data loader.
        
        Args:
            model_name: Pre-trained model name for tokenizer.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_name = model_name
    
    def tokenize_qa(
        self,
        question: str,
        context: str,
        device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize a question-context pair.
        
        Args:
            question: Question text.
            context: Context text.
            device: Target device for tensors.
            
        Returns:
            Dictionary with 'input_ids', 'attention_mask'.
        """
        encoded = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation=TRUNCATION_ENABLED,
            max_length=MAX_SEQUENCE_LENGTH
        )
        
        if device is not None:
            encoded = encoded.to(device)
        
        return encoded
    
    def tokenize_qa_with_offsets(
        self,
        question: str,
        context: str,
        device: Optional[torch.device] = None
    ) -> Dict:
        """
        Tokenize with offset mappings for span location.
        
        Args:
            question: Question text.
            context: Context text.
            device: Target device for tensors.
            
        Returns:
            Dictionary including offset_mapping and sequence_ids.
        """
        encoded = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=TRUNCATION_ENABLED,
            max_length=MAX_SEQUENCE_LENGTH
        )
        
        if device is not None:
            encoded = encoded.to(device)
        
        return encoded
    
    def get_sequence_ids(
        self,
        encoded: Dict,
        batch_index: int = 0
    ) -> List[Optional[int]]:
        """
        Get sequence IDs indicating question vs context tokens.
        
        Args:
            encoded: Output from tokenizer with return_tensors.
            batch_index: Batch index to extract.
            
        Returns:
            List where None=special, 0=question, 1=context.
        """
        return encoded.sequence_ids(batch_index=batch_index)
    
    def get_tokens(
        self,
        input_ids: torch.Tensor,
        batch_index: int = 0
    ) -> List[str]:
        """
        Convert token IDs to token strings.
        
        Args:
            input_ids: Tensor of token IDs.
            batch_index: Batch index to convert.
            
        Returns:
            List of token strings.
        """
        return self.tokenizer.convert_ids_to_tokens(input_ids[batch_index])
    
    def decode_span(
        self,
        input_ids: torch.Tensor,
        start_idx: int,
        end_idx: int,
        batch_index: int = 0,
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode a span of tokens.
        
        Args:
            input_ids: Tensor of token IDs.
            start_idx: Start token index (inclusive).
            end_idx: End token index (inclusive).
            batch_index: Batch index.
            skip_special_tokens: Whether to skip special tokens.
            
        Returns:
            Decoded text span.
        """
        return self.tokenizer.decode(
            input_ids[batch_index, start_idx:end_idx + 1],
            skip_special_tokens=skip_special_tokens
        )


def find_answer_span(
    tokenizer,
    context: str,
    answer_text: str,
    offset_mapping: torch.Tensor,
    sequence_ids: List[Optional[int]],
    batch_index: int = 0
) -> Tuple[int, int]:
    """
    Locate answer span in tokenized context using character offsets.
    
    Args:
        tokenizer: Tokenizer instance.
        context: Original context text.
        answer_text: Answer text to find.
        offset_mapping: Token offset mapping from tokenizer.
        sequence_ids: Sequence IDs (question=0, context=1, special=None).
        batch_index: Batch index.
        
    Returns:
        Tuple of (start_token_index, end_token_index).
        
    Raises:
        ValueError: If answer not found in context or tokens.
    """
    # Find character positions in context
    answer_start_char = context.find(answer_text)
    
    if answer_start_char == -1:
        raise ValueError(f"Answer text '{answer_text}' not found in context.")
    
    answer_end_char = answer_start_char + len(answer_text)
    
    # Find tokens that overlap with answer span
    answer_token_positions = []
    
    for token_idx, (offset, seq_id) in enumerate(
        zip(offset_mapping[batch_index].tolist(), sequence_ids)
    ):
        start_char, end_char = offset
        
        # Only consider context tokens (not question or special)
        if seq_id != 1:
            continue
        
        # Skip empty offsets
        if end_char <= start_char:
            continue
        
        # Check if token overlaps with answer span
        if start_char >= answer_start_char and end_char <= answer_end_char:
            answer_token_positions.append(token_idx)
    
    if not answer_token_positions:
        raise ValueError("Could not locate answer tokens in context.")
    
    start_idx = answer_token_positions[0]
    end_idx = answer_token_positions[-1]
    
    return start_idx, end_idx


def get_token_type(sequence_id: Optional[int]) -> str:
    """
    Get human-readable token type from sequence ID.
    
    Args:
        sequence_id: Sequence ID (None=special, 0=question, 1=context).
        
    Returns:
        Token type string.
    """
    if sequence_id is None:
        return "SPECIAL"
    elif sequence_id == 0:
        return "QUESTION"
    else:
        return "CONTEXT"


def create_token_masks(
    sequence_length: int,
    sequence_ids: List[Optional[int]],
    attention_mask: torch.Tensor,
    batch_index: int = 0,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create masks for valid and protected tokens.
    
    Protected tokens: special tokens ([CLS], [SEP], etc.)
    Valid (adaptive) tokens: question and context tokens with attention=1.
    
    Args:
        sequence_length: Total sequence length.
        sequence_ids: Sequence IDs from tokenizer.
        attention_mask: Attention mask from tokenizer.
        batch_index: Batch index.
        device: Target device.
        
    Returns:
        Tuple of (valid_mask, protected_mask).
    """
    if device is None:
        device = torch.device("cpu")
    
    valid_mask = torch.zeros(sequence_length, dtype=torch.bool, device=device)
    protected_mask = torch.zeros(sequence_length, dtype=torch.bool, device=device)
    
    for idx, seq_id in enumerate(sequence_ids):
        if seq_id is None:
            # Special token - protected
            protected_mask[idx] = True
        elif attention_mask[batch_index, idx].item() == 1:
            # Attended token - valid for adaptation
            valid_mask[idx] = True
    
    return valid_mask, protected_mask
