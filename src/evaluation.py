"""
Evaluation and analysis functions for retention mechanism.
Includes token importance ranking, retention analysis, and metric computation.
"""

from typing import Dict, List, Tuple, Optional
import torch
from src.utils import format_number


class RetentionAnalyzer:
    """
    Analyze retention probabilities and their implications.
    
    This class provides utilities for:
    - Ranking tokens by retention probability
    - Computing memory statistics
    - Analyzing which tokens are retained
    - Computing metrics comparing predictions
    """
    
    def __init__(
        self,
        tokens: List[str],
        probabilities: torch.Tensor,
        protected_mask: torch.Tensor,
        valid_mask: torch.Tensor
    ):
        """
        Initialize retention analyzer.
        
        Args:
            tokens: List of token strings (seq_len,).
            probabilities: Retention probabilities (batch, seq_len) or (seq_len,).
            protected_mask: Protected token mask (seq_len,).
            valid_mask: Valid (adaptive) token mask (seq_len,).
        """
        self.tokens = tokens
        
        # Handle batch dimension
        if probabilities.dim() == 2:
            self.probabilities = probabilities[0]  # Take first batch
        else:
            self.probabilities = probabilities
        
        self.protected_mask = protected_mask
        self.valid_mask = valid_mask
        self.sequence_length = len(tokens)
    
    def get_token_ranking(self, reverse: bool = True) -> List[Tuple[int, str, float, str]]:
        """
        Rank tokens by retention probability.
        
        Args:
            reverse: If True, sort descending (highest probability first).
            
        Returns:
            List of tuples (index, token, probability, type).
            Type is 'PROTECTED', 'ADAPTIVE', or 'IGNORED'.
        """
        ranking = []
        
        for idx in range(self.sequence_length):
            token = self.tokens[idx]
            prob = self.probabilities[idx].item()
            
            # Determine token type
            if self.protected_mask[idx]:
                token_type = "PROTECTED"
            elif self.valid_mask[idx]:
                token_type = "ADAPTIVE"
            else:
                token_type = "IGNORED"
            
            ranking.append((idx, token, prob, token_type))
        
        # Sort by probability
        ranking.sort(key=lambda x: x[2], reverse=reverse)
        
        return ranking
    
    def print_ranking(self, top_k: Optional[int] = None, include_adaptive_only: bool = False):
        """
        Print token ranking.
        
        Args:
            top_k: Only print top K tokens (if None, print all).
            include_adaptive_only: Only show adaptive tokens.
        """
        ranking = self.get_token_ranking(reverse=True)
        
        if include_adaptive_only:
            ranking = [r for r in ranking if r[3] == "ADAPTIVE"]
        
        if top_k is not None:
            ranking = ranking[:top_k]
        
        print("\nToken Ranking (by Retention Probability):")
        print("-" * 80)
        print(f"{'Rank':<6} {'Pos':<4} {'Token':<20} {'Prob':<8} {'Type':<12}")
        print("-" * 80)
        
        for rank, (idx, token, prob, token_type) in enumerate(ranking, 1):
            print(f"{rank:<6} {idx:<4} {token:<20} {prob:<8.4f} {token_type:<12}")
    
    def get_expected_retained_tokens(self) -> float:
        """
        Compute expected number of retained tokens.
        
        Returns:
            Sum of probabilities for adaptive tokens.
        """
        adaptive_probs = self.probabilities * self.valid_mask.to(self.probabilities.device)
        return adaptive_probs.sum().item()
    
    def get_retention_ratio(self) -> float:
        """
        Compute retention ratio (expected retained / total adaptive).
        
        Returns:
            Retention ratio in [0, 1].
        """
        total_adaptive = self.valid_mask.sum().item()
        if total_adaptive == 0:
            return 0.0
        return self.get_expected_retained_tokens() / total_adaptive
    
    def get_token_statistics(self) -> Dict[str, float]:
        """
        Compute statistics about retention probabilities.
        
        Returns:
            Dictionary with min, max, mean, std for adaptive tokens.
        """
        adaptive_probs = self.probabilities * self.valid_mask.to(self.probabilities.device)
        adaptive_probs = adaptive_probs[self.valid_mask > 0]
        
        if len(adaptive_probs) == 0:
            return {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'std': 0.0}
        
        return {
            'min': adaptive_probs.min().item(),
            'max': adaptive_probs.max().item(),
            'mean': adaptive_probs.mean().item(),
            'std': adaptive_probs.std().item(),
        }
    
    def print_summary(self):
        """Print summary statistics."""
        expected_retained = self.get_expected_retained_tokens()
        retention_ratio = self.get_retention_ratio()
        total_adaptive = self.valid_mask.sum().item()
        stats = self.get_token_statistics()
        
        print("\n" + "=" * 80)
        print("RETENTION ANALYSIS SUMMARY")
        print("=" * 80)
        print(f"Total Sequence Length: {self.sequence_length}")
        print(f"Protected Tokens: {self.protected_mask.sum().item()}")
        print(f"Adaptive Tokens: {total_adaptive}")
        print(f"Ignored Tokens: {(~self.protected_mask & ~self.valid_mask).sum().item()}")
        print()
        print(f"Expected Retained Tokens: {format_number(expected_retained, 3)}")
        print(f"Retention Ratio: {format_number(retention_ratio * 100, 2)}%")
        print()
        print(f"Probability Statistics (Adaptive Tokens):")
        print(f"  Min:  {format_number(stats['min'], 4)}")
        print(f"  Max:  {format_number(stats['max'], 4)}")
        print(f"  Mean: {format_number(stats['mean'], 4)}")
        print(f"  Std:  {format_number(stats['std'], 4)}")


def get_top_k_tokens(
    probabilities: torch.Tensor,
    valid_mask: torch.Tensor,
    k: int,
    tokens: Optional[List[str]] = None
) -> Dict:
    """
    Get top-K retained tokens by probability.
    
    Args:
        probabilities: Retention probabilities (batch, seq_len) or (seq_len,).
        valid_mask: Valid token mask (seq_len,).
        k: Number of top tokens to return.
        tokens: Optional token strings for display.
        
    Returns:
        Dictionary with indices, probabilities, and tokens if provided.
    """
    # Handle batch dimension
    if probabilities.dim() == 2:
        probs = probabilities[0]
    else:
        probs = probabilities
    
    # Get indices of valid tokens
    valid_indices = torch.where(valid_mask)[0]
    
    # Get probabilities for valid tokens
    valid_probs = probs[valid_indices]
    
    # Get top-K
    top_k_count = min(k, len(valid_indices))
    top_k_probs, top_k_local_indices = torch.topk(valid_probs, top_k_count)
    
    # Map back to original indices
    top_k_indices = valid_indices[top_k_local_indices]
    
    result = {
        'indices': top_k_indices.cpu().numpy(),
        'probabilities': top_k_probs.cpu().detach().numpy(),
    }
    
    if tokens is not None:
        result['tokens'] = [tokens[i] for i in top_k_indices.cpu().numpy()]
    
    return result


def compare_predictions(
    baseline_start: int,
    baseline_end: int,
    retained_start: int,
    retained_end: int,
    tokens: List[str],
    baseline_tokens: List[str],
    retained_tokens: List[str]
) -> Dict:
    """
    Compare baseline prediction with retained prediction.
    
    Args:
        baseline_start: Baseline model's start index.
        baseline_end: Baseline model's end index.
        retained_start: Retained model's start index.
        retained_end: Retained model's end index.
        tokens: All tokens.
        baseline_tokens: Tokens for baseline span.
        retained_tokens: Tokens for retained span.
        
    Returns:
        Comparison dictionary.
    """
    exact_match = (baseline_start == retained_start and baseline_end == retained_end)
    
    # Token overlap
    baseline_set = set(range(baseline_start, baseline_end + 1))
    retained_set = set(range(retained_start, retained_end + 1))
    
    intersection = len(baseline_set & retained_set)
    union = len(baseline_set | retained_set)
    
    iou = intersection / union if union > 0 else 0.0
    
    if len(baseline_set) > 0:
        recall = intersection / len(baseline_set)
    else:
        recall = 0.0
    
    return {
        'exact_match': exact_match,
        'iou': iou,
        'recall': recall,
        'baseline_answer': ' '.join(baseline_tokens),
        'retained_answer': ' '.join(retained_tokens),
        'baseline_span': f"[{baseline_start}, {baseline_end}]",
        'retained_span': f"[{retained_start}, {retained_end}]",
    }


def print_evaluation_report(
    analyzer: RetentionAnalyzer,
    baseline_metrics: Dict,
    retained_metrics: Optional[Dict] = None,
    comparison: Optional[Dict] = None
):
    """
    Print comprehensive evaluation report.
    
    Args:
        analyzer: RetentionAnalyzer instance.
        baseline_metrics: Baseline metrics dictionary.
        retained_metrics: Retained model metrics (optional).
        comparison: Comparison dictionary (optional).
    """
    print("\n" + "=" * 80)
    print("EVALUATION REPORT")
    print("=" * 80)
    
    # Retention analysis
    analyzer.print_summary()
    
    # Baseline metrics
    print("\n" + "-" * 80)
    print("BASELINE METRICS")
    print("-" * 80)
    print(f"Exact Match: {baseline_metrics.get('exact_match', 0):.4f}")
    print(f"F1 Score: {baseline_metrics.get('f1', 0):.4f}")
    print(f"Precision: {baseline_metrics.get('precision', 0):.4f}")
    print(f"Recall: {baseline_metrics.get('recall', 0):.4f}")
    
    # Retained metrics
    if retained_metrics:
        print("\n" + "-" * 80)
        print("RETAINED MODEL METRICS")
        print("-" * 80)
        print(f"Exact Match: {retained_metrics.get('exact_match', 0):.4f}")
        print(f"F1 Score: {retained_metrics.get('f1', 0):.4f}")
        print(f"Precision: {retained_metrics.get('precision', 0):.4f}")
        print(f"Recall: {retained_metrics.get('recall', 0):.4f}")
    
    # Comparison
    if comparison:
        print("\n" + "-" * 80)
        print("PREDICTION COMPARISON")
        print("-" * 80)
        print(f"Exact Match: {comparison['exact_match']}")
        print(f"IoU: {comparison['iou']:.4f}")
        print(f"Recall: {comparison['recall']:.4f}")
        print(f"Baseline Answer: {comparison['baseline_answer']}")
        print(f"Retained Answer: {comparison['retained_answer']}")
