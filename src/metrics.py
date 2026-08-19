"""
Metrics and evaluation utilities for layer-wise retention.

Tracks:
- Per-layer token counts and retention ratios
- Theoretical attention computation speedup
- CPU/GPU latency
- QA answer accuracy (EM, F1) vs. baseline
- Analysis and reporting
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time


@dataclass
class LayerMetrics:
    """Metrics for a single Transformer layer."""
    layer_idx: int
    tokens_in: int
    tokens_out: int
    retention_ratio: float = field(init=False)
    attention_ops_in: int = field(init=False)
    attention_ops_out: int = field(init=False)
    speedup_factor: float = field(init=False)
    
    def __post_init__(self):
        """Calculate derived metrics."""
        self.retention_ratio = (
            self.tokens_out / self.tokens_in if self.tokens_in > 0 else 1.0
        )
        # Self-attention scales as O(n^2) where n is sequence length
        self.attention_ops_in = self.tokens_in ** 2
        self.attention_ops_out = self.tokens_out ** 2
        # Speedup is inverse of reduction
        self.speedup_factor = (
            self.attention_ops_in / self.attention_ops_out
            if self.attention_ops_out > 0 else 1.0
        )


@dataclass
class SequenceMetrics:
    """Metrics for a complete forward pass through the model."""
    sequence_id: str
    retention_ratio: float  # Target retention ratio
    total_tokens_in: int  # Original sequence length
    total_tokens_out: int  # After all retention layers
    layer_metrics: List[LayerMetrics] = field(default_factory=list)
    
    # QA metrics
    predicted_start: Optional[int] = None
    predicted_end: Optional[int] = None
    ground_truth_start: Optional[int] = None
    ground_truth_end: Optional[int] = None
    exact_match: bool = False
    f1_score: float = 0.0
    
    # Timing
    inference_time_ms: float = 0.0
    
    def get_cumulative_speedup(self) -> float:
        """
        Calculate cumulative theoretical speedup.
        
        Assumes layers process sequentially, so total speedup is
        product of per-layer speedup factors.
        """
        cumulative = 1.0
        for layer in self.layer_metrics:
            cumulative *= layer.speedup_factor
        return cumulative
    
    def get_total_attention_reduction(self) -> float:
        """
        Calculate total reduction in attention operations.
        
        Returns ratio: (total ops with retention) / (total ops baseline)
        """
        ops_baseline = sum(layer.attention_ops_in for layer in self.layer_metrics)
        ops_adaptive = sum(layer.attention_ops_out for layer in self.layer_metrics)
        
        if ops_baseline == 0:
            return 1.0
        
        return ops_adaptive / ops_baseline
    
    def get_tokens_reduction_per_layer(self) -> List[Tuple[int, int, float]]:
        """
        Get token count reduction at each layer.
        
        Returns:
            List of (tokens_in, tokens_out, ratio) tuples
        """
        return [
            (layer.tokens_in, layer.tokens_out, layer.retention_ratio)
            for layer in self.layer_metrics
        ]


class LayerWiseMetrics:
    """
    Collects and manages metrics for layer-wise retention experiments.
    
    Tracks:
    - Per-sequence forward passes
    - Per-layer token reduction
    - QA answer accuracy
    - Latency and computational savings
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        self.sequences: List[SequenceMetrics] = []
        self.current_sequence: Optional[SequenceMetrics] = None
    
    def start_sequence(
        self,
        sequence_id: str,
        retention_ratio: float,
        total_tokens_in: int
    ):
        """
        Mark the start of a new sequence forward pass.
        
        Args:
            sequence_id: Unique identifier for this sequence (e.g., "example_001")
            retention_ratio: Target retention ratio for this pass
            total_tokens_in: Original sequence length
        """
        self.current_sequence = SequenceMetrics(
            sequence_id=sequence_id,
            retention_ratio=retention_ratio,
            total_tokens_in=total_tokens_in,
            total_tokens_out=total_tokens_in  # Will be updated
        )
    
    def record_layer_metrics(
        self,
        layer_idx: int,
        tokens_in: int,
        tokens_out: int
    ):
        """
        Record token counts for a layer.
        
        Args:
            layer_idx: Layer index (0-5 for DistilBERT)
            tokens_in: Number of tokens entering this layer
            tokens_out: Number of tokens after retention
        """
        if self.current_sequence is None:
            raise RuntimeError("No active sequence. Call start_sequence() first.")
        
        layer_metric = LayerMetrics(
            layer_idx=layer_idx,
            tokens_in=tokens_in,
            tokens_out=tokens_out
        )
        self.current_sequence.layer_metrics.append(layer_metric)
        self.current_sequence.total_tokens_out = tokens_out
    
    def record_qa_prediction(
        self,
        predicted_start: int,
        predicted_end: int,
        ground_truth_start: Optional[int] = None,
        ground_truth_end: Optional[int] = None
    ):
        """
        Record QA prediction and compute accuracy metrics.
        
        Args:
            predicted_start: Predicted start position
            predicted_end: Predicted end position
            ground_truth_start: Ground truth start position (optional)
            ground_truth_end: Ground truth end position (optional)
        """
        if self.current_sequence is None:
            raise RuntimeError("No active sequence. Call start_sequence() first.")
        
        self.current_sequence.predicted_start = predicted_start
        self.current_sequence.predicted_end = predicted_end
        self.current_sequence.ground_truth_start = ground_truth_start
        self.current_sequence.ground_truth_end = ground_truth_end
        
        # Compute metrics if ground truth available
        if ground_truth_start is not None and ground_truth_end is not None:
            self._compute_qa_metrics()
    
    def record_inference_time(self, time_ms: float):
        """
        Record total inference time for this sequence.
        
        Args:
            time_ms: Inference time in milliseconds
        """
        if self.current_sequence is None:
            raise RuntimeError("No active sequence. Call start_sequence() first.")
        
        self.current_sequence.inference_time_ms = time_ms
    
    def end_sequence(self):
        """Mark the end of sequence recording and save it."""
        if self.current_sequence is None:
            raise RuntimeError("No active sequence to end.")
        
        self.sequences.append(self.current_sequence)
        self.current_sequence = None
    
    def _compute_qa_metrics(self):
        """Compute EM and F1 scores for QA prediction."""
        if self.current_sequence is None:
            return
        
        seq = self.current_sequence
        
        if (seq.ground_truth_start is None or
            seq.ground_truth_end is None or
            seq.predicted_start is None or
            seq.predicted_end is None):
            return
        
        # Exact Match
        seq.exact_match = (
            seq.predicted_start == seq.ground_truth_start and
            seq.predicted_end == seq.ground_truth_end
        )
        
        # F1: token overlap
        pred_tokens = set(range(seq.predicted_start, seq.predicted_end + 1))
        gt_tokens = set(range(seq.ground_truth_start, seq.ground_truth_end + 1))
        
        if len(pred_tokens) == 0 and len(gt_tokens) == 0:
            seq.f1_score = 1.0
            return
        
        intersection = len(pred_tokens & gt_tokens)
        union = len(pred_tokens | gt_tokens)
        
        if union == 0:
            seq.f1_score = 0.0
        else:
            precision = intersection / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
            recall = intersection / len(gt_tokens) if len(gt_tokens) > 0 else 0.0
            
            if precision + recall == 0:
                seq.f1_score = 0.0
            else:
                seq.f1_score = 2 * (precision * recall) / (precision + recall)
    
    def get_summary(self) -> Dict:
        """
        Get aggregated summary statistics.
        
        Returns:
            Dictionary with summary metrics across all sequences
        """
        if not self.sequences:
            return {}
        
        sequences = self.sequences
        
        # Token reduction stats
        avg_retention_ratio = np.mean([s.retention_ratio for s in sequences])
        final_token_counts = [s.total_tokens_out for s in sequences]
        avg_final_tokens = np.mean(final_token_counts)
        
        # Speedup stats
        cumulative_speedups = [s.get_cumulative_speedup() for s in sequences]
        avg_cumulative_speedup = np.mean(cumulative_speedups)
        
        attention_reductions = [s.get_total_attention_reduction() for s in sequences]
        avg_attention_reduction = np.mean(attention_reductions)
        
        # QA accuracy stats
        ems = [s.exact_match for s in sequences]
        f1s = [s.f1_score for s in sequences]
        
        avg_em = np.mean(ems) if ems else 0.0
        avg_f1 = np.mean(f1s) if f1s else 0.0
        
        # Latency
        inference_times = [s.inference_time_ms for s in sequences]
        avg_latency = np.mean(inference_times) if inference_times else 0.0
        
        return {
            'num_sequences': len(sequences),
            'avg_retention_ratio': avg_retention_ratio,
            'avg_final_tokens': avg_final_tokens,
            'avg_cumulative_speedup': avg_cumulative_speedup,
            'avg_attention_reduction': avg_attention_reduction,
            'avg_exact_match': avg_em,
            'avg_f1': avg_f1,
            'avg_inference_time_ms': avg_latency,
        }
    
    def print_summary(self, retention_ratio: Optional[float] = None):
        """
        Print formatted summary statistics.
        
        Args:
            retention_ratio: Optional retention ratio to include in title
        """
        summary = self.get_summary()
        
        if not summary:
            print("No sequences recorded.")
            return
        
        title = "Layer-Wise Retention Metrics"
        if retention_ratio is not None:
            title += f" (Retention Ratio: {retention_ratio:.0%})"
        
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)
        
        print(f"Sequences:                  {summary['num_sequences']}")
        print(f"Avg Token Retention Ratio:  {summary['avg_retention_ratio']:.1%}")
        print(f"Avg Final Tokens:           {summary['avg_final_tokens']:.1f}")
        print(f"Avg Cumulative Speedup:     {summary['avg_cumulative_speedup']:.2f}x")
        print(f"Avg Attention Reduction:    {summary['avg_attention_reduction']:.1%}")
        print(f"Avg Exact Match:            {summary['avg_exact_match']:.1%}")
        print(f"Avg F1 Score:               {summary['avg_f1']:.3f}")
        print(f"Avg Inference Latency:      {summary['avg_inference_time_ms']:.2f} ms")
        print("=" * 70)
    
    def print_layer_breakdown(self, sequence_idx: int = 0):
        """
        Print per-layer token reduction for a specific sequence.
        
        Args:
            sequence_idx: Index of sequence to analyze (default: first)
        """
        if sequence_idx >= len(self.sequences):
            print(f"Sequence index {sequence_idx} not found.")
            return
        
        seq = self.sequences[sequence_idx]
        
        print("\n" + "=" * 80)
        print(f"Layer-wise Token Reduction: {seq.sequence_id}")
        print("=" * 80)
        print(f"{'Layer':<8} {'Tokens In':<12} {'Tokens Out':<12} {'Ratio':<10} {'Speedup':<10}")
        print("-" * 80)
        
        for layer in seq.layer_metrics:
            ratio_str = f"{layer.retention_ratio:.1%}"
            speedup_str = f"{layer.speedup_factor:.2f}x"
            print(
                f"{layer.layer_idx:<8} "
                f"{layer.tokens_in:<12} "
                f"{layer.tokens_out:<12} "
                f"{ratio_str:<10} "
                f"{speedup_str:<10}"
            )
        
        print("-" * 80)
        total_speedup = seq.get_cumulative_speedup()
        total_reduction = seq.get_total_attention_reduction()
        print(f"{'Total':<8} {seq.total_tokens_in:<12} {seq.total_tokens_out:<12} "
              f"{total_reduction:.1%} reduction")
        print(f"Cumulative Speedup: {total_speedup:.2f}x")
        print("=" * 80)
    
    def print_comparison_table(self, baseline_metrics: Optional['LayerWiseMetrics'] = None):
        """
        Print side-by-side comparison of adaptive vs. baseline metrics.
        
        Args:
            baseline_metrics: Baseline LayerWiseMetrics object for comparison
        """
        adaptive_summary = self.get_summary()
        baseline_summary = baseline_metrics.get_summary() if baseline_metrics else {}
        
        print("\n" + "=" * 100)
        print("Baseline vs. Adaptive Comparison")
        print("=" * 100)
        
        metrics_to_show = [
            ('num_sequences', 'Num Sequences', '{:.0f}'),
            ('avg_retention_ratio', 'Avg Token Retention', '{:.1%}'),
            ('avg_final_tokens', 'Avg Final Tokens', '{:.1f}'),
            ('avg_cumulative_speedup', 'Cumulative Speedup', '{:.2f}x'),
            ('avg_attention_reduction', 'Attention Reduction', '{:.1%}'),
            ('avg_exact_match', 'Exact Match Rate', '{:.1%}'),
            ('avg_f1', 'Avg F1 Score', '{:.3f}'),
            ('avg_inference_time_ms', 'Inference Time (ms)', '{:.2f}'),
        ]
        
        print(f"{'Metric':<30} {'Adaptive':<20} {'Baseline':<20} {'Δ':<15}")
        print("-" * 100)
        
        for key, label, fmt in metrics_to_show:
            adaptive_val = adaptive_summary.get(key, 0.0)
            baseline_val = baseline_summary.get(key, 0.0)
            
            # Format values
            if 'ratio' in fmt or '%' in fmt:
                adaptive_str = fmt.format(adaptive_val)
                baseline_str = fmt.format(baseline_val)
            else:
                adaptive_str = fmt.format(adaptive_val)
                baseline_str = fmt.format(baseline_val)
            
            # Compute delta
            if baseline_val != 0 and isinstance(adaptive_val, (int, float)):
                delta_pct = ((adaptive_val - baseline_val) / baseline_val) * 100
                delta_str = f"{delta_pct:+.1f}%"
            else:
                delta_str = "-"
            
            print(f"{label:<30} {adaptive_str:<20} {baseline_str:<20} {delta_str:<15}")
        
        print("=" * 100)


class InferenceTimer:
    """Context manager for timing inference."""
    
    def __init__(self):
        """Initialize timer."""
        self.start_time = None
        self.elapsed_ms = 0.0
    
    def __enter__(self):
        """Start timer."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timer and record elapsed time."""
        if self.start_time is not None:
            self.elapsed_ms = (time.time() - self.start_time) * 1000
    
    def get_elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.elapsed_ms
