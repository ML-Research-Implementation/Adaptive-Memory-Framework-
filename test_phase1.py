"""
Test script for Phase 1: Layer-wise Retention Architecture

Tests:
1. AdaptiveDistilBertQA model initialization
2. Forward pass with layer-wise token tracking
3. Metrics collection and reporting
4. Comparison with baseline
"""

import torch
from src import (
    AdaptiveDistilBertQA,
    AdaptiveQAInference,
    LayerWiseMetrics,
    InferenceTimer,
    QADataLoader,
    BaselineQAModel,
)
from config import DEVICE


def test_adaptive_model_initialization():
    """Test 1: Initialize adaptive model."""
    print("\n" + "=" * 70)
    print("TEST 1: Adaptive Model Initialization")
    print("=" * 70)
    
    try:
        model = AdaptiveDistilBertQA(
            retention_ratio=0.75
        )
        print("✓ AdaptiveDistilBertQA initialized successfully")
        print(f"  - Device: {model.device}")
        print(f"  - Num Transformer layers: {model.num_layers}")
        print(f"  - Target retention ratio: {model.retention_ratio:.0%}")
        print(f"  - Retention scorers initialized: {len(model.retention_scorers)}")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize AdaptiveDistilBertQA")
        print(f"  Error: {e}")
        return False


def test_forward_pass_single_example():
    """Test 2: Forward pass on single Q&A example."""
    print("\n" + "=" * 70)
    print("TEST 2: Forward Pass with Layer-wise Retention")
    print("=" * 70)
    
    try:
        # Load data
        loader = QADataLoader()
        question = "What is artificial intelligence?"
        context = "Artificial intelligence (AI) is the simulation of human intelligence by computer systems."
        
        encoded = loader.tokenize_qa(question, context)
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        
        tokens_list = loader.get_tokens(input_ids)
        
        print(f"Question: {question}")
        print(f"Context: {context}")
        print(f"Tokens: {' '.join(tokens_list)}")
        print(f"Original sequence length: {input_ids.shape[1]}")
        
        # Initialize model
        model = AdaptiveDistilBertQA(retention_ratio=0.75)
        model.eval()
        
        # Forward pass
        with InferenceTimer() as timer:
            with torch.no_grad():
                start_logits, end_logits, layer_metrics = model(
                    input_ids=input_ids.to(DEVICE),
                    attention_mask=attention_mask.to(DEVICE),
                    return_layer_metrics=True
                )
        
        print(f"\n✓ Forward pass completed in {timer.get_elapsed_ms():.2f} ms")
        
        # Extract predictions
        start_idx = torch.argmax(start_logits, dim=-1).item()
        end_idx = torch.argmax(end_logits, dim=-1).item()
        
        print(f"Predicted answer span: [{start_idx}, {end_idx}]")
        if start_idx < len(tokens_list) and end_idx < len(tokens_list):
            predicted_answer = ' '.join(tokens_list[start_idx:end_idx+1])
            print(f"Predicted tokens: {predicted_answer}")
        
        # Print layer metrics
        print(f"\nLayer-wise Token Reduction:")
        print(f"{'Layer':<8} {'Tokens In':<12} {'Tokens Out':<12} {'Ratio':<10}")
        print("-" * 42)
        
        for i, tokens_in in enumerate(layer_metrics['tokens_per_layer']):
            if i == 0:
                tokens_out = tokens_in  # No retention at input
            else:
                tokens_out = layer_metrics['tokens_per_layer'][i]
            
            # Reconstruct from selection results
            if layer_metrics['selection_results'][i] is not None:
                result = layer_metrics['selection_results'][i]
                print(f"{i:<8} {result.num_original:<12} {result.num_selected:<12} {result.retention_ratio:.1%}")
            else:
                print(f"{i:<8} {tokens_in:<12} {tokens_in:<12} 100.0%")
        
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics_collection():
    """Test 3: Metrics collection and reporting."""
    print("\n" + "=" * 70)
    print("TEST 3: Metrics Collection and Reporting")
    print("=" * 70)
    
    try:
        metrics = LayerWiseMetrics()
        
        # Simulate metrics for a sequence
        metrics.start_sequence("example_001", retention_ratio=0.75, total_tokens_in=31)
        
        # Record layer metrics (example: 75% retention per layer)
        metrics.record_layer_metrics(layer_idx=0, tokens_in=31, tokens_out=24)  # 77%
        metrics.record_layer_metrics(layer_idx=1, tokens_in=24, tokens_out=18)  # 75%
        metrics.record_layer_metrics(layer_idx=2, tokens_in=18, tokens_out=14)  # 78%
        metrics.record_layer_metrics(layer_idx=3, tokens_in=14, tokens_out=11)  # 79%
        metrics.record_layer_metrics(layer_idx=4, tokens_in=11, tokens_out=8)   # 73%
        metrics.record_layer_metrics(layer_idx=5, tokens_in=8, tokens_out=6)    # 75%
        
        # Record QA prediction
        metrics.record_qa_prediction(
            predicted_start=1,
            predicted_end=3,
            ground_truth_start=1,
            ground_truth_end=3
        )
        
        metrics.record_inference_time(45.3)
        metrics.end_sequence()
        
        print("✓ Metrics recorded successfully")
        
        # Print summary
        metrics.print_summary(retention_ratio=0.75)
        
        # Print layer breakdown
        metrics.print_layer_breakdown(sequence_idx=0)
        
        return True
        
    except Exception as e:
        print(f"✗ Metrics collection failed")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comparison_with_baseline():
    """Test 4: Comparison with baseline model."""
    print("\n" + "=" * 70)
    print("TEST 4: Comparison with Baseline")
    print("=" * 70)
    
    try:
        # Load data
        loader = QADataLoader()
        question = "What is AI?"
        context = "AI is intelligence demonstrated by machines."
        
        encoded = loader.tokenize_qa(question, context)
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        
        print(f"Question: {question}")
        print(f"Context: {context}")
        print(f"Sequence length: {input_ids.shape[1]}")
        
        # Baseline prediction
        baseline = BaselineQAModel()
        baseline.print_info()
        
        with InferenceTimer() as timer:
            start_idx_base, end_idx_base, _, info_base = baseline.get_baseline_prediction(
                input_ids=input_ids.to(DEVICE),
                attention_mask=attention_mask.to(DEVICE)
            )
        baseline_time = timer.get_elapsed_ms()
        
        print(f"\n✓ Baseline prediction: [{start_idx_base}, {end_idx_base}]")
        print(f"  Inference time: {baseline_time:.2f} ms")
        
        # Adaptive prediction (100% retention - should match baseline)
        inference = AdaptiveQAInference(retention_ratio=1.0)
        
        with InferenceTimer() as timer:
            start_idx_adapt, end_idx_adapt, layer_metrics = inference.forward(
                input_ids=input_ids.to(DEVICE),
                attention_mask=attention_mask.to(DEVICE)
            )
        adaptive_time = timer.get_elapsed_ms()
        
        print(f"\n✓ Adaptive prediction (100% retention): [{start_idx_adapt}, {end_idx_adapt}]")
        print(f"  Inference time: {adaptive_time:.2f} ms")
        
        # Verify at 100% retention, predictions should be identical
        match = (start_idx_base == start_idx_adapt) and (end_idx_base == end_idx_adapt)
        print(f"\n✓ Predictions match (100% retention): {match}")
        print(f"  Baseline time: {baseline_time:.2f} ms")
        print(f"  Adaptive time: {adaptive_time:.2f} ms")
        
        return match
        
    except Exception as e:
        print(f"✗ Baseline comparison failed")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Phase 1 tests."""
    print("\n" * 2)
    print("+" + "=" * 68 + "+")
    print("|" + " " * 15 + "PHASE 1: LAYER-WISE RETENTION TESTS" + " " * 19 + "|")
    print("+" + "=" * 68 + "+")
    
    results = []
    
    # Run tests
    results.append(("Model Initialization", test_adaptive_model_initialization()))
    results.append(("Forward Pass", test_forward_pass_single_example()))
    results.append(("Metrics Collection", test_metrics_collection()))
    results.append(("Baseline Comparison", test_comparison_with_baseline()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:<40} {status}")
    
    num_passed = sum(1 for _, passed in results if passed)
    num_total = len(results)
    
    print("-" * 70)
    print(f"Total: {num_passed}/{num_total} tests passed")
    print("=" * 70)
    
    if num_passed == num_total:
        print("\nAll Phase 1 tests passed! Ready for Phase 2.")
    else:
        print(f"\n{num_total - num_passed} test(s) failed. Review errors above.")


if __name__ == "__main__":
    main()
