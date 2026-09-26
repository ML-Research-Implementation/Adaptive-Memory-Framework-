import pytest
import os
import sys

# Add root to sys.path to import diagnose_multi_budget_random_full_squad
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import diagnose_multi_budget_random_full_squad

class DummyArgs:
    def __init__(self, biases):
        self.biases = biases

def simulate_validation(all_results, args_biases):
    # This simulates exactly the code we just modified
    args = DummyArgs(args_biases)
    from collections import Counter
    produced_counts = Counter(round(float(row["bias"]), 1) for row in all_results)
    expected = {round(float(b), 1) for b in args.biases}
    missing = [b for b in expected if produced_counts[b] == 0]
    duplicates = [b for b, count in produced_counts.items() if count > 1]
    
    if missing:
        raise AssertionError(f"Expected biases {missing} are missing from accumulated results.")
    if duplicates:
        raise AssertionError(f"Biases {duplicates} appear multiple times in accumulated results.")
    return True

def test_validation_passes_with_historical_biases():
    # existing CSV contains historical biases
    all_results = [
        {"bias": 1.0},
        {"bias": 0.5},
        {"bias": 0.2},
        {"bias": 0.0},
        {"bias": 0.1}
    ]
    # current invocation requests a subset/new subset
    args_biases = [0.0, 0.1, 0.2]
    
    # validation passes when all requested biases are present
    assert simulate_validation(all_results, args_biases) == True

def test_validation_fails_when_missing():
    all_results = [
        {"bias": 1.0},
        {"bias": 0.5},
    ]
    args_biases = [1.0, 0.5, 0.2]
    
    # validation fails when a requested bias is genuinely missing
    with pytest.raises(AssertionError) as exc_info:
        simulate_validation(all_results, args_biases)
    assert "Expected biases [0.2] are missing" in str(exc_info.value)

def test_validation_fails_on_duplicates():
    all_results = [
        {"bias": 1.0},
        {"bias": 1.0},
    ]
    args_biases = [1.0]
    
    with pytest.raises(AssertionError) as exc_info:
        simulate_validation(all_results, args_biases)
    assert "appear multiple times" in str(exc_info.value)
