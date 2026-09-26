import pytest
import os
import sys

# Add root to sys.path to import diagnose_multi_budget_random_full_squad
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class DummyArgs:
    def __init__(self, biases, seeds=None):
        self.biases = biases
        self.seeds = seeds if seeds is not None else [42, 123]

def simulate_validation(all_results, args_biases, args_seeds=None):
    args = DummyArgs(args_biases, args_seeds)
    
    logical_records = []
    for row in all_results:
        b = round(float(row["bias"]), 3)
        row_seeds = tuple(sorted(int(k.split("_")[-1]) for k in row.keys() if k.startswith("Rand_EM_seed_")))
        logical_records.append((b, row_seeds))
        
    from collections import Counter
    produced_counts = Counter(logical_records)
    duplicates = [rec for rec, count in produced_counts.items() if count > 1]
    if duplicates:
        raise AssertionError(f"Duplicate logical records found: {duplicates}")

    expected_seeds = tuple(sorted(args.seeds))
    completed_biases_for_current_run = set(rec[0] for rec in logical_records if rec[1] == expected_seeds)
    missing = [b for b in args.biases if round(float(b), 3) not in completed_biases_for_current_run]
    
    if missing:
        raise AssertionError(f"Expected biases {missing} are missing from accumulated results.")
    return True

def test_multiple_legitimate_rows_per_bias_valid():
    # same bias but different seeds
    all_results = [
        {"bias": 0.1, "Rand_EM_seed_42": 1.0},
        {"bias": 0.1, "Rand_EM_seed_123": 2.0}
    ]
    # Current requested run looking for [123]
    assert simulate_validation(all_results, [0.1], [123]) == True

def test_historical_biases_plus_current_requested_valid():
    all_results = [
        {"bias": 0.05, "Rand_EM_seed_42": 1.0, "Rand_EM_seed_123": 1.0},
        {"bias": 0.10, "Rand_EM_seed_42": 1.0, "Rand_EM_seed_123": 1.0}
    ]
    assert simulate_validation(all_results, [0.05], [42, 123]) == True

def test_same_bias_seed_type_duplicated_fail():
    all_results = [
        {"bias": 0.1, "Rand_EM_seed_42": 1.0},
        {"bias": 0.1, "Rand_EM_seed_42": 1.0}
    ]
    with pytest.raises(AssertionError) as exc:
        simulate_validation(all_results, [0.1], [42])
    assert "Duplicate logical records found" in str(exc.value)

def test_missing_seed_for_requested_bias_fail():
    # Existing row has seed 42, but we requested seed 42 AND 123
    all_results = [
        {"bias": 0.1, "Rand_EM_seed_42": 1.0}
    ]
    with pytest.raises(AssertionError) as exc:
        simulate_validation(all_results, [0.1], [42, 123])
    assert "are missing from accumulated results" in str(exc.value)

def test_precision_collision_no_duplicates():
    all_results = [
        {"bias": 0.15, "Rand_EM_seed_42": 1.0},
        {"bias": 0.1, "Rand_EM_seed_42": 1.0},
    ]
    args_biases = [0.15, 0.1]
    assert simulate_validation(all_results, args_biases, [42]) == True

