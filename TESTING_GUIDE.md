# Testing Guide for AMMR Framework

Complete instructions for testing and validating the AMMR implementation.

## Prerequisites

```bash
pip install torch transformers
```

## Test Levels

### Level 1: Quick Validation (2 minutes)

Verify that all imports work and basic structures load:

```python
# test_imports.py
from config import DEVICE, MODEL_NAME
from src import (
    QADataLoader,
    BaselineQAModel,
    RetentionScorer,
    train_retention_scorer,
    RetentionAnalyzer,
)

print("✓ All imports successful")
print(f"✓ Device: {DEVICE}")
print(f"✓ Model: {MODEL_NAME}")
```

**Run:** `python test_imports.py`

---

### Level 2: Component Testing (10 minutes each)

#### Test 2.1: Data Module
```python
# test_data.py
from src import QADataLoader, find_answer_span, create_token_masks
from config import DEVICE

loader = QADataLoader()

# Test tokenization
question = "What is machine learning?"
context = "Machine learning is a subset of AI."

# Basic tokenization
encoded = loader.tokenize_qa(question, context)
assert "input_ids" in encoded
assert "attention_mask" in encoded
print("✓ Basic tokenization works")

# Tokenization with offsets
encoded_full = loader.tokenize_qa_with_offsets(question, context, DEVICE)
assert "offset_mapping" in encoded_full
print("✓ Offset mapping works")

# Token extraction
tokens = loader.get_tokens(encoded["input_ids"])
assert len(tokens) > 0
print(f"✓ Tokenization: {len(tokens)} tokens extracted")

# Token masks
valid_mask, protected_mask = create_token_masks(
    len(tokens),
    encoded_full.sequence_ids(),
    encoded["attention_mask"],
    device=DEVICE
)
assert valid_mask.sum() > 0
assert protected_mask.sum() > 0
print(f"✓ Masks: {protected_mask.sum().item()} protected, {valid_mask.sum().item()} adaptive")
```

**Run:** `python test_data.py`

#### Test 2.2: Baseline Model
```python
# test_baseline.py
from src import BaselineQAModel
from config import DEVICE
import torch

baseline = BaselineQAModel(freeze_parameters=True)

# Check model info
assert baseline.num_parameters > 0
assert baseline.hidden_dim == 768
assert baseline.num_layers == 6
print(f"✓ Model loaded: {baseline.num_parameters:,} parameters")

# Test hidden state extraction
input_ids = torch.randint(0, 30522, (1, 31))
attention_mask = torch.ones(1, 31)

hidden_states, all_layers = baseline.get_hidden_states(
    input_ids.to(DEVICE),
    attention_mask.to(DEVICE),
    return_all_layers=True
)

assert hidden_states.shape == (1, 31, 768)
assert len(all_layers) == 7  # 6 layers + 1 input embedding
print(f"✓ Hidden states: {hidden_states.shape}")
print(f"✓ All layers extracted: {len(all_layers)} total")
```

**Run:** `python test_baseline.py`

#### Test 2.3: Models Module
```python
# test_models.py
from src import RetentionScorer, AdaptiveMemoryRetention
from config import DEVICE
import torch

# Test RetentionScorer
scorer = RetentionScorer(hidden_dimension=768)
hidden_states = torch.randn(1, 31, 768)

scores, probs = scorer(hidden_states)
assert scores.shape == (1, 31)
assert probs.shape == (1, 31)
assert (probs >= 0).all() and (probs <= 1).all()
print(f"✓ RetentionScorer: scores shape {scores.shape}, probs range [{probs.min():.4f}, {probs.max():.4f}]")

# Test AdaptiveMemoryRetention
retention_module = AdaptiveMemoryRetention(hidden_dimension=768)
protected_mask = torch.zeros(31, dtype=torch.bool)
protected_mask[0] = True  # Protect [CLS]

gated, retention_probs, retention_scores = retention_module(
    hidden_states.to(DEVICE),
    protected_mask.to(DEVICE),
)

assert gated.shape == hidden_states.shape
assert retention_probs[0, 0].item() == 1.0  # Protected token always 1.0
print(f"✓ AdaptiveMemoryRetention: protected token prob = {retention_probs[0, 0].item():.4f}")
```

**Run:** `python test_models.py`

#### Test 2.4: Losses Module
```python
# test_losses.py
from src import (
    calculate_qa_loss,
    calculate_budget_loss,
    calculate_entropy_loss,
    calculate_combined_loss,
)
from src import BaselineQAModel
from config import DEVICE
import torch

baseline = BaselineQAModel(freeze_parameters=True)

# Create dummy data
gated_hidden = torch.randn(1, 31, 768).to(DEVICE)
start_target = torch.tensor([5], device=DEVICE)
end_target = torch.tensor([8], device=DEVICE)
probs = torch.sigmoid(torch.randn(1, 31)).to(DEVICE)
valid_mask = torch.ones(31, dtype=torch.bool, device=DEVICE)
target_budget = 15

# Test individual losses
qa_loss, _, _ = calculate_qa_loss(
    baseline.qa_model, gated_hidden, start_target, end_target
)
assert qa_loss.item() > 0
print(f"✓ QA Loss: {qa_loss.item():.6f}")

budget_loss, expected = calculate_budget_loss(probs, valid_mask, target_budget)
assert budget_loss.item() >= 0
print(f"✓ Budget Loss: {budget_loss.item():.6f} (expected tokens: {expected.item():.1f})")

entropy_loss = calculate_entropy_loss(probs, valid_mask)
assert entropy_loss.item() >= 0
print(f"✓ Entropy Loss: {entropy_loss.item():.6f}")

# Test combined loss
total_loss, loss_dict = calculate_combined_loss(
    qa_loss, budget_loss, entropy_loss,
    budget_weight=0.1, entropy_weight=0.001
)
assert total_loss.item() > 0
assert "total" in loss_dict and "qa" in loss_dict
print(f"✓ Combined Loss: {total_loss.item():.6f}")
```

**Run:** `python test_losses.py`

#### Test 2.5: Training Module
```python
# test_training.py
from src import RetentionScorer, train_retention_scorer, BaselineQAModel
from config import DEVICE
import torch

baseline = BaselineQAModel(freeze_parameters=True)
scorer = RetentionScorer(hidden_dimension=768).to(DEVICE)

# Create dummy data
hidden_states = torch.randn(1, 31, 768).to(DEVICE)
protected_mask = torch.zeros(31, dtype=torch.bool)
protected_mask[0] = True
valid_mask = torch.ones(31, dtype=torch.bool)
start_target = torch.tensor([5], device=DEVICE)
end_target = torch.tensor([8], device=DEVICE)

# Quick training
trainer, result = train_retention_scorer(
    scorer=scorer,
    qa_model=baseline.qa_model,
    hidden_states=hidden_states,
    protected_mask=protected_mask,
    valid_mask=valid_mask,
    start_target=start_target,
    end_target=end_target,
    target_budget=15,
    num_steps=10,  # Quick test
    verbose=False
)

assert result['total'] > 0
assert trainer.current_step == 10
print(f"✓ Training completed: {trainer.current_step} steps")
print(f"✓ Final loss: {result['total']:.6f}")
```

**Run:** `python test_training.py`

#### Test 2.6: Evaluation Module
```python
# test_evaluation.py
from src import RetentionAnalyzer
import torch

tokens = ["[CLS]", "what", "is", "AI", "[SEP]", "AI", "is", "great", "[PAD]"]
probs = torch.tensor([1.0, 0.9, 0.1, 0.95, 1.0, 0.85, 0.2, 0.75, 0.0])
protected_mask = torch.tensor([True, False, False, False, True, False, False, False, False])
valid_mask = torch.tensor([False, True, True, True, False, True, True, True, False])

analyzer = RetentionAnalyzer(tokens, probs, protected_mask, valid_mask)

# Test ranking
ranking = analyzer.get_token_ranking()
assert ranking[0][2] >= ranking[1][2]  # Should be sorted by probability
print(f"✓ Token ranking: top token '{ranking[0][1]}' with prob {ranking[0][2]:.4f}")

# Test statistics
expected = analyzer.get_expected_retained_tokens()
ratio = analyzer.get_retention_ratio()
stats = analyzer.get_token_statistics()
print(f"✓ Analysis: expected={expected:.1f}, ratio={ratio:.2%}, mean_prob={stats['mean']:.4f}")
```

**Run:** `python test_evaluation.py`

---

### Level 3: Integration Testing (5 minutes)

#### Test 3.1: Example Scripts
```bash
python examples.py
```

Should print:
- ✓ Setup example
- ✓ Tokenization example
- ✓ Hidden states example

#### Test 3.2: Full Pipeline
```bash
python main.py
```

Should complete all 21 steps and print detailed analysis.

---

### Level 4: Full Validation (5-10 minutes)

Run all tests in sequence:

```bash
python test_imports.py && \
python test_data.py && \
python test_baseline.py && \
python test_models.py && \
python test_losses.py && \
python test_training.py && \
python test_evaluation.py && \
python examples.py && \
python main.py
```

**Expected Result:** All tests pass, no errors.

---

## Debugging Guide

### Problem: "Device not found"
```python
# Solution
from config import DEVICE
import torch
print(torch.cuda.is_available())  # Should be True for GPU
print(DEVICE)  # Should show cuda:0 or cpu
```

### Problem: "Model not found"
```python
# Solution: First download model
from transformers import DistilBertForQuestionAnswering
model = DistilBertForQuestionAnswering.from_pretrained(
    "distilbert-base-uncased-distilled-squad"
)
```

### Problem: "Loss is NaN"
- Check if probabilities are valid: `assert (probs >= 0).all() and (probs <= 1).all()`
- Check if targets are within sequence length: `assert start_target < seq_len`
- Check if hidden states are not NaN: `assert not torch.isnan(hidden_states).any()`

### Problem: "Out of Memory"
```python
# Reduce batch size or sequence length
from config import MAX_SEQUENCE_LENGTH
# Edit config.py
MAX_SEQUENCE_LENGTH = 256  # Reduce from 512
```

---

## Validation Checklist

Use this checklist to verify complete implementation:

- [ ] All imports work (`test_imports.py`)
- [ ] Data module processes tokens correctly (`test_data.py`)
- [ ] Baseline model loads and extracts hidden states (`test_baseline.py`)
- [ ] RetentionScorer produces valid probabilities (`test_models.py`)
- [ ] All losses compute without errors (`test_losses.py`)
- [ ] Training loop runs and reduces loss (`test_training.py`)
- [ ] Evaluation produces ranking and statistics (`test_evaluation.py`)
- [ ] Examples run without errors (`examples.py`)
- [ ] Full pipeline completes successfully (`main.py`)

---

## Performance Expectations

### Baseline Model
- **Parameters:** ~66.4M (DistilBERT-base)
- **Hidden Dimension:** 768
- **Layers:** 6 Transformer layers
- **Load Time:** ~2-5 seconds
- **Inference Time (31 tokens):** ~0.1-0.2 seconds

### Retention Scorer
- **Parameters:** ~10K-50K (depending on hidden dimension)
- **Training Time (500 steps):** ~30-60 seconds
- **Memory:** ~200MB (with baseline model)

### Total Pipeline
- **Full run:** ~2-5 minutes on CPU, ~30-60 seconds on GPU
- **Memory:** ~500MB-1GB

---

## Next Steps

After validation:
1. Modify `config.py` to test different hyperparameters
2. Implement layer-wise token pruning
3. Train on full SQuAD dataset
4. Measure actual computational savings
