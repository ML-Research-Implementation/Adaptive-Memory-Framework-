# AMMR Project Refactoring Guide

## Overview

The AMMR framework has been refactored from monolithic scripts into a clean, modular architecture. This guide explains the changes and how to use the new structure.

## What Changed

### Before (Monolithic)
```
src/
├── baseline_distilbert.py        (400+ lines)
├── learned_retention_prototype.py (1000+ lines)
└── hoot_test.txt
distilbert_demo.py                 (200+ lines)
```

**Problems:**
- All logic crammed into one file
- Hard to reuse components
- Difficult to test individual pieces
- Configuration scattered throughout
- No clear separation of concerns

### After (Modular)
```
config.py                          # Single config source (90 lines)
main.py                            # Clean pipeline (350 lines)
examples.py                        # Usage examples (150 lines)

src/
├── __init__.py                    # Package exports
├── utils.py                       # Utilities (250 lines)
├── data.py                        # Data loading (220 lines)
├── models.py                      # Models (200 lines)
├── losses.py                      # Loss functions (180 lines)
├── baseline.py                    # Baseline QA (200 lines)
├── training.py                    # Training (300 lines)
└── evaluation.py                  # Analysis (280 lines)
```

**Benefits:**
- ✅ Each module has one responsibility
- ✅ Components are reusable
- ✅ Easy to test and debug
- ✅ All config in one place
- ✅ Clear imports and dependencies

## Key Modules Explained

### 1. config.py - Configuration Hub
**Replaces:** Configuration scattered in each script

**Contains:**
- Model name, dimensions
- Training hyperparameters (learning rate, steps, etc.)
- Loss weights (budget_lambda, entropy_lambda)
- Device selection

**Usage:**
```python
from config import DEVICE, LEARNING_RATE, TRAINING_STEPS
```

### 2. src/utils.py - Common Utilities
**Replaces:** Utility functions spread across files

**Provides:**
- `set_seed()` - Reproducibility
- `print_header()` - Formatted logging
- `count_parameters()` - Model analysis
- `freeze_model()` - Parameter freezing
- `save_checkpoint()` / `load_checkpoint()` - Persistence

### 3. src/data.py - Data Handling
**Replaces:** Tokenization code in baseline_distilbert.py

**Provides:**
- `QADataLoader` - Centralized tokenization
- `find_answer_span()` - Answer localization using offsets
- `create_token_masks()` - Distinguish token types
- `get_token_type()` - Human-readable token labels

### 4. src/models.py - Neural Networks
**Replaces:** RetentionScorer class in learned_retention_prototype.py

**Provides:**
- `RetentionScorer` - MLP for token importance
- `SoftRetentionGate` - Probabilistic gating
- `AdaptiveMemoryRetention` - Complete module

### 5. src/losses.py - Objectives
**Replaces:** Loss calculation functions in learned_retention_prototype.py

**Provides:**
- `calculate_qa_loss()` - Task loss
- `calculate_budget_loss()` - Memory constraint
- `calculate_entropy_loss()` - Decision sharpness
- `calculate_combined_loss()` - Multi-objective combination

### 6. src/baseline.py - Baseline Model
**Replaces:** Baseline setup in baseline_distilbert.py

**Provides:**
- `BaselineQAModel` - Unified baseline interface
- Model loading, freezing, inference
- `compute_baseline_metrics()` - Evaluation metrics

### 7. src/training.py - Training Loop
**Replaces:** Training loop in learned_retention_prototype.py

**Provides:**
- `RetentionScorerTrainer` - Training manager
- `train_retention_scorer()` - Simple training interface
- History tracking, logging, checkpointing

### 8. src/evaluation.py - Analysis
**Replaces:** Analysis code in learned_retention_prototype.py

**Provides:**
- `RetentionAnalyzer` - Token importance analysis
- `get_top_k_tokens()` - Top retained tokens
- `compare_predictions()` - Baseline vs. retained
- `print_evaluation_report()` - Comprehensive reporting

## Migration Guide

### Old Way (Monolithic)
```python
# Everything in learned_retention_prototype.py
import torch
from transformers import DistilBertForQuestionAnswering

# Load model
qa_model = DistilBertForQuestionAnswering.from_pretrained(...)

# Freeze parameters
for p in qa_model.parameters():
    p.requires_grad = False

# ... 1000+ lines of training code
```

### New Way (Modular)
```python
from config import DEVICE, LEARNING_RATE
from src import (
    BaselineQAModel,
    RetentionScorer,
    train_retention_scorer,
)

# Load baseline
baseline = BaselineQAModel(freeze_parameters=True)

# Create scorer
scorer = RetentionScorer().to(DEVICE)

# Train
trainer, result = train_retention_scorer(
    scorer, baseline.qa_model, hidden_states,
    protected_mask, valid_mask, start_target, end_target,
    target_budget, num_steps=TRAINING_STEPS
)
```

## Running the Code

### Full Pipeline
```bash
python main.py
```
Demonstrates complete workflow from loading to evaluation.

### Examples Only
```bash
python examples.py
```
Quick examples of individual components.

### Custom Script
```python
from src import *
from config import *

# Use components as needed
data_loader = QADataLoader()
baseline = BaselineQAModel()
scorer = RetentionScorer()
# ... your code
```

## Dependency Graph

```
main.py
  ├── config.py
  ├── src/__init__.py
  │   ├── src/utils.py
  │   │   └── config.py
  │   ├── src/data.py
  │   │   └── config.py
  │   ├── src/models.py
  │   │   ├── config.py
  │   │   └── torch
  │   ├── src/losses.py
  │   │   ├── torch
  │   │   └── src/models.py
  │   ├── src/baseline.py
  │   │   ├── config.py
  │   │   ├── src/utils.py
  │   │   ├── src/data.py
  │   │   └── transformers
  │   ├── src/training.py
  │   │   ├── config.py
  │   │   ├── src/utils.py
  │   │   ├── src/models.py
  │   │   └── src/losses.py
  │   └── src/evaluation.py
  │       └── torch
  └── transformers
```

**Key Principle:** Dependencies flow downward (no circular dependencies)

## Configuration Management

All settings in one place (`config.py`):

```python
# Change these to experiment
LEARNING_RATE = 1e-3          # Tune training speed
TRAINING_STEPS = 500          # Change number of steps
BUDGET_LAMBDA = 0.10          # Adjust memory constraint weight
ENTROPY_LAMBDA = 0.001        # Adjust decision sharpness
TEMPERATURE = 1.0             # Control probability sharpness
RETENTION_RATIO = 0.50        # Target retention percentage
```

No need to edit multiple files!

## Adding New Features

### Example: Add a new loss function

1. **Add configuration** (config.py):
```python
NEW_LOSS_WEIGHT = 0.05
```

2. **Implement loss** (src/losses.py):
```python
def calculate_new_loss(...):
    # implementation
    return loss
```

3. **Update combined loss** (src/losses.py):
```python
def calculate_combined_loss(..., new_loss_weight=0.0):
    total_loss = qa_loss + budget_loss + entropy_loss + new_loss_weight * new_loss
```

4. **Use in training** (src/training.py):
```python
new_loss = calculate_new_loss(...)
total_loss, loss_dict = calculate_combined_loss(..., new_loss_weight=NEW_LOSS_WEIGHT)
```

5. **No main.py changes needed** - it uses train_retention_scorer() which handles everything!

## Testing Individual Components

```python
# Test data loading
from src import QADataLoader
loader = QADataLoader()
encoded = loader.tokenize_qa("What is X?", "X is Y")
assert "input_ids" in encoded

# Test baseline model
from src import BaselineQAModel
baseline = BaselineQAModel()
start, end, _, info = baseline.get_baseline_prediction(...)
assert isinstance(start, int)

# Test scorer
from src import RetentionScorer
scorer = RetentionScorer()
scores, probs = scorer(hidden_states)
assert probs.shape == hidden_states.shape[:-1]

# Test losses
from src import calculate_qa_loss
loss, _, _ = calculate_qa_loss(qa_model, hidden_states, start_target, end_target)
assert loss.item() > 0
```

## Legacy Files

The old files still exist for reference:
- `distilbert_demo.py` - Original embedding demo
- `src/baseline_distilbert.py` - Original baseline setup
- `src/learned_retention_prototype.py` - Original full prototype

These are **read-only reference**. Use the new modular structure instead.

## Benefits Summary

| Aspect | Before | After |
|--------|--------|-------|
| Configuration | Scattered | Single file |
| Reusability | Low | High |
| Testability | Difficult | Easy |
| Extensibility | Hard | Simple |
| Readability | Overwhelming | Clear |
| Maintenance | Tedious | Straightforward |
| Lines per file | 1000+ | 200-350 |
| Dependencies | Circular | Linear |

## Next Steps

1. **Run the pipeline**
   ```bash
   python main.py
   ```

2. **Review the modular components**
   - Start with `main.py` for high-level flow
   - Dive into individual modules as needed

3. **Extend the framework**
   - Add new components to `src/`
   - Update `config.py` for new settings
   - Keep separation of concerns

4. **Experiment**
   - Modify `config.py` to test different hyperparameters
   - No code changes needed, just configuration

## Questions?

- Read docstrings in each module
- Look at `main.py` for complete workflow
- Check `examples.py` for simple usage patterns
- Review comments in `src/` modules
