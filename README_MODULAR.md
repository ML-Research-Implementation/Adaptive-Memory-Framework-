# AMMR — Adaptive Multi-Level Memory Retention for Efficient Transformer Networks

## Project Overview

**AMMR** is a modular research framework for developing and testing adaptive token retention mechanisms in Transformer-based language models. We use **DistilBERT for Question Answering** as our experimental platform to reduce computational costs while preserving task performance.

### Key Innovation

Instead of processing all tokens through all Transformer layers (which requires quadratic attention computation), AMMR learns to identify and retain task-relevant tokens dynamically. The framework combines:

- **Learnable retention scoring**: A small MLP that predicts which tokens are important
- **Soft probabilistic gating**: Differentiable gate allowing end-to-end training
- **Multi-objective learning**: Balances QA accuracy, memory budget, and decision sharpness
- **Modular architecture**: Clean separation of concerns for extensibility

---

## Project Structure

```
Adaptive-Memory-Framework/
├── config.py                    # Centralized configuration
├── main.py                      # Main training pipeline
├── examples.py                  # Simple usage examples
├── readme.md                    # This file
│
├── src/
│   ├── __init__.py              # Package exports
│   ├── utils.py                 # Utilities (logging, device, checkpoints)
│   ├── data.py                  # Tokenization and data loading
│   ├── models.py                # RetentionScorer and retention modules
│   ├── losses.py                # QA, budget, and entropy losses
│   ├── baseline.py              # Baseline DistilBERT model
│   ├── training.py              # Training loop and trainer class
│   └── evaluation.py            # Analysis and metrics
│
├── distilbert_demo.py           # [Legacy] Embedding demonstration
├── src/baseline_distilbert.py   # [Legacy] Baseline setup
└── src/learned_retention_prototype.py  # [Legacy] Full prototype
```

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| **config.py** | Single source of truth for all hyperparameters and settings |
| **src/utils.py** | Device management, reproducibility, logging, checkpointing |
| **src/data.py** | Tokenization, sequence processing, answer span location |
| **src/models.py** | RetentionScorer, SoftRetentionGate, AdaptiveMemoryRetention |
| **src/losses.py** | QA loss, budget loss, entropy loss, combined loss |
| **src/baseline.py** | Baseline DistilBERT loading and inference |
| **src/training.py** | RetentionScorerTrainer, training loops |
| **src/evaluation.py** | RetentionAnalyzer, ranking, metrics, reports |

---

## Quick Start

### 1. Installation

```bash
pip install torch transformers
```

### 2. Running the Full Pipeline

```bash
python main.py
```

This runs the complete AMMR pipeline:
- Loads baseline DistilBERT model
- Prepares QA data with answer span localization
- Creates and trains a retention scorer
- Analyzes learned retention probabilities
- Compares baseline vs. retained predictions

### 3. Running Examples

```bash
python examples.py
```

Demonstrates individual components (setup, tokenization, hidden states, etc.)

---

## Configuration

All hyperparameters are centralized in `config.py`:

```python
# Model
MODEL_NAME = "distilbert-base-uncased-distilled-squad"

# Retention
RETENTION_RATIO = 0.50              # Target: retain 50% of tokens
TEMPERATURE = 1.0                   # Probability sharpness

# Training
LEARNING_RATE = 1e-3
TRAINING_STEPS = 500
BUDGET_LAMBDA = 0.10                # Weight for budget loss
ENTROPY_LAMBDA = 0.001              # Weight for entropy loss
```

Modify these values to experiment with different configurations.

---

## Core Components

### RetentionScorer

Learns to score each token based on its importance for the QA task:

```python
from src import RetentionScorer

scorer = RetentionScorer(hidden_dimension=768)
scores, probabilities = scorer(hidden_states, temperature=1.0)
```

**Architecture**: Linear → LayerNorm → GELU → Dropout → Linear → Sigmoid

### Data Loading

Handles tokenization and answer span localization:

```python
from src import QADataLoader, find_answer_span

loader = QADataLoader(model_name="distilbert-base-uncased-distilled-squad")
encoded = loader.tokenize_qa_with_offsets(question, context)

start_idx, end_idx = find_answer_span(
    loader.tokenizer, context, answer_text,
    encoded["offset_mapping"], encoded.sequence_ids()
)
```

### Training Loop

```python
from src import train_retention_scorer

trainer, result = train_retention_scorer(
    scorer=scorer,
    qa_model=qa_model,
    hidden_states=hidden_states,
    protected_mask=protected_mask,
    valid_mask=valid_mask,
    start_target=start_idx,
    end_target=end_idx,
    target_budget=target_budget,
    num_steps=500
)
```

### Evaluation and Analysis

```python
from src import RetentionAnalyzer

analyzer = RetentionAnalyzer(tokens, probabilities, protected_mask, valid_mask)
analyzer.print_summary()
analyzer.print_ranking(top_k=10)

expected_retained = analyzer.get_expected_retained_tokens()
retention_ratio = analyzer.get_retention_ratio()
```

---

## Loss Functions

The model is trained with three complementary objectives:

$$\mathcal{L}_{total} = \mathcal{L}_{QA} + \lambda_{budget} \mathcal{L}_{budget} + \lambda_{entropy} \mathcal{L}_{entropy}$$

### QA Loss
Maintains task performance on question-answering:
$$\mathcal{L}_{QA} = \frac{1}{2}[\ell_{CE}(\text{start}) + \ell_{CE}(\text{end})]$$

### Budget Loss
Encourages retention within the memory budget:
$$\mathcal{L}_{budget} = \mathbb{E}[\max(0, N_{retained} - B)]^2$$

where $N_{retained} = \sum_t p_t$ and $B$ is the budget.

### Entropy Loss
Encourages sharp decisions (not ambiguous middle values):
$$\mathcal{L}_{entropy} = -\sum_t [p_t \log p_t + (1-p_t) \log(1-p_t)]$$

---

## Workflow Example

```python
# 1. Import components
from config import DEVICE, TRAINING_STEPS
from src import (
    initialize_reproducibility,
    QADataLoader,
    BaselineQAModel,
    RetentionScorer,
    train_retention_scorer,
    RetentionAnalyzer,
)

# 2. Setup
initialize_reproducibility()
data_loader = QADataLoader()
baseline = BaselineQAModel()

# 3. Prepare data
question = "What is AI?"
context = "AI is artificial intelligence..."
encoded = data_loader.tokenize_qa_with_offsets(question, context, DEVICE)

# 4. Get baseline
hidden_states, _ = baseline.get_hidden_states(
    encoded["input_ids"], encoded["attention_mask"]
)

# 5. Create and train scorer
scorer = RetentionScorer(hidden_dimension=768).to(DEVICE)
trainer, result = train_retention_scorer(
    scorer, baseline.qa_model, hidden_states,
    protected_mask, valid_mask, start_target, end_target,
    target_budget, num_steps=TRAINING_STEPS
)

# 6. Analyze
scorer.eval()
with torch.no_grad():
    _, probs = scorer(hidden_states)

analyzer = RetentionAnalyzer(tokens, probs, protected_mask, valid_mask)
analyzer.print_summary()
```

---

## Current Capabilities

✅ **Implemented**:
- Baseline DistilBERT loading and freezing
- Learnable retention scorer with soft gating
- Multi-objective training (QA + budget + entropy)
- Protected token masking ([CLS], [SEP])
- Token ranking and importance analysis
- Full single-example pipeline

⏳ **In Development**:
- Actual layer-wise token pruning
- Stochastic/differentiable retention gates
- Integration with dynamic layer skipping
- Comparison with efficient Transformer baselines
- Full dataset training (SQuAD, etc.)

❌ **Not Yet**:
- Model pruning (current: soft gating only)
- Computational savings (soft gating doesn't reduce computation)
- Multi-example batch training
- Fine-tuning pretrained parameters

---

## Research Questions

1. **Can task-aware adaptive retention identify important tokens?**
   - Testing: Do learned probabilities correlate with answer positions?

2. **Does soft retention preserve QA performance?**
   - Metric: Exact Match, F1 on retained vs. baseline predictions

3. **Can we achieve meaningful token reduction within budget?**
   - Metric: Expected retained tokens vs. target budget

4. **How does decision sharpness affect performance?**
   - Ablation: Temperature parameter impact

5. **When will actual efficiency gains appear?**
   - After implementing layer-wise token pruning and measuring latency/FLOPs

---

## Next Steps

1. **Implement Stochastic Token Pruning**
   - Move retention decisions inside Transformer
   - Prune tokens between layers, not after

2. **Measure Actual Efficiency**
   - Track computation and memory usage
   - Compare FLOPs with baseline

3. **Scale to Real Datasets**
   - Train on SQuAD or other QA benchmarks
   - Multiple examples, train/val/test splits
   - Statistical significance testing

4. **Ablation Studies**
   - Contribution of budget loss
   - Contribution of entropy loss
   - Temperature sensitivity
   - Scorer architecture variations

5. **Comparison with Baselines**
   - Other token pruning methods
   - Efficient Transformer variants (Performer, Linformer, etc.)
   - Adaptive computation time mechanisms

---

## Key Design Principles

1. **Modularity**: Each component has a single responsibility
2. **Clarity**: Code is readable and well-documented
3. **Configurability**: Hyperparameters in one place
4. **Reproducibility**: Fixed seeds and device management
5. **Extensibility**: Easy to add new components or experiments

---

## File-by-File Guide

### config.py (90 lines)
Centralized configuration. Every setting used across the project.

### src/utils.py (250 lines)
- Device management
- Reproducibility (seeds)
- Model utilities (parameter counting, freezing)
- Checkpoint saving/loading
- Logging helpers

### src/data.py (220 lines)
- `QADataLoader`: Tokenization and data preparation
- `find_answer_span()`: Locate answer in tokenized sequence
- `create_token_masks()`: Distinguish special/question/context tokens

### src/models.py (200 lines)
- `RetentionScorer`: MLP for token importance scoring
- `SoftRetentionGate`: Probabilistic gating
- `AdaptiveMemoryRetention`: Complete module combining both

### src/losses.py (180 lines)
- `calculate_qa_loss()`: Cross-entropy for start/end positions
- `calculate_budget_loss()`: Penalize exceeding token budget
- `calculate_entropy_loss()`: Regularize decision sharpness
- `calculate_combined_loss()`: Multi-objective combination

### src/baseline.py (200 lines)
- `BaselineQAModel`: Wrapper for DistilBERT QA
- Model initialization, freezing, inference
- `compute_baseline_metrics()`: EM, F1, precision, recall

### src/training.py (300 lines)
- `RetentionScorerTrainer`: Training loop management
- `train_retention_scorer()`: High-level training interface
- Gradient clipping, checkpointing, logging

### src/evaluation.py (280 lines)
- `RetentionAnalyzer`: Analysis of learned probabilities
- `get_top_k_tokens()`: Top-K retained token identification
- `compare_predictions()`: Baseline vs. retained comparison
- Comprehensive reporting

### main.py (350 lines)
Complete pipeline demonstrating all components working together.

### examples.py (150 lines)
Three simple examples for quick understanding.

---

## Contributing

When adding new components:

1. Follow the modular structure
2. Add docstrings to all public functions/classes
3. Keep related functionality together
4. Update imports in `src/__init__.py`
5. Add configuration to `config.py` if needed
6. Document in this README

---

## License

This is a research project. Please cite appropriately when using.

---

## Questions?

Review the code comments and docstrings for detailed explanations of each component.
