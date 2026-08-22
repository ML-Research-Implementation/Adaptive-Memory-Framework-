# AMMR — Adaptive Memory Retention for Transformer Networks

## Project Overview

**AMMR** learns which tokens are important for a Transformer model's task performance, enabling selective token retention. This reduces computational cost (quadratic attention complexity) while preserving model accuracy.

**Key Idea:** Instead of keeping all tokens or using binary pruning, AMMR learns soft probabilistic importance scores for each token (0-1 range), allowing fine-grained control over memory-computation trade-offs.

**Task:** Question Answering (DistilBERT on SQuAD-like data)

**Approach:** 
- Load frozen DistilBERT baseline
- Extract hidden representations 
- Train learnable RetentionScorer (small MLP) to predict token importance
- Use multi-objective loss: QA accuracy + budget constraint + probability sharpness

---

## Current Implementation Progress

### ✅ Phase 1: Baseline Architecture
- **config.py** - Centralized hyperparameters
- **src/data.py** - Base tokenization, answer span location, token masking
- **src/models.py** - Initial RetentionScorer (importance predictor) & SoftRetentionGate
- **src/baseline.py** - DistilBERT wrapper, hidden state extraction

### ✅ Phase 2: Layer-wise Physical Token Pruning
- **src/models_adaptive.py** - `TokenSelector` implementing discrete layer-wise Top-K selection. Enables *actual* tensor compaction (e.g. 31 → 24 → 19 tokens).
- **src/losses.py** - Multi-objective loss formulation (QA + Budget + Entropy).
- **src/training_layerwise.py** - `LayerwiseAdaptiveTrainer` for jointly training the 6 layer-wise retention scorers alongside the QA objective.
- **test_batching.py** - Verified index sorting, logit reconstruction via `scatter_`, and exact tensor shape preservation during deterministic Top-K pruning.

### ✅ Phase 3: SQuAD Dataset & Batched Evaluation
- **src/squad_data.py** - Robust data pipeline utilizing Hugging Face `datasets`. Incorporates standard SQuAD token-mapping, handling truncated sliding windows (`doc_stride`).
- **train_squad.py** - Batched training script over SQuAD dataset subsets.
- **evaluate_squad.py** - Baseline vs. AMMR comparative evaluation logic for extracting validation Exact Match (EM), F1 Score, Latency, and Estimated Attention Costs across configured retention ratios.

### ❌ Next Steps (Phase 4 & 5)
- **Stochastic Relaxation (Phase 4):** Swap deterministic Top-K with differentiable Hard-Concrete / Gumbel-Softmax gating to enable backpropagation through the index selection.
- **Full Dataset Training (Phase 5):** End-to-end training over the entire 87k SQuAD training dataset with stochastic gating.

---

## How to Test

### Quick Test: Full Pipeline
```bash
cd d:\Adaptive-Memory-Framework-
python main.py
```

Runs 11 steps:
1. Load model → 2. Prepare data → 3. Locate answer → 4. Baseline prediction
5. Extract hidden states → 6. Create masks → 7. Initialize scorer
8. Train scorer (500 steps) → 9. Analyze tokens → 10. Predict with retention → 11. Report

**Expected**: Completes in 2-5 minutes (CPU) or 30-60 sec (GPU), shows token rankings and training progress.

### Run Examples
```bash
python examples.py
```

### Individual Component Tests
```python
# Test data loading
from src import QADataLoader
loader = QADataLoader()
tokens = loader.get_tokens(loader.tokenize_qa("What is AI?", "AI is..."))

# Test baseline model
from src import BaselineQAModel
baseline = BaselineQAModel(freeze_parameters=True)
hidden, layers = baseline.get_hidden_states(input_ids, attention_mask, return_all_layers=True)

# Test retention scorer
from src import RetentionScorer
scorer = RetentionScorer(hidden_dimension=768)
scores, probs = scorer(hidden_states)

# Test losses
from src import calculate_combined_loss
loss, details = calculate_combined_loss(qa_loss, budget_loss, entropy_loss)

# Test training
from src import train_retention_scorer
trainer, results = train_retention_scorer(scorer, qa_model, hidden_states, ...)

# Test analysis
from src import RetentionAnalyzer
analyzer = RetentionAnalyzer(tokens, probs, protected_mask, valid_mask)
analyzer.print_summary()
ranking = analyzer.get_token_ranking()
```

---

## Key Results: Accuracy-Efficiency Trade-off (Phase 3)

After implementing batched physical token pruning and training our `RetentionScorer` modules on a small SQuAD subset, we ran evaluations across uniform retention schedules (`r = 0.9` to `0.5`). 

| Model / Ratio | Exact Match | F1 Score | Tokens Retained | Attn Cost (est) | Latency/batch |
| --- | --- | --- | --- | --- | --- |
| Baseline | 75.00 | 82.26 | 100.0% | 100.0% | 1579.61 ms |
| AMMR (r=0.90) | 22.00 | 27.15 | 89.9% | 80.8% | 1752.91 ms |
| AMMR (r=0.80) | 10.00 | 13.05 | 79.8% | 63.7% | 1288.76 ms |
| AMMR (r=0.70) | 0.00 | 0.26 | 69.8% | 48.8% | 1188.23 ms |
| AMMR (r=0.60) | 0.00 | 0.76 | 59.9% | 35.8% | 898.80 ms |
| AMMR (r=0.50) | 0.00 | 0.44 | 50.0% | 25.0% | 743.73 ms |

### Explanation of Results

1. **Efficiency Gains Achieved:** 
   The model effectively speeds up sequence processing. At a 50% retention limit, the average CPU latency drops from ~1.58s to ~0.74s, representing over a **50% speedup**. Additionally, the estimated Attention Cost—which scales quadratically $O(N^2)$—drops by a massive **75%**. Note that at $r=0.9$, latency is slightly higher than the baseline due to the non-fused PyTorch overhead of running `top-k` and `scatter_` operations on small sequences.

2. **Accuracy Degradation:**
   As seen in the table, Exact Match and F1 crash heavily even at a 90% retention ratio. Why? Because the `torch.topk` physical selection is a **discrete, non-differentiable operation**. While the model trains, the loss gradients cannot flow back through the index selection step to tell the scorers which "dropped" tokens should have been retained. 

**Next Step (Phase 4):** To resolve the accuracy drop, we will implement **Stochastic Relaxation** (Hard-Concrete / Gumbel-Softmax gating), making the token pruning operation mathematically differentiable.

---

## Documentation

- **TESTING_GUIDE.md** - Comprehensive testing with 4 levels + debugging
- **IMPLEMENTATION_SUMMARY.md** - Detailed feature list and architecture
- **PROJECT_STRUCTURE.md** - File layout and module overview
- **README_MODULAR.md** - Deep architecture explanation
- **QUICKREF.md** - Quick lookup and code patterns
- **REFACTORING_GUIDE.md** - How we refactored from monolithic code

---

## Architecture at a Glance

```
Question & Context
       ↓
QADataLoader (tokenize, locate answer, create masks)
       ↓
BaselineQAModel (frozen DistilBERT)
       ↓
Extract Hidden States (31 × 768)
       ↓
RetentionScorer (small MLP)
       ↓
Soft Gating: h'_t = p_t * h_t
       ↓
Train (500 steps, 3-loss objective)
       ↓
Analyze (token ranking, statistics)
       ↓
Predict with retained representation
```

---

## Getting Started

1. **Validate setup**: `python main.py` (or `python test_imports.py`)
2. **Understand code**: Read QUICKREF.md for patterns
3. **Modify settings**: Edit config.py (hyperparameters)
4. **Extend**: Add new components in src/ and export via src/__init__.py

---

## Status: ✅ Phase 3 Complete, Proceeding to Phase 4

All foundational and data components are working. The framework currently successfully compacts layers physically in batches but suffers from missing gradients due to deterministic Top-K operation. We are now preparing to implement stochastic Hard-Concrete pruning (Phase 4).
