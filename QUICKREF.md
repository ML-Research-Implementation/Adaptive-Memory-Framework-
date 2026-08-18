# AMMR Quick Reference

## File Structure at a Glance

```
📦 Adaptive-Memory-Framework/
│
├── 🔧 config.py              # All settings in ONE place
├── 🎯 main.py                # Complete pipeline demo
├── 📚 examples.py            # Simple usage examples
│
├── 📖 README_MODULAR.md      # Full documentation
├── 🔄 REFACTORING_GUIDE.md   # Migration from old to new
├── 📝 QUICKREF.md            # This file
│
├── 🗂️ src/
│   ├── __init__.py           # Package exports
│   ├── utils.py              # 🔧 Utilities (250 lines)
│   ├── data.py               # 📊 Data loading (220 lines)
│   ├── models.py             # 🧠 Neural networks (200 lines)
│   ├── losses.py             # 📉 Loss functions (180 lines)
│   ├── baseline.py           # 📌 Baseline model (200 lines)
│   ├── training.py           # 🎓 Training loop (300 lines)
│   └── evaluation.py         # 📊 Analysis (280 lines)
│
└── [Legacy files for reference]
    ├── distilbert_demo.py
    ├── readme.md
    └── src/{baseline_distilbert.py, learned_retention_prototype.py}
```

## Import Quick Reference

### Everything from one place
```python
from src import *  # Get all components
```

### Specific imports
```python
# Configuration
from config import DEVICE, LEARNING_RATE, TRAINING_STEPS

# Data
from src import QADataLoader, find_answer_span, create_token_masks

# Models
from src import RetentionScorer, AdaptiveMemoryRetention

# Training
from src import train_retention_scorer, RetentionScorerTrainer

# Evaluation
from src import RetentionAnalyzer, compute_baseline_metrics

# Utilities
from src import initialize_reproducibility, freeze_model, print_header
```

## Typical Workflow

```python
# 1️⃣ SETUP
from config import *
from src import *
initialize_reproducibility()

# 2️⃣ DATA
loader = QADataLoader()
encoded = loader.tokenize_qa_with_offsets(question, context, DEVICE)
start_idx, end_idx = find_answer_span(loader.tokenizer, context, 
                                      answer_text, 
                                      encoded["offset_mapping"], 
                                      encoded.sequence_ids())

# 3️⃣ BASELINE
baseline = BaselineQAModel(freeze_parameters=True)
hidden_states, _ = baseline.get_hidden_states(encoded["input_ids"], 
                                              encoded["attention_mask"])

# 4️⃣ MASKS
valid_mask, protected_mask = create_token_masks(
    seq_len, sequence_ids, attention_mask, device=DEVICE
)

# 5️⃣ TRAINING
scorer = RetentionScorer(hidden_dimension=768).to(DEVICE)
trainer, result = train_retention_scorer(
    scorer, baseline.qa_model, hidden_states,
    protected_mask, valid_mask, 
    start_target, end_target, target_budget,
    num_steps=TRAINING_STEPS, verbose=True
)

# 6️⃣ ANALYSIS
scorer.eval()
with torch.no_grad():
    _, probs = scorer(hidden_states)
    
analyzer = RetentionAnalyzer(tokens, probs, protected_mask, valid_mask)
analyzer.print_summary()
analyzer.print_ranking(top_k=15)
```

## Key Classes

### RetentionScorer
```python
scorer = RetentionScorer(hidden_dimension=768)
scores, probs = scorer(hidden_states, temperature=1.0)
# Returns: scores (batch, seq_len), probs (batch, seq_len)
```

### BaselineQAModel
```python
baseline = BaselineQAModel(model_name="...", freeze_parameters=True)
start, end, answer, info = baseline.get_baseline_prediction(input_ids, attn_mask)
hidden, all_layers = baseline.get_hidden_states(input_ids, attn_mask)
```

### QADataLoader
```python
loader = QADataLoader(model_name="...")
encoded = loader.tokenize_qa(question, context)
tokens = loader.get_tokens(input_ids)
answer = loader.decode_span(input_ids, start, end)
```

### RetentionAnalyzer
```python
analyzer = RetentionAnalyzer(tokens, probs, protected_mask, valid_mask)
analyzer.print_summary()           # Print statistics
ranking = analyzer.get_token_ranking()  # List ranked tokens
expected = analyzer.get_expected_retained_tokens()
ratio = analyzer.get_retention_ratio()
```

### RetentionScorerTrainer
```python
trainer = RetentionScorerTrainer(scorer, qa_model)
result = trainer.train_step(hidden_states, protected_mask, valid_mask, 
                           start_target, end_target, target_budget)
trainer.save_checkpoint("path/to/checkpoint.pt")
trainer.load_checkpoint("path/to/checkpoint.pt")
```

## Loss Functions

```python
from src.losses import *

# Individual losses
qa_loss, start_logits, end_logits = calculate_qa_loss(
    qa_model, gated_hidden, start_target, end_target
)

budget_loss, expected_tokens = calculate_budget_loss(
    probs, valid_mask, target_budget, penalty_mode='excess'
)

entropy_loss = calculate_entropy_loss(probs, valid_mask)

# Combined
total_loss, loss_dict = calculate_combined_loss(
    qa_loss, budget_loss, entropy_loss,
    budget_weight=0.10, entropy_weight=0.001
)
```

## Configuration Parameters

```python
# Model
MODEL_NAME = "distilbert-base-uncased-distilled-squad"
HIDDEN_DIMENSION = 768
NUM_TRANSFORMER_LAYERS = 6

# Retention
RETENTION_RATIO = 0.50              # Retain 50% of tokens
TEMPERATURE = 1.0                   # Probability sharpness

# Training
LEARNING_RATE = 1e-3
TRAINING_STEPS = 500
GRADIENT_CLIP = 1.0
OPTIMIZER_WEIGHT_DECAY = 1e-4

# Loss Weights
BUDGET_LAMBDA = 0.10                # Memory budget weight
ENTROPY_LAMBDA = 0.001              # Decision sharpness weight

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## Running Code

### Full Demo
```bash
python main.py
```

### Simple Examples
```bash
python examples.py
```

### Custom Script
```python
# any_script.py
from src import *
from config import *

# Your code here
```

## Common Tasks

### Task: Change learning rate
```python
# Edit config.py
LEARNING_RATE = 5e-4  # Changed from 1e-3
# Run: python main.py
```

### Task: Change retention ratio
```python
# Edit config.py
RETENTION_RATIO = 0.30  # Retain 30% instead of 50%
# Run: python main.py
```

### Task: Freeze/unfreeze model
```python
from src import freeze_model, unfreeze_model
freeze_model(baseline.qa_model)
unfreeze_model(baseline.qa_model)
```

### Task: Save trained scorer
```python
from src.utils import save_checkpoint
save_checkpoint(scorer, optimizer, step, "checkpoints/scorer.pt")
```

### Task: Load trained scorer
```python
from src.utils import load_checkpoint
step = load_checkpoint(scorer, optimizer, "checkpoints/scorer.pt")
```

### Task: Get model statistics
```python
from src import count_parameters, count_trainable_parameters
total = count_parameters(model)
trainable = count_trainable_parameters(model)
```

## Module Sizes

| Module | Lines | Purpose |
|--------|-------|---------|
| config.py | 90 | Centralized settings |
| src/utils.py | 250 | Common utilities |
| src/data.py | 220 | Tokenization & data |
| src/models.py | 200 | Neural networks |
| src/losses.py | 180 | Loss functions |
| src/baseline.py | 200 | Baseline model |
| src/training.py | 300 | Training loop |
| src/evaluation.py | 280 | Analysis |
| **Total** | **1,720** | **8 focused modules** |

*Compare to old: 1000+ lines in single files*

## Architecture Diagram

```
┌─ config.py (settings)
│
├─ main.py (orchestration)
│   ├─► QADataLoader (data.py)
│   ├─► BaselineQAModel (baseline.py)
│   ├─► RetentionScorer (models.py)
│   ├─► train_retention_scorer (training.py)
│   │   ├─► calculate_qa_loss (losses.py)
│   │   ├─► calculate_budget_loss (losses.py)
│   │   ├─► calculate_entropy_loss (losses.py)
│   │   └─► calculate_combined_loss (losses.py)
│   └─► RetentionAnalyzer (evaluation.py)
│
└─ examples.py (demonstrations)
    ├─► Setup examples
    ├─► Tokenization examples
    └─► Hidden state examples
```

## Performance Tips

```python
# 1. Use batch processing (future enhancement)
loader.tokenize_qa_batch([q1, q2, q3], [c1, c2, c3])

# 2. Gradient accumulation (in RetentionScorerTrainer)
# Already supported via custom training loop

# 3. Mixed precision (add to training.py if needed)
# from torch.cuda.amp import autocast
# with autocast():
#     loss = compute_loss(...)

# 4. Profile code (use torch.profiler)
# from torch.profiler import profile
# with profile() as prof:
#     # Your code
```

## Debugging

```python
# Enable verbose logging
print_header("DEBUG INFO")

# Print model architecture
print(baseline.model)
print(scorer)

# Check tensor shapes
print(f"Hidden states: {hidden_states.shape}")
print(f"Probabilities: {probabilities.shape}")
print(f"Tokens: {len(tokens)}")

# Print training progress
result = trainer.train_step(...)
print(trainer.format_result(result))

# Analyze learned probabilities
analyzer.print_ranking()
print(f"Expected retained: {analyzer.get_expected_retained_tokens()}")
```

## What's Next?

- ✅ Modular architecture complete
- ⏳ Stochastic token pruning (layer-wise)
- ⏳ Real dataset training (SQuAD)
- ⏳ Efficiency benchmarking
- ⏳ Comparison with baselines
- ⏳ Ablation studies

---

**Start here:** Read README_MODULAR.md for full documentation  
**Quick start:** Run `python main.py`  
**Learn modules:** Check `examples.py`  
**Extend code:** Follow patterns in `src/`
