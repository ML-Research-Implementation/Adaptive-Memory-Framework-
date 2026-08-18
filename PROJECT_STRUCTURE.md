

## Directory Layout

```
Adaptive-Memory-Framework/
│
├── config.py                     # Centralized configuration
├── main.py                       # Complete pipeline
├── examples.py                   # Usage examples
│
├── src/
│   ├── __init__.py               # Package exports
│   ├── utils.py                  # Utilities & helpers
│   ├── data.py                   # Data handling
│   ├── models.py                 # Neural networks
│   ├── losses.py                 # Loss functions
│   ├── baseline.py               # Baseline model
│   ├── training.py               # Training loop
│   └── evaluation.py             # Analysis & evaluation
│
├── README_MODULAR.md             # Full documentation
├── QUICKREF.md                   # Quick reference
├── REFACTORING_GUIDE.md          # Migration guide
│
└── [Legacy files for reference]
    ├── distilbert_demo.py
    ├── readme.md
    └── src/{baseline_distilbert.py, learned_retention_prototype.py, ...}
```

## Core Modules

| Module | Purpose |
|--------|---------|
| **config.py** | All hyperparameters and settings |
| **src/utils.py** | Device, reproducibility, logging, checkpoints |
| **src/data.py** | Tokenization and data preparation |
| **src/models.py** | RetentionScorer and retention modules |
| **src/losses.py** | QA, budget, and entropy loss functions |
| **src/baseline.py** | Baseline DistilBERT model |
| **src/training.py** | Training loop and trainer |
| **src/evaluation.py** | Token analysis and metrics |
| **main.py** | Complete pipeline demonstration |
| **examples.py** | Simple usage examples |

## Key Benefits

✅ **Modularity** - Each component has one responsibility  
✅ **Reusability** - Import and use individual components  
✅ **Configurability** - All settings in `config.py`  
✅ **Extensibility** - Easy to add new features  
✅ **Maintainability** - Clear structure and dependencies  
✅ **Testability** - Test each module independently  

## Quick Start

```bash
# Run the complete pipeline
python main.py

# Run usage examples
python examples.py

# Use in your own script
from src import *
from config import *

loader = QADataLoader()
baseline = BaselineQAModel()
scorer = RetentionScorer()
```

## Adding Features

1. Add configuration to `config.py` if needed
2. Implement in the appropriate `src/` module
3. Export in `src/__init__.py`
4. Use in `main.py` or custom scripts

## Documentation

- **README_MODULAR.md** - Complete reference and examples
- **QUICKREF.md** - Quick lookup and common patterns
- **REFACTORING_GUIDE.md** - Migration from old code structure
