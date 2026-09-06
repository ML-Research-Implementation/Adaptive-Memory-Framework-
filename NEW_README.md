# Adaptive Memory Framework (AMMR)

AMMR is a **Differentiable Adaptive Gating** framework for accelerating Question Answering models (specifically DistilBERT) through dynamic, layer-wise token pruning. By learning which tokens to drop and which to retain at each transformer layer, AMMR drastically reduces attention computation cost during inference while maintaining high QA accuracy.

## Key Features

- **Layer-wise Retention Mechanism**: Tokens are dynamically pruned at each layer, reducing the sequence length progressively.
- **Differentiable Hard-Concrete Gating**: Uses stochastic Hard-Concrete gates during training for valid gradient flow, and fast deterministic binary decisions during inference.
- **Teacher-Student Distillation**: 
  - **Logit KD**: T-scaled KL divergence matching the teacher's start/end logits.
  - **Hidden State KD**: Token-aligned Mean Squared Error (MSE) matching the teacher's hidden states.
- **Dynamic Budget Tracking**: Employs a dual-gradient Lagrangian multiplier to dynamically penalize token retention when it exceeds a targeted schedule (e.g., a curriculum that steps down from 95% to 60%).
- **Percentile-based Calibration**: Automatically finds the perfect gating bias threshold via percentile analysis to hit exact target retention budgets (e.g., 70%, 80%) on a validation subset.
- **Mixed Precision Training**: Fully optimized with PyTorch AMP and AdamW for rapid experimentation.

## What We Have Implemented So Far

The project has been implemented in a structured, multi-phase approach:

1. **Phase 1: Baseline Setup**
   - Established the baseline `BaselineQAModel` using standard HuggingFace DistilBERT.
   - Built the SQuAD preprocessing pipeline (`src/squad_data.py`).
2. **Phase 2: Layer-Wise Token Retention**
   - Implemented `AdaptiveDistilBertQA` with custom `TokenSelector` to dynamically prune tokens at each of the 6 transformer layers.
   - Built indexing mechanisms to map compacted hidden states back to original positions for the final QA head.
3. **Phase 3: Stabilization & Checkpointing**
   - Refactored multiple legacy scripts into a unified, stable `train.py`.
   - Built a fully reproducible checkpointing system (`save_checkpoint`/`load_checkpoint`) covering model weights, optimizer states, scheduler states, epoch, step, and Lagrangian variables.
4. **Phase 4: Teacher-Student Distillation**
   - Separated the frozen DistilBERT teacher from the trainable student retention scorers.
   - Implemented multi-objective Knowledge Distillation ($L_{QA} + L_{LogitKD} + L_{HiddenKD}$) using T-scaled KL divergence and token-aligned MSE.
5. **Phase 5: Differentiable Gating & Evaluation (Current State)**
   - Replaced non-differentiable Top-K selection with **Hard-Concrete Gating** (Gumbel-Softmax formulation).
   - Built a dynamic **Dual-Gradient Lagrangian penalty** that enforces target token budgets during training.
   - Created `evaluate_squad.py` with an automated percentile-based calibration function to exactly match efficiency targets.
   - Added `run_ablations.py` to seamlessly execute and compare KD variants.
   - Implemented comprehensive unit tests for core components.

## Setup

1. Install dependencies (PyTorch, Transformers, Datasets, tqdm).
2. Ensure you have the `rajpurkar/squad` dataset available (it will be downloaded automatically via HuggingFace).

## Usage

### 1. Training the Adaptive Model

To train the AMMR student model using the frozen DistilBERT teacher, run the unified training script:

```bash
python train.py --epochs 5 --max_train_samples 10000 --batch_size 16
```

**Key Arguments:**
- `--epochs`: Number of training epochs (Curriculum will scale automatically).
- `--max_train_samples`: Number of training samples to use.
- `--batch_size`: Training batch size.
- `--learning_rate`: AdamW learning rate (default: 3e-3).
- `--resume_from`: Path to a checkpoint to resume training gracefully.

### 2. Evaluation & Calibration

The evaluation script automatically calibrates the model to exact retention thresholds (e.g. 50%, 60%, 70%, 80%) and generates an accuracy-efficiency Pareto curve compared against the baseline.

```bash
python evaluate_squad.py
```

*This will generate `calibration.json` storing the exact threshold biases calculated for your checkpoint.*

### 3. Ablation Studies

To automatically train and evaluate architectural variants (QA-Only, Logit-KD Only, and Full-KD), use the ablation runner:

```bash
python run_ablations.py
```

Add the `--test` flag for a rapid dry-run on a tiny subset of data to verify the pipeline.

### 4. Running Unit Tests

The core components (Hard-Concrete gates, Token Selection, Lagrangian budgets) are backed by unit tests. Run them using:

```bash
python -m unittest tests/test_components.py
```

## Architecture Details

1. **Input**: Sequence of tokens (e.g., 384 length).
2. **Layer `L`**: The sequence passes through the transformer layer.
3. **Retention Scorer**: A lightweight linear layer scores each token.
4. **Token Selector**: Retains tokens where $Logits + Bias > 0$. Special tokens (`[CLS]`, `[SEP]`) are always protected.
5. **Compaction**: The sequence is physically compacted, padding is removed, and the reduced sequence is passed to Layer `L+1`.
6. **Reconstruction**: At the final QA head, the sequence is scattered back to its original length using index preservation, allowing the standard cross-entropy QA loss to function identically to the baseline.
