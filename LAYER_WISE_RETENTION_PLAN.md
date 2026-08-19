# Layer-Wise Retention Implementation Plan

## Overview
Transform the current prototype (end-of-pipeline retention) into an **in-pipeline adaptive computation framework** where retention is applied between every Transformer layer. This plan progresses from deterministic Top-K gating through full stochastic learning, with rigorous metrics and benchmarking at each stage.

---

## Phase 1: Architecture & Preparation (Days 1-2)

### 1.1 Design Custom DistilBERT Forward Pass
**Objective:** Create a new model class that wraps DistilBERT and inserts retention gates between layers.

**Tasks:**
- [ ] Create `src/models_adaptive.py` with `AdaptiveDistilBertQA` class
- [ ] Understand DistilBERT's internal structure:
  - Input embeddings (embedding layer)
  - 6 Transformer layers (each layer: multi-head attention + FFN)
  - Output layer for QA head
- [ ] Design hooks/callbacks to intercept between layers
- [ ] Add configuration for:
  - Target retention ratio per layer (75%, 50%, etc.)
  - Whether to apply retention at each layer
  - Special token preservation strategy
  - Token selection method (Top-K, stochastic, etc.)

**Deliverable:** `AdaptiveDistilBertQA` skeleton with forward pass structure

### 1.2 Design Token Selection & Masking Strategy
**Objective:** Decide how to preserve structural tokens and reduce sequences safely.

**Tasks:**
- [ ] Define token categories:
  - `PROTECTED`: [CLS], [SEP] → always kept
  - `QUESTION`: tokens in question span → may be selectively retained
  - `CONTEXT`: tokens in context span → may be selectively retained
  - `IGNORED`: padding tokens → excluded from retention
- [ ] Design retention algorithm:
  - For each layer, compute retention scores using `RetentionScorer`
  - Select Top-K tokens by score (deterministic)
  - Enforce minimum tokens (e.g., at least [CLS] + some question tokens)
  - Return reduced sequence + mapping for later reconstruction
- [ ] Handle position embeddings & attention masks after token reduction
- [ ] Plan for gradient flow through the selection process

**Deliverable:** Token selection strategy document + pseudocode

### 1.3 Create Metrics & Logging Infrastructure
**Objective:** Set up tracking for all important quantities.

**Tasks:**
- [ ] Create `src/metrics.py` with `LayerWiseMetrics` class to track:
  - **Per-layer:** tokens in, tokens out, retention ratio
  - **Theoretical speedup:** compute O(n²) attention reduction
  - **Latency:** wall-clock time per layer (CPU only initially)
  - **Quality:** final F1, EM, answer match vs. baseline
- [ ] Add real-time visualization hooks (print summaries during inference)
- [ ] Plan logging format for comparison across retention budgets
- [ ] Design experiment tracking (configuration + results per run)

**Deliverable:** `LayerWiseMetrics` class + logging system

---

## Phase 2: Implement Deterministic Top-K Gate (Days 2-4)

### 2.1 Implement AdaptiveDistilBertQA with Top-K Selection
**Objective:** Build the custom forward pass with deterministic retention.

**Tasks:**
- [ ] Implement `AdaptiveDistilBertQA.forward()`:
  1. Embed input → (batch, 31, 768)
  2. For layer in layers[0:6]:
     - Pass through transformer layer → hidden states
     - Compute retention scores via RetentionScorer
     - Apply Top-K selection (keep top 75%, 50%, etc. of adaptive tokens)
     - Enforce protected tokens
     - Log metrics (tokens before/after)
  - [ ] Implement `_select_top_k_tokens()`:
     - Takes scores, protection mask, retention ratio
     - Returns selected token indices + updated attention mask
  - [ ] Handle dimension changes:
     - Position embeddings only for selected tokens
     - Attention masks updated for reduced sequence
     - Attention bias updated for new token indices
  - [ ] Preserve baseline behavior: when retention_ratio=1.0, should match original

**Deliverable:** Working `AdaptiveDistilBertQA` class with Top-K gating

### 2.2 Create Wrapper for Unified Inference
**Objective:** Single interface for running with different retention ratios.

**Tasks:**
- [ ] Create `AdaptiveQAInference` class that:
  - Loads baseline DistilBERT (frozen)
  - Initializes RetentionScorer (untrained initially)
  - Runs adaptive forward pass
  - Collects metrics + computes answer prediction
- [ ] Add methods:
  - `forward_adaptive(input_ids, attention_mask, retention_ratio_per_layer=None)` → answer + metrics
  - `compare_with_baseline()` → side-by-side results
- [ ] Support multiple retention budgets in a single call

**Deliverable:** `AdaptiveQAInference` class

### 2.3 Implement Metrics Collection & Reporting
**Objective:** Capture all data about layer-wise retention.

**Tasks:**
- [ ] Update `LayerWiseMetrics` to:
  - Record token counts per layer
  - Calculate theoretical O(n²) reduction: sum of (n_out[i]² / n_in[i]²) for each layer
  - Estimate actual speedup assuming linear layer processing
  - Log latency per layer (CPU)
- [ ] Create reporting function:
  - Table: Layer | In | Out | Ratio | Speedup Factor
  - Summary: Total attention ops reduction
  - Answer accuracy vs. baseline
- [ ] Add visualization (ASCII tables for now)

**Deliverable:** Metrics system + reporting functions

---

## Phase 3: Validation & Benchmarking (Days 4-5)

### 3.1 Single-Example Tests with Multiple Budgets
**Objective:** Verify deterministic Top-K works correctly without training.

**Tasks:**
- [ ] Test on a single Q&A pair with:
  - Retention ratio = 100% (should match baseline exactly)
  - Retention ratio = 75%
  - Retention ratio = 50%
  - Retention ratio = 25%
- [ ] For each:
  - Print layer-by-layer token counts
  - Check that [CLS], [SEP] are preserved
  - Verify answer prediction (may differ, but should be reasonable)
  - Log total inference latency
  - Calculate theoretical attention speedup
- [ ] Create comparison table: Retention% vs. Answer vs. Speedup

**Deliverable:** Test script + results on single example

### 3.2 Implement Proper QA Answer Extraction
**Objective:** Ensure answer span prediction works with reduced sequences.

**Tasks:**
- [ ] Modify answer extraction to:
  - Map predicted token indices back to original sequence (via retained token indices)
  - Extract answer span from original token list
  - Compare with ground truth
  - Calculate F1 & EM scores
- [ ] Handle edge case: if answer tokens were pruned, score = 0

**Deliverable:** Robust answer extraction + F1/EM calculation

### 3.3 Create Baseline Comparison Module
**Objective:** Standardized comparison against original DistilBERT.

**Tasks:**
- [ ] Create `BaselineComparisonBenchmark` class:
  - Run original baseline on same example
  - Run adaptive version at multiple retention ratios
  - Side-by-side table: Model | Answer | F1 | EM | Latency | Speedup
- [ ] Test on several examples (not just one)

**Deliverable:** Benchmark class + multi-example comparison

---

## Phase 4: Integration with Training Pipeline (Days 5-6)

### 4.1 Integrate RetentionScorer Training
**Objective:** Train the scorer to predict good retention decisions (before going stochastic).

**Tasks:**
- [ ] Modify `src/training.py` to work with layer-wise retention:
  - For each training step:
    - Forward pass through adaptive model
    - Compute QA loss (backprop through reduced sequences)
    - Compute budget loss (encourage hitting target retention ratio)
    - Compute entropy loss (sharpen retention scores)
  - Update RetentionScorer via gradient descent
- [ ] Handle gradient flow:
  - Top-K is non-differentiable → use straight-through estimator or proxy loss
  - Alternatively: train scorer via proxy loss before switching to Top-K
- [ ] Log training metrics per layer

**Deliverable:** Training loop for adaptive model

### 4.2 Create Training Script for Deterministic Version
**Objective:** Train retention scorer with Top-K gating.

**Tasks:**
- [ ] Create `train_layer_wise_topk.py`:
  - Load baseline DistilBERT
  - Initialize RetentionScorer
  - Train for N steps on a single Q&A pair
  - Log loss progression
  - Save trained scorer
- [ ] Run on several examples, measure convergence

**Deliverable:** Working training script + trained model checkpoint

---

## Phase 5: Stochastic Replacement (Days 6-7)

### 5.1 Replace Top-K with Stochastic Gating
**Objective:** Switch from deterministic to learned stochastic retention.

**Tasks:**
- [ ] Create `StochasticRetentionGate` class:
  - Takes retention scores → probability distribution
  - Samples tokens during training (with gradient via Gumbel-softmax or similar)
  - Uses expected retention during inference
  - Differentiable end-to-end
- [ ] Implement Gumbel-softmax sampler for differentiable sampling
- [ ] Update `AdaptiveDistilBertQA` to use stochastic gate
- [ ] Verify: deterministic should be a limiting case of stochastic

**Deliverable:** Stochastic gating mechanism

### 5.2 Retrain Scorer with Stochastic Gating
**Objective:** Learn retention decisions end-to-end.

**Tasks:**
- [ ] Update training loop for stochastic version
- [ ] Compare trained scorers: Top-K vs. stochastic
- [ ] Verify stochastic version converges
- [ ] Measure: same tasks as Top-K version (speedup, F1, etc.)

**Deliverable:** Trained stochastic scorer + comparison

---

## Phase 6: Real Dataset Experiments (Days 7-10)

### 6.1 Prepare SQuAD Dataset Integration
**Objective:** Move beyond single examples to real benchmarking.

**Tasks:**
- [ ] Load SQuAD (or subset) into data pipeline
- [ ] Create batch processing support (if not already present)
- [ ] Implement distributed inference (process multiple examples)
- [ ] Aggregate metrics across dataset

**Deliverable:** SQuAD data loader + batch inference

### 6.2 Run Comprehensive Experiments
**Objective:** Measure accuracy-efficiency trade-off across retention budgets.

**Tasks:**
- [ ] For each retention ratio (100%, 75%, 50%, 25%):
  - Train scorer on dataset (multiple Q&A pairs)
  - Evaluate on test set
  - Record: F1, EM, latency, theoretical speedup, actual speedup
  - Save results to CSV
- [ ] Create comparison plots:
  - X-axis: Theoretical speedup or retention ratio
  - Y-axis: F1 or EM score
  - Show trade-off curve
- [ ] Conduct ablations:
  - Layer-wise vs. global retention ratio
  - Different loss weights (budget, entropy)
  - Impact of protecting special tokens

**Deliverable:** Results CSV + comparison plots + ablation study

### 6.3 Final Analysis & Report
**Objective:** Document findings.

**Tasks:**
- [ ] Generate final report:
  - Method overview
  - Results table + plots
  - Key findings (speedup vs. accuracy loss)
  - Recommendations for deployment
- [ ] Compare against baselines (uncompressed DistilBERT, naive pruning, etc.)

**Deliverable:** Final report + figures

---

## Implementation Order (Recommended)

1. **Phase 1**: Design + architecture (1-2 days)
2. **Phase 2**: Implement deterministic Top-K (2 days)
3. **Phase 3**: Validation (1 day)
4. **Phase 4**: Training integration (1 day)
5. **Phase 5**: Stochastic replacement (1 day)
6. **Phase 6**: Real dataset experiments (3+ days)

**Timeline:** ~10 days for full implementation + experiments

---

## Key Milestones

✅ **Milestone 1:** `AdaptiveDistilBertQA` works with Top-K on single example  
✅ **Milestone 2:** Metrics + benchmarking on multiple retention budgets  
✅ **Milestone 3:** Training loop for layer-wise scorer (deterministic)  
✅ **Milestone 4:** Switch to stochastic gating + retrain  
✅ **Milestone 5:** Full SQuAD experiment with accuracy-efficiency trade-off curves  

---

## Open Questions & Decisions

1. **Token Reduction Strategy**
   - Decision: Top-K deterministic first, then stochastic
   - Alternative: Could use gradient-based selection (learned masking)

2. **Special Token Handling**
   - Current plan: Always keep [CLS], [SEP]
   - Alternative: Learn protection strategy (which tokens are "essential"?)

3. **Attention Mask Updates**
   - When we reduce tokens, how do we update position embeddings?
   - Plan: Use new indices (no gaps in position embeddings)
   - This may require careful attention bias handling

4. **Gradient Flow Through Selection**
   - Top-K is non-differentiable
   - Plan: Use proxy losses (not backprop through Top-K itself)
   - Alternative: Use soft selection (learned gating) from start

5. **Layer-wise vs. Global Budget**
   - Current plan: Fixed retention ratio per layer (e.g., 75% always)
   - Alternative: Learned, adaptive ratio per layer
   - Phase 6 ablation will test this

6. **Real Speedup Measurement**
   - Theoretical speedup is easy (O(n²) reduction)
   - Actual speedup on CPU/GPU depends on implementation
   - Plan: Measure latency with inference timers; may need CUDA profiling later

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/models_adaptive.py` | AdaptiveDistilBertQA class (main mechanism) |
| `src/metrics.py` | LayerWiseMetrics + reporting |
| `src/inference_adaptive.py` | AdaptiveQAInference wrapper |
| `train_layer_wise_topk.py` | Training script (deterministic version) |
| `benchmark_layer_wise.py` | Comparison + metrics reporting |
| `train_layer_wise_stochastic.py` | Training script (stochastic version) |
| `experiments_squad.py` | Full SQuAD experiment pipeline |

---

## Success Criteria

- ✅ Deterministic Top-K works correctly (reproduces baseline at 100%)
- ✅ Token reduction per layer logged and verified
- ✅ Theoretical speedup calculated correctly
- ✅ Training loop optimizes retention scores
- ✅ Stochastic version converges
- ✅ Accuracy-efficiency trade-off demonstrated on real data
- ✅ Final report with clear findings

