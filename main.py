"""
Main training script for AMMR (Adaptive Memory and Token Retention) framework.

This script demonstrates the complete pipeline:
1. Load baseline DistilBERT model
2. Prepare QA data with answer span localization
3. Create and train retention scorer
4. Evaluate retention mechanism
5. Compare baseline vs. retained predictions
"""

import sys
import torch

# Import from modular components
from config import (
    MODEL_NAME,
    DEVICE,
    SEED,
    RETENTION_RATIO,
    LEARNING_RATE,
    TRAINING_STEPS,
    BUDGET_LAMBDA,
    ENTROPY_LAMBDA,
    TEMPERATURE,
)
from src import (
    initialize_reproducibility,
    print_header,
    QADataLoader,
    find_answer_span,
    create_token_masks,
    BaselineQAModel,
    RetentionScorer,
    train_retention_scorer,
    RetentionAnalyzer,
    compute_baseline_metrics,
)


def main():
    """Main training pipeline."""
    
    # =====================================================================
    # SETUP
    # =====================================================================
    
    print_header("AMMR FRAMEWORK - MAIN PIPELINE")
    
    initialize_reproducibility()
    
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Seed: {SEED}")
    
    # =====================================================================
    # STEP 1: LOAD BASELINE MODEL
    # =====================================================================
    
    print_header("STEP 1: LOAD BASELINE MODEL")
    
    baseline = BaselineQAModel(
        model_name=MODEL_NAME,
        device=DEVICE,
        freeze_parameters=True
    )
    baseline.print_info()
    
    # =====================================================================
    # STEP 2: PREPARE DATA
    # =====================================================================
    
    print_header("STEP 2: PREPARE QA DATA")
    
    # Question and context
    question = "What is artificial intelligence?"
    context = (
        "Artificial intelligence is a field of computer science "
        "that focuses on creating systems capable of performing "
        "tasks that normally require human intelligence."
    )
    answer_text = "a field of computer science"
    
    print(f"Question: {question}")
    print(f"Context: {context[:50]}...")
    print(f"Answer: {answer_text}")
    
    # Tokenize with offset mapping
    data_loader = QADataLoader(MODEL_NAME)
    
    encoded = data_loader.tokenize_qa_with_offsets(
        question,
        context,
        device=DEVICE
    )
    
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    offset_mapping = encoded["offset_mapping"]
    sequence_ids = data_loader.get_sequence_ids(encoded)
    tokens = data_loader.get_tokens(input_ids)
    sequence_length = input_ids.shape[1]
    
    print(f"\nSequence Length: {sequence_length}")
    print("Tokens:")
    for idx, token in enumerate(tokens):
        token_type = "SPECIAL" if sequence_ids[idx] is None else ("QUESTION" if sequence_ids[idx] == 0 else "CONTEXT")
        print(f"  {idx:2d}: {token:15s} ({token_type})")
    
    # =====================================================================
    # STEP 3: FIND ANSWER SPAN
    # =====================================================================
    
    print_header("STEP 3: LOCATE GROUND-TRUTH ANSWER SPAN")
    
    start_idx, end_idx = find_answer_span(
        data_loader.tokenizer,
        context,
        answer_text,
        offset_mapping,
        sequence_ids
    )
    
    start_target = torch.tensor([start_idx], dtype=torch.long, device=DEVICE)
    end_target = torch.tensor([end_idx], dtype=torch.long, device=DEVICE)
    
    print(f"Start Index: {start_idx} ({tokens[start_idx]})")
    print(f"End Index: {end_idx} ({tokens[end_idx]})")
    print(f"Answer Span: {data_loader.decode_span(input_ids, start_idx, end_idx)}")
    
    # =====================================================================
    # STEP 4: GET BASELINE PREDICTION
    # =====================================================================
    
    print_header("STEP 4: BASELINE PREDICTION")
    
    baseline_start, baseline_end, _, baseline_info = baseline.get_baseline_prediction(
        input_ids,
        attention_mask
    )
    
    baseline_answer = data_loader.decode_span(
        input_ids,
        baseline_start,
        baseline_end
    )
    
    baseline_metrics = compute_baseline_metrics(
        baseline_info,
        start_idx,
        end_idx,
        tokens
    )
    
    print(f"Predicted Start: {baseline_start} ({tokens[baseline_start]})")
    print(f"Predicted End: {baseline_end} ({tokens[baseline_end]})")
    print(f"Predicted Answer: {baseline_answer}")
    print(f"Exact Match: {baseline_metrics['exact_match']}")
    print(f"F1 Score: {baseline_metrics['f1']:.4f}")
    
    # =====================================================================
    # STEP 5: GET HIDDEN STATES
    # =====================================================================
    
    print_header("STEP 5: EXTRACT HIDDEN STATES")
    
    hidden_states, all_hidden_states = baseline.get_hidden_states(
        input_ids,
        attention_mask,
        return_all_layers=True
    )
    
    print(f"Final Hidden State Shape: {hidden_states.shape}")
    print(f"Hidden Dimension: {hidden_states.shape[-1]}")
    print(f"Number of Layers: {len(all_hidden_states)}")
    
    # =====================================================================
    # STEP 6: CREATE TOKEN MASKS
    # =====================================================================
    
    print_header("STEP 6: CREATE RETENTION MASKS")
    
    valid_mask, protected_mask = create_token_masks(
        sequence_length,
        sequence_ids,
        attention_mask,
        batch_index=0,
        device=DEVICE
    )
    
    total_tokens = sequence_length
    protected_tokens = protected_mask.sum().item()
    adaptive_tokens = valid_mask.sum().item()
    
    target_budget = max(1, round(adaptive_tokens * RETENTION_RATIO))
    
    print(f"Total Tokens: {total_tokens}")
    print(f"Protected Tokens: {protected_tokens}")
    print(f"Adaptive Tokens: {adaptive_tokens}")
    print(f"Retention Ratio: {RETENTION_RATIO}")
    print(f"Target Retained Tokens: {target_budget}")
    
    # =====================================================================
    # STEP 7: CREATE RETENTION SCORER
    # =====================================================================
    
    print_header("STEP 7: CREATE RETENTION SCORER")
    
    scorer = RetentionScorer(
        hidden_dimension=hidden_states.shape[-1]
    ).to(DEVICE)
    
    print(scorer)
    
    # =====================================================================
    # STEP 8: TRAIN RETENTION SCORER
    # =====================================================================
    
    print_header("STEP 8: TRAIN RETENTION SCORER")
    
    trainer, final_result = train_retention_scorer(
        scorer,
        baseline.qa_model,
        hidden_states,
        protected_mask,
        valid_mask,
        start_target,
        end_target,
        target_budget,
        num_steps=TRAINING_STEPS,
        learning_rate=LEARNING_RATE,
        budget_lambda=BUDGET_LAMBDA,
        entropy_lambda=ENTROPY_LAMBDA,
        temperature=TEMPERATURE,
        log_interval=50,
        verbose=True
    )
    
    # =====================================================================
    # STEP 9: ANALYZE RETENTION
    # =====================================================================
    
    print_header("STEP 9: RETENTION ANALYSIS")
    
    scorer.eval()
    with torch.no_grad():
        scores, final_probabilities = scorer(hidden_states, temperature=TEMPERATURE)
        
        # Apply protection
        protected_mask_device = protected_mask.to(final_probabilities.device)
        final_probabilities = torch.where(
            protected_mask_device.unsqueeze(0),
            torch.ones_like(final_probabilities),
            final_probabilities
        )
    
    analyzer = RetentionAnalyzer(
        tokens,
        final_probabilities,
        protected_mask,
        valid_mask
    )
    
    analyzer.print_summary()
    analyzer.print_ranking(top_k=15)
    
    # =====================================================================
    # STEP 10: RETAINED PREDICTION
    # =====================================================================
    
    print_header("STEP 10: PREDICTION WITH RETAINED TOKENS")
    
    # Apply soft gate
    gated_hidden = hidden_states * final_probabilities.unsqueeze(-1)
    
    # Get prediction
    with torch.no_grad():
        qa_head = baseline.get_qa_head()
        logits = qa_head(gated_hidden)
    
    retained_start = torch.argmax(logits[0, :, 0]).item()
    retained_end = torch.argmax(logits[0, :, 1]).item()
    
    retained_answer = data_loader.decode_span(
        input_ids,
        retained_start,
        retained_end
    )
    
    print(f"Predicted Start: {retained_start} ({tokens[retained_start]})")
    print(f"Predicted End: {retained_end} ({tokens[retained_end]})")
    print(f"Predicted Answer: {retained_answer}")
    
    # =====================================================================
    # STEP 11: FINAL REPORT
    # =====================================================================
    
    print_header("FINAL REPORT")
    
    print(f"\nBaseline Prediction: {baseline_answer}")
    print(f"Retained Prediction: {retained_answer}")
    print(f"Ground Truth Answer: {answer_text}")
    print(f"\nBaseline Metrics:")
    print(f"  - Exact Match: {baseline_metrics['exact_match']:.4f}")
    print(f"  - F1 Score: {baseline_metrics['f1']:.4f}")
    print(f"\nRetention Summary:")
    print(f"  - Expected Retained: {analyzer.get_expected_retained_tokens():.1f}/{adaptive_tokens}")
    print(f"  - Retention Ratio: {analyzer.get_retention_ratio():.2%}")
    print(f"\nTraining Summary:")
    print(f"  - Total Steps: {TRAINING_STEPS}")
    print(f"  - Final Total Loss: {final_result['total']:.5f}")
    print(f"  - Final QA Loss: {final_result['qa']:.5f}")
    print(f"  - Final Budget Loss: {final_result['budget']:.5f}")
    print(f"  - Final Entropy Loss: {final_result['entropy']:.5f}")


if __name__ == "__main__":
    main()
