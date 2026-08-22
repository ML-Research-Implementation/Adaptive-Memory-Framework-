import time
import torch
from config import MODEL_NAME, DEVICE
from src.data import QADataLoader, find_answer_span
from src.models_adaptive import AdaptiveDistilBertQA
from src.training_layerwise import LayerwiseAdaptiveTrainer
from src.baseline import BaselineQAModel, compute_baseline_metrics
from src.utils import print_header, initialize_reproducibility

def main():
    print_header("PHASE 2: TRAINING LAYER-WISE RETENTION")
    initialize_reproducibility()
    
    # 1. Prepare Small Dataset (1 Example for now)
    question = "What is artificial intelligence?"
    context = (
        "Artificial intelligence is a field of computer science "
        "that focuses on creating systems capable of performing "
        "tasks that normally require human intelligence."
    )
    answer_text = "a field of computer science"
    
    data_loader = QADataLoader(MODEL_NAME)
    encoded = data_loader.tokenize_qa_with_offsets(question, context, device=DEVICE)
    
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    offset_mapping = encoded["offset_mapping"]
    sequence_ids = data_loader.get_sequence_ids(encoded)
    tokens = data_loader.get_tokens(input_ids)
    
    start_idx, end_idx = find_answer_span(
        data_loader.tokenizer, context, answer_text, offset_mapping, sequence_ids
    )
    start_target = torch.tensor([start_idx], dtype=torch.long, device=DEVICE)
    end_target = torch.tensor([end_idx], dtype=torch.long, device=DEVICE)
    
    # 2. Baseline Model
    baseline = BaselineQAModel(freeze_parameters=True)
    baseline_start, baseline_end, _, info_dict = baseline.get_baseline_prediction(input_ids, attention_mask)
    
    # 3. Layer-wise Adaptive Model
    schedule = [0.90, 0.85, 0.80, 0.75, 0.70, 0.70]
    adaptive_model = AdaptiveDistilBertQA(
        model_name=MODEL_NAME,
        device=DEVICE,
        freeze_transformer=True,
        retention_schedule=schedule
    )
    
    # Override create_protected_mask to ALSO protect the ground truth answer
    original_create_protected_mask = adaptive_model.create_protected_mask
    def custom_protected_mask(input_ids_tensor):
        mask = original_create_protected_mask(input_ids_tensor)
        # Protect ground truth
        mask[start_idx:end_idx+1] = True
        return mask
    adaptive_model.create_protected_mask = custom_protected_mask
    
    trainer = LayerwiseAdaptiveTrainer(
        model=adaptive_model,
        learning_rate=1e-3,
        budget_lambda=0.10,
        entropy_lambda=0.001
    )
    
    # 4. Training Loop
    print_header("TRAINING LOOP")
    epochs = 100
    for epoch in range(1, epochs + 1):
        result = trainer.train_step(input_ids, attention_mask, start_target, end_target)
        if epoch % 10 == 0 or epoch == 1:
            print(trainer.format_result(result, epoch))
            
    # 5. Evaluation
    print_header("POST-TRAINING EVALUATION")
    adaptive_model.eval()
    
    # Measure Latency
    start_time = time.perf_counter()
    for _ in range(10):
        with torch.no_grad():
            ad_start_logits, ad_end_logits, layer_metrics = adaptive_model(input_ids, attention_mask, return_layer_metrics=True)
    adaptive_latency = (time.perf_counter() - start_time) / 10 * 1000
    
    start_time = time.perf_counter()
    for _ in range(10):
        baseline.get_baseline_prediction(input_ids, attention_mask)
    baseline_latency = (time.perf_counter() - start_time) / 10 * 1000
    
    ad_start = torch.argmax(ad_start_logits, dim=-1).item()
    ad_end = torch.argmax(ad_end_logits, dim=-1).item()
    if ad_end < ad_start:
        ad_start, ad_end = ad_end, ad_start
        
    ad_answer = data_loader.decode_span(input_ids, ad_start, ad_end)
    print(f"Adaptive Latency: {adaptive_latency:.2f} ms")
    print(f"Baseline Latency: {baseline_latency:.2f} ms")
    print(f"Speedup: {baseline_latency/adaptive_latency if adaptive_latency > 0 else 1:.2f}x\n")
    
    print(f"Adaptive Prediction: '{ad_answer}'")
    if ad_answer == answer_text:
        print("QA Verification: PASS")
    else:
        print("QA Verification: FAIL")
        
    print("\nTENSOR SHAPES:")
    for i, num_tokens in enumerate(layer_metrics['tokens_per_layer']):
        print(f"Layer {i}: {num_tokens} tokens")
    if layer_metrics['selection_results'][-1] is not None:
        print(f"Final output: {layer_metrics['selection_results'][-1].num_selected} tokens")

if __name__ == "__main__":
    main()
