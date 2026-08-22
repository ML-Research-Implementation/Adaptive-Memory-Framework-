import time
import torch
from config import MODEL_NAME, DEVICE
from src.data import QADataLoader, find_answer_span
from src.models_adaptive import AdaptiveDistilBertQA, AdaptiveQAInference
from src.baseline import BaselineQAModel, compute_baseline_metrics
from src.utils import print_header, initialize_reproducibility

def main():
    print_header("PHASE 1: LAYER-WISE COMPACTION PROTOTYPE")
    initialize_reproducibility()
    
    # 1. Prepare Data
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
    
    print(f"Original Sequence Length: {input_ids.shape[1]}")
    
    # 2. Baseline Model Latency
    print_header("BASELINE MEASUREMENT")
    baseline = BaselineQAModel(freeze_parameters=True)
    
    # Warmup
    for _ in range(3):
        baseline.get_baseline_prediction(input_ids, attention_mask)
        
    start_time = time.perf_counter()
    for _ in range(10):
        baseline_start, baseline_end, _, info_dict = baseline.get_baseline_prediction(input_ids, attention_mask)
    baseline_latency = (time.perf_counter() - start_time) / 10 * 1000  # ms
    
    baseline_answer = data_loader.decode_span(input_ids, baseline_start, baseline_end)
    print(f"Baseline Latency: {baseline_latency:.2f} ms")
    print(f"Baseline Prediction: '{baseline_answer}'")
    
    # 3. Layer-wise Adaptive Model
    print_header("LAYER-WISE ADAPTIVE MODEL")
    
    # We use retention_ratio = 0.75 for physical compaction at each layer
    adaptive_model = AdaptiveDistilBertQA(
        model_name=MODEL_NAME,
        device=DEVICE,
        freeze_transformer=True,
        retention_ratio=0.75
    )
    adaptive_model.eval()
    
    # Warmup
    for _ in range(3):
        adaptive_model(input_ids, attention_mask)
        
    start_time = time.perf_counter()
    for _ in range(10):
        with torch.no_grad():
            adaptive_start_logits, adaptive_end_logits, layer_metrics = adaptive_model(input_ids, attention_mask, return_layer_metrics=True)
    adaptive_latency = (time.perf_counter() - start_time) / 10 * 1000  # ms
    
    ad_start_idx = torch.argmax(adaptive_start_logits, dim=-1).item()
    ad_end_idx = torch.argmax(adaptive_end_logits, dim=-1).item()
    
    # Ensure valid span
    if ad_end_idx < ad_start_idx:
        ad_start_idx, ad_end_idx = ad_end_idx, ad_start_idx
        
    adaptive_answer = data_loader.decode_span(input_ids, ad_start_idx, ad_end_idx)
    
    print(f"Adaptive Latency: {adaptive_latency:.2f} ms")
    speedup = baseline_latency / adaptive_latency if adaptive_latency > 0 else 1.0
    print(f"Speedup: {speedup:.2f}x")
    
    print(f"\nAdaptive Prediction: '{adaptive_answer}'")
    if adaptive_answer == answer_text:
        print("QA Verification: PASS (Model still predicts correct span even with untrained random scorers)")
    else:
        print("QA Verification: FAIL (Untrained random scorers dropped important tokens. Needs training!)")
    
    # 4. Verify Tensor Shapes
    print_header("TENSOR SHAPE VERIFICATION")
    print(f"Layer 0: {input_ids.shape[1]} tokens (Input)")
    for i, num_tokens in enumerate(layer_metrics['tokens_per_layer']):
        print(f"Layer {i+1}: {num_tokens} tokens")
        
    # Let's count final tokens (after Layer 6 selection)
    if layer_metrics['selection_results'][-1] is not None:
        final_tokens = layer_metrics['selection_results'][-1].num_selected
        print(f"Output: {final_tokens} tokens")

if __name__ == "__main__":
    main()
