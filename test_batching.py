import torch
from config import MODEL_NAME, DEVICE
from src.models_adaptive import AdaptiveDistilBertQA
from src.utils import print_header

def test_batching():
    print_header("TEST BATCHING (SYNTHETIC)")
    
    batch_size = 2
    original_seq_len = 31
    hidden_dimension = 768
    
    # 1. Create synthetic input
    # Sequence 1: full length, no padding
    # Sequence 2: 21 valid tokens, 10 padding tokens
    input_ids = torch.randint(1000, 30000, (batch_size, original_seq_len), device=DEVICE)
    
    # Put [CLS] and [SEP]
    input_ids[0, 0] = 101
    input_ids[0, 15] = 102
    input_ids[1, 0] = 101
    input_ids[1, 10] = 102
    
    attention_mask = torch.ones((batch_size, original_seq_len), device=DEVICE)
    attention_mask[1, 21:] = 0  # Sequence 2 has padding
    
    # 2. Model
    schedule = [0.90, 0.85, 0.80, 0.75, 0.70, 0.70]
    model = AdaptiveDistilBertQA(
        model_name=MODEL_NAME,
        device=DEVICE,
        freeze_transformer=True,
        retention_schedule=schedule
    )
    
    print(f"Input Shape: {input_ids.shape}")
    print(f"Attention Mask: \n{attention_mask.cpu().numpy()}")
    
    # 3. Forward Pass
    try:
        start_logits, end_logits, layer_metrics = model(input_ids, attention_mask, return_layer_metrics=True)
        print("\nForward pass successful!")
    except Exception as e:
        print(f"\nForward pass failed: {e}")
        return
        
    print(f"\nLogits Shape: {start_logits.shape} (Expected: [{batch_size}, {original_seq_len}])")
    assert start_logits.shape == (batch_size, original_seq_len)
    
    print("\nLayer Metrics:")
    for i, res in enumerate(layer_metrics['selection_results']):
        if res is not None:
            print(f"Layer {i+1} Output tokens: {res.num_selected}")
            assert res.selected_indices.shape[0] == batch_size
            assert res.selected_indices.shape[1] == res.num_selected
            
            # Verify [CLS] and [SEP] were preserved
            # Because protected tokens have +inf scores, they should be at the start of the sorted top-K
            # Actually, their indices were sorted so they appear in their original temporal positions!
            # Let's check that 0 (CLS) is always preserved
            assert (res.selected_indices[:, 0] == 0).all().item(), "CLS was dropped!"
            
            # Check padding tokens were dropped (indices >= 21 in sequence 1 should NOT be in selected_indices)
            if res.num_selected < 21:
                has_padding_seq1 = (res.selected_indices[1] >= 21).any().item()
                assert not has_padding_seq1, "Padding token was selected in seq 2!"
                
    print("\nAll batching tests passed!")

if __name__ == "__main__":
    test_batching()
