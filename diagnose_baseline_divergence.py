import os
import torch
import numpy as np
from transformers import AutoTokenizer
from config import MODEL_NAME, DEVICE
from src.squad_data import get_squad_dataloaders
from src.baseline import BaselineQAModel
from src.models_adaptive import AdaptiveDistilBertQA
import evaluate_squad

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/content/AMMR_GITHUB/squad_final_checkpoint.pt")
    parser.add_argument("--num-examples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()

def compare_tensors(t1, t2, name):
    if t1 is None and t2 is None:
        return "Both None"
    if t1 is None or t2 is None:
        return "One is None"
    if t1.shape != t2.shape:
        return f"Shape mismatch: {t1.shape} vs {t2.shape}"
    
    # Compute diffs
    diff = (t1 - t2).float().abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    t1_abs = t1.float().abs()
    t1_max = t1_abs.max().item()
    rel_diff = max_diff / max(t1_max, 1e-8)
    
    return f"max_diff={max_diff:.3e}, mean_diff={mean_diff:.3e}, rel_diff={rel_diff:.3e}"

def extract_predictions(start_logits, end_logits):
    # Simply top 1
    start_preds = torch.argmax(start_logits, dim=-1)
    end_preds = torch.argmax(end_logits, dim=-1)
    return start_preds, end_preds

def main():
    args = parse_args()
    
    if not os.path.exists(args.checkpoint):
        if os.path.exists("dummy.pt"):
            args.checkpoint = "dummy.pt"
            
    print("Loading dataloaders...")
    _, val_dl, _, val_data, val_features = get_squad_dataloaders(
        batch_size=args.batch_size,
        max_train_samples=1,
        max_val_samples=args.num_examples,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Fetch exactly one batch for pure deterministic comparison
    batch = next(iter(val_dl))
    input_ids = batch['input_ids'].to(DEVICE)
    attention_mask = batch['attention_mask'].to(DEVICE)
    
    print("\n--- CONDITION A: Baseline ---")
    baseline = BaselineQAModel(freeze_parameters=True).to(DEVICE)
    if os.path.exists(args.checkpoint):
        # Load the AMMR checkpoint into the baseline for an apples-to-apples comparison
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state_dict = ckpt["model_state_dict"]
        
        # Extract just the distilbert and qa_outputs parts
        distilbert_state = {}
        qa_outputs_state = {}
        for k, v in state_dict.items():
            if k.startswith("distilbert."):
                distilbert_state[k.replace("distilbert.", "")] = v
            elif k.startswith("qa_outputs."):
                qa_outputs_state[k.replace("qa_outputs.", "")] = v
                
        baseline.model.distilbert.load_state_dict(distilbert_state, strict=False)
        baseline.model.qa_outputs.load_state_dict(qa_outputs_state, strict=False)
        
    baseline.eval()
    with torch.no_grad():
        out_b = baseline.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        baseline_start = out_b.start_logits
        baseline_end = out_b.end_logits
        baseline_hidden = out_b.hidden_states
        # hidden_states[0] is embedding, [1] is layer1... [6] is layer6
    del baseline
    
    def run_adaptive(flag_kwargs, name):
        print(f"\n--- CONDITION: {name} ---")
        model = AdaptiveDistilBertQA(model_name=MODEL_NAME, device=DEVICE).to(DEVICE)
        if os.path.exists(args.checkpoint):
            evaluate_squad.load_ammr_checkpoint(model, args.checkpoint)
        model.eval()
        
        with torch.no_grad():
            start_logits, end_logits, diagnostics = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                diagnostic_return_layer_states=True,
                **flag_kwargs
            )
            
        print(f"Comparing start_logits: {compare_tensors(baseline_start, start_logits, 'start_logits')}")
        print(f"Comparing end_logits: {compare_tensors(baseline_end, end_logits, 'end_logits')}")
        
        base_s, base_e = extract_predictions(baseline_start, baseline_end)
        s, e = extract_predictions(start_logits, end_logits)
        spans_match = (base_s == s).all() and (base_e == e).all()
        print(f"Predicted spans exact match? {spans_match}")
        
        emb = diagnostics["embedding_output"]
        print(f"Comparing embeddings: {compare_tensors(baseline_hidden[0], emb, 'embeddings')}")
        
        layer_h = diagnostics["layer_hidden_states"]
        layer_m = diagnostics["layer_attention_masks"]
        
        for i in range(6):
            print(f"  Layer {i+1} hidden_states: {compare_tensors(baseline_hidden[i+1], layer_h[i], 'hidden')}")
            # Attention masks for distilbert in baseline aren't returned directly, but they shouldn't change
            
        print(f"Tokens retained at last layer: {diagnostics['tokens_per_layer'][-1]}")
        del model
        
    run_adaptive({"diagnostic_force_all_retain": True}, "B. Adaptive (force_all_retain=True)")
    run_adaptive({"threshold_bias": 1.0}, "C. Adaptive (bias=+1.0)")
    run_adaptive({"diagnostic_no_compaction": True, "diagnostic_force_all_retain": True}, "D. Adaptive (no_compaction, force_all_retain)")

if __name__ == "__main__":
    main()
