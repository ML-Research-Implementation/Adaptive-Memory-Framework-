import time
import torch
import collections
import string
import re
from tqdm import tqdm
from config import MODEL_NAME, DEVICE
from src.squad_data import get_squad_dataloaders
from src.models_adaptive import AdaptiveQAInference
from src.baseline import BaselineQAModel
from src.utils import print_header

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def evaluate_model(model_fn, dataloader, dataset_features, raw_val_data, tokenizer, is_baseline=False):
    if hasattr(model_fn, 'eval'):
        model_fn.eval()
    all_start_logits = []
    all_end_logits = []
    
    total_latency = 0
    num_batches = 0
    
    total_retained_tokens = 0
    total_original_tokens = 0
    
    print(f"Running inference...")
    for batch in tqdm(dataloader, leave=False):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        
        start_time = time.perf_counter()
        with torch.no_grad():
            if is_baseline:
                outputs = model_fn.qa_model(input_ids, attention_mask)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits
                layer_metrics = None
            else:
                start_logits, end_logits, layer_metrics = model_fn(input_ids, attention_mask, return_layer_metrics=True)
        end_time = time.perf_counter()
        
        total_latency += (end_time - start_time)
        num_batches += 1
        
        all_start_logits.append(start_logits.cpu())
        all_end_logits.append(end_logits.cpu())
        
        # Track token retention
        if layer_metrics and layer_metrics.get('selection_results'):
            batch_tokens = 0
            batch_original = 0
            for res in layer_metrics['selection_results']:
                if res is not None:
                    batch_tokens += res.num_selected
                    batch_original += res.num_original
            
            total_retained_tokens += batch_tokens
            total_original_tokens += batch_original
        
    avg_latency = (total_latency / num_batches) * 1000  # ms
    
    all_start_logits = torch.cat(all_start_logits, dim=0)
    all_end_logits = torch.cat(all_end_logits, dim=0)
    
    exact_scores = []
    f1_scores = []
    
    # Map predictions to original texts
    print("Computing metrics...")
    for i, feature in enumerate(dataset_features):
        example_id = feature['example_id']
        # Find original example
        # Since we might have truncated max_val_samples, we just search or map
        # A simpler way is to find it in raw_val_data
        example = None
        for ex in raw_val_data:
            if ex['id'] == example_id:
                example = ex
                break
        if not example:
            continue
            
        gold_answers = [ans for ans in example['answers']['text']]
        if not gold_answers:
            continue
            
        start_logit = all_start_logits[i]
        end_logit = all_end_logits[i]
        
        start_idx = torch.argmax(start_logit).item()
        end_idx = torch.argmax(end_logit).item()
        
        if end_idx < start_idx:
            pred_answer = ""
        else:
            pred_answer = tokenizer.decode(feature['input_ids'][start_idx:end_idx+1], skip_special_tokens=True)
            
        exact_scores.append(max(compute_exact(a, pred_answer) for a in gold_answers))
        f1_scores.append(max(compute_f1(a, pred_answer) for a in gold_answers))
        
    if len(exact_scores) == 0:
        print("WARNING: No exact scores were computed!")
        avg_em = 0.0
        avg_f1 = 0.0
    else:
        avg_em = sum(exact_scores) / len(exact_scores) * 100
        avg_f1 = sum(f1_scores) / len(f1_scores) * 100
    
    avg_retention_ratio = (total_retained_tokens / total_original_tokens * 100) if total_original_tokens > 0 else 100.0
    
    # Estimate Attention cost (Attention is O(N^2)). 
    # Original cost per layer is proportional to N^2. Total original cost = 6 * N^2.
    # New cost = sum((r_i * N)^2).
    # Since we use uniform retention ratio r across 6 layers, cost = 6 * (r * N)^2 = r^2 * (Original Cost)
    # So relative cost is exactly avg_retention_ratio^2
    attention_cost_ratio = (avg_retention_ratio / 100.0) ** 2 * 100.0
    
    return avg_em, avg_f1, avg_latency, avg_retention_ratio, attention_cost_ratio


def main():
    print_header("PHASE 3: SQuAD EVALUATION BASELINE")
    
    # Load just 100 validation examples for fast evaluation in the prototype
    train_dl, val_dl, train_data, val_data, val_features = get_squad_dataloaders(
        batch_size=16, 
        max_train_samples=10, 
        max_val_samples=100
    )
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print_header("EVALUATION")
    
    # Baseline
    baseline = BaselineQAModel(freeze_parameters=True)
    b_em, b_f1, b_lat, _, _ = evaluate_model(
        baseline, val_dl, val_features, val_data, tokenizer, is_baseline=True
    )
    
    # Free memory
    del baseline
    import gc
    gc.collect()
    
    # AMMR
    from src.models_adaptive import AdaptiveDistilBertQA
    from src.utils import load_checkpoint
    import os
    
    ammr = AdaptiveDistilBertQA(model_name=MODEL_NAME, device=DEVICE)
    if os.path.exists("squad_phase3_checkpoint.pt"):
        load_checkpoint(ammr, optimizer=None, checkpoint_path="squad_phase3_checkpoint.pt")
        print("Loaded trained AMMR model.")
    else:
        print("WARNING: Checkpoint not found, evaluating untrained AMMR.")
        
    ratios = [0.90, 0.80, 0.70, 0.60, 0.50]
    results = []
    
    for r in ratios:
        print(f"\nEvaluating AMMR at Ratio: {r}")
        # Apply uniform ratio to all 6 layers
        ammr.retention_schedule = [r] * 6
        a_em, a_f1, a_lat, a_ret_ratio, a_attn_cost = evaluate_model(
            ammr, val_dl, val_features, val_data, tokenizer, is_baseline=False
        )
        results.append({
            'ratio': r,
            'em': a_em,
            'f1': a_f1,
            'retention': a_ret_ratio,
            'attn_cost': a_attn_cost,
            'latency': a_lat
        })
        
    # Print Table
    print_header("EVALUATION RESULTS: ACCURACY-EFFICIENCY TRADE-OFF")
    print(f"{'Model / Ratio':<15} | {'Exact Match':<12} | {'F1 Score':<12} | {'Tokens Retained':<17} | {'Attn Cost (est)':<17} | {'Latency/batch':<15}")
    print("-" * 95)
    print(f"{'Baseline':<15} | {b_em:<12.2f} | {b_f1:<12.2f} | {'100.0%':<17} | {'100.0%':<17} | {b_lat:<10.2f} ms")
    
    for res in results:
        label = f"AMMR (r={res['ratio']:.2f})"
        ret_str = f"{res['retention']:.1f}%"
        attn_str = f"{res['attn_cost']:.1f}%"
        print(f"{label:<15} | {res['em']:<12.2f} | {res['f1']:<12.2f} | {ret_str:<17} | {attn_str:<17} | {res['latency']:<10.2f} ms")
        
    
if __name__ == "__main__":
    main()
