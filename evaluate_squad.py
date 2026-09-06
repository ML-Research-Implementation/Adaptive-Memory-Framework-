import os
import gc
import time
import torch
import collections
import string
import re
from tqdm import tqdm
from transformers import AutoTokenizer
from config import MODEL_NAME, DEVICE
from src.squad_data import get_squad_dataloaders
from src.baseline import BaselineQAModel
from src.models_adaptive import AdaptiveDistilBertQA
from src.utils import print_header
import json


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


def evaluate_model(model, dataloader, dataset_features, raw_val_data, tokenizer, is_baseline=False, threshold_bias=0.0):
    """
    Evaluates a model (Baseline or AMMR) on the SQuAD validation set.
    """
    if hasattr(model, 'eval'):
        model.eval()
    elif hasattr(model, 'qa_model'):
        model.qa_model.eval()
        
    all_start_logits = []
    all_end_logits = []
    
    total_latency = 0
    num_batches = 0
    
    total_retained_tokens = 0
    total_original_tokens = 0
    
    layer_span_survival = [0] * 6
    layer_span_total = [0] * 6
    
    all_retention_scores = []
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(DEVICE)
        
    print(f"Running inference...")
    for batch in tqdm(dataloader, leave=False):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        
        start_time = time.perf_counter()
        with torch.no_grad():
            if is_baseline:
                if hasattr(model, 'qa_model'):
                    outputs = model.qa_model(input_ids, attention_mask)
                else:
                    outputs = model(input_ids, attention_mask)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits
                layer_metrics = None
            else:
                # AMMR forward pass
                start_logits, end_logits, layer_metrics = model(
                    input_ids, attention_mask, return_layer_metrics=True, training=False, threshold_bias=threshold_bias
                )
        end_time = time.perf_counter()
        
        total_latency += (end_time - start_time)
        num_batches += 1
        
        all_start_logits.append(start_logits.cpu())
        all_end_logits.append(end_logits.cpu())
        
        # Track token retention and answer span survival
        if layer_metrics and layer_metrics.get('selection_results'):
            batch_tokens = 0
            batch_original = 0
            
            start_pos = batch.get('start_positions', torch.zeros_like(input_ids[:, 0]))
            end_pos = batch.get('end_positions', torch.zeros_like(input_ids[:, 0]))
            
            for l_idx, res in enumerate(layer_metrics['selection_results']):
                if res is not None:
                    batch_tokens += res.num_selected
                    batch_original += res.num_original
                    
                    for b_idx in range(input_ids.size(0)):
                        s_p = start_pos[b_idx].item()
                        e_p = end_pos[b_idx].item()
                        if s_p == 0 and e_p == 0:
                            continue
                            
                        sel_idx = res.selected_indices[b_idx]
                        span_indices = torch.arange(s_p, e_p + 1, device=DEVICE)
                        survived = torch.all(torch.isin(span_indices, sel_idx)).item()
                        
                        layer_span_survival[l_idx] += survived
                        layer_span_total[l_idx] += 1
            
            
            total_retained_tokens += batch_tokens
            total_original_tokens += batch_original
            
            # Collect scores for calibration if threshold_bias == 0 (baseline pass)
            if threshold_bias == 0.0 and layer_metrics and layer_metrics.get('selection_results'):
                for res in layer_metrics['selection_results']:
                    if res is not None:
                        all_retention_scores.append(res.retention_scores.detach().cpu())
        
    avg_latency = (total_latency / max(1, num_batches)) * 1000  # ms
    
    all_start_logits = torch.cat(all_start_logits, dim=0)
    all_end_logits = torch.cat(all_end_logits, dim=0)
    
    exact_scores = []
    f1_scores = []
    
    print("Computing metrics...")
    for i, feature in enumerate(dataset_features):
        example_id = feature['example_id']
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
    attention_cost_ratio = (avg_retention_ratio / 100.0) ** 2 * 100.0
    compute_reduction = 100.0 - attention_cost_ratio
    
    peak_memory = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2) if torch.cuda.is_available() else 0.0
    
    span_survival_rates = []
    for l_idx in range(6):
        if layer_span_total[l_idx] > 0:
            span_survival_rates.append(layer_span_survival[l_idx] / layer_span_total[l_idx] * 100.0)
        else:
            span_survival_rates.append(100.0)
            
    # Combine retention scores if collected
    all_scores = torch.cat(all_retention_scores).view(-1) if all_retention_scores else None
            
    return avg_em, avg_f1, avg_latency, avg_retention_ratio, attention_cost_ratio, compute_reduction, peak_memory, span_survival_rates, all_scores

def find_best_threshold(scores: torch.Tensor, target_retention_ratio: float) -> float:
    """
    Finds the threshold bias that achieves the target retention ratio using percentiles.
    z = (logits + bias) > 0  => logits > -bias
    """
    # Sort scores or use torch.quantile
    # target_retention_ratio is the top fraction we want to keep
    # e.g. 0.70 means we want top 70%. We find the 30th percentile.
    q = 1.0 - target_retention_ratio
    q = max(0.0, min(1.0, q))
    threshold = torch.quantile(scores.float(), q).item()
    return -threshold


def main():
    print_header("SQuAD EVALUATION: BASELINE VS ADAPTIVE DISTILBERT")
    
    # 1. Load validation subset for fast evaluation
    train_dl, val_dl, train_data, val_data, val_features = get_squad_dataloaders(
        batch_size=16, 
        max_train_samples=10, 
        max_val_samples=200
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 2. Evaluate Baseline
    print_header("1. BASELINE EVALUATION")
    baseline = BaselineQAModel(freeze_parameters=True)
    b_em, b_f1, b_lat, *baseline_stats = evaluate_model(
        baseline, val_dl, val_features, val_data, tokenizer, is_baseline=True
    )
    
    del baseline
    gc.collect()
    
    # 3. Load Trained Adaptive Model
    print_header("2. ADAPTIVE MODEL SETUP")
    ammr = AdaptiveDistilBertQA(model_name=MODEL_NAME, device=DEVICE).to(DEVICE)
    
    checkpoint_candidates = [
        "models/layerwise_scorers_phase4.pt",
        "squad_phase4_checkpoint.pt"
    ]
    
    loaded = False
    for ckpt in checkpoint_candidates:
        if os.path.exists(ckpt):
            try:
                state_dict = torch.load(ckpt, map_location=DEVICE)
                if hasattr(ammr, 'get_retention_scorers'):
                    ammr.get_retention_scorers().load_state_dict(state_dict)
                else:
                    ammr.load_state_dict(state_dict, strict=False)
                print(f"Successfully loaded trained scorers from: {ckpt}")
                loaded = True
                break
            except Exception as e:
                print(f"Notice: Failed loading from {ckpt} ({e}), checking next...")
                
    if not loaded:
        print("WARNING: Checkpoint not found, evaluating with initial scorer weights.")
        
    print("\nStarting Calibration Pass (bias=0.0)...")
    _, _, _, baseline_ret, _, _, _, _, all_scores = evaluate_model(
        ammr, val_dl, val_features, val_data, tokenizer, is_baseline=False, threshold_bias=0.0
    )
    
    target_retentions = [0.80, 0.70, 0.60, 0.50]
    biases = [0.0]
    calibration_data = {}
    
    if all_scores is not None and len(all_scores) > 0:
        for tgt in target_retentions:
            bias = find_best_threshold(all_scores, tgt)
            biases.append(bias)
            calibration_data[f"target_{tgt}"] = bias
            print(f"Calibrated bias for {tgt*100}% retention: {bias:.4f}")
            
        with open("calibration.json", "w") as f:
            json.dump(calibration_data, f, indent=4)
        print("Saved calibration results to calibration.json")
    else:
        biases = [0.0, -1.0, -2.0, -4.0]
        
    results = []
    
    print("\nStarting Adaptive Evaluation Sweep...")
    for bias in biases:
        print(f"\n--> Evaluating AMMR at Bias: {bias:.4f}")
        (
            a_em, a_f1, a_lat, a_ret_ratio, a_attn_cost,
            a_comp_red, a_peak_mem, a_spans, _
        ) = evaluate_model(
            ammr, val_dl, val_features, val_data, tokenizer, is_baseline=False, threshold_bias=bias
        )
        results.append({
            'bias': bias,
            'em': a_em,
            'f1': a_f1,
            'retention': a_ret_ratio,
            'attn_cost': a_attn_cost,
            'latency': a_lat
        })
        
    # 5. Format and print final comparison table
    print_header("EVALUATION RESULTS: ACCURACY-EFFICIENCY TRADE-OFF")
    print(f"{'Model / Bias':<16} | {'Exact Match':<12} | {'F1 Score':<12} | {'Tokens Retained':<17} | {'Attn Cost (est)':<17} | {'Latency/batch':<15}")
    print("-" * 100)
    print(f"{'Baseline':<16} | {b_em:<12.2f} | {b_f1:<12.2f} | {'100.0%':<17} | {'100.0%':<17} | {b_lat:<10.2f} ms")
    
    for res in results:
        label = f"AMMR (b={res['bias']:.1f})"
        ret_str = f"{res['retention']:.1f}%"
        attn_str = f"{res['attn_cost']:.1f}%"
        print(f"{label:<16} | {res['em']:<12.2f} | {res['f1']:<12.2f} | {ret_str:<17} | {attn_str:<17} | {res['latency']:<10.2f} ms")


if __name__ == "__main__":
    main()