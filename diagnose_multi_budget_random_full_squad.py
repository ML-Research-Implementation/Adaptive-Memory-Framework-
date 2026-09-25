import os
import sys
import argparse
import torch
import numpy as np
import time
import json
import csv
from transformers import AutoTokenizer
from config import MODEL_NAME, DEVICE
from src.squad_data import get_squad_dataloaders
from src.baseline import BaselineQAModel
from src.models_adaptive import AdaptiveDistilBertQA
import evaluate_squad

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/content/AMMR_GITHUB/squad_final_checkpoint.pt")
    parser.add_argument("--num-examples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--biases", nargs='+', type=float, default=[1.0, 0.5, 0.2, 0.0, -0.2, -0.4, -0.6])
    parser.add_argument("--seeds", nargs='+', type=int, default=[42, 123, 2026, 7, 19, 37, 101, 256, 512, 999])
    return parser.parse_args()

def custom_evaluate_model(model, dataloader, dataset_features, raw_val_data, tokenizer, is_baseline=False, threshold_bias=0.0, current_bias=None, current_seed=None):
    if hasattr(model, "eval"):
        model.eval()
    elif hasattr(model, "qa_model"):
        model.qa_model.eval()

    all_start_logits, all_end_logits = [], []
    total_latency = 0.0
    num_batches = 0
    total_retained_tokens = 0
    total_original_tokens = 0
    layer_span_survival = [0] * 6
    layer_span_total = [0] * 6
    all_retention_scores = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(DEVICE)

    start_eval_time = time.time()
    for i, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        start_time = time.perf_counter()
        with torch.no_grad():
            if is_baseline:
                outputs = model.qa_model(input_ids, attention_mask) if hasattr(model, "qa_model") else model(input_ids, attention_mask)
                start_logits, end_logits, layer_metrics = outputs.start_logits, outputs.end_logits, None
            else:
                start_logits, end_logits, layer_metrics = evaluate_squad.unpack_student_outputs(model(
                    input_ids, attention_mask, return_layer_metrics=True,
                    training=False, threshold_bias=threshold_bias
                ))
        total_latency += time.perf_counter() - start_time
        num_batches += 1
        all_start_logits.append(start_logits.cpu())
        all_end_logits.append(end_logits.cpu())

        if layer_metrics and layer_metrics.get("selection_results"):
            start_pos = batch.get("start_positions", torch.zeros_like(input_ids[:, 0]))
            end_pos = batch.get("end_positions", torch.zeros_like(input_ids[:, 0]))
            batch_tokens = batch_original = 0
            for layer_idx, result in enumerate(layer_metrics["selection_results"]):
                if result is None:
                    continue
                batch_tokens += result.num_selected
                batch_original += result.num_original
                for batch_idx in range(input_ids.size(0)):
                    start_idx = int(start_pos[batch_idx])
                    end_idx = int(end_pos[batch_idx])
                    if start_idx == 0 and end_idx == 0:
                        continue
                    span = torch.arange(start_idx, end_idx + 1, device=DEVICE)
                    survived = torch.all(torch.isin(span, result.selected_indices[batch_idx])).item()
                    layer_span_survival[layer_idx] += survived
                    layer_span_total[layer_idx] += 1
            total_retained_tokens += batch_tokens
            total_original_tokens += batch_original
            if threshold_bias == 0.0:
                all_retention_scores.extend(
                    result.retention_scores.detach().cpu() for result in layer_metrics["selection_results"] if result is not None
                )

        if num_batches % 10 == 0 or num_batches == len(dataloader):
            elapsed = time.time() - start_eval_time
            rate = num_batches / elapsed if elapsed > 0 else 0
            examples_processed = num_batches * input_ids.size(0)
            print(f"Bias: {current_bias} | Seed: {current_seed} | Batch {num_batches}/{len(dataloader)} | Ex: {examples_processed} | Elapsed: {elapsed:.1f}s | {rate:.1f} batch/s", flush=True)

    if not all_start_logits:
        raise RuntimeError("Evaluation produced no batches.")
    all_start_logits = torch.cat(all_start_logits)
    all_end_logits = torch.cat(all_end_logits)
    raw_by_id = {example["id"]: example for example in raw_val_data}
    exact_scores, f1_scores = [], []
    for index, feature in enumerate(dataset_features):
        if index >= len(all_start_logits):
            break
        example = raw_by_id.get(feature["example_id"])
        if example is None or not example["answers"]["text"]:
            continue
        start_idx = torch.argmax(all_start_logits[index]).item()
        end_idx = torch.argmax(all_end_logits[index]).item()
        prediction = "" if end_idx < start_idx else tokenizer.decode(
            feature["input_ids"][start_idx:end_idx + 1], skip_special_tokens=True
        )
        answers = example["answers"]["text"]
        exact_scores.append(max(evaluate_squad.compute_exact(answer, prediction) for answer in answers))
        f1_scores.append(max(evaluate_squad.compute_f1(answer, prediction) for answer in answers))

    retention = 100.0 * total_retained_tokens / total_original_tokens if total_original_tokens else 100.0
    spans = [100.0 if total == 0 else 100.0 * kept / total for kept, total in zip(layer_span_survival, layer_span_total)]
    scores = torch.cat([value.reshape(-1) for value in all_retention_scores]) if all_retention_scores else None
    return {
        "em": 100.0 * sum(exact_scores) / max(1, len(exact_scores)),
        "f1": 100.0 * sum(f1_scores) / max(1, len(f1_scores)),
        "latency_ms": 1000.0 * total_latency / max(1, num_batches),
        "retention": retention,
        "attention_cost": (retention / 100.0) ** 2 * 100.0,
        "compute_reduction": 100.0 - (retention / 100.0) ** 2 * 100.0,
        "answer_survival": sum(spans) / len(spans) if spans else 100.0,
        "span_survival_rates": spans,
        "all_scores": scores,
    }

def run_evaluation(model, dataloader, val_features, val_data, tokenizer, flag_name, bias, seed_val=None, target_counts_write=None):
    total_valid = 0
    total_selected = 0
    layer_valid = [0] * 6
    layer_selected = [0] * 6
    
    if flag_name is None:
        model._diagnostic_target_counts_write = {}
        model._diagnostic_target_counts_read = None
        model._diagnostic_target_counts_consumed = None
    elif flag_name == "diagnostic_random_seed":
        model._diagnostic_target_counts_write = dict(target_counts_write)
        model._diagnostic_target_counts_read = dict(target_counts_write)
        model._diagnostic_target_counts_consumed = set()
    else:
        model._diagnostic_target_counts_read = None
        model._diagnostic_target_counts_consumed = None
        
    forward_count = 0
    original_forward = model.forward
    
    def new_forward(*args, **kwargs):
        nonlocal total_valid, total_selected, forward_count
        kwargs["diagnostic_batch_id"] = forward_count
        if flag_name:
            if flag_name == "diagnostic_random_seed":
                kwargs[flag_name] = seed_val
            else:
                kwargs[flag_name] = True
        res = original_forward(*args, **kwargs)
        if len(res) == 3 and res[2] is not None:
            results = res[2].get("selection_results", [])
            if len(results) > 0:
                if results[0] is not None:
                    total_valid += int(results[0].actual_valid_counts.sum().item())
                if results[-1] is not None:
                    total_selected += int(results[-1].actual_retained_counts.sum().item())
                for i, result in enumerate(results):
                    if result is not None:
                        layer_valid[i] += int(result.actual_valid_counts.sum().item())
                        layer_selected[i] += int(result.actual_retained_counts.sum().item())
        forward_count += 1
        return res
        
    model.forward = new_forward
    res = custom_evaluate_model(model, dataloader, val_features, val_data, tokenizer, is_baseline=False, threshold_bias=bias, current_bias=bias, current_seed=seed_val if seed_val is not None else "AMMR")
    model.forward = original_forward
    
    res["total_valid"] = total_valid
    res["total_selected"] = total_selected
    res["effective_retention"] = (total_selected / total_valid) * 100.0 if total_valid > 0 else 100.0
    res["layer_valid"] = layer_valid
    res["layer_selected"] = layer_selected
    
    if flag_name is None:
        res["target_counts"] = dict(model._diagnostic_target_counts_write)
        res["target_count_records_captured"] = len(model._diagnostic_target_counts_write)
    elif flag_name == "diagnostic_random_seed":
        initial = len(model._diagnostic_target_counts_write)
        consumed = len(model._diagnostic_target_counts_consumed)
        
        captured_keys = set(model._diagnostic_target_counts_write.keys())
        consumed_keys = model._diagnostic_target_counts_consumed
        missing_keys = captured_keys - consumed_keys
        extra_keys = consumed_keys - captured_keys
        
        res["target_count_records_consumed"] = consumed
        res["target_count_records_remaining"] = initial - consumed
        res["missing_records"] = len(missing_keys)
        res["extra_records"] = len(extra_keys)
        
    return res

def main():
    args = parse_args()
    
    if args.max_examples is not None:
        args.num_examples = args.max_examples
        
    print("\n" + "!" * 80)
    print("WARNING: Optimized Multi-Budget Random Control Diagnostic")
    print(f"Biases: {args.biases}")
    print(f"Seeds: {args.seeds}")
    print(f"Num examples: {args.num_examples}")
    print("!" * 80 + "\n")
    
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        
    print(f"Validating checkpoint: {args.checkpoint}")
    import hashlib
    sha = hashlib.sha256()
    with open(args.checkpoint, 'rb') as f:
        while True:
            chunk = f.read(1024*1024)
            if not chunk:
                break
            sha.update(chunk)
    file_size = os.path.getsize(args.checkpoint)
    print(f"  Size: {file_size} bytes")
    print(f"  SHA256: {sha.hexdigest()}")
    
    ckpt_data = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(ckpt_data, dict):
        raise ValueError("Checkpoint is not a dictionary.")
        
    epoch = ckpt_data.get("epoch")
    step = ckpt_data.get("step")
    target_ratio = ckpt_data.get("target_ratio")
    lam = ckpt_data.get("lagrangian_multiplier")
    
    print(f"  Epoch: {epoch}")
    print(f"  Step: {step}")
    print(f"  Target ratio: {target_ratio}")
    print(f"  Lambda: {lam}")
    
    if step != 6365:
        raise ValueError("WRONG CHECKPOINT FOR EXACT-TARGET EXPERIMENT. Expected step 6365.")
            
    print("Loading datasets and model once...")
    _, val_dl, _, val_data, val_features = get_squad_dataloaders(
        batch_size=args.batch_size,
        max_train_samples=1,
        max_val_samples=args.num_examples,
    )
    
    if args.max_batches is not None:
        val_dl.dataset.data = val_dl.dataset.data[:args.max_batches * args.batch_size]
        val_features = val_features[:args.max_batches * args.batch_size]
        
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    model = AdaptiveDistilBertQA(model_name=MODEL_NAME, device=DEVICE).to(DEVICE)
    if os.path.exists(args.checkpoint):
        evaluate_squad.load_ammr_checkpoint(model, args.checkpoint)
    model.eval()
    
    json_filename = "multi_budget_random_control_resume.json"
    csv_filename = "multi_budget_random_control_full_squad.csv"
    
    all_results = []
    completed_biases = {}
    if os.path.exists(json_filename):
        with open(json_filename, "r") as f:
            all_results = json.load(f)
            for row in all_results:
                completed_biases[row["bias"]] = row

    for bias in args.biases:
        if bias in completed_biases:
            print(f"Skipping already completed bias {bias}")
            continue
            
        print(f"\n======================================")
        print(f"Evaluating AMMR Learned Selection (bias={bias})...")
        try:
            res_ammr = run_evaluation(model, val_dl, val_features, val_data, tokenizer, None, bias=bias)
            
            assert res_ammr.get("target_counts") is not None, "AMMR target_counts is None. Target-count capture failed."
            assert len(res_ammr["target_counts"]) > 0, f"Captured 0 target-count records!"
            
            random_results = []
            target_counts = res_ammr["target_counts"]
            captured = res_ammr.get('target_count_records_captured', 0)
            
            for seed in args.seeds:
                print(f"  Evaluating Random Matched Selection (seed={seed})...")
                res_rand = run_evaluation(model, val_dl, val_features, val_data, tokenizer, "diagnostic_random_seed", bias=bias, seed_val=seed, target_counts_write=target_counts)
                
                # Verify matched retention
                if res_rand['total_selected'] != res_ammr['total_selected']:
                    raise AssertionError(f"Count Mismatch! Random selected {res_rand['total_selected']} != AMMR selected {res_ammr['total_selected']}")
                if res_rand['missing_records'] > 0:
                    raise AssertionError(f"Mismatch! missing_records={res_rand['missing_records']}")
                if res_rand['extra_records'] > 0:
                    raise AssertionError(f"Mismatch! extra_records={res_rand['extra_records']}")
                    
                random_results.append(res_rand)
                
            rand_ems = [r['em'] for r in random_results]
            rand_f1s = [r['f1'] for r in random_results]
            rand_ans_survs = [r['answer_survival'] for r in random_results]
            
            mean_em = float(np.mean(rand_ems))
            std_em = float(np.std(rand_ems))
            mean_f1 = float(np.mean(rand_f1s))
            std_f1 = float(np.std(rand_f1s))
            mean_ans_surv = float(np.mean(rand_ans_survs))
            em_diff = float(res_ammr['em']) - mean_em
            f1_diff = float(res_ammr['f1']) - mean_f1
            
            print(f"\nINVARIANT CHECK for bias={bias}:")
            print(f"  target_count_records_captured: {captured}")
            print(f"  target_count_records_consumed: {random_results[0].get('target_count_records_consumed', 'N/A')}")
            print(f"  missing_records: 0")
            print(f"  extra_records: 0")
            print(f"  Mismatch count (tokens): 0")
            
            print(f"\nRESULTS for bias={bias}:")
            print(f"  AMMR EM={res_ammr['em']:.2f}, F1={res_ammr['f1']:.2f}, EffRet={res_ammr['effective_retention']:.2f}%")
            print(f"  Rand mean EM={mean_em:.2f}, std EM={std_em:.2f}")
            print(f"  Rand mean F1={mean_f1:.2f}, std F1={std_f1:.2f}")
            print(f"  EM diff={em_diff:.2f}, F1 diff={f1_diff:.2f}")
            
            row_dict = {
                "bias": bias,
                "AMMR_EM": res_ammr['em'],
                "AMMR_F1": res_ammr['f1'],
                "AMMR_eff_ret": res_ammr['effective_retention'],
                "AMMR_ans_surv": res_ammr['answer_survival'],
                "Rand_mean_EM": mean_em,
                "Rand_std_EM": std_em,
                "Rand_mean_F1": mean_f1,
                "Rand_std_F1": std_f1,
                "Rand_eff_ret": random_results[0]['effective_retention'],
                "Rand_ans_surv": mean_ans_surv,
                "EM_diff": em_diff,
                "F1_diff": f1_diff
            }
            for i, seed in enumerate(args.seeds):
                row_dict[f"Rand_EM_seed_{seed}"] = rand_ems[i]
                row_dict[f"Rand_F1_seed_{seed}"] = rand_f1s[i]
            
            all_results.append(row_dict)
            completed_biases[bias] = row_dict
            
            with open(json_filename, "w") as f:
                json.dump(all_results, f, indent=2)
            
        except Exception as e:
            consumed = 0
            if 'res_rand' in locals() and res_rand is not None:
                consumed = res_rand.get('target_count_records_consumed', 0)
            raise RuntimeError(
                f"Evaluation failed for bias={bias}.\n"
                f"Stage: AMMR/Random execution\n"
                f"Exception: {e}\n"
                f"Target records captured: {captured if 'captured' in locals() else 'Unknown'}\n"
                f"Target records consumed: {consumed}"
            ) from e

    produced = {round(float(row["bias"]), 1) for row in all_results}
    expected = {round(float(b), 1) for b in args.biases}
    assert produced == expected, f"Produced biases {produced} do not match expected {expected}"
    
    all_results.sort(key=lambda x: x["bias"], reverse=True)
    
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        headers = [
            "bias", "AMMR_EM", "AMMR_F1", "AMMR_eff_ret", "AMMR_ans_surv"
        ]
        for seed in args.seeds:
            headers.append(f"Rand_EM_seed_{seed}")
        for seed in args.seeds:
            headers.append(f"Rand_F1_seed_{seed}")
        headers.extend([
            "Rand_mean_EM", "Rand_std_EM", "Rand_mean_F1", "Rand_std_F1", 
            "Rand_eff_ret", "Rand_ans_surv", "EM_diff", "F1_diff"
        ])
        writer.writerow(headers)
        
        for row in all_results:
            row_vals = [
                f"{row['bias']:.1f}",
                f"{row['AMMR_EM']:.2f}",
                f"{row['AMMR_F1']:.2f}",
                f"{row['AMMR_eff_ret']:.2f}",
                f"{row['AMMR_ans_surv']:.2f}"
            ]
            for seed in args.seeds:
                row_vals.append(f"{row[f'Rand_EM_seed_{seed}']:.2f}")
            for seed in args.seeds:
                row_vals.append(f"{row[f'Rand_F1_seed_{seed}']:.2f}")
            row_vals.extend([
                f"{row['Rand_mean_EM']:.2f}",
                f"{row['Rand_std_EM']:.2f}",
                f"{row['Rand_mean_F1']:.2f}",
                f"{row['Rand_std_F1']:.2f}",
                f"{row['Rand_eff_ret']:.2f}",
                f"{row['Rand_ans_surv']:.2f}",
                f"{row['EM_diff']:.2f}",
                f"{row['F1_diff']:.2f}"
            ])
            writer.writerow(row_vals)
            
    print(f"\n======================================")
    print("COMPLETE MULTI-BUDGET SWEEP")
    print(f"rows: {len(all_results)}")
    print(f"biases: {[round(r['bias'], 1) for r in all_results]}")
    print("missing: []")
    print("duplicate_biases: []")
    print(f"CSV: {csv_filename}")
    
    # Compare against 500-example reference if running smoke test
    if args.num_examples == 500:
        ref_csv = "multi_budget_random_control.csv"
        if os.path.exists(ref_csv):
            print("\nComparing against 500-example reference CSV:")
            with open(ref_csv, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    b = round(float(row["bias"]), 1)
                    if b in completed_biases:
                        cur = completed_biases[b]
                        print(f"Bias {b}: Ref EM={row['AMMR_EM']} F1={row['AMMR_F1']} EffRet={row['AMMR_eff_ret']} | Cur EM={cur['AMMR_EM']:.2f} F1={cur['AMMR_F1']:.2f} EffRet={cur['AMMR_eff_ret']:.2f}")

if __name__ == "__main__":
    main()
