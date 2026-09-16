"""Diagnostic-only AMMR budget coefficient sweep.

This script never trains the production model, writes checkpoints, or changes
production source. It loads one real SQuAD validation batch and evaluates
temporary loss coefficients on cloned scorer/model state.

Example (Colab):
    python diagnose_budget_coefficient_sweep.py \
      --checkpoint /content/drive/MyDrive/ML-Research/AMMR_DIAGNOSTIC_EPOCH5_CHECKPOINT.pt \
      --batch-size 1
"""

import argparse
import copy
import os
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from config import DEVICE, MODEL_NAME
from evaluate_squad import load_ammr_checkpoint
from src.baseline import BaselineQAModel
from src.losses import calculate_distillation_loss, calculate_hidden_state_distillation_loss
from src.models_adaptive import AdaptiveDistilBertQA
from src.squad_data import get_squad_dataloaders

ALPHAS = (1, 10, 100, 1000, 10000)
EPSILON = 1e-8


def build_parser():
    parser = argparse.ArgumentParser(description="Diagnostic-only AMMR budget coefficient sweep")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--target-ratio", type=float, default=None)
    parser.add_argument("--num-examples", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Temporary cloned-model diagnostic step size")
    return parser


def _zero_grads(model):
    for parameter in model.retention_scorers.parameters():
        parameter.grad = None


def _layer_grad_stats(model) -> List[Dict[str, float]]:
    rows = []
    for layer, scorer in enumerate(model.retention_scorers):
        grads = [p.grad.detach().float().reshape(-1) for p in scorer.parameters() if p.grad is not None]
        if grads:
            values = torch.cat(grads)
            norm = float(values.norm().item())
            mean = float(values.mean().item())
            minimum = float(values.min().item())
            maximum = float(values.max().item())
        else:
            norm = mean = minimum = maximum = 0.0
        rows.append({
            "layer": layer,
            "norm": norm,
            "mean": mean,
            "min": minimum,
            "max": maximum,
        })
    return rows


def _score_probability_and_retention(model, batch, target_ratio):
    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    with torch.no_grad():
        _, _, metrics = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_layer_metrics=True,
            training=True,
            minimum_retention_ratio=target_ratio,
            answer_span_mask=None,
        )
        results = [result for result in metrics["selection_results"] if result is not None]
        score_values = torch.cat([result.retention_scores.detach().float().reshape(-1) for result in results])
        probability_values = torch.cat([result.retention_probs.detach().float().reshape(-1) for result in results])
        soft_retention = torch.stack([result.soft_retention_ratio.detach().float() for result in results]).mean()
    return {
        "score_mean": float(score_values.mean().item()),
        "probability_mean": float(probability_values.mean().item()),
        "soft_retention": float(soft_retention.item()),
    }


def _build_production_terms(model, teacher, batch, target_ratio):
    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    start_target = batch["start_positions"].to(DEVICE)
    end_target = batch["end_positions"].to(DEVICE)

    with torch.no_grad():
        teacher_outputs = teacher.model(input_ids, attention_mask, output_hidden_states=True)

    student_start, student_end, metrics = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_layer_metrics=True,
        training=True,
        minimum_retention_ratio=target_ratio,
        answer_span_mask=None,
    )
    qa_loss = (F.cross_entropy(student_start, start_target) + F.cross_entropy(student_end, end_target)) / 2.0
    kd_loss = calculate_distillation_loss(
        student_start, student_end,
        teacher_outputs.start_logits.detach(), teacher_outputs.end_logits.detach(),
        temperature=2.0,
    )
    hidden_kd_loss = torch.zeros((), device=DEVICE)
    results = []
    for layer_idx, result in enumerate(metrics["selection_results"]):
        if result is None:
            continue
        results.append(result)
        hidden_kd_loss = hidden_kd_loss + calculate_hidden_state_distillation_loss(
            student_hidden=metrics["hidden_states"][layer_idx],
            teacher_hidden=teacher_outputs.hidden_states[layer_idx + 1].detach(),
            selected_indices=result.selected_indices,
            attention_mask=result.new_attention_mask,
        )
    soft_retention = torch.stack([result.soft_retention_ratio for result in results]).mean()
    return {
        "qa": qa_loss,
        "kd": kd_loss,
        "hidden": hidden_kd_loss,
        "soft_retention": soft_retention,
        "results": results,
    }


def _global_token_weighted_retention(results):
    numerator = torch.zeros((), device=DEVICE)
    denominator = torch.zeros((), device=DEVICE)
    for result in results:
        valid = result.actual_valid_counts.detach().float().sum()
        probabilities = result.retention_probs.float()
        # Training probabilities are full-sequence tensors; valid counts are
        # reconstructed from the production attention-mask path by retaining
        # only the valid prefix/count per example.
        for row, count in zip(probabilities, result.actual_valid_counts.detach().long()):
            count = int(count.item())
            numerator = numerator + row[:count].sum()
            denominator = denominator + float(count)
    return numerator / denominator.clamp_min(1.0)


def _budget_from_retention(retention, target_ratio, lambda_value):
    violation = torch.relu(torch.as_tensor(target_ratio, device=retention.device) - retention)
    raw_lambda_term = lambda_value * violation
    clamped = torch.clamp(raw_lambda_term, min=0.0, max=10.0)
    return violation, raw_lambda_term, clamped


def _run_gradient_case(model, terms, target_ratio, lambda_value, alpha, retention_key="base"):
    _zero_grads(model)
    retention = terms["soft_retention"] if retention_key == "base" else terms["retention_c"]
    violation, raw_lambda_term, clamped = _budget_from_retention(retention, target_ratio, lambda_value)
    budget_loss = alpha * clamped
    budget_loss.backward(retain_graph=True)
    budget_rows = _layer_grad_stats(model)
    budget_norms = [row["norm"] for row in budget_rows]
    _zero_grads(model)
    combined = terms["qa"] + terms["kd"] + terms["hidden"] + budget_loss.detach() * 0.0
    # Rebuild the budget term so the combined graph remains connected.
    combined = terms["qa"] + terms["kd"] + terms["hidden"] + alpha * clamped
    combined.backward()
    combined_rows = _layer_grad_stats(model)
    _zero_grads(model)
    return {
        "retention": float(retention.detach().item()),
        "violation": float(violation.detach().item()),
        "lambda_violation": float(raw_lambda_term.detach().item()),
        "clamped_budget": float(clamped.detach().item()),
        "scaled_budget": float(budget_loss.detach().item()),
        "budget_rows": budget_rows,
        "combined_rows": combined_rows,
        "budget_norms": budget_norms,
        "combined_norms": [row["norm"] for row in combined_rows],
    }


def _clone_step(model, teacher, batch, target_ratio, lambda_value, alpha):
    clone = copy.deepcopy(model).to(DEVICE)
    clone.train()
    optimizer = AdamW(clone.retention_scorers.parameters(), lr=1e-4)
    before = _score_probability_and_retention(clone, batch, target_ratio)
    terms = _build_production_terms(clone, teacher, batch, target_ratio)
    _, _, clamped = _budget_from_retention(terms["soft_retention"], target_ratio, lambda_value)
    total = terms["qa"] + terms["kd"] + terms["hidden"] + alpha * clamped
    total.backward()
    with torch.no_grad():
        for parameter in clone.retention_scorers.parameters():
            if parameter.grad is not None:
                parameter.add_(-1e-4 * parameter.grad)
    after = _score_probability_and_retention(clone, batch, target_ratio)
    del optimizer
    del clone
    return before, after


def _make_batch(batch):
    # Keep one fixed batch on CPU so cloned diagnostic models can reuse it.
    return {key: value.detach().cpu() for key, value in batch.items() if torch.is_tensor(value)}


def run_diagnostic(args):
    if not os.path.isfile(args.checkpoint):
        raise RuntimeError(f"Checkpoint does not exist: {args.checkpoint}")
    _, val_dl, _, _, _ = get_squad_dataloaders(
        batch_size=args.batch_size,
        max_train_samples=1,
        max_val_samples=max(1, args.num_examples),
    )
    batch = _make_batch(next(iter(val_dl)))
    model = AdaptiveDistilBertQA(model_name=MODEL_NAME, device=DEVICE, freeze_transformer=True).to(DEVICE)
    checkpoint = load_ammr_checkpoint(model, args.checkpoint)
    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    target_ratio = float(args.target_ratio if args.target_ratio is not None else metadata.get("target_ratio", 0.60))
    lambda_value = float(metadata.get("lagrangian_multiplier", 0.0))
    teacher = BaselineQAModel(freeze_parameters=True)
    teacher.qa_model.eval()
    model.train()

    terms = _build_production_terms(model, teacher, batch, target_ratio)
    terms["retention_c"] = _global_token_weighted_retention(terms["results"])
    base_before = _score_probability_and_retention(model, batch, target_ratio)

    print("AMMR BUDGET COEFFICIENT SWEEP")
    print(f"Checkpoint: {os.path.abspath(args.checkpoint)}")
    print(f"Epoch: {metadata.get('epoch', 'N/A')}; Step: {metadata.get('step', 'N/A')}")
    print(f"Target ratio: {target_ratio}; Lambda: {lambda_value}")
    print("Fixed real validation batch: yes")
    print("Normal benchmark-style answer_span_mask=None: yes")
    print("Original production model optimizer.step(): never")

    print("\nTABLE 1")
    print("alpha | budget_grad | combined_grad | budget/combined | soft_retention_delta")
    table1 = []
    for alpha in ALPHAS:
        result = _run_gradient_case(model, terms, target_ratio, lambda_value, alpha)
        before, after = _clone_step(model, teacher, batch, target_ratio, lambda_value, alpha)
        budget_total = sum(result["budget_norms"])
        combined_total = sum(result["combined_norms"])
        ratio = budget_total / max(combined_total, EPSILON)
        delta = after["soft_retention"] - before["soft_retention"]
        table1.append((alpha, budget_total, combined_total, ratio, delta))
        print(f"{alpha} | {budget_total:.8e} | {combined_total:.8e} | {ratio:.8e} | {delta:.8e}")
        print(f"  score mean {before['score_mean']:.8e} -> {after['score_mean']:.8e}; probability mean {before['probability_mean']:.8e} -> {after['probability_mean']:.8e}; retention {before['soft_retention']:.8e} -> {after['soft_retention']:.8e}")

    print("\nTABLE 2")
    print("layer | alpha=1 | alpha=10 | alpha=100 | alpha=1000 | alpha=10000")
    layer_rows = []
    for layer in range(len(model.retention_scorers)):
        values = []
        for alpha in ALPHAS:
            result = _run_gradient_case(model, terms, target_ratio, lambda_value, alpha)
            values.append(result["budget_norms"][layer])
        layer_rows.append(values)
        print(layer, *[f"{value:.8e}" for value in values], sep=" | ")

    print("\nTABLE 3")
    print("formulation | r | violation | budget gradient by layer")
    for name, key in (("BASE", "base"), ("NORMALIZED-C", "c")):
        result = _run_gradient_case(model, terms, target_ratio, lambda_value, 1, retention_key=key)
        layer_gradient_text = [f"{row['norm']:.8e}" for row in result['budget_rows']]
        print(f"{name} | {result['retention']:.8e} | {result['violation']:.8e} | {layer_gradient_text}")
    print(f"BASE initial score/probability/retention: {base_before}")
    print("No production parameters or checkpoint files were modified.")


if __name__ == "__main__":
    run_diagnostic(build_parser().parse_args())
