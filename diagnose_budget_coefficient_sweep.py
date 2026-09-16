"""Diagnostic-only AMMR budget coefficient sweep.

No production model parameters are updated and no checkpoint is written. Every
loss component and every alpha uses a fresh student forward/autograd graph.
"""

import argparse
import copy
import os
from typing import Dict, List

import torch
import torch.nn.functional as F

from config import DEVICE, MODEL_NAME
from evaluate_squad import load_ammr_checkpoint
from src.baseline import BaselineQAModel
from src.losses import calculate_distillation_loss, calculate_hidden_state_distillation_loss
from src.models_adaptive import AdaptiveDistilBertQA
from src.squad_data import get_squad_dataloaders

ALPHAS = (1, 10, 100, 1000, 10000)
EPSILON = 1e-12


def build_parser():
    parser = argparse.ArgumentParser(description="Diagnostic-only AMMR budget coefficient sweep")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--target-ratio", type=float, default=None)
    parser.add_argument("--num-examples", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Temporary diagnostic step size; production scorer LR is reported separately")
    return parser


def _zero_grads(model):
    for parameter in model.retention_scorers.parameters():
        parameter.grad = None


def _layer_grad_stats(model) -> List[Dict[str, float]]:
    rows = []
    for layer, scorer in enumerate(model.retention_scorers):
        values = [p.grad.detach().float().reshape(-1) for p in scorer.parameters() if p.grad is not None]
        flat = torch.cat(values) if values else torch.zeros(1, device=DEVICE)
        rows.append({
            "layer": layer,
            "norm": float(flat.norm().item()),
            "mean": float(flat.mean().item()),
            "min": float(flat.min().item()),
            "max": float(flat.max().item()),
        })
    return rows


def _batch_on_device(batch):
    return {key: value.to(DEVICE) for key, value in batch.items() if torch.is_tensor(value)}


def _teacher_targets(teacher, batch):
    with torch.no_grad():
        return teacher.model(
            batch["input_ids"], batch["attention_mask"], output_hidden_states=True
        )


def _fresh_forward(model, teacher_outputs, batch, target_ratio):
    start_logits, end_logits, metrics = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        return_layer_metrics=True,
        training=True,
        minimum_retention_ratio=target_ratio,
        answer_span_mask=None,
    )
    results = [result for result in metrics["selection_results"] if result is not None]
    qa_loss = (
        F.cross_entropy(start_logits, batch["start_positions"])
        + F.cross_entropy(end_logits, batch["end_positions"])
    ) / 2.0
    kd_loss = calculate_distillation_loss(
        start_logits, end_logits,
        teacher_outputs.start_logits.detach(), teacher_outputs.end_logits.detach(),
        temperature=2.0,
    )
    hidden_loss = torch.zeros((), device=DEVICE)
    for layer_idx, result in enumerate(metrics["selection_results"]):
        if result is None:
            continue
        hidden_loss = hidden_loss + calculate_hidden_state_distillation_loss(
            student_hidden=metrics["hidden_states"][layer_idx],
            teacher_hidden=teacher_outputs.hidden_states[layer_idx + 1].detach(),
            selected_indices=result.selected_indices,
            attention_mask=result.new_attention_mask,
        )
    base_retention = torch.stack([result.soft_retention_ratio for result in results]).mean()
    numerator = torch.zeros((), device=DEVICE)
    denominator = torch.zeros((), device=DEVICE)
    for result in results:
        for row, valid_count in zip(result.retention_probs, result.actual_valid_counts.detach().long()):
            count = int(valid_count.item())
            numerator = numerator + row[:count].sum()
            denominator = denominator + float(count)
    normalized_c_retention = numerator / denominator.clamp_min(1.0)
    return {
        "qa": qa_loss,
        "kd": kd_loss,
        "hidden": hidden_loss,
        "base_retention": base_retention,
        "c_retention": normalized_c_retention,
    }


def _budget_terms(retention, target_ratio, lambda_value):
    violation = torch.relu(torch.as_tensor(target_ratio, device=retention.device) - retention)
    unscaled = lambda_value * violation
    clamped = torch.clamp(unscaled, min=0.0, max=10.0)
    return violation, unscaled, clamped


def _stats_snapshot(model, teacher_outputs, batch, target_ratio):
    with torch.no_grad():
        forward = _fresh_forward(model, teacher_outputs, batch, target_ratio)
        results = []
        # A second fresh forward is intentionally avoided here only because all
        # values are detached and this function is used for reporting.
        model.eval()
        _, _, metrics = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            return_layer_metrics=True, training=True,
            minimum_retention_ratio=target_ratio, answer_span_mask=None,
        )
        model.train()
        results = [result for result in metrics["selection_results"] if result is not None]
        scores = torch.cat([result.retention_scores.float().reshape(-1) for result in results])
        probs = torch.cat([result.retention_probs.float().reshape(-1) for result in results])
    return {
        "score_mean": float(scores.mean().item()),
        "probability_mean": float(probs.mean().item()),
        "soft_retention": float(forward["base_retention"].item()),
    }


def _fresh_gradient(model, teacher_outputs, batch, target_ratio, kind, lambda_value, alpha=1, retention_key="base"):
    model.train()
    _zero_grads(model)
    terms = _fresh_forward(model, teacher_outputs, batch, target_ratio)
    if kind == "qa":
        loss = terms["qa"]
    elif kind == "kd":
        loss = terms["kd"]
    elif kind == "hidden":
        loss = terms["hidden"]
    elif kind == "budget":
        retention = terms["base_retention"] if retention_key == "base" else terms["c_retention"]
        violation, unscaled, clamped = _budget_terms(retention, target_ratio, lambda_value)
        loss = alpha * clamped
        print(
            f"BUDGET CHECK [{retention_key}] alpha={alpha}: "
            f"r={retention.detach().item():.8e} tau={target_ratio:.8e} "
            f"violation={violation.detach().item():.8e} lambda={lambda_value:.8e} "
            f"unscaled={unscaled.detach().item():.8e} clamped={clamped.detach().item():.8e} "
            f"requires_grad={loss.requires_grad}"
        )
        if not loss.requires_grad:
            raise RuntimeError("Budget loss is disconnected from scorer graph")
    elif kind == "combined":
        retention = terms["base_retention"] if retention_key == "base" else terms["c_retention"]
        _, _, clamped = _budget_terms(retention, target_ratio, lambda_value)
        loss = terms["qa"] + terms["kd"] + terms["hidden"] + alpha * clamped
    else:
        raise ValueError(kind)
    loss.backward()
    rows = _layer_grad_stats(model)
    _zero_grads(model)
    return terms, rows


def _finite_difference(model, teacher_outputs, batch, target_ratio, lambda_value):
    terms = _fresh_forward(model, teacher_outputs, batch, target_ratio)
    retention = terms["base_retention"].detach()
    base = float(_budget_terms(retention, target_ratio, lambda_value)[2].item())
    delta = 1e-3
    scorer = model.retention_scorers[0]
    parameter = next(scorer.parameters())
    with torch.no_grad():
        original = parameter.view(-1)[0].item()
        parameter.view(-1)[0].fill_(original + delta)
    plus_terms = _fresh_forward(model, teacher_outputs, batch, target_ratio)
    plus = float(_budget_terms(plus_terms["base_retention"].detach(), target_ratio, lambda_value)[2].item())
    with torch.no_grad():
        parameter.view(-1)[0].fill_(original - delta)
    minus_terms = _fresh_forward(model, teacher_outputs, batch, target_ratio)
    minus = float(_budget_terms(minus_terms["base_retention"].detach(), target_ratio, lambda_value)[2].item())
    with torch.no_grad():
        parameter.view(-1)[0].fill_(original)
    if float(terms["base_retention"].detach().item()) < target_ratio and not (plus >= base - 1e-10 and minus <= base + 1e-10):
        raise RuntimeError(f"Finite-difference budget direction failed: minus={minus}, base={base}, plus={plus}")
    print(f"FINITE DIFFERENCE: lower={minus:.8e} base={base:.8e} higher={plus:.8e}")


def _simulated_step(source_model, teacher, batch, target_ratio, lambda_value, alpha, kind):
    clone = copy.deepcopy(source_model).to(DEVICE)
    clone.train()
    teacher_outputs = _teacher_targets(teacher, batch)
    before = _stats_snapshot(clone, teacher_outputs, batch, target_ratio)
    terms = _fresh_forward(clone, teacher_outputs, batch, target_ratio)
    retention = terms["base_retention"]
    _, _, clamped = _budget_terms(retention, target_ratio, lambda_value)
    if kind == "budget":
        loss = alpha * clamped
    elif kind == "combined":
        loss = terms["qa"] + terms["kd"] + terms["hidden"] + alpha * clamped
    else:
        raise ValueError(kind)
    if not loss.requires_grad:
        raise RuntimeError(f"{kind} simulation loss is disconnected")
    loss.backward()
    with torch.no_grad():
        for parameter in clone.retention_scorers.parameters():
            if parameter.grad is not None:
                parameter.add_(-source_model._diagnostic_lr * parameter.grad)
    after = _stats_snapshot(clone, teacher_outputs, batch, target_ratio)
    if kind == "budget" and after["soft_retention"] > before["soft_retention"] + 1e-8:
        raise RuntimeError(
            f"Budget-only update increased retention: {before['soft_retention']} -> {after['soft_retention']}"
        )
    del clone
    return before, after


def _load_and_prepare(args):
    if not os.path.isfile(args.checkpoint):
        raise RuntimeError(f"Checkpoint does not exist: {args.checkpoint}")
    _, val_dl, _, _, _ = get_squad_dataloaders(
        batch_size=args.batch_size, max_train_samples=1, max_val_samples=max(1, args.num_examples)
    )
    batch = _batch_on_device({key: value.detach().cpu() for key, value in next(iter(val_dl)).items() if torch.is_tensor(value)})
    model = AdaptiveDistilBertQA(model_name=MODEL_NAME, device=DEVICE, freeze_transformer=True).to(DEVICE)
    checkpoint = load_ammr_checkpoint(model, args.checkpoint)
    teacher = BaselineQAModel(freeze_parameters=True)
    teacher.qa_model.eval()
    model._diagnostic_lr = float(args.learning_rate)
    return model, teacher, batch, checkpoint


def run_diagnostic(args):
    model, teacher, batch, checkpoint = _load_and_prepare(args)
    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    target_ratio = float(args.target_ratio if args.target_ratio is not None else metadata.get("target_ratio", 0.60))
    lambda_value = float(metadata.get("lagrangian_multiplier", 0.0))
    teacher_outputs = _teacher_targets(teacher, batch)
    print("AMMR BUDGET COEFFICIENT SWEEP")
    print(f"Checkpoint: {os.path.abspath(args.checkpoint)}")
    print(f"Epoch: {metadata.get('epoch', 'N/A')}; Step: {metadata.get('step', 'N/A')}")
    print(f"Target ratio: {target_ratio}; Lambda: {lambda_value}")
    print("Fixed real validation batch: yes")
    print("Normal benchmark-style answer_span_mask=None: yes")
    print("Original production model optimizer.step(): never")

    _, base_budget_rows = _fresh_gradient(model, teacher_outputs, batch, target_ratio, "budget", lambda_value)
    first_terms = _fresh_forward(model, teacher_outputs, batch, target_ratio)
    if float(first_terms["base_retention"].detach().item()) >= target_ratio:
        raise RuntimeError("Selected batch is not violating tau > r; choose another fixed validation batch")
    _finite_difference(model, teacher_outputs, batch, target_ratio, lambda_value)
    budget_before, budget_after = _simulated_step(model, teacher, batch, target_ratio, lambda_value, 1, "budget")
    print(f"ALPHA=1 DIRECTION CHECK: retention {budget_before['soft_retention']:.8e} -> {budget_after['soft_retention']:.8e}")

    print("\nTABLE 1")
    print("alpha | budget_grad | combined_grad | budget/combined | soft_retention_delta")
    results = {}
    for alpha in ALPHAS:
        _, budget_rows = _fresh_gradient(model, teacher_outputs, batch, target_ratio, "budget", lambda_value, alpha)
        _, combined_rows = _fresh_gradient(model, teacher_outputs, batch, target_ratio, "combined", lambda_value, alpha)
        before, after = _simulated_step(model, teacher, batch, target_ratio, lambda_value, alpha, "combined")
        budget_total = sum(row["norm"] for row in budget_rows)
        combined_total = sum(row["norm"] for row in combined_rows)
        delta = after["soft_retention"] - before["soft_retention"]
        results[alpha] = (budget_rows, combined_rows)
        print(f"{alpha} | {budget_total:.8e} | {combined_total:.8e} | {budget_total / max(combined_total, EPSILON):.8e} | {delta:.8e}")
        print(f"  score mean {before['score_mean']:.8e} -> {after['score_mean']:.8e}; probability mean {before['probability_mean']:.8e} -> {after['probability_mean']:.8e}; retention {before['soft_retention']:.8e} -> {after['soft_retention']:.8e}")

    print("\nTABLE 2")
    print("layer | alpha=1 | alpha=10 | alpha=100 | alpha=1000 | alpha=10000")
    for layer in range(len(model.retention_scorers)):
        print(layer, *[f"{results[alpha][0][layer]['norm']:.8e}" for alpha in ALPHAS], sep=" | ")

    print("\nTABLE 3")
    print("formulation | r | violation | budget gradient by layer")
    for name, key in (("BASE", "base"), ("NORMALIZED-C", "c")):
        terms, rows = _fresh_gradient(model, teacher_outputs, batch, target_ratio, "budget", lambda_value, 1, key)
        retention = terms["base_retention"] if key == "base" else terms["c_retention"]
        violation, _, _ = _budget_terms(retention, target_ratio, lambda_value)
        print(f"{name} | {retention.detach().item():.8e} | {violation.detach().item():.8e} | {[f'{row['norm']:.8e}' for row in rows]}")
    print("No production parameters or checkpoint files were modified.")


if __name__ == "__main__":
    run_diagnostic(build_parser().parse_args())
