import os
import time
import argparse
import traceback
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from config import MODEL_NAME, DEVICE
from src.models_adaptive import AdaptiveDistilBertQA
from src.baseline import BaselineQAModel
from src.squad_data import get_squad_dataloaders
from src.losses import (
    calculate_distillation_loss,
    calculate_lagrangian_budget_loss,
    calculate_hidden_state_distillation_loss
)
from src.utils import set_seed, print_header, save_checkpoint
from src.training_report import new_report, save_report, finalize_report
from src.qa_metrics import evaluate_squad_predictions


def unpack_student_outputs(outputs):
    """Normalize the adaptive student's current three-output API."""
    if not isinstance(outputs, tuple) or len(outputs) != 3:
        raise RuntimeError("Adaptive student must return (start_logits, end_logits, layer_metrics)")
    start_logits, end_logits, layer_metrics = outputs
    return start_logits, end_logits, layer_metrics


def update_retention_lambda(
    lambda_value: float,
    actual_ratio: float,
    target_ratio: float,
    learning_rate: float = 0.005,
    maximum: float = 20.0
):
    """Bounded dual update for a minimum-retention constraint."""
    raw_violation = max(0.0, float(target_ratio) - float(actual_ratio))
    updated = lambda_value + learning_rate * raw_violation
    return min(maximum, max(0.0, updated)), raw_violation


def evaluate(student, val_dl, val_features=None, val_data=None, tokenizer=None):
    was_training = student.training
    student.eval()
    total_loss = 0.0
    feature_logits = []
    total_answer_survival = 0.0
    answer_survival_count = 0
    
    with torch.no_grad():
        for batch in tqdm(val_dl, desc="Validating"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            start_target = batch['start_positions'].to(DEVICE)
            end_target = batch['end_positions'].to(DEVICE)
            
            answer_span_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for batch_idx in range(input_ids.size(0)):
                start_idx = int(start_target[batch_idx].item())
                end_idx = int(end_target[batch_idx].item())
                if 0 <= start_idx <= end_idx < input_ids.size(1):
                    answer_span_mask[batch_idx, start_idx:end_idx + 1] = True
            s_start, s_end, layer_metrics = student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_layer_metrics=True,
                training=False,
                answer_span_mask=answer_span_mask,
                return_original_selection=True
            )
            
            loss_start = F.cross_entropy(s_start, start_target)
            loss_end = F.cross_entropy(s_end, end_target)
            qa_loss = (loss_start + loss_end) / 2
            total_loss += qa_loss.item()
            
            for batch_idx in range(input_ids.size(0)):
                feature_logits.append((s_start[batch_idx].detach().cpu(), s_end[batch_idx].detach().cpu()))
            selections = layer_metrics.get("selection_results", []) if layer_metrics else []
            for batch_idx in range(input_ids.size(0)):
                start_idx = int(start_target[batch_idx].item())
                end_idx = int(end_target[batch_idx].item())
                if not (0 <= start_idx <= end_idx < input_ids.size(1)):
                    continue
                survives = True
                for selection in selections:
                    if selection is None or selection.selected_original_indices is None:
                        continue
                    original = selection.selected_original_indices[batch_idx]
                    survives = survives and bool(((original == start_idx).any() and (original == end_idx).any()).item())
                total_answer_survival += float(survives)
                answer_survival_count += 1
                
    student.train(was_training)
    if val_features is not None and val_data is not None and tokenizer is not None:
        val_em, val_f1, details = evaluate_squad_predictions(
            feature_logits, val_features, val_data, tokenizer
        )
        from src.qa_metrics import summarize_prediction_diagnostics
        diagnostics = summarize_prediction_diagnostics(details)
    else:
        val_em, val_f1 = 0.0, 0.0
        diagnostics = {
            "unique_predicted_answers": 0,
            "prediction_examples": [],
        }
    return (
        total_loss / max(len(val_dl), 1),
        val_em,
        val_f1,
        100.0 * total_answer_survival / max(answer_survival_count, 1),
        diagnostics,
    )

def train(args):
    set_seed(42)
    report = new_report(args)
    report_json_path = getattr(args, "results_json", "training_results.json")
    report_text_path = getattr(args, "results_text", "training_results.txt")
    report_checkpoint_paths = []
    report_start_time = time.time()
    try:
        return _train_with_report(args, report, report_json_path, report_text_path,
                                  report_checkpoint_paths, report_start_time)
    except KeyboardInterrupt:
        finalize_report(report, time.time() - report_start_time, report_checkpoint_paths,
                        status="interrupted", error="Training interrupted by user")
        save_report(report, report_json_path, report_text_path)
        print(f"\nPartial AMMR results saved to {report_json_path} and {report_text_path}")
        raise
    except Exception as exc:
        failure_traceback = traceback.format_exc()
        traceback.print_exc()
        finalize_report(report, time.time() - report_start_time, report_checkpoint_paths,
                        status="failed", error=f"{type(exc).__name__}: {exc}")
        report["failure_traceback"] = failure_traceback
        save_report(report, report_json_path, report_text_path)
        print(f"Partial AMMR results saved to {report_json_path} and {report_text_path}")
        raise

def _train_with_report(args, report, report_json_path, report_text_path,
                       report_checkpoint_paths, report_start_time):
    # Retention is deliberately optimized much more slowly than the frozen
    # task model. The scorer controls a discrete structural decision, so large
    # updates can destroy answer survival between curriculum stages.
    scorer_learning_rate = min(float(args.learning_rate) * 0.1, 3e-4)
    scorer_warmup_epochs = 1
    max_stable_loss = 100.0
    print_header("STABLE TASK-PRESERVING AMMR TRAINING")
    
    train_dl, val_dl, train_data, val_data, val_features = get_squad_dataloaders(
        batch_size=args.batch_size, 
        max_train_samples=args.max_train_samples, 
        max_val_samples=args.max_val_samples
    )
    
    report["config"]["training_examples"] = len(train_data)
    report["config"]["validation_examples"] = len(val_data)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("\nInitializing Teacher (Frozen DistilBERT) and Student (AMMR)...")
    teacher = BaselineQAModel(freeze_parameters=True)
    teacher.qa_model.eval()
    
    student = AdaptiveDistilBertQA(
        model_name=MODEL_NAME, 
        device=DEVICE,
        freeze_transformer=True
    )
    student.unfreeze_scorers()
    student.train()
    
    optimizer = AdamW(student.retention_scorers.parameters(), lr=scorer_learning_rate)
    
    total_steps = len(train_dl) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    lagrangian_multiplier = 0.0
    lagrangian_lr = 0.005
    lagrangian_max = 20.0
    
    start_epoch = 0
    global_step = 0
    
    use_amp = (DEVICE.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None
    
    if args.resume_from and os.path.exists(args.resume_from):
        from src.utils import load_checkpoint
        print(f"Loading checkpoint from {args.resume_from}")
        checkpoint = load_checkpoint(student.retention_scorers, optimizer, args.resume_from, scheduler=scheduler)
        global_step = checkpoint.get('step', 0)
        start_epoch = checkpoint.get('epoch', 0)
        lagrangian_multiplier = checkpoint.get('lagrangian_multiplier', 0.0)
        print(f"Resuming training from epoch {start_epoch}, step {global_step}")

    curriculum = [0.95, 0.90, 0.80, 0.70, 0.60]
    
    lambda_kd = 1.0       
    lambda_h = 1.0        
    lambda_b = 1.0        
    
    print(f"Starting training for {args.epochs} epochs ({total_steps} steps).")
    print(f"Scorer learning rate: {scorer_learning_rate:.2e}; warmup epochs: {scorer_warmup_epochs}")
    
    best_val_score = 0.0
    target_ratio = curriculum[min(max(start_epoch, 0), len(curriculum) - 1)]

    for epoch in range(start_epoch, args.epochs):
        target_ratio = curriculum[min(epoch, len(curriculum)-1)]
        print(f"\n[Epoch {epoch+1}/{args.epochs}] Curriculum Target: {target_ratio*100:.1f}%")
        
        progress_bar = tqdm(train_dl, desc=f"Epoch {epoch+1}")
        
        epoch_retention = 0.0
        epoch_original_tokens = 0.0
        epoch_target = 0.0
        epoch_violation = 0.0
        num_batches = 0
        
        epoch_last_qa_loss = 0.0
        epoch_last_kd_loss = 0.0
        epoch_last_span_survival = 1.0
        epoch_val_answer_survival = 1.0
        epoch_lambda_start = lagrangian_multiplier
        epoch_raw_violation = 0.0
        epoch_optimizer_steps = 0
        epoch_optimizer_skips = {}
        epoch_grad_norm_sum = 0.0
        epoch_first_scorer_checksum = sum(parameter.detach().float().sum().item() for parameter in student.retention_scorers.parameters())
        epoch_first_student_checksum = sum(parameter.detach().float().sum().item() for parameter in student.parameters())

        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            start_target = batch['start_positions'].to(DEVICE)
            end_target = batch['end_positions'].to(DEVICE)
            
            # The floor is enforced per layer against the current valid-token
            # count. The controller uses the normalized ratio directly.
            
            with torch.amp.autocast(device_type=DEVICE.type, enabled=use_amp):
                with torch.no_grad():
                    teacher_outputs = teacher.model(
                        input_ids, 
                        attention_mask, 
                        output_hidden_states=True
                    )
                    t_start = teacher_outputs.start_logits
                    t_end = teacher_outputs.end_logits
                    teacher_hidden_states = teacher_outputs.hidden_states
                    
                # Preserve every answer token through every layer. The span
                # mask is feature-local because SQuAD windows have different
                # answer positions.
                answer_span_mask = torch.zeros_like(input_ids, dtype=torch.bool)
                for batch_idx in range(input_ids.size(0)):
                    start_idx = int(start_target[batch_idx].item())
                    end_idx = int(end_target[batch_idx].item())
                    if 0 <= start_idx < input_ids.size(1) and 0 <= end_idx < input_ids.size(1) and start_idx <= end_idx:
                        answer_span_mask[batch_idx, start_idx:end_idx + 1] = True

                s_start, s_end, layer_metrics = student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_layer_metrics=True,
                    training=True,
                    minimum_retention_ratio=target_ratio,
                    answer_span_mask=answer_span_mask
                )
                
                final_selection = layer_metrics['selection_results'][-1]
                if final_selection is not None:
                    final_indices = final_selection.selected_indices
                    start_kept = (final_indices == start_target.unsqueeze(1)).any(dim=1)
                    end_kept = (final_indices == end_target.unsqueeze(1)).any(dim=1)
                    span_survived = (start_kept & end_kept).float().mean().item()
                else:
                    span_survived = 1.0
                    
                loss_start = F.cross_entropy(s_start, start_target)
                loss_end = F.cross_entropy(s_end, end_target)
                qa_loss = (loss_start + loss_end) / 2
                
                logit_kd_loss = calculate_distillation_loss(s_start, s_end, t_start, t_end, temperature=2.0)
                
                hidden_kd_loss = torch.tensor(0.0, device=DEVICE)
                for layer_idx in range(6):
                    if layer_metrics['selection_results'][layer_idx] is not None:
                        s_hidden = layer_metrics['hidden_states'][layer_idx]
                        t_hidden = teacher_hidden_states[layer_idx + 1]
                        sel_indices = layer_metrics['selection_results'][layer_idx].selected_indices
                        att_mask = layer_metrics['selection_results'][layer_idx].new_attention_mask
                        
                        h_loss = calculate_hidden_state_distillation_loss(
                            student_hidden=s_hidden,
                            teacher_hidden=t_hidden,
                            selected_indices=sel_indices,
                            attention_mask=att_mask,
                            mse_weight=1.0,
                            cos_weight=1.0
                        )
                        hidden_kd_loss += h_loss
                
                # Build the target from the actual pre-selection token counts
                # observed by each active layer.
                # Minimum-retention Lagrangian: under-retention is the
                # violation. A positive multiplier therefore pushes retention
                # upward instead of rewarding collapse.
                actual_ratio = layer_metrics['actual_retention_ratio']
                raw_violation_ratio = torch.relu(
                    torch.as_tensor(target_ratio, device=actual_ratio.device)
                    - actual_ratio
                )
                # A normalized minimum-retention penalty is positive when the
                # model is below target and therefore increases pressure to
                # retain, never to drop more tokens.
                bounded_budget_loss = torch.clamp(
                    lagrangian_multiplier * raw_violation_ratio,
                    min=0.0,
                    max=10.0
                )
                total_loss = qa_loss + lambda_kd * logit_kd_loss + lambda_h * hidden_kd_loss + lambda_b * bounded_budget_loss

            losses_finite = all(torch.isfinite(value).item() for value in (qa_loss, logit_kd_loss, hidden_kd_loss, total_loss))
            losses_stable = all(value.detach().abs().item() <= max_stable_loss for value in (qa_loss, logit_kd_loss, hidden_kd_loss, total_loss))
            warmup = epoch < scorer_warmup_epochs
            should_update_scorer = losses_finite and losses_stable and not warmup
            skip_reasons = []
            if not losses_finite: skip_reasons.append("non_finite_loss")
            if not losses_stable: skip_reasons.append("unstable_loss")
            if warmup: skip_reasons.append("warmup")

            if should_update_scorer:
                if use_amp and scaler is not None:
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(student.retention_scorers.parameters(), max_norm=0.5)
                    if torch.isfinite(grad_norm):
                        scale_before = scaler.get_scale()
                        scaler.step(optimizer)
                        scaler.update()
                        # Scheduler advances only after a real optimizer step.
                        if scaler.get_scale() == scale_before:
                            scheduler.step()
                    else:
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()
                else:
                    total_loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(student.retention_scorers.parameters(), max_norm=0.5)
                    epoch_grad_norm_sum += float(grad_norm.detach().item())
                    if torch.isfinite(grad_norm):
                        optimizer.step()
                        scheduler.step()
                        epoch_optimizer_steps += 1
                    else:
                        optimizer.zero_grad(set_to_none=True)
            else:
                # Do not let unstable gradients alter scorer logits or the
                # scheduler. The frozen teacher/student forward remains usable.
                optimizer.zero_grad(set_to_none=True)
                for reason in skip_reasons or ["non_finite_or_overflow_grad"]:
                    epoch_optimizer_skips[reason] = epoch_optimizer_skips.get(reason, 0) + 1

            with torch.no_grad():
                if losses_finite and losses_stable:
                    lagrangian_multiplier, raw_violation = update_retention_lambda(
                        lagrangian_multiplier,
                        actual_ratio=float(actual_ratio.detach().item()),
                        target_ratio=target_ratio,
                        learning_rate=lagrangian_lr,
                        maximum=lagrangian_max
                    )
                else:
                    raw_violation = max(0.0, target_ratio - float(actual_ratio.detach().item()))

                epoch_raw_violation += raw_violation
                epoch_last_qa_loss = qa_loss.item()
                epoch_last_kd_loss = (logit_kd_loss + hidden_kd_loss).item()
                epoch_last_span_survival = span_survived
                epoch_retention += layer_metrics['actual_retained_tokens'].item()
                epoch_original_tokens += layer_metrics['actual_original_tokens'].item()
                epoch_target += layer_metrics['actual_original_tokens'].item() * target_ratio
                epoch_violation += raw_violation
                num_batches += 1
            
            global_step += 1
            
            if global_step % 10 == 0:
                progress_bar.set_postfix({
                    'L': f"{total_loss.item():.1f}",
                    'QA': f"{qa_loss.item():.1f}",
                    'LogKD': f"{logit_kd_loss.item():.1f}",
                    'HidKD': f"{hidden_kd_loss.item():.1f}",
                    'Ret': f"{float(actual_ratio.item()) * 100.0:.1f}%/{target_ratio * 100.0:.1f}%",
                    'Lam': f"{lagrangian_multiplier:.3f}",
                    'AnsSurv': f"{span_survived*100:.0f}%"
                })
        
        avg_retention = epoch_retention / num_batches
        avg_target = epoch_target / num_batches
        avg_violation = epoch_violation / num_batches
        lambda_after = lagrangian_multiplier
        avg_raw_violation_ratio = epoch_raw_violation / num_batches
        epoch_last_scorer_checksum = sum(parameter.detach().float().sum().item() for parameter in student.retention_scorers.parameters())
        epoch_last_student_checksum = sum(parameter.detach().float().sum().item() for parameter in student.parameters())
        print(f"  Runtime diagnostics: student={epoch_first_student_checksum:.8e}->{epoch_last_student_checksum:.8e}; scorer={epoch_first_scorer_checksum:.8e}->{epoch_last_scorer_checksum:.8e}; steps={epoch_optimizer_steps}; skips={epoch_optimizer_skips}; grad_norm_sum={epoch_grad_norm_sum:.8e}; lambda={epoch_lambda_start:.8e}->{lagrangian_multiplier:.8e}")
        
        actual_retention_pct = 100.0 * epoch_retention / max(epoch_original_tokens, 1e-8)
        target_retention_pct = target_ratio * 100.0
        retention_tolerance = 0.01
        if actual_retention_pct + retention_tolerance < target_retention_pct:
            raise RuntimeError(
                f"Minimum-retention floor violated in epoch {epoch + 1}: "
                f"actual={actual_retention_pct:.2f}% target={target_retention_pct:.2f}%"
            )
        print(f"Epoch {epoch+1} Stability Stats:")
        print(f"  Target retention: {target_retention_pct:.1f}% ({avg_target:.1f} tokens)")
        print(f"  Actual retention: {actual_retention_pct:.1f}% ({avg_retention:.1f} tokens)")
        print(f"  Lambda: {lagrangian_multiplier:.4f} (before={epoch_lambda_start:.4f}, after={lambda_after:.4f})")
        print(f"  Raw violation ratio: {avg_raw_violation_ratio:.6f}")
        print(f"  Answer survival: {100.0 * epoch_last_span_survival:.1f}%")
        print(f"  QA loss: {epoch_last_qa_loss:.4f}")
        print(f"  KD loss: {epoch_last_kd_loss:.4f}")
        print(f"  Minimum-retention floor confirmed: {actual_retention_pct + retention_tolerance >= target_retention_pct}")
        print(f"  Avg normalized violation: {avg_violation:.6f}")
        
        val_loss, val_em, val_f1, epoch_val_answer_survival = evaluate(
            student, val_dl, val_features=val_features, val_data=val_data,
            tokenizer=tokenizer
        )
        print(f"Validation - Epoch {epoch+1}: Loss = {val_loss:.4f}, EM = {val_em:.2f}%, F1 = {val_f1:.2f}%")
        epoch_result = {
            "epoch": epoch + 1,
            "curriculum_retention_target": target_ratio,
            "actual_retention_percentage": actual_retention_pct,
            "target_tokens": avg_target,
            "actual_retained_tokens": avg_retention,
            "retention_violation": avg_raw_violation_ratio,
            "lambda": lagrangian_multiplier,
            "answer_survival_percentage": epoch_val_answer_survival,
            "qa_loss": epoch_last_qa_loss,
            "logit_kd_loss": logit_kd_loss.item(),
            "hidden_state_kd_loss": hidden_kd_loss.item(),
            "total_loss": total_loss.item(),
            "validation_loss": val_loss,
            "validation_em": val_em,
            "validation_f1": val_f1,
            "hard_retention_floor_satisfied": actual_retention_pct + retention_tolerance >= target_retention_pct,
        }
        report["epochs"].append(epoch_result)
        save_report(report, report_json_path, report_text_path)
        
        score = val_f1
        is_best = score > best_val_score
        if is_best:
            best_val_score = score
            save_checkpoint(
                student.retention_scorers, 
                optimizer=optimizer, 
                step=global_step, 
                checkpoint_path="squad_best_checkpoint.pt",
                scheduler_state_dict=scheduler.state_dict(),
                epoch=epoch+1,
                lagrangian_multiplier=lagrangian_multiplier,
                target_ratio=target_ratio
            )
            report_checkpoint_paths.append("squad_best_checkpoint.pt")
            print(f"Saved new best checkpoint (EM: {score:.2f}%)")
            
    # Save final checkpoint
    save_checkpoint(
        student.retention_scorers, 
        optimizer=optimizer, 
        step=global_step, 
        checkpoint_path="squad_final_checkpoint.pt",
        scheduler_state_dict=scheduler.state_dict(),
        epoch=args.epochs,
        lagrangian_multiplier=lagrangian_multiplier,
        target_ratio=target_ratio
    )
    report_checkpoint_paths.append("squad_final_checkpoint.pt")
    finalize_report(report, time.time() - report_start_time, report_checkpoint_paths)
    save_report(report, report_json_path, report_text_path)
    print("\nFINAL AMMR RESULTS")
    print("=" * 80)
    for key, value in report["summary"].items():
        print(f"{key}: {value}")
    print("\nEPOCH RESULTS TABLE")
    for item in report["epochs"]:
        print(f"Epoch {item['epoch']}: target={item['curriculum_retention_target']*100:.1f}% "
              f"actual={item['actual_retention_percentage']:.2f}% QA={item['qa_loss']:.4f} "
              f"LogitKD={item['logit_kd_loss']:.4f} HiddenKD={item['hidden_state_kd_loss']:.4f} "
              f"total={item['total_loss']:.4f} val_EM={item['validation_em']:.2f} val_F1={item['validation_f1']:.2f}")
    print(f"Results saved to {report_json_path} and {report_text_path}")
    print("\nTraining complete.")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AMMR Model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--max_train_samples", type=int, default=5000, help="Max training samples")
    parser.add_argument("--max_val_samples", type=int, default=500, help="Max validation samples")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-3, help="Base learning rate; scorer uses a conservative fraction")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--results_json", type=str, default="training_results.json", help="JSON results path")
    parser.add_argument("--results_text", type=str, default="training_results.txt", help="Text results path")
    
    args = parser.parse_args()
    train(args)
