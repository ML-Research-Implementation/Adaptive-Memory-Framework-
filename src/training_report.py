"""Persistent, human-readable AMMR training reports."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


REQUIRED_EPOCH_FIELDS = [
    "epoch", "curriculum_retention_target", "actual_retention_percentage",
    "target_tokens", "actual_retained_tokens", "retention_violation",
    "lambda", "answer_survival_percentage", "qa_loss", "logit_kd_loss",
    "hidden_state_kd_loss", "total_loss", "validation_loss",
    "validation_em", "validation_f1", "hard_retention_floor_satisfied",
]


def new_report(args: Any) -> Dict[str, Any]:
    return {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "config": {
            "epochs": int(args.epochs),
            "training_examples": None,
            "validation_examples": None,
            "batch_size": int(args.batch_size),
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
        },
        "epochs": [],
        "summary": {},
    }


def save_report(report: Dict[str, Any], json_path: str = "training_results.json",
                text_path: str = "training_results.txt") -> None:
    """Atomically-ish persist the current report, including partial runs."""
    report["updated_at"] = datetime.now(timezone.utc).isoformat()
    json_file = Path(json_path)
    json_file.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["AMMR TRAINING RESULTS", "=" * 80,
             f"Status: {report.get('status', 'unknown')}"]
    config = report.get("config", {})
    lines.extend([
        f"Training examples: {config.get('training_examples')}",
        f"Validation examples: {config.get('validation_examples')}",
        f"Epochs: {config.get('epochs')}",
    ])
    epochs = report.get("epochs", [])
    if epochs:
        lines.append("\nEPOCH RESULTS")
        lines.append("Epoch | Target | Actual | Target tokens | Retained | Violation | Lambda | Answer survival | QA | Logit KD | Hidden KD | Total | Val loss | EM | F1")
        for item in epochs:
            lines.append(
                f"{item['epoch']:>5} | {item['curriculum_retention_target'] * 100:>6.1f}% | "
                f"{item['actual_retention_percentage']:>6.2f}% | {item['target_tokens']:>13.2f} | "
                f"{item['actual_retained_tokens']:>8.2f} | {item['retention_violation']:>9.6f} | "
                f"{item['lambda']:>6.4f} | {item['answer_survival_percentage']:>15.2f}% | "
                f"{item['qa_loss']:>8.4f} | {item['logit_kd_loss']:>8.4f} | "
                f"{item['hidden_state_kd_loss']:>9.4f} | {item['total_loss']:>8.4f} | "
                f"{item['validation_loss']:>8.4f} | {item['validation_em']:>5.2f} | {item['validation_f1']:>5.2f}"
            )
    summary = report.get("summary", {})
    if summary:
        lines.extend(["\nFINAL AMMR RESULTS", "=" * 80])
        for key, value in summary.items():
            lines.append(f"{key}: {value}")
    Path(text_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize_report(report: Dict[str, Any], elapsed_seconds: float,
                    checkpoint_paths: List[str], status: str = "completed",
                    error: Optional[str] = None) -> None:
    epochs = report.get("epochs", [])
    best_em = max(epochs, key=lambda x: x["validation_em"]) if epochs else None
    best_f1 = max(epochs, key=lambda x: x["validation_f1"]) if epochs else None
    final = epochs[-1] if epochs else {}
    floors = [bool(item["hard_retention_floor_satisfied"]) for item in epochs]
    report["status"] = status
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    report["summary"] = {
        "best_validation_em": final_value(best_em, "validation_em"),
        "best_validation_em_epoch": final_value(best_em, "epoch"),
        "best_validation_f1": final_value(best_f1, "validation_f1"),
        "best_validation_f1_epoch": final_value(best_f1, "epoch"),
        "final_validation_em": final.get("validation_em"),
        "final_validation_f1": final.get("validation_f1"),
        "final_retention_percentage": final.get("actual_retention_percentage"),
        "minimum_retention_achieved_percentage": min((x["actual_retention_percentage"] for x in epochs), default=None),
        "hard_retention_floor_satisfied_every_epoch": bool(floors) and all(floors),
        "final_answer_survival_percentage": final.get("answer_survival_percentage"),
        "total_training_time_seconds": elapsed_seconds,
        "training_examples": report["config"].get("training_examples"),
        "validation_examples": report["config"].get("validation_examples"),
        "epochs": report["config"].get("epochs"),
        "checkpoint_paths": checkpoint_paths,
    }
    if error:
        report["error"] = error


def final_value(item: Optional[Dict[str, Any]], key: str) -> Any:
    return item.get(key) if item else None
