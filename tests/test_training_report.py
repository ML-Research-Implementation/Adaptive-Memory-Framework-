import argparse
import json
import tempfile
import unittest
from pathlib import Path

from src.training_report import finalize_report, new_report, save_report


class TestTrainingReport(unittest.TestCase):
    def test_report_contains_required_epoch_fields_and_persists(self):
        args = argparse.Namespace(epochs=1, batch_size=1, max_train_samples=2, max_val_samples=2)
        report = new_report(args)
        report["config"]["training_examples"] = 2
        report["config"]["validation_examples"] = 2
        report["epochs"].append({
            "epoch": 1,
            "curriculum_retention_target": 0.95,
            "actual_retention_percentage": 95.2,
            "target_tokens": 10.0,
            "actual_retained_tokens": 9.52,
            "retention_violation": 0.0,
            "lambda": 0.0,
            "answer_survival_percentage": 100.0,
            "qa_loss": 1.0,
            "logit_kd_loss": 2.0,
            "hidden_state_kd_loss": 3.0,
            "total_loss": 6.0,
            "validation_loss": 0.5,
            "validation_em": 90.0,
            "validation_f1": 95.0,
            "hard_retention_floor_satisfied": True,
        })
        finalize_report(report, 1.5, ["squad_final_checkpoint.pt"])
        with tempfile.TemporaryDirectory() as directory:
            json_path = str(Path(directory) / "training_results.json")
            text_path = str(Path(directory) / "training_results.txt")
            save_report(report, json_path, text_path)
            loaded = json.loads(Path(json_path).read_text(encoding="utf-8"))
            self.assertTrue(Path(json_path).exists())
            self.assertTrue(Path(text_path).exists())
            self.assertEqual(loaded["epochs"][0]["validation_f1"], 95.0)
            self.assertTrue(loaded["summary"]["hard_retention_floor_satisfied_every_epoch"])
            text = Path(text_path).read_text(encoding="utf-8")
            self.assertIn("FINAL AMMR RESULTS", text)
            self.assertIn("EPOCH RESULTS", text)


if __name__ == "__main__":
    unittest.main()
