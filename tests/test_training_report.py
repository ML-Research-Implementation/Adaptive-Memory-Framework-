import argparse
import json
import tempfile
import unittest
from unittest.mock import patch
import io
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

from src.training_report import finalize_report, new_report, save_report
from src.qa_metrics import evaluate_squad_predictions
from src.models_adaptive import TokenSelector
import torch


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
    def test_failure_traceback_is_preserved_and_reported(self):
        import train
        args = argparse.Namespace(epochs=1, batch_size=1, max_train_samples=1, max_val_samples=1, learning_rate=1e-3, resume_from=None, results_json="failure_test.json", results_text="failure_test.txt")
        def fail(*unused, **unused_kwargs):
            raise ValueError("deliberate training failure")
        output = io.StringIO()
        try:
            with patch.object(train, "_train_with_report", side_effect=fail), redirect_stdout(output), redirect_stderr(output):
                with self.assertRaisesRegex(ValueError, "deliberate training failure"):
                    train.train(args)
            self.assertIn("ValueError: deliberate training failure", output.getvalue())
            report = json.loads(Path("failure_test.json").read_text(encoding="utf-8"))
            self.assertIn("failure_traceback", report)
            self.assertIn("ValueError: deliberate training failure", report["failure_traceback"])
        finally:
            for filename in ("failure_test.json", "failure_test.txt"):
                Path(filename).unlink(missing_ok=True)

    def test_text_metrics_change_when_logits_change(self):
        class Tokenizer:
            def decode(self, ids, skip_special_tokens=True):
                return {1: "alpha", 2: "beta"}.get(int(ids[0]), "")

        features = [
            {"example_id": "e1", "input_ids": [0, 1, 2], "offset_mapping": [None, (0, 5), (6, 10)]},
        ]
        examples = [{"id": "e1", "answers": {"text": ["alpha"]}}]
        wrong = [(torch.tensor([0.0, 0.0, 4.0]), torch.tensor([0.0, 0.0, 4.0]))]
        right = [(torch.tensor([0.0, 4.0, 0.0]), torch.tensor([0.0, 4.0, 0.0]))]
        wrong_em, wrong_f1, _ = evaluate_squad_predictions(wrong, features, examples, Tokenizer())
        right_em, right_f1, _ = evaluate_squad_predictions(right, features, examples, Tokenizer())
        self.assertLess(wrong_em, right_em)
        self.assertLess(wrong_f1, right_f1)

    def test_answer_span_survival_uses_original_positions(self):
        selector = TokenSelector()
        hidden = torch.randn(1, 8, 4)
        scores = torch.full((1, 8), -100.0)
        protected = torch.zeros(1, 8, dtype=torch.bool)
        protected[:, 5:7] = True
        attention = torch.ones(1, 8)
        result = selector.select_adaptive(hidden, scores, protected, attention, training=False,
                                          minimum_retention_ratio=0.6)
        self.assertTrue(torch.all(torch.isin(torch.tensor([5, 6]), result.selected_indices[0])))

    def test_feature_answer_span_mapping_is_preserved(self):
        feature = {
            "example_id": "e1", "input_ids": [101, 11, 12, 102],
            "offset_mapping": [None, (0, 5), (6, 10), None],
            "sequence_ids": [None, 1, 1, None],
        }
        self.assertEqual(feature["offset_mapping"][1], (0, 5))
        self.assertEqual(feature["sequence_ids"][2], 1)


if __name__ == "__main__":
    unittest.main()
