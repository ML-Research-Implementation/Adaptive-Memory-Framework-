import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from evaluate_squad import (
    EXPECTED_VALIDATION_EXAMPLES,
    build_parser,
    load_ammr_checkpoint,
)


class FakeScorerContainer:
    def __init__(self):
        self.loaded = None

    def state_dict(self):
        return {"0.weight": torch.tensor([1.0])}

    def load_state_dict(self, state_dict, strict=True):
        self.loaded = state_dict
        self.strict = strict

    def get_retention_scorers(self):
        return self


class TestEvaluationCheckpoint(unittest.TestCase):
    def make_checkpoint(self, state=None):
        return {
            "model_state_dict": state or {"0.weight": torch.tensor([2.0])},
            "step": 10,
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
            "epoch": 1,
            "lagrangian_multiplier": 0.0,
            "target_ratio": 0.95,
        }

    def test_full_training_checkpoint_loads_model_state_dict(self):
        model = FakeScorerContainer()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "checkpoint.pt")
            torch.save(self.make_checkpoint(), path)
            loaded = load_ammr_checkpoint(model, path)
        self.assertIn("model_state_dict", loaded)
        expected = torch.tensor([2.0], device=model.loaded["0.weight"].device)
        self.assertTrue(torch.equal(model.loaded["0.weight"], expected))
        self.assertTrue(model.strict)

    def test_missing_checkpoint_refuses_initial_scorer_fallback(self):
        with self.assertRaisesRegex(RuntimeError, "Refusing to evaluate with initial scorer weights"):
            load_ammr_checkpoint(FakeScorerContainer(), "missing-checkpoint.pt")

    def test_incompatible_checkpoint_fails_strictly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "bad.pt")
            torch.save(self.make_checkpoint({"wrong.key": torch.tensor([1.0])}), path)
            with self.assertRaisesRegex(RuntimeError, "state mismatch"):
                load_ammr_checkpoint(FakeScorerContainer(), path)

    def test_explicit_checkpoint_arguments_are_selected(self):
        args = build_parser().parse_args([
            "--ammr_best_checkpoint", "/best/checkpoint.pt",
            "--ammr_final_checkpoint", "/final/checkpoint.pt",
            "--max_val_samples", "16",
        ])
        self.assertEqual(args.ammr_best_checkpoint, "/best/checkpoint.pt")
        self.assertEqual(args.ammr_final_checkpoint, "/final/checkpoint.pt")
        self.assertEqual(args.max_val_samples, 16)

    def test_full_validation_constant(self):
        self.assertEqual(EXPECTED_VALIDATION_EXAMPLES, 10570)

    def test_evaluation_does_not_require_full_training_count(self):
        from evaluate_squad import EXPECTED_TRAIN_EXAMPLES
        self.assertEqual(EXPECTED_TRAIN_EXAMPLES, 87599)
        self.assertNotEqual(1, EXPECTED_TRAIN_EXAMPLES)

    def test_results_record_explicit_checkpoint_paths(self):
        from evaluate_squad import DEFAULT_BEST_CHECKPOINT, DEFAULT_FINAL_CHECKPOINT
        self.assertTrue(DEFAULT_BEST_CHECKPOINT.endswith("squad_best_checkpoint.pt"))
        self.assertTrue(DEFAULT_FINAL_CHECKPOINT.endswith("squad_final_checkpoint.pt"))


    def test_verify_loaded_model_handles_different_original_lengths(self):
        from evaluate_squad import verify_loaded_model

        class FakeResult:
            def __init__(self, indices, original, selected, ratio):
                self.selected_indices = indices
                self.num_original = original
                self.num_selected = selected
                self.retention_ratio = ratio

        class FakeModel:
            def __init__(self, results):
                self.results = results
            def eval(self):
                pass
            def get_retention_scorers(self):
                class Scorer:
                    def named_parameters(self):
                        return [("0.weight", torch.tensor([1.0]))]
                return Scorer()
            def __call__(self, ids, mask, return_layer_metrics=True, training=False):
                return (None, None, {"selection_results": self.results})

        fresh_indices = torch.tensor([[0, 2, 4]])
        fresh_result = FakeResult(fresh_indices, 10, 3, 0.3)
        fresh = FakeModel([fresh_result])

        loaded_indices = torch.tensor([[0, 2]])
        loaded_result = FakeResult(loaded_indices, 8, 2, 0.25)
        loaded = FakeModel([loaded_result])

        dataloader = [{"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}]

        stats = verify_loaded_model(fresh, loaded, dataloader, batches=1)
        self.assertEqual(stats["different_masks"], 1)
        self.assertEqual(stats["mask_comparisons"], 1)

if __name__ == "__main__":
    unittest.main()
