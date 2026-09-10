import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from evaluate_squad import load_trained_ammr_checkpoint


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
    def test_full_training_checkpoint_loads_model_state_dict(self):
        model = FakeScorerContainer()
        state = {"0.weight": torch.tensor([2.0])}
        checkpoint = {
            "model_state_dict": state,
            "step": 10,
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
            "epoch": 1,
            "lagrangian_multiplier": 0.0,
            "target_ratio": 0.95,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "squad_best_checkpoint.pt")
            torch.save(checkpoint, path)
            loaded = load_trained_ammr_checkpoint(model, path)
        self.assertIn("model_state_dict", loaded)
        self.assertTrue(torch.equal(model.loaded["0.weight"], torch.tensor([2.0])))


if __name__ == "__main__":
    unittest.main()
