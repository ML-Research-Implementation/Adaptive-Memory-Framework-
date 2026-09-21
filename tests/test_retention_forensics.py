import argparse
import unittest
from unittest.mock import patch

import torch

from diagnose_retention_forensics import _layer_row, _parameter_stats, build_parser
from diagnose_soft_vs_hard_retention import _layer_record, _protected_count_from_result
from src.models_adaptive import TokenSelector


class TestRetentionForensics(unittest.TestCase):
    def test_parser_accepts_checkpoint_and_threshold(self):
        args = build_parser().parse_args([
            "--checkpoint", "/content/AMMR_CLEAN_RUN/squad_final_checkpoint.pt",
            "--split", "validation",
            "--num-examples", "1",
            "--threshold", "0.0",
        ])
        self.assertEqual(args.checkpoint, "/content/AMMR_CLEAN_RUN/squad_final_checkpoint.pt")
        self.assertEqual(args.num_examples, 1)
        self.assertEqual(args.threshold, 0.0)

    def test_synthetic_mixed_logits_have_raw_and_floor_separation(self):
        selector = TokenSelector(device=torch.device("cpu"))
        hidden = torch.arange(12 * 4, dtype=torch.float32).reshape(1, 12, 4)
        probabilities = torch.tensor([
            [0.99, 0.98, 0.97, 0.20, 0.15, 0.10,
             0.95, 0.05, 0.30, 0.02, 0.90, 0.01]
        ])
        logits = torch.log(probabilities / (1.0 - probabilities))
        protected = torch.zeros(1, 12, dtype=torch.bool)
        protected[:, 0] = True
        protected[:, 11] = True
        attention = torch.ones(1, 12)
        result = selector.select_adaptive(
            hidden, logits, protected, attention,
            training=False, threshold_bias=0.0, minimum_retention_ratio=0.0
        )
        self.assertTrue(bool((result.raw_retained_counts > 0).item()))
        self.assertLess(result.num_selected, 12)
        self.assertTrue(bool((result.selected_indices[0, 1:] > result.selected_indices[0, :-1]).all()))
        self.assertTrue(bool((result.selected_indices == 0).any()))
        self.assertTrue(bool((result.selected_indices == 11).any()))
        self.assertGreater(result.raw_valid_retained_counts.item(), 0)
        self.assertEqual(result.floor_added_counts.item(), 0)
        self.assertFalse(result.topk_repair_activated)

    def test_floor_and_topk_diagnostics_are_detected(self):
        selector = TokenSelector(device=torch.device("cpu"))
        hidden = torch.randn(1, 12, 4)
        logits = torch.full((1, 12), -10.0)
        protected = torch.zeros(1, 12, dtype=torch.bool)
        attention = torch.ones(1, 12)
        result = selector.select_adaptive(
            hidden, logits, protected, attention,
            training=False, minimum_retention_ratio=0.75
        )
        self.assertEqual(result.raw_valid_retained_counts.item(), 0)
        self.assertEqual(result.actual_retained_counts.item(), 9)
        self.assertEqual(result.floor_added_counts.item(), 9)
        self.assertTrue(result.topk_repair_activated)

    def test_layer_row_reports_nonzero_logits_and_counts(self):
        selector = TokenSelector(device=torch.device("cpu"))
        hidden = torch.randn(1, 12, 4)
        logits = torch.tensor([[-2.0, -1.0, 1.0, 2.0, -3.0, 0.5, -0.5, 0.2, -0.2, 1.5, -1.5, 0.0]])
        protected = torch.zeros(1, 12, dtype=torch.bool)
        attention = torch.ones(1, 12)
        result = selector.select_adaptive(hidden, logits, protected, attention, training=False, minimum_retention_ratio=0.5)
        row = _layer_row(0, result, torch.tensor([[101] + [100] * 10 + [102]]), 0.5, 0.0)
        self.assertNotEqual(row["score"]["max"], 0.0)
        self.assertEqual(row["raw_selected_valid_before_floor"], int(result.raw_valid_retained_counts.sum()))
        self.assertEqual(row["final_retained_tokens"], int(result.actual_retained_counts.sum()))
        self.assertIn("topk_repair_activated", row)

    def test_normal_inference_contract_is_none(self):
        with open("evaluate_squad.py", encoding="utf-8") as handle:
            source = handle.read()
        ammr_call = source[source.index("start_logits, end_logits, layer_metrics = unpack_student_outputs(model("):source.index("        total_latency", source.index("start_logits, end_logits, layer_metrics = unpack_student_outputs(model("))]
        self.assertNotIn("answer_span_mask=", ammr_call)

    # ------------------------------------------------------------------
    # Tests for diagnose_soft_vs_hard_retention._layer_record
    # ------------------------------------------------------------------

    def test_layer_record_does_not_crash_when_raw_protected_counts_none(self):
        """_layer_record must not crash when raw_protected_counts is None.

        Root cause of the original crash: the old code did
        ``result.raw_protected_counts.detach()`` unconditionally.  The
        attribute exists in the class definition but is initialised to None
        and only populated by the inference path.  The fix guards it via
        _protected_count_from_result which returns (0, False) when None.
        """
        selector = TokenSelector(device=torch.device("cpu"))
        hidden = torch.randn(1, 8, 4)
        logits = torch.zeros(1, 8)
        protected = torch.zeros(1, 8, dtype=torch.bool)
        protected[:, 0] = True
        attention = torch.ones(1, 8)
        result = selector.select_adaptive(
            hidden, logits, protected, attention,
            training=False, minimum_retention_ratio=0.0
        )
        # Simulate an older checkpoint object where the attribute is None.
        result.raw_protected_counts = None
        count, available = _protected_count_from_result(result)
        self.assertEqual(count, 0)
        self.assertFalse(available)
        # Must complete without AttributeError or TypeError.
        rec = _layer_record(result, [0.0, -1.0])
        self.assertIn("protected_tokens", rec)
        self.assertEqual(rec["protected_tokens"], 0)
        self.assertFalse(rec["protected_tokens_available"])

    def test_layer_record_uses_raw_protected_counts_when_populated(self):
        """When raw_protected_counts is properly set, report it correctly."""
        selector = TokenSelector(device=torch.device("cpu"))
        hidden = torch.randn(1, 8, 4)
        # All logits strongly positive so raw gate fires for every token.
        logits = torch.ones(1, 8) * 5.0
        protected = torch.zeros(1, 8, dtype=torch.bool)
        protected[:, 0] = True
        protected[:, 7] = True
        attention = torch.ones(1, 8)
        result = selector.select_adaptive(
            hidden, logits, protected, attention,
            training=False, minimum_retention_ratio=0.0
        )
        # raw_protected_counts = gates-fired AND protected.
        # With all-positive logits, both protected tokens fire → count == 2.
        count, available = _protected_count_from_result(result)
        self.assertTrue(available)
        self.assertEqual(count, 2)
        rec = _layer_record(result, [0.0])
        self.assertEqual(rec["protected_tokens"], 2)
        self.assertTrue(rec["protected_tokens_available"])

    def test_layer_record_soft_hard_values_in_unit_interval(self):
        """soft_retention and hard_retention must both lie in [0, 1]."""
        selector = TokenSelector(device=torch.device("cpu"))
        hidden = torch.randn(2, 10, 4)
        logits = torch.randn(2, 10)
        protected = torch.zeros(2, 10, dtype=torch.bool)
        attention = torch.ones(2, 10)
        result = selector.select_adaptive(
            hidden, logits, protected, attention,
            training=False, minimum_retention_ratio=0.3
        )
        rec = _layer_record(result, [0.0, -1.0])
        self.assertGreaterEqual(rec["soft_retention"], 0.0)
        self.assertLessEqual(rec["soft_retention"], 1.0)
        self.assertGreaterEqual(rec["hard_retention"], 0.0)
        self.assertLessEqual(rec["hard_retention"], 1.0)
        self.assertAlmostEqual(
            rec["probability_ge_half"] + rec["probability_lt_half"], 1.0, places=5
        )


if __name__ == "__main__":
    unittest.main()
