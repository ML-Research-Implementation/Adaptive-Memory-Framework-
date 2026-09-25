import argparse
import unittest
from unittest.mock import patch

import torch

from diagnose_retention_forensics import _layer_row, _parameter_stats, build_parser

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
        self.assertGreater(result.raw_retained_counts.item(), 0)
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
        self.assertEqual(result.raw_retained_counts.item(), 0)
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
        self.assertEqual(row["raw_selected_before_floor"], int(result.raw_retained_counts.sum()))
        self.assertEqual(row["final_retained_tokens"], int(result.actual_retained_counts.sum()))
        self.assertIn("topk_repair_activated", row)

    def test_normal_inference_contract_is_none(self):
        with open("evaluate_squad.py", encoding="utf-8") as handle:
            source = handle.read()
        ammr_call = source[source.index("start_logits, end_logits, layer_metrics = unpack_student_outputs(model("):source.index("        total_latency", source.index("start_logits, end_logits, layer_metrics = unpack_student_outputs(model("))]
        self.assertNotIn("answer_span_mask=", ammr_call)

    def test_threshold_bias_changes_hard_selection(self):
        selector = TokenSelector(device=torch.device("cpu"))
        hidden = torch.randn(1, 12, 4)
        logits = torch.tensor([[-0.1, -0.1, -0.1, 0.1, 0.1, 0.1, -0.2, 0.2, 0.0, 0.0, 0.5, -0.5]])
        protected = torch.zeros(1, 12, dtype=torch.bool)
        attention = torch.ones(1, 12)
        
        # Bias 0.0
        result1 = selector.select_adaptive(hidden, logits, protected, attention, training=False, threshold_bias=0.0, minimum_retention_ratio=0.0)
        # Bias -0.15 (less tokens retained because score is lower)
        result2 = selector.select_adaptive(hidden, logits, protected, attention, training=False, threshold_bias=-0.15, minimum_retention_ratio=0.0)
        
        selected1 = result1.actual_retained_counts.item()
        selected2 = result2.actual_retained_counts.item()
        
        raw_selected1 = result1.raw_retained_counts.sum().item()
        raw_selected2 = result2.raw_retained_counts.sum().item()
        
        self.assertLess(selected2, selected1)
        self.assertLess(raw_selected2, raw_selected1)

    def test_diagnostic_sweep_defaults_and_accounting(self):
        import diagnose_soft_vs_hard_retention as diag
        parser = diag.build_parser()
        args = parser.parse_args(["--checkpoint", "dummy.pt"])
        expected_biases = [0.00, -0.05, -0.10, -0.15, -0.20, -0.25, -0.30, -0.35, -0.40, -0.45, -0.50, -0.55, -0.60, -0.65, -0.70, -0.75, -0.80, -0.85, -0.90, -0.95, -1.00]
        self.assertEqual(args.threshold_biases, expected_biases)
        
        with open("diagnose_soft_vs_hard_retention.py", encoding="utf-8") as handle:
            source = handle.read()
            
        self.assertIn("PER-LAYER ACCOUNTING AUDIT", source)
        self.assertIn("floor_req", source)

    @unittest.mock.patch('diagnose_soft_vs_hard_retention.os.path.isfile')
    @unittest.mock.patch('diagnose_soft_vs_hard_retention.load_ammr_checkpoint')
    @unittest.mock.patch('diagnose_soft_vs_hard_retention.get_squad_dataloaders')
    @unittest.mock.patch('diagnose_soft_vs_hard_retention.AdaptiveDistilBertQA')
    def test_diagnostic_accounting_handles_shrinking_sequences(self, mock_model_class, mock_dataloaders, mock_load_checkpoint, mock_isfile):
        import diagnose_soft_vs_hard_retention as diag
        mock_isfile.return_value = True
        
        import torch
        device_1 = torch.device('cpu')
        device_2 = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        
        # Mock dataloader yielding one batch of length 384
        batch = {
            "input_ids": torch.zeros((1, 384), dtype=torch.long, device=device_2),
            "attention_mask": torch.ones((1, 384), dtype=torch.long, device=device_2),
            "start_positions": torch.zeros(1, dtype=torch.long, device=device_2),
            "end_positions": torch.zeros(1, dtype=torch.long, device=device_2),
        }
        mock_dataloaders.return_value = (None, [batch], None, None, None)
        mock_load_checkpoint.return_value = {"target_ratio": 0.6}
        
        # Create a mock model instance
        mock_model = unittest.mock.MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.retention_schedule = [0.6]
        mock_model_class.return_value = mock_model
        
        # Mock selection results: Layer 0 has length 384, Layer 1 has length 171
        from src.models_adaptive import TokenSelectionResult
        
        res0 = TokenSelectionResult(
            selected_indices=torch.zeros((1, 171), dtype=torch.long, device=device_1),
            selected_hidden_states=torch.zeros(1, 171, 4, device=device_1),
            new_attention_mask=torch.ones((1, 171), dtype=torch.long, device=device_2),
            retention_scores=torch.ones((1, 384), device=device_1),
            retention_probs=torch.ones((1, 384), device=device_1),
            num_selected=171,
            num_original=384
        )
        res0.actual_retained_counts = torch.tensor([171], device=device_1)
        res0.raw_retained_counts = torch.tensor([171], device=device_1)
        res0.floor_added_counts = torch.tensor([0], device=device_1)
        res0.actual_valid_counts = torch.tensor([384], device=device_1)
        res0.selected_valid_mask = torch.ones((1, 171), dtype=torch.bool, device=device_1)
        
        res1 = TokenSelectionResult(
            selected_indices=torch.zeros((1, 100), dtype=torch.long, device=device_1),
            selected_hidden_states=torch.zeros(1, 100, 4, device=device_1),
            new_attention_mask=torch.ones((1, 100), dtype=torch.long, device=device_2),
            retention_scores=torch.ones((1, 171), device=device_1), # This is length 171, simulating the shrink!
            retention_probs=torch.ones((1, 171), device=device_1),
            num_selected=100,
            num_original=171
        )
        res1.actual_retained_counts = torch.tensor([100], device=device_1)
        res1.raw_retained_counts = torch.tensor([100], device=device_1)
        res1.floor_added_counts = torch.tensor([0], device=device_1)
        res1.actual_valid_counts = torch.tensor([171], device=device_1)
        res1.selected_valid_mask = torch.ones((1, 100), dtype=torch.bool, device=device_1)
        
        # Mock the forward pass output
        metrics = {"selection_results": [res0, res1]}
        # Model returns (start_logits, end_logits, metrics)
        mock_model.return_value = (torch.zeros((1, 384)), torch.zeros((1, 384)), metrics)
        
        parser = diag.build_parser()
        args = parser.parse_args(["--checkpoint", "dummy.pt", "--num-examples", "1", "--threshold-biases", "0.00"])
        
        try:
            diag.run_diagnostic(args)
        except IndexError as e:
            self.fail(f"run_diagnostic raised IndexError indicating accounting shape mismatch: {e}")

    def test_diagnostic_no_compaction_retains_sequence_length(self):
        selector = TokenSelector(device=torch.device("cpu"))
        hidden = torch.randn(1, 12, 4)
        logits = torch.tensor([[-0.1, -0.1, -0.1, 0.1, 0.1, 0.1, -0.2, 0.2, 0.0, 0.0, 0.5, -0.5]])
        protected = torch.zeros(1, 12, dtype=torch.bool)
        attention = torch.ones(1, 12)
        
        # With normal compaction, sequence length shrinks
        res_compact = selector.select_adaptive(
            hidden, logits, protected, attention, training=False, minimum_retention_ratio=0.0, diagnostic_no_compaction=False
        )
        # With diagnostic_no_compaction=True, sequence length remains original
        res_mask = selector.select_adaptive(
            hidden, logits, protected, attention, training=False, minimum_retention_ratio=0.0, diagnostic_no_compaction=True
        )
        
        # Max retained sequence dimension
        self.assertLess(res_compact.selected_indices.shape[1], 12)
        self.assertEqual(res_mask.selected_indices.shape[1], 12)
        self.assertEqual(res_mask.selected_hidden_states.shape[1], 12)
        
        # Even though sequence didn't shrink physically, actual_retained_counts must still match exactly
        self.assertEqual(res_mask.actual_retained_counts.item(), res_compact.actual_retained_counts.item())
        
        # The unselected tokens in hidden_states should be zeroed
        # And attention mask should have zeroes where dropped
        retained = res_mask.selected_valid_mask[0]
        for i in range(12):
            if not retained[i]:
                self.assertEqual(res_mask.new_attention_mask[0, i].item(), 0)
                self.assertTrue(torch.all(res_mask.selected_hidden_states[0, i] == 0))

    def test_diagnostic_force_all_retain_keeps_all_valid(self):
        selector = TokenSelector(device=torch.device("cpu"))
        hidden = torch.randn(1, 12, 4)
        logits = torch.tensor([[-0.1, -0.1, -0.1, 0.1, 0.1, 0.1, -0.2, 0.2, 0.0, 0.0, 0.5, -0.5]])
        protected = torch.zeros(1, 12, dtype=torch.bool)
        attention = torch.ones(1, 12)
        attention[0, 10:] = 0  # 2 padding tokens, so 10 valid
        
        res = selector.select_adaptive(
            hidden, logits, protected, attention, training=False, minimum_retention_ratio=0.0, diagnostic_force_all_retain=True
        )
        
        # Max retained sequence dimension should match valid tokens
        self.assertEqual(res.selected_indices.shape[1], 10)
        self.assertEqual(res.actual_retained_counts.item(), 10)
        self.assertEqual(res.actual_valid_counts.item(), 10)

    def test_diagnostic_random_seed_matches_count_and_protects(self):
        selector = TokenSelector(device=torch.device("cpu"))
        hidden = torch.randn(1, 12, 4)
        logits = torch.tensor([[-0.1, -0.1, -0.1, 0.1, 0.1, 0.1, -0.2, 0.2, 0.0, 0.0, 0.5, -0.5]])
        protected = torch.zeros(1, 12, dtype=torch.bool)
        protected[0, 0] = True # e.g. CLS
        attention = torch.ones(1, 12)
        attention[0, 10:] = 0  # 2 padding tokens, so 10 valid
        
        # Production run to get expected count
        res_prod = selector.select_adaptive(
            hidden, logits, protected, attention, training=False, minimum_retention_ratio=0.5
        )
        prod_count = res_prod.actual_retained_counts.item()
        # Random run (pass target_count via diagnostic_target_counts)
        target_counts = torch.tensor([prod_count], dtype=torch.long)
        res_rand = selector.select_adaptive(
            hidden, logits, protected, attention, training=False, minimum_retention_ratio=0.5, 
            diagnostic_random_seed=42, diagnostic_target_counts=target_counts
        )
        rand_count = res_rand.actual_retained_counts.item()
        
        # 1. Matches count exactly
        self.assertEqual(rand_count, prod_count)
        
        # 2. Padding is never selected
        # selected_indices should not contain 10 or 11
        selected = res_rand.selected_indices[0].tolist()
        self.assertNotIn(10, selected[:rand_count])
        self.assertNotIn(11, selected[:rand_count])
        
        # 3. Protected tokens remain protected
        self.assertIn(0, selected[:rand_count])

        # 4. Same seed gives same result
        res_rand2 = selector.select_adaptive(
            hidden, logits, protected, attention, training=False, minimum_retention_ratio=0.5, 
            diagnostic_random_seed=42, diagnostic_target_counts=target_counts
        )
        self.assertEqual(res_rand.selected_indices.tolist(), res_rand2.selected_indices.tolist())
        
        # 5. Different seed gives different result
        res_rand3 = selector.select_adaptive(
            hidden, logits, protected, attention, training=False, minimum_retention_ratio=0.5, 
            diagnostic_random_seed=123, diagnostic_target_counts=target_counts
        )
        self.assertNotEqual(res_rand.selected_indices.tolist(), res_rand3.selected_indices.tolist())

    def test_diagnostic_random_seed_respects_target_count(self):
        selector = TokenSelector()
        hidden_states = torch.randn(2, 5, 4)
        retention_scores = torch.tensor([[10.0, 10.0, -10.0, -10.0, -10.0], [10.0, -10.0, -10.0, -10.0, -10.0]])
        protected_mask = torch.tensor([[True, False, False, False, False], [True, False, False, False, False]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])

        # Test diagnostic_target_counts: force exactly 2 tokens to be kept in example 0, and 3 in example 1
        target_counts = torch.tensor([2, 3])
        
        res = selector.select_adaptive(
            hidden_states=hidden_states,
            retention_scores=retention_scores,
            protected_mask=protected_mask,
            attention_mask=attention_mask,
            training=False,
            threshold_bias=0.0,
            minimum_retention_ratio=0.0,
            diagnostic_random_seed=42,
            diagnostic_target_counts=target_counts
        )

        # Example 0: valid=4. Target=2. Keep exactly 2.
        self.assertEqual(int(res.actual_retained_counts[0].item()), 2)
        # Example 1: valid=5. Target=3. Keep exactly 3.
        self.assertEqual(int(res.actual_retained_counts[1].item()), 3)
        
        # Test that padding is not selected (example 0, index 4)
        self.assertNotIn(4, res.selected_indices[0].tolist())

    def test_multi_budget_pipeline(self):
        # We test the end-to-end multi-budget diagnostic capture
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        try:
            from diagnose_multi_budget_random import evaluate_with_fresh_model
            # Use dummy.pt if checkpoint not found
            ckpt = "dummy.pt"
            if not os.path.exists(ckpt):
                # We need a valid model to test, just create a dummy
                torch.save({}, ckpt)
                
            from unittest.mock import patch
            with patch('diagnose_multi_budget_random.evaluate_squad.load_ammr_checkpoint'):
                res_ammr = evaluate_with_fresh_model(
                checkpoint=ckpt,
                num_examples=2,
                batch_size=2,
                flag_name=None,
                bias=0.0
            )
            
            self.assertIsNotNone(res_ammr.get("target_counts"))
            self.assertGreater(len(res_ammr["target_counts"]), 0)
            
            with patch('diagnose_multi_budget_random.evaluate_squad.load_ammr_checkpoint'):
                res_rand = evaluate_with_fresh_model(
                checkpoint=ckpt,
                num_examples=2,
                batch_size=2,
                flag_name="diagnostic_random_seed",
                bias=0.0,
                seed_val=42,
                target_counts_write=res_ammr["target_counts"]
            )
            
            self.assertEqual(res_ammr["total_selected"], res_rand["total_selected"])
            self.assertEqual(res_rand["missing_records"], 0)
            self.assertEqual(res_rand["extra_records"], 0)
        except Exception as e:
            self.fail(f"Pipeline test failed: {e}")

    def test_forced_all_retain_matches_baseline_first_layer(self):
        # We need to strictly compare layer0 between baseline and forced-all-retain
        # because the user reported MaxDiff = 1.699e+00 at layer0.
        import torch
        from config import MODEL_NAME, DEVICE
        from src.baseline import BaselineQAModel
        from src.models_adaptive import AdaptiveDistilBertQA
        import os
        
        from src.squad_data import get_squad_dataloaders
        _, val_dl, _, _, _ = get_squad_dataloaders(
            batch_size=16,
            max_train_samples=1,
            max_val_samples=16,
        )
        batch = next(iter(val_dl))
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)

        baseline = BaselineQAModel(freeze_parameters=True).to(DEVICE)
        adaptive = AdaptiveDistilBertQA(
            model_name=MODEL_NAME, 
            device=DEVICE,
            apply_retention_per_layer=[True, True, True, True, True, True]
        ).to(DEVICE)
        
        # We MUST ensure the weights match exactly
        adaptive.distilbert.load_state_dict(baseline.model.distilbert.state_dict(), strict=False)
        adaptive.qa_outputs.load_state_dict(baseline.model.qa_outputs.state_dict(), strict=False)

        baseline.eval()
        adaptive.eval()

        with torch.no_grad():
            out_b = baseline.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            baseline_hidden = out_b.hidden_states
            
            start_logits, end_logits, diagnostics = adaptive(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                diagnostic_force_all_retain=True,
                diagnostic_return_layer_states=True
            )
            
        embed_diff = torch.abs(baseline_hidden[0] - diagnostics['embedding_output']).max().item()
        layer0_diff = torch.abs(baseline_hidden[1] - diagnostics['layer_hidden_states'][0]).max().item()
        
        # If there's a divergence, it should fail here and report the value
        self.assertEqual(embed_diff, 0.0, f"First divergence at embedding output: {embed_diff}")
        self.assertEqual(layer0_diff, 0.0, f"First divergence at layer0 output: {layer0_diff}")

    def test_matched_retention_random_seeds(self):
        # Validate that running diagnostic_random_seed with multiple seeds
        # produces deterministic behavior for the same seed, different behavior
        # for different seeds, while exact counts match perfectly.
        import torch
        from config import MODEL_NAME, DEVICE
        from src.models_adaptive import AdaptiveDistilBertQA
        
        torch.manual_seed(42)
        model = AdaptiveDistilBertQA(model_name=MODEL_NAME, device=DEVICE).to(DEVICE)
        model.eval()
        
        batch_size = 2
        seq_len = 64
        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(DEVICE)
        attention_mask = torch.ones((batch_size, seq_len)).to(DEVICE)
        # Pretend it's a squad example with CLS/SEP protected
        input_ids[:, 0] = 101
        input_ids[:, -1] = 102
        
        # 1. AMMR Run (capturing target counts)
        model._diagnostic_target_counts_write = {}
        model._diagnostic_target_counts_read = None
        model._diagnostic_target_counts_consumed = None
        
        with torch.no_grad():
            _, _, metrics = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                diagnostic_batch_id=1,
                training=False,
                threshold_bias=0.0,
                return_layer_metrics=True
            )
            
        target_counts = dict(model._diagnostic_target_counts_write)
        ammr_selections = []
        for result in metrics.get("selection_results", []):
            if result is not None:
                ammr_selections.append(result.selected_indices.clone())
                
        # 2. Random Run Seed A
        seeds = [42, 123, 2026, 7, 19, 37, 101, 256, 512, 999]
        self.assertEqual(len(seeds), 10, "Should evaluate exactly 10 seeds")
        
        first_seed_selections = []
        
        for seed in seeds:
            model._diagnostic_target_counts_write = dict(target_counts)
            model._diagnostic_target_counts_read = dict(target_counts)
            model._diagnostic_target_counts_consumed = set()
            
            with torch.no_grad():
                _, _, rand_metrics = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    diagnostic_batch_id=1,
                    diagnostic_random_seed=seed,
                    training=False,
                    threshold_bias=0.0,
                    return_layer_metrics=True
                )
                
            # Verify exact count match for every layer
            for i, result in enumerate(rand_metrics.get("selection_results", [])):
                if result is not None:
                    ammr_target = target_counts[(1, i)]
                    self.assertTrue(torch.equal(result.actual_retained_counts.cpu(), ammr_target))
            
            # Verify consumption
            self.assertEqual(len(model._diagnostic_target_counts_consumed), len(target_counts))
            self.assertEqual(len(target_counts) - len(model._diagnostic_target_counts_consumed), 0)
            
            rand_selections = [r.selected_indices.clone() for r in rand_metrics.get("selection_results", []) if r is not None]
            
            if seed == seeds[0]:
                first_seed_selections = rand_selections
            elif seed == seeds[1]:
                # Verify different seeds yield different selections
                diff = False
                for r1, r2 in zip(first_seed_selections, rand_selections):
                    if not torch.equal(r1, r2):
                        diff = True
                        break
                self.assertTrue(diff, "Different seeds should produce different selections")
                
        # Verify same seed yields deterministic result
        model._diagnostic_target_counts_write = dict(target_counts)
        model._diagnostic_target_counts_read = dict(target_counts)
        model._diagnostic_target_counts_consumed = set()
        
        with torch.no_grad():
            _, _, rand_metrics_deterministic = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                diagnostic_batch_id=1,
                diagnostic_random_seed=seeds[0],
                training=False,
                threshold_bias=0.0,
                return_layer_metrics=True
            )
            
        deterministic_selections = [r.selected_indices.clone() for r in rand_metrics_deterministic.get("selection_results", []) if r is not None]
        for r1, r2 in zip(first_seed_selections, deterministic_selections):
            self.assertTrue(torch.equal(r1, r2), "Same seed should be deterministic")

    def test_multi_budget_script_configuration(self):
        # Verify the script's configured sweep contains exactly these seven biases
        # without running the expensive evaluation.
        import re
        import os
        
        script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "diagnose_multi_budget_random.py")
        self.assertTrue(os.path.exists(script_path), "diagnose_multi_budget_random.py not found")
        
        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        bias_match = re.search(r"biases\s*=\s*\[(.*?)\]", content)
        self.assertIsNotNone(bias_match, "biases list not found in script")
        
        biases_str = bias_match.group(1)
        # Parse the numbers out, handling negatives
        biases_parsed = [float(b.strip()) for b in biases_str.split(",")]
        
        expected_biases = [1.0, 0.5, 0.2, 0.0, -0.2, -0.4, -0.6]
        self.assertEqual(biases_parsed, expected_biases, "Configured biases do not match expected sweep")

    def test_full_squad_script_configuration(self):
        import re
        import os
        
        script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "diagnose_multi_budget_random_full_squad.py")
        self.assertTrue(os.path.exists(script_path), "diagnose_multi_budget_random_full_squad.py not found")
        
        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        bias_match = re.search(r"default=\[(.*?)\]", content)
        self.assertIsNotNone(bias_match, "biases list not found in full squad script")
        
        biases_str = bias_match.group(1)
        biases_parsed = [float(b.strip()) for b in biases_str.split(",")]
        
        expected_biases = [1.0, 0.5, 0.2, 0.0, -0.2, -0.4, -0.6]
        self.assertEqual(biases_parsed, expected_biases, "Configured biases do not match expected sweep in full squad script")
        
        # Verify it writes to correct csv
        self.assertIn("multi_budget_random_control_full_squad.csv", content)

    def test_threshold_bias_changes_selection_mask(self):
        selector = TokenSelector(device=torch.device("cpu"))
        hidden = torch.randn(1, 10, 4)
        logits = torch.tensor([[-2.0, -1.0, 0.5, 1.5, -0.5, 2.0, -3.0, 0.1, -1.5, 1.0]])
        protected = torch.zeros(1, 10, dtype=torch.bool)
        attention = torch.ones(1, 10)
        
        # Bias = 0.0 -> threshold is 0.0. Logits > 0.0 are kept: idx 2, 3, 5, 7, 9 = 5 tokens
        res_0 = selector.select_adaptive(hidden, logits, protected, attention, training=False, threshold_bias=0.0, minimum_retention_ratio=0.0)
        
        # Bias = -1.0 -> threshold is -1.0. Logits > 1.0 are kept: idx 3, 5 = 2 tokens
        res_neg1 = selector.select_adaptive(hidden, logits, protected, attention, training=False, threshold_bias=-1.0, minimum_retention_ratio=0.0)
        
        # Bias = 1.0 -> threshold is 1.0. Logits > -1.0 are kept: idx 1, 2, 3, 4, 5, 7, 9 = 7 tokens
        res_pos1 = selector.select_adaptive(hidden, logits, protected, attention, training=False, threshold_bias=1.0, minimum_retention_ratio=0.0)
        
        self.assertNotEqual(res_0.num_selected, res_neg1.num_selected)
        self.assertNotEqual(res_0.num_selected, res_pos1.num_selected)

    def test_identical_masks_detected(self):
        selector = TokenSelector(device=torch.device("cpu"))
        hidden = torch.randn(1, 10, 4)
        # All logits highly positive
        logits = torch.tensor([[10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]])
        protected = torch.zeros(1, 10, dtype=torch.bool)
        attention = torch.ones(1, 10)
        
        res_a = selector.select_adaptive(hidden, logits, protected, attention, training=False, threshold_bias=0.0, minimum_retention_ratio=0.0)
        res_b = selector.select_adaptive(hidden, logits, protected, attention, training=False, threshold_bias=-1.0, minimum_retention_ratio=0.0)
        
        # Should be identical because logits are huge
        self.assertTrue(torch.equal(res_a.selected_indices, res_b.selected_indices))

if __name__ == "__main__":
    unittest.main()
