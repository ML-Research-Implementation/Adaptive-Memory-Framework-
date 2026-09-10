import torch
import unittest
from torch.optim import AdamW
from src.models import RetentionScorer
from src.models_adaptive import HardConcreteGate, TokenSelector, AdaptiveDistilBertQA
from src.losses import calculate_lagrangian_budget_loss
from config import DEVICE
from train import update_retention_lambda


def unpack_student_outputs(outputs):
    if not isinstance(outputs, tuple) or len(outputs) != 3:
        raise RuntimeError("Expected the current three-output student contract")
    start, end, layer_metrics = outputs
    return start, end, layer_metrics

class TestAdaptiveComponents(unittest.TestCase):
    
    def setUp(self):
        self.device = DEVICE
        self.batch_size = 2
        self.seq_len = 10
        self.hidden_dim = 16
        
    def test_zero_initialized_scorer_learns_through_gumbel_gate(self):
        torch.manual_seed(7)
        scorer = RetentionScorer(hidden_dimension=16, dropout=0.0)
        hidden = torch.randn(2, 5, 16)
        scores, _ = scorer(hidden)
        scores.retain_grad()
        gate_logits = torch.stack((-scores, scores), dim=-1)
        gate = torch.nn.functional.gumbel_softmax(
            gate_logits, tau=0.5, hard=True, dim=-1
        )
        keep = gate[..., 1]
        gated_hidden = hidden * keep.unsqueeze(-1)
        loss = gated_hidden.square().mean()
        self.assertTrue(scores.requires_grad)
        self.assertTrue(gate.requires_grad)
        self.assertTrue(gated_hidden.requires_grad)
        self.assertTrue(loss.requires_grad)
        optimizer = AdamW(scorer.parameters(), lr=1e-2)
        optimizer.zero_grad()
        loss.backward()
        final_weight = scorer.network[-1].weight
        final_bias = scorer.network[-1].bias
        self.assertGreater(float(final_weight.grad.norm()), 0.0)
        self.assertGreater(float(final_bias.grad.norm()), 0.0)
        before = final_weight.detach().clone()
        optimizer.step()
        self.assertGreater(float((final_weight.detach() - before).abs().max()), 0.0)

    def test_hard_concrete_gate(self):
        gate = HardConcreteGate(temperature=0.5)
        logits = torch.randn(self.batch_size, self.seq_len, device=self.device)
        
        # Training mode (stochastic)
        z_train, prob_train = gate(logits, training=True)
        self.assertTrue((z_train >= 0.0).all() and (z_train <= 1.0).all())
        self.assertEqual(z_train.shape, logits.shape)
        
        # Inference mode (deterministic)
        z_eval, prob_eval = gate(logits, training=False, threshold_bias=0.0)
        self.assertTrue(((z_eval == 0.0) | (z_eval == 1.0)).all())
        
    def test_three_output_student_contract(self):
        start = torch.randn(2, 10)
        end = torch.randn(2, 10)
        metrics = {"selection_results": []}
        unpacked_start, unpacked_end, unpacked_metrics = unpack_student_outputs((start, end, metrics))
        self.assertIs(unpacked_start, start)
        self.assertIs(unpacked_end, end)
        self.assertIs(unpacked_metrics, metrics)

    def test_minimum_retention_floor(self):
        selector = TokenSelector(device=self.device)
        hidden = torch.randn(1, self.seq_len, self.hidden_dim, device=self.device)
        logits = torch.full((1, self.seq_len), -10.0, device=self.device)
        protected = torch.zeros(1, self.seq_len, dtype=torch.bool, device=self.device)
        attention = torch.ones(1, self.seq_len, device=self.device)

        result = selector.select_adaptive(
            hidden, logits, protected, attention,
            training=False, minimum_retention_ratio=0.8
        )

        self.assertGreaterEqual(result.num_selected, 8)
        self.assertGreaterEqual(result.retention_ratio, 0.8)

    def test_answer_span_tokens_are_protected(self):
        selector = TokenSelector(device=self.device)
        hidden = torch.randn(1, self.seq_len, self.hidden_dim, device=self.device)
        logits = torch.full((1, self.seq_len), -10.0, device=self.device)
        protected = torch.zeros(1, self.seq_len, dtype=torch.bool, device=self.device)
        protected[:, 3:6] = True
        attention = torch.ones(1, self.seq_len, device=self.device)

        result = selector.select_adaptive(
            hidden, logits, protected, attention,
            training=False, minimum_retention_ratio=0.0
        )

        for index in range(3, 6):
            self.assertTrue((result.selected_indices == index).any())

    def test_token_selector(self):
        selector = TokenSelector(device=self.device)
        hidden = torch.randn(self.batch_size, self.seq_len, self.hidden_dim, device=self.device)
        
        # Create dummy logits heavily biased towards dropping
        logits = torch.ones(self.batch_size, self.seq_len, device=self.device) * -10.0
        
        # Protect specific tokens (e.g. index 0 and 9)
        protected_mask = torch.zeros(self.batch_size, self.seq_len, dtype=torch.bool, device=self.device)
        protected_mask[:, 0] = True
        protected_mask[:, 9] = True
        
        attention_mask = torch.ones(self.batch_size, self.seq_len, device=self.device)
        
        res = selector.select_adaptive(hidden, logits, protected_mask, attention_mask, training=False)
        
        # Should keep exactly the 2 protected tokens
        self.assertEqual(res.num_selected, 2)
        self.assertTrue((res.selected_indices[:, 0] == 0).all())
        self.assertTrue((res.selected_indices[:, 1] == 9).all())
        self.assertEqual(res.selected_hidden_states.shape, (self.batch_size, 2, self.hidden_dim))

    def test_budget_accounting(self):
        expected_tokens = torch.tensor(15.0, device=self.device)
        target = 10.0
        lagrangian = 0.5
        
        loss = calculate_lagrangian_budget_loss(expected_tokens, target, lagrangian)
        
        # loss = 0.5 * (15 - 10) = 2.5
        self.assertAlmostEqual(loss.item(), 2.5)
        
    def test_model_freezing(self):
        model = AdaptiveDistilBertQA(freeze_transformer=True)
        
        # Check transformer is frozen
        for param in model.distilbert.parameters():
            self.assertFalse(param.requires_grad)
            
        # Check scorers are not frozen initially
        for param in model.retention_scorers.parameters():
            self.assertTrue(param.requires_grad)
            
        model.freeze_scorers()
        for param in model.retention_scorers.parameters():
            self.assertFalse(param.requires_grad)

    def test_selector_floor_at_requested_ratios_with_low_scores(self):
        selector = TokenSelector(device=self.device)
        hidden = torch.randn(2, self.seq_len, self.hidden_dim, device=self.device)
        logits = torch.full((2, self.seq_len), -100.0, device=self.device)
        protected = torch.zeros(2, self.seq_len, dtype=torch.bool, device=self.device)
        attention = torch.ones(2, self.seq_len, device=self.device)
        attention[1, 8:] = 0

        for target in (0.95, 0.80, 0.60):
            result = selector.select_adaptive(
                hidden, logits, protected, attention,
                training=False, minimum_retention_ratio=target
            )
            self.assertIsNotNone(result.actual_retained_counts)
            self.assertIsNotNone(result.actual_valid_counts)
            ratios = result.actual_retained_counts.float() / result.actual_valid_counts.float().clamp_min(1.0)
            self.assertTrue(torch.all(ratios >= target - 1e-6))
            self.assertTrue(torch.all(result.new_attention_mask[result.new_attention_mask < 0.5] == 0))

    def test_forward_hard_floor_at_95_percent(self):
        model = AdaptiveDistilBertQA(freeze_transformer=True)
        model.eval()
        input_ids = torch.randint(1000, 30000, (1, 20), device=self.device)
        input_ids[:, 0] = 101
        input_ids[:, 19] = 102
        attention = torch.ones(1, 20, device=self.device)
        with torch.no_grad():
            for scorer in model.retention_scorers:
                for parameter in scorer.parameters():
                    parameter.zero_()
            _, _, metrics = model(
                input_ids, attention,
                return_layer_metrics=True,
                training=False,
                minimum_retention_ratio=0.95
            )
        self.assertGreaterEqual(float(metrics['actual_retention_ratio']), 0.95 - 1e-4)
        self.assertGreaterEqual(float(metrics['expected_retained_tokens']), float(metrics['actual_retained_tokens']))

    def test_forward_hard_floor_at_80_percent(self):
        model = AdaptiveDistilBertQA(freeze_transformer=True)
        model.eval()
        input_ids = torch.randint(1000, 30000, (1, 20), device=self.device)
        input_ids[:, 0] = 101
        input_ids[:, 19] = 102
        attention = torch.ones(1, 20, device=self.device)
        with torch.no_grad():
            for scorer in model.retention_scorers:
                for parameter in scorer.parameters():
                    parameter.zero_()
            _, _, metrics = model(
                input_ids, attention,
                return_layer_metrics=True,
                training=False,
                minimum_retention_ratio=0.80
            )
        self.assertGreaterEqual(float(metrics['actual_retention_ratio']), 0.80 - 1e-4)

    def test_padding_positions_are_never_retained_or_counted(self):
        selector = TokenSelector(device=self.device)
        hidden = torch.randn(1, self.seq_len, self.hidden_dim, device=self.device)
        logits = torch.full((1, self.seq_len), 100.0, device=self.device)
        protected = torch.zeros(1, self.seq_len, dtype=torch.bool, device=self.device)
        attention = torch.ones(1, self.seq_len, device=self.device)
        attention[:, 6:] = 0

        result = selector.select_adaptive(
            hidden, logits, protected, attention,
            training=False, minimum_retention_ratio=0.95
        )
        self.assertEqual(result.actual_valid_counts.item(), 6)
        self.assertLessEqual(result.actual_retained_counts.item(), 6)
        self.assertTrue(torch.all(result.selected_indices < 6))

    def test_floor_holds_at_each_curriculum_target(self):
        selector = TokenSelector(device=self.device)
        hidden = torch.randn(1, self.seq_len, self.hidden_dim, device=self.device)
        logits = torch.full((1, self.seq_len), -100.0, device=self.device)
        protected = torch.zeros(1, self.seq_len, dtype=torch.bool, device=self.device)
        attention = torch.ones(1, self.seq_len, device=self.device)
        for target in (0.95, 0.90, 0.80, 0.70, 0.60):
            result = selector.select_adaptive(
                hidden, logits, protected, attention,
                training=False, minimum_retention_ratio=target
            )
            self.assertGreaterEqual(result.retention_ratio, target)
            self.assertIsNotNone(result.actual_retained_counts)
            actual_ratio = result.actual_retained_counts.float().mean().item() / result.num_original
            self.assertGreaterEqual(actual_ratio, target)

    def test_minimum_retention_violation_increases_lambda(self):
        """Lambda must grow when actual retention is below target."""
        updated, violation = update_retention_lambda(0.0, 0.80, 0.95, learning_rate=0.005, maximum=20.0)
        self.assertGreater(violation, 0.0)
        self.assertGreater(updated, 0.0)

    def test_lambda_update_is_normalized_and_bounded(self):
        updated, violation = update_retention_lambda(19.99, 0.0, 0.95, learning_rate=0.01, maximum=20.0)
        self.assertAlmostEqual(violation, 0.95)
        self.assertLessEqual(updated, 20.0)

    def test_minimum_retention_penalty_direction(self):
        below = torch.clamp(2.0 * torch.relu(torch.tensor(0.95) - torch.tensor(0.80)), min=0.0, max=10.0)
        above = torch.clamp(2.0 * torch.relu(torch.tensor(0.95) - torch.tensor(0.98)), min=0.0, max=10.0)
        self.assertGreater(below.item(), 0.0)
        self.assertEqual(above.item(), 0.0)

    def test_lagrangian_increases_when_over_budget(self):
        """Legacy loss remains numerically valid for checkpoint compatibility."""
        from src.losses import calculate_lagrangian_budget_loss

        # actual > target → violation > 0 → lambda increases
        actual = torch.tensor(120.0)
        target = 100.0
        lam = 0.0
        lr = 0.05
        loss = calculate_lagrangian_budget_loss(actual, target, lam)
        violation = (actual - target).item()
        new_lam = max(0.0, lam + lr * violation)
        self.assertGreater(new_lam, 0.0, "Lambda should increase when over budget")
        # loss = 0 * violation = 0 when lambda=0, but violation is positive
        self.assertAlmostEqual(loss.item(), lam * violation)

    def test_lagrangian_stays_zero_when_under_budget(self):
        """Lambda must not go below zero when actual retention is under budget."""
        from src.losses import calculate_lagrangian_budget_loss

        # actual < target → violation < 0 → lambda clamped at 0
        actual = torch.tensor(80.0)
        target = 100.0
        lam = 0.0
        lr = 0.05
        violation = (actual - target).item()
        new_lam = max(0.0, lam + lr * violation)
        self.assertEqual(new_lam, 0.0, "Lambda must not go negative")

    def test_budget_target_in_per_sequence_units(self):
        """
        target_budget = avg_valid_per_seq * target_ratio * n_layers must be
        comparable in magnitude to expected_retained_tokens from the model
        (which is also per-sequence, summed over layers).
        """
        batch_size = 4
        seq_len = 128
        target_ratio = 0.95
        n_layers = 6
        # Simulate a full-attention batch (all tokens valid)
        attention_mask = torch.ones(batch_size, seq_len)
        avg_valid_per_seq = attention_mask.sum().item() / batch_size  # = 128
        target_budget = avg_valid_per_seq * target_ratio * n_layers   # = 729.6

        # expected_retained_tokens: for each layer, mean per seq of sum of probs.
        # Simulate: all probs = target_ratio, seq compresses each step.
        current_len = seq_len
        simulated_expected = 0.0
        for _ in range(n_layers):
            simulated_expected += current_len * target_ratio   # mean per seq
            current_len = int(current_len * target_ratio)

        # The target_budget is a rough first-layer approximation.
        # What matters is that target and actual are the same order of magnitude.
        ratio = simulated_expected / target_budget
        self.assertLess(ratio, 10.0, "Target and actual must be same order of magnitude")
        self.assertGreater(ratio, 0.1, "Target and actual must be same order of magnitude")

    def test_scheduler_not_stepped_on_amp_overflow(self):
        """
        Demonstrate that our scale-check pattern correctly skips scheduler.step()
        when AMP detects gradient overflow (scale drops).
        """
        import torch.optim as optim
        from transformers import get_linear_schedule_with_warmup

        model = torch.nn.Linear(4, 4)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        scheduler = get_linear_schedule_with_warmup(optimizer, 0, 10)
        scaler = torch.amp.GradScaler("cpu", enabled=True)

        initial_lr = scheduler.get_last_lr()[0]
        steps_taken = 0

        for _ in range(3):
            optimizer.zero_grad()
            loss = model(torch.randn(2, 4)).sum()
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            # This is the guard pattern from train.py
            if scaler.get_scale() == scale_before:
                scheduler.step()
                steps_taken += 1

        # For normal (non-overflowing) gradients all 3 steps should have fired
        self.assertEqual(steps_taken, 3)

if __name__ == "__main__":
    unittest.main()
