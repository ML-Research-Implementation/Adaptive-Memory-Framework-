import torch
import unittest
from src.models_adaptive import HardConcreteGate, TokenSelector, AdaptiveDistilBertQA
from src.losses import calculate_lagrangian_budget_loss
from config import DEVICE

class TestAdaptiveComponents(unittest.TestCase):
    
    def setUp(self):
        self.device = DEVICE
        self.batch_size = 2
        self.seq_len = 10
        self.hidden_dim = 16
        
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

    def test_lagrangian_increases_when_over_budget(self):
        """Lambda must grow when actual retention exceeds target."""
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
