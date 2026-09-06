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

if __name__ == "__main__":
    unittest.main()
