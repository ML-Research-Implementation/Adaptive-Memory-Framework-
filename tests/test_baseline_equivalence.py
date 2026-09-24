import unittest
import torch
import os
from config import MODEL_NAME, DEVICE
from src.baseline import BaselineQAModel
from src.models_adaptive import AdaptiveDistilBertQA
import evaluate_squad

class TestBaselineEquivalence(unittest.TestCase):
    def test_baseline_vs_forced_all_retain(self):
        # We test that AdaptiveDistilBertQA with forced-all-retain produces the exact same hidden states
        # and logits as BaselineQAModel, given the same weights and inputs.
        
        batch_size = 2
        seq_len = 16
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(DEVICE)
        # Add CLS and SEP
        input_ids[:, 0] = 101
        input_ids[:, -1] = 102
        
        attention_mask = torch.ones(batch_size, seq_len).to(DEVICE)
        
        baseline = BaselineQAModel(freeze_parameters=True).to(DEVICE)
        baseline.eval()
        
        adaptive = AdaptiveDistilBertQA(model_name=MODEL_NAME, device=DEVICE).to(DEVICE)
        # Copy exact weights from baseline to adaptive so they are identical
        adaptive.distilbert.load_state_dict(baseline.model.distilbert.state_dict(), strict=False)
        adaptive.qa_outputs.load_state_dict(baseline.model.qa_outputs.state_dict(), strict=False)
        adaptive.eval()
        
        with torch.no_grad():
            out_b = baseline.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            baseline_start = out_b.start_logits
            baseline_end = out_b.end_logits
            baseline_hidden = out_b.hidden_states
            
            start_logits, end_logits, diagnostics = adaptive(
                input_ids=input_ids,
                attention_mask=attention_mask,
                diagnostic_force_all_retain=True,
                diagnostic_return_layer_states=True
            )
            
        # Check logits
        diff_start = (baseline_start - start_logits).abs().max().item()
        diff_end = (baseline_end - end_logits).abs().max().item()
        
        self.assertLess(diff_start, 1e-4, f"Start logits differ by {diff_start}")
        self.assertLess(diff_end, 1e-4, f"End logits differ by {diff_end}")
        
        # Check embeddings
        diff_emb = (baseline_hidden[0] - diagnostics["embedding_output"]).abs().max().item()
        self.assertLess(diff_emb, 1e-4, f"Embeddings differ by {diff_emb}")
        
        # Check each layer
        for i in range(6):
            diff_h = (baseline_hidden[i+1] - diagnostics["layer_hidden_states"][i]).abs().max().item()
            self.assertLess(diff_h, 1e-4, f"Layer {i+1} hidden states differ by {diff_h}")

if __name__ == "__main__":
    unittest.main()
