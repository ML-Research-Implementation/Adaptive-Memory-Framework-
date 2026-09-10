import unittest
from unittest.mock import patch
import torch
import torch.nn as nn
from src.baseline import BaselineQAModel
class Output:
    pass
class FakeQA(nn.Module):
    def __init__(self):
        super().__init__(); self.weight=nn.Parameter(torch.ones(1)); self.distilbert=nn.Identity(); self.qa_outputs=nn.Identity()
    def forward(self,input_ids=None,attention_mask=None,**kwargs):
        out=Output(); out.start_logits=self.weight.expand(input_ids.size(0),input_ids.size(1)); out.end_logits=out.start_logits+1; return out
class TestBaseline(unittest.TestCase):
    @patch("src.baseline.DistilBertForQuestionAnswering.from_pretrained")
    def test_module_eval_forward_and_frozen_parameters(self, loader):
        loader.return_value=FakeQA(); baseline=BaselineQAModel(freeze_parameters=True); self.assertIsInstance(baseline,nn.Module); self.assertFalse(baseline.training); self.assertTrue(all(not p.requires_grad for p in baseline.parameters())); output=baseline(torch.ones(2,4,dtype=torch.long),torch.ones(2,4)); self.assertEqual(output.start_logits.shape,(2,4)); baseline.train(); self.assertTrue(baseline.training); baseline.eval(); self.assertFalse(baseline.training)
if __name__=="__main__": unittest.main()
