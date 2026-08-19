import torch
from transformers import DistilBertForQuestionAnswering

model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')
qa_outputs = model.qa_outputs

# Create dummy hidden states
hidden_states = torch.randn(1, 10, 768)

# Test what qa_outputs returns
result = qa_outputs(hidden_states)
print(f"Type: {type(result)}")
print(f"Result: {result}")

if isinstance(result, torch.Tensor):
    print(f"Shape: {result.shape}")
elif isinstance(result, tuple):
    print(f"Tuple length: {len(result)}")
    for i, r in enumerate(result):
        print(f"  Element {i}: type={type(r)}, shape={r.shape if hasattr(r, 'shape') else 'N/A'}")
