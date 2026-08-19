import inspect
from transformers import DistilBertForQuestionAnswering

model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')
layer = model.distilbert.transformer.layer[0]

# Print forward signature
sig = inspect.signature(layer.forward)
print("TransformerBlock forward signature:")
print(f"Parameters: {list(sig.parameters.keys())}")
print(f"\nFull signature:")
print(sig)
