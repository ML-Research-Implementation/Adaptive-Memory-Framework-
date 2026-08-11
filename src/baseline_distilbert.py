import torch
from transformers import (
    AutoTokenizer,
    DistilBertForQuestionAnswering
)

MODEL_NAME = "distilbert-base-uncased-distilled-squad"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("DEVICE")
print("=" * 70)

print(device)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


qa_model = DistilBertForQuestionAnswering.from_pretrained(
    MODEL_NAME
)

qa_model = qa_model.to(device)

qa_model.eval()




encoder = qa_model.distilbert


question = "What is artificial intelligence?"

context = """
Artificial intelligence is a field of computer science
that focuses on creating systems capable of performing
tasks that normally require human intelligence.
"""


inputs = tokenizer(
    question,
    context,
    return_tensors="pt"
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}


print("\n" + "=" * 70)
print("TOKENIZATION")
print("=" * 70)

tokens = tokenizer.convert_ids_to_tokens(
    inputs["input_ids"][0]
)

print("Tokens:")
print(tokens)

print("\nInput IDs shape:")
print(inputs["input_ids"].shape)


with torch.no_grad():

    encoder_outputs = encoder(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )


hidden_states = encoder_outputs.last_hidden_state


print("\n" + "=" * 70)
print("DISTILBERT ENCODER")
print("=" * 70)

print(
    "Hidden state shape:",
    hidden_states.shape
)


batch_size = hidden_states.shape[0]
sequence_length = hidden_states.shape[1]
hidden_dimension = hidden_states.shape[2]

print("\nBatch size:", batch_size)
print("Sequence length:", sequence_length)
print("Hidden dimension:", hidden_dimension)


with torch.no_grad():

    encoder_outputs = encoder(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        output_hidden_states=True
    )


all_hidden_states = encoder_outputs.hidden_states


print("\n" + "=" * 70)
print("DISTILBERT LAYER REPRESENTATIONS")
print("=" * 70)

print(
    "Number of hidden-state tensors:",
    len(all_hidden_states)
)

for layer_index, hidden in enumerate(all_hidden_states):

    print(
        f"Layer {layer_index}: {hidden.shape}"
    )


with torch.no_grad():

    outputs = qa_model(
        **inputs
    )


start_logits = outputs.start_logits
end_logits = outputs.end_logits


print("\n" + "=" * 70)
print("QUESTION ANSWERING")
print("=" * 70)

print(
    "Start logits shape:",
    start_logits.shape
)

print(
    "End logits shape:",
    end_logits.shape
)


start_index = torch.argmax(
    start_logits,
    dim=-1
).item()

end_index = torch.argmax(
    end_logits,
    dim=-1
).item()


print(
    "\nPredicted start index:",
    start_index
)

print(
    "Predicted end index:",
    end_index
)




answer_tokens = inputs["input_ids"][
    0,
    start_index:end_index + 1
]

answer = tokenizer.decode(
    answer_tokens,
    skip_special_tokens=True
)


print("\n" + "=" * 70)
print("PREDICTED ANSWER")
print("=" * 70)

print(answer)