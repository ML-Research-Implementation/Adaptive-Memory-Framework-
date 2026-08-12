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


# Get the DistilBERT encoder inside the QA model

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
print("STEP 5: TOKENIZATION")
print("=" * 70)

tokens = tokenizer.convert_ids_to_tokens(
    inputs["input_ids"][0]
)

print("Tokens:")
print(tokens)

print("\nInput IDs:")
print(inputs["input_ids"])

print("\nInput IDs shape:")
print(inputs["input_ids"].shape)


with torch.no_grad():

    encoder_outputs = encoder(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )


hidden_states = encoder_outputs.last_hidden_state


print("\n" + "=" * 70)
print("STEP 6: DISTILBERT ENCODER")
print("=" * 70)

print(
    "Final hidden state shape:",
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
print("STEP 7: DISTILBERT LAYER REPRESENTATIONS")
print("=" * 70)

print(
    "Number of hidden-state tensors:",
    len(all_hidden_states)
)


for layer_index, layer_hidden in enumerate(all_hidden_states):

    print(
        f"Layer {layer_index}: "
        f"{layer_hidden.shape}"
    )



with torch.no_grad():

    outputs = qa_model(
        **inputs
    )


start_logits = outputs.start_logits
end_logits = outputs.end_logits


print("\n" + "=" * 70)
print("STEP 8: QUESTION ANSWERING")
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
print("STEP 10: PREDICTED ANSWER")
print("=" * 70)

print(answer)



print("\n" + "=" * 70)
print("STEP 11: TOKEN IMPORTANCE ANALYSIS")
print("=" * 70)


# We use the final Transformer layer.

final_layer = all_hidden_states[-1]


print(
    "Final layer shape:",
    final_layer.shape
)


token_importance = torch.norm(
    final_layer,
    p=2,
    dim=-1
)


print(
    "\nToken importance shape:",
    token_importance.shape
)


print("\n" + "=" * 70)
print("STEP 12: TOKEN IMPORTANCE SCORES")
print("=" * 70)


importance_values = (
    token_importance[0]
    .cpu()
    .tolist()
)


for index, (token, score) in enumerate(
    zip(tokens, importance_values)
):

    print(
        f"{index:2d} | "
        f"{token:20s} | "
        f"Importance = {score:.4f}"
    )



print("\n" + "=" * 70)
print("STEP 13: TOKENS RANKED BY IMPORTANCE")
print("=" * 70)


# Sort token positions from highest importance
# to lowest importance.

sorted_indices = torch.argsort(
    token_importance[0],
    descending=True
)


for rank, token_index in enumerate(
    sorted_indices.tolist()
):

    token = tokens[token_index]

    score = importance_values[token_index]

    print(
        f"{rank + 1:2d} | "
        f"Position = {token_index:2d} | "
        f"Token = {token:20s} | "
        f"Score = {score:.4f}"
    )


print("\n" + "=" * 70)
print("STEP 14: TOP-K TOKEN RETENTION")
print("=" * 70)


# We will keep 50% of the tokens.

retention_ratio = 0.50


num_tokens = sequence_length


num_keep = max(
    1,
    int(num_tokens * retention_ratio)
)


print(
    "Original number of tokens:",
    num_tokens
)

print(
    "Retention ratio:",
    retention_ratio
)

print(
    "Number of tokens to keep:",
    num_keep
)


# Get the most important K tokens.

top_k_indices = sorted_indices[:num_keep]


print("\nSelected token positions:")

print(
    top_k_indices.tolist()
)


print("\nSelected tokens:")


for token_index in top_k_indices.tolist():

    print(
        f"Position {token_index:2d}: "
        f"{tokens[token_index]}"
    )

print("\n" + "=" * 70)
print("STEP 15: RETENTION MASK")
print("=" * 70)


retention_mask = torch.zeros(
    sequence_length,
    dtype=torch.bool
)


retention_mask[top_k_indices] = True


for index, token in enumerate(tokens):

    if retention_mask[index]:

        status = "KEEP"

    else:

        status = "DROP"


    print(
        f"{index:2d} | "
        f"{token:20s} | "
        f"{status}"
    )


print("\n" + "=" * 70)
print("STEP 16: RETAINED HIDDEN REPRESENTATIONS")
print("=" * 70)


# final_layer shape:
#
# [batch_size, sequence_length, hidden_dimension]
#
# Example:
#
# [1, 31, 768]


selected_hidden_states = final_layer[
    0,
    retention_mask
]


print(
    "Original hidden state shape:",
    final_layer.shape
)


print(
    "Retained hidden state shape:",
    selected_hidden_states.shape
)


print(
    "\nOriginal tokens:",
    sequence_length
)


print(
    "Retained tokens:",
    selected_hidden_states.shape[0]
)


print(
    "Removed tokens:",
    sequence_length -
    selected_hidden_states.shape[0]
)



print("\n" + "=" * 70)
print("EXPERIMENT SUMMARY")
print("=" * 70)


print(
    "Model:",
    MODEL_NAME
)

print(
    "Device:",
    device
)

print(
    "Original tokens:",
    sequence_length
)

print(
    "Retention ratio:",
    retention_ratio
)

print(
    "Retained tokens:",
    selected_hidden_states.shape[0]
)

print(
    "Removed tokens:",
    sequence_length -
    selected_hidden_states.shape[0]
)

print(
    "Hidden dimension:",
    hidden_dimension
)

print(
    "Predicted answer:",
    answer
)

print("=" * 70)