 import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    DistilBertForQuestionAnswering
)




MODEL_NAME = "distilbert-base-uncased-distilled-squad"

RETENTION_RATIO = 0.50

LEARNING_RATE = 1e-3

TRAINING_STEPS = 200

BUDGET_LAMBDA = 0.10

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)



print("=" * 75)
print("STEP 1: DEVICE")
print("=" * 75)

print("Device:", device)


 

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


 

qa_model = DistilBertForQuestionAnswering.from_pretrained(
    MODEL_NAME
)

qa_model = qa_model.to(device)

qa_model.eval()


 
for parameter in qa_model.parameters():
    parameter.requires_grad = False


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


input_ids = inputs["input_ids"]

attention_mask = inputs["attention_mask"]

tokens = tokenizer.convert_ids_to_tokens(
    input_ids[0]
)

sequence_length = input_ids.shape[1]


print("\n" + "=" * 75)
print("STEP 5: TOKENIZATION")
print("=" * 75)

for index, token in enumerate(tokens):

    print(
        f"{index:2d} -> {token}"
    )

print("\nInput shape:")
print(input_ids.shape)


 

with torch.no_grad():

    encoder_outputs = encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True
    )


hidden_states = encoder_outputs.last_hidden_state

all_hidden_states = encoder_outputs.hidden_states


print("\n" + "=" * 75)
print("STEP 6: DISTILBERT REPRESENTATIONS")
print("=" * 75)

print(
    "Final hidden state:",
    hidden_states.shape
)

print(
    "Number of hidden-state tensors:",
    len(all_hidden_states)
)

for layer_index, layer_hidden in enumerate(
    all_hidden_states
):

    print(
        f"Layer {layer_index}: "
        f"{layer_hidden.shape}"
    )


 

with torch.no_grad():

    baseline_outputs = qa_model(
        **inputs
    )


baseline_start_logits = (
    baseline_outputs.start_logits
)

baseline_end_logits = (
    baseline_outputs.end_logits
)


baseline_start = torch.argmax(
    baseline_start_logits,
    dim=-1
).item()

baseline_end = torch.argmax(
    baseline_end_logits,
    dim=-1
).item()


baseline_answer = tokenizer.decode(
    input_ids[
        0,
        baseline_start:baseline_end + 1
    ],
    skip_special_tokens=True
)


print("\n" + "=" * 75)
print("STEP 7: BASELINE QA")
print("=" * 75)

print("Start index:", baseline_start)

print("End index:", baseline_end)

print("Answer:", baseline_answer)


 


answer_text = "a field of computer science"


 
offset_inputs = tokenizer(
    question,
    context,
    return_offsets_mapping=True,
    return_tensors="pt"
)


offset_mapping = offset_inputs[
    "offset_mapping"
][0]


 

answer_start_char = context.find(
    answer_text
)

if answer_start_char == -1:

    raise ValueError(
        "Answer text was not found inside the context."
    )


answer_end_char = (
    answer_start_char
    +
    len(answer_text)
)


 
answer_token_positions = []

for index, (start, end) in enumerate(
    offset_mapping.tolist()
):

    if (
        start >= answer_start_char
        and end <= answer_end_char
        and end > start
    ):

        answer_token_positions.append(
            index
        )


if len(answer_token_positions) == 0:

    raise ValueError(
        "Could not identify answer token positions."
    )


start_target_index = (
    answer_token_positions[0]
)

end_target_index = (
    answer_token_positions[-1]
)


print("\n" + "=" * 75)
print("STEP 8: GROUND-TRUTH ANSWER")
print("=" * 75)

print(
    "Answer:",
    answer_text
)

print(
    "Start token:",
    start_target_index,
    tokens[start_target_index]
)

print(
    "End token:",
    end_target_index,
    tokens[end_target_index]
)


start_target = torch.tensor(
    [start_target_index],
    dtype=torch.long,
    device=device
)

end_target = torch.tensor(
    [end_target_index],
    dtype=torch.long,
    device=device
)


 

    """
    Learnable token-retention scorer.

    Input:

        h_t ∈ R^768

    Output:

        s_t ∈ R

    Then:

        p_t = sigmoid(s_t)

    where p_t is the learned retention probability.
    """

    def __init__(self, hidden_dimension):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                hidden_dimension,
                hidden_dimension // 4
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dimension // 4,
                1
            )
        )


    def forward(self, hidden_states):

        scores = self.network(
            hidden_states
        )

        scores = scores.squeeze(-1)

        probabilities = torch.sigmoid(
            scores
        )

        return scores, probabilities


 

hidden_dimension = hidden_states.shape[-1]

retention_scorer = RetentionScorer(
    hidden_dimension
).to(device)


print("\n" + "=" * 75)
print("STEP 10: RETENTION SCORER")
print("=" * 75)

print(
    retention_scorer
)


 

target_tokens = max(
    1,
    int(sequence_length * RETENTION_RATIO)
)


print("\n" + "=" * 75)
print("STEP 11: MEMORY BUDGET")
print("=" * 75)

print(
    "Original tokens:",
    sequence_length
)

print(
    "Retention ratio:",
    RETENTION_RATIO
)

print(
    "Target tokens:",
    target_tokens
)


 
def calculate_budget_loss(
    probabilities,
    target_tokens
):

     

    expected_tokens = probabilities.sum(
        dim=-1
    )


    
    excess = F.relu(
        expected_tokens - target_tokens
    )


    budget_loss = (
        excess ** 2
    ).mean()


    return (
        budget_loss,
        expected_tokens
    )


 

def apply_soft_gate(
    hidden_states,
    probabilities
):

    # p_t * h_t

    gate = probabilities.unsqueeze(-1)

    gated_hidden_states = (
        hidden_states * gate
    )

    return gated_hidden_states


 
def calculate_task_loss(
    gated_hidden_states,
    start_target,
    end_target
):

    

    logits = qa_model.qa_classifier(
        gated_hidden_states
    )


    start_logits = logits[..., 0]

    end_logits = logits[..., 1]


    start_loss = F.cross_entropy(
        start_logits,
        start_target
    )


    end_loss = F.cross_entropy(
        end_logits,
        end_target
    )


    task_loss = (
        start_loss + end_loss
    ) / 2


    return (
        task_loss,
        start_logits,
        end_logits
    )


 
optimizer = torch.optim.AdamW(
    retention_scorer.parameters(),
    lr=LEARNING_RATE
)


 

retention_scorer.eval()

with torch.no_grad():

    initial_scores, initial_probabilities = (
        retention_scorer(
            hidden_states
        )
    )


print("\n" + "=" * 75)
print("STEP 16: INITIAL RETENTION PROBABILITIES")
print("=" * 75)

for index, token in enumerate(tokens):

    probability = (
        initial_probabilities[0, index]
        .item()
    )

    print(
        f"{index:2d} | "
        f"{token:20s} | "
        f"{probability:.4f}"
    )


 
print("\n" + "=" * 75)
print("STEP 17: TRAINING")
print("=" * 75)


retention_scorer.train()


for step in range(
    1,
    TRAINING_STEPS + 1
):

    optimizer.zero_grad()


    
    scores, probabilities = (
        retention_scorer(
            hidden_states
        )
    )


     

    gated_hidden_states = (
        apply_soft_gate(
            hidden_states,
            probabilities
        )
    )


    
    (
        task_loss,
        start_logits,
        end_logits
    ) = calculate_task_loss(
        gated_hidden_states,
        start_target,
        end_target
    )


 

    (
        budget_loss,
        expected_tokens
    ) = calculate_budget_loss(
        probabilities,
        target_tokens
    )


    

    total_loss = (
        task_loss
        +
        BUDGET_LAMBDA * budget_loss
    )


     

    total_loss.backward()


    

    torch.nn.utils.clip_grad_norm_(
        retention_scorer.parameters(),
        max_norm=1.0
    )


    optimizer.step()


    
    if (
        step == 1
        or step % 20 == 0
        or step == TRAINING_STEPS
    ):

        print(
            f"Step {step:3d} | "
            f"Total={total_loss.item():.4f} | "
            f"Task={task_loss.item():.4f} | "
            f"Budget={budget_loss.item():.4f} | "
            f"Expected Tokens="
            f"{expected_tokens.item():.2f}"
        )


 

retention_scorer.eval()


with torch.no_grad():

    final_scores, final_probabilities = (
        retention_scorer(
            hidden_states
        )
    )


print("\n" + "=" * 75)
print("STEP 18: FINAL LEARNED RETENTION PROBABILITIES")
print("=" * 75)


for index, token in enumerate(tokens):

    probability = (
        final_probabilities[0, index]
        .item()
    )

    print(
        f"{index:2d} | "
        f"{token:20s} | "
        f"P(retain)={probability:.4f}"
    )


# =====================================================================
# STEP 19: RANK TOKENS BY LEARNED IMPORTANCE
# =====================================================================

print("\n" + "=" * 75)
print("STEP 19: LEARNED TOKEN RANKING")
print("=" * 75)


ranked_indices = torch.argsort(
    final_probabilities[0],
    descending=True
)


for rank, index in enumerate(
    ranked_indices.tolist()
):

    print(
        f"{rank + 1:2d} | "
        f"Position={index:2d} | "
        f"Token={tokens[index]:20s} | "
        f"P={final_probabilities[0, index].item():.4f}"
    )


 

with torch.no_grad():

    expected_retained_tokens = (
        final_probabilities.sum()
        .item()
    )


expected_retention_ratio = (
    expected_retained_tokens
    /
    sequence_length
)


print("\n" + "=" * 75)
print("STEP 20: MEMORY ANALYSIS")
print("=" * 75)

print(
    "Original tokens:",
    sequence_length
)

print(
    "Target tokens:",
    target_tokens
)

print(
    "Expected retained tokens:",
    round(
        expected_retained_tokens,
        3
    )
)

print(
    "Expected retention ratio:",
    round(
        expected_retention_ratio,
        4
    )
)


 

threshold = 0.50

binary_mask = (
    final_probabilities
    >= threshold
)


print("\n" + "=" * 75)
print("STEP 21: BINARY INTERPRETATION")
print("=" * 75)


binary_count = (
    binary_mask.sum()
    .item()
)


print(
    "Threshold:",
    threshold
)

print(
    "Tokens with P >= threshold:",
    binary_count
)


for index, token in enumerate(tokens):

    probability = (
        final_probabilities[0, index]
        .item()
    )

    status = (
        "KEEP"
        if probability >= threshold
        else "DROP"
    )

    print(
        f"{index:2d} | "
        f"{token:20s} | "
        f"P={probability:.4f} | "
        f"{status}"
    )


 
with torch.no_grad():

    gated_hidden_states = (
        apply_soft_gate(
            hidden_states,
            final_probabilities
        )
    )


    gated_logits = (
        qa_model.qa_classifier(
            gated_hidden_states
        )
    )


gated_start_logits = (
    gated_logits[..., 0]
)

gated_end_logits = (
    gated_logits[..., 1]
)


gated_start = torch.argmax(
    gated_start_logits,
    dim=-1
).item()


gated_end = torch.argmax(
    gated_end_logits,
    dim=-1
).item()


if gated_end >= gated_start:

    gated_answer = tokenizer.decode(
        input_ids[
            0,
            gated_start:gated_end + 1
        ],
        skip_special_tokens=True
    )

else:

    gated_answer = ""


print("\n" + "=" * 75)
print("STEP 22: QA AFTER LEARNED RETENTION")
print("=" * 75)

print(
    "Baseline answer:",
    baseline_answer
)

print(
    "Retention start:",
    gated_start
)

print(
    "Retention end:",
    gated_end
)

print(
    "Retention answer:",
    gated_answer
)


 
print("\n" + "=" * 75)
print("FINAL SUMMARY")
print("=" * 75)

print(
    "Model:",
    MODEL_NAME
)

print(
    "Device:",
    device
)

print(
    "Sequence length:",
    sequence_length
)

print(
    "Target retention ratio:",
    RETENTION_RATIO
)

print(
    "Target tokens:",
    target_tokens
)

print(
    "Expected retained tokens:",
    round(
        expected_retained_tokens,
        3
    )
)

print(
    "Expected retention ratio:",
    round(
        expected_retention_ratio,
        4
    )
)

print(
    "Baseline answer:",
    baseline_answer
)

print(
    "Retention answer:",
    gated_answer
)

print("=" * 75)
