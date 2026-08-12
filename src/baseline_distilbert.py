 import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    DistilBertForQuestionAnswering
)


# =====================================================================
# CONFIGURATION
# =====================================================================

MODEL_NAME = "distilbert-base-uncased-distilled-squad"

RETENTION_RATIO = 0.50

LEARNING_RATE = 1e-3

TRAINING_STEPS = 100

BUDGET_LAMBDA = 0.05

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =====================================================================
# STEP 1: DEVICE
# =====================================================================

print("=" * 75)
print("STEP 1: DEVICE")
print("=" * 75)

print("Device:", device)


# =====================================================================
# STEP 2: LOAD TOKENIZER
# =====================================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# =====================================================================
# STEP 3: LOAD PRETRAINED DISTILBERT QA MODEL
# =====================================================================

qa_model = DistilBertForQuestionAnswering.from_pretrained(
    MODEL_NAME
)

qa_model = qa_model.to(device)

qa_model.eval()


# We will initially freeze DistilBERT and the QA head.
#
# This lets us study the retention mechanism separately.

for parameter in qa_model.parameters():

    parameter.requires_grad = False


encoder = qa_model.distilbert


# =====================================================================
# STEP 4: INPUT
# =====================================================================

question = "What is artificial intelligence?"

context = """
Artificial intelligence is a field of computer science
that focuses on creating systems capable of performing
tasks that normally require human intelligence.
"""


# =====================================================================
# STEP 5: TOKENIZATION
# =====================================================================

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

print("Tokens:")

for index, token in enumerate(tokens):

    print(
        f"{index:2d} -> {token}"
    )

print("\nInput shape:")
print(input_ids.shape)


# =====================================================================
# STEP 6: BASELINE DISTILBERT FORWARD PASS
# =====================================================================

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


# =====================================================================
# STEP 7: BASELINE QA PREDICTION
# =====================================================================

with torch.no_grad():

    baseline_outputs = qa_model(
        **inputs
    )


baseline_start_logits = baseline_outputs.start_logits

baseline_end_logits = baseline_outputs.end_logits


baseline_start = torch.argmax(
    baseline_start_logits,
    dim=-1
).item()

baseline_end = torch.argmax(
    baseline_end_logits,
    dim=-1
).item()


baseline_answer = tokenizer.decode(
    input_ids[0, baseline_start:baseline_end + 1],
    skip_special_tokens=True
)


print("\n" + "=" * 75)
print("STEP 7: BASELINE QA")
print("=" * 75)

print("Start index:", baseline_start)

print("End index:", baseline_end)

print("Answer:", baseline_answer)


# =====================================================================
# STEP 8: LEARNABLE RETENTION SCORER
# =====================================================================

class RetentionScorer(nn.Module):

    """
    Learnable token-retention scorer.

    Input:
        h_t ∈ R^768

    Output:
        s_t ∈ R

    Then:

        p_t = sigmoid(s_t)

    where p_t represents the learned probability/
    strength of retaining token t.
    """

    def __init__(self, hidden_dimension):

        super().__init__()

        self.scorer = nn.Sequential(

            nn.Linear(
                hidden_dimension,
                hidden_dimension // 4
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dimension // 4,
                1
            )
        )


    def forward(self, hidden_states):

        # hidden_states:
        #
        # [batch, sequence_length, hidden_dimension]

        scores = self.scorer(
            hidden_states
        )

        # [batch, sequence_length, 1]
        scores = scores.squeeze(-1)

        # [batch, sequence_length]
        probabilities = torch.sigmoid(
            scores
        )

        return scores, probabilities


# =====================================================================
# STEP 9: CREATE RETENTION SCORER
# =====================================================================

hidden_dimension = hidden_states.shape[-1]

retention_scorer = RetentionScorer(
    hidden_dimension
).to(device)


print("\n" + "=" * 75)
print("STEP 9: RETENTION SCORER")
print("=" * 75)

print(
    "Hidden dimension:",
    hidden_dimension
)

print(
    "Retention scorer created successfully."
)


# =====================================================================
# STEP 10: INITIAL RETENTION PROBABILITIES
# =====================================================================

with torch.no_grad():

    initial_scores, initial_probabilities = (
        retention_scorer(hidden_states)
    )


print("\n" + "=" * 75)
print("STEP 10: INITIAL RETENTION PROBABILITIES")
print("=" * 75)

for index, token in enumerate(tokens):

    probability = (
        initial_probabilities[0, index]
        .item()
    )

    print(
        f"{index:2d} | "
        f"{token:20s} | "
        f"P(retain) = {probability:.4f}"
    )


# =====================================================================
# STEP 11: SOFT RETENTION GATE
# =====================================================================

def apply_soft_retention(
    hidden_states,
    retention_probabilities
):

    """
    Apply a differentiable retention gate.

    h'_t = p_t * h_t

    This is intentionally a SOFT gate.

    We are NOT physically removing tokens yet.
    """

    gate = retention_probabilities.unsqueeze(-1)

    gated_hidden_states = (
        hidden_states * gate
    )

    return gated_hidden_states


# =====================================================================
# STEP 12: MEMORY BUDGET
# =====================================================================

target_tokens = max(
    1,
    int(sequence_length * RETENTION_RATIO)
)


target_ratio = (
    target_tokens / sequence_length
)


print("\n" + "=" * 75)
print("STEP 12: MEMORY BUDGET")
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
    "Target retained tokens:",
    target_tokens
)

print(
    "Target retention ratio:",
    target_ratio
)


# =====================================================================
# STEP 13: MEMORY BUDGET LOSS
# =====================================================================

def memory_budget_loss(
    retention_probabilities,
    target_tokens
):

    """
    Expected number of retained tokens:

        E[T] = Σ p_t

    Penalize the model when the expected number
    of retained tokens exceeds the target.
    """

    expected_tokens = (
        retention_probabilities.sum(
            dim=-1
        )
    )

    excess_tokens = F.relu(
        expected_tokens - target_tokens
    )

    loss = (
        excess_tokens ** 2
    ).mean()

    return loss, expected_tokens


# =====================================================================
# STEP 14: QA LOSS + RETENTION LOSS
# =====================================================================

def calculate_total_loss(
    hidden_states,
    retention_probabilities,
    start_positions,
    end_positions,
    target_tokens
):

    # ---------------------------------------------------------------
    # Apply soft retention
    # ---------------------------------------------------------------

    gated_hidden_states = apply_soft_retention(
        hidden_states,
        retention_probabilities
    )


    # ---------------------------------------------------------------
    # QA HEAD
    # ---------------------------------------------------------------

    logits = qa_model.qa_classifier(
        gated_hidden_states
    )


    start_logits = logits[..., 0]

    end_logits = logits[..., 1]


    # ---------------------------------------------------------------
    # QA TASK LOSS
    # ---------------------------------------------------------------

    start_loss = F.cross_entropy(
        start_logits,
        start_positions
    )

    end_loss = F.cross_entropy(
        end_logits,
        end_positions
    )

    task_loss = (
        start_loss + end_loss
    ) / 2


    # ---------------------------------------------------------------
    # MEMORY LOSS
    # ---------------------------------------------------------------

    budget_loss, expected_tokens = (
        memory_budget_loss(
            retention_probabilities,
            target_tokens
        )
    )


    # ---------------------------------------------------------------
    # TOTAL LOSS
    # ---------------------------------------------------------------

    total_loss = (
        task_loss
        +
        BUDGET_LAMBDA * budget_loss
    )


    return (
        total_loss,
        task_loss,
        budget_loss,
        expected_tokens,
        start_logits,
        end_logits
    )


# =====================================================================
# STEP 15: DEFINE TRAINING TARGET
# =====================================================================

# For this demonstration we use the answer span that
# the pretrained QA model predicted.
#
# This is NOT yet our final dataset-training procedure.
#
# Later we will use real labelled datasets such as SQuAD.

start_target = torch.tensor(
    [baseline_start],
    dtype=torch.long,
    device=device
)

end_target = torch.tensor(
    [baseline_end],
    dtype=torch.long,
    device=device
)


# =====================================================================
# STEP 16: OPTIMIZER
# =====================================================================

optimizer = torch.optim.AdamW(
    retention_scorer.parameters(),
    lr=LEARNING_RATE
)


# =====================================================================
# STEP 17: TRAIN RETENTION SCORER
# =====================================================================

print("\n" + "=" * 75)
print("STEP 17: TRAINING RETENTION SCORER")
print("=" * 75)


retention_scorer.train()


for step in range(
    1,
    TRAINING_STEPS + 1
):

    optimizer.zero_grad()


    # ---------------------------------------------------------------
    # Get current retention probabilities
    # ---------------------------------------------------------------

    scores, probabilities = (
        retention_scorer(hidden_states)
    )


    # ---------------------------------------------------------------
    # Calculate losses
    # ---------------------------------------------------------------

    (
        total_loss,
        task_loss,
        budget_loss,
        expected_tokens,
        start_logits,
        end_logits
    ) = calculate_total_loss(
        hidden_states=hidden_states,
        retention_probabilities=probabilities,
        start_positions=start_target,
        end_positions=end_target,
        target_tokens=target_tokens
    )


    # ---------------------------------------------------------------
    # Backpropagation
    # ---------------------------------------------------------------

    total_loss.backward()


    # Gradient clipping makes the training more stable.

    torch.nn.utils.clip_grad_norm_(
        retention_scorer.parameters(),
        max_norm=1.0
    )


    optimizer.step()


    # ---------------------------------------------------------------
    # Display progress
    # ---------------------------------------------------------------

    if (
        step == 1
        or step % 10 == 0
        or step == TRAINING_STEPS
    ):

        print(
            f"Step {step:3d} | "
            f"Total Loss = {total_loss.item():.4f} | "
            f"Task Loss = {task_loss.item():.4f} | "
            f"Budget Loss = {budget_loss.item():.4f} | "
            f"Expected Tokens = "
            f"{expected_tokens.mean().item():.2f}"
        )


# =====================================================================
# STEP 18: FINAL RETENTION PROBABILITIES
# =====================================================================

retention_scorer.eval()


with torch.no_grad():

    final_scores, final_probabilities = (
        retention_scorer(hidden_states)
    )


print("\n" + "=" * 75)
print("STEP 18: LEARNED RETENTION PROBABILITIES")
print("=" * 75)


for index, token in enumerate(tokens):

    probability = (
        final_probabilities[0, index]
        .item()
    )

    print(
        f"{index:2d} | "
        f"{token:20s} | "
        f"P(retain) = {probability:.4f}"
    )


# =====================================================================
# STEP 19: EXPECTED MEMORY USAGE
# =====================================================================

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
print("STEP 19: MEMORY ANALYSIS")
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
        2
    )
)

print(
    "Expected retention ratio:",
    round(
        expected_retention_ratio,
        4
    )
)


# =====================================================================
# STEP 20: SOFT-GATED REPRESENTATIONS
# =====================================================================

with torch.no_grad():

    gated_hidden_states = (
        apply_soft_retention(
            hidden_states,
            final_probabilities
        )
    )


print("\n" + "=" * 75)
print("STEP 20: SOFT-GATED REPRESENTATIONS")
print("=" * 75)

print(
    "Original representation:",
    hidden_states.shape
)

print(
    "Gated representation:",
    gated_hidden_states.shape
)


# =====================================================================
# STEP 21: PREDICTION AFTER RETENTION
# =====================================================================

with torch.no_grad():

    gated_logits = qa_model.qa_classifier(
        gated_hidden_states
    )


gated_start_logits = gated_logits[..., 0]

gated_end_logits = gated_logits[..., 1]


gated_start = torch.argmax(
    gated_start_logits,
    dim=-1
).item()

gated_end = torch.argmax(
    gated_end_logits,
    dim=-1
).item()


# Make sure the predicted span is valid.

if gated_end < gated_start:

    gated_answer = ""

else:

    gated_answer = tokenizer.decode(
        input_ids[
            0,
            gated_start:gated_end + 1
        ],
        skip_special_tokens=True
    )


print("\n" + "=" * 75)
print("STEP 21: QA AFTER LEARNED RETENTION")
print("=" * 75)

print(
    "Predicted start:",
    gated_start
)

print(
    "Predicted end:",
    gated_end
)

print(
    "Predicted answer:",
    gated_answer
)


# =====================================================================
# STEP 22: FINAL SUMMARY
# =====================================================================

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
    "Original tokens:",
    sequence_length
)

print(
    "Target retained tokens:",
    target_tokens
)

print(
    "Expected retained tokens:",
    round(
        expected_retained_tokens,
        2
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