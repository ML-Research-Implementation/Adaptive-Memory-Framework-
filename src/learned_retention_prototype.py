import os
import random
import numpy as np

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

SEED = 42

RETENTION_RATIO = 0.50

LEARNING_RATE = 1e-3

TRAINING_STEPS = 500

BUDGET_LAMBDA = 0.10

GRADIENT_CLIP = 1.0

SAVE_PATH = "retention_scorer.pt"


# =====================================================================
# REPRODUCIBILITY
# =====================================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# =====================================================================
# DEVICE
# =====================================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def print_header(title):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def count_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


# =====================================================================
# STEP 1: LOAD TOKENIZER
# =====================================================================

print_header("STEP 1: MODEL")

print("Model:", MODEL_NAME)
print("Device:", device)
print("Seed:", SEED)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# =====================================================================
# STEP 2: LOAD PRETRAINED DISTILBERT
# =====================================================================

qa_model = (
    DistilBertForQuestionAnswering
    .from_pretrained(MODEL_NAME)
)


qa_model = qa_model.to(device)

qa_model.eval()


# ---------------------------------------------------------------------
# Freeze the complete pretrained model.
#
# We are studying the retention mechanism independently first.
# ---------------------------------------------------------------------

for parameter in qa_model.parameters():

    parameter.requires_grad = False


encoder = qa_model.distilbert


print("\nTotal DistilBERT parameters:")

print(
    f"{count_parameters(qa_model):,}"
)


# =====================================================================
# STEP 3: QUESTION AND CONTEXT
# =====================================================================

question = (
    "What is artificial intelligence?"
)


context = (
    "Artificial intelligence is a field of computer science "
    "that focuses on creating systems capable of performing "
    "tasks that normally require human intelligence."
)


# Ground-truth answer for this demonstration.

answer_text = (
    "a field of computer science"
)


# =====================================================================
# STEP 4: TOKENIZATION
# =====================================================================

encoded = tokenizer(
    question,
    context,
    return_tensors="pt",
    return_offsets_mapping=True,
    truncation=True,
    max_length=512
)


input_ids = encoded["input_ids"]

attention_mask = encoded["attention_mask"]

offset_mapping = encoded["offset_mapping"]


# ---------------------------------------------------------------------
# Determine which tokens belong to which sequence.
#
# sequence_ids():
#
# None → special token
# 0    → question
# 1    → context
# ---------------------------------------------------------------------

sequence_ids = encoded.sequence_ids(
    batch_index=0
)


tokens = tokenizer.convert_ids_to_tokens(
    input_ids[0]
)


sequence_length = input_ids.shape[1]


print_header("STEP 4: TOKENIZATION")

print(
    "Sequence length:",
    sequence_length
)


for index, token in enumerate(tokens):

    sequence_type = sequence_ids[index]

    if sequence_type is None:

        segment = "SPECIAL"

    elif sequence_type == 0:

        segment = "QUESTION"

    else:

        segment = "CONTEXT"


    print(
        f"{index:3d} | "
        f"{token:20s} | "
        f"{segment}"
    )


# Move model inputs to device.

model_inputs = {
    "input_ids": input_ids.to(device),
    "attention_mask": attention_mask.to(device)
}


# =====================================================================
# STEP 5: IDENTIFY ANSWER SPAN CORRECTLY
# =====================================================================

answer_start_char = context.find(
    answer_text
)


if answer_start_char == -1:

    raise ValueError(
        "Answer text was not found in context."
    )


answer_end_char = (
    answer_start_char
    +
    len(answer_text)
)


# ---------------------------------------------------------------------
# Find tokens belonging to the answer.
#
# IMPORTANT:
# We only consider context tokens.
# ---------------------------------------------------------------------

answer_token_positions = []


for index, ((start, end), sequence_id) in enumerate(
    zip(
        offset_mapping[0].tolist(),
        sequence_ids
    )
):

    if sequence_id != 1:

        continue


    if start is None or end is None:

        continue


    if (
        start >= answer_start_char
        and end <= answer_end_char
        and end > start
    ):

        answer_token_positions.append(
            index
        )


if not answer_token_positions:

    raise RuntimeError(
        "Could not locate answer token positions."
    )


start_target_index = (
    answer_token_positions[0]
)


end_target_index = (
    answer_token_positions[-1]
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


print_header("STEP 5: GROUND-TRUTH ANSWER")

print(
    "Answer:",
    answer_text
)

print(
    "Answer start token:",
    start_target_index,
    tokens[start_target_index]
)

print(
    "Answer end token:",
    end_target_index,
    tokens[end_target_index]
)


# =====================================================================
# STEP 6: DISTILBERT ENCODER
# =====================================================================

with torch.no_grad():

    encoder_outputs = encoder(
        **model_inputs,
        output_hidden_states=True
    )


hidden_states = (
    encoder_outputs.last_hidden_state
)


all_hidden_states = (
    encoder_outputs.hidden_states
)


hidden_dimension = (
    hidden_states.shape[-1]
)


print_header("STEP 6: DISTILBERT")

print(
    "Final hidden states:",
    hidden_states.shape
)

print(
    "Hidden dimension:",
    hidden_dimension
)

print(
    "Transformer layers:",
    len(all_hidden_states) - 1
)


for layer_index, layer_hidden in enumerate(
    all_hidden_states
):

    print(
        f"Layer {layer_index}: "
        f"{tuple(layer_hidden.shape)}"
    )


# =====================================================================
# STEP 7: BASELINE QA
# =====================================================================

with torch.no_grad():

    baseline_outputs = qa_model(
        **model_inputs
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


print_header("STEP 7: BASELINE QA")

print(
    "Predicted start:",
    baseline_start
)

print(
    "Predicted end:",
    baseline_end
)

print(
    "Predicted answer:",
    baseline_answer
)


# =====================================================================
# STEP 8: CREATE VALID TOKEN MASK
# =====================================================================

# We don't want padding or special tokens to consume our
# retention budget.
#
# However, [CLS] and [SEP] are structurally important for
# Transformer input, so we mark them separately as protected.


valid_token_mask = torch.zeros(
    sequence_length,
    dtype=torch.bool
)


protected_token_mask = torch.zeros(
    sequence_length,
    dtype=torch.bool
)


for index, sequence_id in enumerate(
    sequence_ids
):

    token_id = input_ids[0, index].item()


    # Special tokens

    if sequence_id is None:

        protected_token_mask[index] = True

        continue


    # Question or context token

    if attention_mask[0, index].item() == 1:

        valid_token_mask[index] = True


# ---------------------------------------------------------------------
# Memory-budget tokens are only normal question/context tokens.
# ---------------------------------------------------------------------

budget_token_count = (
    valid_token_mask.sum().item()
)


target_budget = max(
    1,
    round(
        budget_token_count
        * RETENTION_RATIO
    )
)


print_header("STEP 8: MEMORY BUDGET")

print(
    "Total sequence tokens:",
    sequence_length
)

print(
    "Budget-eligible tokens:",
    budget_token_count
)

print(
    "Retention ratio:",
    RETENTION_RATIO
)

print(
    "Target retained tokens:",
    target_budget
)


# =====================================================================
# STEP 9: LEARNABLE RETENTION SCORER
# =====================================================================

class RetentionScorer(nn.Module):

    """
    Learnable token importance network.

    Input:
        h_t ∈ R^768

    Output:
        s_t ∈ R

    Retention probability:

        p_t = sigmoid(s_t)
    """

    def __init__(
        self,
        hidden_dimension,
        dropout=0.10
    ):

        super().__init__()


        intermediate_dimension = max(
            64,
            hidden_dimension // 4
        )


        self.network = nn.Sequential(

            nn.Linear(
                hidden_dimension,
                intermediate_dimension
            ),

            nn.LayerNorm(
                intermediate_dimension
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                intermediate_dimension,
                1
            )
        )


        # Start with a neutral probability.
        #
        # sigmoid(0) = 0.5

        nn.init.zeros_(
            self.network[-1].weight
        )

        nn.init.zeros_(
            self.network[-1].bias
        )


    def forward(
        self,
        hidden_states
    ):

        scores = self.network(
            hidden_states
        )


        scores = scores.squeeze(-1)


        probabilities = torch.sigmoid(
            scores
        )


        return (
            scores,
            probabilities
        )


# =====================================================================
# STEP 10: CREATE RETENTION MODEL
# =====================================================================

retention_scorer = RetentionScorer(
    hidden_dimension
).to(device)


print_header("STEP 10: RETENTION SCORER")

print(retention_scorer)

print(
    "\nTrainable parameters:",
    f"{count_parameters(retention_scorer):,}"
)


# =====================================================================
# STEP 11: RETENTION PROBABILITY FUNCTION
# =====================================================================

def get_retention_probabilities(
    scorer,
    hidden_states,
    protected_mask
):

    scores, probabilities = scorer(
        hidden_states
    )


    # Protected tokens must remain active.

    protected_mask = (
        protected_mask
        .to(probabilities.device)
    )


    probabilities = torch.where(
        protected_mask.unsqueeze(0),
        torch.ones_like(probabilities),
        probabilities
    )


    return (
        scores,
        probabilities
    )


# =====================================================================
# STEP 12: BUDGET LOSS
# =====================================================================

def calculate_budget_loss(
    probabilities,
    valid_mask,
    target_budget
):

    valid_mask = (
        valid_mask
        .to(probabilities.device)
    )


    # Only count tokens that are eligible
    # for adaptive retention.

    expected_tokens = (
        probabilities
        *
        valid_mask.unsqueeze(0)
    ).sum(dim=-1)


    # Penalize excess memory.

    excess = F.relu(
        expected_tokens
        -
        target_budget
    )


    budget_loss = (
        excess ** 2
    ).mean()


    return (
        budget_loss,
        expected_tokens
    )


# =====================================================================
# STEP 13: SOFT RETENTION
# =====================================================================

def apply_soft_retention(
    hidden_states,
    probabilities
):

    """
    Soft retention:

        h'_t = p_t h_t

    This is differentiable.

    IMPORTANT:
    This does not physically remove tokens.
    """

    return (
        hidden_states
        *
        probabilities.unsqueeze(-1)
    )


# =====================================================================
# STEP 14: TASK LOSS
# =====================================================================

def calculate_task_loss(
    gated_hidden_states
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
        start_loss
        +
        end_loss
    ) / 2


    return (
        task_loss,
        start_logits,
        end_logits
    )


# =====================================================================
# STEP 15: OPTIMIZER
# =====================================================================

optimizer = torch.optim.AdamW(
    retention_scorer.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# =====================================================================
# STEP 16: TRAINING
# =====================================================================

print_header("STEP 16: TRAINING RETENTION SCORER")

retention_scorer.train()


for step in range(
    1,
    TRAINING_STEPS + 1
):

    optimizer.zero_grad(
        set_to_none=True
    )


    # ---------------------------------------------------------------
    # Calculate probabilities
    # ---------------------------------------------------------------

    scores, probabilities = (
        get_retention_probabilities(
            retention_scorer,
            hidden_states,
            protected_token_mask
        )
    )


    # ---------------------------------------------------------------
    # Apply soft gate
    # ---------------------------------------------------------------

    gated_hidden_states = (
        apply_soft_retention(
            hidden_states,
            probabilities
        )
    )


    # ---------------------------------------------------------------
    # Task loss
    # ---------------------------------------------------------------

    (
        task_loss,
        start_logits,
        end_logits
    ) = calculate_task_loss(
        gated_hidden_states
    )


    # ---------------------------------------------------------------
    # Memory loss
    # ---------------------------------------------------------------

    (
        budget_loss,
        expected_tokens
    ) = calculate_budget_loss(
        probabilities,
        valid_token_mask,
        target_budget
    )


    # ---------------------------------------------------------------
    # Total objective
    # ---------------------------------------------------------------

    total_loss = (
        task_loss
        +
        BUDGET_LAMBDA
        *
        budget_loss
    )


    # ---------------------------------------------------------------
    # Backpropagation
    # ---------------------------------------------------------------

    total_loss.backward()


    # ---------------------------------------------------------------
    # Gradient clipping
    # ---------------------------------------------------------------

    gradient_norm = (
        torch.nn.utils.clip_grad_norm_(
            retention_scorer.parameters(),
            GRADIENT_CLIP
        )
    )


    optimizer.step()


    # ---------------------------------------------------------------
    # Logging
    # ---------------------------------------------------------------

    if (
        step == 1
        or step % 25 == 0
        or step == TRAINING_STEPS
    ):

        print(
            f"Step {step:4d} | "
            f"Total={total_loss.item():.5f} | "
            f"Task={task_loss.item():.5f} | "
            f"Budget={budget_loss.item():.5f} | "
            f"ExpectedTokens="
            f"{expected_tokens.mean().item():.3f} | "
            f"GradNorm="
            f"{float(gradient_norm):.4f}"
        )


# =====================================================================
# STEP 17: FINAL RETENTION PROBABILITIES
# =====================================================================

retention_scorer.eval()


with torch.no_grad():

    final_scores, final_probabilities = (
        get_retention_probabilities(
            retention_scorer,
            hidden_states,
            protected_token_mask
        )
    )


print_header(
    "STEP 17: FINAL RETENTION PROBABILITIES"
)


for index, token in enumerate(tokens):

    probability = (
        final_probabilities[0, index]
        .item()
    )


    if protected_token_mask[index]:

        token_type = "PROTECTED"

    elif valid_token_mask[index]:

        token_type = "ADAPTIVE"

    else:

        token_type = "IGNORED"


    print(
        f"{index:3d} | "
        f"{token:20s} | "
        f"P={probability:.4f} | "
        f"{token_type}"
    )


# =====================================================================
# STEP 18: MEMORY ANALYSIS
# =====================================================================

with torch.no_grad():

    expected_retained_tokens = (
        (
            final_probabilities
            *
            valid_token_mask
            .to(device)
            .unsqueeze(0)
        )
        .sum()
        .item()
    )


expected_retention_ratio = (
    expected_retained_tokens
    /
    budget_token_count
)


print_header(
    "STEP 18: MEMORY ANALYSIS"
)


print(
    "Budget-eligible tokens:",
    budget_token_count
)

print(
    "Target retained tokens:",
    target_budget
)

print(
    "Expected retained tokens:",
    f"{expected_retained_tokens:.3f}"
)

print(
    "Expected retention ratio:",
    f"{expected_retention_ratio:.4f}"
)


# =====================================================================
# STEP 19: LEARNED TOKEN RANKING
# =====================================================================

print_header(
    "STEP 19: LEARNED TOKEN IMPORTANCE RANKING"
)


adaptive_indices = [
    index
    for index in range(sequence_length)
    if valid_token_mask[index]
]


adaptive_indices.sort(
    key=lambda index:
        final_probabilities[
            0,
            index
        ].item(),
    reverse=True
)


for rank, index in enumerate(
    adaptive_indices
):

    probability = (
        final_probabilities[
            0,
            index
        ].item()
    )


    print(
        f"{rank + 1:3d} | "
        f"Position={index:2d} | "
        f"Token={tokens[index]:20s} | "
        f"P={probability:.4f}"
    )


# =====================================================================
# STEP 20: TOP-K INTERPRETATION
# =====================================================================
#
# This is NOT stochastic Bernoulli sampling.
#
# It is simply a deterministic interpretation:
#
# "If we were allowed to retain K tokens, which ones would
#  the learned scorer choose?"
#


top_k = min(
    target_budget,
    len(adaptive_indices)
)


selected_indices = (
    adaptive_indices[:top_k]
)


selected_index_set = set(
    selected_indices
)


print_header(
    "STEP 20: TOP-K INTERPRETATION"
)


print(
    "Top-K:",
    top_k
)


for index in range(sequence_length):

    token = tokens[index]


    if protected_token_mask[index]:

        status = "PROTECTED"

    elif index in selected_index_set:

        status = "KEEP"

    elif valid_token_mask[index]:

        status = "DROP"

    else:

        status = "IGNORE"


    probability = (
        final_probabilities[
            0,
            index
        ].item()
    )


    print(
        f"{index:3d} | "
        f"{token:20s} | "
        f"P={probability:.4f} | "
        f"{status}"
    )


# =====================================================================
# STEP 21: SOFT-GATED QA
# =====================================================================

with torch.no_grad():

    gated_hidden_states = (
        apply_soft_retention(
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


print_header(
    "STEP 21: SOFT-GATED QA"
)


print(
    "Ground truth:",
    answer_text
)

print(
    "Baseline prediction:",
    baseline_answer
)

print(
    "Retention prediction:",
    gated_answer
)


# =====================================================================
# STEP 22: ANSWER TOKEN RETENTION CHECK
# =====================================================================

print_header(
    "STEP 22: ANSWER TOKEN RETENTION"
)


answer_probabilities = (
    final_probabilities[
        0,
        answer_token_positions
    ]
)


print(
    "Answer tokens:"
)


for index in answer_token_positions:

    print(
        f"Position={index:2d} | "
        f"Token={tokens[index]:20s} | "
        f"P={final_probabilities[0, index].item():.4f}"
    )


print(
    "\nAverage answer-token retention:",
    f"{answer_probabilities.mean().item():.4f}"
)


# =====================================================================
# STEP 23: SAVE RETENTION SCORER
# =====================================================================

torch.save(
    {
        "model_name": MODEL_NAME,
        "hidden_dimension": hidden_dimension,
        "retention_ratio": RETENTION_RATIO,
        "state_dict": retention_scorer.state_dict()
    },
    SAVE_PATH
)


print_header(
    "STEP 23: MODEL SAVED"
)


print(
    "Saved retention scorer to:",
    os.path.abspath(SAVE_PATH)
)


# =====================================================================
# FINAL SUMMARY
# =====================================================================

print_header(
    "FINAL SUMMARY"
)


print(
    "Pretrained model:",
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
    "Budget-eligible tokens:",
    budget_token_count
)

print(
    "Target retention:",
    f"{RETENTION_RATIO * 100:.1f}%"
)

print(
    "Target tokens:",
    target_budget
)

print(
    "Expected retained tokens:",
    f"{expected_retained_tokens:.3f}"
)

print(
    "Expected retention:",
    f"{expected_retention_ratio * 100:.2f}%"
)

print(
    "Ground-truth answer:",
    answer_text
)

print(
    "Baseline answer:",
    baseline_answer
)

print(
    "Soft-gated answer:",
    gated_answer
)

print("=" * 80)