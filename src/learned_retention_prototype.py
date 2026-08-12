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

ENTROPY_LAMBDA = 0.001

TEMPERATURE = 1.0

GRADIENT_CLIP = 1.0

CHECKPOINT_PATH = "retention_scorer.pt"


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
# UTILITY
# =====================================================================

def print_header(title):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def count_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


# =====================================================================
# STEP 1 — LOAD MODEL
# =====================================================================

print_header("STEP 1 — MODEL SETUP")

print("Model:", MODEL_NAME)
print("Device:", device)
print("Seed:", SEED)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


qa_model = (
    DistilBertForQuestionAnswering
    .from_pretrained(MODEL_NAME)
)


qa_model = qa_model.to(device)

qa_model.eval()


# Freeze pretrained model.
# Only the retention scorer will be trained.

for parameter in qa_model.parameters():

    parameter.requires_grad = False


encoder = qa_model.distilbert


print(
    "DistilBERT parameters:",
    f"{count_parameters(qa_model):,}"
)


# =====================================================================
# STEP 2 — INPUT
# =====================================================================

print_header("STEP 2 — INPUT DATA")


question = (
    "What is artificial intelligence?"
)


context = (
    "Artificial intelligence is a field of computer science "
    "that focuses on creating systems capable of performing "
    "tasks that normally require human intelligence."
)


answer_text = (
    "a field of computer science"
)


print("Question:")
print(question)

print("\nContext:")
print(context)

print("\nGround-truth answer:")
print(answer_text)


# =====================================================================
# STEP 3 — TOKENIZATION
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

sequence_ids = encoded.sequence_ids(
    batch_index=0
)


tokens = tokenizer.convert_ids_to_tokens(
    input_ids[0]
)


sequence_length = input_ids.shape[1]


model_inputs = {

    "input_ids":
        input_ids.to(device),

    "attention_mask":
        attention_mask.to(device)
}


print_header("STEP 3 — TOKENIZATION")


for index, token in enumerate(tokens):

    sequence_id = sequence_ids[index]

    if sequence_id is None:
        token_type = "SPECIAL"

    elif sequence_id == 0:
        token_type = "QUESTION"

    else:
        token_type = "CONTEXT"

    print(
        f"{index:3d} | "
        f"{token:20s} | "
        f"{token_type}"
    )


print("\nSequence length:", sequence_length)


# =====================================================================
# STEP 4 — FIND GROUND-TRUTH ANSWER SPAN
# =====================================================================

answer_start_char = context.find(
    answer_text
)


if answer_start_char == -1:

    raise ValueError(
        "Ground-truth answer was not found in context."
    )


answer_end_char = (
    answer_start_char
    + len(answer_text)
)


answer_token_positions = []


for index, (offset, sequence_id) in enumerate(
    zip(
        offset_mapping[0].tolist(),
        sequence_ids
    )
):

    start, end = offset

    # Only context tokens.

    if sequence_id != 1:
        continue

    # Ignore empty offsets.

    if end <= start:
        continue

    if (
        start >= answer_start_char
        and end <= answer_end_char
    ):

        answer_token_positions.append(index)


if not answer_token_positions:

    raise RuntimeError(
        "Could not locate answer tokens."
    )


start_target_index = answer_token_positions[0]

end_target_index = answer_token_positions[-1]


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


print_header("STEP 4 — GROUND-TRUTH SPAN")


print(
    "Start:",
    start_target_index,
    tokens[start_target_index]
)

print(
    "End:",
    end_target_index,
    tokens[end_target_index]
)


# =====================================================================
# STEP 5 — DISTILBERT REPRESENTATIONS
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


hidden_dimension = hidden_states.shape[-1]


print_header(
    "STEP 5 — DISTILBERT REPRESENTATIONS"
)


print(
    "Final hidden state:",
    tuple(hidden_states.shape)
)

print(
    "Hidden dimension:",
    hidden_dimension
)

print(
    "Transformer layers:",
    len(all_hidden_states) - 1
)


for layer_index, layer in enumerate(
    all_hidden_states
):

    print(
        f"Layer {layer_index}: "
        f"{tuple(layer.shape)}"
    )


# =====================================================================
# STEP 6 — BASELINE QA
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


print_header("STEP 6 — BASELINE QA")


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
# STEP 7 — TOKEN MASKS
# =====================================================================

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

    # Special tokens are protected.

    if sequence_id is None:

        protected_token_mask[index] = True

        continue


    # Question/context tokens are adaptive.

    if attention_mask[0, index].item() == 1:

        valid_token_mask[index] = True


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


print_header(
    "STEP 7 — RETENTION BUDGET"
)


print(
    "Total sequence tokens:",
    sequence_length
)

print(
    "Adaptive tokens:",
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
# STEP 8 — LEARNABLE RETENTION SCORER
# =====================================================================

class RetentionScorer(nn.Module):

    """
    Learnable token retention network.

    h_t
       ↓
    MLP
       ↓
    score s_t
       ↓
    sigmoid
       ↓
    p_t

    p_t = retention probability
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


        # Start from approximately p = 0.5.

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


        # Temperature controls probability sharpness.

        probabilities = torch.sigmoid(
            scores / TEMPERATURE
        )


        return (
            scores,
            probabilities
        )


# =====================================================================
# STEP 9 — CREATE SCORER
# =====================================================================

retention_scorer = RetentionScorer(
    hidden_dimension
).to(device)


print_header(
    "STEP 9 — RETENTION SCORER"
)


print(retention_scorer)


print(
    "\nTrainable parameters:",
    f"{count_parameters(retention_scorer):,}"
)


# =====================================================================
# STEP 10 — GET RETENTION PROBABILITIES
# =====================================================================

def get_probabilities(
    hidden_states,
    scorer,
    protected_mask
):

    scores, probabilities = scorer(
        hidden_states
    )


    protected_mask = (
        protected_mask
        .to(probabilities.device)
    )


    # Special tokens remain fully retained.

    probabilities = torch.where(

        protected_mask.unsqueeze(0),

        torch.ones_like(
            probabilities
        ),

        probabilities
    )


    return (
        scores,
        probabilities
    )


# =====================================================================
# STEP 11 — SOFT RETENTION
# =====================================================================

def apply_soft_retention(
    hidden_states,
    probabilities
):

    return (
        hidden_states
        *
        probabilities.unsqueeze(-1)
    )


# =====================================================================
# STEP 12 — MEMORY LOSS
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


    expected_tokens = (
        probabilities
        *
        valid_mask.unsqueeze(0)
    ).sum(dim=-1)


    excess = F.relu(
        expected_tokens
        -
        target_budget
    )


    loss = (
        excess ** 2
    ).mean()


    return (
        loss,
        expected_tokens
    )


# =====================================================================
# STEP 13 — ENTROPY REGULARIZATION
# =====================================================================

def calculate_entropy(
    probabilities,
    valid_mask
):

    valid_mask = (
        valid_mask
        .to(probabilities.device)
    )


    p = probabilities.clamp(
        min=1e-7,
        max=1 - 1e-7
    )


    entropy = -(
        p * torch.log(p)
        +
        (1 - p)
        *
        torch.log(1 - p)
    )


    masked_entropy = (
        entropy
        *
        valid_mask.unsqueeze(0)
    )


    return masked_entropy.sum() / (
        valid_mask.sum()
        +
        1e-8
    )


# =====================================================================
# STEP 14 — QA LOSS
# =====================================================================

def calculate_qa_loss(
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


    loss = (
        start_loss
        +
        end_loss
    ) / 2


    return (
        loss,
        start_logits,
        end_logits
    )


# =====================================================================
# STEP 15 — OPTIMIZER
# =====================================================================

optimizer = torch.optim.AdamW(
    retention_scorer.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# =====================================================================
# STEP 16 — INITIAL PROBABILITIES
# =====================================================================

retention_scorer.eval()


with torch.no_grad():

    _, initial_probabilities = (
        get_probabilities(
            hidden_states,
            retention_scorer,
            protected_token_mask
        )
    )


print_header(
    "STEP 16 — INITIAL PROBABILITIES"
)


for index, token in enumerate(tokens):

    probability = (
        initial_probabilities[
            0,
            index
        ].item()
    )


    print(
        f"{index:3d} | "
        f"{token:20s} | "
        f"P={probability:.4f}"
    )


# =====================================================================
# STEP 17 — TRAIN RETENTION SCORER
# =====================================================================

print_header(
    "STEP 17 — TRAINING"
)


retention_scorer.train()


for step in range(
    1,
    TRAINING_STEPS + 1
):

    optimizer.zero_grad(
        set_to_none=True
    )


    # ---------------------------------------------------------------
    # Retention probabilities
    # ---------------------------------------------------------------

    _, probabilities = (
        get_probabilities(
            hidden_states,
            retention_scorer,
            protected_token_mask
        )
    )


    # ---------------------------------------------------------------
    # Soft retention
    # ---------------------------------------------------------------

    gated_hidden_states = (
        apply_soft_retention(
            hidden_states,
            probabilities
        )
    )


    # ---------------------------------------------------------------
    # QA objective
    # ---------------------------------------------------------------

    (
        task_loss,
        start_logits,
        end_logits
    ) = calculate_qa_loss(
        gated_hidden_states
    )


    # ---------------------------------------------------------------
    # Memory objective
    # ---------------------------------------------------------------

    (
        memory_loss,
        expected_tokens
    ) = calculate_budget_loss(
        probabilities,
        valid_token_mask,
        target_budget
    )


    # ---------------------------------------------------------------
    # Entropy
    # ---------------------------------------------------------------

    entropy = calculate_entropy(
        probabilities,
        valid_token_mask
    )


    # ---------------------------------------------------------------
    # Total loss
    #
    # We minimize entropy so probabilities become more decisive.
    # ---------------------------------------------------------------

    total_loss = (
        task_loss
        +
        BUDGET_LAMBDA * memory_loss
        +
        ENTROPY_LAMBDA * entropy
    )


    # ---------------------------------------------------------------
    # Backpropagation
    # ---------------------------------------------------------------

    total_loss.backward()


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
            f"QA={task_loss.item():.5f} | "
            f"Budget={memory_loss.item():.5f} | "
            f"Entropy={entropy.item():.5f} | "
            f"Expected="
            f"{expected_tokens.item():.3f} | "
            f"Grad="
            f"{float(gradient_norm):.4f}"
        )


# =====================================================================
# STEP 18 — FINAL PROBABILITIES
# =====================================================================

retention_scorer.eval()


with torch.no_grad():

    final_scores, final_probabilities = (
        get_probabilities(
            hidden_states,
            retention_scorer,
            protected_token_mask
        )
    )


print_header(
    "STEP 18 — FINAL RETENTION PROBABILITIES"
)


for index, token in enumerate(tokens):

    probability = (
        final_probabilities[
            0,
            index
        ].item()
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
# STEP 19 — MEMORY ANALYSIS
# =====================================================================

with torch.no_grad():

    adaptive_mask = (
        valid_token_mask
        .to(device)
        .unsqueeze(0)
    )


    expected_retained_tokens = (
        final_probabilities
        *
        adaptive_mask
    ).sum().item()


expected_ratio = (
    expected_retained_tokens
    /
    budget_token_count
)


print_header(
    "STEP 19 — MEMORY ANALYSIS"
)


print(
    "Adaptive tokens:",
    budget_token_count
)

print(
    "Target tokens:",
    target_budget
)

print(
    "Expected retained:",
    f"{expected_retained_tokens:.3f}"
)

print(
    "Expected retention:",
    f"{expected_ratio * 100:.2f}%"
)


# =====================================================================
# STEP 20 — TOKEN RANKING
# =====================================================================

adaptive_indices = [

    index

    for index in range(
        sequence_length
    )

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


print_header(
    "STEP 20 — LEARNED TOKEN RANKING"
)


for rank, index in enumerate(
    adaptive_indices,
    start=1
):

    probability = (
        final_probabilities[
            0,
            index
        ].item()
    )


    print(
        f"{rank:3d} | "
        f"Position={index:2d} | "
        f"Token={tokens[index]:20s} | "
        f"P={probability:.4f}"
    )


# =====================================================================
# STEP 21 — TOP-K SELECTION
# =====================================================================

top_k = min(
    target_budget,
    len(adaptive_indices)
)


selected_indices = set(
    adaptive_indices[:top_k]
)


print_header(
    "STEP 21 — TOP-K INTERPRETATION"
)


print(
    "Top-K:",
    top_k
)


for index, token in enumerate(tokens):

    probability = (
        final_probabilities[
            0,
            index
        ].item()
    )


    if protected_token_mask[index]:

        status = "PROTECTED"

    elif index in selected_indices:

        status = "KEEP"

    elif valid_token_mask[index]:

        status = "DROP"

    else:

        status = "IGNORE"


    print(
        f"{index:3d} | "
        f"{token:20s} | "
        f"P={probability:.4f} | "
        f"{status}"
    )


# =====================================================================
# STEP 22 — SOFT-GATED QA
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


gated_start = torch.argmax(
    gated_logits[..., 0],
    dim=-1
).item()


gated_end = torch.argmax(
    gated_logits[..., 1],
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
    "STEP 22 — SOFT-GATED QA"
)


print(
    "Ground truth:",
    answer_text
)

print(
    "Baseline:",
    baseline_answer
)

print(
    "After retention:",
    gated_answer
)


# =====================================================================
# STEP 23 — ANSWER TOKEN ANALYSIS
# =====================================================================

print_header(
    "STEP 23 — ANSWER TOKEN RETENTION"
)


answer_probabilities = (
    final_probabilities[
        0,
        answer_token_positions
    ]
)


for index in answer_token_positions:

    print(
        f"Position={index:2d} | "
        f"Token={tokens[index]:20s} | "
        f"P={final_probabilities[0, index].item():.4f}"
    )


print(
    "\nAverage answer-token probability:",
    f"{answer_probabilities.mean().item():.4f}"
)


# =====================================================================
# STEP 24 — SAVE CHECKPOINT
# =====================================================================

checkpoint = {

    "model_name":
        MODEL_NAME,

    "hidden_dimension":
        hidden_dimension,

    "retention_ratio":
        RETENTION_RATIO,

    "target_budget":
        target_budget,

    "temperature":
        TEMPERATURE,

    "state_dict":
        retention_scorer.state_dict()
}


torch.save(
    checkpoint,
    CHECKPOINT_PATH
)


print_header(
    "STEP 24 — CHECKPOINT"
)


print(
    "Saved:",
    os.path.abspath(
        CHECKPOINT_PATH
    )
)


# =====================================================================
# FINAL SUMMARY
# =====================================================================

print_header(
    "FINAL SUMMARY"
)


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
    "Adaptive tokens:",
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
    "Expected retained:",
    f"{expected_retained_tokens:.3f}"
)

print(
    "Expected retention:",
    f"{expected_ratio * 100:.2f}%"
)

print(
    "Ground truth:",
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