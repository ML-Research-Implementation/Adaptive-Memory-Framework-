import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast

# -------------------------------------------------------------------
# STEP 1: TOKENIZATION & INPUT PREPARATION
# -------------------------------------------------------------------
model_name = "distilbert-base-cased"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

text = "DistilBERT is fast!"
inputs = tokenizer(text, return_tensors="pt")

input_ids = inputs["input_ids"] # Shape: [1, sequence_length]
seq_len = input_ids.size(1)

print("=" * 60)
print("STEP 1: TOKENIZATION")
print("=" * 60)
print("Raw Text:  ", text)
print("Tokens:    ", tokenizer.convert_ids_to_tokens(input_ids[0]))
print("Token IDs: ", input_ids[0].tolist())
print("Tensor Shape:", input_ids.shape)


# -------------------------------------------------------------------
# STEP 2: EMBEDDING LAYER (Word Vector + Position Vector)
# -------------------------------------------------------------------
class DistilBertEmbeddingsFromScratch(nn.Module):
    def __init__(self, vocab_size=30522, max_position_embeddings=512, dim=768):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, dim)
        self.position_embeddings = nn.Embedding(max_position_embeddings, dim)
        self.LayerNorm = nn.LayerNorm(dim)
        
    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)
        
        # 1. Look up static word vector
        word_embeds = self.word_embeddings(input_ids)
        # 2. Look up position vector
        pos_embeds = self.position_embeddings(position_ids)
        
        # 3. Element-wise addition
        embeddings = word_embeds + pos_embeds
        embeddings = self.LayerNorm(embeddings)
        return embeddings

embedding_layer = DistilBertEmbeddingsFromScratch()
embeddings = embedding_layer(input_ids)

print("\n" + "=" * 60)
print("STEP 2: EMBEDDINGS")
print("=" * 60)
print("Output Embedding Shape:", embeddings.shape) # [1, seq_len, 768]


# -------------------------------------------------------------------
# STEP 3: SINGLE TRANSFORMER LAYER (Self-Attention + FFN)
# -------------------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim=768, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Linear projections for Query, Key, Value
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        self.out_lin = nn.Linear(dim, dim)
        
    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        
        # Create Q, K, V matrices
        q = self.q_lin(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_lin(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_lin(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores: (Q * K^T) / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Weighted sum over Values: Weights * V
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        return self.out_lin(context)

attention = MultiHeadSelfAttention()
contextual_vectors = attention(embeddings)

print("\n" + "=" * 60)
print("STEP 3: SELF-ATTENTION")
print("=" * 60)
print("Contextual Output Shape:", contextual_vectors.shape) # [1, seq_len, 768]


# -------------------------------------------------------------------
# STEP 4: KNOWLEDGE DISTILLATION (Softmax Temperature & Loss)
# -------------------------------------------------------------------
def distillation_loss_fn(student_logits, teacher_logits, labels, T=5.0, alpha=0.5):
    # 1. Soft targets loss (KL Divergence at Temperature T)
    soft_student = F.log_softmax(student_logits / T, dim=-1)
    soft_teacher = F.softmax(teacher_logits / T, dim=-1)
    distillation_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T ** 2)
    
    # 2. Hard label loss (Cross Entropy)
    student_loss = F.cross_entropy(student_logits, labels)
    
    # 3. Combined Loss
    return (alpha * distillation_loss) + ((1 - alpha) * student_loss)

# Simulated outputs for vocabulary size of 30522 tokens
teacher_logits = torch.randn(1, 30522) * 2.0  # Teacher (BERT) output
student_logits = torch.randn(1, 30522) * 1.5  # Student (DistilBERT) output
true_label = torch.tensor([3435])             # Token ID for "fast"

loss = distillation_loss_fn(student_logits, teacher_logits, true_label, T=5.0)

print("\n" + "=" * 60)
print("STEP 4: KNOWLEDGE DISTILLATION LOSS")
print("=" * 60)
print("Calculated Distillation Loss:", loss.item())


# -------------------------------------------------------------------
# STEP 5: TASK HEAD (Question Answering Head)
# -------------------------------------------------------------------
class QAHead(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        # Maps each 768-D vector to 2 scores: [start_logit, end_logit]
        self.qa_outputs = nn.Linear(dim, 2)
        
    def forward(self, hidden_states):
        logits = self.qa_outputs(hidden_states) # Shape: [1, seq_len, 2]
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)

qa_head = QAHead()
start_logits, end_logits = qa_head(contextual_vectors)

print("\n" + "=" * 60)
print("STEP 5: QA TASK HEAD")
print("=" * 60)
print("Start Logits Shape:", start_logits.shape) # [1, seq_len]
print("End Logits Shape:  ", end_logits.shape)   # [1, seq_len]
print("Predicted Start Token Index:", torch.argmax(start_logits).item())
print("Predicted End Token Index:  ", torch.argmax(end_logits).item())
