import torch
import torch.nn as nn
import math
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=None):
        super().__init__()
        self.dropout = nn.Droput(dropout) if dropout else None
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)
        attention_scores = torch.matmul(q,k.transpose(-2,-1))/torch.sqrt(torch.tensor(d_k,dtype=torch.float32))

        # apply the mask (if provided)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, v)
        return output, attention_weights




# small dimension for clarity
d_k = 4
seq_len = 3
batch_size = 1

Q = torch.eye(seq_len).unsqueeze(0) # shape (1, 3, 3)
K = Q.clone()
V = torch.randn(batch_size, seq_len, d_k)  # random values

attn = ScaledDotProductAttention()
output, weights = attn(Q, K, V)

print("Attention weights:\n", weights)
print("Output:\n", output)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear layers to project Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Final layer after concatenatino
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)
    
    def forward(self, q, k, v, mask=None):
        # Project
        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)

        batch_size, seq_len, _ = Q.size()
        # Reshape for multi-head
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)

        # Apply scaled dot-product attention
        attention_output, attention_weights = self.attention(Q, K, V, mask)

        # Concatenate heads
        attention_output = attention_output.transpose(1,2).contiguous().view(batch_size, seq_len, self.d_model)

        # Final linear layer
        attention_output = self.W_o(attention_output)

        return attention_output, attention_weights


class MLP(nn.Module):
    def __init__(self, d_model, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * d_model
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model)
    
    def forward(self, x, mask=None):
        x_norm = self.ln1(x)
        # Attention block
        attention_output, _ = self.mha(x_norm, x_norm, x_norm, mask)
        x = x + attention_output

        # Feedforward block
        mlp_output = self.mlp(self.ln2(x))
        x = x + mlp_output

        return x
