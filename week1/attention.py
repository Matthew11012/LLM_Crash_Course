import torch
import torch.nn as nn
import math
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=None):
        super().__init__()
        self.dropout = nn.droput(dropout) if dropout else None
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





