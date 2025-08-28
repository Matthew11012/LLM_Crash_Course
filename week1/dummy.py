import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# import your model (TinyGPT / Transformer) from your file
from model import TinyGPT  

# 1. Setup
torch.manual_seed(42)

vocab_size = 100
seq_len = 8
batch_size = 1

# Dummy input/output (just random ints as tokens)
x = torch.randint(0, vocab_size, (batch_size, seq_len))
y = torch.randint(0, vocab_size, (batch_size, seq_len))

# 2. Model
model = TinyGPT(
    d_model=32,
    n_heads=4,
    n_layers=2,
    vocab_size=vocab_size,
    max_seq_len=seq_len
)

# 3. Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 4. Training loop
for step in range(500):  # a few hundred steps
    optimizer.zero_grad()
    logits = model(x)  # (B, T, vocab_size)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}, loss = {loss.item():.4f}")
