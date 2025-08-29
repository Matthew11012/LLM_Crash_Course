import torch
from torch.utils.data import Dataset

class CharTokenizer:
    def __init__(self, text):
        # build vocab
        self.chars = sorted(list(set(text)))
        self.stoi = {ch:i for i,ch in enumerate(text)}
        self.itos = {i:ch for i,ch in enumerate(text)}
        self.vocab_size = len(self.chars)
    
    def encode(self, s):
        return [self.stoi[ch] for ch in s]
    
    def decode(self, s):
        return [self.itos[ch] for ch in s]

class CharDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len=16):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.data = tokenizer.encode(text)

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+self.seq_len+1], dtype=torch.long)
        return x, y
    
