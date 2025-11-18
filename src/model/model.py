import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

# model.py
"""Minimal character-level transformer for binary classification."""
import math
import torch
import torch.nn as nn


class ByteEmbedding(nn.Module):
    """Simple embedding layer for byte tokens."""
    def __init__(self, vocab_size: int = 96, d_model: int = 32):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T)  ->  (B, T, d)
        return self.emb(x) * math.sqrt(self.emb.embedding_dim)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention (no mask needed)."""
    def __init__(self, d_model: int = 32, heads: int = 2):
        super().__init__()
        assert d_model % heads == 0
        self.d_model, self.heads = d_model, heads
        self.d_k = d_model // heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # (B,T,3d) -> 3×(B,T,d)
        q = q.view(B, T, self.heads, self.d_k).transpose(1, 2)  # (B,h,T,d_k)
        k = k.view(B, T, self.heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.heads, self.d_k).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v  # (B,h,T,d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, d)
        return self.out(out)


class MiniTransformer(nn.Module):
    """2-layer encoder + max-pool + sigmoid."""
    def __init__(self,
                 vocab_size: int = 96,
                 d_model: int = 32,
                 heads: int = 2,
                 layers: int = 2,
                 max_len: int = 75):
        super().__init__()
        self.embedding = ByteEmbedding(vocab_size, d_model)
        self.pos_enc = nn.Parameter(torch.randn(max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=64,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T)  long tensor
        x = self.embedding(x) + self.pos_enc  # (B,T,d)
        x = self.encoder(x)                   # (B,T,d)
        z = self.pool(x.transpose(1, 2)).squeeze(-1)  # (B,d)
        return 1 - torch.sigmoid(self.fc(z)).squeeze(-1)   # (B,) -> flipped to represent phishing probability well 


if __name__ == "__main__":
    m = MiniTransformer()
    print(m(torch.randint(0, 96, (4, 75))).shape)  # -> torch.Size([4])