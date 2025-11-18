import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

"""Unit tests for tokenizer and model."""

import pytest
import torch
from src.data.tokenizer import url_to_ids, MAX_LEN
from src.model.model import MiniTransformer

def test_token_length():
    """Checking if the function truncates so that the returned 
    token sequence has exactly MAX_LEN tokens."""
    assert len(url_to_ids("a" * 200))  == MAX_LEN

def test_padding():
    assert url_to_ids("hi")[-1] == 0  # padded

def test_model_output_shape():
    m = MiniTransformer()
    assert m(torch.randint(0, 96, (7, MAX_LEN))).shape == (7,)

def test_sigmoid_range():
    m = MiniTransformer()
    m.eval()
    with torch.no_grad():
        out = m(torch.randint(0, 96, (5, MAX_LEN)))
    assert out.min() >= 0.0 and out.max() <= 1.0

def test_overfit_small_batch():
    """Overfitting 32 samples to > 99 % accuracy."""
    m = MiniTransformer()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

    x = torch.randint(0, 96, (32, MAX_LEN))
    y = torch.randint(0, 2, (32,)).float()
    for _ in range(300):
        opt.zero_grad()
        loss = torch.nn.BCELoss()(m(x), y)
        loss.backward()
        opt.step()
        
    m.eval()
    with torch.no_grad():
        preds = (m(x) > 0.5).float()
        acc = (preds == y).float().mean().item()
    assert acc >= 0.99