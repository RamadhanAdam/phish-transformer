import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

"""training the mini-transformer on UCI URL data."""

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from data.tokenizer import url_to_ids
from model.model import MiniTransformer


def load_csv(path : str):
    """Return (token_ds, labels) tensors."""
    df = pd.read_csv(path)
    urls, labels = df["URL"].values, df["label"].values
    ids = torch.tensor([url_to_ids(u) for u in urls], dtype= torch.long)
    lbl = torch.tensor(labels, dtype=torch.float32)

    return ids, lbl

def train(model, train_loader, val_x, val_y, epochs: int = 5, lr: float = 3e-4):
    """Train with early-stop on best validation acc."""

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr =lr)
    loss_fn = nn.BCELoss()
    best_acc = 0.0

    for epoch in range(epochs):
        model.train() # activating training mode
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)

            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
        
        # validation
        model.eval()
        with torch.no_grad():
            preds = (model(val_x.to(device)) > 0.5).cpu()
            acc = (preds == val_y).float().mean().item()

            print(f"Epoch {epoch} : val-acc {acc:3f}")
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), "./models/phish_model.pt")

    print("Saved best model to phish_model.pt")      

if __name__ == "__main__":
    train_x, train_y = load_csv("./datasets/train.csv")
    val_x, val_y = load_csv("./datasets/val.csv")
    train_ds = TensorDataset(train_x, train_y)
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
    model = MiniTransformer()
    train(model, train_dl, val_x, val_y, epochs=5)