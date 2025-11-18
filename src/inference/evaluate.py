import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

"""Evaluating the model on test set"""

# Adding project root to PYTHONPATH
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import pandas as pd
import sklearn.metrics
import matplotlib.pyplot as plt
from data.tokenizer import url_to_ids

model = torch.jit.load("./models/phish_model_ts.pt")
test_df = pd.read_csv("./datasets/test.csv")
test_x = torch.tensor([url_to_ids(u) for u in test_df["URL"]], dtype = torch.long)

with torch.no_grad():
    scores = model(test_x).numpy()

y_true = test_df["label"].values

fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, scores)
auc = sklearn.metrics.roc_auc_score(y_true, scores)

plt.plot(fpr, tpr, label = f"AUC = {auc:.3f}")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.legend()
plt.savefig("roc.png")
print(f"Test AUC: {auc:.3f} Accuracy: {sklearn.metrics.accuracy_score(y_true, scores > 0.5):.3f}")