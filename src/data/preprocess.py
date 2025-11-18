"""Clean UCI dataset and create train/ val/ test splits."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

def clean_and_split(in_file : str = "./datasets/PhiUSIIL_Phishing_URL_Dataset.csv",
                    train_frac : float = 0.6, val_frac : float = 0.2, seed : int = 42):
    
    "I keep only URL + Label, I remove junk, and split 80/10/10"

    df = pd.read_csv(in_file, usecols = ["URL", "label"])
    df = df.dropna().drop_duplicates()
    df["URL"] = df["URL"].astype(str).str.strip()

    # I then drop tiny URLS
    df = df[df["URL"].str.len() > 10]

    # deterministic split and shuffling
    df = df.sample(frac=1, random_state = seed).reset_index(drop = True)
    n = len(df) # storing the number of rows in the dataset

    # Computing split boundaries
    train_end = int(train_frac * n)
    val_end = int((train_frac + val_frac) * n)

    # Position based slicing
    df.iloc[:train_end].to_csv("./datasets/train.csv", index = False)
    df.iloc[train_end:val_end].to_csv("./datasets/val.csv", index=False)
    df.iloc[val_end:].to_csv("./datasets/test.csv", index=False)

    print(f"Train: {train_end}, Val: {val_end-train_end}, Test: {n-val_end}")

if __name__ == "__main__":
    clean_and_split()