"""
export.py

Exporting a trained MiniTransformer PyTorch model to TorchScript format for inference.

Steps:
1. Loading the MiniTransformer model architecture.
2. Loading trained weights from 'phish_model.pt'.
3. Converting the model to TorchScript using torch.jit.script.
4. Saving the TorchScript model as 'phish_model_ts.pt'.
5. Printing the file size of the exported model.
"""

import torch
from pathlib import Path
import sys

# Adding project root to Python path for importing modules
project_root = Path(__file__).resolve().parents[2]  # go up to project root
sys.path.insert(0, str(project_root / "src"))

from model.model import MiniTransformer

def export():
    """
    Exporting the trained MiniTransformer to TorchScript.

    Requirements:
    - Having 'phish_model.pt' existing in the project root.

    Output:
    - Saving 'phish_model_ts.pt' in the project root.
    - Printing the file size of the exported model in KB.
    """

    # Initializing the model
    model = MiniTransformer()

    # Loading trained weights
    model_path = project_root / "models" / "phish_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    # Setting model to evaluation mode
    model.eval()

    # Converting the model to TorchScript
    with torch.no_grad():
        scripted = torch.jit.script(model)

    # Saving the TorchScript model
    output_path = project_root / "models" / "phish_model.pt"
    scripted.save(output_path)

    # Printing the model file size
    size_kb = len(scripted.save_to_buffer()) / 1024
    print(f"Exported TorchScript model to '{output_path}' | size: {size_kb:.1f} KB")

if __name__ == "__main__":
    export()