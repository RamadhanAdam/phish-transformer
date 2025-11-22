"""End to end demo: train -> export -> api -> eval"""

import subprocess, sys, os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)

steps = [
    ("pre-process", "python src/data/preprocess.py"),
    ("train", "python src/training/train.py"),
    ("export", "python src/model/export.py"),
    ("test", "pytest tests"),
    ("eval", "python src/inference/evaluate.py")
]

for name, cmd in steps:
    print(f"\n ------ {name.upper()} ------")
    subprocess.run(cmd, shell = True, check = True)

print("\n All done. \n\n Run 'flask --app src/inference/app run --port 8000' to serve")