"""
Make the trained model write, without retraining it.

    python generate.py                 500 characters starting from nothing
    python generate.py "ROMEO:"        continue from your own opening
    python generate.py "ROMEO:" 1000   and choose the length
"""

import sys
from pathlib import Path

import torch

from gpt import MiniGPT

saved = torch.load(Path(__file__).parent / "model.pt")
characters = saved["characters"]
to_number = {ch: i for i, ch in enumerate(characters)}

model = MiniGPT(vocab_size=len(characters), context=saved["context"])
model.load_state_dict(saved["weights"])

opening = sys.argv[1] if len(sys.argv) > 1 else "\n"
length = int(sys.argv[2]) if len(sys.argv) > 2 else 500

unknown = sorted(set(opening) - set(characters))
if unknown:
    sys.exit(f"The model has never seen these characters: {unknown}")

tokens = torch.tensor([[to_number[ch] for ch in opening]])
written = model.generate(tokens, length)[0].tolist()
print("".join(characters[i] for i in written))
