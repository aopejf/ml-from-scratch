"""
Pack the trained model so a web page can run it with no server.

    python export_web.py

Writes two files into ../docs (the folder GitHub Pages serves):
    model.bin    every adjustable number, one after another, as 16-bit decimals
    model.json   the model's size, where each block of numbers starts, and the tokenizer

docs/story-model.js reads these and does the model's arithmetic in JavaScript.
It also writes docs/check.json: the scores this Python model gives for a few
prompts, so the JavaScript version can be checked against it.
"""

import json
from pathlib import Path

import numpy as np
import torch

from model import load_trained

here = Path(__file__).parent
out = here.parent / "docs"
out.mkdir(exist_ok=True)

model, tokenizer, saved = load_trained()
tensors, numbers, offset = {}, [], 0
for name, tensor in model.state_dict().items():
    if name.endswith(".mask"):
        continue  # the "no looking ahead" triangle; the page gets that for free by only ever looking back
    values = tensor.numpy().astype(np.float16).ravel()
    tensors[name] = {"shape": list(tensor.shape), "offset": offset}
    numbers.append(values)
    offset += len(values)

np.concatenate(numbers).tofile(out / "model.bin")
info = {
    "config": saved["config"],
    "vocab_size": tokenizer.vocab_size,
    "end_token": tokenizer.end_token,
    "parameters": offset,
    "steps": saved.get("step", 0),
    "tensors": tensors,
    "merges": tokenizer.merges,
}
(out / "model.json").write_text(json.dumps(info), encoding="utf-8")

# Reference answers from the Python model, using the same rounded numbers the page will get.
for name, tensor in model.state_dict().items():
    if not name.endswith(".mask"):
        tensor.copy_(tensor.half().float())
checks = []
for prompt in ["Once upon a time, there was a little dinosaur who", "Lily found a shiny key in the garden.", "The robot was sad because", "Café ☕ time! 123"]:
    tokens = tokenizer.encode(prompt)
    with torch.no_grad():
        scores, _ = model(torch.tensor([tokens]))
    checks.append({"prompt": prompt, "tokens": tokens, "scores": [round(s, 4) for s in scores[0, -1].tolist()]})
(out / "check.json").write_text(json.dumps(checks), encoding="utf-8")

print(f"Wrote {offset:,} numbers to docs/model.bin ({(out / 'model.bin').stat().st_size / 1e6:.1f} MB), plus model.json and check.json")
