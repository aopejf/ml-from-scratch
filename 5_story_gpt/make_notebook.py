"""
Writes train_colab.ipynb, the notebook that trains the bigger model on a free
Google Colab graphics card. Kept as a script so the notebook is easy to change.

    python make_notebook.py
"""

import json
from pathlib import Path

HOURS = 2.5


def text(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": "\n".join(lines)}


cells = [
    text(
        "# Train the bigger story GPT on a free graphics card",
        "",
        "The laptop model has 3.4 million adjustable numbers and trained for 6.5 hours on a laptop processor.",
        "This notebook trains a model four times the size (13.9 million) on a graphics card instead.",
        "",
        "**Before you start:** in the menu, choose **Runtime > Change runtime type > T4 GPU > Save**.",
        "",
        f"Then choose **Runtime > Run all**. It takes about {HOURS + 0.5:.0f} hours. Keep this browser tab open.",
        "At the end it downloads `model_big.pt` to your computer.",
    ),
    text("## 1. Check the graphics card is on"),
    code(
        "import torch",
        'assert torch.cuda.is_available(), "No graphics card. Choose Runtime > Change runtime type > T4 GPU, then run again."',
        "print(torch.cuda.get_device_name(0))",
    ),
    text("## 2. Get the code"),
    code(
        "!git clone --depth 1 https://github.com/aopejf/ml-from-scratch.git",
        "%cd ml-from-scratch/5_story_gpt",
    ),
    text("## 3. Download the stories (about 420 MB) and turn them into numbers", "", "It reuses the word pieces the laptop model learned, so the two models can be compared fairly."),
    code(
        "!mkdir -p data",
        "!curl -L -o data/valid.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
        "!curl -L -r 0-419430399 -o data/train_part.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
        "!python prepare_data.py --keep-tokenizer",
    ),
    text(
        "## 4. Train",
        "",
        "It prints the loss and a short sample of its writing every 100 steps.",
        "If Colab disconnects part-way, run steps 1 to 3 again and the work so far is lost; a free session usually lasts long enough.",
    ),
    code(f"!python train.py {HOURS} --big"),
    text("## 5. Read a story and download the model"),
    code(
        "from model import load_trained, write",
        'model, tokenizer, saved = load_trained("model_big.pt")',
        "print(f\"Loss on unseen stories: {saved['unseen_loss']:.3f}  (the laptop model scored 1.95)\\n\")",
        'print(write(model, tokenizer, "Once upon a time, there was a little dinosaur who", 250)[0])',
    ),
    code(
        "from google.colab import files",
        'for name in ["model_big.pt", "samples_big.md", "result_big.png"]:',
        "    files.download(name)",
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
out = Path(__file__).with_name("train_colab.ipynb")
out.write_text(json.dumps(notebook, indent=1), encoding="utf-8", newline="\n")
print(f"Wrote {out.name}")
