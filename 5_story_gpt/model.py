"""
The story model. It is the same MiniGPT class as project 3 (same hand-written
attention), built bigger and fed word pieces instead of letters.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "3_mini_gpt"))
from gpt import MiniGPT  # noqa: E402

from tokenizer import Tokenizer  # noqa: E402

here = Path(__file__).parent

# The size of the model. Project 3 used context 128, width 128, 4 heads, 4 layers.
# Sized so that one night on a laptop processor is enough to train it.
CONFIG = dict(context=128, width=192, n_heads=6, n_layers=4, dropout=0.0)


def new_model(vocab_size, config=CONFIG):
    return MiniGPT(vocab_size, **config)


@torch.no_grad()
def write(model, tokenizer, prompt, max_pieces=200, temperature=0.8, top_k=40, on_piece=None):
    """
    Continue `prompt`. Stops at the end-of-story marker or after max_pieces.
    top_k: only the 40 likeliest next pieces are considered, which stops the
    occasional wild guess from derailing the story.
    """
    model.eval()
    tokens = tokenizer.encode(prompt) or [tokenizer.end_token]
    start = len(tokens)
    for _ in range(max_pieces):
        recent = torch.tensor([tokens[-model.context :]])
        scores, _ = model(recent)
        scores = scores[0, -1] / temperature
        if top_k:
            cutoff = torch.topk(scores, top_k).values[-1]
            scores[scores < cutoff] = float("-inf")
        next_token = torch.multinomial(F.softmax(scores, dim=-1), 1).item()
        if next_token == tokenizer.end_token:
            break
        tokens.append(next_token)
        if on_piece:
            on_piece(tokenizer.piece(next_token))
    model.train()
    return tokenizer.decode(tokens), tokenizer.decode(tokens[start:])


def load_trained(path=None):
    """Load a finished (or part-trained) model and its tokenizer."""
    tokenizer = Tokenizer.load(here / "tokenizer.json")
    path = Path(path) if path else here / "model.pt"
    if not path.exists():
        path = here / "checkpoint.pt"
    saved = torch.load(path, map_location="cpu")
    model = new_model(tokenizer.vocab_size, saved["config"])
    model.load_state_dict(saved["model"])
    model.eval()
    return model, tokenizer, saved
