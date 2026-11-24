"""
Project 3: a miniature GPT.

The model has one job: given some text, predict the next character.
Do that well enough, feed its own guesses back in, and it writes.

From here on PyTorch does two jobs that projects 1 and 2 did by hand:
  - loss.backward()   the autograd engine from project 2, but fast
  - the optimizer     Adam from project 1

The attention mechanism below is written out step by step. PyTorch has a
ready-made attention layer; it is deliberately not used.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Lets every character look back at the characters before it and decide
    which ones matter for guessing what comes next.

    Each character produces three lists of numbers:
      query   "what am I looking for?"
      key     "what do I contain?"
      value   "what I will hand over if you pick me"

    A character's query is compared with every earlier character's key.
    A strong match means "pay attention to that one". The character then
    takes a blend of the values, weighted by how strong each match was.

    Several "heads" do this side by side, each free to look for something
    different (one might track the last vowel, another the start of the line).
    """

    def __init__(self, width, n_heads, context, dropout):
        super().__init__()
        assert width % n_heads == 0
        self.n_heads = n_heads
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)
        # The mask: a triangle of 1s. Position 5 may look at positions 0-5, never 6+.
        # Without it the model could cheat by reading the answer.
        self.register_buffer("mask", torch.tril(torch.ones(context, context)))
        self.last_weights = None  # kept so the attention can be drawn later

    def forward(self, x):
        batch, length, width = x.shape
        head_width = width // self.n_heads

        def split_into_heads(t):
            return t.view(batch, length, self.n_heads, head_width).transpose(1, 2)

        q = split_into_heads(self.query(x))
        k = split_into_heads(self.key(x))
        v = split_into_heads(self.value(x))

        # 1. Compare every query with every key (a dot product). Big number = relevant.
        scores = q @ k.transpose(-2, -1) / math.sqrt(head_width)
        # 2. Hide the future: scores for later positions become "minus infinity".
        scores = scores.masked_fill(self.mask[:length, :length] == 0, float("-inf"))
        # 3. Turn each row of scores into weights that add up to 1.
        weights = F.softmax(scores, dim=-1)
        self.last_weights = weights.detach()
        # 4. Blend the values using those weights.
        blended = self.dropout(weights) @ v

        blended = blended.transpose(1, 2).contiguous().view(batch, length, width)
        return self.dropout(self.output(blended))


class Block(nn.Module):
    """
    One layer of the model: gather information (attention), then think about
    it (a small ordinary network). The "x + ..." means each step adds to what
    is already there instead of replacing it, which keeps deep stacks trainable.
    """

    def __init__(self, width, n_heads, context, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = SelfAttention(width, n_heads, context, dropout)
        self.norm2 = nn.LayerNorm(width)
        self.think = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.think(self.norm2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, context=128, width=128, n_heads=4, n_layers=4, dropout=0.1):
        super().__init__()
        self.context = context
        # Every character gets a list of numbers describing "what I am"...
        self.char_embedding = nn.Embedding(vocab_size, width)
        # ...and every position gets one describing "where I am".
        self.position_embedding = nn.Embedding(context, width)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(width, n_heads, context, dropout) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(width)
        # The last step: one score per possible next character.
        self.to_scores = nn.Linear(width, vocab_size)

    def forward(self, tokens, targets=None):
        batch, length = tokens.shape
        positions = torch.arange(length, device=tokens.device)
        x = self.dropout(self.char_embedding(tokens) + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x)
        scores = self.to_scores(self.final_norm(x))

        loss = None
        if targets is not None:
            # Cross-entropy, the loss from project 0, with 65 possible answers instead of 2.
            loss = F.cross_entropy(scores.view(-1, scores.size(-1)), targets.view(-1))
        return scores, loss

    @torch.no_grad()
    def generate(self, tokens, n_new, temperature=0.8):
        """Write n_new characters, one at a time, each based on everything so far."""
        self.eval()
        for _ in range(n_new):
            recent = tokens[:, -self.context :]  # the model can only see this far back
            scores, _ = self(recent)
            probabilities = F.softmax(scores[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)  # roll weighted dice
            tokens = torch.cat([tokens, next_token], dim=1)
        self.train()
        return tokens
