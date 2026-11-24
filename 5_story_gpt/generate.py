"""
Ask the trained model to continue a story.

    python generate.py "Once upon a time, a cat found a"
    python generate.py "Lily was sad because" 300      (300 = longest length, in word pieces)
"""

import sys

from model import load_trained, write

prompt = sys.argv[1] if len(sys.argv) > 1 else "Once upon a time"
length = int(sys.argv[2]) if len(sys.argv) > 2 else 200

model, tokenizer, _ = load_trained()
print(prompt, end="", flush=True)
write(model, tokenizer, prompt, length, on_piece=lambda piece: print(piece, end="", flush=True))
print()
