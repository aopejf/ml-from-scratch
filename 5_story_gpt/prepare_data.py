"""
Step 1 of the story GPT: learn the word pieces, then turn the stories into numbers.

    python prepare_data.py
    python prepare_data.py --keep-tokenizer   reuse the word pieces already in tokenizer.json

Reads   data/train_part.txt, data/valid.txt   (TinyStories)
Writes  tokenizer.json                        the learned word pieces
        data/train.bin, data/valid.bin        the stories as piece numbers
"""

import sys
import time
from array import array
from pathlib import Path

import numpy as np

from tokenizer import END, Tokenizer

here = Path(__file__).parent
VOCAB_SIZE = 4096
LEARN_FROM_MB = 60  # the pieces are learned from the first part of the text; more changes little

start = time.time()
print("Reading the stories...")
text = (here / "data" / "train_part.txt").read_bytes().decode("utf-8", "replace")
text = text[: text.rindex(END)]  # the download stops mid-story; drop the unfinished one
valid_text = (here / "data" / "valid.txt").read_bytes().decode("utf-8", "replace")
print(f"  {len(text) / 1e6:.0f} million characters to train on, {len(valid_text) / 1e6:.0f} million held back")

if "--keep-tokenizer" in sys.argv:
    tokenizer = Tokenizer.load(here / "tokenizer.json")
    print(f"\nUsing the {tokenizer.vocab_size:,} word pieces already learned")
else:
    print(f"\nLearning {VOCAB_SIZE:,} word pieces...")
    tokenizer = Tokenizer.train(text[: LEARN_FROM_MB * 1_000_000], VOCAB_SIZE)
    tokenizer.save(here / "tokenizer.json")
    print(f"  done, {tokenizer.vocab_size:,} pieces   ({time.time() - start:.0f}s)")

example = "Once upon a time, there was a little dinosaur who loved pancakes."
pieces = [tokenizer.piece(t) for t in tokenizer.encode(example)]
print(f"\nExample: {example}")
print("  becomes " + "|".join(pieces))
print(f"  {len(example)} letters -> {len(pieces)} pieces")
assert tokenizer.decode(tokenizer.encode(example)) == example


def to_numbers(text, name):
    numbers = array("H")
    stories = text.split(END)
    for i, story in enumerate(stories):
        numbers.extend(tokenizer.encode(story))
        numbers.append(tokenizer.end_token)
        if i % 100_000 == 0 and i:
            print(f"  {name}: {i:,} of {len(stories):,} stories   ({time.time() - start:.0f}s)", flush=True)
    result = np.frombuffer(numbers, dtype=np.uint16)
    result.tofile(here / "data" / f"{name}.bin")
    print(f"  {name}: {len(stories):,} stories, {len(result):,} pieces, {len(text) / len(result):.2f} letters per piece")
    return result


print("\nTurning the stories into numbers...")
valid = to_numbers(valid_text, "valid")
assert tokenizer.decode(valid[:2000].tolist()) == valid_text.replace(END, "")[: len(tokenizer.decode(valid[:2000].tolist()))]
train = to_numbers(text, "train")
print(f"\nFinished in {(time.time() - start) / 60:.1f} minutes")
