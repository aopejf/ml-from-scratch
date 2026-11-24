"""
Score the story model with numbers, not just by reading its stories.

    python evaluate.py            200 stories
    python evaluate.py 50         a quicker run

For each test, the model is given the opening of a story it has never seen
(the first two sentences) and asked to finish it. Its endings are scored on
four things, and the real endings, written for the dataset, get the same
scores so there is something to compare against.

  Finishes        does it bring the story to an end by itself?
  Keeps the hero  does the name from the opening show up again?
  Repeats itself  share of four-word phrases it has already used (lower is better)
  New names       how many names turn up that were never introduced (lower is better)

Writes evaluation.md.
"""

import re
import sys
from pathlib import Path

import numpy as np
import torch

from model import load_trained, write
from tokenizer import END

here = Path(__file__).parent
N_STORIES = int(sys.argv[1]) if len(sys.argv) > 1 else 200
MAX_PIECES = 300
NOT_NAMES = set(
    "The A An One Once He She It They We You I But And So Then When After Suddenly Finally There This That His Her "
    "Their Every Each All As At In On From With Now Soon Later Just Thank Thanks Yes No Oh Wow Let Can Do What Why How "
    "Please Mom Dad Mommy Daddy Mama Papa Grandma Grandpa If Not My Your Our Maybe Look Come Go Don Sorry Hello Hi Okay "
    "Of To For Is Was Are Were Be Will Would Could Should Have Had Has Did Who Where Here Today Tomorrow Yesterday "
    "Inside Outside Together Everyone Everything Nobody Even Because While Before Still Also Sometimes Always Never "
    "Many Some Both Two Three Four Five Good Great Little Big Mr Mrs Miss".split()
)


def names_in(text):
    return {w for w in re.findall(r"\b[A-Z][a-z]{2,}\b", text) if w not in NOT_NAMES}


def repeated_phrases(text):
    words = re.findall(r"[a-z']+", text.lower())
    phrases = [tuple(words[i : i + 4]) for i in range(len(words) - 3)]
    return 1 - len(set(phrases)) / len(phrases) if phrases else 0.0


def score(opening, ending, finished):
    hero = names_in(opening)
    return {
        "Finishes the story": float(finished),
        "Keeps the hero": float(bool(hero & names_in(ending))),
        "Repeats itself": repeated_phrases(ending),
        "New names per story": float(len(names_in(ending) - hero)),
        "Length in words": float(len(ending.split())),
    }


model, tokenizer, saved = load_trained()
torch.manual_seed(0)
rng = np.random.default_rng(0)

# Openings: unseen stories whose first two sentences introduce a named character.
stories = [s.strip() for s in (here / "data" / "valid.txt").read_text(encoding="utf-8", errors="replace").split(END)]
tests = []
for i in rng.permutation(len(stories)):
    parts = re.split(r"(?<=[.!?])\s+", stories[i], maxsplit=2)
    if len(parts) == 3 and names_in(parts[0] + " " + parts[1]) and 40 < len(parts[2].split()) < 220:
        tests.append((parts[0] + " " + parts[1], parts[2]))
    if len(tests) == N_STORIES:
        break
print(f"Scoring {len(tests)} unseen story openings...", flush=True)

model_scores, real_scores, examples = [], [], []
for n, (opening, real_ending) in enumerate(tests, 1):
    count = [0]
    _, ending = write(model, tokenizer, opening, MAX_PIECES, on_piece=lambda piece: count.__setitem__(0, count[0] + 1))
    model_scores.append(score(opening, ending, count[0] < MAX_PIECES))
    real_scores.append(score(opening, real_ending, True))
    if n <= 3:
        examples.append((opening, ending))
    if n % 25 == 0:
        print(f"  {n} of {len(tests)}", flush=True)

rows = []
for key in model_scores[0]:
    ours, real = np.mean([s[key] for s in model_scores]), np.mean([s[key] for s in real_scores])
    as_percent = key in ("Finishes the story", "Keeps the hero", "Repeats itself")
    show = (lambda v: f"{v:.0%}") if as_percent else (lambda v: f"{v:.1f}")
    rows.append((key, show(ours), show(real)))

print(f"\n{'':24s}{'This model':>12s}{'Real stories':>14s}")
for key, ours, real in rows:
    print(f"{key:24s}{ours:>12s}{real:>14s}")

with open(here / "evaluation.md", "w", encoding="utf-8") as f:
    f.write("# Story GPT: scored test\n\n")
    f.write(f"{len(tests)} openings of stories the model never saw. The model wrote an ending for each. ")
    f.write("The real endings were scored the same way for comparison.\n\n")
    f.write("| | This model | Real stories |\n|---|---|---|\n")
    for key, ours, real in rows:
        f.write(f"| {key} | {ours} | {real} |\n")
    f.write(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters, {saved.get('step', 0):,} training steps.\n")
    f.write("\n## Three of the endings it wrote\n")
    for opening, ending in examples:
        f.write(f"\n**{opening}**{ending}\n")
print("\nSaved evaluation.md")
