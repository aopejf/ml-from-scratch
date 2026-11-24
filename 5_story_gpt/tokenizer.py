"""
A word-piece tokenizer, written from scratch. Its technical name is
byte-pair encoding (BPE), the same method GPT models use.

Project 3 fed the model one letter at a time. That works, but the model
spends most of its effort learning to spell. Here the text is cut into
common chunks instead ("the", " want", "ed", " happy"), so the model can
spend its effort on what the words mean.

How the chunks are found:

  1. Start with single bytes (letters, spaces, punctuation).
  2. Count every pair of neighbouring pieces in the text.
  3. Glue the most common pair into one new piece.
  4. Repeat until there are as many pieces as you asked for.

Common words end up as one piece. Rare words stay as several.
"""

import json
import re
from collections import Counter

END = "<|endoftext|>"  # the marker between stories
# Cuts text into words, each keeping the space in front of it.
WORDS = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+")


class Tokenizer:
    def __init__(self, merges=None):
        self.merges = [tuple(m) for m in (merges or [])]  # pairs, in the order they were glued
        self.pieces = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.pieces.append(self.pieces[a] + self.pieces[b])
        self.rank = {pair: 256 + i for i, pair in enumerate(self.merges)}
        self.end_token = len(self.pieces)  # one extra number that means "story over"
        self.vocab_size = len(self.pieces) + 1
        self.cache = {}

    # ----- learning the pieces -------------------------------------------

    @classmethod
    def train(cls, text, vocab_size, report_every=500):
        word_counts = Counter()
        for story in text.split(END):
            word_counts.update(WORDS.findall(story))
        words = [list(w.encode("utf-8")) for w in word_counts]
        counts = list(word_counts.values())
        print(f"  {sum(counts):,} words, {len(words):,} different ones")

        pair_counts = Counter()  # how often each neighbouring pair appears
        pair_where = {}  # which words contain it
        for i, (word, n) in enumerate(zip(words, counts)):
            for pair in zip(word, word[1:]):
                pair_counts[pair] += n
                pair_where.setdefault(pair, set()).add(i)

        merges = []
        pieces = [bytes([i]) for i in range(256)]
        while 256 + len(merges) + 1 < vocab_size and pair_counts:
            best = max(pair_counts, key=pair_counts.get)
            if pair_counts[best] < 2:
                break
            new_piece = 256 + len(merges)
            merges.append(best)
            pieces.append(pieces[best[0]] + pieces[best[1]])
            for i in list(pair_where.pop(best, ())):
                word, n = words[i], counts[i]
                for pair in zip(word, word[1:]):  # forget this word's old pairs
                    pair_counts[pair] -= n
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                merged, j = [], 0
                while j < len(word):  # glue every occurrence of the pair
                    if j + 1 < len(word) and (word[j], word[j + 1]) == best:
                        merged.append(new_piece)
                        j += 2
                    else:
                        merged.append(word[j])
                        j += 1
                words[i] = merged
                for pair in zip(merged, merged[1:]):  # count its new pairs
                    pair_counts[pair] += n
                    pair_where.setdefault(pair, set()).add(i)
            if len(merges) % report_every == 0:
                print(f"  {len(merges):5d} pieces learned, latest: {pieces[-1].decode('utf-8', 'replace')!r}", flush=True)
        return cls(merges)

    # ----- using them ----------------------------------------------------

    def _encode_word(self, word):
        tokens = self.cache.get(word)
        if tokens is None:
            tokens = list(word.encode("utf-8"))
            while len(tokens) > 1:
                # Glue pairs in the same order they were learned.
                pair = min(zip(tokens, tokens[1:]), key=lambda p: self.rank.get(p, 1e9))
                if pair not in self.rank:
                    break
                new_piece, merged, j = self.rank[pair], [], 0
                while j < len(tokens):
                    if j + 1 < len(tokens) and (tokens[j], tokens[j + 1]) == pair:
                        merged.append(new_piece)
                        j += 2
                    else:
                        merged.append(tokens[j])
                        j += 1
                tokens = merged
            self.cache[word] = tokens
        return tokens

    def encode(self, text):
        """Text in, list of piece numbers out."""
        tokens = []
        for i, story in enumerate(text.split(END)):
            if i > 0:
                tokens.append(self.end_token)
            for word in WORDS.findall(story):
                tokens.extend(self._encode_word(word))
        return tokens

    def decode(self, tokens):
        """Piece numbers in, text out."""
        return b"".join(self.pieces[t] for t in tokens if t != self.end_token).decode("utf-8", "replace")

    def piece(self, token):
        return END if token == self.end_token else self.pieces[token].decode("utf-8", "replace")

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"merges": self.merges}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f)["merges"])
