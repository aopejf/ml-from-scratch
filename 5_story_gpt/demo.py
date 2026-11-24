"""
A page you can type into.

    python demo.py

Then open http://localhost:8000 in a browser. Type the start of a story and
the model finishes it, one word piece at a time. Under the story it shows the
pieces it was choosing between for the next word, with its odds for each.

Uses only Python's built-in web server. Nothing leaves this computer.
"""

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlparse

import torch
import torch.nn.functional as F

from model import load_trained, write

here = Path(__file__).parent
HOST, PORT = os.environ.get("HOST", "127.0.0.1"), int(os.environ.get("PORT", 8000))  # a web host sets these
model, tokenizer, saved = load_trained()
one_at_a_time = Lock()
INFO = {
    "parameters": sum(p.numel() for p in model.parameters()),
    "pieces": tokenizer.vocab_size,
    "steps": saved.get("step", 0),
}


@torch.no_grad()
def next_piece_odds(text, how_many=8):
    """The model's favourite next pieces after `text`, with the odds it gives each."""
    tokens = tokenizer.encode(text)[-model.context :]
    if not tokens:
        return []
    scores, _ = model(torch.tensor([tokens], device=next(model.parameters()).device))
    odds = F.softmax(scores[0, -1], dim=-1)
    top = torch.topk(odds, how_many)
    return [{"piece": tokenizer.piece(i), "odds": round(p, 4)} for p, i in zip(top.values.tolist(), top.indices.tolist())]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, body, kind):
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        ask = parse_qs(url.query)
        if url.path == "/":
            self.send((here / "demo.html").read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/info":
            self.send(json.dumps(INFO).encode(), "application/json")
        elif url.path == "/write":
            prompt = ask.get("prompt", ["Once upon a time"])[0][:2000]
            temperature = min(max(float(ask.get("temperature", ["0.8"])[0]), 0.1), 1.5)
            length = min(max(int(ask.get("length", ["200"])[0]), 1), 400)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")  # asks a web host to pass each piece on at once
            self.end_headers()

            def say(message):
                self.wfile.write(f"data: {json.dumps(message)}\n\n".encode())
                self.wfile.flush()

            try:
                with one_at_a_time:
                    story, _ = write(model, tokenizer, prompt, length, temperature, on_piece=lambda piece: say({"piece": piece}))
                    say({"done": True, "odds": next_piece_odds(story)})
            except (BrokenPipeError, ConnectionError):
                pass  # the browser tab was closed mid-story
        else:
            self.send_error(404)


if __name__ == "__main__":
    print(f"Model: {INFO['parameters']:,} adjustable numbers, trained for {INFO['steps']:,} steps")
    print(f"Open http://localhost:{PORT} in a browser. Press Ctrl+C here to stop.")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
