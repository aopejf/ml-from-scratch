"""
Gather everything the demo page needs into one folder, ready to put online.

    python make_space.py

It fills space_upload/ with the model, the code and two small setup files for
Hugging Face Spaces (a free host for machine learning demos). Upload the
contents of that folder to a new Space and the page is live on the internet.
The README has the steps.
"""

import shutil
from pathlib import Path

here = Path(__file__).parent
out = here / "space_upload"
shutil.rmtree(out, ignore_errors=True)
out.mkdir()

for file in ["demo.py", "demo.html", "model.py", "tokenizer.py", "model.pt", "tokenizer.json", "../3_mini_gpt/gpt.py"]:
    shutil.copy(here / file, out)

# Tells Hugging Face what this is and which port the page is served on.
(out / "README.md").write_text(
    """---
title: Story GPT from scratch
emoji: 📖
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

A 3.4 million parameter GPT that writes short children's stories, with the tokenizer and
attention written by hand and the model trained overnight on a laptop processor.

Type the start of a story and it finishes it.
Code and write-up: https://github.com/aopejf/ml-from-scratch/tree/main/5_story_gpt
""",
    encoding="utf-8",
)

# The recipe for the small computer that runs the page.
(out / "Dockerfile").write_text(
    """FROM python:3.12-slim
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" HOST=0.0.0.0 PORT=7860
WORKDIR /home/user/app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user . .
EXPOSE 7860
CMD ["python", "demo.py"]
""",
    encoding="utf-8",
    newline="\n",
)

print(f"Ready in {out}:")
for file in sorted(out.iterdir()):
    print(f"  {file.name:<16} {file.stat().st_size / 1e6:6.2f} MB")
