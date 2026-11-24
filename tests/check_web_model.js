/*
Checks the JavaScript story model against the Python one.

    node tests/check_web_model.js

docs/check.json holds the scores the Python model gave for a few prompts
(written by 5_story_gpt/export_web.py). This runs the same prompts through
docs/story-model.js and fails if the pieces or the scores differ.
*/

const fs = require("fs");
const path = require("path");
const { Tokenizer, StoryModel, unpackFloat16, softmax } = require("../docs/story-model.js");

const docs = path.join(__dirname, "..", "docs");
const info = JSON.parse(fs.readFileSync(path.join(docs, "model.json")));
const file = fs.readFileSync(path.join(docs, "model.bin"));
const numbers = unpackFloat16(file.buffer.slice(file.byteOffset, file.byteOffset + file.length));
const model = new StoryModel(info, numbers);
const tokenizer = new Tokenizer(info.merges);

let failed = false;
for (const check of JSON.parse(fs.readFileSync(path.join(docs, "check.json")))) {
  const tokens = tokenizer.encode(check.prompt);
  const sameTokens = JSON.stringify(tokens) === JSON.stringify(check.tokens);
  const started = performance.now();
  const scores = model.readAll(tokens);
  const perPiece = (performance.now() - started) / tokens.length;
  let worst = 0;
  for (let i = 0; i < scores.length; i++) worst = Math.max(worst, Math.abs(scores[i] - check.scores[i]));
  const mine = softmax(scores), theirs = softmax(check.scores);
  let worstOdds = 0;
  for (let i = 0; i < mine.length; i++) worstOdds = Math.max(worstOdds, Math.abs(mine[i] - theirs[i]));
  const ok = sameTokens && worst < 0.01;
  failed ||= !ok;
  console.log(`${ok ? "ok  " : "FAIL"} ${JSON.stringify(check.prompt)}  pieces match: ${sameTokens}  biggest score gap: ${worst.toFixed(5)}  biggest odds gap: ${(worstOdds * 100).toFixed(4)}%  ${perPiece.toFixed(1)} ms per piece`);
}
process.exit(failed ? 1 : 0);
