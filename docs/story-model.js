/*
The story GPT, written a second time in plain JavaScript so it can run inside
a web page with no server and no libraries.

It is the same arithmetic as 3_mini_gpt/gpt.py and 5_story_gpt/tokenizer.py:
  Tokenizer   text <-> word pieces (byte-pair encoding)
  StoryModel  word pieces in, a score for every possible next piece out

One difference from the Python version, for speed: it reads one piece at a
time and remembers what each layer worked out for the earlier pieces (a
"key/value cache"), instead of re-reading the whole story for every new word.
The answers are the same; check.html proves it against the Python model.
*/

const END = "<|endoftext|>";
const WORDS = / ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+/g;

class Tokenizer {
  constructor(merges) {
    this.pieces = [];
    for (let i = 0; i < 256; i++) this.pieces.push(Uint8Array.of(i));
    this.rank = new Map(); // pair -> the piece it becomes, in the order the pairs were learned
    merges.forEach(([a, b], i) => {
      const glued = new Uint8Array(this.pieces[a].length + this.pieces[b].length);
      glued.set(this.pieces[a]);
      glued.set(this.pieces[b], this.pieces[a].length);
      this.pieces.push(glued);
      this.rank.set(a * 65536 + b, 256 + i);
    });
    this.endToken = this.pieces.length;
    this.vocabSize = this.pieces.length + 1;
    this.cache = new Map();
    this.utf8 = new TextEncoder();
  }

  encodeWord(word) {
    let tokens = this.cache.get(word);
    if (tokens) return tokens;
    tokens = Array.from(this.utf8.encode(word));
    while (tokens.length > 1) {
      // Glue pairs in the same order they were learned: find the earliest-learned pair present.
      let best = Infinity, bestA = 0, bestB = 0;
      for (let i = 0; i + 1 < tokens.length; i++) {
        const rank = this.rank.get(tokens[i] * 65536 + tokens[i + 1]);
        if (rank !== undefined && rank < best) { best = rank; bestA = tokens[i]; bestB = tokens[i + 1]; }
      }
      if (best === Infinity) break;
      const merged = [];
      for (let i = 0; i < tokens.length; ) {
        if (i + 1 < tokens.length && tokens[i] === bestA && tokens[i + 1] === bestB) { merged.push(best); i += 2; }
        else { merged.push(tokens[i]); i += 1; }
      }
      tokens = merged;
    }
    this.cache.set(word, tokens);
    return tokens;
  }

  encode(text) {
    const tokens = [];
    text.split(END).forEach((story, i) => {
      if (i > 0) tokens.push(this.endToken);
      for (const word of story.match(WORDS) || []) tokens.push(...this.encodeWord(word));
    });
    return tokens;
  }

  bytes(token) { return token === this.endToken ? new Uint8Array(0) : this.pieces[token]; }
  piece(token) { return token === this.endToken ? END : new TextDecoder().decode(this.pieces[token]); }
}

// The file stores each number in 16 bits to halve the download. This unpacks them.
function unpackFloat16(buffer) {
  const packed = new Uint16Array(buffer), out = new Float32Array(packed.length);
  for (let i = 0; i < packed.length; i++) {
    const h = packed[i], sign = h & 0x8000 ? -1 : 1, exponent = (h >> 10) & 31, fraction = h & 1023;
    out[i] = sign * (exponent === 0 ? fraction * 2 ** -24 : exponent === 31 ? (fraction ? NaN : Infinity) : (1 + fraction / 1024) * 2 ** (exponent - 15));
  }
  return out;
}

// y = W x + b, where W is stored row by row.
function linear(W, b, x, nOut, y) {
  const nIn = x.length;
  for (let o = 0, w = 0; o < nOut; o++) {
    let sum = b ? b[o] : 0;
    for (let i = 0; i < nIn; i++, w++) sum += W[w] * x[i];
    y[o] = sum;
  }
  return y;
}

// Rescale a list of numbers to average 0 and spread 1, then apply a learned stretch and shift.
function layerNorm(x, gain, shift, y) {
  const n = x.length;
  let mean = 0, spread = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  for (let i = 0; i < n; i++) spread += (x[i] - mean) ** 2;
  const scale = 1 / Math.sqrt(spread / n + 1e-5);
  for (let i = 0; i < n; i++) y[i] = (x[i] - mean) * scale * gain[i] + shift[i];
  return y;
}

// GELU: the smooth on/off switch between the two halves of each layer's "thinking" step.
function gelu(x) {
  const z = Math.abs(x) / Math.SQRT2, t = 1 / (1 + 0.3275911 * z);
  const erf = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-z * z);
  return 0.5 * x * (1 + (x < 0 ? -erf : erf));
}

class StoryModel {
  constructor(info, numbers) {
    Object.assign(this, { context: info.config.context, width: info.config.width, heads: info.config.n_heads, layers: info.config.n_layers, vocab: info.vocab_size });
    const get = name => {
      const { shape, offset } = info.tensors[name];
      return numbers.subarray(offset, offset + shape.reduce((a, b) => a * b, 1));
    };
    this.pieceEmbedding = get("char_embedding.weight");
    this.positionEmbedding = get("position_embedding.weight");
    this.blocks = [];
    for (let l = 0; l < this.layers; l++) {
      const p = `blocks.${l}.`;
      this.blocks.push({
        norm1: [get(p + "norm1.weight"), get(p + "norm1.bias")],
        query: get(p + "attention.query.weight"), key: get(p + "attention.key.weight"), value: get(p + "attention.value.weight"),
        out: [get(p + "attention.output.weight"), get(p + "attention.output.bias")],
        norm2: [get(p + "norm2.weight"), get(p + "norm2.bias")],
        think1: [get(p + "think.0.weight"), get(p + "think.0.bias")],
        think2: [get(p + "think.2.weight"), get(p + "think.2.bias")],
        keys: new Float32Array(this.context * this.width), // what each earlier piece "contains"
        values: new Float32Array(this.context * this.width), // what each earlier piece hands over
      });
    }
    this.finalNorm = [get("final_norm.weight"), get("final_norm.bias")];
    this.toScores = [get("to_scores.weight"), get("to_scores.bias")];
    const W = this.width;
    this.x = new Float32Array(W); this.h = new Float32Array(W); this.q = new Float32Array(W);
    this.blend = new Float32Array(W); this.added = new Float32Array(W); this.wide = new Float32Array(4 * W);
    this.weights = new Float32Array(this.context); this.scores = new Float32Array(this.vocab);
    this.length = 0;
  }

  reset() { this.length = 0; }

  // Read one more piece. Returns a score for every piece that could come next.
  read(token) {
    const { width: W, heads: H, x, h, q, blend, added, wide, weights } = this;
    const at = this.length, headWidth = W / H, scale = 1 / Math.sqrt(headWidth);
    for (let i = 0; i < W; i++) x[i] = this.pieceEmbedding[token * W + i] + this.positionEmbedding[at * W + i];

    for (const block of this.blocks) {
      // Attention: look back over every piece so far and gather what is relevant.
      layerNorm(x, block.norm1[0], block.norm1[1], h);
      linear(block.query, null, h, W, q);
      linear(block.key, null, h, W, block.keys.subarray(at * W, at * W + W));
      linear(block.value, null, h, W, block.values.subarray(at * W, at * W + W));
      for (let head = 0; head < H; head++) {
        const from = head * headWidth;
        let biggest = -Infinity;
        for (let j = 0; j <= at; j++) { // only positions up to this one: it cannot look ahead
          let match = 0;
          for (let i = 0; i < headWidth; i++) match += q[from + i] * block.keys[j * W + from + i];
          weights[j] = match * scale;
          if (weights[j] > biggest) biggest = weights[j];
        }
        let total = 0;
        for (let j = 0; j <= at; j++) { weights[j] = Math.exp(weights[j] - biggest); total += weights[j]; }
        for (let i = 0; i < headWidth; i++) {
          let sum = 0;
          for (let j = 0; j <= at; j++) sum += weights[j] * block.values[j * W + from + i];
          blend[from + i] = sum / total;
        }
      }
      linear(block.out[0], block.out[1], blend, W, added);
      for (let i = 0; i < W; i++) x[i] += added[i];

      // Thinking: a small ordinary network works on what was gathered.
      layerNorm(x, block.norm2[0], block.norm2[1], h);
      linear(block.think1[0], block.think1[1], h, 4 * W, wide);
      for (let i = 0; i < 4 * W; i++) wide[i] = gelu(wide[i]);
      linear(block.think2[0], block.think2[1], wide, W, added);
      for (let i = 0; i < W; i++) x[i] += added[i];
    }

    this.length += 1;
    layerNorm(x, this.finalNorm[0], this.finalNorm[1], h);
    return linear(this.toScores[0], this.toScores[1], h, this.vocab, this.scores);
  }

  // Read a list of pieces from a clean start; returns the scores after the last one.
  readAll(tokens) {
    this.reset();
    let scores = null;
    for (const token of tokens.slice(-this.context)) scores = this.read(token);
    return scores;
  }
}

function softmax(scores) {
  let biggest = -Infinity, total = 0;
  for (const s of scores) if (s > biggest) biggest = s;
  const odds = new Float32Array(scores.length);
  for (let i = 0; i < scores.length; i++) { odds[i] = Math.exp(scores[i] - biggest); total += odds[i]; }
  for (let i = 0; i < odds.length; i++) odds[i] /= total;
  return odds;
}

// Draw the next piece: keep the top 40 guesses, then roll weighted dice.
function choose(scores, temperature, topK = 40) {
  const order = Array.from(scores.keys()).sort((a, b) => scores[b] - scores[a]).slice(0, topK);
  const odds = softmax(order.map(i => scores[i] / temperature));
  let roll = Math.random();
  for (let i = 0; i < order.length; i++) { roll -= odds[i]; if (roll <= 0) return order[i]; }
  return order[0];
}

function topOdds(scores, tokenizer, howMany = 8) {
  const odds = softmax(scores);
  return Array.from(odds.keys()).sort((a, b) => odds[b] - odds[a]).slice(0, howMany).map(i => ({ piece: tokenizer.piece(i), odds: odds[i] }));
}

async function loadStoryModel(folder = ".", onProgress = () => {}) {
  const info = await (await fetch(`${folder}/model.json`)).json();
  const reply = await fetch(`${folder}/model.bin`);
  const size = info.parameters * 2, bytes = new Uint8Array(size), reader = reply.body.getReader();
  for (let got = 0; ; ) {
    const { done, value } = await reader.read();
    if (done) break;
    bytes.set(value, got);
    got += value.length;
    onProgress(got / size);
  }
  return { info, model: new StoryModel(info, unpackFloat16(bytes.buffer)), tokenizer: new Tokenizer(info.merges) };
}

// ---------------------------------------------------------------------------
// When this file is run as a background worker, it writes stories on request,
// so the page stays responsive while the model works.
// ---------------------------------------------------------------------------
if (typeof WorkerGlobalScope !== "undefined" && self instanceof WorkerGlobalScope) {
  let loaded = null, job = 0;
  const ready = loadStoryModel(".", share => postMessage({ loading: share })).then(result => {
    loaded = result;
    postMessage({ ready: { parameters: result.info.parameters, pieces: result.info.vocab_size, steps: result.info.steps } });
  }).catch(problem => postMessage({ failed: String(problem) }));

  onmessage = async ({ data }) => {
    const mine = ++job; // a newer request, or "stop", cancels the one in progress
    if (data.stop) return;
    await ready;
    if (!loaded) return;
    const { model, tokenizer } = loaded;
    const decoder = new TextDecoder();
    let tokens = tokenizer.encode(data.prompt);
    if (!tokens.length) tokens = [tokenizer.endToken];
    let scores = model.readAll(tokens);
    for (let written = 0; written < data.length; written++) {
      const next = choose(scores, data.temperature);
      if (next === tokenizer.endToken) break;
      tokens.push(next);
      postMessage({ id: data.id, piece: decoder.decode(tokenizer.bytes(next), { stream: true }) });
      // Its memory holds 128 pieces. When full, keep the most recent half and carry on.
      scores = model.length < model.context ? model.read(next) : model.readAll(tokens.slice(-model.context / 2));
      if (written % 4 === 3) await new Promise(resume => setTimeout(resume)); // let a "stop" message in
      if (mine !== job) return;
    }
    postMessage({ id: data.id, done: true, odds: topOdds(scores, tokenizer) });
  };
}

if (typeof module !== "undefined") module.exports = { Tokenizer, StoryModel, unpackFloat16, softmax, choose, topOdds };
