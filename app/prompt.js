// Prompt tokenization matching ComfyUI's Anima tokenizer (comfy/text_encoders/anima.py):
//  - "(text:1.2)" / "(text)" weighting, "\(" "\)" escapes (comfy/sd1_clip.py token_weights)
//  - Qwen3 ids: every segment tokenized without special tokens, weights dropped;
//    an empty prompt becomes the single pad token 151643
//  - T5 ids: every segment tokenized without its end token, one </s> appended at the end;
//    the per-token weights scale the LLM adapter output rows

import { Tokenizer } from "./vendor/tokenizers.min.mjs";

const QWEN_PAD = 151643;

function parseParentheses(s) {
  const out = [];
  let cur = "";
  let depth = 0;
  for (const ch of s) {
    if (ch === "(") {
      if (depth === 0) {
        if (cur) out.push(cur);
        cur = "(";
      } else cur += ch;
      depth++;
    } else if (ch === ")") {
      depth--;
      if (depth === 0) {
        out.push(cur + ")");
        cur = "";
      } else cur += ch;
    } else cur += ch;
  }
  if (cur) out.push(cur);
  return out;
}

function tokenWeights(s, weight) {
  const out = [];
  for (let x of parseParentheses(s)) {
    let w = weight;
    if (x.length >= 2 && x[0] === "(" && x[x.length - 1] === ")") {
      x = x.slice(1, -1);
      const i = x.lastIndexOf(":");
      w *= 1.1;
      if (i > 0) {
        const num = x.slice(i + 1).trim();
        if (num !== "" && !Number.isNaN(Number(num))) {
          w = Number(num);
          x = x.slice(0, i);
        }
      }
      out.push(...tokenWeights(x, w));
    } else out.push([x, weight]);
  }
  return out;
}

function parseWeights(text) {
  const esc = text.replaceAll("\\)", "\0\x01").replaceAll("\\(", "\0\x02");
  return tokenWeights(esc, 1.0).map(([s, w]) => [s.replaceAll("\0\x01", ")").replaceAll("\0\x02", "("), w]);
}

export class AnimaTokenizer {
  static async load(fetchJson) {
    const [qj, qc, tj, tc] = await Promise.all([
      fetchJson("tokenizers/qwen/tokenizer.json"),
      fetchJson("tokenizers/qwen/tokenizer_config.json"),
      fetchJson("tokenizers/t5/tokenizer.json"),
      fetchJson("tokenizers/t5/tokenizer_config.json"),
    ]);
    return new AnimaTokenizer(new Tokenizer(qj, qc), new Tokenizer(tj, tc));
  }

  constructor(qwen, t5) {
    this.qwen = qwen;
    this.t5 = t5;
    this.t5End = t5.encode("", { add_special_tokens: true }).ids.at(-1);
  }

  encode(text) {
    const segs = parseWeights(text).filter(([s]) => s !== "");
    const qwenIds = [];
    const t5Ids = [];
    const t5Weights = [];
    for (const [s, w] of segs) {
      qwenIds.push(...this.qwen.encode(s, { add_special_tokens: false }).ids);
      for (const id of this.t5.encode(s, { add_special_tokens: false }).ids) {
        t5Ids.push(id);
        t5Weights.push(w);
      }
    }
    if (!qwenIds.length) qwenIds.push(QWEN_PAD);
    t5Ids.push(this.t5End);
    t5Weights.push(1.0);
    return { qwenIds, t5Ids, t5Weights };
  }
}
