import { RNG } from "./rng.ts";
import { TransformerLM, type TransformerConfig } from "./model.ts";
import { AdamW, lrSchedule } from "./optim.ts";
import { crossEntropy, softmaxRows } from "./ops.ts";
import { TASKS, type Task, VOCAB_SIZE, decode, encode } from "./tasks.ts";

export interface TrainerOptions {
  task: Task["id"];
  seed: number;
  nLayer: number;
  nHead: number;
  nEmbd: number;
  batchSize: number;
  lr: number;
  warmup: number;
  totalSteps: number;
}

export const DEFAULT_OPTIONS: TrainerOptions = {
  task: "reverse",
  seed: 1337,
  nLayer: 2,
  nHead: 4,
  nEmbd: 48,
  batchSize: 32,
  lr: 3e-3,
  warmup: 50,
  totalSteps: 1500,
};

export interface StepStats {
  step: number;
  loss: number;
  lr: number;
  gradNorm: number;
  ms: number;
}

export interface Prediction {
  prompt: string;
  output: string;
  expected: string;
  correct: boolean;
  /** Full token string the attention maps refer to (prompt + output minus the last token). */
  tokens: string;
  /** [layer][head] -> T×T row-major attention matrix */
  attention: number[][][];
  /** For each generated position, the model's top-3 alternatives. */
  topk: Array<Array<{ token: string; p: number }>>;
}

export class Trainer {
  opts: TrainerOptions;
  task: Task;
  model: TransformerLM;
  optim: AdamW;
  rng: RNG;
  step = 0;

  constructor(opts: Partial<TrainerOptions> = {}) {
    this.opts = { ...DEFAULT_OPTIONS, ...opts };
    this.task = TASKS[this.opts.task];
    this.rng = new RNG(this.opts.seed);
    const config: TransformerConfig = {
      vocabSize: VOCAB_SIZE,
      blockSize: 16,
      nLayer: this.opts.nLayer,
      nHead: this.opts.nHead,
      nEmbd: this.opts.nEmbd,
    };
    this.model = new TransformerLM(config, this.rng);
    this.optim = new AdamW(this.model.params, { lr: this.opts.lr, weightDecay: 0.01, clipNorm: 1 });
  }

  get seqLen(): number {
    return this.task.promptLen + this.task.answerLen;
  }

  /** Builds (inputs, targets, loss-mask) for a batch of freshly sampled problems. */
  makeBatch(B: number, rng = this.rng) {
    const T = this.seqLen - 1;
    const x = new Int32Array(B * T), y = new Int32Array(B * T), w = new Float64Array(B * T);
    for (let b = 0; b < B; b++) {
      const { prompt, answer } = this.task.sample(rng);
      const ids = encode(prompt + answer);
      for (let t = 0; t < T; t++) {
        x[b * T + t] = ids[t];
        y[b * T + t] = ids[t + 1];
        // Only grade the model on answer tokens; the prompt is random noise it can't predict.
        w[b * T + t] = t + 1 >= this.task.promptLen ? 1 : 0;
      }
    }
    return { x, y, w, T };
  }

  trainStep(): StepStats {
    const t0 = performance.now();
    const { batchSize, lr, warmup, totalSteps } = this.opts;
    const { x, y, w, T } = this.makeBatch(batchSize);
    const { logits } = this.model.forward(x, batchSize, T);
    const loss = crossEntropy(logits, y, w);
    this.optim.zeroGrad();
    loss.backward();
    const curLr = lrSchedule(this.step, lr, warmup, totalSteps);
    const gradNorm = this.optim.step(curLr);
    this.step++;
    return { step: this.step, loss: loss.data[0], lr: curLr, gradNorm, ms: performance.now() - t0 };
  }

  /** Greedy decoding for a batch of prompts that share the same length. */
  generate(prompts: string[]): string[] {
    const B = prompts.length, P = this.task.promptLen, A = this.task.answerLen;
    const seqs = prompts.map((p) => Array.from(encode(p)));
    for (let k = 0; k < A; k++) {
      const T = P + k;
      const ids = new Int32Array(B * T);
      for (let b = 0; b < B; b++) ids.set(seqs[b], b * T);
      const { logits } = this.model.forward(ids, B, T);
      const V = logits.cols;
      for (let b = 0; b < B; b++) {
        const row = (b * T + T - 1) * V;
        let best = 0;
        for (let j = 1; j < V; j++) if (logits.data[row + j] > logits.data[row + best]) best = j;
        seqs[b].push(best);
      }
    }
    return seqs.map((s) => decode(s.slice(P)));
  }

  /** Exact-match accuracy on fresh problems drawn from a separate, fixed RNG stream. */
  evaluate(n = 128, seed = 99991): { accuracy: number; tokenAccuracy: number } {
    const rng = new RNG(seed);
    const probs = Array.from({ length: n }, () => this.task.sample(rng));
    const outs = this.generate(probs.map((p) => p.prompt));
    let exact = 0, tok = 0;
    outs.forEach((o, i) => {
      if (o === probs[i].answer) exact++;
      for (let j = 0; j < o.length; j++) if (o[j] === probs[i].answer[j]) tok++;
    });
    return { accuracy: exact / n, tokenAccuracy: tok / (n * this.task.answerLen) };
  }

  /** Runs one prompt and returns everything the UI needs to explain the model's decision. */
  predict(prompt: string): Prediction {
    const err = this.task.validate(prompt);
    if (err) throw new Error(err);
    const output = this.generate([prompt])[0];
    const full = prompt + output;
    const T = full.length - 1;
    const ids = encode(full.slice(0, T));
    const { logits, attention } = this.model.forward(ids, 1, T);
    const V = logits.cols;
    const probs = softmaxRows(logits.data, T, V);
    const { nHead } = this.model.config;
    const att = attention.map((layer) =>
      Array.from({ length: nHead }, (_, h) => Array.from(layer.subarray(h * T * T, (h + 1) * T * T))),
    );
    const topk: Prediction["topk"] = [];
    for (let t = this.task.promptLen - 1; t < T; t++) {
      const row = Array.from(probs.subarray(t * V, (t + 1) * V)).map((p, j) => ({ token: decode([j]), p }));
      row.sort((a, b) => b.p - a.p);
      topk.push(row.slice(0, 3));
    }
    const expected = this.task.solve(prompt);
    return { prompt, output, expected, correct: output === expected, tokens: full.slice(0, T), attention: att, topk };
  }
}
