import { RNG } from "./rng.ts";
import { Tensor } from "./tensor.ts";
import {
  add, addBias, causalSelfAttention, crossEntropy, embedding, gelu, layerNorm, matmul,
} from "./ops.ts";
import { TransformerLM } from "./model.ts";

/**
 * Finite-difference gradient checking.
 *
 * For an op f, we build the scalar L = Σ f(inputs) ⊙ R with a fixed random R (so every
 * output element matters with a different weight), backprop to get the analytic gradient,
 * then nudge individual input elements by ±ε and compare against
 *   (L(x+ε) − L(x−ε)) / 2ε.
 * A max relative error around 1e-7 means the hand-written backward is right.
 */

export interface GradCheckResult {
  name: string;
  maxRelError: number;
  checked: number;
  passed: boolean;
}

function weightedSum(out: Tensor, R: Float64Array): Tensor {
  // L = Σ out[i] * R[i] — implemented as an op so it participates in autograd.
  let s = 0;
  for (let i = 0; i < out.size; i++) s += out.data[i] * R[i];
  const L = new Tensor(new Float64Array([s]), 1, 1, [out]);
  L.backwardFn = () => {
    if (!out.grad) return;
    for (let i = 0; i < out.size; i++) out.grad[i] += L.grad![0] * R[i];
  };
  return L;
}

export function gradcheck(
  name: string,
  inputs: Tensor[],
  f: () => Tensor,
  { eps = 1e-5, tol = 1e-5, samplesPerInput = 24, seed = 7 } = {},
): GradCheckResult {
  const rng = new RNG(seed);
  for (const t of inputs) t.asParam();
  const probe = f();
  const R = new Float64Array(probe.size);
  for (let i = 0; i < R.length; i++) R[i] = rng.normal();
  const lossAt = () => {
    const out = f();
    if (out.size === 1) return out.data[0] * R[0];
    let s = 0;
    for (let i = 0; i < out.size; i++) s += out.data[i] * R[i];
    return s;
  };

  for (const t of inputs) t.grad!.fill(0);
  const out = f();
  const L = out.size === 1 ? weightedSum(out, R) : weightedSum(out, R);
  L.backward();

  let maxRel = 0, checked = 0;
  for (const t of inputs) {
    const n = Math.min(samplesPerInput, t.size);
    for (let s = 0; s < n; s++) {
      const i = t.size <= samplesPerInput ? s : rng.int(t.size);
      const orig = t.data[i];
      t.data[i] = orig + eps;
      const lp = lossAt();
      t.data[i] = orig - eps;
      const lm = lossAt();
      t.data[i] = orig;
      const num = (lp - lm) / (2 * eps);
      const ana = t.grad![i];
      const rel = Math.abs(num - ana) / Math.max(1e-6, Math.abs(num) + Math.abs(ana));
      if (rel > maxRel) maxRel = rel;
      checked++;
    }
  }
  return { name, maxRelError: maxRel, checked, passed: maxRel < tol };
}

/** Runs the full suite: every primitive op, then a whole model forward pass end-to-end. */
export function runAllGradChecks(): GradCheckResult[] {
  const rng = new RNG(42);
  const r = (rows: number, cols: number, std = 1) => Tensor.randn(rows, cols, std, rng);
  const results: GradCheckResult[] = [];

  { const a = r(5, 7), w = r(7, 4); results.push(gradcheck("matmul", [a, w], () => matmul(a, w))); }
  { const a = r(4, 6), b = r(4, 6); results.push(gradcheck("add (residual)", [a, b], () => add(a, b))); }
  { const x = r(5, 6), b = r(1, 6); results.push(gradcheck("bias broadcast", [x, b], () => addBias(x, b))); }
  {
    const tbl = r(9, 5), ids = new Int32Array([3, 1, 3, 8, 0, 3]);
    results.push(gradcheck("embedding (scatter-add)", [tbl], () => embedding(tbl, ids)));
  }
  {
    const x = r(4, 8, 2), g = r(1, 8), b = r(1, 8);
    results.push(gradcheck("layernorm", [x, g, b], () => layerNorm(x, g, b)));
  }
  { const x = r(5, 6, 2); results.push(gradcheck("gelu", [x], () => gelu(x))); }
  {
    const B = 2, T = 5, C = 8, H = 2;
    const qkv = r(B * T, 3 * C);
    results.push(gradcheck("causal self-attention (fused)", [qkv], () => causalSelfAttention(qkv, B, T, H).out));
  }
  {
    const logits = r(6, 5, 2);
    const tgt = new Int32Array([0, 4, 2, 2, 1, 3]);
    const w = new Float64Array([0, 1, 1, 0, 1, 1]);
    results.push(gradcheck("cross-entropy (masked)", [logits], () => crossEntropy(logits, tgt, w)));
  }
  {
    // End-to-end: every parameter of a small model, gradient of the actual training loss.
    const model = new TransformerLM({ vocabSize: 7, blockSize: 6, nLayer: 2, nHead: 2, nEmbd: 8 }, new RNG(3));
    // Larger init so the check exercises non-trivial gradients everywhere.
    for (const p of model.params) if (p.name.includes(".g") === false) for (let i = 0; i < p.size; i++) p.data[i] += rng.normal() * 0.3;
    const B = 2, T = 5;
    const ids = new Int32Array(B * T).map(() => rng.int(7));
    const tgt = new Int32Array(B * T).map(() => rng.int(7));
    results.push(
      gradcheck("full model forward → loss", model.params, () => crossEntropy(model.forward(ids, B, T).logits, tgt), {
        samplesPerInput: 6,
      }),
    );
  }
  return results;
}
