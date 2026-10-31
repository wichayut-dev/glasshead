import { Tensor } from "./tensor.ts";

/**
 * Differentiable operations. Each op computes its forward result and attaches a
 * closure that, given out.grad, accumulates gradients into its inputs.
 * Every one of these is verified against finite differences in tests/gradcheck.test.ts.
 */

function make(data: Float64Array, rows: number, cols: number, parents: Tensor[]): Tensor {
  return new Tensor(data, rows, cols, parents);
}

// ---------------------------------------------------------------------------
// Linear algebra
// ---------------------------------------------------------------------------

/** out[n,m] = a[n,k] @ w[k,m] */
export function matmul(a: Tensor, w: Tensor): Tensor {
  const n = a.rows, k = a.cols, m = w.cols;
  if (w.rows !== k) throw new Error(`matmul shape mismatch ${n}x${k} @ ${w.rows}x${m}`);
  const A = a.data, W = w.data;
  const O = new Float64Array(n * m);
  // i-p-j loop order keeps the innermost loop walking contiguous memory.
  for (let i = 0; i < n; i++) {
    const oi = i * m;
    for (let p = 0; p < k; p++) {
      const av = A[i * k + p];
      if (av === 0) continue;
      const wp = p * m;
      for (let j = 0; j < m; j++) O[oi + j] += av * W[wp + j];
    }
  }
  const out = make(O, n, m, [a, w]);
  out.backwardFn = () => {
    const G = out.grad!;
    if (a.grad) {
      const dA = a.grad;
      for (let i = 0; i < n; i++) {
        const gi = i * m;
        for (let p = 0; p < k; p++) {
          const wp = p * m;
          let s = 0;
          for (let j = 0; j < m; j++) s += G[gi + j] * W[wp + j];
          dA[i * k + p] += s;
        }
      }
    }
    if (w.grad) {
      const dW = w.grad;
      for (let i = 0; i < n; i++) {
        const gi = i * m;
        for (let p = 0; p < k; p++) {
          const av = A[i * k + p];
          if (av === 0) continue;
          const wp = p * m;
          for (let j = 0; j < m; j++) dW[wp + j] += av * G[gi + j];
        }
      }
    }
  };
  return out;
}

/** Elementwise a + b (same shape). Used for residual connections. */
export function add(a: Tensor, b: Tensor): Tensor {
  if (a.rows !== b.rows || a.cols !== b.cols) throw new Error("add shape mismatch");
  const O = new Float64Array(a.size);
  for (let i = 0; i < O.length; i++) O[i] = a.data[i] + b.data[i];
  const out = make(O, a.rows, a.cols, [a, b]);
  out.backwardFn = () => {
    const G = out.grad!;
    if (a.grad) for (let i = 0; i < G.length; i++) a.grad[i] += G[i];
    if (b.grad) for (let i = 0; i < G.length; i++) b.grad[i] += G[i];
  };
  return out;
}

/** x[n,m] + bias[1,m] broadcast over rows. */
export function addBias(x: Tensor, bias: Tensor): Tensor {
  const n = x.rows, m = x.cols;
  if (bias.size !== m) throw new Error("addBias shape mismatch");
  const O = new Float64Array(x.size);
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) O[i * m + j] = x.data[i * m + j] + bias.data[j];
  const out = make(O, n, m, [x, bias]);
  out.backwardFn = () => {
    const G = out.grad!;
    if (x.grad) for (let i = 0; i < G.length; i++) x.grad[i] += G[i];
    if (bias.grad) for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) bias.grad[j] += G[i * m + j];
  };
  return out;
}

/** Fully connected layer: x @ W + b */
export function linear(x: Tensor, w: Tensor, b: Tensor): Tensor {
  return addBias(matmul(x, w), b);
}

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

/** Gathers rows of `table` by index: out[i] = table[ids[i]]. Backward is a scatter-add. */
export function embedding(table: Tensor, ids: Int32Array): Tensor {
  const C = table.cols, n = ids.length;
  const O = new Float64Array(n * C);
  for (let i = 0; i < n; i++) {
    const id = ids[i];
    if (id < 0 || id >= table.rows) throw new Error(`embedding index ${id} out of range`);
    O.set(table.data.subarray(id * C, id * C + C), i * C);
  }
  const out = make(O, n, C, [table]);
  out.backwardFn = () => {
    if (!table.grad) return;
    const G = out.grad!;
    for (let i = 0; i < n; i++) {
      const base = ids[i] * C;
      for (let j = 0; j < C; j++) table.grad[base + j] += G[i * C + j];
    }
  };
  return out;
}

// ---------------------------------------------------------------------------
// Normalisation & activations
// ---------------------------------------------------------------------------

/** Row-wise LayerNorm with learnable gain and bias. */
export function layerNorm(x: Tensor, gain: Tensor, bias: Tensor, eps = 1e-5): Tensor {
  const n = x.rows, m = x.cols;
  const X = x.data;
  const O = new Float64Array(x.size);
  const xhat = new Float64Array(x.size);
  const rstd = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const o = i * m;
    let mean = 0;
    for (let j = 0; j < m; j++) mean += X[o + j];
    mean /= m;
    let v = 0;
    for (let j = 0; j < m; j++) {
      const d = X[o + j] - mean;
      v += d * d;
    }
    v /= m;
    const r = 1 / Math.sqrt(v + eps);
    rstd[i] = r;
    for (let j = 0; j < m; j++) {
      const h = (X[o + j] - mean) * r;
      xhat[o + j] = h;
      O[o + j] = h * gain.data[j] + bias.data[j];
    }
  }
  const out = make(O, n, m, [x, gain, bias]);
  out.backwardFn = () => {
    const G = out.grad!;
    for (let i = 0; i < n; i++) {
      const o = i * m;
      // dxhat = G * gain; dx = rstd * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat))
      let sumD = 0, sumDX = 0;
      for (let j = 0; j < m; j++) {
        const d = G[o + j] * gain.data[j];
        sumD += d;
        sumDX += d * xhat[o + j];
        if (gain.grad) gain.grad[j] += G[o + j] * xhat[o + j];
        if (bias.grad) bias.grad[j] += G[o + j];
      }
      if (x.grad) {
        const mD = sumD / m, mDX = sumDX / m, r = rstd[i];
        for (let j = 0; j < m; j++) {
          const d = G[o + j] * gain.data[j];
          x.grad[o + j] += r * (d - mD - xhat[o + j] * mDX);
        }
      }
    }
  };
  return out;
}

const GELU_K = Math.sqrt(2 / Math.PI);

/** GELU (tanh approximation). */
export function gelu(x: Tensor): Tensor {
  const X = x.data;
  const O = new Float64Array(x.size);
  const T = new Float64Array(x.size);
  for (let i = 0; i < X.length; i++) {
    const v = X[i];
    const t = Math.tanh(GELU_K * (v + 0.044715 * v * v * v));
    T[i] = t;
    O[i] = 0.5 * v * (1 + t);
  }
  const out = make(O, x.rows, x.cols, [x]);
  out.backwardFn = () => {
    if (!x.grad) return;
    const G = out.grad!;
    for (let i = 0; i < X.length; i++) {
      const v = X[i], t = T[i];
      const d = 0.5 * (1 + t) + 0.5 * v * (1 - t * t) * GELU_K * (1 + 3 * 0.044715 * v * v);
      x.grad[i] += G[i] * d;
    }
  };
  return out;
}

// ---------------------------------------------------------------------------
// Attention (fused, hand-derived backward)
// ---------------------------------------------------------------------------

export interface AttentionResult {
  out: Tensor;
  /** Attention probabilities laid out as [batch][head][query][key], zeros above the diagonal. */
  probs: Float64Array;
}

/**
 * Multi-head causal self-attention as one fused op.
 *
 * qkv: [B*T, 3C] — the output of a single linear layer, holding Q | K | V side by side.
 * Returns [B*T, C] with the heads concatenated back together.
 *
 * Backward (per batch b, head h), with S = scale * Q K^T, P = softmax(S), O = P V:
 *   dV  = P^T dO
 *   dP  = dO V^T
 *   dS  = P ⊙ (dP − rowsum(P ⊙ dP))
 *   dQ  = scale * dS K,   dK = scale * dS^T Q
 */
export function causalSelfAttention(qkv: Tensor, B: number, T: number, nHead: number): AttentionResult {
  const C3 = qkv.cols, C = C3 / 3;
  if (C % nHead !== 0) throw new Error("nEmbd must be divisible by nHead");
  if (qkv.rows !== B * T) throw new Error("attention: qkv rows must equal B*T");
  const hs = C / nHead;
  const scale = 1 / Math.sqrt(hs);
  const X = qkv.data;
  const O = new Float64Array(B * T * C);
  const P = new Float64Array(B * nHead * T * T);

  for (let b = 0; b < B; b++) {
    for (let h = 0; h < nHead; h++) {
      const pBase = (b * nHead + h) * T * T;
      const qOff = h * hs, kOff = C + h * hs, vOff = 2 * C + h * hs;
      for (let i = 0; i < T; i++) {
        const qi = (b * T + i) * C3 + qOff;
        const pRow = pBase + i * T;
        let max = -Infinity;
        for (let j = 0; j <= i; j++) {
          const kj = (b * T + j) * C3 + kOff;
          let s = 0;
          for (let d = 0; d < hs; d++) s += X[qi + d] * X[kj + d];
          s *= scale;
          P[pRow + j] = s;
          if (s > max) max = s;
        }
        let sum = 0;
        for (let j = 0; j <= i; j++) {
          const e = Math.exp(P[pRow + j] - max);
          P[pRow + j] = e;
          sum += e;
        }
        const oi = (b * T + i) * C + h * hs;
        for (let j = 0; j <= i; j++) {
          const p = (P[pRow + j] /= sum);
          const vj = (b * T + j) * C3 + vOff;
          for (let d = 0; d < hs; d++) O[oi + d] += p * X[vj + d];
        }
      }
    }
  }

  const out = make(O, B * T, C, [qkv]);
  out.backwardFn = () => {
    if (!qkv.grad) return;
    const G = out.grad!, dX = qkv.grad;
    const dP = new Float64Array(T);
    for (let b = 0; b < B; b++) {
      for (let h = 0; h < nHead; h++) {
        const pBase = (b * nHead + h) * T * T;
        const qOff = h * hs, kOff = C + h * hs, vOff = 2 * C + h * hs;
        for (let i = 0; i < T; i++) {
          const gi = (b * T + i) * C + h * hs;
          const qi = (b * T + i) * C3 + qOff;
          const pRow = pBase + i * T;
          let dot = 0;
          for (let j = 0; j <= i; j++) {
            const vj = (b * T + j) * C3 + vOff;
            const p = P[pRow + j];
            let s = 0;
            for (let d = 0; d < hs; d++) {
              s += G[gi + d] * X[vj + d];
              dX[vj + d] += p * G[gi + d]; // dV
            }
            dP[j] = s;
            dot += p * s;
          }
          for (let j = 0; j <= i; j++) {
            const dS = P[pRow + j] * (dP[j] - dot) * scale;
            if (dS === 0) continue;
            const kj = (b * T + j) * C3 + kOff;
            for (let d = 0; d < hs; d++) {
              dX[qi + d] += dS * X[kj + d]; // dQ
              dX[kj + d] += dS * X[qi + d]; // dK
            }
          }
        }
      }
    }
  };
  return { out, probs: P };
}

// ---------------------------------------------------------------------------
// Loss
// ---------------------------------------------------------------------------

/**
 * Weighted mean cross-entropy over rows of logits[n, V].
 * weights[i] = 0 masks a row out (e.g. prompt positions the model shouldn't be graded on).
 */
export function crossEntropy(logits: Tensor, targets: Int32Array, weights?: Float64Array): Tensor {
  const n = logits.rows, V = logits.cols;
  const L = logits.data;
  const probs = new Float64Array(n * V);
  let total = 0, wsum = 0;
  for (let i = 0; i < n; i++) {
    const w = weights ? weights[i] : 1;
    const o = i * V;
    let max = -Infinity;
    for (let j = 0; j < V; j++) if (L[o + j] > max) max = L[o + j];
    let sum = 0;
    for (let j = 0; j < V; j++) {
      const e = Math.exp(L[o + j] - max);
      probs[o + j] = e;
      sum += e;
    }
    for (let j = 0; j < V; j++) probs[o + j] /= sum;
    if (w !== 0) {
      total += -w * Math.log(Math.max(probs[o + targets[i]], 1e-300));
      wsum += w;
    }
  }
  const denom = wsum > 0 ? wsum : 1;
  const out = make(new Float64Array([total / denom]), 1, 1, [logits]);
  out.backwardFn = () => {
    if (!logits.grad) return;
    const g = out.grad![0] / denom;
    for (let i = 0; i < n; i++) {
      const w = weights ? weights[i] : 1;
      if (w === 0) continue;
      const o = i * V;
      for (let j = 0; j < V; j++) logits.grad[o + j] += g * w * probs[o + j];
      logits.grad[o + targets[i]] -= g * w;
    }
  };
  return out;
}

/** Row-wise softmax with no autograd — used for inference/visualisation only. */
export function softmaxRows(data: Float64Array, rows: number, cols: number, temperature = 1): Float64Array {
  const out = new Float64Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    const o = i * cols;
    let max = -Infinity;
    for (let j = 0; j < cols; j++) max = Math.max(max, data[o + j] / temperature);
    let sum = 0;
    for (let j = 0; j < cols; j++) sum += out[o + j] = Math.exp(data[o + j] / temperature - max);
    for (let j = 0; j < cols; j++) out[o + j] /= sum;
  }
  return out;
}
