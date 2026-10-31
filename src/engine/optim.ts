import { Tensor } from "./tensor.ts";

export interface AdamWOptions {
  lr: number;
  beta1?: number;
  beta2?: number;
  eps?: number;
  weightDecay?: number;
  /** Clip the global gradient L2 norm to this value (0 disables). */
  clipNorm?: number;
}

/**
 * AdamW (Loshchilov & Hutter): Adam with weight decay decoupled from the gradient,
 * plus global-norm gradient clipping. Decay only applies to params flagged `decay`
 * (weight matrices & embeddings, not biases or LayerNorm gains).
 */
export class AdamW {
  params: Tensor[];
  m: Float64Array[];
  v: Float64Array[];
  t = 0;
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
  weightDecay: number;
  clipNorm: number;

  constructor(params: Tensor[], opts: AdamWOptions) {
    this.params = params;
    this.m = params.map((p) => new Float64Array(p.size));
    this.v = params.map((p) => new Float64Array(p.size));
    this.lr = opts.lr;
    this.beta1 = opts.beta1 ?? 0.9;
    this.beta2 = opts.beta2 ?? 0.99;
    this.eps = opts.eps ?? 1e-8;
    this.weightDecay = opts.weightDecay ?? 0.01;
    this.clipNorm = opts.clipNorm ?? 1.0;
  }

  zeroGrad(): void {
    for (const p of this.params) p.grad!.fill(0);
  }

  /** Global L2 norm of all gradients. */
  gradNorm(): number {
    let s = 0;
    for (const p of this.params) for (const g of p.grad!) s += g * g;
    return Math.sqrt(s);
  }

  /** Applies one update and returns the pre-clip gradient norm. */
  step(lr = this.lr): number {
    this.t++;
    const norm = this.gradNorm();
    const clip = this.clipNorm > 0 && norm > this.clipNorm ? this.clipNorm / norm : 1;
    const bc1 = 1 - Math.pow(this.beta1, this.t);
    const bc2 = 1 - Math.pow(this.beta2, this.t);
    for (let k = 0; k < this.params.length; k++) {
      const p = this.params[k];
      const decay = (p as Tensor & { decay?: boolean }).decay ? this.weightDecay : 0;
      const W = p.data, G = p.grad!, M = this.m[k], V = this.v[k];
      for (let i = 0; i < W.length; i++) {
        const g = G[i] * clip;
        M[i] = this.beta1 * M[i] + (1 - this.beta1) * g;
        V[i] = this.beta2 * V[i] + (1 - this.beta2) * g * g;
        const mHat = M[i] / bc1, vHat = V[i] / bc2;
        W[i] -= lr * (mHat / (Math.sqrt(vHat) + this.eps) + decay * W[i]);
      }
    }
    return norm;
  }
}

/** Linear warmup then cosine decay to `minFrac` of the peak LR. */
export function lrSchedule(step: number, peak: number, warmup: number, total: number, minFrac = 0.1): number {
  if (step < warmup) return (peak * (step + 1)) / warmup;
  const progress = Math.min(1, (step - warmup) / Math.max(1, total - warmup));
  return peak * (minFrac + (1 - minFrac) * 0.5 * (1 + Math.cos(Math.PI * progress)));
}
