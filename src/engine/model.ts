import { RNG } from "./rng.ts";
import { Tensor } from "./tensor.ts";
import { add, causalSelfAttention, embedding, gelu, layerNorm, linear } from "./ops.ts";

export interface TransformerConfig {
  vocabSize: number;
  blockSize: number;
  nLayer: number;
  nHead: number;
  nEmbd: number;
}

interface Block {
  ln1g: Tensor; ln1b: Tensor;
  wqkv: Tensor; bqkv: Tensor;
  wo: Tensor; bo: Tensor;
  ln2g: Tensor; ln2b: Tensor;
  wfc: Tensor; bfc: Tensor;
  wproj: Tensor; bproj: Tensor;
}

export interface ForwardResult {
  /** [B*T, vocab] */
  logits: Tensor;
  /** One entry per layer, each laid out [B][head][query][key]. */
  attention: Float64Array[];
}

/**
 * A decoder-only transformer with pre-LayerNorm blocks (pre-LayerNorm, GELU MLP,
 * learned positional embeddings), built entirely on the hand-written autograd ops.
 */
export class TransformerLM {
  config: TransformerConfig;
  wte: Tensor;
  wpe: Tensor;
  blocks: Block[] = [];
  lnfg: Tensor;
  lnfb: Tensor;
  head: Tensor;
  headb: Tensor;
  params: Tensor[] = [];

  constructor(config: TransformerConfig, rng: RNG) {
    this.config = config;
    const { vocabSize: V, blockSize, nLayer, nEmbd: C } = config;
    const std = 0.02;
    // Standard trick: scale residual-path projections so the residual stream's variance
    // doesn't grow with depth.
    const resStd = std / Math.sqrt(2 * nLayer);
    const p = (t: Tensor, name: string, decay: boolean) => {
      t.name = name;
      t.asParam();
      (t as Tensor & { decay?: boolean }).decay = decay;
      this.params.push(t);
      return t;
    };
    this.wte = p(Tensor.randn(V, C, std, rng), "wte", true);
    this.wpe = p(Tensor.randn(blockSize, C, std, rng), "wpe", true);
    for (let l = 0; l < nLayer; l++) {
      this.blocks.push({
        ln1g: p(Tensor.fill(1, C, 1), `h${l}.ln1.g`, false),
        ln1b: p(Tensor.zeros(1, C), `h${l}.ln1.b`, false),
        wqkv: p(Tensor.randn(C, 3 * C, std, rng), `h${l}.attn.wqkv`, true),
        bqkv: p(Tensor.zeros(1, 3 * C), `h${l}.attn.bqkv`, false),
        wo: p(Tensor.randn(C, C, resStd, rng), `h${l}.attn.wo`, true),
        bo: p(Tensor.zeros(1, C), `h${l}.attn.bo`, false),
        ln2g: p(Tensor.fill(1, C, 1), `h${l}.ln2.g`, false),
        ln2b: p(Tensor.zeros(1, C), `h${l}.ln2.b`, false),
        wfc: p(Tensor.randn(C, 4 * C, std, rng), `h${l}.mlp.wfc`, true),
        bfc: p(Tensor.zeros(1, 4 * C), `h${l}.mlp.bfc`, false),
        wproj: p(Tensor.randn(4 * C, C, resStd, rng), `h${l}.mlp.wproj`, true),
        bproj: p(Tensor.zeros(1, C), `h${l}.mlp.bproj`, false),
      });
    }
    this.lnfg = p(Tensor.fill(1, C, 1), "lnf.g", false);
    this.lnfb = p(Tensor.zeros(1, C), "lnf.b", false);
    this.head = p(Tensor.randn(C, V, std, rng), "lm_head", true);
    this.headb = p(Tensor.zeros(1, V), "lm_head.b", false);
  }

  get numParams(): number {
    return this.params.reduce((s, t) => s + t.size, 0);
  }

  /** ids holds B sequences of length T, flattened row-major. */
  forward(ids: Int32Array, B: number, T: number): ForwardResult {
    if (T > this.config.blockSize) throw new Error(`sequence length ${T} > blockSize`);
    if (ids.length !== B * T) throw new Error("ids length must be B*T");
    const pos = new Int32Array(B * T);
    for (let b = 0; b < B; b++) for (let t = 0; t < T; t++) pos[b * T + t] = t;

    let x = add(embedding(this.wte, ids), embedding(this.wpe, pos));
    const attention: Float64Array[] = [];
    for (const blk of this.blocks) {
      const h = layerNorm(x, blk.ln1g, blk.ln1b);
      const att = causalSelfAttention(linear(h, blk.wqkv, blk.bqkv), B, T, this.config.nHead);
      attention.push(att.probs);
      x = add(x, linear(att.out, blk.wo, blk.bo));
      const h2 = layerNorm(x, blk.ln2g, blk.ln2b);
      x = add(x, linear(gelu(linear(h2, blk.wfc, blk.bfc)), blk.wproj, blk.bproj));
    }
    const logits = linear(layerNorm(x, this.lnfg, this.lnfb), this.head, this.headb);
    return { logits, attention };
  }

  /** Serialises all weights (for save / load / sharing a trained model). */
  exportWeights(): Record<string, number[]> {
    const out: Record<string, number[]> = {};
    for (const p of this.params) out[p.name] = Array.from(p.data, (v) => Math.round(v * 1e6) / 1e6);
    return out;
  }

  importWeights(w: Record<string, number[]>): void {
    for (const p of this.params) {
      const src = w[p.name];
      if (!src || src.length !== p.size) throw new Error(`weight ${p.name} missing or wrong size`);
      p.data.set(src);
    }
  }
}
