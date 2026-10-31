import { RNG } from "./rng.ts";

/**
 * A 2-D tensor that records the operation that produced it, so gradients can be
 * pushed back through the graph with reverse-mode automatic differentiation.
 *
 * Everything is stored row-major in a Float64Array. Double precision costs a bit of
 * speed but makes finite-difference gradient checks accurate to ~1e-8, which is how
 * the tests prove every backward pass is correct.
 */
export class Tensor {
  data: Float64Array;
  grad: Float64Array | null = null;
  rows: number;
  cols: number;
  parents: Tensor[];
  backwardFn: (() => void) | null = null;
  requiresGrad: boolean;
  /** Human-readable name, used for parameters and debugging. */
  name: string;

  constructor(data: Float64Array, rows: number, cols: number, parents: Tensor[] = [], name = "") {
    if (data.length !== rows * cols) {
      throw new Error(`Tensor ${name}: data length ${data.length} != ${rows}x${cols}`);
    }
    this.data = data;
    this.rows = rows;
    this.cols = cols;
    this.parents = parents;
    this.requiresGrad = parents.some((p) => p.requiresGrad);
    this.name = name;
  }

  get size(): number {
    return this.data.length;
  }

  static zeros(rows: number, cols: number, name = ""): Tensor {
    return new Tensor(new Float64Array(rows * cols), rows, cols, [], name);
  }

  static fill(rows: number, cols: number, value: number, name = ""): Tensor {
    return new Tensor(new Float64Array(rows * cols).fill(value), rows, cols, [], name);
  }

  static randn(rows: number, cols: number, std: number, rng: RNG, name = ""): Tensor {
    const d = new Float64Array(rows * cols);
    for (let i = 0; i < d.length; i++) d[i] = rng.normal() * std;
    return new Tensor(d, rows, cols, [], name);
  }

  /** Mark as a trainable leaf. Parameters keep a persistent grad buffer. */
  asParam(): Tensor {
    this.requiresGrad = true;
    this.grad = new Float64Array(this.size);
    return this;
  }

  /**
   * Reverse-mode autodiff. Topologically sorts the graph (iteratively, so deep
   * graphs never blow the call stack), seeds d(this)/d(this) = 1, and calls each
   * node's backward closure in reverse order. Parameter grads ACCUMULATE, so the
   * optimizer is responsible for zeroing them.
   */
  backward(): void {
    if (this.size !== 1) throw new Error("backward() must be called on a scalar");
    const order: Tensor[] = [];
    const visited = new Set<Tensor>();
    const stack: Array<[Tensor, number]> = [[this, 0]];
    while (stack.length) {
      const top = stack[stack.length - 1];
      const [node, idx] = top;
      if (idx === 0) {
        if (visited.has(node)) {
          stack.pop();
          continue;
        }
        visited.add(node);
      }
      if (idx < node.parents.length) {
        top[1]++;
        const p = node.parents[idx];
        if (p.requiresGrad && !visited.has(p)) stack.push([p, 0]);
      } else {
        stack.pop();
        order.push(node);
      }
    }
    for (const n of order) if (!n.grad) n.grad = new Float64Array(n.size);
    this.grad![0] += 1;
    for (let i = order.length - 1; i >= 0; i--) {
      const fn = order[i].backwardFn;
      if (fn) fn();
    }
  }
}
