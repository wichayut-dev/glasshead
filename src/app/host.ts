import { Trainer, type TrainerOptions, type Prediction, type StepStats } from "../engine/trainer.ts";
import { runAllGradChecks, type GradCheckResult } from "../engine/gradcheck.ts";

/** Messages the UI sends to the training host. */
export type ToHost =
  | { type: "init"; opts: Partial<TrainerOptions>; weights?: Record<string, number[]>; step?: number }
  | { type: "run" }
  | { type: "pause" }
  | { type: "probe"; prompt: string }
  | { type: "gradcheck" }
  | { type: "export" };

/** Messages the training host sends back to the UI. */
export type FromHost =
  | { type: "ready"; params: number; opts: TrainerOptions; layers: { name: string; shape: [number, number] }[] }
  | { type: "progress"; stats: StepStats[]; running: boolean }
  | { type: "eval"; step: number; accuracy: number; tokenAccuracy: number }
  | { type: "probe"; step: number; prediction: Prediction }
  | { type: "gradcheck"; results: GradCheckResult[]; ms: number }
  | { type: "export"; data: string }
  | { type: "error"; message: string };

/**
 * The training loop, written once and run either inside a Web Worker (normal case)
 * or on the main thread in small time slices (fallback when workers are unavailable).
 */
export class TrainingHost {
  trainer: Trainer | null = null;
  running = false;
  probePrompt = "";
  post: (m: FromHost) => void;
  schedule: (fn: () => void) => void;
  sliceMs: number;

  constructor(post: (m: FromHost) => void, schedule: (fn: () => void) => void, sliceMs = 120) {
    this.post = post;
    this.schedule = schedule;
    this.sliceMs = sliceMs;
  }

  handle(msg: ToHost): void {
    try {
      switch (msg.type) {
        case "init": {
          this.running = false;
          const t = new Trainer(msg.opts);
          if (msg.weights) t.model.importWeights(msg.weights);
          if (msg.step) t.step = msg.step;
          this.trainer = t;
          this.probePrompt = t.task.probe;
          this.post({
            type: "ready",
            params: t.model.numParams,
            opts: t.opts,
            layers: t.model.params.map((p) => ({ name: p.name, shape: [p.rows, p.cols] })),
          });
          this.sendProbe();
          this.sendEval();
          break;
        }
        case "run":
          if (!this.running) {
            this.running = true;
            this.loop();
          }
          break;
        case "pause":
          this.running = false;
          this.post({ type: "progress", stats: [], running: false });
          break;
        case "probe":
          this.probePrompt = msg.prompt;
          this.sendProbe();
          break;
        case "gradcheck": {
          const t0 = performance.now();
          const results = runAllGradChecks();
          this.post({ type: "gradcheck", results, ms: performance.now() - t0 });
          break;
        }
        case "export": {
          if (!this.trainer) return;
          this.post({
            type: "export",
            data: JSON.stringify({ opts: this.trainer.opts, step: this.trainer.step, weights: this.trainer.model.exportWeights() }),
          });
          break;
        }
      }
    } catch (e) {
      this.post({ type: "error", message: e instanceof Error ? e.message : String(e) });
    }
  }

  private sendProbe(): void {
    const t = this.trainer;
    if (!t) return;
    try {
      this.post({ type: "probe", step: t.step, prediction: t.predict(this.probePrompt) });
    } catch (e) {
      this.post({ type: "error", message: e instanceof Error ? e.message : String(e) });
    }
  }

  private sendEval(): void {
    const t = this.trainer;
    if (!t) return;
    const r = t.evaluate(64);
    this.post({ type: "eval", step: t.step, ...r });
  }

  private loop = (): void => {
    const t = this.trainer;
    if (!this.running || !t) return;
    const stats: StepStats[] = [];
    const start = performance.now();
    let probeDue = false, evalDue = false;
    do {
      const s = t.trainStep();
      stats.push(s);
      if (s.step % 20 === 0) probeDue = true;
      if (s.step % 100 === 0) evalDue = true;
      if (s.step >= t.opts.totalSteps) this.running = false;
    } while (this.running && performance.now() - start < this.sliceMs);
    this.post({ type: "progress", stats, running: this.running });
    if (probeDue || !this.running) this.sendProbe();
    if (evalDue || !this.running) this.sendEval();
    if (this.running) this.schedule(this.loop);
  };
}
