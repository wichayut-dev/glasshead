/**
 * Headless training from the command line:
 *   node scripts/train.ts <task> [steps] [seed] [--save] [--lr=] [--nEmbd=] [--nLayer=] [--nHead=] [--batchSize=]
 *   node scripts/train.ts sort 1500 --save
 * --save writes pretrained/<task>.json (Float32 weights as base64 + the recorded
 * loss / accuracy curves), which the build embeds in the web demo.
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { Trainer } from "../src/engine/trainer.ts";

const args = process.argv.slice(2).filter((a) => !a.startsWith("--"));
const task = (args[0] ?? "reverse") as "reverse" | "sort" | "add";
const steps = Number(args[1] ?? 1500);
const seed = Number(args[2] ?? 1337);
const save = process.argv.includes("--save");
const flag = (name: string) => {
  const a = process.argv.find((x) => x.startsWith(`--${name}=`));
  return a ? { [name]: Number(a.slice(name.length + 3)) } : {};
};

const trainer = new Trainer({ task, totalSteps: steps, seed, ...flag("lr"), ...flag("nEmbd"), ...flag("nLayer"), ...flag("nHead"), ...flag("batchSize") });
console.log(`task=${task} params=${trainer.model.numParams.toLocaleString()} steps=${steps} seed=${seed} lr=${trainer.opts.lr} d=${trainer.opts.nEmbd} L=${trainer.opts.nLayer}`);

const loss: number[] = [];
const evals: [number, number, number][] = [];
const ev = () => {
  const { accuracy, tokenAccuracy } = trainer.evaluate(64);
  evals.push([trainer.step, accuracy, tokenAccuracy]);
  return { accuracy, tokenAccuracy };
};
ev();

const t0 = performance.now();
let ema = 0;
for (let s = 0; s < steps; s++) {
  const st = trainer.trainStep();
  loss.push(Math.round(st.loss * 1e5) / 1e5);
  ema = s === 0 ? st.loss : 0.95 * ema + 0.05 * st.loss;
  if (st.step % 100 === 0 || st.step === steps) {
    ev();
    const big = trainer.evaluate(512, 4242);
    console.log(
      `step ${String(st.step).padStart(5)} | loss ${ema.toFixed(4)} | exact ${(big.accuracy * 100).toFixed(1).padStart(5)}% | ` +
        `token ${(big.tokenAccuracy * 100).toFixed(1).padStart(5)}% | ${st.ms.toFixed(0)} ms/step`,
    );
  }
}
console.log(`done in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
const p = trainer.predict(trainer.task.probe);
console.log(`probe ${p.prompt}${p.output}  (expected ${p.expected}) ${p.correct ? "✓" : "✗"}`);

if (save) {
  const weights: Record<string, string> = {};
  for (const prm of trainer.model.params) {
    weights[prm.name] = Buffer.from(new Float32Array(prm.data).buffer).toString("base64");
  }
  mkdirSync("pretrained", { recursive: true });
  writeFileSync(`pretrained/${task}.json`, JSON.stringify({ opts: trainer.opts, step: trainer.step, loss, evals, weights }));
  console.log(`saved pretrained/${task}.json`);
}
