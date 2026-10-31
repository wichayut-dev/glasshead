import { test } from "node:test";
import assert from "node:assert/strict";
import { Trainer } from "../src/engine/trainer.ts";
import { TASKS, decode, encode } from "../src/engine/tasks.ts";
import { RNG } from "../src/engine/rng.ts";
import { lrSchedule } from "../src/engine/optim.ts";

test("tokenizer round-trips", () => {
  assert.equal(decode(encode("479+386=5680")), "479+386=5680");
  assert.throws(() => encode("abc"));
});

test("task solvers are correct", () => {
  assert.equal(TASKS.reverse.solve("31415926>"), "62951413");
  assert.equal(TASKS.sort.solve("73019482>"), "01234789");
  assert.equal(TASKS.add.solve("479+386="), "5680"); // 865, ones digit first
  assert.equal(TASKS.add.solve("999+999="), "8991"); // 1998
  const rng = new RNG(1);
  for (let i = 0; i < 200; i++) {
    for (const t of Object.values(TASKS)) {
      const { prompt, answer } = t.sample(rng);
      assert.equal(prompt.length, t.promptLen);
      assert.equal(answer.length, t.answerLen);
      assert.equal(t.validate(prompt), null);
    }
  }
});

test("training is bit-for-bit reproducible from a seed", () => {
  const run = () => {
    const t = new Trainer({ task: "sort", seed: 7, nEmbd: 16, nHead: 2, nLayer: 1, batchSize: 4 });
    return Array.from({ length: 5 }, () => t.trainStep().loss);
  };
  assert.deepEqual(run(), run());
});

test("LR schedule warms up then decays to the floor", () => {
  assert.ok(lrSchedule(0, 1, 10, 100) < lrSchedule(9, 1, 10, 100));
  assert.equal(lrSchedule(9, 1, 10, 100), 1);
  assert.ok(Math.abs(lrSchedule(100, 1, 10, 100) - 0.1) < 1e-12);
});

test("the model actually learns: reverse reaches >95% token accuracy", () => {
  const t = new Trainer({ task: "reverse", seed: 1337, nEmbd: 32, nHead: 2, nLayer: 1, batchSize: 16, totalSteps: 200 });
  const before = t.evaluate(64).tokenAccuracy;
  for (let i = 0; i < 200; i++) t.trainStep();
  const after = t.evaluate(64).tokenAccuracy;
  assert.ok(before < 0.3, `untrained accuracy should be near chance, got ${before}`);
  assert.ok(after > 0.95, `trained token accuracy ${after}`);
});
