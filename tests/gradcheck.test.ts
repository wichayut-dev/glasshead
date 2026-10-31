import { test } from "node:test";
import assert from "node:assert/strict";
import { runAllGradChecks } from "../src/engine/gradcheck.ts";

// Every backward pass in the engine is compared against central finite differences.
for (const r of runAllGradChecks()) {
  test(`gradcheck: ${r.name}`, () => {
    assert.ok(r.passed, `${r.name}: max relative error ${r.maxRelError.toExponential(2)} over ${r.checked} elements`);
  });
}
