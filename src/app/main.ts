import { TrainingHost, type FromHost, type ToHost } from "./host.ts";
import { TASKS, type Task } from "../engine/tasks.ts";
import { RNG } from "../engine/rng.ts";
import type { TrainerOptions, Prediction } from "../engine/trainer.ts";
import { applyStatic, getLang, initialLang, saveLang, setLang, t, type Lang } from "./i18n.ts";

// Injected at build time (scripts/build.mjs).
declare const __WORKER_SRC__: string;
declare const __PRETRAINED__: Record<
  string,
  { opts: TrainerOptions; step: number; loss: number[]; evals: [number, number, number][]; weights: Record<string, string> }
>;

type TaskId = Task["id"];
const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = {
  task: "reverse" as TaskId,
  running: false,
  step: 0,
  totalSteps: 1500,
  losses: [] as number[],
  evals: [] as [number, number, number][],
  msEma: 0,
  source: "" as "" | "pretrained" | "live",
  prediction: null as Prediction | null,
  nHead: 4,
  params: 0,
  probeStep: 0,
  proof: null as null | { results: { name: string; maxRelError: number; checked: number; passed: boolean }[]; ms: number },
};

// ---------------------------------------------------------------------------
// Training host: Web Worker, with an in-page fallback
// ---------------------------------------------------------------------------
let send: (m: ToHost) => void;
let mode = "worker";
let lastInit: ToHost | null = null;
let gotReply = false;

function useMainThread() {
  mode = "main thread";
  $("mode-label").textContent = t("mode.main");
  const host = new TrainingHost(onHost, (fn) => setTimeout(fn, 16), 45);
  send = (m) => setTimeout(() => host.handle(m), 0);
  if (lastInit) send(lastInit);
}

function startHost() {
  try {
    const url = URL.createObjectURL(new Blob([__WORKER_SRC__], { type: "text/javascript" }));
    const w = new Worker(url);
    w.onmessage = (e) => { gotReply = true; onHost(e.data); };
    w.onerror = () => { if (!gotReply) { w.terminate(); useMainThread(); } };
    send = (m) => w.postMessage(m);
    $("mode-label").textContent = t("mode.worker");
    setTimeout(() => { if (!gotReply && mode === "worker") { w.terminate(); useMainThread(); } }, 2500);
  } catch {
    useMainThread();
  }
}

function sendInit(m: ToHost) {
  lastInit = m;
  send(m);
}

// ---------------------------------------------------------------------------
// Messages from the host
// ---------------------------------------------------------------------------
function onHost(m: FromHost) {
  switch (m.type) {
    case "ready":
      $("ro-params").textContent = m.params.toLocaleString();
      state.params = m.params;
      $("arch-count").textContent = t("params.total", { n: m.params.toLocaleString() });
      $("arch").innerHTML = m.layers
        .map((l) => `<div><b>${l.name}</b> ${l.shape[0]}×${l.shape[1]}</div>`)
        .join("");
      state.totalSteps = m.opts.totalSteps;
      state.nHead = m.opts.nHead;
      break;
    case "progress":
      // Steps still in flight from a live run that was replaced by a pretrained model.
      if (state.source === "pretrained") {
        state.running = false;
        renderReadouts();
        break;
      }
      for (const s of m.stats) {
        state.losses.push(s.loss);
        state.step = s.step;
        state.msEma = state.msEma ? 0.9 * state.msEma + 0.1 * s.ms : s.ms;
      }
      state.running = m.running;
      if (!m.running && state.step >= state.totalSteps) state.source = "live";
      renderReadouts();
      drawLoss();
      break;
    case "eval":
      if (state.source === "live" || state.running) {
        const last = state.evals[state.evals.length - 1];
        if (!last || last[0] !== m.step) state.evals.push([m.step, m.accuracy, m.tokenAccuracy]);
        drawAcc();
      }
      $("ro-acc").textContent = `${(m.accuracy * 100).toFixed(0)}%`;
      break;
    case "probe":
      state.prediction = m.prediction;
      state.probeStep = m.step;
      $("probe-step").textContent = t("probe.step", { step: m.step });
      renderProbe();
      renderAttention();
      break;
    case "gradcheck":
      state.proof = { results: m.results, ms: m.ms };
      renderProof(m.results, m.ms);
      break;
    case "error":
      $("probe-err").textContent = TASKS[state.task].validate(currentPrompt()) ? t(`err.${state.task}`) : m.message;
      break;
    case "export":
      break;
  }
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------
function readOpts(): Partial<TrainerOptions> {
  const v = (id: string) => ($(id) as HTMLInputElement).value;
  return {
    task: state.task,
    nLayer: +v("hp-layers"),
    nHead: +v("hp-heads"),
    nEmbd: +v("hp-width"),
    batchSize: +v("hp-batch"),
    lr: +v("hp-lr"),
    totalSteps: +v("hp-steps"),
    seed: +v("hp-seed") || 1,
  };
}

function trainFromScratch() {
  const opts = readOpts();
  state.losses = [];
  state.evals = [];
  state.step = 0;
  state.msEma = 0;
  state.source = "live";
  state.totalSteps = opts.totalSteps!;
  sendInit({ type: "init", opts });
  send({ type: "probe", prompt: currentPrompt() });
  send({ type: "run" });
  state.running = true;
  $("loss-src").textContent = "";
  $("rail-note").textContent = t("note.live");
  renderReadouts();
  drawLoss();
  drawAcc();
}

function b64ToArray(b64: string): number[] {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return Array.from(new Float32Array(bytes.buffer));
}

function loadPretrained() {
  const p = __PRETRAINED__[state.task];
  if (!p) return;
  const weights: Record<string, number[]> = {};
  for (const k in p.weights) weights[k] = b64ToArray(p.weights[k]);
  // Show the hyperparameters these weights were trained with.
  const set = (id: string, v: number) => (($(id) as HTMLInputElement).value = String(v));
  set("hp-layers", p.opts.nLayer);
  set("hp-heads", p.opts.nHead);
  set("hp-width", p.opts.nEmbd);
  set("hp-batch", p.opts.batchSize);
  set("hp-lr", p.opts.lr);
  set("hp-steps", p.opts.totalSteps);
  set("hp-seed", p.opts.seed);
  state.losses = p.loss.slice();
  state.evals = p.evals.slice();
  state.step = p.step;
  state.totalSteps = p.opts.totalSteps;
  state.source = "pretrained";
  state.running = false;
  state.msEma = 0;
  sendInit({ type: "init", opts: p.opts, weights, step: p.step });
  send({ type: "probe", prompt: currentPrompt() });
  $("loss-src").textContent = t("loss.recorded");
  $("rail-note").textContent = t("note.pretrained");
  renderReadouts();
  drawLoss();
  drawAcc();
}

function setTask(id: TaskId) {
  state.task = id;
  for (const b of document.querySelectorAll<HTMLButtonElement>(".seg button")) {
    b.setAttribute("aria-pressed", String(b.dataset.task === id));
  }
  const task = TASKS[id];
  ($("probe-input") as HTMLInputElement).value = task.probe;
  renderTaskText();
  $("probe-err").textContent = "";
  loadPretrained();
}

function renderTaskText() {
  const id = state.task;
  const ex = TASKS[id].sample(new RNG(5));
  $("task-fmt").innerHTML = `${escape(ex.prompt)}<b>${escape(ex.answer)}</b><br><span>${t(`desc.${id}`)}</span>`;
  $("insight").textContent = t(`insight.${id}`);
}

function currentPrompt(): string {
  return ($("probe-input") as HTMLInputElement).value.trim();
}

let probeTimer = 0;
function onProbeInput() {
  clearTimeout(probeTimer);
  probeTimer = window.setTimeout(() => {
    const p = currentPrompt();
    const err = TASKS[state.task].validate(p);
    $("probe-err").textContent = err ? t(`err.${state.task}`) : "";
    if (!err) send({ type: "probe", prompt: p });
  }, 180);
}

function randomProblem() {
  const seed = (Math.random() * 1e9) | 0;
  const { prompt } = TASKS[state.task].sample(new RNG(seed));
  ($("probe-input") as HTMLInputElement).value = prompt;
  $("probe-err").textContent = "";
  send({ type: "probe", prompt });
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
function escape(s: string) {
  return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);
}

function renderReadouts() {
  const st = $("ro-state");
  const s = state.running ? "training" : state.step > 0 ? "trained" : "idle";
  st.dataset.s = s;
  st.textContent = t(state.running ? "state.training" : state.source === "pretrained" ? "state.loaded" : state.step > 0 ? "state.paused" : "state.idle");
  if (!state.running && state.step >= state.totalSteps && state.source === "live") st.textContent = t("state.done");
  $("ro-step").textContent = `${state.step}`;
  $("ro-ms").textContent = state.msEma ? state.msEma.toFixed(0) : "—";
  const pause = $<HTMLButtonElement>("btn-pause");
  pause.textContent = t(state.running ? "btn.pause" : "btn.resume");
  pause.disabled = !state.running && (state.source !== "live" || state.step >= state.totalSteps);
}

function renderProbe() {
  const p = state.prediction;
  if (!p) return;
  const toks = [...p.prompt].map((c) => `<span class="tok">${escape(c)}</span>`);
  const outs = [...p.output].map(
    (c, i) =>
      `<span class="tok out${c === p.expected[i] ? "" : " wrong"}" title="${escape(t("probe.expectedTip", { answer: p.expected[i] }))}">${escape(c)}</span>`,
  );
  const verdict = p.correct
    ? `<span class="verdict ok">${t("probe.correct")}</span>`
    : `<span class="verdict no">${escape(t("probe.expected", { answer: p.expected }))}</span>`;
  $("probe-tokens").innerHTML = toks.join("") + outs.join("") + `&nbsp;&nbsp;${verdict}`;
  $("probe-topk").innerHTML = p.topk
    .map(
      (col) =>
        `<div class="col">${col
          .map(
            (o, i) =>
              `<div class="bar${i ? " second" : ""}"><span class="mono">${escape(o.token)}</span><span style="--w:${(o.p * 100).toFixed(1)}%" title="${(o.p * 100).toFixed(1)}%"></span></div>`,
          )
          .join("")}</div>`,
    )
    .join("");
}

function css(name: string) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function parseColor(c: string): [number, number, number] {
  const probe = document.createElement("canvas").getContext("2d")!;
  probe.fillStyle = c;
  const v = probe.fillStyle as string;
  if (v.startsWith("#")) return [1, 3, 5].map((i) => parseInt(v.slice(i, i + 2), 16)) as [number, number, number];
  const m = v.match(/\d+/g)!;
  return [+m[0], +m[1], +m[2]];
}

function setupCanvas(cv: HTMLCanvasElement, w: number, h: number) {
  const dpr = window.devicePixelRatio || 1;
  cv.width = Math.round(w * dpr);
  cv.height = Math.round(h * dpr);
  const ctx = cv.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

interface Axis { x0: number; x1: number; y0: number; y1: number }

function drawFrame(ctx: CanvasRenderingContext2D, W: number, H: number, pad: Axis, xMax: number, yTicks: { v: number; label: string; y: number }[]) {
  ctx.clearRect(0, 0, W, H);
  ctx.font = `11px ${css("--f-mono")}`;
  ctx.fillStyle = css("--muted");
  ctx.strokeStyle = css("--line");
  ctx.lineWidth = 1;
  ctx.textBaseline = "middle";
  ctx.textAlign = "right";
  for (const t of yTicks) {
    ctx.beginPath();
    ctx.moveTo(pad.x0, Math.round(t.y) + 0.5);
    ctx.lineTo(W - pad.x1, Math.round(t.y) + 0.5);
    ctx.stroke();
    ctx.fillText(t.label, pad.x0 - 6, t.y);
  }
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const step = niceStep(xMax / 4);
  for (let v = 0; v <= xMax; v += step) {
    const x = pad.x0 + (v / xMax) * (W - pad.x0 - pad.x1);
    ctx.textAlign = x > W - pad.x1 - 16 ? "right" : "center";
    ctx.fillText(String(v), Math.min(x, W - 2), H - pad.y1 + 6);
  }
}

function niceStep(raw: number) {
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const n = raw / p;
  return (n < 1.5 ? 1 : n < 3.5 ? 2.5 : n < 7.5 ? 5 : 10) * p;
}

function drawLoss() {
  const cv = $<HTMLCanvasElement>("chart-loss");
  const W = cv.clientWidth, H = cv.clientHeight;
  if (!W) return;
  const ctx = setupCanvas(cv, W, H);
  const pad = { x0: 44, x1: 10, y0: 10, y1: 22 };
  const xMax = Math.max(state.totalSteps, state.losses.length, 10);
  const lo = -4, hi = 1; // log10 range: 1e-4 .. 10
  const yOf = (l: number) => {
    const v = Math.min(hi, Math.max(lo, Math.log10(Math.max(l, 1e-6))));
    return pad.y0 + ((hi - v) / (hi - lo)) * (H - pad.y0 - pad.y1);
  };
  const xOf = (i: number) => pad.x0 + (i / xMax) * (W - pad.x0 - pad.x1);
  const ticks = [10, 1, 0.1, 0.01, 0.001, 0.0001].map((v) => ({ v, label: v >= 1 ? String(v) : v.toExponential(0).replace("e-", "e−"), y: yOf(v) }));
  drawFrame(ctx, W, H, pad, xMax, ticks);
  const L = state.losses;
  if (!L.length) {
    ctx.fillStyle = css("--muted");
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(t("loss.empty"), W / 2, H / 2);
    return;
  }
  ctx.strokeStyle = css("--line");
  ctx.lineWidth = 1;
  ctx.beginPath();
  L.forEach((l, i) => (i ? ctx.lineTo(xOf(i + 1), yOf(l)) : ctx.moveTo(xOf(1), yOf(l))));
  ctx.stroke();
  ctx.strokeStyle = css("--accent");
  ctx.lineWidth = 2;
  ctx.beginPath();
  let ema = L[0];
  L.forEach((l, i) => {
    ema = i ? 0.95 * ema + 0.05 * l : l;
    i ? ctx.lineTo(xOf(i + 1), yOf(ema)) : ctx.moveTo(xOf(1), yOf(ema));
  });
  ctx.stroke();
  const ex = xOf(L.length), ey = yOf(ema);
  ctx.fillStyle = css("--accent");
  ctx.beginPath();
  ctx.arc(ex, ey, 3.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.font = `600 11px ${css("--f-mono")}`;
  ctx.textAlign = ex > W - 70 ? "right" : "left";
  ctx.textBaseline = "bottom";
  ctx.fillText(ema.toFixed(4), ex + (ex > W - 70 ? -6 : 6), ey - 4);
}

function drawAcc() {
  const cv = $<HTMLCanvasElement>("chart-acc");
  const W = cv.clientWidth, H = cv.clientHeight;
  if (!W) return;
  const ctx = setupCanvas(cv, W, H);
  const pad = { x0: 44, x1: 10, y0: 10, y1: 22 };
  const xMax = Math.max(state.totalSteps, 10);
  const yOf = (a: number) => pad.y0 + (1 - a) * (H - pad.y0 - pad.y1);
  const xOf = (s: number) => pad.x0 + (s / xMax) * (W - pad.x0 - pad.x1);
  drawFrame(ctx, W, H, pad, xMax, [0, 0.25, 0.5, 0.75, 1].map((v) => ({ v, label: `${v * 100}%`, y: yOf(v) })));
  const E = state.evals;
  if (!E.length) return;
  const line = (idx: 1 | 2, color: string, width: number) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    E.forEach((e, i) => (i ? ctx.lineTo(xOf(e[0]), yOf(e[idx])) : ctx.moveTo(xOf(e[0]), yOf(e[idx]))));
    ctx.stroke();
  };
  line(2, css("--muted"), 1.25);
  line(1, css("--good"), 2);
  const last = E[E.length - 1];
  ctx.fillStyle = css("--good");
  ctx.beginPath();
  ctx.arc(xOf(last[0]), yOf(last[1]), 3.5, 0, Math.PI * 2);
  ctx.fill();
}

// Attention heatmaps -----------------------------------------------------------
const hover = { layer: -1, head: -1, q: -1, k: -1 };

function renderAttention() {
  const p = state.prediction;
  if (!p) return;
  const grid = $("att-grid");
  const L = p.attention.length, Hh = p.attention[0].length;
  const want = L * Hh;
  if (grid.children.length !== want) {
    grid.innerHTML = "";
    for (let l = 0; l < L; l++)
      for (let h = 0; h < Hh; h++) {
        const cell = document.createElement("div");
        cell.className = "att-cell";
        cell.innerHTML = `<div class="label">${t("att.cell", { l: l + 1, h: h + 1 })}</div><canvas aria-label="attention map layer ${l + 1} head ${h + 1}"></canvas><div class="att-tip">&nbsp;</div>`;
        const cv = cell.querySelector("canvas")!;
        cv.addEventListener("mousemove", (e) => onHeatHover(e, cv, l, h));
        cv.addEventListener("mouseleave", () => {
          Object.assign(hover, { layer: -1, head: -1, q: -1, k: -1 });
          renderAttention();
        });
        grid.appendChild(cell);
      }
  }
  const cells = grid.querySelectorAll<HTMLDivElement>(".att-cell");
  for (let l = 0; l < L; l++)
    for (let h = 0; h < Hh; h++) drawHeat(cells[l * Hh + h], p, l, h);
}

const LABEL = 18;

function drawHeat(cell: HTMLDivElement, p: Prediction, l: number, h: number) {
  const cv = cell.querySelector("canvas")!;
  const S = cv.clientWidth;
  if (!S) return;
  const ctx = setupCanvas(cv, S, S);
  const T = p.tokens.length;
  const cs = (S - LABEL) / T;
  const A = p.attention[l][h];
  const lo = parseColor(css("--surface-2")), hi = parseColor(css("--heat-hi"));
  const promptLen = p.prompt.length;
  ctx.clearRect(0, 0, S, S);
  for (let q = 0; q < T; q++)
    for (let k = 0; k <= q; k++) {
      const a = Math.pow(A[q * T + k], 0.6);
      const c = lo.map((v, i) => Math.round(v + (hi[i] - v) * a));
      ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
      ctx.fillRect(LABEL + k * cs, LABEL + q * cs, Math.ceil(cs), Math.ceil(cs));
    }
  // Boundary between prompt and answer.
  ctx.strokeStyle = css("--signal");
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  const yb = LABEL + (promptLen - 1) * cs + 0.5;
  ctx.beginPath();
  ctx.moveTo(LABEL, yb);
  ctx.lineTo(S, yb);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.font = `${Math.min(11, cs * 0.8)}px ${css("--f-mono")}`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (let i = 0; i < T; i++) {
    const isH = hover.layer === l && hover.head === h;
    ctx.fillStyle = isH && (i === hover.k) ? css("--signal") : css("--muted");
    ctx.fillText(p.tokens[i], LABEL + (i + 0.5) * cs, LABEL / 2);
    ctx.fillStyle = isH && i === hover.q ? css("--signal") : i >= promptLen - 1 ? css("--ink") : css("--muted");
    ctx.fillText(p.tokens[i], LABEL / 2, LABEL + (i + 0.5) * cs);
  }
  if (hover.layer === l && hover.head === h && hover.q >= 0) {
    ctx.strokeStyle = css("--signal");
    ctx.lineWidth = 2;
    ctx.strokeRect(LABEL + hover.k * cs, LABEL + hover.q * cs, cs, cs);
  }
}

function onHeatHover(e: MouseEvent, cv: HTMLCanvasElement, l: number, h: number) {
  const p = state.prediction;
  if (!p) return;
  const r = cv.getBoundingClientRect();
  const T = p.tokens.length;
  const cs = (r.width - LABEL) / T;
  const k = Math.floor((e.clientX - r.left - LABEL) / cs);
  const q = Math.floor((e.clientY - r.top - LABEL) / cs);
  const tip = cv.parentElement!.querySelector(".att-tip")!;
  if (q < 0 || k < 0 || q >= T || k > q) {
    tip.innerHTML = "&nbsp;";
    return;
  }
  Object.assign(hover, { layer: l, head: h, q, k });
  const w = p.attention[l][h][q * T + k];
  const next = q >= p.prompt.length - 1 ? t("att.writes", { tok: escape((p.prompt + p.output)[q + 1]) }) : "";
  tip.innerHTML = t("att.tip", { q, qt: escape(p.tokens[q]), next, k, kt: escape(p.tokens[k]), pct: (w * 100).toFixed(1) });
  drawHeat(cv.parentElement as HTMLDivElement, p, l, h);
}

// Gradient check -----------------------------------------------------------------
function renderProof(results: { name: string; maxRelError: number; checked: number; passed: boolean }[], ms: number) {
  const table = $<HTMLTableElement>("proof-table");
  table.hidden = false;
  table.tBodies[0].innerHTML = results
    .map(
      (r) =>
        `<tr><td>${escape(r.name)}</td><td class="mono">${r.maxRelError.toExponential(1)}</td><td><span class="pill ${r.passed ? "ok" : "no"}">${t(r.passed ? "proof.pass" : "proof.fail")}</span></td></tr>`,
    )
    .join("");
  const n = results.reduce((s, r) => s + r.checked, 0);
  $("proof-note").textContent = t("proof.note", { n, s: (ms / 1000).toFixed(1) });
  const b = $<HTMLButtonElement>("btn-gradcheck");
  b.disabled = false;
  b.textContent = t("proof.again");
}

// ---------------------------------------------------------------------------
// Wire up
// ---------------------------------------------------------------------------
// Language ------------------------------------------------------------------------
function setLanguage(l: Lang) {
  setLang(l);
  saveLang(l);
  applyStatic();
  for (const b of document.querySelectorAll<HTMLButtonElement>(".lang button")) {
    b.setAttribute("aria-pressed", String(b.dataset.lang === l));
  }
  $("mode-label").textContent = t(mode === "worker" ? "mode.worker" : "mode.main");
  if (state.params) $("arch-count").textContent = t("params.total", { n: state.params.toLocaleString() });
  if (state.prediction) $("probe-step").textContent = t("probe.step", { step: state.probeStep });
  $("loss-src").textContent = state.source === "pretrained" ? t("loss.recorded") : "";
  $("rail-note").textContent = state.source === "pretrained" ? t("note.pretrained") : state.source === "live" ? t("note.live") : "";
  const err = $("probe-err");
  if (err.textContent) err.textContent = t(`err.${state.task}`);
  const gc = $<HTMLButtonElement>("btn-gradcheck");
  gc.textContent = gc.disabled ? t("proof.checking") : t(state.proof ? "proof.again" : "proof.run");
  if (state.proof) renderProof(state.proof.results, state.proof.ms);
  renderTaskText();
  renderReadouts();
  renderProbe();
  $("att-grid").innerHTML = ""; // rebuilt with translated labels
  renderAttention();
  drawLoss();
  drawAcc();
}
for (const b of document.querySelectorAll<HTMLButtonElement>(".lang button")) {
  b.addEventListener("click", () => setLanguage(b.dataset.lang as Lang));
}
setLang(initialLang());
applyStatic();
for (const b of document.querySelectorAll<HTMLButtonElement>(".lang button")) {
  b.setAttribute("aria-pressed", String(b.dataset.lang === getLang()));
}

startHost();
for (const b of document.querySelectorAll<HTMLButtonElement>(".seg button")) {
  b.addEventListener("click", () => setTask(b.dataset.task as TaskId));
}
$("btn-scratch").addEventListener("click", trainFromScratch);
$("btn-pretrained").addEventListener("click", loadPretrained);
$("btn-pause").addEventListener("click", () => {
  if (state.running) send({ type: "pause" });
  else if (state.source === "live" && state.step < state.totalSteps) {
    send({ type: "run" });
    state.running = true;
    renderReadouts();
  }
});
$("btn-random").addEventListener("click", randomProblem);
$("probe-input").addEventListener("input", onProbeInput);
$("btn-gradcheck").addEventListener("click", () => {
  const b = $<HTMLButtonElement>("btn-gradcheck");
  b.disabled = true;
  b.textContent = t("proof.checking");
  send({ type: "gradcheck" });
});

const redraw = () => { drawLoss(); drawAcc(); renderAttention(); };
let rz = 0;
window.addEventListener("resize", () => { cancelAnimationFrame(rz); rz = requestAnimationFrame(redraw); });
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);
new MutationObserver(redraw).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
document.fonts?.ready.then(redraw);

setTask("reverse");
