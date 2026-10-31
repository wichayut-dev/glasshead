// UI strings in English and Thai. Static elements carry data-i18n="key" (text) or
// data-i18n-html="key" (markup); dynamic text calls t(key, vars).

export type Lang = "en" | "th";

const en = {
  "lang.label": "Language",
  "mast.eyebrow": "Transformer from first principles · no ML libraries",
  "mast.lede":
    "A small transformer, the same kind of neural network behind today’s AI chatbots, with its <b>own autograd engine</b>, written from scratch in TypeScript. It trains live in your browser on algorithmic tasks, and every attention head is drawn as it learns, so you can see the algorithm the network discovers.",
  "ro.state": "State",
  "ro.step": "Step",
  "ro.acc": "Exact acc.",
  "ro.params": "Params",
  "ro.ms": "ms / step",
  "state.idle": "idle",
  "state.training": "training",
  "state.loaded": "loaded",
  "state.paused": "paused",
  "state.done": "done",
  "panel.task": "Task",
  "task.reverse": "Reverse",
  "task.sort": "Sort",
  "task.add": "Add",
  "desc.reverse": "8 digits, then write them backwards",
  "desc.sort": "8 digits, then write them in order",
  "desc.add": "two 3-digit numbers; sum written ones-digit first",
  "btn.scratch": "Train from scratch",
  "btn.pause": "Pause",
  "btn.resume": "Resume",
  "btn.pretrained": "Load trained model",
  "hp.summary": "Hyperparameters",
  "hp.layers": "Layers",
  "hp.heads": "Heads",
  "hp.width": "Width (d_model)",
  "hp.batch": "Batch",
  "hp.lr": "Peak LR",
  "hp.steps": "Steps",
  "hp.seed": "Seed",
  "hp.note": "Same seed gives the same initial weights, batches and loss curve. Changes apply on the next “Train from scratch”.",
  "mode.worker": "web worker",
  "mode.main": "main thread",
  "note.live": "Training on fresh random problems every step. Watch the attention maps sharpen.",
  "note.pretrained":
    "These weights were trained by this same engine (seed 1337). Press “Train from scratch” to watch it learn live.",
  "loss.title": "Training loss",
  "loss.sub": "cross-entropy, log scale",
  "loss.smoothed": "smoothed",
  "loss.batch": "per batch",
  "loss.recorded": "· recorded run, same code",
  "loss.empty": "no training yet",
  "acc.title": "Held-out accuracy",
  "acc.sub": "64 unseen problems",
  "acc.exact": "exact match",
  "acc.token": "per token",
  "probe.title": "Ask the model",
  "probe.prompt": "Prompt",
  "probe.random": "Random problem",
  "probe.topk": "Top-3 next-token probabilities at each answer position",
  "probe.step": "model at step {step}",
  "probe.correct": "correct",
  "probe.expected": "expected {answer}",
  "probe.expectedTip": "expected {answer}",
  "err.reverse": "Type exactly 8 digits followed by '>' (e.g. 31415926>)",
  "err.sort": "Type exactly 8 digits followed by '>' (e.g. 73019482>)",
  "err.add": "Format: 3 digits, '+', 3 digits, '=' (e.g. 047+385=)",
  "att.title": "Attention, every layer and head",
  "att.sub": "rows = query position · columns = key position",
  "att.cell": "Layer {l} · head {h}",
  "att.tip": "pos {q} '<b>{qt}</b>'{next} looks at pos {k} '<b>{kt}</b>' · {pct}%",
  "att.writes": " → writes <b>{tok}</b>",
  "insight.reverse":
    "A clean anti-diagonal: the k-th output token attends to input position (8 − k). The model has learned to compute a pointer from its own position.",
  "insight.sort":
    "Each output token attends to the input digit that comes next in sorted order. Attention works as a content-addressed lookup for 'the smallest digit larger than what I just wrote'.",
  "insight.add":
    "The sum is written right-to-left (ones digit first). Each output digit attends to the matching column of both numbers, plus the previous column to work out the carry.",
  "proof.title": "Prove the gradients",
  "proof.body":
    "Every backward pass is hand-written. This runs a finite-difference check of each one in your browser, then of the whole model end to end.",
  "proof.run": "Run gradient check",
  "proof.again": "Run again",
  "proof.checking": "Checking…",
  "proof.op": "Operation",
  "proof.err": "Max rel. error",
  "proof.pass": "pass",
  "proof.fail": "fail",
  "proof.note": "{n} gradient entries compared against central differences (ε = 1e−5) in {s} s. Pass threshold: 1e−5.",
  "fused.title": "Fused attention backward",
  "fused.body": "Attention is one op with a hand-derived gradient, so the engine never materialises per-head tensors.",
  "params.title": "Parameters",
  "params.total": "{n} total",
  footer:
    "Engine: reverse-mode autodiff on Float64 tensors · pre-LayerNorm transformer block · GELU · AdamW with warmup + cosine decay · gradient clipping · seeded RNG. Training runs in a Web Worker so the page stays responsive.",
};

export type Key = keyof typeof en;

const th: Record<Key, string> = {
  "lang.label": "ภาษา",
  "mast.eyebrow": "Transformer ที่สร้างจากหลักการพื้นฐาน · ไม่ใช้ไลบรารี ML",
  "mast.lede":
    "โมเดล Transformer ขนาดเล็ก (โครงสร้างเดียวกับ AI แชทบอทในปัจจุบัน) ที่มี<b>ระบบ autograd ของตัวเอง</b> เขียนขึ้นใหม่ทั้งหมดด้วย TypeScript เทรนสดในเบราว์เซอร์ด้วยโจทย์เชิงอัลกอริทึม และวาด attention ทุก head ระหว่างที่มันเรียนรู้ ทำให้เห็นว่าโมเดลค้นพบอัลกอริทึมอะไร",
  "ro.state": "สถานะ",
  "ro.step": "Step",
  "ro.acc": "แม่นยำ",
  "ro.params": "พารามิเตอร์",
  "ro.ms": "ms / step",
  "state.idle": "รอ",
  "state.training": "กำลังเทรน",
  "state.loaded": "โหลดแล้ว",
  "state.paused": "หยุดชั่วคราว",
  "state.done": "เสร็จ",
  "panel.task": "โจทย์",
  "task.reverse": "กลับลำดับ",
  "task.sort": "เรียงลำดับ",
  "task.add": "บวกเลข",
  "desc.reverse": "ตัวเลข 8 หลัก แล้วเขียนกลับหลัง",
  "desc.sort": "ตัวเลข 8 หลัก แล้วเขียนเรียงจากน้อยไปมาก",
  "desc.add": "เลข 3 หลักสองตัว เขียนผลบวกจากหลักหน่วยก่อน",
  "btn.scratch": "เทรนใหม่ตั้งแต่ศูนย์",
  "btn.pause": "หยุดชั่วคราว",
  "btn.resume": "เทรนต่อ",
  "btn.pretrained": "โหลดโมเดลที่เทรนแล้ว",
  "hp.summary": "ไฮเปอร์พารามิเตอร์",
  "hp.layers": "จำนวนชั้น",
  "hp.heads": "จำนวน head",
  "hp.width": "ความกว้าง (d_model)",
  "hp.batch": "Batch",
  "hp.lr": "Learning rate สูงสุด",
  "hp.steps": "จำนวน step",
  "hp.seed": "Seed",
  "hp.note": "Seed เดียวกันจะได้น้ำหนักเริ่มต้น ข้อมูลแต่ละ batch และกราฟ loss เหมือนเดิมทุกครั้ง ค่าที่เปลี่ยนจะมีผลเมื่อกด “เทรนใหม่ตั้งแต่ศูนย์”",
  "mode.worker": "web worker",
  "mode.main": "main thread",
  "note.live": "แต่ละ step เทรนด้วยโจทย์สุ่มชุดใหม่ ลองดู attention ค่อย ๆ คมชัดขึ้น",
  "note.pretrained": "น้ำหนักชุดนี้เทรนด้วยโค้ดเดียวกันนี้ (seed 1337) กด “เทรนใหม่ตั้งแต่ศูนย์” เพื่อดูมันเรียนรู้สด ๆ",
  "loss.title": "Loss ระหว่างเทรน",
  "loss.sub": "cross-entropy, สเกล log",
  "loss.smoothed": "ค่าเฉลี่ย",
  "loss.batch": "แต่ละ batch",
  "loss.recorded": "· ผลที่บันทึกไว้ โค้ดเดียวกัน",
  "loss.empty": "ยังไม่ได้เทรน",
  "acc.title": "ความแม่นยำกับโจทย์ที่ไม่เคยเห็น",
  "acc.sub": "โจทย์ใหม่ 64 ข้อ",
  "acc.exact": "ถูกทั้งคำตอบ",
  "acc.token": "ถูกรายตัว",
  "probe.title": "ลองถามโมเดล",
  "probe.prompt": "โจทย์",
  "probe.random": "สุ่มโจทย์",
  "probe.topk": "3 ตัวเลือกที่โมเดลให้ความน่าจะเป็นสูงสุด ในแต่ละตำแหน่งของคำตอบ",
  "probe.step": "โมเดลที่ step {step}",
  "probe.correct": "ถูกต้อง",
  "probe.expected": "คำตอบที่ถูกคือ {answer}",
  "probe.expectedTip": "คำตอบที่ถูก: {answer}",
  "err.reverse": "พิมพ์ตัวเลข 8 หลักแล้วตามด้วย '>' (เช่น 31415926>)",
  "err.sort": "พิมพ์ตัวเลข 8 หลักแล้วตามด้วย '>' (เช่น 73019482>)",
  "err.add": "รูปแบบ: เลข 3 หลัก '+' เลข 3 หลัก '=' (เช่น 047+385=)",
  "att.title": "Attention ทุกชั้นและทุก head",
  "att.sub": "แถว = ตำแหน่งที่กำลังเขียน · คอลัมน์ = ตำแหน่งที่มอง",
  "att.cell": "ชั้น {l} · head {h}",
  "att.tip": "ตำแหน่ง {q} '<b>{qt}</b>'{next} มองตำแหน่ง {k} '<b>{kt}</b>' · {pct}%",
  "att.writes": " → เขียน <b>{tok}</b>",
  "insight.reverse":
    "เห็นเป็นเส้นทแยงชัดเจน: ตัวที่ k ของคำตอบมองไปที่ตำแหน่ง (8 − k) ของโจทย์ แปลว่าโมเดลเรียนรู้ที่จะคำนวณ \"ตัวชี้\" จากตำแหน่งของตัวเอง",
  "insight.sort":
    "แต่ละตัวของคำตอบมองไปที่ตัวเลขในโจทย์ที่มาถัดไปตามลำดับ attention ทำหน้าที่เหมือนการค้นหาด้วยเนื้อหา คือหา \"ตัวเลขที่เล็กที่สุดที่มากกว่าตัวที่เพิ่งเขียน\"",
  "insight.add":
    "ผลบวกเขียนจากขวาไปซ้าย (หลักหน่วยก่อน) แต่ละหลักของคำตอบมองไปที่หลักเดียวกันของเลขทั้งสองตัว และหลักก่อนหน้าเพื่อคำนวณการทด",
  "proof.title": "พิสูจน์ gradient",
  "proof.body":
    "backward pass ทุกตัวเขียนด้วยมือ ปุ่มนี้จะตรวจแต่ละตัวด้วย finite difference ในเบราว์เซอร์ แล้วตรวจทั้งโมเดลแบบ end to end",
  "proof.run": "ตรวจ gradient",
  "proof.again": "ตรวจอีกครั้ง",
  "proof.checking": "กำลังตรวจ…",
  "proof.op": "การคำนวณ",
  "proof.err": "Relative error สูงสุด",
  "proof.pass": "ผ่าน",
  "proof.fail": "ไม่ผ่าน",
  "proof.note": "เทียบ gradient {n} ค่ากับ central difference (ε = 1e−5) ใช้เวลา {s} วินาที เกณฑ์ผ่าน: 1e−5",
  "fused.title": "Backward ของ attention แบบ fused",
  "fused.body": "attention เป็น op เดียวที่ derive สูตร gradient ด้วยมือ จึงไม่ต้องสร้าง tensor แยกของแต่ละ head",
  "params.title": "พารามิเตอร์",
  "params.total": "ทั้งหมด {n}",
  footer:
    "Engine: autodiff แบบ reverse-mode บน tensor Float64 · บล็อก Transformer แบบ pre-LayerNorm · GELU · AdamW พร้อม warmup + cosine decay · gradient clipping · RNG แบบกำหนด seed การเทรนรันใน Web Worker หน้าเว็บจึงไม่ค้าง",
};

const dict: Record<Lang, Record<Key, string>> = { en, th };

let current: Lang = "en";

export function getLang(): Lang {
  return current;
}

export function setLang(l: Lang): void {
  current = l;
}

export function t(key: Key, vars: Record<string, string | number> = {}): string {
  return dict[current][key].replace(/\{(\w+)\}/g, (_, k) => String(vars[k] ?? `{${k}}`));
}

/** Fills every element marked data-i18n / data-i18n-html. */
export function applyStatic(root: ParentNode = document): void {
  for (const el of root.querySelectorAll<HTMLElement>("[data-i18n]")) el.textContent = t(el.dataset.i18n as Key);
  for (const el of root.querySelectorAll<HTMLElement>("[data-i18n-html]")) el.innerHTML = t(el.dataset.i18nHtml as Key);
  document.documentElement.lang = current;
}

export function initialLang(): Lang {
  try {
    const saved = localStorage.getItem("glasshead.lang");
    if (saved === "en" || saved === "th") return saved;
  } catch {
    /* storage unavailable */
  }
  return navigator.language?.toLowerCase().startsWith("th") ? "th" : "en";
}

export function saveLang(l: Lang): void {
  try {
    localStorage.setItem("glasshead.lang", l);
  } catch {
    /* storage unavailable */
  }
}
