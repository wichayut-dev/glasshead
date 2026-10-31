import { RNG } from "./rng.ts";

/** Shared vocabulary for every task, so one architecture can learn any of them. */
export const VOCAB = "0123456789+=>";
export const VOCAB_SIZE = VOCAB.length;

export function encode(s: string): Int32Array {
  const out = new Int32Array(s.length);
  for (let i = 0; i < s.length; i++) {
    const id = VOCAB.indexOf(s[i]);
    if (id < 0) throw new Error(`character '${s[i]}' is not in the vocabulary`);
    out[i] = id;
  }
  return out;
}

export function decode(ids: ArrayLike<number>): string {
  let s = "";
  for (let i = 0; i < ids.length; i++) s += VOCAB[ids[i]];
  return s;
}

export interface Task {
  id: "reverse" | "sort" | "add";
  title: string;
  /** The model sees `promptLen` tokens, then must produce `answerLen` tokens. */
  promptLen: number;
  answerLen: number;
  sample(rng: RNG): { prompt: string; answer: string };
  /** Ground-truth algorithm, used for scoring. */
  solve(prompt: string): string;
  /** Returns an error message if a user-typed prompt is not valid for this task. */
  validate(prompt: string): string | null;
  /** A prompt for the attention probe that makes the learned algorithm easy to see. */
  probe: string;
  /** What the attention maps should reveal once the model has learned the task. */
  insight: string;
}

const DIGITS = 8;

function randDigits(rng: RNG, n: number): string {
  let s = "";
  for (let i = 0; i < n; i++) s += String(rng.int(10));
  return s;
}

function checkDigits(prompt: string, n: number, sep: string): string | null {
  if (prompt.length !== n + 1) return `type exactly ${n} digits followed by '${sep}'`;
  if (!/^\d+$/.test(prompt.slice(0, n))) return "only digits 0-9 before the separator";
  if (prompt[n] !== sep) return `the prompt must end with '${sep}'`;
  return null;
}

export const TASKS: Record<Task["id"], Task> = {
  reverse: {
    id: "reverse",
    title: "Reverse",
    promptLen: DIGITS + 1,
    answerLen: DIGITS,
    sample(rng) {
      const d = randDigits(rng, DIGITS);
      return { prompt: d + ">", answer: d.split("").reverse().join("") };
    },
    solve(prompt) {
      return prompt.slice(0, DIGITS).split("").reverse().join("");
    },
    validate: (p) => checkDigits(p, DIGITS, ">"),
    probe: "31415926>",
    insight:
      "A clean anti-diagonal: the k-th output token attends to input position (8 − k). The model has learned to compute a pointer from its own position.",
  },
  sort: {
    id: "sort",
    title: "Sort",
    promptLen: DIGITS + 1,
    answerLen: DIGITS,
    sample(rng) {
      const d = randDigits(rng, DIGITS);
      return { prompt: d + ">", answer: d.split("").sort().join("") };
    },
    solve(prompt) {
      return prompt.slice(0, DIGITS).split("").sort().join("");
    },
    validate: (p) => checkDigits(p, DIGITS, ">"),
    probe: "73019482>",
    insight:
      "Each output token attends to the input digit that comes next in sorted order — attention acts as a content-addressed lookup for 'the smallest digit larger than what I just wrote'.",
  },
  add: {
    id: "add",
    title: "Add",
    promptLen: 8,
    answerLen: 4,
    sample(rng) {
      const a = rng.int(1000), b = rng.int(1000);
      const prompt = `${String(a).padStart(3, "0")}+${String(b).padStart(3, "0")}=`;
      return { prompt, answer: TASKS.add.solve(prompt) };
    },
    solve(prompt) {
      const a = Number(prompt.slice(0, 3)), b = Number(prompt.slice(4, 7));
      // Answer is written least-significant digit first, the way you carry by hand.
      return String(a + b).padStart(4, "0").split("").reverse().join("");
    },
    validate(p) {
      if (!/^\d{3}\+\d{3}=$/.test(p)) return "format: 3 digits, '+', 3 digits, '=' (e.g. 047+385=)";
      return null;
    },
    probe: "479+386=",
    insight:
      "The sum is written right-to-left (ones digit first). Each output digit attends to the matching column of BOTH numbers, plus the previous column to work out the carry.",
  },
};
