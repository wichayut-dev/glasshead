import { TrainingHost, type ToHost } from "./host.ts";

// Runs the training loop off the main thread so the UI stays at 60fps while the model trains.
const ctx = self as unknown as { postMessage(m: unknown): void; onmessage: ((e: MessageEvent) => void) | null };
const host = new TrainingHost((m) => ctx.postMessage(m), (fn) => setTimeout(fn, 0), 150);
ctx.onmessage = (e: MessageEvent<ToHost>) => host.handle(e.data);
