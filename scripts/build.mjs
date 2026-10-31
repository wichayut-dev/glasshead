// Builds the whole lab into ONE self-contained HTML file (dist/index.html):
// the worker is bundled to a string and started from a Blob URL, and the pretrained
// weights are embedded, so the page works offline and from any static host.
import { build } from "esbuild";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";

const bundle = async (entry) => {
  const r = await build({ entryPoints: [entry], bundle: true, format: "iife", minify: true, write: false, target: "es2020" });
  return r.outputFiles[0].text;
};

const worker = await bundle("src/app/worker.ts");
const main = await bundle("src/app/main.ts");

const pretrained = {};
for (const t of ["reverse", "sort", "add"]) {
  const f = `pretrained/${t}.json`;
  if (existsSync(f)) pretrained[t] = JSON.parse(readFileSync(f, "utf8"));
  else console.warn(`warning: ${f} missing (run: node scripts/train.ts ${t} --save)`);
}

// Never let embedded text close the <script> element early.
const safe = (s) => s.replace(/<\/(script)/gi, "<\\/$1");
const template = readFileSync("src/app/index.html", "utf8");
const html =
  template +
  `\n<script>const __WORKER_SRC__=${safe(JSON.stringify(worker))};const __PRETRAINED__=${safe(JSON.stringify(pretrained))};</script>` +
  `\n<script>${safe(main)}</script>\n`;

mkdirSync("dist", { recursive: true });
writeFileSync("dist/index.html", html);
console.log(`dist/index.html  ${(html.length / 1024).toFixed(0)} KB  (worker ${(worker.length / 1024).toFixed(0)} KB, app ${(main.length / 1024).toFixed(0)} KB)`);
