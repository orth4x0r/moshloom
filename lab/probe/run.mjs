// P1 runner: opens lab/probe/index.html over http and over file, headless, and reports the
// three answers. Usage: node lab/probe/run.mjs
import { chromium } from 'playwright';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const root = path.resolve(import.meta.dirname);
const page = path.join(root, 'index.html');
const body = fs.readFileSync(page);

const server = http.createServer((req, res) => {
  res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
  res.end(body);
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const port = server.address().port;

const ARGS = [
  '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=swiftshader',
  '--use-webgpu-adapter=swiftshader', '--ignore-gpu-blocklist',
];

const targets = [
  { name: 'http://127.0.0.1', url: `http://127.0.0.1:${port}/` },
  { name: 'file://', url: pathToFileURL(page).href },
];

const browser = await chromium.launch({ args: ARGS });
let bad = 0;
for (const t of targets) {
  const p = await browser.newPage();
  const errs = [];
  p.on('pageerror', (e) => errs.push(e.message));
  await p.goto(t.url);
  await p.waitForFunction(() => document.getElementById('out').textContent.includes('verdict'), null, { timeout: 20000 })
    .catch(() => {});
  const text = await p.textContent('#out');
  const start = text.indexOf('{'), end = text.lastIndexOf('}');
  let r = null;
  try { r = JSON.parse(text.slice(start, end + 1)); } catch { /* leave null */ }
  const ok = !!(r && r.blobWorker && r.gpu && r.drew && r.lum > 0.02);
  if (!ok) bad++;
  console.log(`${t.name.padEnd(18)} blobWorker=${r ? r.blobWorker : '?'} gpu=${r ? r.gpu : '?'} drew=${r ? r.drew : '?'} lum=${r ? r.lum : '?'} ${ok ? 'PASS' : 'FAIL'}${r && r.err ? '  err: ' + r.err : ''}${errs.length ? '  pageerrors: ' + errs[0] : ''}`);
  await p.close();
}
await browser.close();
server.close();
console.log(bad ? '\nFAIL' : '\nPASS - blob workers and WebGPU-in-worker both work here');
process.exit(bad ? 1 : 0);
