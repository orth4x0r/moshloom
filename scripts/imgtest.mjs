// Golden-image comparator for binary P6 PPM files. No dependencies, plain ESM.
// Usage: node scripts/imgtest.mjs <dirA> <dirB> [--tol N] [--max-frac F] [--allow-missing]
import fs from 'node:fs';
import path from 'node:path';

const USAGE = `Usage: node scripts/imgtest.mjs <dirA> <dirB> [options]

Compare every *.ppm in <dirA> against the baseline of the same name in <dirB>.

Options:
  --tol N         max per-channel delta for a pixel to still count as equal (default 2)
  --max-frac F    max fraction of differing pixels allowed per image (default 0.002 = 0.2%)
  --allow-missing images in <dirA> with no baseline in <dirB> are skipped, not failed
  -h, --help      show this help

Exit codes: 0 all within tolerance, 1 mismatch / size mismatch / missing baseline,
2 usage error.`;

function die(msg) {
  console.error(`imgtest: ${msg}`);
  console.error(USAGE);
  process.exit(2);
}

// Header tokens are ASCII separated by arbitrary whitespace; pixel data follows
// the newline (or lone CR, space, tab) after maxval.
function parsePPM(buf, name) {
  const text = buf.toString('latin1', 0, Math.min(buf.length, 256));
  const head = /^P6\s+(\d+)\s+(\d+)\s+(\d+)(?:\r\n|\r|\n|[ \t])/.exec(text);
  if (!head) throw new Error(`${name}: not a P6 PPM or unparseable header`);
  const w = Number(head[1]), h = Number(head[2]), max = Number(head[3]);
  if (w <= 0 || h <= 0) throw new Error(`${name}: bad dimensions "${w}x${h}"`);
  if (max !== 255) throw new Error(`${name}: unsupported maxval ${max} (only 255)`);
  const need = w * h * 3;
  const start = head[0].length;
  if (buf.length - start < need) {
    throw new Error(`${name}: expected ${need} pixel bytes, got ${buf.length - start}`);
  }
  return { w, h, data: buf.subarray(start, start + need) };
}

function diffImages(a, b, tol) {
  let differing = 0, maxDelta = 0;
  for (let o = 0; o < a.data.length; o += 3) {
    let m = 0;
    for (let c = 0; c < 3; c++) {
      const d = Math.abs(a.data[o + c] - b.data[o + c]);
      if (d > m) m = d;
    }
    if (m > maxDelta) maxDelta = m;
    if (m > tol) differing++;
  }
  return { differing, maxDelta, frac: differing / (a.data.length / 3) };
}

const argv = process.argv.slice(2);
let tol = 2, maxFrac = 0.002, allowMissing = false;
const positional = [];

for (let i = 0; i < argv.length; i++) {
  const arg = argv[i];
  if (arg === '-h' || arg === '--help') { console.log(USAGE); process.exit(0); }
  else if (arg === '--allow-missing') allowMissing = true;
  else if (arg === '--tol' || arg === '--max-frac') {
    const n = Number(argv[++i]);
    if (!Number.isFinite(n) || n < 0) die(`${arg} needs a non-negative number`);
    if (arg === '--tol') tol = n; else maxFrac = n;
  } else if (arg.startsWith('-')) die(`unknown option ${arg}`);
  else positional.push(arg);
}
if (positional.length !== 2) die('need exactly two directories');
const [dirA, dirB] = positional.map(p => path.resolve(p));

function listPPM(dir, label) {
  if (!fs.existsSync(dir) || !fs.statSync(dir).isDirectory()) die(`${label} not found: ${dir}`);
  return fs.readdirSync(dir).filter(f => f.toLowerCase().endsWith('.ppm')).sort();
}

const names = listPPM(dirA, 'image directory');
const baselines = listPPM(dirB, 'baseline directory');
if (names.length === 0) die(`no .ppm files in ${dirA}`);

const rows = [];
let passed = 0, failed = 0, skipped = 0;
for (const name of names) {
  const row = { name, size: '-', differing: '-', frac: '-', maxDelta: '-' };
  if (!baselines.includes(name)) {
    if (allowMissing) { skipped++; row.result = 'skipped (no baseline)'; }
    else { failed++; row.result = 'FAIL missing baseline'; }
    rows.push(row);
    continue;
  }
  try {
    const a = parsePPM(fs.readFileSync(path.join(dirA, name)), name);
    const b = parsePPM(fs.readFileSync(path.join(dirB, name)), `baseline/${name}`);
    if (a.w !== b.w || a.h !== b.h) {
      failed++;
      row.result = `FAIL size ${a.w}x${a.h} vs ${b.w}x${b.h}`;
    } else {
      const d = diffImages(a, b, tol);
      const ok = d.frac <= maxFrac;
      if (ok) passed++; else failed++;
      Object.assign(row, {
        size: `${a.w}x${a.h}`, differing: String(d.differing),
        frac: `${(d.frac * 100).toFixed(3)}%`, maxDelta: String(d.maxDelta),
        result: ok ? 'ok' : 'FAIL over tolerance',
      });
    }
  } catch (e) {
    failed++;
    row.result = `FAIL ${e.message}`;
  }
  rows.push(row);
}

const width = Math.max(4, ...rows.map(r => r.name.length));
const head = `${'image'.padEnd(width)}  ${'size'.padEnd(9)}  ${'diff px'.padStart(9)}  ${'frac'.padStart(8)}  ${'maxΔ'.padStart(5)}  result`;
console.log(`tol=${tol}  max-frac=${maxFrac} (${(maxFrac * 100).toFixed(2)}%)`);
console.log(head);
console.log('-'.repeat(head.length));
for (const r of rows) {
  console.log(`${r.name.padEnd(width)}  ${r.size.padEnd(9)}  ${r.differing.padStart(9)}  ${r.frac.padStart(8)}  ${r.maxDelta.padStart(5)}  ${r.result}`);
}
console.log('-'.repeat(head.length));
console.log(`${rows.length} image(s): ${passed} within tolerance, ${failed} failed, ${skipped} skipped`);
process.exit(failed === 0 ? 0 : 1);
