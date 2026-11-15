// Headless-Chrome WebGPU harness. Renders through the offscreen hook and reads pixels back.
// Usage: node gpu_test.mjs
// Requires: npm install && npx playwright install chromium, and index.html built (npm run pack).
// Exits non-zero on any WebGPU validation error, page error, or a black render.
import { chromium } from 'playwright';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { spawn, spawnSync } from 'node:child_process';
import crypto from 'node:crypto';

const HARDWARE = process.argv.includes('--hardware');

// ---- GPU run lock (H3) -------------------------------------------------------
// AGENTS.md rule 6: one headed-Chrome run at a time on this machine, and nothing alongside it.
// A brief that says "do not" was already ignored once, so the harness enforces it:
//   --hardware                     takes the EXCLUSIVE gpu.lock
//   anything else (npm test,       takes its own sw.<pid>.lock; two of these may overlap
//     --record, --no-golden)
// Both are created atomically ('wx') in one directory shared by every checkout and worktree of
// this user, so a hardware run on the primary checkout stops a SwiftShader run in any worktree.
// A run that would break the rule prints one `gpu lock: refused ...` line and exits 3 before any
// browser is launched. `--lock-self-test` exercises all of it with no browser at all.
const LOCK_REFUSED = 3;
const GPU_LOCK = 'gpu.lock';
const swMarker = (pid) => `sw.${pid}.lock`;
const lockDirFor = () => process.env.MOSHLOOM_LOCK_DIR || path.join(os.homedir(), '.moshloom');

// `process.kill(pid, 0)` in try/catch. On Windows EPERM means the process exists but is not ours.
function pidAlive(pid) {
  if (!Number.isInteger(pid) || pid <= 0) return false;
  try { process.kill(pid, 0); return true; }
  catch (e) { return e.code === 'EPERM'; }
}

function readRecord(file) {
  try {
    // Strip a BOM: a lock written by another tool is still a lock, and an unreadable pid reads
    // as 0, which would let a live run be swept as stale.
    const j = JSON.parse(fs.readFileSync(file, 'utf8').replace(/^\uFEFF/, ''));
    if (j && Number.isInteger(j.pid)) return j;
  } catch (e) {}
  return null;
}

const lockNames = (dir) => {
  try { return fs.readdirSync(dir).filter((n) => n === GPU_LOCK || /^sw\.\d+\.lock$/.test(n)); }
  catch (e) { return []; }
};

// openSync('wx') creates the file empty and the record is written a moment later. A lock with no
// readable record that is this young is a run being born, not a stale lock: deleting it there would
// let two runs both pass their checks.
const YOUNG_MS = 10000;
function beingWritten(file) {
  try { return Date.now() - fs.statSync(file).mtimeMs < YOUNG_MS; } catch (e) { return false; }
}

// Stale = the recorded pid is not running. A live lock is never removed: it is exactly the run
// the rule protects. A record with no readable pid cannot be attributed, so it counts as stale
// once it is older than YOUNG_MS.
function removeStaleLocks(dir) {
  for (const n of lockNames(dir)) {
    const file = path.join(dir, n);
    const rec = readRecord(file);
    if (!rec && beingWritten(file)) continue;
    const pid = rec ? rec.pid : 0;
    if (pidAlive(pid)) continue;
    try {
      fs.unlinkSync(file);
      console.log(`gpu lock: removed stale ${file} (pid ${pid})`);
    } catch (e) {}
  }
}

// A SwiftShader run is blocked by a live gpu.lock and by nothing else (two may overlap).
// A hardware run is blocked by a live gpu.lock or by any live sw marker.
function liveConflicts(dir, hardware) {
  const out = [];
  for (const n of lockNames(dir)) {
    const file = path.join(dir, n);
    const rec = readRecord(file);
    if (!rec) {
      // a run that has created its lock but not yet written it: treat it as live
      if (beingWritten(file) && (n === GPU_LOCK || hardware)) out.push({ file, pid: 0, start: 'being written', cwd: 'unknown' });
      continue;
    }
    if (!pidAlive(rec.pid)) continue;
    if (n === GPU_LOCK || hardware) out.push({ file, ...rec });
  }
  return out;
}

const holderText = (h) => `pid ${h.pid} in ${h.file} (started ${h.start || 'unknown start'}, cwd ${h.cwd || 'unknown cwd'})`;

/** Take the lock for a run of this kind: {ok:true, file} or {ok:false, holders|why}. */
function acquireRunLock({ dir, hardware, argv = process.argv }) {
  try { fs.mkdirSync(dir, { recursive: true }); }
  catch (e) { return { ok: false, why: `cannot create lock dir ${dir}: ${e.message}` }; }
  removeStaleLocks(dir);
  const clash = liveConflicts(dir, hardware);
  if (clash.length) return { ok: false, holders: clash };
  const file = path.join(dir, hardware ? GPU_LOCK : swMarker(process.pid));
  const record = JSON.stringify({ pid: process.pid, start: new Date().toISOString(), cwd: process.cwd(),
    argv: argv.slice(2), kind: hardware ? 'hardware' : 'swiftshader' }, null, 1);
  let fd;
  try { fd = fs.openSync(file, 'wx'); }          // atomic: throws EEXIST if the file is already there
  catch (e) {
    if (e.code === 'EEXIST') {
      const rec = readRecord(file);
      return { ok: false, holders: [{ file, pid: rec ? rec.pid : 0, start: rec && rec.start, cwd: rec && rec.cwd }] };
    }
    return { ok: false, why: `cannot create ${file}: ${e.message}` };
  }
  try { fs.writeSync(fd, record); } finally { fs.closeSync(fd); }
  // Close the check-then-create window: re-read the other side now that our file exists, and back
  // off if a run of the opposite kind got in between. Whichever side sees the other first wins.
  const after = liveConflicts(dir, hardware).filter((h) => h.pid !== process.pid);
  if (after.length) {
    try { fs.unlinkSync(file); } catch (e) {}
    return { ok: false, holders: after };
  }
  return { ok: true, file };
}

// Release on every exit path. `exit` fires for a normal return, process.exit(), an uncaught throw
// and a signal handler that exits, so one synchronous unlink handler covers all of them.
const heldLocks = new Set();
let releaseInstalled = false;
function installReleaseHandlers() {
  if (releaseInstalled) return;
  releaseInstalled = true;
  process.on('exit', () => {
    for (const file of heldLocks) { try { fs.unlinkSync(file); } catch (e) {} }
    heldLocks.clear();
  });
  // Windows delivers Ctrl+C as SIGINT; SIGTERM is raised by a parent or a test. Both exit, which
  // runs the handler above. SIGBREAK is Ctrl+Break on Windows.
  for (const [sig, code] of [['SIGINT', 130], ['SIGTERM', 143]]) process.on(sig, () => process.exit(code));
  if (process.platform === 'win32') process.on('SIGBREAK', () => process.exit(131));
}

/** Acquire or refuse, before any browser exists. */
function takeRunLock(hardware) {
  const lock = acquireRunLock({ dir: lockDirFor(), hardware });
  if (!lock.ok) {
    const ms = Math.round(process.uptime() * 1000);      // process start -> refusal line
    const who = lock.holders ? lock.holders.map(holderText).join('; ') : lock.why;
    console.log(`gpu lock: refused after ${ms} ms for a ${hardware ? 'hardware' : 'SwiftShader'} run: ${who}`
      + ' - that run owns the GPU; wait for it to finish (a lock whose pid is dead is removed automatically).');
    process.exit(LOCK_REFUSED);
  }
  heldLocks.add(lock.file);
  installReleaseHandlers();
  return lock.file;
}

const SELF_URL = import.meta.url;

// Used only by --lock-self-test: a real second process takes a lock and leaves by a chosen path,
// so the release handler is exercised for real instead of simulated in-process.
async function lockChildMode(mode) {
  const dir = lockDirFor();
  const lock = acquireRunLock({ dir, hardware: mode === 'hardware' });
  if (!lock.ok) {
    console.log(`child ${process.pid}: refused ${lock.holders ? lock.holders.map(holderText).join('; ') : lock.why}`);
    process.exit(LOCK_REFUSED);
  }
  heldLocks.add(lock.file);
  installReleaseHandlers();
  console.log(`child ${process.pid}: holding ${lock.file}`);
  if (mode === 'throw') throw new Error('self-test: the release handler must survive a thrown error');
  if (mode === 'hold') {
    // Concurrency case: rendezvous with the other SwiftShader run. The ready file is written only
    // while the marker is held, so seeing the peer's ready file proves both markers coexisted.
    const ready = (marker) => marker + '.ready';
    fs.writeFileSync(ready(lock.file), String(process.pid));
    const mine = path.basename(ready(lock.file));
    const deadline = Date.now() + 20000;
    let peer = 0;
    while (Date.now() < deadline && !peer) {
      let names = [];
      try { names = fs.readdirSync(dir); } catch (e) {}
      for (const n of names) {
        if (n === mine || !/^sw\.\d+\.lock\.ready$/.test(n)) continue;
        try { peer = Number(fs.readFileSync(path.join(dir, n), 'utf8')) || 1; } catch (e) {}
      }
      if (!peer) await new Promise((r) => setTimeout(r, 50));
    }
    console.log(`child ${process.pid}: holding ${path.basename(lock.file)},`
      + ` peer marker ${peer ? 'sw.' + peer + '.lock seen at the same time' : 'NEVER SEEN'}`);
    process.exit(peer ? 0 : 1);
  }
  process.exit(0);                                   // the normal-exit path
}

/** --lock-self-test: every case here, no browser, exit 0/1. */
async function runLockSelfTest() {
  const started = Date.now();
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'moshloom-lock-'));
  let pass = 0, fail = 0;
  const ok = (name, cond, detail = '') => {
    if (cond) pass++; else fail++;
    console.log(`${cond ? 'PASS' : 'FAIL'} ${name}${detail ? ' :: ' + detail : ''}`);
  };
  const dir = (tag) => fs.mkdtempSync(path.join(tmp, tag + '-'));
  const writeFake = (file, pid) => fs.writeFileSync(file,
    JSON.stringify({ pid, start: '2026-01-01T00:00:00.000Z', cwd: tmp, argv: ['--fake'] }));
  // A recorded console so the stale-removal lines can be asserted instead of just printed.
  const quiet = (fn) => {
    const real = console.log, lines = [];
    console.log = (...a) => lines.push(a.map(String).join(' '));
    try { return { value: fn(), lines }; } finally { console.log = real; }
  };
  // A pid that is provably dead: a child that has already exited and been reaped.
  const deadPid = () => {
    const r = spawnSync(process.execPath, ['-e', '0'], {});
    if (r.pid && !pidAlive(r.pid)) return r.pid;
    for (let p = 65000; p < 66000; p++) if (!pidAlive(p)) return p;
    return 999999;
  };
  const envFor = (d, mode) => ({ ...process.env, MOSHLOOM_LOCK_DIR: d, MOSHLOOM_LOCK_CHILD: mode });
  const childCode = `await import(${JSON.stringify(SELF_URL)})`;
  const childSync = (d, mode) => spawnSync(process.execPath, ['--input-type=module', '-e', childCode],
    { env: envFor(d, mode), encoding: 'utf8', timeout: 60000 });

  try {
    // 1. a live hardware lock blocks a SwiftShader run
    {
      const d = dir('hw-blocks-sw');
      const g = path.join(d, GPU_LOCK);
      writeFake(g, process.pid);
      const { value } = quiet(() => acquireRunLock({ dir: d, hardware: false }));
      ok('a live hardware lock blocks a SwiftShader run',
        !value.ok && !!value.holders && value.holders[0].pid === process.pid && fs.existsSync(g),
        `refused=${!value.ok}, holder pid ${value.holders && value.holders[0].pid}, gpu.lock kept=${fs.existsSync(g)}`);
      ok('the refused SwiftShader run left no marker of its own',
        !fs.existsSync(path.join(d, swMarker(process.pid))));
    }

    // 2. a live SwiftShader marker blocks a hardware run
    {
      const d = dir('sw-blocks-hw');
      const m = path.join(d, swMarker(process.pid));
      writeFake(m, process.pid);
      const { value } = quiet(() => acquireRunLock({ dir: d, hardware: true }));
      const g = path.join(d, GPU_LOCK);
      ok('a live SwiftShader marker blocks a hardware run',
        !value.ok && value.holders.some((h) => h.file === m) && !fs.existsSync(g),
        `refused=${!value.ok}, holder ${path.basename(m)}, gpu.lock created=${fs.existsSync(g)}`);
    }

    // 3. stale locks of both kinds are removed, and the run continues
    {
      const d = dir('stale');
      const dead = deadPid();
      const g = path.join(d, GPU_LOCK), s = path.join(d, swMarker(dead));
      writeFake(g, dead); writeFake(s, dead);
      const { value, lines } = quiet(() => acquireRunLock({ dir: d, hardware: false }));
      const format = lines.every((l) => new RegExp(`^gpu lock: removed stale .* \\(pid ${dead}\\)$`).test(l));
      ok('stale locks of both kinds are removed and the run continues',
        value.ok && lines.length === 2 && format && !fs.existsSync(g) && !fs.existsSync(s),
        `${lines.length} stale line(s), both in "gpu lock: removed stale <file> (pid ${dead})" form=${format}`);
      for (const l of lines) console.log('  ' + l);
      if (value.ok) fs.unlinkSync(value.file);
    }

    // 3b. a hardware run after a stale gpu.lock takes the exclusive lock (the acquire half of it)
    {
      const d = dir('stale-hw');
      const dead = deadPid();
      writeFake(path.join(d, GPU_LOCK), dead);
      const { value, lines } = quiet(() => acquireRunLock({ dir: d, hardware: true }));
      const body = value.ok ? JSON.parse(fs.readFileSync(value.file, 'utf8')) : {};
      ok('a stale gpu.lock is removed and a hardware run then takes it exclusively',
        value.ok && path.basename(value.file) === GPU_LOCK && body.pid === process.pid
          && typeof body.start === 'string' && Array.isArray(body.argv),
        `lock=${value.ok ? path.basename(value.file) : 'none'}, record pid ${body.pid}, cwd ${body.cwd}`);
      for (const l of lines) console.log('  ' + l);
      if (value.ok) fs.unlinkSync(value.file);
    }

    // 4. own marker released after a normal exit, and after a thrown error (real child processes)
    {
      const d = dir('exit');
      const r = childSync(d, 'sw');
      const file = r.pid ? path.join(d, swMarker(r.pid)) : '';
      ok('the marker is released after a normal exit',
        r.status === 0 && /child \d+: holding/.test(r.stdout || '') && !!file && !fs.existsSync(file),
        `child pid ${r.pid} exit ${r.status}, marker gone=${!!file && !fs.existsSync(file)}, child said "${(r.stdout || '').trim()}"`);
    }
    {
      const d = dir('throw');
      const r = childSync(d, 'throw');
      const file = r.pid ? path.join(d, swMarker(r.pid)) : '';
      ok('the marker is released after a thrown error',
        r.status !== 0 && /child \d+: holding/.test(r.stdout || '') && !!file && !fs.existsSync(file),
        `child pid ${r.pid} exit ${r.status} (non-zero expected), marker gone=${!!file && !fs.existsSync(file)}`);
    }
    {
      const d = dir('exit-hw');
      const r = childSync(d, 'hardware');
      const g = path.join(d, GPU_LOCK);
      ok('the exclusive gpu.lock is released after a normal exit',
        r.status === 0 && /child \d+: holding/.test(r.stdout || '') && !fs.existsSync(g),
        `child pid ${r.pid} exit ${r.status}, gpu.lock gone=${!fs.existsSync(g)}`);
    }

    // 5. two concurrent SwiftShader runs do not block each other
    {
      const d = dir('concurrent');
      const a = spawn(process.execPath, ['--input-type=module', '-e', childCode],
        { env: envFor(d, 'hold'), stdio: ['ignore', 'pipe', 'pipe'] });
      const b = spawn(process.execPath, ['--input-type=module', '-e', childCode],
        { env: envFor(d, 'hold'), stdio: ['ignore', 'pipe', 'pipe'] });
      let outA = '', outB = '';
      a.stdout.on('data', (c) => { outA += c; });
      b.stdout.on('data', (c) => { outB += c; });
      const codeOf = (p) => new Promise((res) => p.on('close', res));
      const [ca, cb] = await Promise.all([codeOf(a), codeOf(b)]);
      ok('two concurrent SwiftShader runs do not block each other',
        ca === 0 && cb === 0 && /peer marker sw\.\d+\.lock seen at the same time/.test(outA)
          && /peer marker sw\.\d+\.lock seen at the same time/.test(outB) && lockNames(d).length === 0,
        `exits ${ca}/${cb}; "${outA.trim().split('\n').pop()}" | "${outB.trim().split('\n').pop()}"`);
    }
    // 6. a gpu.lock that exists but is not written yet is a run being born, not a stale lock
    {
      const d = dir('young');
      const g = path.join(d, GPU_LOCK);
      fs.closeSync(fs.openSync(g, 'wx'));
      const { value, lines } = quiet(() => acquireRunLock({ dir: d, hardware: false }));
      ok('an empty, just-created gpu.lock blocks a SwiftShader run and is not swept',
        !value.ok && fs.existsSync(g) && lines.length === 0,
        `refused=${!value.ok}, gpu.lock kept=${fs.existsSync(g)}, stale lines ${lines.length}`);
    }
  } finally {
    try { fs.rmSync(tmp, { recursive: true, force: true }); } catch (e) {}
  }
  console.log(`lock self-test:${pass} passed, ${fail} failed, ${Date.now() - started} ms, no browser launched`);
  process.exit(fail ? 1 : 0);
}

if (process.env.MOSHLOOM_LOCK_CHILD) await lockChildMode(process.env.MOSHLOOM_LOCK_CHILD);
if (process.argv.includes('--lock-self-test')) await runLockSelfTest();
takeRunLock(HARDWARE);
// ---- end GPU run lock -------------------------------------------------------

const root = import.meta.dirname;
const html = path.join(root, 'index.html');
if (!fs.existsSync(html)) { console.error('index.html is missing - run: npm run pack'); process.exit(1); }

const fixture = (name) => {
  const p = path.join(root, 'fixtures', name);
  return fs.existsSync(p) ? p : null;
};
const clipImg = fixture('clip.png');
const clipVid = fixture('clip.mp4') || fixture('clip.webm');
if (!clipImg || !clipVid) { console.error('fixtures/clip.png and a clip video are required'); process.exit(1); }

const outDir = path.join(root, 'out');
fs.mkdirSync(outDir, { recursive: true });

// ---- export decode: one separate headless Chromium (H4) ----------------------
// Every export check used to decode its own file in the page that produced it: a <video>, a seek
// to 0.5 s, drawImage onto a 160x90 canvas, luminance and an FNV hash. In `--hardware` mode that
// page is a headed window, and Chrome does not reliably produce decoded video frames for a window
// that is covered or in the background, so the same file decoded lit in one run and black in the
// next three with no code change. The check must not depend on what is on screen.
//
// So the exporting page now hands the file's bytes (base64) to Node and never decodes anything,
// and Node decodes in ONE separate headless Chromium with no GPU: launched lazily on the first
// file, one page reused for all of them, at a file:// URL inside this repo. The computation is
// the same video -> seek 0.5 s -> 160x90 -> luminance / FNV hash, so the numbers stay comparable
// with the ones already recorded. It is part of this run and inside this run's GPU lock; it is
// closed on the normal path and by an exit handler on every other one.
//
// `--use-angle=swiftshader` is the one flag and it is not decoration: drawImage(video) rasterizes
// the frame, and the default backend rounds YUV->RGB differently from the software one the
// exporting page ran under `npm test`. The same bytes decoded by both, measured on the H4 branch:
//
//   decode backend                     worker export      media clip frame
//   default (machine GPU)              0.109              630426149, lum 0.022
//   --use-angle=swiftshader            0.111              1500507020, lum 0.023   <- the recorded values
//   channel:'chromium', new headless   0 (black frame)    1620850465
//
// The recorded values are 0.111 and 1500507020, so the decode browser takes the software raster:
// one decoder on every machine, no adapter in the answer, and the pinned numbers keep their
// meaning. (New headless with a full Chromium build is not an option either: it drew black.)
const SWIFTSHADER_RASTER = ['--use-angle=swiftshader'];
const decodeHtml = path.join(outDir, 'decode.html');
let decodeBrowser = null, decodeP = null;

// Playwright serializes this function and runs it in the decode page, so it may reference nothing
// outside its own argument. Same steps, same numbers as the inline decodes it replaces.
const decodeInPage = async ({ b64, mime }) => {
  const bin = atob(b64);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  const blob = new Blob([u8], mime ? { type: mime } : undefined);
  const hashOf = (a) => { let x = 2166136261 >>> 0; for (let i = 0; i < a.length; i++) x = Math.imul(x ^ a[i], 16777619) >>> 0; return x >>> 0; };
  const url = URL.createObjectURL(blob);
  const v = document.createElement('video'); v.muted = true; v.src = url; v.load();
  await new Promise((res) => { v.onloadeddata = res; v.onerror = res; setTimeout(res, 4000); });
  v.currentTime = 0.5;
  await new Promise((r) => { v.onseeked = r; setTimeout(r, 800); });
  const c = document.createElement('canvas'); c.width = 160; c.height = 90;
  const g = c.getContext('2d'); g.drawImage(v, 0, 0, 160, 90);
  const d = g.getImageData(0, 0, 160, 90).data;
  let sum = 0; for (let i = 0; i < d.length; i += 4) sum += d[i] + d[i + 1] + d[i + 2];
  const out = { hash: hashOf(d), lum: +(sum / (160 * 90 * 3 * 255)).toFixed(3),
    w: v.videoWidth, h: v.videoHeight, dur: +(v.duration || 0).toFixed(2) };
  URL.revokeObjectURL(url); v.remove(); c.remove();
  return out;
};

async function openDecodePage() {
  if (decodeP) return decodeP;
  fs.writeFileSync(decodeHtml, '<!doctype html><meta charset="utf-8"><title>moshloom export decode</title>\n');
  decodeBrowser = await chromium.launch({ headless: true, args: SWIFTSHADER_RASTER });
  // Belt and braces for the "close it on every exit path" rule: the normal path awaits
  // closeDecodePage(), and a throw, process.exit() or a signal still takes the browser down here.
  process.on('exit', () => { try { if (decodeBrowser) decodeBrowser.close(); } catch (e) {} });
  const pg = await decodeBrowser.newPage();
  await pg.goto(pathToFileURL(decodeHtml).href);
  decodeP = pg;
  console.log(`decode: export files are decoded in a separate headless Chromium, never in the exporting page (${path.relative(process.cwd(), decodeHtml) || decodeHtml})`);
  return decodeP;
}

// A decode that cannot run must not take the suite down with a stack: the file then reads as
// undecodable and the existing assertion for that check fails with its existing message.
async function decodeVideoFile(b64, mime, label) {
  try {
    const pg = await openDecodePage();
    return await pg.evaluate(decodeInPage, { b64, mime: mime || '' });
  } catch (e) {
    console.log(`decode: ${label} could not be decoded in the headless page (${e.message.split('\n')[0]})`);
    return { hash: 0, lum: 0, w: 0, h: 0, dur: 0 };
  }
}

async function closeDecodePage() {
  if (!decodeBrowser) return;
  const b = decodeBrowser;
  decodeBrowser = null; decodeP = null;
  try { await b.close(); } catch (e) {}
}

// HARDWARE is read at the top of the file, before the lock is taken.
// The committed goldens are SwiftShader references. A real adapter renders the same scenes with
// its own precision/colour path, so hardware mode checks rendering and export but skips that
// comparison unless a hardware baseline is requested explicitly.
const CHECK_GOLDENS = !process.argv.includes('--no-golden')
  && !(HARDWARE && !process.argv.includes('--hardware-golden'));
const SWIFTSHADER_ARGS = [
  '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=swiftshader',
  '--use-webgpu-adapter=swiftshader', '--ignore-gpu-blocklist',
  '--autoplay-policy=no-user-gesture-required',
];
const HARDWARE_ARGS = [
  '--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--autoplay-policy=no-user-gesture-required',
  '--disable-backgrounding-occluded-windows', '--disable-renderer-backgrounding',
  '--disable-background-timer-throttling', '--disable-features=CalculateNativeWinOcclusion',
];

async function launch() {
  if (HARDWARE) {
    try { return await chromium.launch({ channel: 'chrome', headless: false, args: HARDWARE_ARGS }); }
    catch (e) {
      try { return await chromium.launch({ headless: false, args: HARDWARE_ARGS }); }
      catch (e2) { throw new Error('--hardware needs a headed browser with a real WebGPU adapter.\n' + e.message + '\n' + e2.message); }
    }
  }
  try {
    return await chromium.launch({ args: SWIFTSHADER_ARGS });
  } catch (e) {
    try { return await chromium.launch({ channel: 'chrome', args: SWIFTSHADER_ARGS }); }
    catch { throw new Error('could not launch Chromium. Run: npx playwright install chromium\n' + e.message); }
  }
}

const browser = await launch();
if (HARDWARE) console.log('hardware run: headed Chrome, real GPU; export must produce a file'
  + (CHECK_GOLDENS ? '' : '; SwiftShader goldens skipped (pass --hardware-golden to compare)'));
const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
const errs = [];
const skips = [];
// Chrome on file:// logs this while a test decodes a blob URL into a <video>; the frame still decodes.
const benignConsole = (t) => /Not allowed to load local resource: blob:/.test(t);
page.on('pageerror', (e) => errs.push('PAGEERR ' + e.message));
page.on('console', (m) => { if (m.type() === 'error' && !benignConsole(m.text())) errs.push('CONSOLE ' + m.text().slice(0, 500)); });

// the offscreen hook: headless SwiftShader loses the device when presenting to a canvas
await page.addInitScript(() => { window.__MOSH_OFFSCREEN = true; });
await page.goto(pathToFileURL(html).href);
await page.waitForFunction(() => {
  const m = document.querySelector('#mode');
  return m && m.textContent && m.textContent !== '.';
}, null, { timeout: 30000 }).catch(() => {});
await page.waitForTimeout(1500);

await page.waitForFunction(() => window.__featReady === true, null, { timeout: 30000 }).catch(() => {});   // analysis worker round trip
const modeLabel = (await page.textContent('#mode'))?.trim();
const modeNow = await page.evaluate(() => (typeof mode === 'string' ? mode : null));
const err = (await page.textContent('#err'))?.trim();
console.log(`mode: ${modeNow} (${modeLabel}) | err: ${err || 'none'}`);
if (modeNow !== 'gpu') {
  console.log('NOTE: not on the WebGPU path; the GPU suite has nothing to test.');
  console.log(errs.slice(0, 6).join('\n') || 'no page errors');
  await browser.close();
  process.exit(1);
}

// Each section starts from the same tempo and a known project. The suite reuses one page, and
// without this a tempo left behind by Auto-arrange changes every later plan and export.
const resetSection = async () => {
  await page.evaluate(() => {
    pause();
    bpm = 120; offs = 0; X.set_grid(bpm, offs);
    const b = $('bpm'); if (b) b.value = bpm.toFixed(2);
    blocks = [newBlock(0, { len: 256, mosh: 0, glitch: 0, var: 1 })];
    fx = []; sel = 0; fsel = 0; selK = 'blk';
    push();
  });
};

// Thumbnails are asynchronous: drawTimeline queues thumbsSoon (350 ms) and genThumbs then renders
// one thumbnail per block, in the page in main mode and in the render worker in worker mode. A
// fixed wait is a guess in both directions, so wait for the real condition: every #track .blk has
// a background image. On timeout the counts are reported and the run fails; it never passes on a
// timeline that is still empty.
const THUMB_TIMEOUT_MS = 60000;
async function waitForThumbs(pg, label) {
  const counts = () => pg.evaluate(() => ({
    cached: thumbs.size,
    drawn: [...document.querySelectorAll('#track .blk')].filter((d) => d.style.backgroundImage).length,
    blocks: blocks.length,
  }));
  const t0 = Date.now();
  let timedOut = false;
  try {
    await pg.waitForFunction(() => {
      const els = [...document.querySelectorAll('#track .blk')];
      return els.length > 0 && els.length === blocks.length && els.every((d) => d.style.backgroundImage);
    }, null, { timeout: THUMB_TIMEOUT_MS, polling: 100 });
  } catch (e) { timedOut = true; }
  const info = await counts();
  const ms = Date.now() - t0;
  // A separate line, so the pinned `thumbs: ...` line below keeps its exact format.
  console.log(`${label} wait: ${ms} ms for ${info.drawn}/${info.blocks} block thumbnails (budget ${THUMB_TIMEOUT_MS} ms)`);
  return { ok: !timedOut, info, ms };
}

const shots = [];
async function grab(name) {
  const px = await page.evaluate(async () => {
    if (typeof G === 'undefined' || !G.off) return null;
    const d = G.dev, w = G.w, h = G.h;
    const buf = d.createBuffer({ size: w * h * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const e = d.createCommandEncoder();
    e.copyTextureToBuffer({ texture: G.off }, { buffer: buf, bytesPerRow: w * 4 }, [w, h]);
    d.queue.submit([e.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const a = new Uint8Array(buf.getMappedRange());
    let s = '';
    const CH = 0x8000;
    for (let i = 0; i < a.length; i += CH) s += String.fromCharCode.apply(null, a.subarray(i, i + CH));
    buf.unmap(); buf.destroy();
    return { w, h, fmt: G.fmt, b64: btoa(s) };
  });
  if (!px) { errs.push(`grab ${name}: no G.off`); return; }
  const raw = Buffer.from(px.b64, 'base64');
  const n = px.w * px.h;
  const rgb = Buffer.alloc(n * 3);
  const bgr = px.fmt.startsWith('bgra');
  let lit = 0;
  for (let i = 0; i < n; i++) {
    const r = raw[i * 4 + (bgr ? 2 : 0)], g = raw[i * 4 + 1], b = raw[i * 4 + (bgr ? 0 : 2)];
    rgb[i * 3] = r; rgb[i * 3 + 1] = g; rgb[i * 3 + 2] = b;
    if (r | g | b) lit++;
  }
  const file = path.join(outDir, `${name}.ppm`);
  fs.writeFileSync(file, Buffer.concat([Buffer.from(`P6 ${px.w} ${px.h} 255\n`), rgb]));
  const frac = lit / n;
  shots.push({ name, frac });
  if (frac < 0.02) errs.push(`${name}: image is ${(frac * 100).toFixed(1)}% lit (blank)`);
  console.log(`  ${name.padEnd(28)} ${(frac * 100).toFixed(1)}% lit`);
}

// ---- media ----
await resetSection();
await page.selectOption('#res', '640');
await page.setInputFiles('#mfile', clipImg);
await page.waitForFunction(() => /added|Could not/.test($('status').textContent), null, { timeout: 20000 }).catch(() => {});
console.log('media image:', (await page.textContent('#status'))?.trim());
await page.setInputFiles('#mfile', clipVid);
await page.waitForTimeout(1500);
console.log('media video:', (await page.textContent('#status'))?.trim(), '|', await page.evaluate(() => G.media.map((m) => `${m.kind} ${m.w}x${m.h}`).join(', ')));

// ---- custom shader: template compiles, broken code reports an error ----
await resetSection();
await page.click('#bShd');
await page.fill('#sName', 'Rings');
await page.click('#sSave');
await page.waitForTimeout(1200);
console.log('shaders:', await page.evaluate(() => shaders.map((s) => `${s.name}:${s.ok ? 'ok' : 'ERR'}`).join(',')));
await page.click('#sNew');
await page.fill('#sCode', '@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {\n  let q = XY(pos);\n  return fin(vec3f(q.x, nope, 0.));\n}');
await page.click('#sSave');
await page.waitForTimeout(1200);
const serr = (await page.textContent('#serr'))?.trim() || '';
console.log('bad shader ->', serr.slice(0, 120));
if (!/line|error|unknown|undeclared/i.test(serr)) errs.push('broken shader compiled without an error message');
await page.click('#sClose');

// ---- scenes and effects ----
await resetSection();
const shot = async (name, sc, extra, fxs) => {
  await page.evaluate(([sc, extra, fxs]) => {
    blocks = [newBlock(sc, Object.assign({ len: 256, mosh: 0, glitch: 0, var: 1, rgb: 0.3 }, extra))];
    fx = fxs; push(); pos = 20; renderAt(600, 12);
  }, [sc, extra || {}, fxs || []]);
  await grab(name);
};

await shot('scene_media_image', 16, { clip: 0 });
await shot('scene_media_video', 16, { clip: 1, pal: 2 });
await shot('scene_shader', 17, {});
for (let t = 0; t < 8; t++) {
  await shot(`fx_${t}`, [9, 16, 7, 7, 0, 10, 15, 12][t], { clip: 0, mosh: t === 2 || t === 3 || t === 6 ? 0.7 : 0 },
    [{ type: t, start: 0, len: 999, amt: 0.9, par: t === 0 ? 0.4 : 0.6 }]);
}
await shot('mods_bloom', 10, { mods: [[6, 0, 0.5], [1, 4, 0.5], [0, 2, 0]], bloom: 0.6, mosh: 0.6 });

// ---- fluid (effect type 8): renders, is deterministic from a seek, and costs ----
await resetSection();
const fluid = await page.evaluate(async (hardware) => {
  const hashOffset = async () => {
    const d = G.dev, w = G.w, h = G.h;
    const buf = d.createBuffer({ size: w * h * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const e = d.createCommandEncoder();
    e.copyTextureToBuffer({ texture: G.off }, { buffer: buf, bytesPerRow: w * 4 }, [w, h]);
    d.queue.submit([e.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const a = new Uint8Array(buf.getMappedRange().slice(0));
    buf.unmap(); buf.destroy();
    let x = 2166136261 >>> 0;
    for (let i = 0; i < a.length; i++) x = Math.imul(x ^ a[i], 16777619) >>> 0;
    return x;
  };
  const run = async (amt) => {
    blocks = [newBlock(0, { len: 256, mosh: 0.9, glitch: 0, var: 1, rgb: 0.3 })];
    fx = amt > 0 ? [{ type: 8, start: 0, len: 999, amt, par: 0.4 }] : [];
    push(); gpuReset(); X.reset();
    const d = G.dev;
    for (let i = 0; i < 12; i++) gpuFrame(588 + i);        // preroll: the mosh needs a reference frame
    await d.queue.onSubmittedWorkDone();
    const t = performance.now();
    for (let i = 0; i < 60; i++) gpuFrame(600 + i);
    await d.queue.onSubmittedWorkDone();
    return { ms: (performance.now() - t) / 60, hash: await hashOffset() };
  };
  const errsBefore = G.errs.length;
  // Always run one warm-up pass. In a headed hardware window, wall-clock waits are paced by
  // presentation, so the off/on delta is noise; keep it only for the software run.
  const warm = await run(0);
  const off = hardware ? null : warm;
  const on1 = await run(0.9), on2 = await run(0.9);
  return { offMs: off ? off.ms : null, onMs: on1.ms, delta: off ? on1.ms - off.ms : null,
    same: on1.hash === on2.hash, h1: on1.hash >>> 0, h2: on2.hash >>> 0, newErrs: G.errs.length - errsBefore };
}, HARDWARE);
if (HARDWARE) {
  console.log(`fluid (hardware): 60 frames from a seek twice: ${fluid.same ? 'identical' : 'DIFFERENT'}; ` +
    'wall-clock off/on comparison skipped in headed mode (see profile)');
} else {
  console.log(`fluid (SwiftShader, software): ${fluid.offMs.toFixed(2)} ms/frame off, ${fluid.onMs.toFixed(2)} on (+${fluid.delta.toFixed(2)}), ` +
    `60 frames from a seek twice: ${fluid.same ? 'identical' : 'DIFFERENT'}`);
}
if (!fluid.same) errs.push(`fluid is not deterministic: ${fluid.h1.toString(16)} vs ${fluid.h2.toString(16)}`);
if (fluid.newErrs) errs.push(`fluid produced ${fluid.newErrs} GPU errors`);
await page.evaluate(() => {
  blocks = [newBlock(0, { len: 256, mosh: 0.9, glitch: 0, var: 1, rgb: 0.3 })];
  fx = [{ type: 8, start: 0, len: 999, amt: 0.9, par: 0.4 }];
  push(); gpuReset(); pos = 20; renderAt(600, 12);
});
await grab('fx_fluid');

// ---- profiler (task 0c): when the device supports timestamps, per-pass numbers must arrive ----
await resetSection();
const prof = await page.evaluate(async () => {
  if (!G.tsQ) return { skipped: true, why: (G.tsWhy || '').slice(0, 80) };
  G.prof = true;
  blocks = [newBlock(0, { len: 256, mosh: 0.9, glitch: 0, var: 1, rgb: 0.3 })];
  fx = [{ type: 8, start: 0, len: 999, amt: 0.9, par: 0.35 }];
  push(); gpuReset(); X.reset();
  for (let i = 0; i < 10; i++) { gpuFrame(600 + i); await G.dev.queue.onSubmittedWorkDone(); }
  const T = G.timings; G.prof = false;
  return { skipped: false, ts: !!(T && T.ts), total: T ? T.total : 0, scene: T ? T.scene : 0, fluid: T ? T.fluid : 0, stamps: G.tsN };
});
if (prof.skipped) console.log(`profile: no timestamp support here (${prof.why || 'feature absent'}), overlay uses wall clock`);
else {
  console.log(`profile: ${prof.stamps} stamps, total ${prof.total.toFixed(2)} ms, scene ${prof.scene.toFixed(2)}, fluid ${prof.fluid.toFixed(2)}`);
  if (!prof.ts) errs.push('timestamp queries are supported but the profiler produced no per-pass timings');
  if (!(prof.total > 0)) errs.push('profiler reported a zero frame time');
}

// ---- thumbnails ----
await resetSection();
await page.evaluate(() => { autoArrange(); });
const thumbsWait = await waitForThumbs(page, 'thumbs');
const thumbInfo = thumbsWait.info;
console.log('thumbs:', `${thumbInfo.cached} cached, ${thumbInfo.drawn}/${thumbInfo.blocks} blocks have one`);
if (!thumbsWait.ok) errs.push(`thumbnails: only ${thumbInfo.drawn}/${thumbInfo.blocks} blocks had one after ${thumbsWait.ms} ms (timed out)`);
else if (thumbInfo.drawn === 0) errs.push('no block thumbnails were generated');

// ---- Auto-arrange must not flicker faster than the flash limit, at any tempo ----
await resetSection();
const aaFails = await page.evaluate(() => {
  const fails = [];
  for (const bpm of [60, 90, 120, 150, 180]) {
    setGrid(bpm, 0);
    autoArrange();
    const n = Math.min(nf, 30 * 60);
    let prev = -1, lastChange = -1, bad = 0, changes = 0;
    for (let f = 0; f < n; f++) {
      const p = new Float32Array(X.memory.buffer, X.plan(f), 17);
      if (p[0] === 1) continue;                       // black frame: no scene id to track
      const id = p[16];
      if (prev >= 0 && id !== prev) {
        changes++;
        if (lastChange >= 0 && f - lastChange < 12) bad++;
        lastChange = f;
      }
      prev = id;
    }
    if (bad) fails.push(bpm + ' bpm: ' + bad + ' scene changes closer than 12 frames');
    if (!changes) fails.push(bpm + ' bpm: Auto-arrange produced no scene changes at all');
  }
  return fails;
});
for (const f of aaFails) errs.push('auto-arrange: ' + f);
if (!aaFails.length) console.log('auto-arrange: no full-frame change closer than 12 frames at 60/90/120/150/180 BPM');

await page.evaluate(() => { pause(); });
await page.screenshot({ path: path.join(outDir, 'ui.png') });

const gpuErrs = await page.evaluate(() => (typeof G !== 'undefined' && G.errs ? G.errs.slice(0, 8) : []));
const lost = await page.evaluate(() => (typeof G !== 'undefined' && G.lostMsg) ? G.lostMsg : 'no');
console.log('gpu errors:', gpuErrs.length ? gpuErrs.join(' || ').slice(0, 900) : 'none', '| lost:', lost);

// ---- projects (step A): schema, round trip, both stores, migrate, and the dialog ----
await resetSection();
const projFails = await page.evaluate(async () => {
  const fails = [];
  const ck = (c, m) => { if (!c) fails.push(m); };
  try {
  ck(typeof SCHEMA === 'number', `SCHEMA is ${typeof SCHEMA}, not a number`);
  blocks = [
    newBlock(0, { len: 8, mosh: 0.4, bloom: 0.5, mods: [[1, 1, 0.4], [6, 2, 0.2], [0, 0, 0]] }),
    newBlock(7, { len: 8, var: 3 }),
    newBlock(16, { len: 8, clip: 0 }),
  ];
  if (shaders.length) blocks.push(newBlock(17, { len: 8 }));
  fx = [{ type: 8, start: 0, len: 32, amt: 0.9, par: 0.4 }, { type: 3, start: 8, len: 8, amt: 0.5, par: 0.5 }];
  push();
  let p = null;
  try { p = project(); } catch (e) { fails.push('project() threw: ' + e.message); }
  if (p) {
    ck(p.v === SCHEMA && typeof p.bpm === 'number' && typeof p.offset === 'number', 'project() is missing v/bpm/offset');
    ck(Array.isArray(p.blocks) && p.blocks.length === blocks.length, 'project() lost blocks');
    ck(Array.isArray(p.fx) && p.fx.length === 2, 'project() lost effects');
    ck(Array.isArray(p.shaders), 'project() lost shaders');
  }
  // round trip through the same path the paste box uses
  const a = JSON.stringify(project());
  try { await applyProject(JSON.parse(a)); } catch (e) { fails.push('applyProject threw: ' + e.message); }
  const b = JSON.stringify(project());
  if (a !== b) fails.push('round trip changed the project');
  // autosave must capture an edit, not just exist
  try { localStorage.removeItem('moshloom'); } catch (e) {}
  blocks[0].mosh = 0.77; push();
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem('moshloom')); } catch (e) {}
  ck(!!(saved && saved.blocks && saved.blocks[0]) && Math.abs(saved.blocks[0].mosh - 0.77) < 1e-6, 'autosave is missing or stale');
  // named store: save, change, load
  try {
    const s = { t1: project() };
    localStorage.setItem('moshloom:projects', JSON.stringify(s));
    blocks = [newBlock(2, { len: 4 })]; push();
    await applyProject(JSON.parse(localStorage.getItem('moshloom:projects')).t1);
    ck(blocks.length === a ? false : blocks.length >= 3 && Math.abs(blocks[0].mosh - 0.77) < 1e-6, 'named project did not restore');
  } catch (e) { fails.push('named store: ' + e.message); }
  // migrate
  const mk = (o) => Object.assign({ app: 'moshloom', bpm: 120, offset: 0, blocks: [newBlock(0)], fx: [], shaders: [] }, o);
  try { const q = mk({}); delete q.v; await applyProject(q); } catch (e) { fails.push('migrate rejected a project with no v: ' + e.message); }
  let v99 = null;
  try { await applyProject(mk({ v: 99 })); } catch (e) { v99 = e.message; }
  ck(v99 !== null, 'migrate accepted v:99');
  if (v99) ck(/99/.test(v99) && /\b2\b/.test(v99), 'v:99 message does not name both versions: ' + v99);
  let empty = null;
  try { await applyProject(mk({ blocks: [] })); } catch (e) { empty = e.message; }
  ck(empty !== null, 'migrate accepted an empty blocks array');
  // the writer now lives in the renderer; it must still produce exactly the engine's record
  blocks = [newBlock(2, { len: 4, mosh: 0.25, strobe: 1, glitch: 0.5, rgb: 1, var: 3, dens: 0.75, bloom: 0.5,
    mods: [[1, 1, 0.25], [6, 2, 0.5], [0, 0, 0]] })];
  fx = [{ type: 8, start: 2, len: 4, amt: 0.75, par: 0.25 }];
  push();
  const S = X.block_floats();
  const rec = Array.from(new Float32Array(X.memory.buffer, X.blocks_ptr(), S));
  const frec = Array.from(new Float32Array(X.memory.buffer, X.fx_ptr(), X.fx_floats()));
  const want = { 1: 4, 2: 2, 6: 1, 7: 0.25, 8: 1, 9: 0.5, 10: 1, 11: 3, 12: 0.75, 13: 0.5 };
  for (const k in want) if (Math.abs(rec[k] - want[k]) > 1e-6) fails.push(`writer: block[${k}] is ${rec[k]}, expected ${want[k]}`);
  const mods = [14, 15, 16, 17, 18, 19, 20, 21, 22].map((i) => rec[i]).join(',');
  if (mods !== '1,1,0.25,6,2,0.5,0,0,0') fails.push('writer: modulation slots are ' + mods);
  if (!(frec[2] === 8 && Math.abs(frec[3] - 0.75) < 1e-6)) fails.push('writer: effect record is ' + frec.slice(0, 5));
  } catch (e) { fails.push('uncaught: ' + e.message); }
  return fails;
});
for (const f of projFails) errs.push('projects: ' + f);
if (!projFails.length) console.log('projects: schema, round trip, autosave, named store, migrate ok');

// the dialog itself
const before = errs.length;
await page.click('#bProj');
await page.waitForTimeout(200);
const dlgOpen = await page.evaluate(() => getComputedStyle($('dlg')).display !== 'none');
await page.click('#pjCopy').catch(() => {});
await page.evaluate(() => { $('pj').value = $('pj').value; });
await page.click('#pjApply').catch(() => {});
await page.click('#pjClose').catch(() => {});
if (!dlgOpen) errs.push('projects: #bProj did not open the Project dialog');
if (errs.length > before) errs.push('projects: the dialog raised a page error');

// ---- analysis worker (T1b1): features come from the worker and are loaded into the render wasm ----
// plan()[2] is skipres/freeze, not time; scene A's clock is plan()[18] (pack_scene writes o[2]).
await resetSection();
const aw = await page.evaluate(() => {
  const plan = (f) => new Float32Array(X.memory.buffer, X.plan(f), 512).slice();
  const S = X.block_floats();
  const b = new Array(S).fill(0);
  b[1] = 8; b[5] = 2; b[6] = 1; b[10] = 1; b[12] = 0.5;    // variant 0: plan[18] is the clock alone
  new Float32Array(X.memory.buffer, X.blocks_ptr(), S).set(b);
  X.set_nblocks(1); X.reset();
  const a = plan(30), c = plan(31);
  return { worker: window.__analysisWorker, source: window.__featSource, nf,
    t30: a[18], t31: c[18], skipres: a[2] };
});
if (aw.source !== 'worker') errs.push('analysis: features did not arrive from the worker (' + aw.source + ')');
if (!(aw.t30 > 0)) errs.push('analysis: scene clock plan[18] is zero after loading worker features');
if (!(aw.t31 > aw.t30)) errs.push('analysis: the clock did not advance between frames');
console.log(`analysis worker: ${aw.worker}, features from ${aw.source}, nf ${aw.nf}, clock ${aw.t30.toFixed(3)} -> ${aw.t31.toFixed(3)}`);

// ---- render-mode probe (T1b2): lazy, never at boot in main mode ----
const probeInfo = await page.evaluate(() => ({
  worker: window.__workerGpu, header: $('mode').textContent, mode,
  engine: $('engine') ? $('engine').value : null,
  options: $('engine') ? [...$('engine').options].map((o) => o.value) : [],
}));
if (probeInfo.worker) errs.push('mode probe: ran at boot in main mode (' + probeInfo.worker + ')');
else if (/worker/.test(probeInfo.header)) errs.push('mode probe: header claims a worker answer before the probe ran: ' + probeInfo.header);
else console.log(`mode probe: skipped in main mode (header: ${probeInfo.header}); it runs only for worker mode or recovery`);
if (probeInfo.engine !== 'main') errs.push('engine control: #engine is ' + probeInfo.engine + ', expected main');
for (const v of ['main', 'worker', 'cpu'])
  if (!probeInfo.options.includes(v)) errs.push('engine control: #engine has no ' + v + ' option');
const rwErr = await page.evaluate(async () => {
  const old = rw; rw = null;
  try { await rwCall('frame', { f: 0 }); return 'resolved'; }
  catch (e) { return e.message; }
  finally { rw = old; }
});
if (!/no render worker/i.test(rwErr)) errs.push('rwCall with no worker did not reject clearly: ' + rwErr);
else console.log('rwCall: no worker -> "' + rwErr + '"');

// ---- worker mode: lazy probe, smoke test, and one main-vs-worker pixel comparison ----
// Both sides are fresh pages, exactly like the old lab script: this is a mode comparison, not a
// comparison against whatever state the long-lived suite page happens to be in.
let mainCmpPage = null, workerPage = null, workerOut = null, mainRef = null, workerWhy = '';
try {
  mainCmpPage = await browser.newPage({ viewport: { width: 1400, height: 900 } });
  const merrs = [];
  mainCmpPage.on('pageerror', (e) => merrs.push('PAGEERR ' + e.message));
  mainCmpPage.on('console', (m) => { if (m.type() === 'error' && !benignConsole(m.text())) merrs.push('CONSOLE ' + m.text().slice(0, 500)); });
  await mainCmpPage.addInitScript(() => { window.__MOSH_OFFSCREEN = true; });
  await mainCmpPage.goto(pathToFileURL(html).href + '?mode=main&v=' + Date.now());
  await mainCmpPage.waitForFunction(() => window.__featReady === true, null, { timeout: 30000 }).catch(() => {});
  mainRef = await mainCmpPage.evaluate(async () => {
    if (typeof mode !== 'string' || mode !== 'gpu' || typeof G === 'undefined' || !G.off)
      return { skipped: 'main GPU/offscreen path is not available' };
    const hashOf = (a) => { let x = 2166136261 >>> 0; for (let i = 0; i < a.length; i++) x = Math.imul(x ^ a[i], 16777619) >>> 0; return x >>> 0; };
    const lumOf = (a) => { let s = 0; for (let i = 0; i < a.length; i += 4) s += a[i] + a[i + 1] + a[i + 2]; return +(s / (a.length / 4 * 3 * 255)).toFixed(4); };
    bpm = 120; offs = 0; X.set_grid(bpm, offs);      // pin both sides to the same grid
    blocks = [newBlock(9, { len: 256, mosh: 0.6, glitch: 0, var: 1, rgb: 0.3 })];
    fx = [{ type: 8, start: 0, len: 999, amt: 0.7, par: 0.4 }]; push();
    gpuReset(); X.reset();
    for (let i = 0; i < 12; i++) gpuFrame(588 + i);
    for (let i = 0; i < 20; i++) gpuFrame(600 + i);
    const plan = Array.from(new Float32Array(X.memory.buffer, X.plan(619), 14)).map((v) => +v.toFixed(3));
    const d = G.dev, w = G.w, h = G.h;
    const buf = d.createBuffer({ size: w * h * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const e = d.createCommandEncoder();
    e.copyTextureToBuffer({ texture: G.off }, { buffer: buf, bytesPerRow: w * 4 }, [w, h]);
    d.queue.submit([e.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const a = new Uint8Array(buf.getMappedRange().slice(0)); buf.unmap(); buf.destroy();
    return { w, h, hash: hashOf(a), lum: lumOf(a), plan };
  });
  if (mainRef.skipped) throw new Error('main comparison page: ' + mainRef.skipped);
  if (merrs.length) throw new Error('main comparison page: ' + merrs[0]);

  workerPage = await browser.newPage({ viewport: { width: 1400, height: 900 } });
  const werrs = [];
  workerPage.on('pageerror', (e) => werrs.push('PAGEERR ' + e.message));
  workerPage.on('console', (m) => { if (m.type() === 'error' && !benignConsole(m.text())) werrs.push('CONSOLE ' + m.text().slice(0, 500)); });
  await workerPage.addInitScript(() => { window.__MOSH_OFFSCREEN = true; });
  await workerPage.goto(pathToFileURL(html).href + '?mode=worker&offscreen=1&v=' + Date.now());
  await workerPage.waitForFunction(() => window.__featReady === true, null, { timeout: 30000 }).catch(() => {});
  const ws = await workerPage.evaluate(() => ({
    runMode: window.__runMode, workerGpu: window.__workerGpu, header: $('mode').textContent,
    note: typeof gpuNote === 'string' ? gpuNote : '', err: $('err').textContent,
  }));
  if (ws.runMode !== 'worker+gpu') {
    workerWhy = (ws.workerGpu && ws.workerGpu !== 'yes' ? ws.workerGpu : 'run mode is ' + (ws.runMode || 'unknown'))
      + (ws.note ? ' (' + ws.note + ')' : '');
  } else {
    workerOut = await workerPage.evaluate(async () => {
      const hashOf = (a) => { let x = 2166136261 >>> 0; for (let i = 0; i < a.length; i++) x = Math.imul(x ^ a[i], 16777619) >>> 0; return x >>> 0; };
      const lumOf = (a) => { let s = 0; for (let i = 0; i < a.length; i += 4) s += a[i] + a[i + 1] + a[i + 2]; return +(s / (a.length / 4 * 3 * 255)).toFixed(4); };
      const readback = async () => { const r = await rwCall('readback', {}); const a = new Uint8Array(r.buf);
        return { w: r.w, h: r.h, fmt: r.fmt, bytes: a.length, hash: hashOf(a), lum: lumOf(a) }; };
      bpm = 120; offs = 0; X.set_grid(bpm, offs);      // same grid as the main reference
      blocks = [newBlock(9, { len: 256, mosh: 0.6, glitch: 0, var: 1, rgb: 0.3 })];
      fx = [{ type: 8, start: 0, len: 999, amt: 0.7, par: 0.4 }]; push();
      const featAck = await rwCall('features', {
        nf, bpm, offs,
        arrays: [0, 1, 2, 3, 4, 5, 6].map((i) => new Float32Array(X.memory.buffer, X.feat(i), nf).slice()),
        mel: new Float32Array(X.memory.buffer, X.mel_ptr(), nf * 24).slice(),
      });
      await rwCall('project', { blocks, fx, bpm, offs });
      // comparison first: exactly the old lab sequence, with no extra frames before it
      await rwCall('seek', {});
      for (let i = 0; i < 12; i++) await rwCall('frame', { f: 588 + i });
      for (let i = 0; i < 20; i++) await rwCall('frame', { f: 600 + i });
      const planW = await rwCall('plan', { f: 619 });
      const cmp = await readback();
      // smoke: the worker was initialised at boot; prove frames render and read back
      await rwCall('seek', {});
      for (let i = 0; i < 4; i++) await rwCall('frame', { f: 600 + i });
      const smoke = await readback();
      return { featAck, smoke, cmp: { ...cmp, plan: (planW.p || []).slice(0, 14).map((v) => +v.toFixed(3)) } };
    });
    if (!(workerOut.smoke.bytes === workerOut.smoke.w * workerOut.smoke.h * 4) || !(workerOut.smoke.lum > 0.01))
      errs.push(`worker smoke: readback ${workerOut.smoke.w}x${workerOut.smoke.h} ${workerOut.smoke.bytes} bytes, lum ${workerOut.smoke.lum}`);
    else console.log(`worker smoke: init + 4 frames + readback ${workerOut.smoke.w}x${workerOut.smoke.h}, lum ${workerOut.smoke.lum}`);
    if (workerOut.cmp.hash !== mainRef.hash)
      errs.push(`worker: pixels differ from main (${mainRef.hash >>> 0} vs ${workerOut.cmp.hash >>> 0}, lum ${mainRef.lum} vs ${workerOut.cmp.lum})`);
    else if (JSON.stringify(workerOut.cmp.plan) !== JSON.stringify(mainRef.plan))
      errs.push('worker: the plan differs from main on the same frame');
    else console.log(`worker: main vs worker ${mainRef.w}x${mainRef.h} hash ${workerOut.cmp.hash >>> 0} lum ${workerOut.cmp.lum} IDENTICAL`);
    // T1c: export runs in the worker and encodes from an offscreen texture, not the canvas.
    // The page only hands the bytes over; Node decodes them in the headless page above.
    const workerExport = await workerPage.evaluate(async () => {
      const b64Of = async (blob) => {
        const a = new Uint8Array(await blob.arrayBuffer());
        let s = '';
        for (let i = 0; i < a.length; i += 0x8000) s += String.fromCharCode.apply(null, a.subarray(i, i + 0x8000));
        return btoa(s);
      };
      const savedNf = nf;
      nf = 45; blocks = [newBlock(9, { len: 256, mosh: 0.6, glitch: 0, var: 1, rgb: 0.3 })];
      fx = []; lastBlob = null; push();
      await exportVideo();
      const st = $('status').textContent;
      const b = lastBlob ? lastBlob.blob : null;
      nf = savedNf;
      if (!b) return { status: st, blob: 0, mode, renderIn };
      return { status: st, blob: b.size, type: b.type, b64: await b64Of(b), mode, renderIn };
    });
    if (workerExport.b64) Object.assign(workerExport, await decodeVideoFile(workerExport.b64, workerExport.type, 'worker export'));
    if (!(workerExport.blob > 0)) errs.push('worker export: no file (' + workerExport.status + ')');
    else if (!(workerExport.w > 0) || !(workerExport.h > 0) || !(workerExport.dur > 0))
      errs.push('worker export: the file does not decode');
    else if (!(workerExport.lum > 0.01)) errs.push('worker export decoded to black frames');
    else {
      const sz = workerExport.blob >= 1e5 ? (workerExport.blob / 1e6).toFixed(2) + ' MB' : Math.max(1, Math.round(workerExport.blob / 1024)) + ' KB';
      console.log(`worker export: ${sz} ${workerExport.type} ${workerExport.w}x${workerExport.h} ${workerExport.dur}s, decoded luminance ${workerExport.lum}`);
    }
    // T1c: the Size control must resize the worker's textures, not the page's transferred canvas.
    await workerPage.selectOption('#res', '640');
    await workerPage.waitForTimeout(600);
    const workerResize = await workerPage.evaluate(async () => {
      const r = await rwCall('readback', {});
      return { w: r.w, h: r.h, W, H, run: window.__runMode };
    });
    if (workerResize.w !== 640 || workerResize.h !== 360 || workerResize.W !== 640 || workerResize.H !== 360)
      errs.push(`worker resize: readback ${workerResize.w}x${workerResize.h}, page ${workerResize.W}x${workerResize.H}`);
    else console.log(`worker resize: ${workerResize.w}x${workerResize.h}, ${workerResize.run}`);
    // A video Media block exercises the request/transfer path, and two exports must land on the
    // same clip frame at the same timestamp.
    await workerPage.bringToFront();
    await workerPage.setInputFiles('#mfile', clipVid);
    await workerPage.waitForFunction(() => /added|Could not/.test($('status').textContent), null, { timeout: 20000 }).catch(() => {});
    const mediaExport = await workerPage.evaluate(async () => {
      const b64Of = async (blob) => {
        const a = new Uint8Array(await blob.arrayBuffer());
        let s = '';
        for (let i = 0; i < a.length; i += 0x8000) s += String.fromCharCode.apply(null, a.subarray(i, i + 0x8000));
        return btoa(s);
      };
      const savedNf = nf;
      nf = 30; blocks = [newBlock(16, { len: 256, clip: 0, mosh: 0.5, glitch: 0, var: 1, rgb: 0.3 })];
      fx = [];
      const run = async () => {
        lastBlob = null; push(); await exportVideo();
        const b = lastBlob ? lastBlob.blob : null;
        if (!b) return { status: $('status').textContent, blob: 0 };
        return { status: $('status').textContent, blob: b.size, type: b.type, b64: await b64Of(b) };
      };
      const a = await run(), b = await run();
      nf = savedNf;
      return { a, b };
    });
    // Both files go through the one headless page, so a difference between the two hashes comes
    // from the encoded frames, not from which page decoded them.
    if (mediaExport.a.b64) Object.assign(mediaExport.a, await decodeVideoFile(mediaExport.a.b64, mediaExport.a.type, 'worker media export (first)'));
    if (mediaExport.b.b64) Object.assign(mediaExport.b, await decodeVideoFile(mediaExport.b.b64, mediaExport.b.type, 'worker media export (second)'));
    if (!(mediaExport.a.blob > 0) || !(mediaExport.b.blob > 0))
      errs.push('worker media export: no file (' + mediaExport.a.status + ' / ' + mediaExport.b.status + ')');
    else if (!(mediaExport.a.lum > 0.01)) errs.push('worker media export decoded to black frames');
    else if (mediaExport.a.hash !== mediaExport.b.hash)
      errs.push(`worker media export is not repeatable: ${mediaExport.a.hash >>> 0} vs ${mediaExport.b.hash >>> 0}`);
    else console.log(`worker media export: clip frame hash ${mediaExport.a.hash >>> 0} twice, luminance ${mediaExport.a.lum}`);
    // T1c: thumbnails render in the worker too.
    await workerPage.evaluate(() => { autoArrange(); });
    const workerThumbsWait = await waitForThumbs(workerPage, 'worker thumbnails');
    const workerThumbs = workerThumbsWait.info;
    console.log(`worker thumbnails: ${workerThumbs.cached} cached, ${workerThumbs.drawn}/${workerThumbs.blocks} blocks have one`);
    if (!workerThumbsWait.ok) errs.push(`worker thumbnails: only ${workerThumbs.drawn}/${workerThumbs.blocks} blocks had one after ${workerThumbsWait.ms} ms (timed out)`);
    else if (workerThumbs.drawn === 0) errs.push('worker thumbnails: no block thumbnail was generated');
    if (werrs.length) errs.push('worker page: ' + werrs[0]);
  }
  // The header control must restart through recovery. A machine can always reach CPU.
  await workerPage.selectOption('#engine', 'cpu');
  await workerPage.waitForFunction(() => window.__runMode === 'main+cpu', null, { timeout: 15000 }).catch(() => {});
  const cpuSwitch = await workerPage.evaluate(() => ({ mode: window.__runMode, header: $('mode').textContent, engine: $('engine').value }));
  if (cpuSwitch.mode !== 'main+cpu' || cpuSwitch.engine !== 'cpu')
    errs.push(`engine control: choosing CPU ended in ${cpuSwitch.mode} (select ${cpuSwitch.engine})`);
  else console.log(`engine control: worker page switched to ${cpuSwitch.mode} (${cpuSwitch.header})`);
  // CPU -> worker through the same control; where worker+GPU is unavailable it must fall back cleanly.
  const hadWorker = ws.workerGpu === 'yes';
  await workerPage.selectOption('#engine', 'worker');
  await workerPage.waitForFunction(() => window.__runMode && window.__runMode !== 'main+cpu', null, { timeout: 20000 }).catch(() => {});
  const wSwitch = await workerPage.evaluate(() => ({ mode: window.__runMode, header: $('mode').textContent, engine: $('engine').value }));
  if (hadWorker && wSwitch.mode !== 'worker+gpu')
    errs.push(`engine control: choosing worker ended in ${wSwitch.mode} (select ${wSwitch.engine})`);
  else console.log(`engine control: CPU -> ${wSwitch.mode} (${wSwitch.header})`);
  // A plain fragment must select a mode without a query string (the published page has none).
  await workerPage.goto(pathToFileURL(html).href + '#cpu');
  await workerPage.waitForFunction(() => window.__featReady === true, null, { timeout: 30000 }).catch(() => {});
  const hashMode = await workerPage.evaluate(() => ({ mode: window.__runMode, header: $('mode').textContent }));
  if (hashMode.mode !== 'main+cpu') errs.push('hash mode: #cpu booted into ' + hashMode.mode);
  else console.log(`engine control: plain #cpu booted ${hashMode.mode} (${hashMode.header})`);
} catch (e) {
  workerWhy = workerWhy || ('comparison pages failed: ' + e.message);
} finally {
  if (mainCmpPage) await mainCmpPage.close();
  if (workerPage) await workerPage.close();
}
if (workerWhy) {
  skips.push('worker: ' + workerWhy);
  console.log(`worker: skipped here (${workerWhy})`);
}

// ---- spontaneous worker failures (H2): the device-lost message and onerror both recover ---------
// README, Recovery: "The worker's own `device-lost` message and the worker `onerror` call the same
// function. A silent `terminate()` cannot be made to look like a crash, so the mechanism is tested
// by inviting it explicitly; the spontaneous triggers are wired but not simulated in the suite."
// These sections invite exactly those two triggers, each on its own fresh worker page, through the
// handlers index.src.html installs on `rw` (rw.onmessage / rw.onerror), then assert the step-down
// the README documents (worker+gpu -> main+gpu), a fresh canvas, a rendered frame and no errors.
for (const name of ['device-lost', 'onerror']) {
  let rp = null;
  try {
    rp = await browser.newPage({ viewport: { width: 1400, height: 900 } });
    const rerrs = [];
    rp.on('pageerror', (e) => rerrs.push('PAGEERR ' + e.message));
    rp.on('console', (m) => { if (m.type() === 'error' && !benignConsole(m.text())) rerrs.push('CONSOLE ' + m.text().slice(0, 500)); });
    await rp.addInitScript(() => { window.__MOSH_OFFSCREEN = true; });
    await rp.goto(pathToFileURL(html).href + '?mode=worker&offscreen=1&v=' + Date.now());
    await rp.waitForFunction(() => window.__featReady === true, null, { timeout: 30000 }).catch(() => {});
    const before = await rp.evaluate(() => ({
      runMode: window.__runMode, workerGpu: window.__workerGpu,
      note: typeof gpuNote === 'string' ? gpuNote : '',
    }));
    if (before.runMode !== 'worker+gpu') {
      const why = (before.workerGpu && before.workerGpu !== 'yes' ? before.workerGpu : 'run mode is ' + (before.runMode || 'unknown'))
        + (before.note ? ' (' + before.note + ')' : '');
      skips.push(`recovery ${name}: ` + why);
      console.log(`recovery ${name}: skipped here (${why})`);
      continue;
    }
    // Tag the live canvas and take the error count, then deliver the failure through the handler
    // the page actually installed. Frame 0 of a project is a black frame by plan (gpuFrame clears
    // and returns), so ask for frame 600 - the frame the rest of the suite reads back - by moving
    // the playhead before the trigger: recover() replays the current frame. recover() sets
    // window.__runMode in its finally block.
    await rp.evaluate((kind) => {
      window.__h2old = $('cv');
      window.__h2gpuErrs = (typeof G !== 'undefined' && G.errs) ? G.errs.length : 0;
      pos = 20;                                  // 20 s * 30 fps = frame 600
      if (kind === 'device-lost') rw.onmessage({ data: PROTO.make('device-lost', { message: 'test' }, null) });
      else rw.onerror(new ErrorEvent('error', { message: 'test' }));
    }, name);
    await rp.waitForFunction(() => window.__runMode && window.__runMode !== 'worker+gpu', null, { timeout: 45000, polling: 100 }).catch(() => {});
    const after = await rp.evaluate(async () => {
      const lumOf = (a) => {
        let s = 0;
        for (let i = 0; i < a.length; i += 4) s += a[i] + a[i + 1] + a[i + 2];
        return +(s / (a.length / 4 * 3 * 255)).toFixed(4);
      };
      let lum = null, route = 'no readable frame';
      if (mode === 'gpu' && typeof G !== 'undefined' && G.ok && G.off) {
        const d = G.dev, w = G.w, h = G.h;
        const buf = d.createBuffer({ size: w * h * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const e = d.createCommandEncoder();
        e.copyTextureToBuffer({ texture: G.off }, { buffer: buf, bytesPerRow: w * 4 }, [w, h]);
        d.queue.submit([e.finish()]);
        await d.queue.onSubmittedWorkDone();
        await buf.mapAsync(GPUMapMode.READ);
        lum = lumOf(new Uint8Array(buf.getMappedRange()));
        buf.unmap(); buf.destroy();
        route = 'G.off texture';
      } else {
        const c = $('cv');
        const g = (typeof ctx2d !== 'undefined' && ctx2d) ? ctx2d : null;
        if (g) { lum = lumOf(g.getImageData(0, 0, c.width, c.height).data); route = '2D canvas'; }
      }
      const old = window.__h2old;
      const now = $('cv');
      return {
        runMode: window.__runMode, route, lum,
        swapped: now !== old && !old.isConnected,
        gpuNew: ((typeof G !== 'undefined' && G.errs) ? G.errs.length : 0) - window.__h2gpuErrs,
      };
    });
    console.log(`recovery ${name}: ${before.runMode} -> ${after.runMode}, `
      + `${after.swapped ? 'new canvas' : 'SAME canvas'}, lum ${after.lum} (${after.route}), `
      + `page errors ${rerrs.length}, validation errors ${after.gpuNew}`);
    if (after.runMode !== 'main+gpu')
      errs.push(`recovery ${name}: ${before.runMode} stepped down to ${after.runMode}, expected main+gpu`);
    if (!after.swapped) errs.push(`recovery ${name}: the canvas element was not replaced`);
    if (!(after.lum > 0.01)) errs.push(`recovery ${name}: no frame rendered after recovery (lum ${after.lum} via ${after.route})`);
    if (after.gpuNew) errs.push(`recovery ${name}: ${after.gpuNew} new WebGPU validation error(s)`);
    if (rerrs.length) errs.push(`recovery ${name}: page error: ${rerrs[0]}`);
  } catch (e) {
    errs.push(`recovery ${name}: ${e.message}`);
  } finally {
    if (rp) await rp.close();
  }
}

// ---- export guard: an encoder that yields no chunks must fail, not report "done · 0.0 MB" ----
await resetSection();
const emptyExport = await page.evaluate(async () => {
  const RealVE = window.VideoEncoder, RealVF = window.VideoFrame;
  class FakeVE {
    static async isConfigSupported() { return { supported: true }; }
    constructor() { this.encodeQueueSize = 0; this.state = 'configured'; }
    configure() {}
    encode() {}
    async flush() {}
    close() { this.state = 'closed'; }
  }
  class FakeVF { close() {} }
  const savedNf = nf;
  window.VideoEncoder = FakeVE; window.VideoFrame = FakeVF;
  nf = 2; blocks = [newBlock(9, { len: 256, mosh: 0, glitch: 0, var: 1 })]; fx = []; lastBlob = null; push();
  try { await exportVideo(); } finally {
    window.VideoEncoder = RealVE; window.VideoFrame = RealVF; nf = savedNf;
  }
  return { status: $('status').textContent, blob: lastBlob ? lastBlob.blob.size : 0 };
});
if (emptyExport.blob !== 0) errs.push('export guard: an empty encode left a ' + emptyExport.blob + '-byte blob');
else if (!/export produced no video data|empty file/i.test(emptyExport.status))
  errs.push('export guard: an empty encode did not fail loudly (' + emptyExport.status + ')');
else if (/export done/.test(emptyExport.status)) errs.push('export guard: reported done for an empty file');
else console.log('export guard: empty encode -> "' + emptyExport.status + '"');

// ---- export from the VISIBLE canvas, decoded and checked for pixels ----
// The offscreen hook never presents, so drop it and reallocate: frames then go to the canvas,
// which is the only way to verify that new VideoFrame(canvas) captures what was just rendered.
await page.bringToFront();
await page.waitForTimeout(300);
const exp = await page.evaluate(async () => {
  const b64Of = async (blob) => {
    const a = new Uint8Array(await blob.arrayBuffer());
    let s = '';
    for (let i = 0; i < a.length; i += 0x8000) s += String.fromCharCode.apply(null, a.subarray(i, i + 0x8000));
    return btoa(s);
  };
  delete window.__MOSH_OFFSCREEN;
  gpuAttach({ offscreen: false });       // ENV.offscreen is what gpuResize reads; deleting the global is not enough
  gpuResize(G.w, G.h);
  // Present one frame before exporting: an occluded/just-focused canvas can otherwise leave the
  // first captures blank on a real GPU.
  gpuFrame(600);
  await G.dev.queue.onSubmittedWorkDone();
  await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  const savedNf = nf; nf = 45;
  blocks = [newBlock(9, { len: 256, mosh: 0.6, glitch: 0, var: 1, rgb: 0.3 })];
  fx = []; lastBlob = null; push();
  await exportVideo();
  const st = $('status').textContent;
  const b = lastBlob ? lastBlob.blob : null;
  const info = { status: st, blob: b ? b.size : 0, mode, err: $('err').textContent,
    lost: (typeof G !== 'undefined' && G.lostMsg) || '', ok: (typeof G !== 'undefined' && !!G.ok) };
  nf = savedNf;
  if (!b) return info;
  // The decode happens in Node's headless page, never here: this page may be a headed window.
  return { ...info, type: b.type, b64: await b64Of(b) };
});
const lostDevice = exp.mode !== 'gpu' || !!exp.lost || !exp.ok || /lost|device lost/i.test(exp.status || '');
if (!lostDevice && exp.b64) Object.assign(exp, await decodeVideoFile(exp.b64, exp.type, 'export'));
if (lostDevice) {
  const why = exp.lost || (/GPU was lost[^.;]*/i.exec(exp.status || '') || [])[0]
    || (exp.mode !== 'gpu' ? 'no presentable WebGPU canvas' : 'the GPU device was lost');
  skips.push('export: lost the GPU device (' + why + ')');
  console.log('export: skipped here (lost the GPU device: ' + why + ')');
} else if (!(exp.blob > 0)) {
  // SwiftShader cannot encode a frame taken from a presented WebGPU canvas; on hardware this path
  // produces a valid file (verified: 2.28 MB mp4 at 1280x720, decoded luminance 0.13 and 0.29).
  if (/encoding error|closed codec|cannot call 'encode'|encoder.*closed/i.test(exp.status || '')) {
    skips.push('export: software adapter cannot encode from the presented canvas');
    console.log('export: skipped here (software adapter cannot encode from the presented canvas)');
  } else errs.push('export produced no usable file: ' + exp.status);
} else {
  const sizeText = exp.blob >= 1e5 ? (exp.blob / 1e6).toFixed(2) + ' MB'
    : Math.max(1, Math.round(exp.blob / 1024)) + ' KB';
  console.log(`export: ${sizeText} ${exp.type} ${exp.w}x${exp.h} ${exp.dur}s, decoded luminance ${exp.lum}`);
  if (!(exp.w > 0) || !(exp.h > 0) || !(exp.dur > 0)) errs.push('exported video is not decodable');
  else if (!(exp.lum > 0.01)) errs.push('export decoded to black frames - the canvas was captured before presentation');
}
if (HARDWARE) {
  const exportSkips = skips.filter((s) => s.startsWith('export:'));
  if (exportSkips.length) errs.push('hardware run: export skipped on the real GPU (' + exportSkips.join('; ') + ')');
}

// ---- goldens ------------------------------------------------------------------
// These images belong to one engine.wasm. Same binary: identical pixels on any machine.
// A different compiler changes the mosh frames substantially (the feedback amplifies rounding),
// so the hash is checked first and a mismatch is reported as "re-record deliberately", never as
// a pixel diff that would look like a code regression.
const goldenDir = path.join(root, 'fixtures', 'golden');
const goldenMeta = path.join(goldenDir, 'meta.json');
const GOLDEN_CASES = ['scene_media_image', 'scene_media_video', 'scene_shader',
  'fx_0', 'fx_1', 'fx_2', 'fx_3', 'fx_4', 'fx_5', 'fx_6', 'fx_7', 'mods_bloom', 'fx_fluid'];
const engineSha = crypto.createHash('sha256').update(fs.readFileSync(path.join(root, 'engine.wasm'))).digest('hex');
const RECORD = process.argv.includes('--record');
const buildInfo = (() => { try { return JSON.parse(fs.readFileSync(path.join(root, 'engine.build.json'), 'utf8')); } catch (e) { return null; } })();
const clangVersion = (() => {
  try { const b = JSON.parse(fs.readFileSync(path.join(root, 'engine.build.json'), 'utf8'));
        if (b.clang && b.clang !== 'unknown') return b.clang; } catch (e) {}
  const cands = ['clang', 'C:/Program Files/LLVM/bin/clang.exe', '/usr/bin/clang'];
  for (const c of cands) { const r = spawnSync(c, ['--version'], { encoding: 'utf8' });
    if (r.status === 0) return r.stdout.split('\n')[0].trim() + ' (probed, not from engine.build.json)'; }
  return 'unknown';
})();

if (CHECK_GOLDENS && fs.existsSync(goldenDir)) {
  if (RECORD) {
    const r = spawnSync(process.execPath, [path.join(root, 'scripts', 'imgtest.mjs'), outDir, goldenDir,
      '--tol', '0', '--max-frac', '0', '--allow-missing'], { encoding: 'utf8' });
    console.log('recording goldens; differences against the previous set:');
    if (r.stdout) console.log(r.stdout.trim());
    for (const n of GOLDEN_CASES) {
      const src = path.join(outDir, n + '.ppm');
      if (!fs.existsSync(src)) { errs.push('golden: ' + n + '.ppm was not rendered'); continue; }
      fs.copyFileSync(src, path.join(goldenDir, n + '.ppm'));
    }
    fs.writeFileSync(goldenMeta, JSON.stringify({
      engineSha256: engineSha, clang: clangVersion,
      date: new Date().toISOString().slice(0, 10), images: GOLDEN_CASES.map((n) => n + '.ppm'),
    }, null, 1));
    console.log(`recorded ${GOLDEN_CASES.length} goldens for engine.wasm ${engineSha.slice(0, 16)} (${clangVersion})`);
  } else {
    let meta = null;
    try { meta = JSON.parse(fs.readFileSync(goldenMeta, 'utf8')); } catch (e) {}
    if (!meta) errs.push('golden: fixtures/golden/meta.json is missing - run npm run golden:record');
    else if (meta.engineSha256 !== engineSha) {
      console.log('\nFAIL: engine.wasm is not the build these goldens were recorded with.');
      console.log(`  goldens: ${meta.engineSha256.slice(0, 16)}  (${meta.clang}, ${meta.date})`);
      const currentClang = (buildInfo && buildInfo.engineSha256 === engineSha) ? buildInfo.clang
        : 'unknown (engine.build.json describes a different binary)';
      console.log(`  current: ${engineSha.slice(0, 16)}  (${currentClang})`);
      console.log('  Either restore the committed engine.wasm, or re-record with npm run golden:record and say why in the commit.');
      errs.push('golden hash mismatch');
    } else {
    // the page must be packed from the same binary the goldens were recorded with
    const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
    const mb = html.match(/WASM_B64="([A-Za-z0-9+/=]+)"/);
    if (mb) {
      const embedded = crypto.createHash('sha256').update(Buffer.from(mb[1], 'base64')).digest('hex');
      if (embedded !== engineSha) {
        console.log('\nFAIL: index.html embeds a different engine.wasm than the one on disk - run npm run pack.');
        errs.push('index.html is stale');
      }
    }
      const r = spawnSync(process.execPath, [path.join(root, 'scripts', 'imgtest.mjs'), outDir, goldenDir,
        '--tol', '2', '--max-frac', '0.002'], { encoding: 'utf8' });
      if (r.stdout) console.log(r.stdout.trim());
      if (r.status !== 0) errs.push('golden images drifted (see the table above)');
    }
  }
}


await closeDecodePage();
await browser.close();

const all = [...errs, ...gpuErrs.map((e) => 'GPU ' + e)];
if (all.length) {
  console.log('\nFAIL\n' + all.slice(0, 10).join('\n'));
  process.exit(1);
}
if (skips.length) console.log('skipped: ' + skips.join('; '));
console.log(`\nok: ${shots.length} frames rendered, no validation errors (ppm + ui.png in ${path.relative(process.cwd(), outDir) || '.'})`);
