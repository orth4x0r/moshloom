#!/usr/bin/env node
// scripts/gate.mjs - reduce an `npm test` log to the one gate line AGENTS.md rule 3 asks for:
// goldens N/13 with the differing pixel count, CPU ms/frame, and every skip with its reason.
//
//   node scripts/gate.mjs <logfile>    parse a saved log
//   node scripts/gate.mjs --run        run `npm test`, tee it to out/gate.log (UTF-8), parse that
//
// Exactly one line on stdout, then one of three exit codes:
//   0  PASS     1  FAIL     2  the log is truncated or unparseable (no verdict line)
//
// Logs written by `powershell> npm test > out/x.log` are UTF-16 LE with a BOM; everything else is
// read as UTF-8. The grammar below is from test.mjs and gpu_test.mjs; scripts/testdata/ holds the
// real logs this parser is pinned against.
//
//   ms/frame 10.35  worst 26.6  frames 3840  lit 99.9%     <- the CPU number (line starts with it)
//   fluid (SwiftShader, ...): 60.74 ms/frame off, ...      <- per-effect timing, NOT the CPU number
//   13 image(s): 13 within tolerance, 0 failed, 0 skipped  <- goldens summary
//   fx_0.ppm  640x360  0  0.000%  0  ok                    <- golden row; diff px follows the size
//   gpu errors: none | lost: no                            <- WebGPU validation errors
//   PAGEERR ... / CONSOLE ...                              <- page errors, listed in the FAIL block
//   skipped: export: ...                                   <- only printed when the run passed
//   worker: skipped here (no: probe timed out)             <- how a skip shows up in a FAILed run
//   ok: 13 frames rendered, no validation errors (...)     <- verdict   (test.mjs also prints a
//   FAIL                                                   <- verdict    bare `ok`, which is not one)
//
// A FAIL is reported whenever the log shows: goldens short of all within tolerance, any differing
// pixel, any page or validation error, a FAIL line, a missing CPU line in a full `npm test` log, a
// missing golden comparison on a SwiftShader run, or a skip other than the SwiftShader export skip.

import fs from 'node:fs';
import path from 'node:path';
import { spawn, spawnSync } from 'node:child_process';
import { pathToFileURL } from 'node:url';

/** The one skip a SwiftShader run is allowed to print. Verbatim from gpu_test.mjs. */
export const SOFTWARE_EXPORT_SKIP = 'export: software adapter cannot encode from the presented canvas';

const LOG_NAME = path.join('out', 'gate.log');

/** UTF-16 LE when the file starts with FF FE, UTF-8 otherwise. TextDecoder drops a UTF-8 BOM. */
export function decodeLog(buf) {
  if (buf.length >= 2 && buf[0] === 0xff && buf[1] === 0xfe) {
    return new TextDecoder('utf-16le').decode(buf.subarray(2));
  }
  return new TextDecoder('utf-8').decode(buf);
}

/**
 * Parse a decoded log. Pure, no I/O: scripts/gate.test.mjs drives this over every testdata file.
 * Returns the fields for the gate line plus `reasons`, the list of findings that force a FAIL.
 */
export function parseLog(text) {
  const lines = text.split(/\r?\n/);
  const reasons = [];

  // -- what kind of run wrote this log ------------------------------------------------
  const hardware = lines.some((l) => /^hardware run: /.test(l))
    || lines.some((l) => /^> node gpu_test\.mjs .*--hardware\b/.test(l));
  // `npm test` runs test.mjs before gpu_test.mjs, and test.mjs prints this banner line.
  const cpuHarnessRan = lines.some((l) => /^block record \d+ floats,/.test(l))
    || lines.some((l) => /^> node test\.mjs && node gpu_test\.mjs\s*$/.test(l));

  // -- CPU: only `ms/frame` at column 0. The `fluid (...)` lines contain "ms/frame" too. -----
  const cpuLine = lines.find((l) => /^ms\/frame \d/.test(l));
  const cpu = (() => {
    if (cpuLine) {
      const ms = /^ms\/frame (\d+(?:\.\d+)?)/.exec(cpuLine)[1];
      return { ms, note: null };
    }
    if (hardware) return { ms: null, note: 'hardware run' };
    if (cpuHarnessRan) {
      reasons.push('CPU line is missing from a full npm test log');
      return { ms: null, note: 'missing line from a full npm test log' };
    }
    return { ms: null, note: 'partial log: test.mjs did not run' };
  })();

  // -- goldens -----------------------------------------------------------------------------
  const summary = lines.map((l) => /^(\d+) image\(s\): (\d+) within tolerance, (\d+) failed, (\d+) skipped\s*$/.exec(l))
    .find(Boolean);
  const rows = lines.map((l) => /^(\S+\.ppm)\s+\d+x\d+\s+(\d+)\s+[\d.]+%\s+\d+\s+\S/.exec(l)).filter(Boolean);
  const diffPixels = rows.reduce((n, m) => n + Number(m[2]), 0);
  let goldens;
  if (summary) {
    const total = Number(summary[1]);
    const within = Number(summary[2]);
    const failed = Number(summary[3]);
    goldens = { kind: 'compared', total, within, failed, diffPixels };
    if (within !== total || failed !== 0) reasons.push(`goldens: ${within}/${total} within tolerance, ${failed} failed`);
    if (diffPixels !== 0) reasons.push(`goldens: ${diffPixels} differing pixels total`);
    if (rows.length && rows.length !== total) reasons.push(`goldens: table has ${rows.length} rows for ${total} images`);
  } else if (hardware) {
    // `npm run test:gpu:hw` compares no goldens unless it is passed --hardware-golden: the
    // committed goldens are SwiftShader references. That is a skip to name, never a 13/13.
    goldens = { kind: 'skipped-hardware', total: null, within: null, failed: null, diffPixels: null };
  } else {
    goldens = { kind: 'missing', total: null, within: null, failed: null, diffPixels: null };
    reasons.push('no golden comparison in a SwiftShader run');
  }

  // -- verdict -----------------------------------------------------------------------------
  const okLines = lines.filter((l) => /^ok: /.test(l));
  const failLines = lines.filter((l) => /^FAIL(?::|$)/.test(l));
  const verdict = failLines.length ? 'fail' : okLines.length ? 'ok' : null;
  if (verdict === 'fail') {
    for (const l of failLines) reasons.push(l === 'FAIL' ? 'FAIL line present' : l);
  }

  // -- page errors: gpu_test.mjs lists them in the FAIL block, so count those lines. A FAIL block
  // stops at 10 entries, so this is a lower bound when the list is full; a passing run has none.
  let pageErrors = null;
  if (verdict) {
    pageErrors = lines.filter((l) => /^(?:PAGEERR|CONSOLE) /.test(l)).length;
    if (pageErrors) reasons.push(`${pageErrors} page error line(s) in the FAIL block`);
  }

  // -- WebGPU validation errors ------------------------------------------------------------
  const gpuLine = lines.map((l) => /^gpu errors: (.*)$/.exec(l)).find(Boolean);
  let validationErrors = null;
  if (gpuLine) {
    const body = gpuLine[1].replace(/\s*\|\s*lost:.*$/, '').trim();
    validationErrors = body === 'none' ? 0 : body.split(' || ').filter((s) => s.trim()).length;
    if (validationErrors) reasons.push(`${validationErrors} WebGPU validation error(s)`);
  } else if (verdict === 'ok') {
    validationErrors = 0; // the ok line itself asserts `no validation errors`
  }

  // -- skips: the `skipped:` summary when the run passed, the `skipped here (...)` line when not --
  const skipSummary = lines.map((l) => /^skipped: (.*)$/.exec(l)).find(Boolean);
  const inline = lines.map((l) => /^([^:]+): skipped here \((.*)\)$/.exec(l)).filter(Boolean)
    .map((m) => `${m[1]}: ${m[2]}`);
  const skipItems = skipSummary ? skipSummary[1].split('; ').map((s) => s.trim()).filter(Boolean) : inline;
  const skips = skipItems.length ? skipItems.join('; ') : null;
  for (const s of skipItems) {
    if (hardware) reasons.push(`skipped on a hardware run: ${s}`);
    else if (s !== SOFTWARE_EXPORT_SKIP) reasons.push(`unexpected skip: ${s}`);
  }

  return { hardware, verdict, goldens, cpu, pageErrors, validationErrors, skips, reasons };
}

const na = (why) => `n/a (${why})`;

/** The gate line. One line, always the same shape. */
export function formatGate(r) {
  const goldens = r.goldens.kind === 'compared'
    ? `${r.goldens.within}/${r.goldens.total}, ${r.goldens.diffPixels} differing px`
    : r.goldens.kind === 'skipped-hardware'
      ? 'skipped (hardware)'
      : na('no golden comparison in the log');
  const cpu = r.cpu.ms !== null ? `${r.cpu.ms} ms/frame` : na(r.cpu.note);
  const page = r.pageErrors !== null ? `${r.pageErrors}` : na('no verdict line: the run may not have finished');
  const validation = r.validationErrors !== null ? `${r.validationErrors}` : na('no "gpu errors:" line');
  const pass = gateExit(r) === 0;
  return `gate: ${pass ? 'PASS' : 'FAIL'} · goldens ${goldens} · CPU ${cpu} · page errors ${page}`
    + ` · validation errors ${validation} · skips: ${r.skips === null ? 'none' : r.skips}`;
}

/** 0 PASS, 1 FAIL, 2 truncated or unparseable. */
export function gateExit(r) {
  if (r.verdict === null) return 2;
  return r.reasons.length ? 1 : 0;
}

/** Read a log file and return the line plus exit code. */
export function gateFile(file) {
  const result = parseLog(decodeLog(fs.readFileSync(file)));
  return { result, line: formatGate(result), code: gateExit(result) };
}

async function run() {
  // Tee `npm test` into out/gate.log as UTF-8 while it runs, then gate the file that was written,
  // so `--run` and `<logfile>` go through exactly the same parser.
  fs.mkdirSync(path.dirname(LOG_NAME), { recursive: true });
  const log = fs.createWriteStream(LOG_NAME);
  // npm is npm.cmd on Windows, and Node refuses to spawn a .cmd without a shell; having the shell
  // parse one fixed command string also avoids the shell+args deprecation warning.
  const child = process.platform === 'win32'
    ? spawn('npm test', { stdio: ['ignore', 'pipe', 'pipe'], shell: true })
    : spawn('npm', ['test'], { stdio: ['ignore', 'pipe', 'pipe'] });
  const tee = (buf) => { process.stdout.write(buf); log.write(buf); };
  child.stdout.on('data', tee);
  child.stderr.on('data', tee);
  child.on('error', (e) => console.error(`gate: could not start npm test: ${e.message}`));
  const npmCode = await new Promise((resolve) => child.on('close', resolve));
  await new Promise((resolve) => log.end(resolve));
  // The exit code comes from the log: a run that died before printing a verdict is exit 2
  // (truncated), a different finding from a FAIL line. But a non-zero npm exit is never a PASS,
  // whatever the log says.
  const { result, code } = gateFile(LOG_NAME);
  let extra = 0;
  if (npmCode !== 0) { result.reasons.push(`npm test exited ${npmCode}`); extra = 1; }
  // pretest re-packs index.html, so a committed index.html that pack.py would not produce shows
  // up as a working-tree diff here. This is the pack-freshness check (there is no hosted CI).
  const diff = spawnSync('git', ['diff', '--quiet', '--', 'index.html']);
  if (diff.status !== 0) {
    result.reasons.push(diff.status === 1 ? 'index.html differs from the pack.py output (stale or hand-edited)'
      : `git diff index.html could not run (${diff.status})`);
    extra = 1;
  }
  console.log(formatGate(result));
  if (result.reasons.length) console.log('gate reasons: ' + result.reasons.join('; '));
  process.exit(code === 0 && extra ? 1 : code);
}

function main(argv) {
  if (argv.includes('--run')) return run();
  const file = argv.find((a) => !a.startsWith('--'));
  if (!file) {
    console.error('usage: node scripts/gate.mjs <logfile>\n       node scripts/gate.mjs --run');
    process.exit(2);
  }
  let line, code;
  try {
    ({ line, code } = gateFile(file));
  } catch (e) {
    console.error(`gate: cannot read ${file}: ${e.message}`);
    process.exit(2);
  }
  console.log(line);
  process.exit(code);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main(process.argv.slice(2));
}
