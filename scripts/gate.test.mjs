#!/usr/bin/env node
// scripts/gate.test.mjs - the gate parser against real logs.
//
// Every file in scripts/testdata/ is a saved `npm test` / `npm run test:gpu:hw` log with its
// expected gate line, exit code and fields written out here by hand. No expectation is derived
// from the parser. `node scripts/gate.test.mjs`, exit 0 when every case matches.
//
// The logs are real runs, transcribed (not edited) from out/ logs: one is still UTF-16 LE so the
// decoder stays tested, the rest are UTF-8. synthetic-truncated.log is the one deliberate cut.

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { decodeLog, parseLog, formatGate, gateExit, SOFTWARE_EXPORT_SKIP } from './gate.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const dir = path.join(here, 'testdata');

// file -> what the parser must say about it.
const CASES = [
  {
    file: 'pass-swiftshader.log', encoding: 'utf-16le', code: 0, verdict: 'ok',
    why: 'a full `npm test` run, goldens clean, the one allowed skip',
    goldens: { kind: 'compared', total: 13, within: 13, failed: 0, diffPixels: 0 },
    cpu: { ms: '10.35', note: null },
    pageErrors: 0, validationErrors: 0,
    skips: 'export: software adapter cannot encode from the presented canvas',
    reasons: [],
    line: 'gate: PASS · goldens 13/13 (0 differing) · CPU 10.35 ms/frame · page errors 0'
      + ' · validation errors 0 · skips: export: software adapter cannot encode from the presented canvas',
  },
  {
    file: 'pass-hardware.log', encoding: 'utf-8', code: 0, verdict: 'ok',
    why: '`npm run test:gpu:hw`: no CPU harness, goldens skipped, never a 13/13',
    goldens: { kind: 'skipped-hardware', total: null, within: null, failed: null, diffPixels: null },
    cpu: { ms: null, note: 'hardware run' },
    pageErrors: 0, validationErrors: 0,
    skips: null,
    reasons: [],
    line: 'gate: PASS · goldens skipped (hardware) · CPU n/a (hardware run) · page errors 0'
      + ' · validation errors 0 · skips: none',
  },
  {
    file: 'fail-worker-pixels.log', encoding: 'utf-8', code: 1, verdict: 'fail',
    why: 'goldens clean and every step ran, but the log ends in a FAIL',
    goldens: { kind: 'compared', total: 13, within: 13, failed: 0, diffPixels: 0 },
    cpu: { ms: '10.67', note: null },
    pageErrors: 0, validationErrors: 0,
    skips: SOFTWARE_EXPORT_SKIP,
    reasons: ['FAIL line present'],
    line: `gate: FAIL · goldens 13/13 (0 differing) · CPU 10.67 ms/frame · page errors 0`
      + ` · validation errors 0 · skips: ${SOFTWARE_EXPORT_SKIP}`,
  },
  {
    file: 'fail-goldens-drift.log', encoding: 'utf-8', code: 1, verdict: 'fail',
    why: 'hardware run whose goldens differ, and an unexpected worker skip',
    goldens: { kind: 'compared', total: 13, within: 0, failed: 13, diffPixels: 2351169 },
    cpu: { ms: null, note: 'hardware run' },
    pageErrors: 0, validationErrors: 0,
    skips: 'worker: no: probe timed out',
    reasons: [
      'goldens: 0/13 within tolerance, 13 failed',
      'goldens: 2351169 differing pixels total',
      'FAIL line present',
      'skipped on a hardware run: worker: no: probe timed out',
    ],
    line: 'gate: FAIL · goldens 0/13 (13 differing) · CPU n/a (hardware run) · page errors 0'
      + ' · validation errors 0 · skips: worker: no: probe timed out',
  },
  {
    file: 'fail-hardware-goldens.log', encoding: 'utf-8', code: 1, verdict: 'fail',
    why: '`--hardware --hardware-golden`: a hardware run that did compare, so not "skipped"',
    goldens: { kind: 'compared', total: 13, within: 0, failed: 13, diffPixels: 2351169 },
    cpu: { ms: null, note: 'hardware run' },
    pageErrors: 0, validationErrors: 0,
    skips: null,
    reasons: [
      'goldens: 0/13 within tolerance, 13 failed',
      'goldens: 2351169 differing pixels total',
      'FAIL line present',
    ],
    line: 'gate: FAIL · goldens 0/13 (13 differing) · CPU n/a (hardware run) · page errors 0'
      + ' · validation errors 0 · skips: none',
  },
  {
    file: 'fail-page-errors.log', encoding: 'utf-8', code: 1, verdict: 'fail',
    why: 'goldens clean, but two CONSOLE lines in the FAIL block',
    goldens: { kind: 'compared', total: 13, within: 13, failed: 0, diffPixels: 0 },
    cpu: { ms: '9.95', note: null },
    pageErrors: 2, validationErrors: 0,
    skips: SOFTWARE_EXPORT_SKIP,
    reasons: ['FAIL line present', '2 page error line(s) in the FAIL block'],
    line: `gate: FAIL · goldens 13/13 (0 differing) · CPU 9.95 ms/frame · page errors 2`
      + ` · validation errors 0 · skips: ${SOFTWARE_EXPORT_SKIP}`,
  },
  {
    file: 'partial-gpu-only.log', encoding: 'utf-8', code: 1, verdict: 'fail',
    why: '`node gpu_test.mjs` on its own: no CPU number exists, and none is invented',
    goldens: { kind: 'compared', total: 13, within: 13, failed: 0, diffPixels: 0 },
    cpu: { ms: null, note: 'partial log: test.mjs did not run' },
    pageErrors: 0, validationErrors: 0,
    skips: SOFTWARE_EXPORT_SKIP,
    reasons: ['FAIL line present'],
    line: `gate: FAIL · goldens 13/13 (0 differing) · CPU n/a (partial log: test.mjs did not run)`
      + ` · page errors 0 · validation errors 0 · skips: ${SOFTWARE_EXPORT_SKIP}`,
  },
  {
    file: 'synthetic-truncated.log', encoding: 'utf-8', code: 2, verdict: null,
    why: 'synthetic: the same passing run cut inside the golden table, so there is no verdict',
    goldens: { kind: 'missing', total: null, within: null, failed: null, diffPixels: null },
    cpu: { ms: '10.35', note: null },
    pageErrors: null, validationErrors: 0,
    skips: SOFTWARE_EXPORT_SKIP,
    reasons: ['no golden comparison in a SwiftShader run'],
    line: `gate: FAIL · goldens n/a (no golden comparison in the log) · CPU 10.35 ms/frame`
      + ` · page errors n/a (no verdict line: the run may not have finished) · validation errors 0`
      + ` · skips: ${SOFTWARE_EXPORT_SKIP}`,
  },
];

// Grammar traps that no saved log in testdata/ happens to contain. Inline snippets, not files:
// they pin the parser's reading of a line, not the behaviour of a run.
const TRAPS = [
  {
    why: 'the `fluid (...): 60.74 ms/frame` line is not the CPU number',
    log: [
      'block record 24 floats, effect record 5 floats, render 640x360',
      'ms/frame 10.35  worst 26.6  frames 3840  lit 99.9%',
      'fluid (SwiftShader, software): 60.74 ms/frame off, 114.35 on (+53.61)',
      '13 image(s): 13 within tolerance, 0 failed, 0 skipped',
      'gpu errors: none | lost: no',
      '',
      'ok: 13 frames rendered, no validation errors (ppm + ui.png in out)',
    ].join('\n'),
    expect: (r) => { assert.equal(r.cpu.ms, '10.35'); assert.equal(gateExit(r), 0); },
  },
  {
    why: "test.mjs prints a bare `ok`, which is not the verdict: a log that stops there is truncated",
    log: 'block record 24 floats, effect record 5 floats, render 640x360\nms/frame 9.00  worst 9  frames 10  lit 1%\nok\n',
    expect: (r) => { assert.equal(r.verdict, null); assert.equal(gateExit(r), 2); },
  },
  {
    why: 'a `FAIL: ...` line from test.mjs is a verdict, not a detail',
    log: 'block record 24 floats, effect record 5 floats, render 640x360\nms/frame 9.00  worst 9  frames 10  lit 1%\nFAIL: 2 strobe/transition checks\n',
    expect: (r) => { assert.equal(r.verdict, 'fail'); assert.equal(gateExit(r), 1); },
  },
  {
    why: 'a full npm test log without an `ms/frame` line is a FAIL, not an n/a',
    log: [
      '> node test.mjs && node gpu_test.mjs',
      'block record 24 floats, effect record 5 floats, render 640x360',
      '13 image(s): 13 within tolerance, 0 failed, 0 skipped',
      'gpu errors: none | lost: no',
      '',
      'ok: 13 frames rendered, no validation errors (ppm + ui.png in out)',
    ].join('\n'),
    expect: (r) => {
      assert.equal(r.cpu.ms, null);
      assert.deepEqual(r.reasons, ['CPU line is missing from a full npm test log']);
      assert.equal(gateExit(r), 1);
    },
  },
  {
    why: 'a skip other than the SwiftShader export skip is a FAIL even on a green run',
    log: [
      'block record 24 floats, effect record 5 floats, render 640x360',
      'ms/frame 10.00  worst 12  frames 10  lit 99%',
      '13 image(s): 13 within tolerance, 0 failed, 0 skipped',
      'gpu errors: none | lost: no',
      'skipped: worker: no: probe timed out',
      '',
      'ok: 13 frames rendered, no validation errors (ppm + ui.png in out)',
    ].join('\n'),
    expect: (r) => {
      assert.equal(r.verdict, 'ok');
      assert.deepEqual(r.reasons, ['unexpected skip: worker: no: probe timed out']);
      assert.equal(gateExit(r), 1);
    },
  },
  {
    why: 'any differing pixel fails, even when the images are inside the tolerance band',
    log: [
      'block record 24 floats, effect record 5 floats, render 640x360',
      'ms/frame 10.00  worst 12  frames 10  lit 99%',
      '13 image(s): 13 within tolerance, 0 failed, 0 skipped',
      'image                  size         diff px      frac   maxd  result',
      'fx_0.ppm               640x360            3    0.001%      2  ok',
      'gpu errors: none | lost: no',
      '',
      'ok: 13 frames rendered, no validation errors (ppm + ui.png in out)',
    ].join('\n'),
    expect: (r) => {
      assert.equal(r.verdict, 'ok');
      assert.ok(r.reasons.includes('goldens: 3 differing pixels total'), r.reasons.join(' / '));
      assert.equal(gateExit(r), 1);
    },
  },
];

let failures = 0;
const fail = (what, e) => { failures++; console.log(`FAIL ${what}\n  ${String(e.message).split('\n').join('\n  ')}`); };

for (const c of CASES) {
  const file = path.join(dir, c.file);
  try {
    const buf = fs.readFileSync(file);
    // the one file kept in the encoding PowerShell writes
    if (c.encoding === 'utf-16le') assert.deepEqual([...buf.subarray(0, 2)], [0xff, 0xfe], 'expected a UTF-16 LE BOM');
    else assert.notDeepEqual([...buf.subarray(0, 2)], [0xff, 0xfe], 'expected UTF-8, not UTF-16 LE');
    const text = decodeLog(buf);
    assert.ok(!text.includes('\u0000'), 'decoded text still holds NUL bytes: wrong encoding');
    assert.ok(text.endsWith('\n'), 'the log lost its last line');

    const r = parseLog(text);
    assert.equal(r.verdict, c.verdict, 'verdict');
    assert.deepEqual(r.goldens, c.goldens, 'goldens');
    assert.deepEqual(r.cpu, c.cpu, 'cpu');
    assert.equal(r.pageErrors, c.pageErrors, 'page errors');
    assert.equal(r.validationErrors, c.validationErrors, 'validation errors');
    assert.equal(r.skips, c.skips, 'skips');
    assert.deepEqual(r.reasons, c.reasons, 'reasons');
    assert.equal(formatGate(r), c.line, 'gate line');
    assert.equal(gateExit(r), c.code, 'exit code');
    console.log(`ok   ${c.file.padEnd(28)} exit ${c.code}  ${c.why}`);
  } catch (e) {
    fail(c.file, e);
  }
}

for (const t of TRAPS) {
  try {
    t.expect(parseLog(t.log));
    console.log(`ok   grammar: ${t.why}`);
  } catch (e) {
    fail(`grammar: ${t.why}`, e);
  }
}

// Windows checks testdata out with CRLF (text=auto + core.autocrlf=true); CI reads it as LF. Both
// readings have to produce the same gate line, so compare a CRLF copy against the real file.
try {
  const text = decodeLog(fs.readFileSync(path.join(dir, 'pass-swiftshader.log')));
  assert.deepEqual(parseLog(text.replace(/\n/g, '\r\n')), parseLog(text), 'CRLF and LF parse the same log differently');
  console.log('ok   grammar: a CRLF checkout of the same log parses identically');
} catch (e) {
  fail('grammar: CRLF checkout', e);
}

console.log(`\ngate.test: ${CASES.length} testdata files, ${TRAPS.length + 1} grammar checks, ${failures} failed`);
process.exit(failures ? 1 : 0);
