// CPU renderer harness. No GPU, no audio fixture: features come from X.synth().
// Usage: node test.mjs [width] [frames] [--ppm]
// Exits non-zero if the renderer drifts between runs or produces a black sequence.
import fs from 'node:fs';
import path from 'node:path';

const root = import.meta.dirname;
const { instance } = await WebAssembly.instantiate(fs.readFileSync(path.join(root, 'engine.wasm')), {});
const X = instance.exports;

const W = +process.argv[2] || 640;
const H = Math.round(W * 9 / 16);
const WANT_PPM = process.argv.includes('--ppm');
const ARG_FRAMES = +process.argv[3] || 0;

const STRIDE = X.block_floats();
const FXSTRIDE = X.fx_floats();
if (!(STRIDE >= 12 && STRIDE <= 64)) throw new Error(`implausible block record: ${STRIDE} floats`);
if (FXSTRIDE !== 5) throw new Error(`effect record is ${FXSTRIDE} floats, expected 5`);
console.log(`block record ${STRIDE} floats, effect record ${FXSTRIDE} floats, render ${W}x${H}`);

X.init(W, H);
const nf = X.synth(30 * 128, 120);           // 128 s of synthetic features at 120 BPM
const end = Math.min(ARG_FRAMES || nf, nf);
const PX = W * H * 4;

// ---- a fixed arrangement, written at the engine's own stride ----
const TRS = [0, 2, 1, 3, 4, 2, 3, 2];
const blocks = [];
for (let i = 0; i < 34; i++) {
  const b = new Array(STRIDE).fill(0);
  b[0] = i * 8;                              // start (beats)
  b[1] = 8;                                  // len
  b[2] = i % 8;                              // scene
  b[3] = 0;                                  // palette (auto)
  b[4] = TRS[i % 8];                         // transition
  b[5] = 2;                                  // transition length
  b[6] = 1;                                  // speed
  b[7] = 0.3 + 0.6 * ((i * 37) % 10) / 10;   // mosh
  b[8] = i % 3;                              // strobe (0..2)
  b[9] = 0.6;                                // glitch
  b[10] = 1;                                 // rgb split
  b[11] = i;                                 // variant
  b[12] = 0.5;                               // density
  b[13] = 0;                                 // bloom
  // three modulation slots: (source, target, amount)
  b[14] = 1; b[15] = 1; b[16] = 0.4;         // bass -> mosh
  b[17] = 5; b[18] = 0; b[19] = 0.3;         // pulse -> density
  b[20] = 0; b[21] = 0; b[22] = 0;           // off
  blocks.push(b);
}
new Float32Array(X.memory.buffer, X.blocks_ptr(), blocks.length * STRIDE).set(blocks.flat());
X.set_nblocks(blocks.length);

// exercise the effects lane too
const effects = [[0, 999, 3, 0.35, 0.5], [64, 32, 6, 0.6, 0.5]];   // Zoom, Echo
new Float32Array(X.memory.buffer, X.fx_ptr(), effects.length * FXSTRIDE).set(effects.flat());
X.set_nfx(effects.length);

// ---- render ----
const SHOTS = [30, 90, 150, 210];
const first = new Map();
let worst = 0, litFrames = 0;
const t0 = performance.now();
for (let f = 0; f < end; f++) {
  const a = performance.now();
  X.render(f);
  worst = Math.max(worst, performance.now() - a);
  const px = new Uint8Array(X.memory.buffer, X.frame_ptr(), PX);
  let h = 2166136261 >>> 0, lit = 0;
  for (let i = 0; i < PX; i += 4) {
    h = Math.imul(h ^ px[i], 16777619) >>> 0;
    if (px[i] | px[i + 1] | px[i + 2]) lit++;
  }
  if (lit > PX / 4 / 20) litFrames++;
  if (SHOTS.includes(f)) {
    first.set(f, h);
    if (WANT_PPM) {
      const rgb = Buffer.alloc(W * H * 3);
      for (let i = 0; i < W * H; i++) { rgb[i * 3] = px[i * 4]; rgb[i * 3 + 1] = px[i * 4 + 1]; rgb[i * 3 + 2] = px[i * 4 + 2]; }
      fs.writeFileSync(path.join(root, `s_${f}.ppm`), Buffer.concat([Buffer.from(`P6 ${W} ${H} 255\n`), rgb]));
    }
  }
}
const per = (performance.now() - t0) / end;
console.log(`ms/frame ${per.toFixed(2)}  worst ${worst.toFixed(1)}  frames ${end}  lit ${(100 * litFrames / end).toFixed(1)}%`);

// ---- determinism: replay from a reset and compare the sample frames ----
X.reset();
let drift = 0;
const last = Math.max(...SHOTS);
for (let f = 0; f <= last; f++) {
  X.render(f);
  if (!SHOTS.includes(f)) continue;
  const px = new Uint8Array(X.memory.buffer, X.frame_ptr(), PX);
  let h = 2166136261 >>> 0;
  for (let i = 0; i < PX; i += 4) h = Math.imul(h ^ px[i], 16777619) >>> 0;
  if (h !== first.get(f)) { drift++; console.log(`  frame ${f} drifted: ${first.get(f).toString(16)} -> ${h.toString(16)}`); }
}

// ---- strobe and transition behaviour (fixes 2, 3, 6) ----
// plan()[8] is P.inv (1 = invert, 2 = tint), plan()[16] is scene A's id.
const strobeFails = [];
const ckS = (c, m) => { if (!c) strobeFails.push(m); };
const planAt = (f) => new Float32Array(X.memory.buffer, X.plan(f), 512).slice();
function setup(bpm, strobe, opts = {}) {
  X.init(320, 180);
  const nf = X.synth(30 * 24, bpm);
  const S = X.block_floats();
  const mk = (o) => {
    const b = new Array(S).fill(0);
    b[0] = 0; b[1] = 8; b[2] = 0; b[3] = 0; b[4] = 0; b[5] = 2; b[6] = 1;
    b[7] = 0; b[8] = strobe; b[9] = 0; b[10] = 1; b[11] = 0; b[12] = 0.5;
    return Object.assign(b, o);
  };
  const blocks = [mk({})];
  if (opts.trans !== undefined) blocks.push(mk({ 0: opts.at ?? 8, 2: opts.scene2 ?? 3, 4: opts.trans, 5: opts.tlen ?? 4, 8: opts.strobe2 ?? 0 }));
  new Float32Array(X.memory.buffer, X.blocks_ptr(), blocks.length * S).set(blocks.flat());
  X.set_nblocks(blocks.length);
  X.reset();
  return nf;
}
const invSeries = (n) => { const a = []; for (let f = 0; f < n; f++) a.push(planAt(f)[8]); return a; };
const startsOf = (s) => { const a = []; for (let f = 0; f < s.length; f++) if (s[f] === 1) a.push(f); return a; };

// 1. 120 BPM, strobe 2: every beat is invert then tint, and nothing else fires
{ const s = invSeries(setup(120, 2));
  for (let f = 0; f < 90; f += 15) {
    ckS(s[f] === 1, `120/strobe2: frame ${f} inv=${s[f]}, want 1`);
    ckS(s[f + 1] === 2, `120/strobe2: frame ${f + 1} inv=${s[f + 1]}, want 2 (the tint frame)`);
  }
  for (let f = 0; f < 90; f++) { const m = f % 15; if (m !== 0 && m !== 1 && s[f]) ckS(false, `120/strobe2: stray inv ${s[f]} at ${f}`); }
}
// 2. strobe 1: the same, on every second beat
{ const s = invSeries(setup(120, 1));
  for (const f of [0, 30, 60]) { ckS(s[f] === 1, `120/strobe1: frame ${f} inv=${s[f]}, want 1`); ckS(s[f + 1] === 2, `120/strobe1: frame ${f + 1} inv=${s[f + 1]}, want 2`); }
}
// 3. 180 BPM, strobe 2: beats 10 frames apart, so every second beat, still as a pair
{ const s = invSeries(setup(180, 2)); const st = startsOf(s);
  ckS(st.length >= 4, `180/strobe2: only ${st.length} flashes`);
  ckS(st.every((v, i) => i === 0 || v - st[i - 1] >= 12), `180/strobe2: flash starts <12 apart: ${st}`);
  ckS(st.every((f) => s[f + 1] === 2), `180/strobe2: tint missing after ${st.join(',')}`);
}
// 4. 180 BPM, strobe 3: every beat, no limit
{ const s = invSeries(setup(180, 3)); const st = startsOf(s);
  ckS(st.length >= 8, `180/strobe3: only ${st.length} flashes`);
  ckS(st.every((v, i) => i === 0 || v - st[i - 1] === 10), `180/strobe3: not every beat: ${st}`);
}
// 5. a Strobe transition in the mix must not add a second invert
for (const bpm of [90, 120, 150, 180]) {
  const s = invSeries(setup(bpm, 2, { trans: 4 }));
  const st = startsOf(s);
  ckS(st.every((v, i) => i === 0 || v - st[i - 1] >= 12), `${bpm} bpm with a Strobe transition: starts <12 apart: ${st.slice(0, 12)}`);
}
// 6. the scene alternation itself must be at least 12 frames per switch (fixed below 75 bpm, opt-out for strobe 3)
for (const bpm of [60, 90, 120, 150, 180, 200]) {
  setup(bpm, 2, { trans: 4 });
  let prevId = -1, lastChange = -1, bad = 0, switches = 0;
  for (let f = 0; f < 30 * 6; f++) { const id = planAt(f)[16];
    if (prevId >= 0 && id !== prevId) { switches++; if (lastChange >= 0 && f - lastChange < 12) bad++; lastChange = f; }
    prevId = id; }
  ckS(bad === 0, `${bpm} bpm: ${bad} scene switches closer than 12 frames`);
  ckS(switches > 0, `${bpm} bpm: the Strobe transition never alternated`);
}
{ // strobe 3 on the incoming block keeps the half-beat flicker
  setup(120, 2, { trans: 4, strobe2: 3 });
  let prevId = -1, lastChange = -1, gaps = [];
  for (let f = 0; f < 120; f++) { const id = planAt(f)[16];
    if (prevId >= 0 && id !== prevId) { if (lastChange >= 0) gaps.push(f - lastChange); lastChange = f; }
    prevId = id; }
  ckS(gaps.length > 0 && gaps.every((g) => g <= 8), `strobe 3: expected half-beat switches, got ${gaps.slice(0, 8)}`);
}
// 7. a backwards frame jump without reset() must not suppress flashes
{ setup(120, 2);
  for (let f = 600; f < 640; f++) planAt(f);
  let hits = 0; for (let f = 100; f < 140; f++) if (planAt(f)[8] !== 0) hits++;
  ckS(hits > 0, 'a backwards frame jump suppressed every flash');
}
if (strobeFails.length) { for (const m of strobeFails.slice(0, 12)) console.log('  ' + m); console.log(`FAIL: ${strobeFails.length} strobe/transition checks`); }
else console.log('strobe/transition: pairs, 12-frame spacing, switch rate and jump recovery ok');

let bad = 0;
if (strobeFails.length) bad++;
if (drift) { console.log(`FAIL: ${drift}/${SHOTS.length} sample frames changed between runs`); bad++; }
if (litFrames < end * 0.5) { console.log(`FAIL: only ${litFrames}/${end} frames had visible pixels`); bad++; }
if (WANT_PPM) console.log(`wrote ${SHOTS.map(f => `s_${f}.ppm`).join(', ')}`);
console.log(bad ? 'FAIL' : 'ok');
process.exit(bad ? 1 : 0);
