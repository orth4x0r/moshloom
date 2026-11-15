// check.mjs - compiles every modules/*.wgsl against the real WG_HEAD, in headless Chromium.
//
// The shader text is built exactly the way gpu.js builds it: WG_HEAD (the WGSU struct + samplers +
// helper functions) followed by the module, followed by a trivial fragment entry point that
// references one exported symbol of the module. Each result is compiled with
// shaderModule.getCompilationInfo() and reported as PASS/FAIL.
//
// WG_HEAD is taken from gpu.js by evaluating its (pure JS) prefix up to `const WG_VS=`, so the
// `${structText('SU',SU)}` interpolation resolves to the real struct. Nothing in gpu.js is
// imported or written. If that ever stops working the script says so and falls back to a minimal,
// clearly-marked stand-in header.
//
// Usage: node modules/check.mjs        (exit code 0 = all modules compile)
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';

const root = path.dirname(import.meta.dirname); // repo root
const modDir = import.meta.dirname;
const GPU = path.join(root, 'gpu.js');

// Launch flags mirror gpu_test.mjs.
const ARGS = [
  '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=swiftshader',
  '--use-webgpu-adapter=swiftshader', '--ignore-gpu-blocklist',
];

// ---- the modules, and the fragment body that touches each one's exports ----
const MODULES = [
  {
    file: 'noise.wgsl',
    body: `let a = nz_fbm2(q, 4) + nz_snoise2(q) + nz_snoise3(vec3f(q, 0.5)); let v = a + nz_vnoise3(vec3f(q, 0.5)) + nz_ridged2(q, 3) + nz_warp2(q, 0.2, 2).x + nz_sfbm2(q, 2) + nz_hash11(1.) + nz_hash22(q).x + nz_hash33(vec3f(q, 0.)).z;`,
  },
  {
    file: 'sdf2d.wgsl',
    body: `var vv: array<vec2f, 8>; for (var i = 0; i < 8; i++) { vv[i] = vec2f(0.1 * f32(i), 0.1); } let v = sd2_circle(q, 0.3) + sd2_box(q, vec2f(0.2)) + sd2_box_round(q, vec2f(0.2), 0.05) + sd2_box_frame(q, vec2f(0.3), 0.02) + sd2_segment(q, vec2f(0.), vec2f(0.5)) + sd2_segment_w(q, vec2f(0.), vec2f(0.5), 0.02) + sd2_line(q, vec2f(0.), vec2f(0.5)) + sd2_tri(q, vec2f(0.), vec2f(1., 0.), vec2f(0., 1.)) + sd2_ngon(q, 5., 0.4) + sd2_ngon_rot(q, 5., 0.4, 0.2) + sd2_star(q, 5., 0.4, 0.5) + sd2_poly_pts(q, vv) + sd2_poly_n(q, vv, 4) + sd2_smin(0., 1., 0.2) + sd2_smin_exp(0., 1., 0.2) + sd2_smax(0., 1., 0.2) + sd2_smin_c(0., 1., 0.2).x + sd2_fill(0.1, 0.01) + sd2_stroke(0.1, 0.02, 0.01) + sd2_ring(q, 0.4, 0.02) + SD2_BIG * 0.;`,
  },
  {
    file: 'color.wgsl',
    body: `let ip = vec2i(pos.xy); let v = col_hsv2rgb(vec3f(q.x, 0.5, 1.)).x + col_rgb2hsv(vec3f(0.5)).z + col_hue(vec3f(1., 0., 0.)) + col_luma(vec3f(0.5)) + col_sat(vec3f(0.5), 2.).x + col_cos(0.3, vec3f(0.5), vec3f(0.5), vec3f(1.), vec3f(0.)).x + col_pal(0.3, 1.5).y + col_warm(0.2).z + col_cool(0.2).z + col_fire(0.2).z + col_gray(0.5).x + col_bayer4(ip) + col_bayer8(ip) + col_dither(vec3f(0.5), 0.5, 0.1).x + col_quant(vec3f(0.5), 4.).x + col_quant_d(vec3f(0.5), 4., 0.5, 1.).x;`,
  },
  {
    file: 'polar.wgsl',
    body: `let v = pol_to(q).x + pol_from(0.5, 0.2).y + pol_to01(q).y + pol_r(q) + pol_th(q) + pol_th01(q) + pol_rot(q, 0.3).x + pol_repeat_th(0.3, 6.) + pol_repeat(q, 6.).y + pol_mirror(3.) + pol_kaleido(q, 6.).x + pol_kaleido_rot(q, 6., 0.2).y + pol_swirl(q, 0.3, 0.5).x + pol_log(q, 2.).x + pol_exp(vec2f(0., 0.), 2.).y + pol_tile(q, 8., 3.).x;`,
  },
  {
    file: 'audio.wgsl',
    body: `let v = aud_band(3) + aud_band01(5.5) + aud_norm(1) + aud_peak() + aud_avg() + aud_sum(2, 9) + aud_centroid() + aud_hist(4., 2.) + aud_hist_max(4.) + aud_hist_avg(4.) + aud_onset(4., 2., 4.) + aud_wave(3., 1.) + aud_gate(0.5, 0.2, 0.1) + aud_pulse_at(2.);`,
  },
  {
    file: 'numogram.wgsl',
    body: `let v = ZP[0].x + f32(LP[0]) + f32(SZ[0].x) + f32(GT[0].x) + num_zone(2).x + f32(num_current(1)) + num_syzygy(1).y + num_gate(2).x + num_seg(q, vec2f(0.), vec2f(0.5)) + num_rot(q, 0.3).y + num_labyrinth(q, 2).x;`,
  },
];

// Names the modules may never declare, because WG_HEAD already owns them.
const HEAD_OWNED = ['XY', 'lut', 'mel', 'melh', 'ca', 'rule', 'h1', 'h2', 'fin'];

const FALLBACK_HEAD = `// ---- STAND-IN HEADER (gpu.js WG_HEAD could not be extracted). Only the helper
// signatures the modules call are present; this is NOT the shipping head. ----
const PI = 3.14159265; const TAU = 6.2831853;
fn mel(i: i32) -> f32 { return 0.; }
fn melh(band: f32, age: f32) -> f32 { return 0.; }
fn XY(p: vec4f) -> vec2f { return p.xy; }
fn fin(c: vec3f) -> vec4f { return vec4f(clamp(c, vec3f(0.), vec3f(1.)), 1.); }
`;

function readHead() {
  const src = fs.readFileSync(GPU, 'utf8');
  const vsAt = src.indexOf('const WG_VS=');
  if (vsAt < 0) throw new Error('could not find `const WG_VS=` in gpu.js');
  // Everything up to (not including) WG_VS is pure JS that defines WG_SU and WG_HEAD.
  const prefix = src.slice(0, vsAt);
  const head = new Function(prefix + '\nreturn WG_HEAD;')();
  if (typeof head !== 'string' || !head.includes('fn fin(')) {
    throw new Error('WG_HEAD did not evaluate to the expected shader text');
  }
  return { head, how: 'gpu.js prefix evaluated (WG_SU + WG_HEAD), includes the real SU struct' };
}

// ---- static checks: no bindings, no entry points, no stolen names, no cross-module collisions ----
function staticCheck(file, text, seen) {
  const errs = [];
  const lines = text.split('\n');
  lines.forEach((l, n) => {
    const code = l.replace(/\/\/.*$/, '');
    if (/@group|@binding/.test(code)) errs.push(`line ${n + 1}: declares a binding (@group/@binding)`);
    if (/@vertex|@fragment|@compute/.test(code)) errs.push(`line ${n + 1}: declares an entry point`);
    const fn = code.match(/\bfn\s+([A-Za-z_]\w*)/);
    if (fn) {
      if (HEAD_OWNED.includes(fn[1])) errs.push(`line ${n + 1}: '${fn[1]}' collides with a WG_HEAD helper`);
      const key = 'fn ' + fn[1];
      if (seen.has(key) && seen.get(key) !== file) errs.push(`line ${n + 1}: '${fn[1]}' already declared by ${seen.get(key)}`);
      else seen.set(key, file);
    }
    // numogram's tables are shared on purpose; every other module-level const must be prefixed
    const cn = code.match(/^\s*const\s+([A-Za-z_]\w*)/);
    if (cn && !['ZP', 'SZ', 'GT', 'LP'].includes(cn[1])) {
      const key = 'const ' + cn[1];
      if (seen.has(key) && seen.get(key) !== file) errs.push(`line ${n + 1}: '${cn[1]}' already declared by ${seen.get(key)}`);
      else seen.set(key, file);
    }
  });
  return errs;
}

// ---- build the shaders ----
let head, how;
try { ({ head, how } = readHead()); }
catch (e) {
  console.log(`WARN: ${e.message}`);
  console.log('WARN: falling back to a minimal stand-in WG_HEAD (see modules/check.mjs).');
  head = FALLBACK_HEAD; how = 'stand-in header (extraction failed)';
}
console.log(`WG_HEAD: ${how}, ${head.split('\n').length} lines`);

const seen = new Map();
let staticFail = false;
const list = [];
for (const m of MODULES) {
  const p = path.join(modDir, m.file);
  if (!fs.existsSync(p)) { console.log(`FAIL ${m.file}: missing`); staticFail = true; continue; }
  const text = fs.readFileSync(p, 'utf8');
  const sErrs = staticCheck(m.file, text, seen);
  if (sErrs.length) { staticFail = true; console.log(`FAIL ${m.file} (static)\n  ` + sErrs.join('\n  ')); continue; }
  // the module is appended right after WG_HEAD, same as `FS = b => WG_HEAD + WG_VS + b`
  const pre = head + '\n';
  const code = pre + text + '\n' +
    `@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {\n` +
    `  let q = XY(pos);\n  ${m.body}\n  return fin(vec3f(v));\n}\n`;
  const offset = (pre.match(/\n/g) || []).length;
  list.push({ name: m.file, code, offset });
}

// ---- compile in headless Chromium ----
const browser = await chromium.launch({ args: ARGS });
let fatal = null;
let results = [];
try {
  const page = await browser.newPage();
  // A routed https origin is a secure context and needs no file on disk.
  await page.route('**/*', (route) => route.fulfill({ status: 200, contentType: 'text/html', body: '<!doctype html><html><body></body></html>' }));
  await page.goto('https://modules.moshloom.test/');
  const res = await page.evaluate(async ({ list }) => {
    if (!navigator.gpu) return { fatal: 'navigator.gpu is missing' };
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { fatal: 'requestAdapter() returned null' };
    const device = await adapter.requestDevice();
    const out = [];
    for (const item of list) {
      device.pushErrorScope('validation');
      let msgs = [];
      try {
        const mod = device.createShaderModule({ code: item.code, label: item.name });
        const info = await mod.getCompilationInfo();
        msgs = info.messages.map((x) => ({ type: x.type, line: x.lineNum, col: x.lineColumn, msg: x.message }));
      } catch (e) {
        msgs = [{ type: 'error', line: 0, col: 0, msg: String(e && e.message || e) }];
      }
      const scoped = await device.popErrorScope();
      if (!msgs.length && scoped) msgs.push({ type: 'error', line: 0, col: 0, msg: scoped.message });
      out.push({ name: item.name, offset: item.offset, msgs });
    }
    device.destroy();
    return { out };
  }, { list });
  if (res && res.fatal) fatal = res.fatal;
  results = (res && res.out) || [];
} finally {
  await browser.close();
}

// ---- report ----
if (fatal) {
  console.log(`FAIL: could not compile anything - ${fatal}`);
  console.log('      (need chromium with WebGPU; flags are the same as gpu_test.mjs)');
  process.exit(1);
}

let failed = staticFail;
const modLine = (item, msg) => {
  // report the line inside the module where possible, plus the absolute line
  if (!msg.line) return '';
  const rel = msg.line - item.offset;
  return rel >= 1 ? `module line ${rel} (of ${item.code.split('\n').length}, absolute ${msg.line})` : `header line ${msg.line}`;
};

for (const r of results) {
  const item = list.find((x) => x.name === r.name);
  const errs = r.msgs.filter((m) => m.type === 'error');
  const warns = r.msgs.filter((m) => m.type !== 'error');
  if (!errs.length) {
    console.log(`PASS ${r.name}${warns.length ? ` (${warns.length} warning(s))` : ''}`);
  } else {
    failed = true;
    const f = errs[0];
    console.log(`FAIL ${r.name}: ${modLine(item, f)}: ${f.msg}`);
    for (const e of errs.slice(1, 4)) console.log(`     + ${modLine(item, e)}: ${e.msg}`);
  }
}

if (failed) {
  console.log('\nFAIL');
  process.exit(1);
}
console.log(`\nok: ${results.length} modules compile against WG_HEAD`);
