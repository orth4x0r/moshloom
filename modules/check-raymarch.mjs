// modules/check-raymarch.mjs - compile check for the 3D raymarch modules.
//
//   node modules/check-raymarch.mjs
//
// What it does
//   1. reads gpu.js and extracts WG_HEAD by evaluating the WG_LAYOUT..WG_HEAD
//      block of that file (that block is pure JS). If extraction ever breaks,
//      a documented stand-in header with the same U/lut/XY surface is used and
//      the output says so.
//   2. appends modules/sdf3d.wgsl, modules/raymarch.wgsl and a minimal test
//      scene (sphere; trivial map and shade) exactly the way gpu.js will
//      assemble them: header first, scene last.
//   3. compiles the module in headless Chromium (SwiftShader WebGPU, the same
//      launch flags as gpu_test.mjs) and prints PASS, or the first WGSL error
//      with its file, line and column.
//
// It needs no bind groups, textures or canvas: createShaderModule type-checks
// and validates the whole module, including functions the test scene never
// calls, so every primitive/operator in sdf3d.wgsl is covered.
//
// Exit code: 0 on PASS, 1 on any error (missing export, compile error, WebGPU
// unavailable).

import { chromium } from 'playwright';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';

const root = path.resolve(import.meta.dirname, '..');
const modDir = path.join(root, 'modules');

// Same flags as gpu_test.mjs: software WebGPU that works headless.
const ARGS = [
  '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=swiftshader',
  '--use-webgpu-adapter=swiftshader', '--ignore-gpu-blocklist',
];

// ---------------------------------------------------------------------------
// 1. source
// ---------------------------------------------------------------------------

const REQUIRED = {
  'modules/sdf3d.wgsl': [
    'sd3_sphere', 'sd3_box', 'sd3_round_box', 'sd3_torus', 'sd3_capsule',
    'sd3_cylinder', 'sd3_plane', 'sd3_gyroid', 'sd3_octahedron',
    'op3_union', 'op3_sub', 'op3_inter', 'op3_smooth_union',
    'op3_repeat_inf', 'op3_repeat_lim', 'op3_mirror', 'op3_twist',
    'op3_bend', 'op3_polar', 'op3_fold',
  ],
  'modules/raymarch.wgsl': [
    'rm_march', 'rm_normal', 'rm_ao', 'rm_shadow', 'rm_camera', 'rm_camera_at',
    'rm_look', 'rm_cam_path', 'rm_shade_ray', 'fs',
  ],
};

function readModule(name) {
  const p = path.join(modDir, name);
  if (!fs.existsSync(p)) {
    console.error(`missing ${name} - run from the repo root`);
    process.exit(1);
  }
  return fs.readFileSync(p, 'utf8');
}

// Extract WG_HEAD from gpu.js. The slice from WG_LAYOUT to WG_VS contains only
// data and pure functions, so it can be evaluated with new Function().
function extractHead() {
  const src = fs.readFileSync(path.join(root, 'gpu.js'), 'utf8');
  const a = src.indexOf('const WG_LAYOUT');
  const b = src.indexOf('const WG_VS');
  if (a < 0 || b < 0 || b <= a) return { head: null, why: 'WG_LAYOUT/WG_VS markers not found' };
  let head;
  try {
    head = new Function(src.slice(a, b) + '\nreturn WG_HEAD;')();
  } catch (e) {
    return { head: null, why: 'evaluating the header block failed: ' + e.message };
  }
  if (typeof head !== 'string' || head.length < 200) return { head: null, why: 'WG_HEAD is not a shader string' };
  for (const need of ['struct SU', 'var<uniform> U', 'fn lut(', 'fn XY(', 'textureSampleLevel']) {
    if (!head.includes(need)) return { head: null, why: `WG_HEAD is missing ${need}` };
  }
  return { head, why: null };
}

// Documented stand-in, used only if gpu.js parsing breaks. It mirrors WG_SU +
// WG_HEAD: the SU fields the modules read, lut() and XY(). Keep it in sync with
// the real header; the check prints STAND-IN when this is used.
const STANDIN_HEAD = `
// STAND-IN for WG_HEAD (gpu.js could not be parsed)
struct SU { res: vec2f, T: f32, hue: f32, bass: f32, mid: f32, high: f32, pulse: f32,
  vari: f32, dens: f32, beat: f32, frame: f32, pal: f32, aux0: f32, aux1: f32, aux2: f32,
  l0: vec4f, l1: vec4f, l2: vec4f, mel: array<vec4f, 6> };
@group(0) @binding(0) var<uniform> U: SU;
const PI = 3.14159265; const TAU = 6.2831853;
@group(0) @binding(1) var lin: sampler;
@group(0) @binding(2) var palT: texture_2d<f32>;
fn lut(v: f32) -> vec3f { return textureSampleLevel(palT, lin, vec2f(v, (U.pal + 0.5) / 8.), 0.).rgb; }
fn XY(p: vec4f) -> vec2f { return (p.xy - U.res * 0.5) / (U.res.y * 0.5); }
`;

const SCENE = `
// ---- minimal test scene appended last (order-independent module scope) ----
fn map(p: vec3f) -> vec2f {
  let d = sd3_sphere(p - vec3f(0.0, 0.25 * sin(U.T), 0.0), 1.0);
  return vec2f(d, 0.35);
}
fn shade(p: vec3f, n: vec3f, rd: vec3f, mat: f32, steps: f32) -> vec3f {
  return lut(mat * 0.6 + 0.08 + 0.22 * dot(n, -rd));
}
`;

const sdf3d = readModule('sdf3d.wgsl');
const raymarch = readModule('raymarch.wgsl');
const gotHead = extractHead();
const head = gotHead.head ?? STANDIN_HEAD;

// required exports
let missing = 0;
for (const [file, names] of Object.entries(REQUIRED)) {
  const src = file.endsWith('sdf3d.wgsl') ? sdf3d : raymarch;
  for (const n of names) {
    if (!new RegExp(`\\bfn\\s+${n}\\s*\\(`).test(src)) {
      console.error(`FAIL missing export ${n} in ${file}`);
      missing++;
    }
  }
}
if (missing) process.exit(1);

const parts = [
  { label: gotHead.head ? 'gpu.js WG_HEAD' : 'STAND-IN WG_HEAD', code: head },
  { label: 'modules/sdf3d.wgsl', code: sdf3d },
  { label: 'modules/raymarch.wgsl', code: raymarch },
  { label: 'test scene', code: SCENE },
];

// line table, so a compiler line number maps back to a file
let line = 1;
let code = '';
for (const p of parts) {
  p.start = line;
  p.lines = (p.code.match(/\n/g) || []).length + (p.code.endsWith('\n') ? 0 : 1);
  line += p.lines;
  code += p.code;
  if (!p.code.endsWith('\n')) code += '\n';
}
const allLines = code.split('\n');

function locate(n) {
  let hit = parts[0];
  for (const p of parts) if (n >= p.start) hit = p;
  return hit;
}

const count = (src, re) => [...src.matchAll(re)].map((m) => m[1]);
const sd3Fns = count(sdf3d, /\bfn\s+((?:sd3|op3)_\w+)\s*\(/g);
const rmFns = count(raymarch, /\bfn\s+(rm_\w+)\s*\(/g);

// ---------------------------------------------------------------------------
// 2. compile
// ---------------------------------------------------------------------------

async function launch() {
  try {
    return await chromium.launch({ args: ARGS });
  } catch (e) {
    try { return await chromium.launch({ channel: 'chrome', args: ARGS }); }
    catch { throw new Error('could not launch Chromium. Run: npx playwright install chromium\n' + e.message); }
  }
}

// WebGPU needs a secure context; a data:/about: page is not one, so serve the
// tiny probe document from localhost (which is treated as secure).
const server = http.createServer((req, res) => {
  res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
  res.end('<!doctype html><meta charset="utf-8"><title>raymarch check</title>');
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const url = `http://127.0.0.1:${server.address().port}/`;

let browser;
try {
  browser = await launch();
  const page = await browser.newPage();
  const pageErrors = [];
  page.on('pageerror', (e) => pageErrors.push(e.message));
  await page.goto(url);

  const result = await page.evaluate(async ({ wgsl }) => {
    if (!navigator.gpu) return { fatal: 'navigator.gpu is missing (WebGPU flags rejected)' };
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { fatal: 'no WebGPU adapter (SwiftShader unavailable)' };
    const device = await adapter.requestDevice();
    const mod = device.createShaderModule({ code: wgsl, label: 'sdf3d+raymarch' });
    const info = await mod.getCompilationInfo();
    const messages = info.messages.map((m) => ({
      type: m.type, line: m.lineNum, col: m.linePos, msg: m.message,
    }));
    device.destroy();
    return { messages };
  }, { wgsl: code });

  if (result.fatal) {
    console.error('FAIL ' + result.fatal);
    process.exitCode = 1;
  } else {
    const errors = result.messages.filter((m) => m.type === 'error');
    const warnings = result.messages.filter((m) => m.type === 'warning');

    if (errors.length === 0) {
      const headNote = gotHead.head
        ? `gpu.js WG_HEAD (extracted, ${head.split('\n').length} lines)`
        : `STAND-IN WG_HEAD (${gotHead.why}; ${head.split('\n').length} lines)`;
      console.log(`head:    ${headNote}`);
      console.log(`sdf3d:   ${sd3Fns.length} functions: ${sd3Fns.join(', ')}`);
      console.log(`raymarch:${rmFns.length} functions: ${rmFns.join(', ')}`);
      console.log(`budget:  ${/RM_MAX_STEPS: u32 = 64u/.test(raymarch) ? '64' : '?'} steps, far ${/RM_FAR: f32 = 40\.0/.test(raymarch) ? '40' : '?'}`);
      console.log(`compiled ${parts.length} chunks, ${allLines.length} lines, 0 errors` +
        (warnings.length ? `, ${warnings.length} warnings` : ''));
      for (const w of warnings) {
        console.log(`  warning ${locate(w.line).label}:${w.line}:${w.col} ${w.msg}`);
      }
      console.log('PASS');
    } else {
      const e = errors[0];
      const part = locate(e.line);
      const src = allLines[e.line - 1] ?? '';
      const col = Math.max(1, e.col);
      console.error('FAIL WGSL compile error');
      console.error(`  ${part.label}:${e.line}:${col}: ${e.msg}`);
      console.error(`  ${String(e.line).padStart(5)} | ${src}`);
      console.error(`        | ${' '.repeat(Math.min(src.length, col - 1))}^`);
      if (errors.length > 1) {
        console.error(`  (${errors.length} errors total; first shown)`);
        for (const x of errors.slice(1, 5)) {
          console.error(`  ${locate(x.line).label}:${x.line}:${x.col} ${x.msg}`);
        }
      }
      process.exitCode = 1;
    }
    if (pageErrors.length) {
      console.error('page errors: ' + pageErrors.join(' || '));
      process.exitCode = 1;
    }
  }
} catch (e) {
  console.error('FAIL ' + (e.message || String(e)));
  process.exitCode = 1;
} finally {
  if (browser) await browser.close();
  server.close();
}
