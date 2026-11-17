#!/usr/bin/env node
// Extracts every WGSL module from gpu.js into native/crates/mosh-shaders/wgsl/*.wgsl.
//
//   node scripts/extract_wgsl.mjs            (writes the files)
//   node scripts/extract_wgsl.mjs --check    (writes nothing; exits 1 if the tree is stale)
//
// gpu.js keeps its shaders as template strings built on top of a few shared pieces (`WG_SU`,
// `WG_HEAD`, `WG_VS`, `FS(...)`, `WG_FU`, `WG_FL_HEAD`, ...). Those pieces are plain JavaScript
// with no DOM in them, so instead of re-parsing the file with a regex this script evaluates the
// shader section *as it is* — everything above `let WASM=null;` — and writes the resulting
// strings. gpu.js is only read, never written.
//
// The output is committed: `build.rs` compiles it to SPIR-V at build time and the extraction is
// a reviewer-visible diff of the web shaders. Re-run this by hand after gpu.js changes, then
// commit the .wgsl files with the change.
//
// One wrinkle worth knowing: a scene in gpu.js is only the *fragment* half, because `FS(...)`
// prepends the uniform struct, the helper functions, the texture bindings and the vertex stage.
// A .wgsl file here must be a complete, compilable module, so each one is written out as the
// full text gpu.js hands to `createShaderModule` (`G.mod(...)`), not as the bare fragment.

import { mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');
const GPU_JS = join(root, 'gpu.js');
const OUT_DIR = join(root, 'native', 'crates', 'mosh-shaders', 'wgsl');
const MARKER = 'let WASM=null;';

const check = process.argv.includes('--check');

const gpu = readFileSync(GPU_JS, 'utf8');
const cut = gpu.indexOf(MARKER);
if (cut < 0) {
  throw new Error(`gpu.js: marker "${MARKER}" not found; the shader section moved`);
}
// Everything above the marker is the shader section: layout tables, structText, SU/FU/FB,
// WG_SU/WG_HEAD/WG_VS/FS, WG_SCENES, WG_TEMPLATE, the reaction-diffusion and feedback passes and
// the seven fluid passes. No DOM, no async, no calls into the renderer.
const shaderSection = gpu.slice(0, cut);

const NAMES = [
  'WG_SCENES', 'WG_TEMPLATE', 'WG_RDSIM', 'WG_MIX', 'WG_COPY', 'WG_PRE', 'WG_ME', 'WG_FB',
  'WG_OUT', 'WG_FADV', 'WG_FFOR', 'WG_FCURL', 'WG_FVOR', 'WG_FDIV', 'WG_FJAC', 'WG_FGRD',
  'WG_HEAD', 'WG_VS', 'WG_SU', 'SU', 'layout', 'structText',
];
const shaders = new Function(`${shaderSection};return {${NAMES.join(',')}};`)();

// Scene slugs, in WG_SCENES order. The display names come from SCENES in index.src.html; scene 16
// is "Media" (an image or video clip), not one of the 16 named visualizers.
const SCENE_SLUGS = [
  'scene00_tunnel', 'scene01_loom', 'scene02_numogram', 'scene03_flesh', 'scene04_rotor',
  'scene05_moire', 'scene06_spectrum', 'scene07_cells', 'scene08_punch_cards', 'scene09_vortex',
  'scene10_kaleido', 'scene11_reaction', 'scene12_grid', 'scene13_glyph_rain', 'scene14_scan',
  'scene15_swarm', 'scene16_media',
];

// Files, in the order they are written. `origin` names the gpu.js constant, so a file traces back.
const files = [];
const add = (name, src, origin) => {
  if (files.some((f) => f.name === name)) throw new Error(`duplicate module name ${name}`);
  files.push({ name, src, origin });
};

if (shaders.WG_SCENES.length !== SCENE_SLUGS.length) {
  throw new Error(`gpu.js has ${shaders.WG_SCENES.length} scenes, the slug table has ${SCENE_SLUGS.length}`);
}
shaders.WG_SCENES.forEach((src, i) => add(SCENE_SLUGS[i], src, `WG_SCENES[${i}]`));

// The custom-scene template is a fragment body: gpu.js compiles it as WG_HEAD+WG_VS+code (see the
// custom shader path in gpu.js). Write the same complete module here.
add('custom_template', shaders.WG_HEAD + shaders.WG_VS + shaders.WG_TEMPLATE, 'WG_TEMPLATE');
// The shared prefix on its own: `web_prelude()` in Rust returns exactly this text.
add('web_prelude', shaders.WG_HEAD + shaders.WG_VS, 'WG_HEAD + WG_VS');

// The same prelude with the native uniform struct: identical field order, `p0..p7` appended on
// top, so `LayerUniforms` (Rust) is a superset of the web block and a native custom scene can read
// both halves. The layout table in gpu.js builds the struct, so the field order cannot drift.
const WEB_SU = shaders.structText('SU', shaders.SU);
if (!shaders.WG_HEAD.startsWith(shaders.WG_SU) || !shaders.WG_SU.includes(WEB_SU)) {
  throw new Error('gpu.js: WG_HEAD no longer starts with WG_SU, cannot build layer_prelude');
}
if (shaders.SU.bytes !== 208) {
  throw new Error(`gpu.js: SU is ${shaders.SU.bytes} bytes, the Rust WEB_UNIFORM_BYTES says 208`);
}
const LAYER_FIELDS = [...shaders.SU.fields, ...Array.from({ length: 8 }, (_, i) => ({ name: `p${i}`, type: 'f32' }))];
const LAYER = shaders.layout(LAYER_FIELDS);
if (LAYER.bytes !== 240) {
  throw new Error(`layer uniform struct is ${LAYER.bytes} bytes, LayerUniforms expects 240`);
}
const layerSu = shaders.WG_SU
  .replace(WEB_SU, shaders.structText('LU', LAYER))
  .replace('var<uniform> U: SU;', 'var<uniform> U: LU;');
add(
  'layer_prelude',
  layerSu + shaders.WG_HEAD.slice(shaders.WG_SU.length) + shaders.WG_VS,
  'WG_HEAD + WG_VS with LayerUniforms (SU + p0..p7)',
);
add('rd_sim', shaders.WG_RDSIM, 'WG_RDSIM');

add('fx_mix', shaders.WG_MIX, 'WG_MIX');
add('fx_copy', shaders.WG_COPY, 'WG_COPY');
add('fx_pre', shaders.WG_PRE, 'WG_PRE');
add('motion_estimate', shaders.WG_ME, 'WG_ME');
add('fx_feedback', shaders.WG_FB, 'WG_FB');
add('fx_output', shaders.WG_OUT, 'WG_OUT');

add('fluid_advect', shaders.WG_FADV, 'WG_FADV');
add('fluid_force', shaders.WG_FFOR, 'WG_FFOR');
add('fluid_curl', shaders.WG_FCURL, 'WG_FCURL');
add('fluid_vorticity', shaders.WG_FVOR, 'WG_FVOR');
add('fluid_divergence', shaders.WG_FDIV, 'WG_FDIV');
add('fluid_jacobi', shaders.WG_FJAC, 'WG_FJAC');
add('fluid_gradient', shaders.WG_FGRD, 'WG_FGRD');

for (const f of files) {
  if (typeof f.src !== 'string' || f.src.length === 0) {
    throw new Error(`${f.origin}: empty shader text`);
  }
  if (!/\bfn (fs|vs)\s*\(/.test(f.src)) {
    throw new Error(`${f.origin}: no entry point, the shader section moved`);
  }
  f.text = `// extracted by scripts/extract_wgsl.mjs from gpu.js: ${f.origin}\n// Do not edit: change gpu.js and re-run the script.\n${f.src.replace(/\r\n/g, '\n').replace(/\s*$/, '')}\n`;
}

const expected = new Map(files.map((f) => [`${f.name}.wgsl`, f.text]));

function existing() {
  try {
    return readdirSync(OUT_DIR).filter((n) => n.endsWith('.wgsl'));
  } catch {
    return [];
  }
}

if (check) {
  const stale = [];
  for (const name of existing()) if (!expected.has(name)) stale.push(`extra: ${name}`);
  for (const [name, text] of expected) {
    let have = null;
    try {
      have = readFileSync(join(OUT_DIR, name), 'utf8');
    } catch {
      stale.push(`missing: ${name}`);
      continue;
    }
    if (have !== text && have.replace(/\r\n/g, '\n') !== text) stale.push(`changed: ${name}`);
  }
  if (stale.length) {
    console.error(`wgsl/ is stale, run node scripts/extract_wgsl.mjs:\n  ${stale.join('\n  ')}`);
    process.exit(1);
  }
  console.log(`wgsl/: ${expected.size} modules match gpu.js`);
  process.exit(0);
}

mkdirSync(OUT_DIR, { recursive: true });
// Drop .wgsl files that no longer have a source, so a renamed pass cannot leave a stale module
// behind that still compiles and still gets a `spirv()` entry.
for (const name of existing()) if (!expected.has(name)) rmSync(join(OUT_DIR, name));
for (const [name, text] of expected) writeFileSync(join(OUT_DIR, name), text);

const bytes = [...expected.values()].reduce((n, t) => n + Buffer.byteLength(t), 0);
console.log(`wgsl/: wrote ${expected.size} modules, ${bytes} bytes`);
