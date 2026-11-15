// Headless WebGPU check for lab/fluid/index.html — same Chromium flags as gpu_test.mjs.
// Usage: node lab/fluid/verify.mjs
import { chromium } from 'playwright';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const html = path.join(import.meta.dirname, 'index.html');

const ARGS = [
  '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=swiftshader',
  '--use-webgpu-adapter=swiftshader', '--ignore-gpu-blocklist',
];

const browser = await chromium.launch({ args: ARGS });
const page = await browser.newPage({ viewport: { width: 1200, height: 800 } });

const errs = [];
page.on('pageerror', (e) => errs.push('PAGEERR ' + e.message));
page.on('console', (m) => {
  const t = m.text();
  if (m.type() === 'error') errs.push('CONSOLE ' + t.slice(0, 400));
  else if (t.startsWith('[fluid]')) console.log('  console:', t);
});

// headless SwiftShader loses the device when presenting to a canvas
await page.addInitScript(() => { window.__FLUID_OFFSCREEN = true; });
await page.goto(pathToFileURL(html).href);

const boot = await page.evaluate(() => window.__lab.ready);
if (!boot || !boot.ok) {
  console.error('init failed:', boot);
  console.error(errs.join('\n'));
  await browser.close();
  process.exit(1);
}
console.log(`init ok | adapter: ${boot.adapter} | grid ${boot.grid.join('x')} | offscreen ${boot.offscreen}`);

const labErrs = await page.evaluate(() => window.__lab.errors.slice());
if (labErrs.length) { console.error('device errors:', labErrs); await browser.close(); process.exit(1); }

await page.evaluate(() => window.__lab.pause());

// --- 200 steps
const runMs = await page.evaluate(() => window.__lab.run(200));
const state = await page.evaluate(() => window.__lab.state());
console.log(`200 steps, ${runMs.toFixed(2)} ms/step wall (one submit + onSubmittedWorkDone each)`);
console.log(`beat: ${state.beats} impulses in ${state.time.toFixed(2)} s of sim time`);
if (state.beats < 3) errs.push(`expected >=3 beat impulses, saw ${state.beats}`);

// --- readback of the velocity texture
const vel = await page.evaluate(async () => {
  const f = window.__lab.fluid, dev = window.__lab.device, W = f.W, H = f.H;
  const bytesPerRow = W * 8;                       // rgba16float
  const buf = dev.createBuffer({ size: bytesPerRow * H, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = dev.createCommandEncoder();
  enc.copyTextureToBuffer({ texture: f.velocityTexture }, { buffer: buf, bytesPerRow }, { width: W, height: H });
  dev.queue.submit([enc.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const u16 = new Uint16Array(buf.getMappedRange().slice(0));
  buf.unmap(); buf.destroy();

  const half = (h) => {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e === 0) return s * 6.103515625e-5 * (m / 1024);
    if (e === 31) return m ? NaN : s * Infinity;
    return s * Math.pow(2, e - 15) * (1 + m / 1024);
  };
  let nz = 0, bad = 0, max = 0, sum = 0, n = 0;
  const hist = [];
  for (let i = 0; i < u16.length; i += 4) {
    const vx = half(u16[i]), vy = half(u16[i + 1]);
    if (!Number.isFinite(vx) || !Number.isFinite(vy)) { bad++; continue; }
    const sp = Math.hypot(vx, vy);
    if (sp > 0) nz++;
    if (sp > max) max = sp;
    sum += sp; n++;
    if (i % 400 === 0) hist.push(+sp.toFixed(2));
  }
  return { cells: n, nonzero: nz, bad, max, mean: sum / Math.max(n, 1), sample: hist.slice(0, 8) };
});

console.log(`velocity readback: ${vel.nonzero}/${vel.cells} cells non-zero (${(100 * vel.nonzero / vel.cells).toFixed(1)}%)`);
console.log(`  max ${vel.max.toFixed(1)} cells/s, mean ${vel.mean.toFixed(2)} cells/s, non-finite ${vel.bad}`);
if (vel.nonzero === 0) errs.push('velocity field is all zeros after impulses');
if (vel.bad > 0) errs.push(`${vel.bad} non-finite velocity cells`);
if (!(vel.max > 1)) errs.push(`velocity magnitude too small (${vel.max})`);

// --- image-advection toggle path + readout text
const velShot = await page.evaluate(() => window.__lab.grab());
if (velShot) {
  console.log(`velocity display: mean ${velShot.mean.toFixed(1)}/255, lit ${(velShot.lit * 100).toFixed(1)}%, max ${velShot.max}`);
  if (velShot.lit < 0.02) errs.push('velocity display is essentially black');
}

const imgOK = await page.evaluate(async () => {
  document.getElementById('imgmode').checked = true;
  document.getElementById('imgmode').dispatchEvent(new Event('change'));
  const t0 = performance.now();
  await window.__lab.run(30);
  return { ms: (performance.now() - t0) / 30, checked: document.getElementById('imgmode').checked };
});
console.log(`image-advect mode: ${imgOK.ms.toFixed(2)} ms/step`);

const imgShot = await page.evaluate(() => window.__lab.grab());
if (imgShot) {
  console.log(`image display: mean ${imgShot.mean.toFixed(1)}/255, lit ${(imgShot.lit * 100).toFixed(1)}%, max ${imgShot.max}`);
  if (imgShot.lit < 0.3) errs.push('advected test image is not visible');
}

const hud = await page.evaluate(() => ({ ms: document.getElementById('ms').textContent, state: window.__lab.state() }));
console.log(`on-page readout: "${hud.ms}" ms (${hud.state.samples} samples)`);
if (!/^\d+\.\d\d$/.test(hud.ms)) errs.push(`HUD ms readout looks wrong: ${hud.ms}`);

const finalErrs = await page.evaluate(() => window.__lab.errors.slice());
errs.push(...finalErrs.map((e) => 'GPU ' + e));

await browser.close();

if (errs.length) {
  console.error('\nFAIL');
  for (const e of errs) console.error('  ' + e);
  process.exit(1);
}
console.log('\nPASS — no WebGPU validation errors, field is alive, readout present.');
