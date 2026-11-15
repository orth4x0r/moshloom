// One-time fixture generator for the test suite. Run with: node scripts/make-fixtures.mjs
// Produces fixtures/clip.png and fixtures/clip.mp4 (or .webm when H.264 encoding is unavailable).
// The outputs are committed; `npm test` never runs this.
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';

const root = path.resolve(import.meta.dirname, '..');
const out = path.join(root, 'fixtures');
fs.mkdirSync(out, { recursive: true });

const browser = await chromium.launch({ args: ['--ignore-gpu-blocklist'] });
const page = await browser.newPage();
// file:// is a secure context in Chrome; about:blank is not, and WebCodecs refuses to encode there.
const host = path.join(out, '.fixture-host.html');
fs.writeFileSync(host, '<!doctype html><meta charset="utf-8"><title>fixtures</title>');
await page.goto('file:///' + host.replace(/\\/g, '/'));
await page.addScriptTag({ path: path.join(root, 'node_modules/mp4-muxer/build/mp4-muxer.js') });
await page.addScriptTag({ path: path.join(root, 'node_modules/webm-muxer/build/webm-muxer.js') });

const b64 = (u8) => {
  let s = '';
  for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
  return btoa(s);
};

const png = await page.evaluate(async () => {
  const W = 160, H = 90;
  const cv = new OffscreenCanvas(W, H);
  const g = cv.getContext('2d');
  const grd = g.createLinearGradient(0, 0, W, H);
  grd.addColorStop(0, '#120018'); grd.addColorStop(0.5, '#ff2a5a'); grd.addColorStop(1, '#00e5ff');
  g.fillStyle = grd; g.fillRect(0, 0, W, H);
  for (let i = 0; i < 5; i++) {
    g.beginPath();
    g.arc(W * (0.2 + 0.15 * i), H * (0.5 + 0.22 * Math.sin(i * 1.7)), 6 + i * 4, 0, Math.PI * 2);
    g.fillStyle = i % 2 ? 'rgba(0,0,0,.55)' : 'rgba(255,255,255,.6)';
    g.fill();
  }
  g.fillStyle = '#000';
  for (let y = 0; y < H; y += 4) g.fillRect(0, y, W, 1);
  const blob = await cv.convertToBlob({ type: 'image/png' });
  const u8 = new Uint8Array(await blob.arrayBuffer());
  let s = '';
  for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
  return btoa(s);
});

const video = await page.evaluate(async () => {
  const W = 160, H = 90, N = 45, FPS = 30;
  const cv = new OffscreenCanvas(W, H);
  const g = cv.getContext('2d');
  const toB64 = (u8) => {
    let s = '';
    for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
    return btoa(s);
  };
  const draw = (i) => {
    const t = i / FPS;
    g.fillStyle = '#08000c'; g.fillRect(0, 0, W, H);
    for (let k = 0; k < 6; k++) {
      g.beginPath();
      g.arc(W / 2 + Math.cos(t * 2 + k) * 40, H / 2 + Math.sin(t * 3 + k * 1.3) * 26, 6 + k * 3, 0, Math.PI * 2);
      g.fillStyle = `hsl(${(k * 55 + t * 90) % 360} 100% 55% / .78)`;
      g.fill();
    }
    g.fillStyle = 'rgba(255,255,255,.85)';
    g.fillRect((t * 90) % W, 0, 3, H);
  };
  const cands = [
    { codec: 'avc1.42E01F', mux: 'mp4', vcodec: 'avc' },
    { codec: 'vp09.00.10.08', mux: 'webm', vcodec: 'V_VP9' },
    { codec: 'vp8', mux: 'webm', vcodec: 'V_VP8' },
  ];
  for (const c of cands) {
    try {
      const sup = await VideoEncoder.isConfigSupported({ codec: c.codec, width: W, height: H, bitrate: 500000, framerate: FPS });
      if (!sup.supported) continue;
      const M = c.mux === 'mp4' ? Mp4Muxer : WebMMuxer;
      const target = new M.ArrayBufferTarget();
      const mux = new M.Muxer({
        target,
        video: { codec: c.vcodec, width: W, height: H, frameRate: FPS },
        ...(c.mux === 'mp4' ? { fastStart: 'in-memory' } : {}),
        firstTimestampBehavior: 'offset',
      });
      let err = null;
      const enc = new VideoEncoder({ output: (ch, meta) => mux.addVideoChunk(ch, meta), error: (e) => { err = e; } });
      enc.configure({ codec: c.codec, width: W, height: H, bitrate: 500000, framerate: FPS });
      for (let i = 0; i < N; i++) {
        draw(i);
        const f = new VideoFrame(cv, { timestamp: Math.round(i * 1e6 / FPS), duration: Math.round(1e6 / FPS) });
        enc.encode(f, { keyFrame: i % 15 === 0 });
        f.close();
      }
      await enc.flush();
      enc.close();
      if (err) throw err;
      mux.finalize();
      const u8 = new Uint8Array(target.buffer);
      return { ext: c.mux, codec: c.codec, size: u8.length, b64: toB64(u8) };
    } catch (e) {
      (window.__fixtureErrors ||= []).push(c.codec + ': ' + (e && e.message ? e.message : e));
    }
  }
  return { error: 'no supported video encoder: ' + ((window.__fixtureErrors || []).join(' | ') || 'isConfigSupported said no for all') };
});

await browser.close();

if (video.error) { console.error('fixture video failed:', video.error); process.exit(1); }

fs.writeFileSync(path.join(out, 'clip.png'), Buffer.from(png, 'base64'));
const vidName = `clip.${video.ext}`;
fs.writeFileSync(path.join(out, vidName), Buffer.from(video.b64, 'base64'));
const total = fs.statSync(path.join(out, 'clip.png')).size + fs.statSync(path.join(out, vidName)).size;
console.log(`fixtures/clip.png  ${fs.statSync(path.join(out, 'clip.png')).size} bytes`);
console.log(`fixtures/${vidName}  ${fs.statSync(path.join(out, vidName)).size} bytes  (${video.codec})`);
console.log(`total ${(total / 1024).toFixed(0)} KB`);
if (total > 200 * 1024) console.warn('WARNING: fixtures exceed the 200 KB budget');
