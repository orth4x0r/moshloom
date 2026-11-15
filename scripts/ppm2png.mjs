// Minimal PPM (P6) -> PNG converter, dependency free. Handy for looking at the harness output.
// Usage: node scripts/ppm2png.mjs <in.ppm> [out.png]
import fs from 'node:fs';
import zlib from 'node:zlib';

const [, , input, output] = process.argv;
if (!input) { console.error('usage: node scripts/ppm2png.mjs <in.ppm> [out.png]'); process.exit(2); }

const buf = fs.readFileSync(input);
// header: P6 <w> <h> <max>, arbitrary whitespace, then raw bytes
let i = 0;
const token = () => {
  while (i < buf.length && (buf[i] === 0x20 || buf[i] === 0x0a || buf[i] === 0x0d || buf[i] === 0x09)) i++;
  const s = i;
  while (i < buf.length && !(buf[i] === 0x20 || buf[i] === 0x0a || buf[i] === 0x0d || buf[i] === 0x09)) i++;
  return buf.toString('ascii', s, i);
};
const magic = token();
if (magic !== 'P6') { console.error('not a P6 ppm:', magic); process.exit(2); }
const w = +token(), h = +token(), max = +token();
i++; // single whitespace after maxval
if (max !== 255) { console.error('only maxval 255 is supported'); process.exit(2); }
const px = buf.subarray(i, i + w * h * 3);
if (px.length < w * h * 3) { console.error('truncated ppm'); process.exit(2); }

const crcTable = new Int32Array(256);
for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; crcTable[n] = c; }
const crc32 = (b, start, end) => { let c = -1; for (let n = start; n < end; n++) c = crcTable[(c ^ b[n]) & 0xff] ^ (c >>> 8); return (c ^ -1) >>> 0; };
const chunk = (type, data) => {
  const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
  const body = Buffer.concat([Buffer.from(type, 'ascii'), data]);
  const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(body, 0, body.length));
  return Buffer.concat([len, body, crc]);
};

const ihdr = Buffer.alloc(13);
ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4);
ihdr[8] = 8; ihdr[9] = 2; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
const raw = Buffer.alloc((w * 3 + 1) * h);
for (let y = 0; y < h; y++) {
  raw[y * (w * 3 + 1)] = 0;
  px.copy(raw, y * (w * 3 + 1) + 1, y * w * 3, (y + 1) * w * 3);
}
const png = Buffer.concat([
  Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
  chunk('IHDR', ihdr),
  chunk('IDAT', zlib.deflateSync(raw, { level: 9 })),
  chunk('IEND', Buffer.alloc(0)),
]);
const out = output || input.replace(/\.ppm$/i, '.png');
fs.writeFileSync(out, png);
console.log(`${out} ${w}x${h} ${png.length} bytes`);
