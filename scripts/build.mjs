// Cross-platform build: compile engine.c to engine.wasm, then pack index.html.
// Usage: npm run build
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const root = path.resolve(import.meta.dirname, '..');

// Keep these in step with build.sh / build.bat. -fno-finite-math-only stops clang from
// folding `a<b?a:b` into a libm fminf call, which a freestanding build cannot link.
const FLAGS = [
  '--target=wasm32', '-O3', '-ffast-math', '-fno-finite-math-only', '-msimd128',
  '-nostdlib', '-ffreestanding', '-fvisibility=hidden',
  '-Wl,--no-entry', '-Wl,--export-dynamic', '-Wl,--export=__heap_base',
  '-Wl,-z,stack-size=1048576',
];

function run(cmd, args) {
  const r = spawnSync(cmd, args, { cwd: root, stdio: 'inherit' });
  if (r.error && r.error.code === 'ENOENT') return null;   // not installed, try the next candidate
  return r.status;
}

const clangs = [
  'clang',
  process.env.LLVM_BIN && path.join(process.env.LLVM_BIN, 'clang'),
  'C:/Program Files/LLVM/bin/clang.exe',
  '/usr/bin/clang',
  '/opt/homebrew/opt/llvm/bin/clang',
].filter(Boolean);

let built = false;
let usedClang = null;
for (const c of clangs) {
  const st = run(c, [...FLAGS, '-o', 'engine.wasm', 'engine.c']);
  if (st === null) continue;
  if (st !== 0) { console.error(`clang failed (${c})`); process.exit(st); }
  console.log(`compiled engine.wasm with ${c}`);
  usedClang = c;
  built = true;
  break;
}
if (!built) {
  console.error('clang not found. Install LLVM: winget install LLVM.LLVM (Windows), or use build.sh / build.bat.');
  console.error('A prebuilt engine.wasm is committed, so `npm test` still works without it.');
  process.exit(1);
}

const pys = process.platform === 'win32'
  ? [['python', ['pack.py']], ['py', ['-3', 'pack.py']]]
  : [['python3', ['pack.py']], ['python', ['pack.py']]];

let packed = false;
for (const [cmd, args] of pys) {
  const st = run(cmd, args);
  if (st === null) continue;
  if (st !== 0) { console.error(`${cmd} pack.py failed`); process.exit(st); }
  packed = true;
  break;
}
if (!packed) { console.error('python not found - needed to run pack.py'); process.exit(1); }

if (usedClang) {
  const v = spawnSync(usedClang, ['--version'], { encoding: 'utf8' });
  console.log('compiler: ' + (v.status === 0 ? v.stdout.split('\n')[0].trim() : usedClang));
}
const wasmPath = path.join(root, 'engine.wasm');
try {
  const sha = (await import('node:crypto')).createHash('sha256').update(fs.readFileSync(wasmPath)).digest('hex');
  const ver = usedClang ? spawnSync(usedClang, ['--version'], { encoding: 'utf8' }) : { status: 1 };
  fs.writeFileSync(path.join(root, 'engine.build.json'), JSON.stringify({
    engineSha256: sha,
    clang: ver.status === 0 ? ver.stdout.split('\n')[0].trim() : 'unknown',
    date: new Date().toISOString().slice(0, 10),
  }, null, 1) + '\n');
  console.log('wrote engine.build.json');
} catch (e) { console.log('could not write engine.build.json: ' + e.message); }
const wasm = fs.statSync(path.join(root, 'engine.wasm')).size;
const html = fs.statSync(path.join(root, 'index.html')).size;
console.log(`engine.wasm ${wasm} bytes, index.html ${html} bytes`);
