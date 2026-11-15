// Milestone archive for the external reviewer.
//
// Usage:
//   npm run zip -- --sha <rev> [--out <path>] [--check-pack] <log> [<log> ...]
//   node scripts/zip.mjs --self-test
//
// The tree half of the archive is `git archive --format=zip` of <rev>, copied entry for entry
// (same compressed bytes, so the tree half is exactly what git wrote). Logs are added under
// <prefix>/logs/ and MANIFEST.txt last. The archive is then read back from disk and every
// entry is inflated, CRC-checked and re-hashed; the zip reader below is the only thing that
// looks at the finished file.
//
// Node built-ins only: zlib for deflate/inflate, crypto for sha256, child_process for git and
// pack.py. crc32 is our own table so the code does not depend on a recent zlib.crc32.
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import zlib from 'node:zlib';

const root = path.resolve(import.meta.dirname, '..');
const PREFIX_NAME = 'logs';

// ---------------------------------------------------------------- small helpers

class Fail extends Error {}

function fail(msg) {
  throw new Fail(msg);
}

function sha256(buf) {
  return createHash('sha256').update(buf).digest('hex');
}

function git(args, opts = {}) {
  const r = spawnSync('git', args, {
    cwd: opts.cwd || root,
    encoding: opts.encoding === undefined ? 'utf8' : opts.encoding,
    maxBuffer: 1 << 28,
  });
  if (r.error) fail(`git ${args[0]} could not start: ${r.error.message}`);
  if (r.status !== 0) {
    const err = (r.stderr || '').toString().trim();
    fail(`git ${args.join(' ')} failed (${r.status})${err ? ': ' + err : ''}`);
  }
  return r.stdout;
}

function bytes(n) {
  return n.toLocaleString('en-US');
}

// ---------------------------------------------------------------- crc32

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(buf) {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

// ---------------------------------------------------------------- entry name validator
//
// The archive must never carry a `node_modules` or `out` path segment: those are local
// install/output trees, and shipping one to the reviewer is the failure this check exists for.
// A segment is a whole path component, so `about/x` passes while `a/out/y` does not.

function segmentsOf(name) {
  return name.replace(/\\/g, '/').split('/').filter((s) => s.length > 0);
}

function forbiddenSegment(name) {
  for (const s of segmentsOf(name)) {
    const low = s.toLowerCase();
    if (low === 'node_modules') return 'node_modules';
    if (low === 'out') return 'out';
  }
  return null;
}

function validateEntries(names) {
  const problems = [];
  const seen = new Map();
  for (const name of names) {
    const hit = forbiddenSegment(name);
    if (hit) problems.push({ name, reason: `forbidden "${hit}" path segment` });
    const key = name.replace(/\\/g, '/').toLowerCase();
    if (seen.has(key)) problems.push({ name, reason: `duplicate entry (also ${seen.get(key)})` });
    else seen.set(key, name);
  }
  return { ok: problems.length === 0, problems };
}

// ---------------------------------------------------------------- zip reader

const SIG_LOCAL = 0x04034b50;
const SIG_CENTRAL = 0x02014b50;
const SIG_EOCD = 0x06054b50;

function readZip(buf, label) {
  const minEocd = Math.max(0, buf.length - 65557);
  let eocd = -1;
  for (let i = buf.length - 22; i >= minEocd; i--) {
    if (buf.readUInt32LE(i) !== SIG_EOCD) continue;
    if (i + 22 + buf.readUInt16LE(i + 20) === buf.length) { eocd = i; break; }
  }
  if (eocd < 0) fail(`${label}: no zip end-of-central-directory record found`);

  const count = buf.readUInt16LE(eocd + 10);
  const cdSize = buf.readUInt32LE(eocd + 12);
  const cdOffset = buf.readUInt32LE(eocd + 16);
  if (count === 0xffff || cdSize === 0xffffffff || cdOffset === 0xffffffff) {
    fail(`${label}: zip64 archives are not supported by this tool`);
  }
  if (cdOffset + cdSize > eocd) fail(`${label}: central directory runs past the EOCD record`);

  const entries = [];
  let p = cdOffset;
  for (let i = 0; i < count; i++) {
    if (p + 46 > buf.length || buf.readUInt32LE(p) !== SIG_CENTRAL) {
      fail(`${label}: central directory entry ${i} is malformed at offset ${p}`);
    }
    const nameLen = buf.readUInt16LE(p + 28);
    const extraLen = buf.readUInt16LE(p + 30);
    const commentLen = buf.readUInt16LE(p + 32);
    const e = {
      versionMadeBy: buf.readUInt16LE(p + 4),
      versionNeeded: buf.readUInt16LE(p + 6),
      flags: buf.readUInt16LE(p + 8),
      method: buf.readUInt16LE(p + 10),
      time: buf.readUInt16LE(p + 12),
      date: buf.readUInt16LE(p + 14),
      crc: buf.readUInt32LE(p + 16),
      csize: buf.readUInt32LE(p + 20),
      usize: buf.readUInt32LE(p + 24),
      externalAttrs: buf.readUInt32LE(p + 38),
      localOffset: buf.readUInt32LE(p + 42),
      name: buf.subarray(p + 46, p + 46 + nameLen).toString('utf8'),
    };
    p += 46 + nameLen + extraLen + commentLen;

    const lh = e.localOffset;
    if (lh + 30 > buf.length || buf.readUInt32LE(lh) !== SIG_LOCAL) {
      fail(`${label}: entry "${e.name}" has no local header at offset ${lh}`);
    }
    const lNameLen = buf.readUInt16LE(lh + 26);
    const lExtraLen = buf.readUInt16LE(lh + 28);
    const dataStart = lh + 30 + lNameLen + lExtraLen;
    if (dataStart + e.csize > buf.length) {
      fail(`${label}: entry "${e.name}" runs past the end of the file`);
    }
    e.dataStart = dataStart;
    e.compressed = buf.subarray(dataStart, dataStart + e.csize);
    entries.push(e);
  }
  return entries;
}

function entryData(entry, label) {
  if (entry.method === 0) return Buffer.from(entry.compressed);
  if (entry.method === 8) {
    try {
      return zlib.inflateRawSync(entry.compressed);
    } catch (e) {
      fail(`${label}: entry "${entry.name}" does not inflate: ${e.message}`);
    }
  }
  return fail(`${label}: entry "${entry.name}" uses unsupported compression method ${entry.method}`);
}

// ---------------------------------------------------------------- zip writer

function dosDateTime(d) {
  const time = (d.getHours() << 11) | (d.getMinutes() << 5) | (d.getSeconds() >> 1);
  const date = ((d.getFullYear() - 1980) << 9) | ((d.getMonth() + 1) << 5) | d.getDate();
  return { time, date };
}

function writeZip(entries) {
  const locals = [];
  const centrals = [];
  let offset = 0;

  for (const e of entries) {
    const name = Buffer.from(e.name, 'utf8');
    if (e.csize > 0xfffffffe || e.usize > 0xfffffffe) {
      fail(`entry "${e.name}" is too large for a non-zip64 archive`);
    }
    const local = Buffer.alloc(30);
    local.writeUInt32LE(SIG_LOCAL, 0);
    local.writeUInt16LE(e.versionNeeded || 20, 4);
    local.writeUInt16LE(e.flags === undefined ? 0x0800 : e.flags, 6);
    local.writeUInt16LE(e.method, 8);
    local.writeUInt16LE(e.time, 10);
    local.writeUInt16LE(e.date, 12);
    local.writeUInt32LE(e.crc, 14);
    local.writeUInt32LE(e.csize, 18);
    local.writeUInt32LE(e.usize, 22);
    local.writeUInt16LE(name.length, 26);
    local.writeUInt16LE(0, 28);
    locals.push(local, name, e.compressed);

    const central = Buffer.alloc(46);
    central.writeUInt32LE(SIG_CENTRAL, 0);
    central.writeUInt16LE(e.versionMadeBy === undefined ? 0x0314 : e.versionMadeBy, 4);
    central.writeUInt16LE(e.versionNeeded || 20, 6);
    central.writeUInt16LE(e.flags === undefined ? 0x0800 : e.flags, 8);
    central.writeUInt16LE(e.method, 10);
    central.writeUInt16LE(e.time, 12);
    central.writeUInt16LE(e.date, 14);
    central.writeUInt32LE(e.crc, 16);
    central.writeUInt32LE(e.csize, 20);
    central.writeUInt32LE(e.usize, 24);
    central.writeUInt16LE(name.length, 28);
    central.writeUInt16LE(0, 30);           // extra
    central.writeUInt16LE(0, 32);           // comment
    central.writeUInt16LE(0, 34);           // disk
    central.writeUInt16LE(0, 36);           // internal attributes
    central.writeUInt32LE(e.externalAttrs === undefined ? (0o100644 << 16) >>> 0 : e.externalAttrs, 38);
    central.writeUInt32LE(offset, 42);
    centrals.push(central, name);

    offset += local.length + name.length + e.compressed.length;
  }

  const cd = Buffer.concat(centrals);
  if (entries.length > 0xfffe) fail('too many entries for a non-zip64 archive');
  const eocd = Buffer.alloc(22);
  eocd.writeUInt32LE(SIG_EOCD, 0);
  eocd.writeUInt16LE(0, 4);
  eocd.writeUInt16LE(0, 6);
  eocd.writeUInt16LE(entries.length, 8);
  eocd.writeUInt16LE(entries.length, 10);
  eocd.writeUInt32LE(cd.length, 12);
  eocd.writeUInt32LE(offset, 16);
  eocd.writeUInt16LE(0, 20);
  return Buffer.concat([...locals, cd, eocd]);
}

function deflatedEntry(name, data, when) {
  const deflated = zlib.deflateRawSync(data, { level: 9 });
  const method = deflated.length < data.length ? 8 : 0;
  const payload = method === 8 ? deflated : data;
  return {
    name,
    method,
    flags: 0x0800,
    time: when.time,
    date: when.date,
    crc: crc32(data),
    csize: payload.length,
    usize: data.length,
    externalAttrs: (0o100644 << 16) >>> 0,
    versionMadeBy: 0x0314,
    versionNeeded: 20,
    compressed: payload,
  };
}

// ---------------------------------------------------------------- self-test

function selfTest() {
  const cases = [
    {
      what: 'clean tree + log + manifest',
      names: ['moshloom-abc/index.html', 'moshloom-abc/gpu.js', 'moshloom-abc/logs/a.log', 'moshloom-abc/MANIFEST.txt'],
      expect: 'accept',
    },
    { what: 'node_modules/x', names: ['moshloom-abc/node_modules/x'], expect: 'reject' },
    { what: 'a/out/y', names: ['moshloom-abc/a/out/y'], expect: 'reject' },
    // `about` is a different segment: rejecting it would be a substring bug, not a check.
    { what: 'about/x', names: ['moshloom-abc/about/x'], expect: 'accept' },
    { what: 'nested a/node_modules/b/c', names: ['moshloom-abc/a/node_modules/b/c'], expect: 'reject' },
    { what: 'root out/x', names: ['out/x'], expect: 'reject' },
    { what: 'trailing segment pkg/out', names: ['moshloom-abc/pkg/out'], expect: 'reject' },
    { what: 'lookalike node_modulesx/y', names: ['moshloom-abc/node_modulesx/y'], expect: 'accept' },
    { what: 'lookalike logs/out.log', names: ['moshloom-abc/logs/out.log'], expect: 'accept' },
    { what: 'backslash a\\out\\y', names: ['moshloom-abc\\a\\out\\y'], expect: 'reject' },
    { what: 'uppercase OUT/y', names: ['moshloom-abc/OUT/y'], expect: 'reject' },
    { what: 'duplicate entry', names: ['moshloom-abc/x', 'moshloom-abc/x'], expect: 'reject' },
  ];

  let pass = 0;
  const out = [];
  for (const c of cases) {
    const r = validateEntries(c.names);
    const got = r.ok ? 'accept' : 'reject';
    const ok = got === c.expect;
    if (ok) pass++;
    const detail = r.problems.length ? ` (${r.problems.map((p) => p.reason).join('; ')})` : '';
    out.push(`  ${ok ? 'PASS' : 'FAIL'} ${c.what} -> ${got}${detail}`);
  }
  out.push(`self-test ${pass === cases.length ? 'OK' : 'FAILED'} (${pass}/${cases.length} assertions)`);
  console.log(out.join('\n'));
  return pass === cases.length;
}

// ---------------------------------------------------------------- manifest

function makeManifest({ sha, commitLine, createdIso, prefix, logs, treeCount }) {
  const lines = [
    'moshloom milestone archive',
    `created: ${createdIso}`,
    `sha:     ${sha}`,
    `commit:  ${commitLine}`,
    `prefix:  ${prefix}/`,
    `entries: ${treeCount + logs.length + 1} (tree ${treeCount}, logs ${logs.length}, manifest 1)`,
    'logs:',
  ];
  for (const l of logs) {
    lines.push(`  ${l.size} bytes  sha256 ${l.sha256}  ${PREFIX_NAME}/${l.name}`);
  }
  lines.push('');
  return lines.join('\n');
}

// ---------------------------------------------------------------- args

function parseArgs(argv) {
  const opts = { sha: null, out: null, checkPack: false, selfTest: false, logs: [], help: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--self-test') opts.selfTest = true;
    else if (a === '--check-pack') opts.checkPack = true;
    else if (a === '--help' || a === '-h') opts.help = true;
    else if (a === '--sha' || a === '--out') {
      const v = argv[++i];
      if (v === undefined) fail(`${a} needs a value`);
      if (a === '--sha') opts.sha = v; else opts.out = v;
    } else if (a.startsWith('--sha=')) opts.sha = a.slice(6);
    else if (a.startsWith('--out=')) opts.out = a.slice(6);
    else if (a.startsWith('--')) fail(`unknown option ${a}`);
    else opts.logs.push(a);
  }
  return opts;
}

const USAGE = `usage: npm run zip -- --sha <rev> [--out <path>] [--check-pack] <log> [<log> ...]
       node scripts/zip.mjs --self-test

  --sha <rev>     revision to archive; resolved to a full SHA (required)
  --out <path>    archive path, default out/moshloom-<shortsha>.zip
  --check-pack    also rebuild index.html with "python pack.py" in a temporary worktree of
                  <rev> and fail unless it matches the committed index.html byte for byte
  --self-test     unit-test the entry-name validator and exit`;

// ---------------------------------------------------------------- check-pack

function checkPack({ sha, treeEntries, tmp }) {
  const wt = path.join(tmp, 'packwt');
  const nodeModules = path.join(root, 'node_modules');
  const link = path.join(wt, 'node_modules');
  let added = false;
  try {
    git(['worktree', 'add', '--detach', wt, sha]);
    added = true;
    if (!fs.existsSync(nodeModules)) {
      fail(`--check-pack needs node_modules in ${root} because pack.py inlines the muxers`);
    }
    fs.symlinkSync(nodeModules, link, process.platform === 'win32' ? 'junction' : 'dir');

    const py = process.platform === 'win32' ? ['python', ['pack.py']] : ['python3', ['pack.py']];
    const r = spawnSync(py[0], py[1], { cwd: wt, encoding: 'utf8' });
    if (r.error) fail(`--check-pack could not run ${py[0]}: ${r.error.message}`);
    if (r.status !== 0) {
      const msg = (r.stderr || r.stdout || '').trim();
      fail(`--check-pack: ${py[0]} pack.py failed in the temporary worktree (${r.status})${msg ? ': ' + msg : ''}`);
    }

    const produced = fs.readFileSync(path.join(wt, 'index.html'));
    const entry = treeEntries.find((e) => e.name === 'index.html' || e.name.endsWith('/index.html'));
    const shipped = entryData(entry, 'tree');
    const norm = (b) => Buffer.from(b.toString('latin1').replace(/\r\n/g, '\n'), 'latin1');
    if (Buffer.compare(produced, shipped) === 0) {
      console.log(`check-pack: index.html ${bytes(produced.length)} bytes, sha256 ${sha256(produced).slice(0, 16)}... matches pack.py output (byte-identical)`);
      return { mode: 'byte-identical', size: produced.length, sha256: sha256(produced) };
    }
    if (Buffer.compare(norm(produced), norm(shipped)) === 0) {
      console.log(`check-pack: index.html matches pack.py output after newline normalization (${bytes(produced.length)} vs ${bytes(shipped.length)} bytes: pack.py writes native line endings)`);
      return { mode: 'newline-normalized', size: shipped.length, sha256: sha256(shipped) };
    }
    let at = 0;
    while (at < produced.length && at < shipped.length && produced[at] === shipped[at]) at++;
    fail(`--check-pack: the archive's index.html differs from pack.py's output at ${sha.slice(0, 7)} ` +
      `(archive ${bytes(shipped.length)} bytes sha256 ${sha256(shipped).slice(0, 16)}..., ` +
      `pack.py ${bytes(produced.length)} bytes sha256 ${sha256(produced).slice(0, 16)}..., first difference at byte ${at})`);
  } finally {
    // Unlink the node_modules link before the worktree goes away: a recursive delete through a
    // Windows junction takes the target with it, and that target is this checkout's real
    // node_modules. Measured once the hard way; never again.
    let lst = null;
    try { lst = fs.lstatSync(link); } catch { /* never created */ }
    if (lst && lst.isSymbolicLink()) fs.unlinkSync(link);
    // anything else left at that path was created inside the temp directory, so it is ours
    if (added) {
      const rm = spawnSync('git', ['worktree', 'remove', '--force', wt], { cwd: root, encoding: 'utf8' });
      if (rm.status !== 0) {
        // keep the failure reason (if any) intact, but never leave a worktree registered
        try { fs.rmSync(wt, { recursive: true, force: true }); } catch { /* reported below */ }
        spawnSync('git', ['worktree', 'prune'], { cwd: root, encoding: 'utf8' });
        console.error(`zip.mjs: note: \`git worktree remove\` reported ${rm.status}; removed the directory and pruned instead`);
      }
    }
    if (added && !fs.existsSync(nodeModules)) {
      console.error(`zip.mjs: warning: ${nodeModules} is gone after the pack check; run npm ci`);
    }
  }
}

// ---------------------------------------------------------------- main

function main(argv) {
  const opts = parseArgs(argv);
  if (opts.help) {
    console.log(USAGE);
    return 0;
  }
  if (opts.selfTest) return selfTest() ? 0 : 1;

  if (!opts.sha) fail('--sha <rev> is required\n\n' + USAGE);
  if (opts.logs.length === 0) fail('no log files given; pass at least one test log\n\n' + USAGE);

  // resolve the revision to a full SHA first, so a typo never produces an archive
  if (opts.sha.startsWith('-')) fail(`--sha "${opts.sha}" looks like an option, not a revision`);
  const resolved = spawnSync('git', ['rev-parse', '--verify', '--quiet', `${opts.sha}^{commit}`], {
    cwd: root, encoding: 'utf8',
  });
  if (resolved.status !== 0 || !resolved.stdout.trim()) {
    fail(`--sha "${opts.sha}" does not resolve to a commit in ${root}`);
  }
  const sha = resolved.stdout.trim();
  if (!/^[0-9a-f]{40}$/.test(sha)) fail(`--sha "${opts.sha}" resolved to "${sha}", not a full SHA`);

  const shortsha = sha.slice(0, 7);
  const prefix = `moshloom-${shortsha}`;
  const outPath = path.resolve(root, opts.out || path.join('out', `${prefix}.zip`));

  // Everything below cleans up after itself: a temp directory for git archive / the pack
  // worktree, and the output path, which must not exist after a failed run.
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'moshloom-zip-'));
  let wroteOut = false;
  try {
    // logs: reject missing and empty files before anything is written
    const logs = [];
    const problems = [];
    for (const given of opts.logs) {
      const abs = path.resolve(process.cwd(), given);
      if (!fs.existsSync(abs) || !fs.statSync(abs).isFile()) { problems.push(`${given}: not a file`); continue; }
      const n = fs.statSync(abs).size;
      if (n === 0) { problems.push(`${given}: 0 bytes`); continue; }
      const name = path.basename(abs);
      if (logs.some((l) => l.name === name)) { problems.push(`${given}: duplicate log basename "${name}"`); continue; }
      logs.push({ name, path: abs, size: n });
    }
    if (problems.length) fail('bad log arguments (nothing archived):\n  ' + problems.join('\n  '));

    // 1. the tree, straight from git
    const treeZipPath = path.join(tmp, 'tree.zip');
    git(['archive', '--format=zip', `--prefix=${prefix}/`, '-o', treeZipPath, sha]);
    const treeZip = fs.readFileSync(treeZipPath);
    const treeEntries = readZip(treeZip, 'git archive');

    const v = validateEntries(treeEntries.map((e) => e.name));
    if (!v.ok) {
      fail('the tree at ' + shortsha + ' carries entries this archive must not ship:\n  ' +
        v.problems.map((p) => `${p.name}: ${p.reason}`).join('\n  '));
    }
    if (!treeEntries.some((e) => e.name === `${prefix}/index.html`)) {
      fail(`the tree at ${shortsha} has no ${prefix}/index.html; pack it before archiving`);
    }

    // 2. manifest input: git's own subject line, and the creation time
    const commitLine = git(['log', '-1', '--format=%H %s', sha]).trim();
    const created = new Date();
    const createdIso = created.toISOString();
    const when = dosDateTime(created);

    // 3. logs, read once: the size and sha256 in the manifest describe the bytes we ship
    const logEntries = [];
    const manifestLogs = [];
    for (const l of logs) {
      const data = fs.readFileSync(l.path);
      if (data.length === 0) fail(`${l.path}: 0 bytes`);
      logEntries.push(deflatedEntry(`${prefix}/${PREFIX_NAME}/${l.name}`, data, when));
      manifestLogs.push({ name: l.name, size: data.length, sha256: sha256(data) });
    }

    manifestLogs.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
    const finalManifest = makeManifest({
      sha, commitLine, createdIso, prefix,
      logs: manifestLogs, treeCount: treeEntries.length,
    });

    const allEntries = [
      ...treeEntries.map((e) => ({
        name: e.name, method: e.method, flags: e.flags, time: e.time, date: e.date,
        crc: e.crc, csize: e.csize, usize: e.usize, externalAttrs: e.externalAttrs,
        versionMadeBy: e.versionMadeBy, versionNeeded: e.versionNeeded, compressed: e.compressed,
      })),
      ...logEntries,
      deflatedEntry(`${prefix}/MANIFEST.txt`, Buffer.from(finalManifest, 'utf8'), when),
    ];

    const vFinal = validateEntries(allEntries.map((e) => e.name));
    if (!vFinal.ok) {
      fail('refusing to write an archive with bad entry names:\n  ' +
        vFinal.problems.map((p) => `${p.name}: ${p.reason}`).join('\n  '));
    }

    // 4. the pack check runs before the archive exists, so a failure leaves nothing behind
    const pack = opts.checkPack ? checkPack({ sha, treeEntries, tmp }) : null;

    // 5. write, then read the file back and check it as a stranger would
    fs.mkdirSync(path.dirname(outPath), { recursive: true });
    const zipBuf = writeZip(allEntries);
    fs.writeFileSync(outPath, zipBuf);
    wroteOut = true;

    const back = fs.readFileSync(outPath);
    const readBack = readZip(back, path.basename(outPath));
    const bad = validateEntries(readBack.map((e) => e.name));
    if (!bad.ok) {
      fail('read-back found bad entries in ' + outPath + ':\n  ' +
        bad.problems.map((p) => `${p.name}: ${p.reason}`).join('\n  '));
    }
    if (readBack.length !== allEntries.length) {
      fail(`read-back has ${readBack.length} entries, expected ${allEntries.length}`);
    }
    let checked = 0;
    for (const e of readBack) {
      if (!e.name.startsWith(prefix + '/')) fail(`read-back entry "${e.name}" is outside ${prefix}/`);
      const data = entryData(e, path.basename(outPath));
      if (data.length !== e.usize) fail(`read-back entry "${e.name}" is ${data.length} bytes, header says ${e.usize}`);
      if (crc32(data) !== e.crc) fail(`read-back entry "${e.name}" fails its CRC`);
      checked++;
    }

    // the manifest must describe the bytes actually in the archive
    const manifestEntry = readBack.find((e) => e.name === `${prefix}/MANIFEST.txt`);
    if (!manifestEntry) fail('read-back has no MANIFEST.txt');
    const manifestBack = entryData(manifestEntry, 'manifest').toString('utf8');
    if (!manifestBack.includes(sha)) fail('MANIFEST.txt does not contain the full SHA');
    if (!manifestBack.includes(commitLine)) fail('MANIFEST.txt does not contain the commit line');
    for (const l of manifestLogs) {
      const entry = readBack.find((e) => e.name === `${prefix}/${PREFIX_NAME}/${l.name}`);
      if (!entry) fail(`read-back has no logs/${l.name}`);
      const data = entryData(entry, 'log');
      if (data.length !== l.size) fail(`logs/${l.name}: ${data.length} bytes, manifest says ${l.size}`);
      if (sha256(data) !== l.sha256) fail(`logs/${l.name}: sha256 does not match the manifest`);
      if (!manifestBack.includes(`${l.size} bytes`) || !manifestBack.includes(l.sha256) ||
          !manifestBack.includes(`${PREFIX_NAME}/${l.name}`)) {
        fail(`MANIFEST.txt does not describe logs/${l.name}`);
      }
    }

    const logBytes = manifestLogs.reduce((n, l) => n + l.size, 0);
    console.log(`zip: ${path.relative(root, outPath).replace(/\\/g, '/')} ${bytes(back.length)} bytes`);
    console.log(`sha: ${sha} (${commitLine.split(' ').slice(1).join(' ')})`);
    console.log(`entries: ${readBack.length} (tree ${treeEntries.length}, logs ${logEntries.length}, manifest 1), logs ${bytes(logBytes)} bytes`);
    console.log(`read back: ${checked}/${readBack.length} entries inflated, CRC ok; 0 forbidden segments, 0 duplicates, all under ${prefix}/`);
    console.log(`logs: ${manifestLogs.length}/${logs.length} size + sha256 match the manifest`);
    if (pack) console.log(`check-pack: PASS (${pack.mode})`);
    return 0;
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
    if (!wroteOut && fs.existsSync(outPath)) {
      // a stale zip from an earlier run would let exit != 0 sit next to an archive
      fs.rmSync(outPath, { force: true });
      console.error(`zip.mjs: removed stale ${path.relative(root, outPath).replace(/\\/g, '/')}; a failed run leaves no archive behind`);
    }
  }
}

let code = 1;
try {
  code = main(process.argv.slice(2));
} catch (e) {
  if (e instanceof Fail) {
    console.error('zip.mjs: ' + e.message);
    code = 1;
  } else {
    console.error('zip.mjs: unexpected failure: ' + (e && e.stack ? e.stack : e));
    code = 1;
  }
}
process.exitCode = code;
