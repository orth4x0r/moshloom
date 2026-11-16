// ---------- WebGPU renderer: scenes, motion estimation, mosh feedback, output ----------
// ---- uniform layouts. These tables are the only description of SU and FU: the WGSL
// struct text, the buffer byte sizes and every write offset below are derived from them.
const WG_LAYOUT={ f32:{a:4,s:4}, vec2f:{a:8,s:8}, vec4f:{a:16,s:16}, Op:{a:16,s:32} };
function layout(fields){
  let off=0; const at={};
  for(const f of fields){ const t=WG_LAYOUT[f.type]; off=Math.ceil(off/t.a)*t.a; at[f.name]=off/4; off+=t.s*(f.count||1); }
  const bytes=Math.ceil(off/16)*16; return {at,bytes,f32:bytes/4,fields};
}
function structText(name,L){
  return `struct ${name} { ${L.fields.map(f=>`${f.name}: ${f.count?`array<${f.type},${f.count}>`:f.type}`).join(', ')} };`;
}
const SU=layout([
  {name:'res',type:'vec2f'}, {name:'T',type:'f32'}, {name:'hue',type:'f32'}, {name:'bass',type:'f32'},
  {name:'mid',type:'f32'}, {name:'high',type:'f32'}, {name:'pulse',type:'f32'}, {name:'vari',type:'f32'},
  {name:'dens',type:'f32'}, {name:'beat',type:'f32'}, {name:'frame',type:'f32'}, {name:'pal',type:'f32'},
  {name:'aux0',type:'f32'}, {name:'aux1',type:'f32'}, {name:'aux2',type:'f32'},
  {name:'l0',type:'vec4f'}, {name:'l1',type:'vec4f'}, {name:'l2',type:'vec4f'},
  {name:'mel',type:'vec4f',count:6},
]);
const WG_SU=`
${structText('SU',SU)}
@group(0) @binding(0) var<uniform> U: SU;
const PI = 3.14159265; const TAU = 6.2831853;
fn h1(n: f32) -> f32 { return fract(sin(n * 127.1) * 43758.5453); }
fn h2(p: vec2f) -> f32 { return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453); }
`;
const WG_HEAD=WG_SU+`
@group(0) @binding(1) var lin: sampler;
@group(0) @binding(2) var palT: texture_2d<f32>;
@group(0) @binding(3) var caT: texture_2d<f32>;
@group(0) @binding(4) var melT: texture_2d<f32>;
@group(0) @binding(5) var rdT: texture_2d<f32>;
@group(0) @binding(6) var mediaT: texture_2d<f32>;
fn lut(v: f32) -> vec3f { return textureSampleLevel(palT, lin, vec2f(v, (U.pal + 0.5) / 8.), 0.).rgb; }
fn mel(i: i32) -> f32 { let k = ((i % 24) + 24) % 24; return U.mel[k / 4][k % 4]; }
fn melh(band: f32, age: f32) -> f32 { return textureSampleLevel(melT, lin, vec2f((clamp(band, 0., 23.) + 0.5) / 24., (clamp(age, 0., 127.) + 0.5) / 128.), 0.).r * 1.3; }
fn XY(p: vec4f) -> vec2f { return (p.xy - U.res * 0.5) / (U.res.y * 0.5); }
fn ca(x: i32, y: i32, r: i32) -> f32 { return textureLoad(caT, vec2i(x & 511, y & 511), 0)[r]; }
fn rule() -> i32 { return ((i32(U.vari) % 3) + 3) % 3; }
fn fin(c: vec3f) -> vec4f { return vec4f(clamp(c, vec3f(0.), vec3f(1.)), 1.); }
`;
const WG_VS=`@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { return vec4f(f32((i << 1u) & 2u) * 2. - 1., f32(i & 2u) * 2. - 1., 0., 1.); }
`;
const FS=b=>WG_HEAD+WG_VS+b;
const WG_SCENES=[
// 0 tunnel
FS(`@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos); let r = length(q) + 1e-4; let th = atan2(q.y, q.x); let n = round(2. + U.dens * 8.);
  let u = 0.35 / r + U.T * 0.8;
  let pp = sin(TAU * u) * cos(n * th + TAU * (0.3 * sin(u * 1.5 + U.T) + U.T * 0.1));
  var c = lut(u * 0.22 + 0.16 * pp + U.hue);
  let m = select(0.55, 1., sin(25.133 * u) > 0.) * pow(min(r * 2., 1.), 1.25);
  let g = max(0., 1. - r * r * 8.);
  return fin(c * m + g * g * U.pulse * U.bass); }`),
// 1 loom
FS(`@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos); let zoom = (13. + 6. * sin(U.T * 0.21)) * (0.5 + U.dens); let a = 0.25 * sin(U.T * 0.07); let cs = cos(a); let sn = sin(a);
  let g = vec2f((q.x * cs - q.y * sn) * zoom + U.T * 1.5, (q.x * sn + q.y * cs) * zoom + U.T * 0.4);
  let ig = floor(g); let f = g - ig; let ix = i32(ig.x); let iy = i32(ig.y); let sx = sin(PI * f.x); let sy = sin(PI * f.y);
  var c: vec3f;
  if (ca(ix, iy + i32(U.T * 3.), rule()) > 0.5) { c = lut(f32(ix) * 0.013 + U.hue) * (0.35 + mel(ix)) * (0.25 + 0.75 * sx) * (0.6 + 0.4 * sy); }
  else { c = lut(f32(iy) * 0.021 + U.hue + 0.5) * 0.8 * (0.25 + 0.75 * sy) * (0.6 + 0.4 * sx); }
  return fin(c); }`),
// 2 numogram
FS(`const ZP = array<vec2f, 10>(vec2f(-.25,.75), vec2f(-.55,0.), vec2f(-.3,.35), vec2f(-.25,-.75), vec2f(-.3,-.35), vec2f(.3,-.35), vec2f(.25,-.75), vec2f(.3,.35), vec2f(.55,0.), vec2f(.25,.75));
const LP = array<i32, 6>(1, 2, 7, 8, 5, 4);
const SZ = array<vec2i, 5>(vec2i(0,9), vec2i(1,8), vec2i(2,7), vec2i(3,6), vec2i(4,5));
const GT = array<vec2i, 6>(vec2i(2,3), vec2i(4,1), vec2i(5,6), vec2i(7,1), vec2i(8,9), vec2i(6,3));
fn seg(p: vec2f, a: vec2f, b: vec2f) -> f32 { let pa = p - a; let ba = b - a; let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-9), 0., 1.); return length(pa - ba * h); }
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos); let r = length(q) + 1e-4; let px = 2. / U.res.y; let ss = U.res.x / 640.;
  let tw = f32(i32(U.vari) % 4);
  let rg = step(0.96, sin(log(r) * 14. - U.T * 1.5 + atan2(q.y, q.x) * tw));
  var c = rg * (0.10 * U.pulse * U.l1.rgb + 0.18 * U.l0.rgb);
  var zp = ZP; var lp = LP; var sz = SZ; var gt = GT; let ph = fract(U.T * 0.13);
  for (var k = 0; k < 5; k++) {
    let p = (ph + f32(k)) / 5.; let s = 0.12 * exp2(p * 5.5) * (0.5 + U.dens); let b = pow(sin(PI * p), 1.5);
    let ang = U.T * 0.11 * select(-1., 1., (k & 1) == 1) + f32(k); let cs = cos(ang); let sn = sin(ang);
    var pt: array<vec2f, 10>;
    for (var z = 0; z < 10; z++) { pt[z] = vec2f(zp[z].x * cs - zp[z].y * sn, zp[z].x * sn + zp[z].y * cs) * s; }
    let th = select(2.2, 1.2, s < 1.2) * ss * px;
    var d0 = 1e9; var d1 = 1e9; var d2 = 1e9;
    for (var i = 0; i < 6; i++) { d0 = min(d0, seg(q, pt[lp[i]], pt[lp[(i + 1) % 6]])); d2 = min(d2, seg(q, pt[gt[i].x], pt[gt[i].y])); }
    for (var i = 0; i < 5; i++) { d1 = min(d1, seg(q, pt[sz[i].x], pt[sz[i].y])); }
    c = max(c, U.l0.rgb * b * clamp((th * 0.5 - d0) / px + 0.5, 0., 1.));
    c = max(c, U.l2.rgb * b * 0.8 * clamp((ss * px * 0.5 - d2) / px + 0.5, 0., 1.));
    c = max(c, U.l1.rgb * b * clamp((th * 0.5 - d1) / px + 0.5, 0., 1.));
    for (var z = 0; z < 10; z++) {
      let e = mel((z * 2 + 3) % 24); let rad = s * (0.05 + 0.07 * e); let dd = length(q - pt[z]);
      let ain = clamp((rad - dd) / px + 0.5, 0., 1.); c = c * (1. - ain);
      if (e > 0.75) { c = mix(c, vec3f(b), ain); } else { c = max(c, U.l0.rgb * b * clamp((th * 0.5 - abs(dd - rad)) / px + 0.5, 0., 1.)); }
    }
  }
  return fin(c); }`),
// 3 flesh
FS(`@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos); var f = 0.;
  for (var k = 0; k < 6; k++) { let fk = f32(k); let cc = vec2f(1.2 * sin(U.T * (0.23 + 0.07 * fk) + fk * 2.1), 0.7 * cos(U.T * (0.19 + 0.05 * fk) + fk * 1.3));
    let w = (0.05 + 0.03 * fk + 0.06 * U.bass) * (0.4 + 1.2 * U.dens); let d = q - cc; f += w / (dot(d, d) + 0.01); }
  let v = log(1. + f) * 0.33 - U.T * 0.07 + U.hue * 0.3;
  return fin(lut(v) * select(0.7, 1., sin(v * 37.7) > 0.)); }`),
// 4 rotor
FS(`@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos); let a = U.T * 1.1; let cs = cos(a); let sn = sin(a); let fq = (5. + 9. * U.mid) * (0.4 + 1.2 * U.dens);
  let st = sin((q.x * cs + q.y * sn) * fq) * sin((-q.x * sn + q.y * cs) * 2.5 + U.T * 2.);
  var c = select(lut(U.hue * 2. + 0.55) * 0.9, lut(U.hue * 2. + 0.13), st > 0.);
  if (abs(length(q) - (0.3 + 0.9 * fract(U.T * 0.5))) < 0.04) { c = 1. - c; }
  return fin(c); }`),
// 5 moire
FS(`@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos); let cc = vec2f(0.6 * sin(U.T * 0.31), 0.35 * cos(U.T * 0.43)); let fq = (38. + 22. * U.mid) * (0.4 + 1.2 * U.dens);
  let v = (sin(length(q - cc) * fq - U.T * 3.) + sin(length(q + cc) * fq + U.T * 3.)) * 0.25 + 0.5;
  return fin(lut(v * 0.5 + U.hue) * select(0.4, 1., v > 0.5)); }`),
// 6 spectrum
FS(`@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos); let asp = U.res.x / U.res.y; var bf: f32; var age: f32; let sc = 0.5 + U.dens;
  if ((i32(U.vari) & 1) == 1) { bf = abs(atan2(q.y, q.x)) / PI * 23.; age = length(q) * 70. * sc; } else { bf = abs(q.x) / asp * 23.; age = (q.y + 1.) * 60. * sc; }
  age = floor(min(age, 127.)); let v = melh(bf, age);
  return fin(lut(v * 0.45 + U.hue + age * 0.004) * min(1., v * v * 1.4)); }`),
// 7 cells
FS(`@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos) * (0.6 + 0.8 * U.dens); var d1 = 1e9; var d2 = 1e9; var i1 = 0;
  for (var k = 0; k < 14; k++) { let fk = f32(k); let d = q - vec2f(1.6 * sin(U.T * (0.11 + 0.03 * fk) + fk * 1.7), 0.9 * cos(U.T * (0.13 + 0.02 * fk) + fk * 2.9)); let e = dot(d, d);
    if (e < d1) { d2 = d1; d1 = e; i1 = k; } else if (e < d2) { d2 = e; } }
  d1 = sqrt(d1); d2 = sqrt(d2);
  var c = lut(f32(i1) * 0.071 + d1 * 0.9 - U.T * 0.05 + U.hue) * select(select(0.6, 1., sin(d1 * 40. - U.T * 4.) > 0.), 0.08, d2 - d1 < 0.03);
  if (d1 < 0.05 * (1. + 2. * U.bass * U.pulse)) { c = vec3f(1.); }
  return fin(c); }`),
// 8 punch cards
FS(`@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos); let asp = U.res.x / U.res.y; let cols = round(12. + U.dens * 24.); let cw = 2. * asp / cols;
  let g = vec2f((q.x + asp) / cw, q.y / (cw * 1.25) + U.T * 1.4); let ig = floor(g); let f = g - ig - 0.5; let ix = i32(ig.x); let iy = i32(ig.y);
  let row = ((iy % 8) + 8) % 8; let card = floor(f32(iy) / 8.);
  var c = (lut(U.hue + card * 0.09 + 0.1) * 0.28 + 0.06) * (0.85 + 0.15 * h2(floor(pos.xy / 2.)));
  if (row == 7) { c = vec3f(0.01); if (((ix % 6) + 6) % 6 == 0 && length(f) < 0.28) { c = U.l1.rgb * (0.3 + U.pulse); } }
  else { let on = ca(ix + i32(card) * 7, iy, rule()) > 0.5; let hole = max(abs(f.x) - 0.26, abs(f.y) - 0.34);
    if (on && hole < 0.) { let e = mel(i32(f32(ix) * 24. / cols)); c = lut(U.hue + 0.5 + e * 0.3 + card * 0.05) * (0.25 + 1.3 * e) + vec3f(exp(-abs(q.y) * 5.) * U.pulse); }
    else if (on && hole < 0.07) { c *= 0.4; } }
  return fin(c); }`),
// 9 vortex
FS(`@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos); let r = length(q) + 1e-4; let th = atan2(q.y, q.x); let n = round(2. + U.dens * 6.); let lr = log(r);
  let spin = 2.5 * sin(U.beat * PI * 0.25) + U.T * 0.3;
  let v = sin(th * n - lr * 8. + spin * n + 0.8 * sin(lr * 3. - U.T));
  var c = lut(lr * 0.35 - U.T * 0.2 + U.hue + select(0., 0.27, v > 0.));
  c *= (0.5 + 0.5 * step(0., sin(lr * 20. - U.T * 6.))) * smoothstep(0.02, 0.18, r);
  return fin(c); }`),
// 10 kaleidoscope
FS(`@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos); let r = length(q); let n = round(3. + U.dens * 10.); let sg = TAU / n;
  var a = atan2(q.y, q.x) + U.T * 0.1; a = abs((a - sg * floor(a / sg)) - sg * 0.5);
  var p = r * vec2f(cos(a), sin(a)); p += 0.3 * sin(p.yx * 3. + U.T * vec2f(0.7, 0.9)); p *= 2. + sin(U.T * 0.2) + U.bass * 0.5;
  let v = sin(p.x * 4. + U.T) + sin(p.y * 5. - U.T * 1.3) + sin((p.x + p.y) * 3. + U.T * 0.5) + sin(length(p) * 6. - U.T * 2.);
  return fin(lut(v * 0.18 + U.hue + r * 0.2) * (0.55 + 0.45 * step(0., sin(v * 6.)))); }`),
// 11 reaction-diffusion (display; the simulation is WG_RDSIM)
FS(`@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let uv = pos.xy / U.res; let e = vec2f(1. / 320., 1. / 180.);
  let v = textureSampleLevel(rdT, lin, uv, 0.).g;
  let gx = textureSampleLevel(rdT, lin, uv + vec2f(e.x, 0.), 0.).g - textureSampleLevel(rdT, lin, uv - vec2f(e.x, 0.), 0.).g;
  let gy = textureSampleLevel(rdT, lin, uv + vec2f(0., e.y), 0.).g - textureSampleLevel(rdT, lin, uv - vec2f(0., e.y), 0.).g;
  var c = lut(v * 1.5 + U.hue * 0.3 + 0.08) * smoothstep(0.03, 0.22, v) + lut(U.hue * 0.3 + 0.62) * length(vec2f(gx, gy)) * 2.2;
  c += U.l0.rgb * U.pulse * U.bass * 0.25 * smoothstep(0.1, 0.3, v);
  return fin(c); }`),
// 12 cyberspace grid
FS(`@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos); let asp = U.res.x / U.res.y; let h = -0.08 + 0.04 * sin(U.T * 0.3); let dy = q.y - h; let n = (2. + U.dens * 8.) * 0.25;
  let z = 0.6 / (abs(dy) + 1e-3); let wx = q.x * z; let wz = z + U.T * 1.6;
  let gx = abs(fract(wx * n) - 0.5); let gz = abs(fract(wz * n) - 0.5);
  let ax = 1. - smoothstep(0., min(0.5, 0.004 * z * n * 4. + 0.012), gx); let az = 1. - smoothstep(0., min(0.5, 0.004 * z * z * n * 4. + 0.012), gz);
  var c = select(U.l1.rgb, U.l0.rgb, dy > 0.) * max(ax, az) * exp(-z * 0.12) * (0.7 + 0.6 * U.pulse * U.bass);
  c += U.l2.rgb * exp(-abs(dy) * 14.) * (0.4 + U.pulse * 0.8);
  let band = (q.x / asp * 0.5 + 0.5) * 24.; let sk = mel(i32(band)) * 0.24;
  if (dy < 0. && -dy < sk && fract(band) > 0.12) { c += lut(U.hue + floor(band) * 0.03) * (0.35 + 0.65 * step(0.5, fract(dy * 40.))); }
  return fin(c); }`),
// 13 glyph rain
FS(`@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos); let asp = U.res.x / U.res.y; let N = round(20. + U.dens * 60.); let cw = 2. * asp / N; let ch = cw * 1.5;
  let gx = (q.x + asp) / cw; let gy = (q.y + 1.) / ch; let ix = floor(gx); let iy = floor(gy); let rows = 2. / ch;
  let sp = 0.25 + 0.9 * h1(ix + 3.); let b = fract(gy / rows * 0.7 - U.T * sp * 0.35 - h1(ix + 17.) * 7.);
  let s = floor(fract(vec2f(gx, gy)) * vec2f(4., 6.));
  let seed = floor(U.T * 1.5 * h1(ix * 3. + iy) + h1(iy + ix * 5.) * 9.);
  var bit = step(0.5, h2(vec2f(ix * 7. + s.x + seed * 13., iy * 11. + s.y)));
  if (s.x > 2.5 || s.y > 4.5) { bit = 0.; }
  var c = U.l0.rgb * pow(b, 4.) * bit * (0.35 + mel(i32(ix))) * 1.3;
  if (b > 0.965) { c = vec3f(bit); }
  c += U.l1.rgb * bit * 0.6 * step(0.985, h2(vec2f(ix, iy) + floor(U.beat))) * U.pulse;
  return fin(c); }`),
// 14 scan interference
FS(`@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let uv = pos.xy / U.res; let y = fract(uv.y + U.T * 0.03 + 0.015 * sin(U.T * 0.7)); let ln = floor(y * 90.); let tt = floor(U.T * 3.);
  let tear = step(0.9, h2(vec2f(ln, tt))) * 0.24 * (h2(vec2f(ln, tt + 9.)) - 0.5);
  let x = fract(uv.x + tear + 0.04 * U.bass * sin(y * 9. + U.T * 3.) + 0.01 * sin(y * 140. + U.T * 8.) * U.mid);
  let nb = round(4. + U.dens * 12.); let bi = floor(x * nb);
  var c = lut(bi / nb * 0.85 + U.hue) * (0.75 + 0.25 * step(0.5, fract(bi * 0.5)));
  if (y > 0.74) { let b2 = floor(x * nb * 3.); c = vec3f(h1(b2 + floor(U.beat))) * select(U.l0.rgb, vec3f(1.), h1(b2 * 3. + 1.) > 0.5); }
  c *= 1. - 0.6 * smoothstep(0., 0.08, abs(fract(uv.y * 1.3 - U.T * 0.21) - 0.5) - 0.32);
  if (y < 0.05) { c = vec3f(0.02); if (x > 0.1 && x < 0.16) { c = vec3f(0.9); } }
  c *= 0.78 + 0.22 * sin(pos.y * PI * 360. / U.res.y);
  c += (h2(floor(pos.xy * 180. / U.res.y) + U.frame * 0.37) - 0.5) * 0.5 * (0.15 + U.high);
  c += 0.12 * sin(uv.x * 220. + uv.y * 130. + U.T * 20.) * U.mid;
  return fin(c); }`),
// 15 swarm (instanced streaks, additive)
WG_HEAD+`struct VO { @builtin(position) p: vec4f, @location(0) uv: vec2f, @location(1) col: vec3f };
fn ppos(i: f32, t: f32) -> vec2f {
  let a = h1(i) * TAU; let b = h1(i + 31.) * TAU; let f1 = 0.3 + h1(i + 7.) * 0.9; let f2 = 0.2 + h1(i + 13.) * 0.8;
  var p = vec2f(sin(t * f1 + a) + 0.6 * sin(t * f2 * 2.3 + b), cos(t * f2 + b) + 0.6 * cos(t * f1 * 1.7 + a)) * vec2f(1.1, 0.65);
  let c = vec2f(0.8 * sin(t * 0.37), 0.5 * cos(t * 0.29));
  p = mix(p, c, 0.35 + 0.45 * sin(t * 0.5 + h1(i + 3.) * 2.));
  return c + (p - c) * (1. + 1.2 * U.pulse * U.bass); }
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
  var cx = array<vec2f, 6>(vec2f(0., -1.), vec2f(1., -1.), vec2f(0., 1.), vec2f(0., 1.), vec2f(1., -1.), vec2f(1., 1.));
  let i = f32(ii); let asp = U.res.x / U.res.y; let p0 = ppos(i, U.T); let p1 = ppos(i, U.T - 0.09);
  var dir = p0 - p1; let len = length(dir); if (len > 1e-5) { dir = dir / len; } else { dir = vec2f(1., 0.); }
  let w = 0.006 + 0.012 * h1(i + 5.) + 0.01 * U.bass; let c = cx[vi];
  let p = mix(p1 - dir * w, p0 + dir * w, c.x) + vec2f(-dir.y, dir.x) * w * c.y;
  var o: VO; o.p = vec4f(p.x / asp, -p.y, 0., 1.); o.uv = c;
  o.col = (lut(h1(i + 9.) * 0.5 + U.hue) + 0.15) * (0.5 + mel(i32(ii % 24u))); return o; }
@fragment fn fs(v: VO) -> @location(0) vec4f { let a = (1. - v.uv.y * v.uv.y) * (0.25 + 0.75 * v.uv.x);
  return vec4f(v.col * a * a, 1.); }`,
// 16 media (image / video clip): cover fit, slow drift, pulse zoom; a chosen palette maps luminance through it
FS(`@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let asp = U.res.x / U.res.y; var p = pos.xy / U.res - 0.5;
  if (U.aux0 > asp) { p.x *= asp / U.aux0; } else { p.y *= U.aux0 / asp; }
  let zoom = 1. / (1.1 + 0.06 * sin(U.T * 0.3) + 0.12 * U.bass * U.pulse * U.dens * 2.);
  p = p * zoom + 0.02 * vec2f(sin(U.T * 0.2), cos(U.T * 0.17));
  var c = textureSampleLevel(mediaT, lin, p + 0.5, 0.).rgb;
  if (U.aux1 > 0.5) { c = lut(dot(c, vec3f(0.3, 0.59, 0.11)) * 0.8 + U.hue); }
  return fin(c); }`)
];
const WG_TEMPLATE=`// Custom visual. Uniforms: U.T (audio-driven time), U.hue, U.bass, U.mid, U.high, U.pulse (1 on the beat, decaying),
// U.beat, U.dens (Density slider 0..1), U.vari (Variant), U.res.  Helpers: XY(pos) centred coords, lut(v) palette colour,
// mel(i) spectrum band 0..23, h1(x)/h2(vec2) hashes, fin(rgb).
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos);
  let r = length(q);
  let v = sin(r * (10. + 30. * U.dens) - U.T * 3.) + sin(atan2(q.y, q.x) * 5. + U.T);
  let c = lut(v * 0.2 + U.hue) * (0.6 + 0.4 * U.pulse);
  return fin(c);
}`;
const WG_RDSIM=WG_SU+WG_VS+`@group(0) @binding(1) var rdT: texture_2d<f32>;
fn at(p: vec2i) -> vec2f { let sz = vec2i(320, 180); return textureLoad(rdT, (p + sz) % sz, 0).rg; }
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let p = vec2i(pos.xy); let c = at(p); let uv = pos.xy / vec2f(320., 180.);
  let lap = 0.2 * (at(p + vec2i(1, 0)) + at(p + vec2i(-1, 0)) + at(p + vec2i(0, 1)) + at(p + vec2i(0, -1)))
          + 0.05 * (at(p + vec2i(1, 1)) + at(p + vec2i(-1, 1)) + at(p + vec2i(1, -1)) + at(p + vec2i(-1, -1))) - c;
  let F = 0.022 + 0.034 * U.dens + 0.008 * sin(uv.x * 6. + U.T * 0.1) * sin(uv.y * 5.); let K = 0.060 + 0.004 * cos(uv.y * 4. - U.T * 0.07);
  let abb = c.r * c.g * c.g;
  var a = c.r + lap.r - abb + F * (1. - c.r); var b = c.g + 0.5 * lap.g + abb - (K + F) * c.g;
  if (U.aux2 > 0.5) { a = 1.; b = select(0., 1., h2(floor(pos.xy / 6.)) > 0.93); }
  if (U.aux0 >= 0. && length((uv - vec2f(U.aux0, U.aux1)) * vec2f(16. / 9., 1.)) < 0.035) { b = 1.; }
  return vec4f(clamp(a, 0., 1.), clamp(b, 0., 1.), 0., 1.); }`;
const FU=layout([
  {name:'res',type:'vec2f'}, {name:'iframe',type:'f32'}, {name:'skipres',type:'f32'}, {name:'leak',type:'f32'},
  {name:'mvScale',type:'f32'}, {name:'coarse',type:'f32'}, {name:'sh',type:'f32'}, {name:'inv',type:'f32'},
  {name:'fade',type:'f32'}, {name:'xf',type:'f32'}, {name:'frame',type:'f32'}, {name:'nops',type:'f32'},
  {name:'gain',type:'f32'}, {name:'p0',type:'f32'}, {name:'p1',type:'f32'},
  {name:'fxa',type:'vec4f'}, {name:'fxb',type:'vec4f'}, {name:'fxc',type:'vec4f'}, {name:'fxd',type:'vec4f'},
  {name:'fpa',type:'vec4f'}, {name:'fpb',type:'vec4f'}, {name:'fpc',type:'vec4f'}, {name:'fpd',type:'vec4f'},
  {name:'ops',type:'Op',count:48},
]);
// fxa = fold, mirror, zoom, spin; fxb = streak, crush, echo, negative; fxc/fxd = fluid and three spare
// fpa..fpd are the matching parameters. fxa..fxd and fpa..fpd are each adjacent, so a single write loop fills them.
const WG_FU=`
struct Op { rect: vec4f, src: vec4f };
${structText('FU',FU)}
@group(0) @binding(0) var<uniform> F: FU;
`+WG_VS;

// uniform for the fluid solver
const FB=layout([
  {name:'texel',type:'vec2f'}, {name:'dt',type:'f32'}, {name:'visc',type:'f32'},
  {name:'imp',type:'vec4f'},                       // centre x, y, strength, unused
  {name:'beat',type:'f32'}, {name:'bass',type:'f32'}, {name:'pulse',type:'f32'}, {name:'gain',type:'f32'},
  {name:'vort',type:'f32'}, {name:'pad0',type:'f32'}, {name:'pad1',type:'f32'}, {name:'pad2',type:'f32'},
]);
const WG_FB_STRUCT=`
${structText('FUB',FB)}`;
const WG_MIX=WG_FU+`@group(0) @binding(1) var aT: texture_2d<f32>; @group(0) @binding(2) var bT: texture_2d<f32>;
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f { let p = vec2i(pos.xy); return mix(textureLoad(aT, p, 0), textureLoad(bT, p, 0), F.xf); }`;
const WG_COPY=WG_VS+`@group(0) @binding(0) var aT: texture_2d<f32>;
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f { return vec4f(clamp(textureLoad(aT, vec2i(pos.xy), 0).rgb, vec3f(0.), vec3f(1.)), 1.); }`;
const WG_PRE=WG_FU+`@group(0) @binding(1) var ms: sampler; @group(0) @binding(2) var inT: texture_2d<f32>;
fn tx(uv: vec2f) -> vec3f { return textureSampleLevel(inT, ms, uv, 0.).rgb; }
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let asp = vec2f(F.res.x / F.res.y, 1.); var p = (pos.xy / F.res - 0.5) * asp;
  if (F.fxa.y > 0.) { p.x = -abs(p.x); if (F.fpa.y > 0.5) { p.y = -abs(p.y); } }
  var uv = p / asp + 0.5; var c = tx(uv);
  if (F.fxa.x > 0.) { let n = round(2. + F.fpa.x * 10.); let sg = 6.2831853 / n; let r = length(p); var a = atan2(p.y, p.x) + 1.5707963;
    a = abs((a - sg * floor(a / sg)) - sg * 0.5); let uf = r * vec2f(sin(a), -cos(a)) / asp + 0.5; c = mix(c, tx(uf), F.fxa.x); uv = mix(uv, uf, step(0.5, F.fxa.x)); }
  if (F.fxb.y > 0.) { let bs = mix(4., 40., F.fpb.y) * F.res.y / 360.; c = mix(c, tx((floor(uv * F.res / bs) + 0.5) * bs / F.res), F.fxb.y);
    let L = mix(24., 2.5, F.fxb.y); c = floor(c * L + 0.5) / L; }
  if (F.fxb.x > 0.) { let K = mix(3., 14., F.fpb.x); let s = tx(vec2f(uv.x, (floor(uv.y * K) + 0.02) / K)); if (dot(s, vec3f(0.3, 0.59, 0.11)) > 1. - F.fxb.x * 0.85) { c = s; } }
  return vec4f(c, 1.); }`;
const WG_ME=WG_FU+`@group(0) @binding(1) var ls: sampler; @group(0) @binding(2) var curT: texture_2d<f32>; @group(0) @binding(3) var prvT: texture_2d<f32>;
const GS = vec2f(320., 180.);
fn lumc(p: vec2f) -> f32 { return clamp(dot(textureSampleLevel(curT, ls, p / GS, 0.).rgb, vec3f(0.3, 0.59, 0.11)), 0., 1.); }
fn lump(p: vec2f) -> f32 { return clamp(dot(textureSampleLevel(prvT, ls, p / GS, 0.).rgb, vec3f(0.3, 0.59, 0.11)), 0., 1.); }
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let o = floor(pos.xy) * 8.; var cur: array<f32, 64>; var best = 0.;
  for (var i = 0; i < 64; i++) { let p = o + vec2f(f32(i % 8), f32(i / 8)) + 0.5; cur[i] = lumc(p); best += abs(cur[i] - lump(p)) * 255.; }
  var bd = vec2f(0.);
  if (best > 0.5) {
    for (var dy = -6; dy <= 6; dy++) { for (var dx = -6; dx <= 6; dx++) {
      if (dx == 0 && dy == 0) { continue; }
      let d = vec2f(f32(dx), f32(dy)); let lo = o + d;
      if (lo.x < 0. || lo.y < 0. || lo.x + 8. > 320. || lo.y + 8. > 184.) { continue; }
      var cost = 3. * (abs(d.x) + abs(d.y));
      for (var i = 0; i < 64; i++) { cost += abs(cur[i] - lump(lo + vec2f(f32(i % 8), f32(i / 8)) + 0.5)) * 255.; if (cost >= best) { break; } }
      if (cost < best) { best = cost; bd = d; }
    } }
  }
  let mv = clamp(bd / GS * F.gain, vec2f(-56. / 640., -56. / 360.), vec2f(56. / 640., 56. / 360.));
  return vec4f(mv, 0., 1.); }`;
const WG_FB=WG_FU+`@group(0) @binding(1) var ns: sampler; @group(0) @binding(2) var srcT: texture_2d<f32>; @group(0) @binding(3) var prvT: texture_2d<f32>;
@group(0) @binding(4) var bufT: texture_2d<f32>; @group(0) @binding(5) var mvT: texture_2d<f32>;
@group(0) @binding(6) var velT: texture_2d<f32>;
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  var uv = pos.xy / F.res; var sw = false;
  for (var i = i32(F.nops) - 1; i >= 0; i--) { let o = F.ops[i];
    if (uv.x >= o.rect.x && uv.x < o.rect.x + o.rect.z && uv.y >= o.rect.y && uv.y < o.rect.y + o.rect.w) {
      if (o.src.z == 2.) { uv = o.rect.xy + o.rect.zw * 0.5; } else { uv = vec2f(fract(uv.x + o.src.x), uv.y + o.src.y); }
      sw = o.src.z == 4.; break; } }
  let s = textureSampleLevel(srcT, ns, uv, 0.).rgb; var c = s;
  if (F.iframe < 0.5) {
    var bc = clamp(vec2i(floor(uv * vec2f(40., 22.5))), vec2i(0), vec2i(39, 22)); if (F.coarse > 0.5) { bc = bc & vec2i(-2); }
    var d = round(textureLoad(mvT, bc, 0).xy * F.mvScale * F.res) / F.res;
    if (F.fxc.x > 0.) { d += textureSampleLevel(velT, ns, uv, 0.).xy * F.fxc.x * 0.03; }   // Fluid: the solver's velocity pushes the pixels
    let asp = vec2f(F.res.x / F.res.y, 1.); var q = (uv + d - 0.5) * asp;     // feedback zoom / spin
    let an = 0.03 * F.fxa.w * select(1., -1., F.fpa.w > 0.5); let z = 1. - 0.035 * F.fxa.z * select(1., -1., F.fpa.z > 0.5);
    q = vec2f(q.x * cos(an) - q.y * sin(an), q.x * sin(an) + q.y * cos(an)) * z;
    c = textureSampleLevel(bufT, ns, q / asp + 0.5, 0.).rgb;
    if (F.skipres < 0.5) { c = c + s - textureSampleLevel(prvT, ns, uv, 0.).rgb; c = mix(c, s, F.leak); }
    if (F.fxb.z > 0.) { c = mix(c, max(c, textureSampleLevel(bufT, ns, uv, 0.).rgb * (0.72 + 0.22 * F.fpb.z)), F.fxb.z); }
    c = clamp(c, vec3f(-0.15), vec3f(1.15));
  }
  if (sw) { c = c.bgr; }
  return vec4f(c, 1.); }`;
const WG_OUT=WG_FU+`@group(0) @binding(1) var ns: sampler; @group(0) @binding(2) var bufT: texture_2d<f32>;
fn hh(p: vec2f) -> f32 { return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453); }
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let uv = pos.xy / F.res; let o = vec2f(F.sh, 0.);
  var c = clamp(vec3f(textureSampleLevel(bufT, ns, uv - o, 0.).r, textureSampleLevel(bufT, ns, uv, 0.).g, textureSampleLevel(bufT, ns, uv + o, 0.).b), vec3f(0.), vec3f(1.));
  c = mix(c, 1. - c, F.fxb.w);
  if (F.inv == 1.) { c = 1. - c; } else if (F.inv == 2.) { c = c * vec3f(1.6, 0.24, 0.16); }
  let v = (uv - 0.5) * 2.; let vig = 1. - 0.45 * min(dot(v, v), 1.6);
  let n = (hh(floor(pos.xy * 360. / F.res.y) + F.frame * 0.618) - 0.5) * 0.11;
  return vec4f((c * vig + n) * F.fade, 1.); }`;

// ---------- stable fluids (256x144) ----------
// Seven passes, twenty of them Jacobi. Vue: velocity is in UV units per second, so the
// feedback pass can add it to a UV displacement directly.
const WG_FL_HEAD=WG_VS+WG_FB_STRUCT+`
@group(0) @binding(0) var<uniform> Q: FUB;
@group(0) @binding(1) var smp: sampler;
@group(0) @binding(2) var tA: texture_2d<f32>;
@group(0) @binding(3) var tB: texture_2d<f32>;
@group(0) @binding(4) var tC: texture_2d<f32>;
@group(0) @binding(5) var tD: texture_2d<f32>;
fn UV(pos: vec4f) -> vec2f { return pos.xy * Q.texel; }
fn VEL(p: vec2f) -> vec2f { return textureSampleLevel(tA, smp, p, 0.).xy; }
`;
const WG_FADV=WG_FL_HEAD+`
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let p = UV(pos); let v = VEL(p);
  let o = textureSampleLevel(tA, smp, p - v * Q.dt, 0.);
  return vec4f(o.xy * 0.999, 0., 1.); }`;
const WG_FFOR=WG_FL_HEAD+`
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let p = UV(pos); var v = VEL(p);
  let mv = textureSampleLevel(tB, smp, p, 0.).xy;
  v += (mv / max(Q.dt, 0.0001) - v) * clamp(Q.gain * Q.dt, 0., 1.);        // block motion as a body force
  if (Q.imp.z > 0.) { let d = p - Q.imp.xy; let r = length(d) + 0.001;      // kick impulse, hashed centre
    v += (d / r) * Q.imp.z * exp(-r * r * 60.); }
  v += vec2f(p.y - 0.5, 0.5 - p.x) * Q.bass * 0.35 * Q.dt;                  // slow swirl
  v *= 1. / (1. + Q.visc * 9. * Q.dt);
  return vec4f(clamp(v, vec2f(-6.), vec2f(6.)), 0., 1.); }`;
const WG_FCURL=WG_FL_HEAD+`
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let p = UV(pos); let dx = vec2f(Q.texel.x, 0.); let dy = vec2f(0., Q.texel.y);
  let l = textureSampleLevel(tA, smp, p - dx, 0.).y; let r = textureSampleLevel(tA, smp, p + dx, 0.).y;
  let b = textureSampleLevel(tA, smp, p - dy, 0.).x; let t = textureSampleLevel(tA, smp, p + dy, 0.).x;
  return vec4f(((r - l) - (t - b)) * 0.5, 0., 0., 1.); }`;
const WG_FVOR=WG_FL_HEAD+`
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let p = UV(pos); let dx = vec2f(Q.texel.x, 0.); let dy = vec2f(0., Q.texel.y);
  let l = textureSampleLevel(tB, smp, p - dx, 0.).x; let r = textureSampleLevel(tB, smp, p + dx, 0.).x;
  let b = textureSampleLevel(tB, smp, p - dy, 0.).x; let t = textureSampleLevel(tB, smp, p + dy, 0.).x;
  var n = vec2f(r - l, t - b); n = n / (length(n) + 0.0001);
  let w = textureSampleLevel(tB, smp, p, 0.).x;
  var v = textureSampleLevel(tA, smp, p, 0.).xy;
  v += vec2f(n.y, -n.x) * w * Q.vort * Q.dt;
  return vec4f(clamp(v, vec2f(-6.), vec2f(6.)), 0., 1.); }`;
const WG_FDIV=WG_FL_HEAD+`
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let p = UV(pos); let dx = vec2f(Q.texel.x, 0.); let dy = vec2f(0., Q.texel.y);
  let l = textureSampleLevel(tA, smp, p - dx, 0.).x; let r = textureSampleLevel(tA, smp, p + dx, 0.).x;
  let b = textureSampleLevel(tA, smp, p - dy, 0.).y; let t = textureSampleLevel(tA, smp, p + dy, 0.).y;
  return vec4f(((r - l) + (t - b)) * 0.5, 0., 0., 1.); }`;
const WG_FJAC=WG_FL_HEAD+`
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let p = UV(pos); let dx = vec2f(Q.texel.x, 0.); let dy = vec2f(0., Q.texel.y);
  let d = textureSampleLevel(tB, smp, p, 0.).x;
  let l = textureSampleLevel(tA, smp, p - dx, 0.).x; let r = textureSampleLevel(tA, smp, p + dx, 0.).x;
  let b = textureSampleLevel(tA, smp, p - dy, 0.).x; let t = textureSampleLevel(tA, smp, p + dy, 0.).x;
  return vec4f((l + r + b + t - d) * 0.25, 0., 0., 1.); }`;
const WG_FGRD=WG_FL_HEAD+`
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let p = UV(pos); let dx = vec2f(Q.texel.x, 0.); let dy = vec2f(0., Q.texel.y);
  var v = textureSampleLevel(tA, smp, p, 0.).xy;
  let l = textureSampleLevel(tB, smp, p - dx, 0.).x; let r = textureSampleLevel(tB, smp, p + dx, 0.).x;
  let b = textureSampleLevel(tB, smp, p - dy, 0.).x; let t = textureSampleLevel(tB, smp, p + dy, 0.).x;
  v -= vec2f(r - l, t - b) * 0.5;
  return vec4f(v, 0., 1.); }`;

let WASM=null;                       // the renderer's own wasm exports (gpuWasm sets it)
function gpuWasm(x){ if(arguments.length) WASM=x; return WASM; }
let markPass=null;   // set by gpuFrame while the stats overlay is on
// ---- environment ----
// The renderer touches no DOM of its own. The caller supplies a document-like object (anything
// with createElement) when media is used, and says whether to render to a texture instead of a
// canvas. Everything else it needs (navigator.gpu, performance) exists in a worker too.
const ENV={doc:null,offscreen:false};
function gpuAttach(o){ if(o&&o.doc)ENV.doc=o.doc; if(o&&o.offscreen!==undefined)ENV.offscreen=!!o.offscreen; return ENV; }
// The mode probe runs on a throwaway canvas: transferControlToOffscreen() is one-shot, so the
// real canvas must never be spent on a capability test.
// The worker says which step it is on before each step that can block, so the page can tell a slow
// adapter from a worker that never started: progress re-arms the stall timer, an answer ends it.
const PROBE_SRC = `
const stage = (s) => self.postMessage({ stage: s });
self.onmessage = async (ev) => {
  const out = { gpu: false, drew: false, lum: 0, err: null };
  try {
    stage('start');
    if (!self.navigator.gpu) throw new Error('no navigator.gpu in worker');
    stage('adapter');
    const ad = await navigator.gpu.requestAdapter();
    if (!ad) throw new Error('no adapter in worker');
    stage('device');
    const dev = await ad.requestDevice();
    dev.addEventListener('uncapturederror', (e) => { out.err = e.error.message; });
    const fmt = navigator.gpu.getPreferredCanvasFormat();
    const ctx = ev.data.canvas.getContext('webgpu');
    ctx.configure({ device: dev, format: fmt, alphaMode: 'opaque' });
    const enc = dev.createCommandEncoder();
    const p = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(),
      loadOp: 'clear', storeOp: 'store', clearValue: [0.1, 0.4, 0.9, 1] }] });
    p.end();
    const tex = dev.createTexture({ size: [8, 8], format: fmt,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const p2 = enc.beginRenderPass({ colorAttachments: [{ view: tex.createView(),
      loadOp: 'clear', storeOp: 'store', clearValue: [0.1, 0.4, 0.9, 1] }] });
    p2.end();
    const ROW = 256;                                   // bytesPerRow must be a multiple of 256
    const buf = dev.createBuffer({ size: ROW * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: ROW }, [8, 8]);
    stage('submit');
    dev.queue.submit([enc.finish()]);
    stage('readback');
    await buf.mapAsync(GPUMapMode.READ);
    const a = new Uint8Array(buf.getMappedRange());
    let sum = 0;
    for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) {
      const i = y * ROW + x * 4; sum += a[i] + a[i + 1] + a[i + 2];
    }
    out.lum = +(sum / (8 * 8 * 3 * 255)).toFixed(3);
    out.gpu = true; out.drew = true;
  } catch (e) { out.err = String((e && e.message) || e); }
  self.postMessage(out);
};`;
// Probe budgets, in ms. Idle, the whole probe is a few hundred milliseconds, so a stall budget of
// 5 s is more than ten times the room that needs. Under load it is not enough: against a SwiftShader
// adapter that another Chromium was already driving, a fresh worker page reported `no: probe timed
// out` in 2 of 2 tries at 5 s, so a single stall is not a verdict. Attempt 2 is a new worker with a
// new throwaway canvas and double the room; both attempts share one cap, so a worker that never
// answers holds the caller for at most 15 s and then reports "no answer", never "no worker".
const PROBE_STALL_MS = [5000, 10000];        // per attempt: no message for this long ends it
const PROBE_TOTAL_MS = 15000;                // both attempts together, setup included
const probeNow = () => (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
// One attempt on a fresh worker and a fresh throwaway canvas. Resolves with the worker's answer
// (ok or not: both are final), a refusal, or a stall. `stall` is how long the attempt may go with
// no message from the worker; `deadline` is the absolute time the whole probe must be done by.
function gpuProbeAttempt(stall, deadline){
  return new Promise((resolve) => {
    const doc = ENV.doc;
    if (!doc) return resolve({ ok: false, why: 'no document to make a throwaway canvas', ms: 0 });
    const t0 = probeNow();
    let w = null, canvas = null;
    try {
      canvas = doc.createElement('canvas'); canvas.width = 8; canvas.height = 8;
      w = new Worker(URL.createObjectURL(new Blob([PROBE_SRC], { type: 'text/javascript' })));
    } catch (e) { return resolve({ ok: false, why: 'worker refused: ' + ((e && e.message) || e), ms: Math.round(probeNow() - t0) }); }
    let settled = false, timer = 0, stage = 'start';
    const done = (r) => {
      if (settled) return; settled = true;
      clearTimeout(timer);
      try { w.terminate(); } catch (e) {}
      try { canvas.remove(); } catch (e) {}
      r.stage = stage; r.ms = Math.round(probeNow() - t0);
      resolve(r);
    };
    // Progress re-arms the timer; the absolute deadline still ends an attempt that keeps talking.
    const arm = () => {
      clearTimeout(timer);
      const left = Math.min(stall, deadline - probeNow());
      if (!(left > 0)) return done({ ok: false, timedOut: true, why: 'no answer in time' });
      timer = setTimeout(() => done({ ok: false, timedOut: true,
        why: 'no answer within ' + Math.round(stall) + ' ms (stage: ' + stage + ')' }), left);
    };
    w.onmessage = (ev) => {
      const d = ev.data || {};
      if (d.stage) { stage = d.stage; return arm(); }          // progress, not an answer
      done({ ok: !!(d.drew && d.lum > 0.02), why: d.err || '', lum: d.lum });
    };
    w.onerror = (ev) => { done({ ok: false, why: ev.message || 'worker error' }); };
    try {
      const off = canvas.transferControlToOffscreen();
      w.postMessage({ canvas: off }, [off]);
    } catch (e) { return done({ ok: false, why: 'transfer refused: ' + ((e && e.message) || e) }); }
    arm();
  });
}
// The probe the app calls: attempt 1, and only after a stall a second attempt with more room. An
// answer of either kind is final; a stall is not, so it comes back as `timedOut` for the caller to
// decide what to do with (the app does not cache it). `onAttempt({attempt, retry})` fires as each
// attempt starts, so the UI can say it is still checking instead of claiming an answer.
async function gpuWorkerProbe(onAttempt){
  const t0 = probeNow(), deadline = t0 + PROBE_TOTAL_MS;
  let tries = 0, last = null;
  for (let i = 0; i < PROBE_STALL_MS.length; i++) {
    const left = deadline - probeNow();
    if (!(left > 0)) break;
    tries++;
    if (typeof onAttempt === 'function') { try { onAttempt({ attempt: tries, retry: i > 0 }); } catch (e) {} }
    const r = await gpuProbeAttempt(Math.min(PROBE_STALL_MS[i], left), deadline);
    if (!r.timedOut) { r.tries = tries; r.ms = Math.round(probeNow() - t0); return r; }
    last = r;
  }
  const ms = Math.round(probeNow() - t0);
  return { ok: false, timedOut: true, lum: last && last.lum, ms, tries: tries || 1, stage: last && last.stage,
    why: 'probe timed out after ' + ms + ' ms (' + (tries || 1) + ((tries || 1) === 1 ? ' attempt' : ' attempts')
      + (last ? ', last: ' + last.why : '') + ')' };
}
// Live texture memory for one renderer, in bytes. It takes its G as an argument and reads nothing
// else, because there are two renderers now: the page's own (main mode) and the render worker's
// (worker mode). The stats overlay asks this for whichever one is drawing, and the worker sends
// the same number in its `stats` message, so the two can never compute it differently.
function gpuTexBytes(g){
  const s=g||G;
  if(!(s&&s.ok&&s.w)) return 0;
  const w=s.w,h=s.h;
  let b=((s.src?s.src.length:0)+(s.buf?s.buf.length:0))*w*h*8;
  b+=3*w*h*8;                                  // tmpA, tmpB, tmpC
  if(s.off) b+=w*h*4;
  b+=2*320*180*8+40*23*8+160*90*8+160*90*4;    // reaction-diffusion, motion, thumbnails
  b+=1024*8*4+512*512*4+24*128+4;              // palette, cellular automaton, spectrum history, blank
  b+=6*256*144*8;                              // fluid solver
  return b;
}
function gpuDispose(){
  try{
    const ts=[...(G.src||[]),...(G.buf||[]),G.tmpA,G.tmpB,G.tmpC,G.off,G.half];
    if(G.fluid) ts.push(G.fluid.velA,G.fluid.velB,G.fluid.prs0,G.fluid.prs1,G.fluid.div,G.fluid.curl);
    for(const t of ts) if(t&&t.destroy) t.destroy();
    G.ok=false; G.valid=false;
  }catch(e){}
}
const G={ok:false,valid:false,si:0,bi:0,ri:0,mvScale:1,rdInit:false,errs:[],media:[],scenes:[]};
async function gpuInit(cv){
  if(!navigator.gpu) throw new Error('no WebGPU in this browser');
  const ad=await navigator.gpu.requestAdapter(); if(!ad) throw new Error('no GPU adapter');
  const adHas=f=>!!(ad.features&&ad.features.has&&ad.features.has(f));
  const tsFeats=['timestamp-query','chromium-experimental-timestamp-query-inside-passes'].filter(adHas);
  let dev;
  try{ dev=await ad.requestDevice(tsFeats.length?{requiredFeatures:tsFeats}:undefined); }
  catch(e){ dev=await ad.requestDevice(); }          // an experimental feature may be refused; never lose the GPU path for it
  G.dev=dev;
  G.tsQ=dev.features.has('timestamp-query');
  if(G.tsQ){
    try{
      G.qs=dev.createQuerySet({type:'timestamp',count:64});
      G.rb=dev.createBuffer({size:512,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
      G.rbMap=dev.createBuffer({size:512,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
      // Some builds allow stamps only on a command encoder, some only inside passes, and some
      // require an experimental flag for passes. Probe once; a refusal is not an error.
      const probe=dev.createTexture({size:[1,1],format:'rgba8unorm',usage:GPUTextureUsage.RENDER_ATTACHMENT});
      const pe=dev.createCommandEncoder();
      const pp=pe.beginRenderPass({colorAttachments:[{view:probe.createView(),loadOp:'clear',storeOp:'discard'}]});
      pp.writeTimestamp(G.qs,0); pp.end(); probe.destroy();
    }catch(e){ G.tsQ=false; G.qs=G.rb=G.rbMap=null; G.tsWhy=(e&&e.message)||String(e); }
  }
  dev.lost.then(i=>{G.ok=false;G.lostMsg=i.message||i.reason;if(G.onLost)G.onLost(G.lostMsg);});
  dev.addEventListener('uncapturederror',e=>{G.errs.push(e.error.message);console.error('WebGPU:',e.error.message);if(G.onError)G.onError(e.error.message);});
  G.ctx=cv.getContext('webgpu'); if(!G.ctx) throw new Error('no WebGPU canvas context');
  G.fmt=navigator.gpu.getPreferredCanvasFormat(); G.ctx.configure({device:dev,format:G.fmt,alphaMode:'opaque'});
  const T=GPUTextureUsage, tex=(w,h,format,usage)=>{const t=dev.createTexture({size:[w,h],format,usage});t.v=t.createView();return t;};
  G.tex=tex; const RT=T.TEXTURE_BINDING|T.RENDER_ATTACHMENT;
  const lut=new Float32Array(WASM.memory.buffer,WASM.pal_ptr(),8*1024*3), pb=new Uint8Array(8*1024*4);
  for(let i=0;i<8*1024;i++){pb[i*4]=lut[i*3]*255;pb[i*4+1]=lut[i*3+1]*255;pb[i*4+2]=lut[i*3+2]*255;pb[i*4+3]=255;}
  G.palT=tex(1024,8,'rgba8unorm',T.TEXTURE_BINDING|T.COPY_DST); dev.queue.writeTexture({texture:G.palT},pb,{bytesPerRow:4096},[1024,8]);
  const ca=new Uint8Array(WASM.memory.buffer,WASM.ca_ptr(),3*512*512), cb=new Uint8Array(512*512*4);
  for(let i=0;i<512*512;i++){cb[i*4]=ca[i]*255;cb[i*4+1]=ca[262144+i]*255;cb[i*4+2]=ca[524288+i]*255;cb[i*4+3]=255;}
  G.caT=tex(512,512,'rgba8unorm',T.TEXTURE_BINDING|T.COPY_DST); dev.queue.writeTexture({texture:G.caT},cb,{bytesPerRow:2048},[512,512]);
  G.melT=tex(24,128,'r8unorm',T.TEXTURE_BINDING|T.COPY_DST); G.melB=new Uint8Array(24*128);
  G.blank=tex(1,1,'rgba8unorm',T.TEXTURE_BINDING|T.COPY_DST); dev.queue.writeTexture({texture:G.blank},new Uint8Array([40,40,40,255]),{bytesPerRow:4},[1,1]);
  G.rd=[0,1].map(()=>tex(320,180,'rgba16float',RT)); G.mvT=tex(40,23,'rgba16float',RT);
  G.thT=tex(160,90,'rgba16float',RT); G.th8=tex(160,90,'rgba8unorm',RT|T.COPY_SRC);
  G.lin=dev.createSampler({magFilter:'linear',minFilter:'linear',addressModeU:'repeat',addressModeV:'repeat'});
  G.linC=dev.createSampler({magFilter:'linear',minFilter:'linear'});
  G.linM=dev.createSampler({magFilter:'linear',minFilter:'linear',addressModeU:'mirror-repeat',addressModeV:'mirror-repeat'});
  G.nm=dev.createSampler({addressModeU:'mirror-repeat',addressModeV:'mirror-repeat'});
  G.nr=dev.createSampler({addressModeU:'repeat',addressModeV:'clamp-to-edge'});
  const ub=n=>dev.createBuffer({size:n,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  G.uA=ub(SU.bytes); G.uB=ub(SU.bytes); G.uT=ub(SU.bytes); G.uF=ub(FU.bytes); G.fuData=new Float32Array(FU.f32); G.suData=new Float32Array(SU.f32);
  const V=GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT;
  G.sceneBGL=dev.createBindGroupLayout({entries:[{binding:0,visibility:V,buffer:{}},{binding:1,visibility:V,sampler:{}},
    ...[2,3,4,5,6].map(binding=>({binding,visibility:V,texture:{}}))]});
  G.sceneLayout=dev.createPipelineLayout({bindGroupLayouts:[G.sceneBGL]});
  G.mod=async(code,name)=>{dev.pushErrorScope('validation');const m=dev.createShaderModule({code});const info=await m.getCompilationInfo();await dev.popErrorScope();
    const er=info.messages.filter(x=>x.type==='error');if(er.length)throw new Error(name+': '+er[0].message+' (line '+er[0].lineNum+')');return m;};
  G.pipe=(m,format,layout,blend)=>dev.createRenderPipelineAsync({layout,vertex:{module:m,entryPoint:'vs'},fragment:{module:m,entryPoint:'fs',targets:[blend?{format,blend}:{format}]},primitive:{topology:'triangle-list'}});
  const add={color:{srcFactor:'one',dstFactor:'one',operation:'add'},alpha:{srcFactor:'one',dstFactor:'one',operation:'add'}};
  const sm=await Promise.all(WG_SCENES.map((c,i)=>G.mod(c,'scene '+i)));
  G.scenes=await Promise.all(sm.map((m,i)=>G.pipe(m,'rgba16float',G.sceneLayout,i===15?add:null)));
  const ms=await Promise.all([[WG_RDSIM,'rd'],[WG_MIX,'mix'],[WG_PRE,'prefx'],[WG_ME,'motion'],[WG_FB,'feedback'],[WG_OUT,'output'],[WG_COPY,'copy']].map(a=>G.mod(a[0],a[1])));
  [G.pRd,G.pMix,G.pPre,G.pMe,G.pFb,G.pOut,G.pCopy]=await Promise.all(ms.map((m,i)=>G.pipe(m,i===5?G.fmt:i===6?'rgba8unorm':'rgba16float','auto')));
  // ---- fluid solver: fixed 256x144 grid, independent of the canvas size ----
  G.fluidBGL=dev.createBindGroupLayout({entries:[{binding:0,visibility:V,buffer:{}},{binding:1,visibility:V,sampler:{}},
    ...[2,3,4,5].map(binding=>({binding,visibility:V,texture:{}}))]});
  G.fluidLayout=dev.createPipelineLayout({bindGroupLayouts:[G.fluidBGL]});
  const fms=await Promise.all([[WG_FADV,'fluid advect'],[WG_FFOR,'fluid forces'],[WG_FCURL,'fluid curl'],
    [WG_FVOR,'fluid vorticity'],[WG_FDIV,'fluid divergence'],[WG_FJAC,'fluid jacobi'],[WG_FGRD,'fluid gradient']]
    .map(a=>G.mod(a[0],a[1])));
  [G.pFad,G.pFfo,G.pFcu,G.pFvo,G.pFdi,G.pFja,G.pFgr]=await Promise.all(fms.map(m=>G.pipe(m,'rgba16float',G.fluidLayout)));
  G.uFl=ub(FB.bytes); G.flData=new Float32Array(FB.f32);
  const ft=()=>G.tex(256,144,'rgba16float',RT);
  const FL=G.fluid={w:256,h:144,velA:ft(),velB:ft(),prs0:ft(),prs1:ft(),div:ft(),curl:ft()};
  const fbg=(a,b,c,d)=>dev.createBindGroup({layout:G.fluidBGL,entries:[
    {binding:0,resource:{buffer:G.uFl}},{binding:1,resource:G.linC},
    {binding:2,resource:(a||G.blank).v},{binding:3,resource:(b||G.blank).v},
    {binding:4,resource:(c||G.blank).v},{binding:5,resource:(d||G.blank).v}]});
  FL.gAdv=fbg(FL.velA); FL.gFor=fbg(FL.velB,G.mvT); FL.gCur=fbg(FL.velA); FL.gVor=fbg(FL.velA,FL.curl);
  FL.gDiv=fbg(FL.velB); FL.gJ0=fbg(FL.prs0,FL.div); FL.gJ1=fbg(FL.prs1,FL.div); FL.gGrd=fbg(FL.velB,FL.prs0);
  G.fluidOn=false;
  assertLayouts();
  G.ok=true;
}
// one step of the solver: every pass goes into the caller's encoder, so a frame is one submit
function fluidStep(enc,pl){
  const FL=G.fluid, Q=G.flData, q=FB.at;
  Q.fill(0);
  Q[q.texel]=1/FL.w; Q[q.texel+1]=1/FL.h;
  Q[q.dt]=1/30; Q[q.visc]=pl[504]; Q[q.gain]=1.6; Q[q.vort]=0.35;
  Q[q.bass]=pl[6]; Q[q.pulse]=pl[9]; Q[q.beat]=pl[11];
  if(pl[9]>0.7&&pl[6]>0.3){                       // kick: radial impulse at a centre hashed from the beat
    const b=Math.floor(pl[11]);
    Q[q.imp]=0.15+0.7*(((Math.sin(b*12.9898)*43758.5453)%1+1)%1);
    Q[q.imp+1]=0.15+0.7*(((Math.sin(b*78.233)*43758.5453)%1+1)%1);
    Q[q.imp+2]=1.6*pl[6]*pl[9];
  }
  G.dev.queue.writeBuffer(G.uFl,0,Q);
  const run=(pipe,group,view)=>{
    const p=enc.beginRenderPass({colorAttachments:[{view,loadOp:'clear',storeOp:'store',clearValue:[0,0,0,0]}]});
    p.setPipeline(pipe); p.setBindGroup(0,group); if(markPass) markPass(p,'fluid'); p.draw(3); p.end();
  };
  run(G.pFad,FL.gAdv,FL.velB.v);                    // advect
  run(G.pFfo,FL.gFor,FL.velA.v);                    // forces -> current velocity
  run(G.pFcu,FL.gCur,FL.curl.v);                    // curl
  run(G.pFvo,FL.gVor,FL.velB.v);                    // vorticity confinement
  run(G.pFdi,FL.gDiv,FL.div.v);                     // divergence
  for(let i=0;i<20;i++) run(G.pFja,i&1?FL.gJ1:FL.gJ0,(i&1?FL.prs0:FL.prs1).v);   // even count, ends in prs0
  run(G.pFgr,FL.gGrd,FL.velA.v);                    // subtract pressure gradient -> current velocity
}
function fluidClear(enc){
  for(const t of [G.fluid.velA,G.fluid.velB,G.fluid.prs0,G.fluid.prs1]){
    const p=enc.beginRenderPass({colorAttachments:[{view:t.v,loadOp:'clear',storeOp:'store',clearValue:[0,0,0,0]}]});
    p.end();
  }
}
function gpuReset(){ G.valid=false; G.fluidOn=false; }   // call next to WASM.reset(): seek, play start, resize, export
// ---- offscreen frame capture: the frame an export encodes --------------------------------
// An export must never read the presented canvas. A canvas snapshot taken when the window is not
// being composited (a background tab, an occluded window) is blank on a real adapter, and the
// result changed between runs, so every export switches to an offscreen texture, renders into
// G.off and reads that texture back. Only the texture path is independent of the window.
// One MAP_READ buffer per export, reused for every frame. bytesPerRow must be a multiple of 256.
function gpuCapOpen(w,h){
  const row=Math.ceil(w*4/256)*256;
  return {w,h,row,bgra:G.fmt.indexOf('bgra')===0,
    buf:G.dev.createBuffer({size:row*h,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ})};
}
function gpuCapClose(cap){ try{cap.buf.destroy();}catch(e){} }
// Copies G.off into a tight RGBA frame in `cap` and returns it with the luminance sampled from
// those same bytes. The guard has to measure what the encoder is handed: the feedback buffer stays
// bright while a blank capture is encoded, which is how black exports shipped green once.
async function gpuCapFrame(cap){
  const dev=G.dev;
  await dev.queue.onSubmittedWorkDone();
  const ce=dev.createCommandEncoder();
  ce.copyTextureToBuffer({texture:G.off},{buffer:cap.buf,bytesPerRow:cap.row},[cap.w,cap.h]);
  dev.queue.submit([ce.finish()]);
  await cap.buf.mapAsync(GPUMapMode.READ);
  const mapped=new Uint8Array(cap.buf.getMappedRange());
  const copy=new Uint8Array(mapped.length); copy.set(mapped); cap.buf.unmap();
  let sum=0,cnt=0;
  for(let i=0;i<copy.length;i+=4*97){sum+=copy[i]+copy[i+1]+copy[i+2];cnt++;}
  const lum=cnt?sum/(cnt*3*255):0;
  // G.fmt is the canvas format, so an bgra8unorm adapter hands us BGR: swap before VideoFrame('RGBA').
  if(cap.bgra)for(let i=0;i<copy.length;i+=4){const b=copy[i];copy[i]=copy[i+2];copy[i+2]=b;}
  return {copy,lum};
}
// Mean luminance of the last feedback buffer, 0..1. Not a capture check: use gpuCapFrame's lum for
// "what did the encoder get".
async function gpuProbe(){
  if(!G.ok||!G.valid) return null;
  const dev=G.dev, T=GPUTextureUsage;
  const tex=dev.createTexture({size:[8,8],format:G.fmt,usage:T.RENDER_ATTACHMENT|T.COPY_SRC});
  const enc=dev.createCommandEncoder();
  const p=enc.beginRenderPass({colorAttachments:[{view:tex.createView(),loadOp:'clear',storeOp:'store',clearValue:[0,0,0,1]}]});
  p.setPipeline(G.pOut);
  p.setBindGroup(0,dev.createBindGroup({layout:G.pOut.getBindGroupLayout(0),entries:[
    {binding:0,resource:{buffer:G.uF}},{binding:1,resource:G.nr},{binding:2,resource:G.buf[G.bi].v}]}));
  p.draw(3); p.end();
  const ROW=256;                                     // bytesPerRow must be a multiple of 256
  const buf=dev.createBuffer({size:ROW*8,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  enc.copyTextureToBuffer({texture:tex},{buffer:buf,bytesPerRow:ROW},[8,8]);
  dev.queue.submit([enc.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const a=new Uint8Array(buf.getMappedRange().slice(0)); buf.unmap(); buf.destroy(); tex.destroy();
  let s=0;
  for(let y=0;y<8;y++) for(let x=0;x<8;x++){const i=y*ROW+x*4; s+=a[i]+a[i+1]+a[i+2];}
  return s/(8*8*3*255);
}
// the layout table is the only description of the uniforms; this catches an edit that breaks
// alignment or leaves the buffers out of step with it, instead of producing garbage uniforms
function assertLayouts(){
  for(const [name,L] of [['SU',SU],['FU',FU],['FB',FB]]){
    if(L.bytes%16||L.f32*4!==L.bytes) throw new Error(name+' layout is not a 16-byte multiple');
    for(const k in L.at) if(!Number.isInteger(L.at[k])) throw new Error(name+'.'+k+' is not float-aligned');
  }
  if(G.uF.size!==FU.bytes||G.uA.size!==SU.bytes||G.uFl.size!==FB.bytes) throw new Error('uniform buffers do not match the layout table');
  if(G.suData.byteLength!==SU.bytes||G.fuData.byteLength!==FU.bytes||G.flData.byteLength!==FB.bytes) throw new Error('uniform staging arrays do not match the layout table');
}
// compile a user shader into scene slot 17+i; resolves to null or an error string
async function gpuCompile(i,code){
  try{ const m=await G.mod(WG_HEAD+WG_VS+code,'shader'); G.dev.pushErrorScope('validation'); const p=await G.pipe(m,'rgba16float',G.sceneLayout,null); const e=await G.dev.popErrorScope();
    if(e) return e.message; G.scenes[17+i]=p; return null; }
  catch(e){ return (e.message||String(e)).replace(/\(line (\d+)\)/,(m,n)=>'(line '+Math.max(1,n-(WG_HEAD+WG_VS).split('\n').length+1)+')'); }
}
async function gpuAddMedia(file){
  const T=GPUTextureUsage, u=T.TEXTURE_BINDING|T.COPY_DST|T.RENDER_ATTACHMENT, m={name:file.name,file};
  if(file.type.startsWith('video')){ const doc=ENV.doc; if(!doc) throw new Error('media needs a document: pass one to gpuAttach');
    const el=doc.createElement('video'); el.muted=true; el.loop=true; el.playsInline=true; el.src=URL.createObjectURL(file);
    await new Promise((res,rej)=>{el.onloadeddata=res;el.onerror=()=>rej(new Error('cannot decode this video'));}); m.kind='video'; m.el=el; m.dur=el.duration||1; const sc=Math.min(1,1280/el.videoWidth); m.w=Math.max(2,Math.round(el.videoWidth*sc)); m.h=Math.max(2,Math.round(el.videoHeight*sc));
    m.cv=doc.createElement('canvas'); m.cv.width=m.w; m.cv.height=m.h; m.c2=m.cv.getContext('2d',{willReadFrequently:true});
    m.tex=G.tex(m.w,m.h,'rgba8unorm',u); try{el.currentTime=Math.min(0.05,m.dur/2);await new Promise(r=>{el.onseeked=r;setTimeout(r,500);});}catch(e){} gpuVideoFrame(m); }
  else { if(!ENV.doc) throw new Error('media needs a document: pass one to gpuAttach'); let bmp=await createImageBitmap(file); if(bmp.width>1920){const b2=await createImageBitmap(bmp,{resizeWidth:1920,resizeHeight:Math.round(bmp.height*1920/bmp.width),resizeQuality:'high'});bmp.close();bmp=b2;}
    m.kind='image'; m.w=bmp.width; m.h=bmp.height; m.tex=G.tex(m.w,m.h,'rgba8unorm',u); G.dev.queue.copyExternalImageToTexture({source:bmp},{texture:m.tex},[m.w,m.h]); bmp.close(); }
  G.media.push(m); return G.media.length-1;
}
function gpuVideoFrame(m){   // upload the video's current frame through a 2D canvas (works wherever images do)
  if(!m.el) return false;    // the worker receives video frames as textures, not as DOM elements
  if(m.el.readyState<2) return false;
  try{ m.c2.drawImage(m.el,0,0,m.w,m.h); G.dev.queue.copyExternalImageToTexture({source:m.cv},{texture:m.tex},[m.w,m.h]); m.got=true; return true; }catch(e){ m.err=e.message; return false; }
}
function gpuResize(w,h){
  const T=GPUTextureUsage, u=T.TEXTURE_BINDING|T.RENDER_ATTACHMENT;
  for(const t of [...(G.src||[]),...(G.buf||[]),G.tmpA,G.tmpB,G.tmpC,G.off]) if(t) t.destroy();
  const mk=()=>G.tex(w,h,'rgba16float',u);
  G.src=[mk(),mk()]; G.buf=[mk(),mk()]; G.tmpA=mk(); G.tmpB=mk(); G.tmpC=mk(); G.w=w; G.h=h; G.valid=false; G.mvScale=1; G.off=null;
  if(ENV.offscreen) G.off=G.tex(w,h,G.fmt,u|T.COPY_SRC);   // test hook: render to a texture instead of the canvas
}
function gpuSU(buf,pl,o,f,w,h){
  const d=G.suData, sc=pl[o], S=SU.at;
  d[S.res]=w; d[S.res+1]=h;
  d[S.T]=pl[o+2]; d[S.hue]=pl[o+3]; d[S.bass]=pl[o+6]; d[S.mid]=pl[o+7]; d[S.high]=pl[o+8]; d[S.pulse]=pl[o+9];
  d[S.vari]=pl[o+4]; d[S.dens]=pl[o+5]; d[S.beat]=pl[11]; d[S.frame]=f; d[S.pal]=pl[o+1];
  d[S.aux0]=-1; d[S.aux1]=0; d[S.aux2]=0;
  if(sc===11){ if(!G.rdInit)d[S.aux2]=1; if(pl[o+9]>0.7&&pl[o+6]>0.3){const b=Math.floor(pl[11]);d[S.aux0]=0.1+0.8*(((Math.sin(b*12.9898)*43758.5453)%1+1)%1);d[S.aux1]=0.1+0.8*(((Math.sin(b*78.233)*43758.5453)%1+1)%1);} }
  if(sc===16){ const m=G.media[pl[o+4]]; d[S.aux0]=m?m.w/m.h:1; d[S.aux1]=pl[o+19]; }
  for(let j=0;j<3;j++){const L=S['l'+j];d[L]=pl[o+10+j*3];d[L+1]=pl[o+11+j*3];d[L+2]=pl[o+12+j*3];d[L+3]=1;}
  for(let k=0;k<24;k++)d[S.mel+k]=pl[64+k];
  G.dev.queue.writeBuffer(buf,0,d); return sc;
}
function gpuScenePass(enc,view,sc,ubuf,clip,dens){
  const dev=G.dev, m=sc===16?G.media[clip]:null;
  if(m&&m.kind==='video') gpuVideoFrame(m);
  const p=enc.beginRenderPass({colorAttachments:[{view,loadOp:'clear',storeOp:'store',clearValue:[0,0,0,1]}]});
  p.setPipeline(G.scenes[sc]||G.scenes[0]);
  if(markPass) markPass(p,'scene');
  p.setBindGroup(0,dev.createBindGroup({layout:G.sceneBGL,entries:[{binding:0,resource:{buffer:ubuf}},{binding:1,resource:G.lin},{binding:2,resource:G.palT.v},{binding:3,resource:G.caT.v},
    {binding:4,resource:G.melT.v},{binding:5,resource:G.rd[G.ri].v},{binding:6,resource:(m?m.tex:G.blank).v}]}));
  if(sc===15)p.draw(6,Math.round(150+1300*dens)); else p.draw(3); p.end();
}
function gpuMel(f){ const nf=WASM.get_nf(); const mel=new Float32Array(WASM.memory.buffer,WASM.mel_ptr(),nf*24), mb=G.melB;
  for(let a=0;a<128;a++){const o=Math.max(0,f-a)*24;for(let k=0;k<24;k++)mb[a*24+k]=Math.min(255,mel[o+k]*196);}
  G.dev.queue.writeTexture({texture:G.melT},mb,{bytesPerRow:24},[24,128]); }
function gpuFrame(f){
  const dev=G.dev, pl=new Float32Array(WASM.memory.buffer,WASM.plan(f),512), enc=dev.createCommandEncoder();
  const bg=(p,entries)=>dev.createBindGroup({layout:p.getBindGroupLayout(0),entries:entries.map((r,i)=>({binding:i,resource:r}))});
  const full=(view,pipeline,group,seg,end)=>{const p=enc.beginRenderPass({colorAttachments:[{view,loadOp:'clear',storeOp:'store',clearValue:[0,0,0,1]}]});if(pipeline){p.setPipeline(pipeline);p.setBindGroup(0,group);if(markPass)markPass(p,seg||'post');p.draw(3);}if(end&&markPass)markPass(p,'end');p.end();};
  const canvasView=G.off?G.off.v:G.ctx.getCurrentTexture().createView();
  const prof=G.prof&&G.tsQ&&!G.profBusy;
  if(prof){ G.tsN=0; G.tsSeg=[]; }
  markPass=prof?((p,seg)=>{ if(G.tsN<63){ p.writeTimestamp(G.qs,G.tsN); G.tsSeg.push(seg); G.tsN++; } }):null;
  if(pl[0]){ full(canvasView,null); dev.queue.submit([enc.finish()]); G.valid=false; return; }
  gpuMel(f);
  const xf=pl[10], iframe=pl[1]>0||!G.valid, freeze=pl[3]>0;
  const scA=gpuSU(G.uA,pl,16,f,G.w,G.h), scB=xf>=0?gpuSU(G.uB,pl,40,f,G.w,G.h):-1;
  const rdU=scA===11?G.uA:scB===11?G.uB:null;
  if(rdU){ for(let i=0;i<12;i++){ full(G.rd[1-G.ri].v,G.pRd,bg(G.pRd,[{buffer:rdU},G.rd[G.ri].v])); G.ri^=1; } G.rdInit=true; }
  const fu=G.fuData, Fw=FU.at; fu.fill(0); const n=pl[12];
  fu[Fw.res]=G.w; fu[Fw.res+1]=G.h;
  fu[Fw.iframe]=iframe?1:0; fu[Fw.skipres]=pl[2]; fu[Fw.leak]=pl[4];
  fu[Fw.coarse]=pl[6]; fu[Fw.sh]=pl[7]; fu[Fw.inv]=pl[8]; fu[Fw.fade]=pl[9];
  fu[Fw.xf]=Math.max(0,xf); fu[Fw.frame]=f; fu[Fw.nops]=n; fu[Fw.gain]=pl[5];
  if(freeze) G.mvScale*=1.02; else if(!iframe) G.mvScale=1;
  fu[Fw.mvScale]=G.mvScale;
  for(let k=0;k<8;k++){fu[Fw.fxa+k]=pl[480+k];fu[Fw.fpa+k]=pl[496+k];}   // fxa..fxd and fpa..fpd are adjacent in the table
  for(let k=0;k<8;k++){fu[Fw.fxc+k]=pl[488+k];fu[Fw.fpc+k]=pl[504+k];}
  for(let i=0;i<n;i++){const o=96+i*8,q=Fw.ops+i*8;fu[q]=pl[o+1];fu[q+1]=pl[o+2];fu[q+2]=pl[o+3];fu[q+3]=pl[o+4];fu[q+4]=pl[o+5];fu[q+5]=pl[o+6];fu[q+6]=pl[o];}
  dev.queue.writeBuffer(G.uF,0,fu);
  const cur=G.src[G.si], prv=G.src[1-G.si], pre=pl[480]>0||pl[481]>0||pl[484]>0||pl[485]>0, tgt=pre?G.tmpC:cur;
  if(xf<0) gpuScenePass(enc,tgt.v,scA,G.uA,pl[20],pl[21]);
  else { gpuScenePass(enc,G.tmpA.v,scA,G.uA,pl[20],pl[21]); gpuScenePass(enc,G.tmpB.v,scB,G.uB,pl[44],pl[45]); full(tgt.v,G.pMix,bg(G.pMix,[{buffer:G.uF},G.tmpA.v,G.tmpB.v])); }
  if(pre) full(cur.v,G.pPre,bg(G.pPre,[{buffer:G.uF},G.linM,G.tmpC.v]),'pre');
  if(!iframe&&!freeze) full(G.mvT.v,G.pMe,bg(G.pMe,[{buffer:G.uF},G.linC,cur.v,prv.v]),'motion');
  if(pl[488]>0){ if(!G.fluidOn){ fluidClear(enc); G.fluidOn=true; } fluidStep(enc,pl); } else G.fluidOn=false;
  full(G.buf[1-G.bi].v,G.pFb,bg(G.pFb,[{buffer:G.uF},G.nm,cur.v,prv.v,G.buf[G.bi].v,G.mvT.v,G.fluid.velA.v]),'feedback'); G.bi^=1;
  full(canvasView,G.pOut,bg(G.pOut,[{buffer:G.uF},G.nr,G.buf[G.bi].v]),'output',true);
  markPass=null;
  if(prof&&G.tsN>1){ enc.resolveQuerySet(G.qs,0,G.tsN,G.rb,0); enc.copyBufferToBuffer(G.rb,0,G.rbMap,0,G.tsN*8); }
  dev.queue.submit([enc.finish()]); G.si^=1; G.valid=true;
  if(prof&&G.tsN>1){
    G.profBusy=true; const segs=G.tsSeg.slice(), n=G.tsN, buf=G.rbMap;
    buf.mapAsync(GPUMapMode.READ).then(()=>{
      const raw=new BigInt64Array(buf.getMappedRange().slice(0)); buf.unmap(); G.profBusy=false;
      const at=i=>Number(raw[i])/1e6;                 // GPU timestamps are 64-bit nanoseconds
      const acc={scene:0,pre:0,motion:0,fluid:0,feedback:0,output:0,post:0};
      for(let i=0;i<n-1;i++){ const ms=Math.max(0,at(i+1)-at(i)); if(acc[segs[i]]!==undefined) acc[segs[i]]+=ms; }
      G.timings={...acc,total:Math.max(0,at(n-1)-at(0)),ts:true};
    }).catch(()=>{ G.profBusy=false; });
  } else if(G.prof){
    const t0=performance.now();
    dev.queue.onSubmittedWorkDone().then(()=>{ G.timings={total:performance.now()-t0,ts:false}; });
  }
}
// 160x90 thumbnail of block i at frame f -> ImageData
async function gpuThumb(i,f){
  const dev=G.dev, pl=new Float32Array(WASM.memory.buffer,WASM.thumb_plan(i,f),512), enc=dev.createCommandEncoder();
  gpuMel(f); const sc=gpuSU(G.uT,pl,16,f,160,90);
  gpuScenePass(enc,G.thT.v,sc,G.uT,pl[20],pl[21]);
  const p=enc.beginRenderPass({colorAttachments:[{view:G.th8.v,loadOp:'clear',storeOp:'store'}]}); p.setPipeline(G.pCopy);
  p.setBindGroup(0,dev.createBindGroup({layout:G.pCopy.getBindGroupLayout(0),entries:[{binding:0,resource:G.thT.v}]})); p.draw(3); p.end();
  const buf=dev.createBuffer({size:768*90,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  enc.copyTextureToBuffer({texture:G.th8},{buffer:buf,bytesPerRow:768},[160,90]); dev.queue.submit([enc.finish()]);
  await buf.mapAsync(GPUMapMode.READ); const a=new Uint8Array(buf.getMappedRange()), out=new Uint8ClampedArray(160*90*4);
  for(let y=0;y<90;y++) out.set(a.subarray(y*768,y*768+640),y*640);
  buf.unmap(); buf.destroy(); return new ImageData(out,160,90);
}

// Write a project into the renderer's wasm. This used to live in the UI; the worker that will
// render frames needs it, and it is also what keeps `project` messages non-resetting.
// Returns null, or a message to show, instead of throwing inside a render loop.
function gpuSetProject(p){
  if(!WASM) return 'renderer has no wasm instance';
  const S=WASM.block_floats(), F=WASM.fx_floats();
  const blocks=p.blocks||[], fx=p.fx||[], bpm=+p.bpm||120, offs=+p.offs||0;
  const lead=offs>0?Math.ceil(offs*bpm/60):0;
  const a=new Float32Array(WASM.memory.buffer, WASM.blocks_ptr(), 512*S);
  const n=Math.min(blocks.length,512); let sum=0;
  for(let i=0;i<n;i++){
    const b=blocks[i], m=b.mods||[[0,0,0],[0,1,0],[0,2,0]];
    const st=i?sum:-lead, ln=i?b.len:b.len+lead;
    const rec=[st,ln,b.scene,b.pal,b.trans,b.tlen,b.speed,b.mosh,b.strobe,b.glitch,b.rgb,b.scene===16?b.clip:b.var,b.dens,b.bloom,
      m[0][0],m[0][1],m[0][2],m[1][0],m[1][1],m[1][2],m[2][0],m[2][1],m[2][2],0];
    if(rec.length!==S) return `block writer sends ${rec.length} floats, the engine's record is ${S}`;
    a.set(rec,i*S); sum+=b.len;
  }
  WASM.set_nblocks(n);
  const fa=new Float32Array(WASM.memory.buffer, WASM.fx_ptr(), 256*F), fn=Math.min(fx.length,256);
  for(let i=0;i<fn;i++){ const e=fx[i], rec=[e.start,e.len,e.type,e.amt,e.par];
    if(rec.length!==F) return `effect writer sends ${rec.length} floats, the engine's record is ${F}`;
    fa.set(rec,i*F); }
  WASM.set_nfx(fn);
  return null;
}

// Load analysis made elsewhere into the renderer's wasm. The order is not cosmetic: the feature
// arrays and the mel table come first, then set_grid() derives fPulse/fTT/fHue from them, and
// plan() reads those every frame -- set_grid before the arrays copies leaves U.T at zero and the
// picture freezes. Used by the page and by the render worker, so the sequence exists once.
// Returns null, or a message to show, instead of throwing inside a render loop.
function gpuLoadFeatures(p){
  if(!WASM) return 'renderer has no wasm instance';
  const o=p||{}, n=+o.nf||0, arr=o.arrays||[];
  WASM.load_features(n);
  for(let i=0;i<arr.length;i++)
    new Float32Array(WASM.memory.buffer, WASM.feat(i), n).set(arr[i]);
  new Float32Array(WASM.memory.buffer, WASM.mel_ptr(), n*24).set(o.mel);
  WASM.set_grid(o.bpm, o.offs);
  return null;
}

// The renderer's public surface: everything a page or a worker needs to render a project with
// only its own wasm instance. exportVideo stays in the UI.
const Renderer={attach:gpuAttach,init:gpuInit,frame:gpuFrame,resize:gpuResize,thumb:gpuThumb,
  addMedia:gpuAddMedia,compileShader:gpuCompile,probe:gpuProbe,reset:gpuReset,dispose:gpuDispose,
  setProject:gpuSetProject,loadFeatures:gpuLoadFeatures,texBytes:gpuTexBytes};
