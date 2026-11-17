// The web app's palT: 1024x8 rgba8unorm, built by engine.c `init_tables()` from KEYS/KEYN and
// sampled by `lut()`. Generated once, at the first frame that needs a scene, because
// `mosh-render` has no texture upload (so the table cannot be copied in from the CPU).
//
// No bindings: a pixel only needs its own coordinate. x is the entry (0..1023), y the palette
// (0..7); palette 0 is black, exactly as engine.c leaves it.
const KEYN = array<i32, 8>(1, 8, 6, 6, 6, 6, 6, 6);
var<private> KEYS: array<array<vec3f, 8>, 8> = array<array<vec3f, 8>, 8>(
  array<vec3f, 8>(vec3f(0.), vec3f(0.), vec3f(0.), vec3f(0.), vec3f(0.), vec3f(0.), vec3f(0.), vec3f(0.)),
  array<vec3f, 8>(vec3f(0., 0., 0.), vec3f(1., .02, .05), vec3f(1., .45, 0.), vec3f(0., 0., 0.), vec3f(.1, 1., .15), vec3f(0., 0., 0.), vec3f(.6, .05, 1.), vec3f(1., 1., 1.)),
  array<vec3f, 8>(vec3f(0., 0., 0.), vec3f(.1, 1., .35), vec3f(0., .12, .05), vec3f(0., 0., 0.), vec3f(1., .05, .7), vec3f(0., 0., 0.), vec3f(0.), vec3f(0.)),
  array<vec3f, 8>(vec3f(.05, .02, .2), vec3f(.85, .85, .95), vec3f(1., .25, .6), vec3f(0., 0., 0.), vec3f(.2, .9, .8), vec3f(.1, .05, .3), vec3f(0.), vec3f(0.)),
  array<vec3f, 8>(vec3f(0., 0., 0.), vec3f(.35, 0., .02), vec3f(1., .05, .05), vec3f(1., .5, .1), vec3f(1., .75, .7), vec3f(.5, 0., .2), vec3f(0.), vec3f(0.)),
  array<vec3f, 8>(vec3f(0., 0., 0.), vec3f(1., 1., 1.), vec3f(0., 0., 0.), vec3f(.45, .45, .45), vec3f(0., 0., 0.), vec3f(1., 1., 1.), vec3f(0.), vec3f(0.)),
  array<vec3f, 8>(vec3f(0., 0., 0.), vec3f(.05, .05, .6), vec3f(.5, .1, 1.), vec3f(0., 0., 0.), vec3f(.1, .9, 1.), vec3f(1., 1., 1.), vec3f(0.), vec3f(0.)),
  array<vec3f, 8>(vec3f(0., 0., 0.), vec3f(1., .95, 0.), vec3f(.4, 1., 0.), vec3f(0., 0., 0.), vec3f(1., .1, .6), vec3f(.1, 0., .15), vec3f(0.), vec3f(0.)),
);

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let entry = i32(pos.x);
  let pal = i32(pos.y);
  if (pal < 1 || pal > 7) { return vec4f(0., 0., 0., 1.); }
  let k = KEYN[pal];
  // engine.c: v = i/1024*k; smoothstep between key a and key (a+1)%k.
  let v = f32(entry) / 1024. * f32(k);
  let a = i32(floor(v));
  let b = (a + 1) % k;
  var f = v - floor(v);
  f = f * f * (3. - 2. * f);
  return vec4f(mix(KEYS[pal][a], KEYS[pal][b], f), 1.);
}
