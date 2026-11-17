// What an unknown visualizer id draws: black, so a typo in a project file is a black layer and
// never a panic or a stale picture.
@fragment fn fs_main() -> @location(0) vec4f { return vec4f(0., 0., 0., 1.); }
