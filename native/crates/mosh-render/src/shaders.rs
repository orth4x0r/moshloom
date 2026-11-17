//! The crate's own SPIR-V, generated from `src/shaders/*.wgsl` by `build.rs`.

include!(concat!(env!("OUT_DIR"), "/shaders_gen.rs"));
