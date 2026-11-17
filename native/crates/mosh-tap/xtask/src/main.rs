//! nih-plug's bundler, so `mosh-tap` can be bundled from inside its own crate directory.
//!
//! The bundler walks up to the workspace root (`native/`), runs
//! `cargo build -p mosh-tap --release`, and copies the resulting DLL into
//! `native/target/bundled/<name>.vst3/Contents/x86_64-win/<name>.vst3`.
//!
//! Usage, from `native/`:
//!
//! ```text
//! cargo run --manifest-path crates/mosh-tap/xtask/Cargo.toml -- bundle mosh-tap --release
//! ```
//!
//! `cargo xtask` (as in nih-plug's own README) needs a `native/.cargo/config.toml` alias, which is
//! outside this lane's files: the coordinator adds it if the short form is wanted.

fn main() -> nih_plug_xtask::Result<()> {
    nih_plug_xtask::main()
}
