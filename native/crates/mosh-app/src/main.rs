//! mosh-app: see docs/superpowers/plans/2026-11-17-native-port.md, Task 2.6.
//!
//! The app never exits on its own and never shows a dialog: an error is one line on stderr, and
//! the show keeps running.

fn main() {
    if let Err(error) = mosh_app::shell::run() {
        eprintln!("moshloom: {error}");
        std::process::exit(1);
    }
}
