//! mosh-feed: plays an audio file into the `master` shared-memory region in real time, with a
//! constant-tempo 4/4 transport, exactly as Moshloom Tap would from Ableton's master. Lets the app
//! run in plugin mode with real music and no sound coming out of the speakers.
//!
//!     mosh-feed <file> [--bpm <f64>] [--start-s <f64>] [--seconds <f64>]
//!
//! `--bpm` defaults to the tempo in the file name. The transport starts at the file's beat 1 at
//! 0:00 (true for every tuning track) and `--start-s` skips ahead without breaking the grid.

use std::time::{Duration, Instant};

use mosh_analysis::offline::{blocks, read_audio, tempo_from_name};
use mosh_core::ShowRole;
use mosh_ipc::IpcWriter;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(path) = args.first() else {
        eprintln!("usage: mosh-feed <file> [--bpm <f64>] [--start-s <f64>] [--seconds <f64>]");
        std::process::exit(2);
    };
    let flag = |name: &str| -> Option<f64> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse().ok())
    };
    let Some(bpm) = flag("--bpm").or_else(|| tempo_from_name(path)) else {
        eprintln!("mosh-feed: no tempo in the file name; pass --bpm");
        std::process::exit(2);
    };
    let start_s = flag("--start-s").unwrap_or(0.0).max(0.0);
    let seconds = flag("--seconds").unwrap_or(f64::INFINITY);

    let (samples, rate, channels) = match read_audio(path) {
        Ok(audio) => audio,
        Err(error) => {
            eprintln!("mosh-feed: {error}");
            std::process::exit(1);
        }
    };
    let mut writer = match IpcWriter::create(ShowRole::Master, rate, channels) {
        Ok(writer) => writer,
        Err(error) => {
            eprintln!("mosh-feed: cannot open the master region: {error:?}");
            std::process::exit(1);
        }
    };
    const BLOCK: usize = 256;
    let block_s = BLOCK as f64 / f64::from(rate);
    let skip = (start_s / block_s) as usize;
    eprintln!("mosh-feed: {path} at {bpm} BPM, {rate} Hz, {channels} ch, from {start_s:.1} s");

    let t0 = Instant::now();
    for (n, block) in blocks(&samples, rate, channels, BLOCK, bpm, 0.0).skip(skip).enumerate() {
        let elapsed = n as f64 * block_s;
        if elapsed >= seconds {
            break;
        }
        let due = t0 + Duration::from_secs_f64(elapsed);
        if let Some(wait) = due.checked_duration_since(Instant::now()) {
            std::thread::sleep(wait);
        }
        let frames = block.frames() as u32;
        writer.push_block(block.transport.as_ref(), &block.samples, frames);
    }
}
