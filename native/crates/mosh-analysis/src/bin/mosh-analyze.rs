//! `mosh-analyze`: play a track through the offline harness.
//!
//! Task 1.5 of `docs/superpowers/plans/2026-11-17-native-port.md`:
//!
//! ```text
//! mosh-analyze <file>... [--bpm <f64>] [--offset-s <f64>] [--labels <file.toml>]
//!             [--propose] [--out <file.toml>]
//! ```
//!
//! Without `--propose` or `--labels` this only decodes: sample rate, channels, duration, and the
//! tempo hint in the file name. That path works in every build.
//!
//! `--propose` writes `<file stem>.labels.toml` (or `--out`) with the marks the intensity model
//! fired. `--labels` scores a hand-corrected file against the same run, per kind, with hits,
//! misses and mean timing error in bars. Both need the `live-analysis` cargo feature, which is
//! off by default until `Analyzer` (Task 1.3) and `IntensityModel` (Task 1.4) land on main.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use mosh_analysis::offline;

const USAGE: &str = "\
usage: mosh-analyze <file>... [--bpm <f64>] [--offset-s <f64>] [--labels <file.toml>]
                    [--propose] [--out <file.toml>]

Decodes WAV (PCM 8/16/24/32-bit, IEEE float 32/64-bit) and MP3, and prints the sample rate,
the channel count, the duration and the tempo written in the file name (`93`, `137bpm`).

  --bpm <f64>       tempo for the synthetic transport; defaults to the file name
  --offset-s <f64>  seconds before the first downbeat (default 0)
  --labels <file>   score this label file against the detected marks
  --propose         write the detected marks as <file stem>.labels.toml
  --out <file>      where --propose writes; needs exactly one input file
  -h, --help        this text

--propose and --labels need a build with --features live-analysis.";

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let args = match Args::parse(&argv) {
        Ok(args) => args,
        Err(message) => {
            eprintln!("mosh-analyze: {message}");
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    if args.help {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    if args.paths.is_empty() {
        eprintln!("mosh-analyze: no input files");
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    }

    let mut failures = 0usize;
    for path in &args.paths {
        if let Err(message) = run_one(&args, path) {
            eprintln!("mosh-analyze: {message}");
            failures += 1;
        }
    }
    if failures == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

#[derive(Debug, Default, PartialEq)]
struct Args {
    paths: Vec<PathBuf>,
    bpm: Option<f64>,
    offset_s: Option<f64>,
    labels: Option<PathBuf>,
    propose: bool,
    out: Option<PathBuf>,
    help: bool,
}

impl Args {
    fn parse(argv: &[String]) -> Result<Args, String> {
        let mut args = Args::default();
        let mut index = 0usize;
        while index < argv.len() {
            let arg = argv[index].as_str();
            let (flag, inline) = match arg.split_once('=') {
                Some((flag, value)) => (flag, Some(value)),
                None => (arg, None),
            };
            match flag {
                "-h" | "--help" => args.help = true,
                "--propose" => args.propose = true,
                "--bpm" => {
                    let value = take_value(argv, &mut index, inline, "--bpm")?;
                    args.bpm = Some(parse_number("--bpm", &value)?);
                }
                "--offset-s" => {
                    let value = take_value(argv, &mut index, inline, "--offset-s")?;
                    args.offset_s = Some(parse_number("--offset-s", &value)?);
                }
                "--labels" => {
                    let value = take_value(argv, &mut index, inline, "--labels")?;
                    args.labels = Some(PathBuf::from(value));
                }
                "--out" => {
                    let value = take_value(argv, &mut index, inline, "--out")?;
                    args.out = Some(PathBuf::from(value));
                }
                _ if flag.starts_with("--") => return Err(format!("unknown flag {flag}")),
                _ => args.paths.push(PathBuf::from(arg)),
            }
            index += 1;
        }

        if args.out.is_some() && !args.propose {
            return Err("--out only makes sense with --propose".to_string());
        }
        if args.out.is_some() && args.paths.len() > 1 {
            return Err("--out writes one file, so give exactly one input file".to_string());
        }
        if let Some(bpm) = args.bpm
            && (!bpm.is_finite() || bpm <= 0.0)
        {
            return Err(format!("--bpm must be a positive number, got {bpm}"));
        }
        Ok(args)
    }

    fn wants_analysis(&self) -> bool {
        self.propose || self.labels.is_some()
    }
}

fn take_value(
    argv: &[String],
    index: &mut usize,
    inline: Option<&str>,
    flag: &str,
) -> Result<String, String> {
    if let Some(value) = inline {
        return Ok(value.to_string());
    }
    *index += 1;
    argv.get(*index)
        .cloned()
        .ok_or_else(|| format!("{flag} needs a value"))
}

fn parse_number(flag: &str, value: &str) -> Result<f64, String> {
    value
        .parse::<f64>()
        .map_err(|_| format!("{flag} expects a number, got {value:?}"))
}

fn run_one(args: &Args, path: &Path) -> Result<(), String> {
    let (samples, rate, channels) = offline::read_audio(path).map_err(|e| e.to_string())?;
    let frames = if channels == 0 {
        0
    } else {
        samples.len() / usize::from(channels)
    };
    let duration_s = if rate == 0 {
        0.0
    } else {
        frames as f64 / f64::from(rate)
    };
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("?")
        .to_string();
    let hint = offline::tempo_from_name(&name);

    match hint {
        Some(bpm) => println!(
            "{name}: {rate} Hz, {channels} ch, {frames} frames, {duration_s:.3} s, tempo hint {bpm}"
        ),
        None => println!(
            "{name}: {rate} Hz, {channels} ch, {frames} frames, {duration_s:.3} s, tempo hint none"
        ),
    }

    if args.wants_analysis() {
        run_analysis(args, path, &samples, rate, channels, hint)?;
    }
    Ok(())
}

#[cfg(not(feature = "live-analysis"))]
fn run_analysis(
    args: &Args,
    path: &Path,
    _samples: &[f32],
    _rate: u32,
    _channels: u16,
    _hint: Option<f64>,
) -> Result<(), String> {
    let _ = (args, path);
    Err(
        "--propose and --labels need the live-analysis cargo feature (Analyzer, Task 1.3, and \
         IntensityModel, Task 1.4, are not on main yet): cargo run -p mosh-analysis \
         --features live-analysis --bin mosh-analyze -- <file> --bpm <n> --propose"
            .to_string(),
    )
}

#[cfg(feature = "live-analysis")]
fn run_analysis(
    args: &Args,
    path: &Path,
    samples: &[f32],
    rate: u32,
    channels: u16,
    hint: Option<f64>,
) -> Result<(), String> {
    use mosh_analysis::offline::Labels;

    let labelled = match &args.labels {
        Some(labels_path) => Some(Labels::load(labels_path).map_err(|e| e.to_string())?),
        None => None,
    };
    let bpm = args
        .bpm
        .or_else(|| labelled.as_ref().and_then(|labels| labels.bpm))
        .or(hint)
        .ok_or_else(|| {
            format!(
                "{}: no tempo: pass --bpm, put bpm in the label file, or rename the track with one",
                path.display()
            )
        })?;
    let offset_s = args
        .offset_s
        .or_else(|| labelled.as_ref().and_then(|labels| labels.offset_s))
        .unwrap_or(0.0);

    let run = offline::analyze(samples, rate, channels, bpm, offset_s);
    println!(
        "  analysed {:.2} s at {bpm} bpm, offset {offset_s} s: {} bars, {} marks",
        run.duration_s,
        run.bars,
        run.labels.marks.len()
    );
    for (kind, count) in run.labels.counts() {
        println!("  proposed {:<9} {count}", kind.as_str());
    }

    let mut detected = run.labels.clone();
    detected.bpm = Some(bpm);
    detected.offset_s = (offset_s != 0.0).then_some(offset_s);

    if args.propose {
        let out = args.out.clone().unwrap_or_else(|| Labels::path_for(path));
        detected.save(&out).map_err(|e| e.to_string())?;
        println!("  wrote {}", out.display());
    }

    if let Some(labels) = &labelled {
        print!("{}", offline::score(labels, &detected).report());
    }
    Ok(())
}
