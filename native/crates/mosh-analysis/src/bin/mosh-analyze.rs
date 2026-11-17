//! `mosh-analyze`: play a track through the offline harness.
//!
//! Task 1.5 of `docs/superpowers/plans/2026-11-17-native-port.md`:
//!
//! ```text
//! mosh-analyze <file>... [--bpm <f64>] [--offset-s <f64>] [--labels <file.toml>]
//!             [--tempos <tempos.toml>] [--propose] [--out <file.toml>] [--dump]
//! ```
//!
//! Without `--propose`, `--labels` or `--dump` this only decodes: sample rate, channels, duration,
//! and the tempo hint in the file name. That path works in every build.
//!
//! `--dump` prints one line per closed bar: the bar, its raw short-term level, the four normalised
//! terms the intensity model used, the intensity, the reading the level was taken from, the level
//! and the events that bar fired. It is a tuning instrument, not a report: nothing reads it back.
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
                    [--tempos <tempos.toml>] [--propose] [--out <file.toml>] [--dump]

Decodes WAV (PCM 8/16/24/32-bit, IEEE float 32/64-bit) and MP3, and prints the sample rate,
the channel count, the duration and the tempo written in the file name (`93`, `137bpm`).

  --bpm <f64>         tempo for the synthetic transport; defaults to the file name
  --offset-s <f64>    seconds before the first downbeat (default 0)
  --from-s <f64>      analyse from this second into the file, as a run that starts mid-track
                      would (the transport starts a fresh grid there); default 0
  --until-s <f64>     stop at this second; default the end of the file
  --tempos <file>     tempo per file name, as \"<file name>\" = <bpm>; beats the file name
  --labels <file>     score this label file against the detected marks
  --propose           write the detected marks as <file stem>.labels.toml
  --out <file>        where --propose writes; needs exactly one input file
  --dump              one line per bar: level, level inputs, intensity and events
  -h, --help          this text

--propose, --labels and --dump need a build with --features live-analysis.";

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
    let tempos = match &args.tempos {
        Some(path) => match offline::load_tempos(path) {
            Ok(map) => Some(map),
            Err(error) => {
                eprintln!("mosh-analyze: {error}");
                return ExitCode::from(2);
            }
        },
        None => None,
    };
    for path in &args.paths {
        if let Err(message) = run_one(&args, path, tempos.as_ref()) {
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
    tempos: Option<PathBuf>,
    from_s: Option<f64>,
    until_s: Option<f64>,
    propose: bool,
    out: Option<PathBuf>,
    dump: bool,
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
                "--dump" => args.dump = true,
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
                "--tempos" => {
                    let value = take_value(argv, &mut index, inline, "--tempos")?;
                    args.tempos = Some(PathBuf::from(value));
                }
                "--from-s" => {
                    let value = take_value(argv, &mut index, inline, "--from-s")?;
                    args.from_s = Some(parse_number("--from-s", &value)?);
                }
                "--until-s" => {
                    let value = take_value(argv, &mut index, inline, "--until-s")?;
                    args.until_s = Some(parse_number("--until-s", &value)?);
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
        self.propose || self.labels.is_some() || self.dump
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

fn run_one(
    args: &Args,
    path: &Path,
    tempos: Option<&offline::TempoMap>,
) -> Result<(), String> {
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
    let hint = tempos
        .and_then(|map| offline::tempo_for(map, path))
        .or_else(|| offline::tempo_from_name(&name));

    match hint {
        Some(bpm) => println!(
            "{name}: {rate} Hz, {channels} ch, {frames} frames, {duration_s:.3} s, tempo hint {bpm}"
        ),
        None => println!(
            "{name}: {rate} Hz, {channels} ch, {frames} frames, {duration_s:.3} s, tempo hint none"
        ),
    }

    if args.wants_analysis() {
        let (from_s, until_s) = (args.from_s.unwrap_or(0.0), args.until_s);
        if from_s < 0.0 || until_s.is_some_and(|until| until <= from_s) {
            return Err(format!(
                "--from-s {from_s} / --until-s {until_s:?} do not describe a window"
            ));
        }
        // A run that starts mid-track has no history: the analyzer's window begins where the
        // stream began, exactly as it does when the tap is started at 0:30.
        let first = ((from_s * f64::from(rate)) as usize).min(frames) * usize::from(channels);
        let last = until_s
            .map(|until| ((until * f64::from(rate)) as usize).min(frames))
            .unwrap_or(frames)
            * usize::from(channels);
        run_analysis(args, path, &samples[first..last], rate, channels, hint)?;
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
    if args.dump {
        dump_rows(&run);
    }
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

/// `--dump`: one line per bar, the section table, the unsupervised checks, then a `#` summary
/// line the parser may skip.
#[cfg(feature = "live-analysis")]
fn dump_rows(run: &offline::AnalysisRun) {
    use mosh_core::EventFlags;

    const NAMES: [(&str, EventFlags); 6] = [
        ("build", EventFlags::BUILD_START),
        ("drop", EventFlags::DROP),
        ("breakdown", EventFlags::BREAKDOWN),
        ("up", EventFlags::LEVEL_UP),
        ("down", EventFlags::LEVEL_DOWN),
        ("phrase", EventFlags::PHRASE),
    ];
    let mut counts = [0usize; 4];
    for row in &run.rows {
        let names: Vec<&str> = NAMES
            .iter()
            .filter(|(_, flag)| row.update.events.contains(*flag))
            .map(|(name, _)| *name)
            .collect();
        let mut events = if names.is_empty() {
            "-".to_string()
        } else {
            names.join("+")
        };
        if row.drop && !row.update.events.contains(EventFlags::DROP) {
            // The mid-bar pass fired this drop on the downbeat, a bar earlier than the close.
            events = if events == "-" {
                "drop(downbeat)".to_string()
            } else {
                format!("{events}+drop(downbeat)")
            };
        }
        println!(
            "  bar {:>4}  db {:>8.2}  lvl {:>6.2}  early {:>6.2}  loud {:.3}  low {:.3}  onset {:.3}  bright {:.3}  int {:.3}  read {:.3}  level {}  sec {}  {:.3}  tgt {:.3}  nov {:.3}  {events}",
            row.stats.bar_index,
            row.stats.loudness_db,
            row.raw.level_db,
            row.raw.early_db,
            row.update.terms.loudness,
            row.update.terms.low,
            row.update.terms.onset,
            row.update.terms.brightness,
            row.update.intensity,
            row.update.reading,
            row.update.level,
            row.section.index,
            row.section.energy,
            row.section.target,
            row.section.novelty,
        );
        let slot = usize::from(row.update.level.clamp(1, 4)) - 1;
        counts[slot] += 1;
    }
    print!("{}", run.eval.sections_report());
    print!("{}", run.eval.report());
    println!(
        "# levels l1={} l2={} l3={} l4={} bars={}",
        counts[0],
        counts[1],
        counts[2],
        counts[3],
        run.rows.len()
    );
}
