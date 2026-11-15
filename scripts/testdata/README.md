# scripts/testdata

Saved `npm test` / `npm run test:gpu:hw` logs that `scripts/gate.mjs` is pinned against, with the
expected gate line for each one written out by hand in `scripts/gate.test.mjs`.

These are real runs from `out/`, transcribed and nothing else:

- **encoding**: one file (`fail-export-skip.log`) is kept in the encoding PowerShell redirection
  produces, **UTF-16 LE with a BOM**, so the decoder stays tested. The rest are UTF-8. Line endings
  are LF in every file.
- **paths**: no log contained an absolute path or a user name, so the scrub in the transcription
  (`C:\Users\<name>\...` -> `<repo>\...`) had nothing to rewrite.
- **mojibake**: the UTF-16 capture mangled non-ASCII that `gpu_test.mjs` prints - `·` arrives as
  `┬╖` and `Δ` as `╬ö`. That is left exactly as captured. The parser matches ASCII only, so the
  gate line does not depend on it.
- `synthetic-truncated.log` is the single deliberate exception: `fail-export-skip.log` cut inside
  the golden table, to exercise the truncated/unparseable exit (2).

| file | source | what it pins |
| --- | --- | --- |
| `pass-swiftshader.log` | `out/gate.log` on main 240c736 | a full `npm test` run after X2: 13/13 goldens, 0 differing, CPU 14.85 ms/frame, main export runs on SwiftShader, no skip -> PASS, exit 0. |
| `fail-export-skip.log` | `out/final-test2.log` | the same kind of run from before X2, with the old SwiftShader export skip: since X2 any skip FAILs -> FAIL, exit 1. Also the UTF-16 LE sample. |
| `pass-hardware.log` | `out/final-hardware2.log` | `npm run test:gpu:hw`: goldens are skipped on hardware, so `goldens skipped (hardware)` and `CPU n/a (hardware run)`, never a 13/13 -> PASS, exit 0. |
| `fail-worker-pixels.log` | `out/test-after-fixes.log` | goldens clean and every step ran, but the log ends in a `FAIL` (worker pixels differ) -> FAIL, exit 1. |
| `fail-goldens-drift.log` | `out/gpu-hardware.log` | 0/13 within tolerance, 2351169 differing pixels, and an unexpected `worker:` skip -> FAIL, exit 1. |
| `fail-hardware-goldens.log` | `out/gpu-hardware-golden.log` | `--hardware --hardware-golden`: a hardware run that did compare goldens, so the line is not `skipped (hardware)` -> FAIL, exit 1. |
| `fail-page-errors.log` | `out/test-after-hw-fixes.log` | goldens 13/13, but two `CONSOLE` lines in the FAIL block -> `page errors 2`, FAIL, exit 1. |
| `partial-gpu-only.log` | `out/gpu-test-check.log` | `node gpu_test.mjs` alone: there is no CPU number, and none is invented -> `CPU n/a (partial log: test.mjs did not run)`, FAIL, exit 1. |
| `synthetic-truncated.log` | `out/final-test2.log`, cut | no verdict line -> exit 2, with each unknown field printed as `n/a (<reason>)`. |

`out/` is gitignored, so the raw files stay out of the repository.

    node scripts/gate.test.mjs                    # every file above, exit 0 when all match
    node scripts/gate.mjs scripts/testdata/pass-swiftshader.log
