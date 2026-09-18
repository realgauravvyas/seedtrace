# Changelog

All notable changes to `seedtrace` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

- Roadmap: distributed / multi-rank RNG capture, notebook magic
  (`%%seedtrace`), JSON schema publication for `run.json`, hazard
  plugins via entry points.

## [0.1.0] - 2026-09-18

First public release.

### Added

- RNG stream fingerprinting: `python.random`, `numpy.random`,
  `torch.cpu`, and per-device `torch.cuda.N` states, reduced to SHA-256
  digests with zero runtime dependencies.
- `seedtrace.auto()` / `seedtrace.mark()` embedding API: ordered
  checkpoints with bit-exact value digests (tensors, arrays, nested
  containers, NaN/Inf-safe floats).
- `seedtrace run`: child-process tracer with `--seed` and `--hashseed`
  pinning; records `run.json` per execution.
- `seedtrace compare`: earliest-divergence bisection between two runs
  with cause classification (RNG-state drift vs. equal-RNG value drift
  vs. control-flow split).
- `seedtrace audit`: static AST hazard scanner (set-iteration order,
  unsorted `os.listdir`, non-deterministic CUDA ops, unseeded
  DataLoader workers, cudnn.benchmark/TF32, unseeded RNG consumers)
  plus live-environment checks; `--json` and `--fail-on` for CI.
- `seedtrace variance`: repeat-N metric spread measurement with a
  `--tol` CI gate.
- `seedtrace determinism-patch`: conservative drop-in hardening
  snippet.
- Example gallery (`examples/`) and CI matrix across Linux/macOS/
  Windows, Python 3.9–3.13.
