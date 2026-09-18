# seedtrace

[![CI](https://github.com/realgauravvyas/seedtrace/actions/workflows/ci.yml/badge.svg)](https://github.com/realgauravvyas/seedtrace/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Dependencies: 0](https://img.shields.io/badge/dependencies-0-green)](pyproject.toml)

**Find out why two runs with the same seed gave different results.**

```
Same seed. Same code. Same machine. Different score. Again.
```

Every ML practitioner has lost an afternoon to this. You seeded `random`,
`numpy`, and `torch` — and your metric *still* moves between runs. The
PyTorch docs hand you a wall of flags (`use_deterministic_algorithms`,
`CUBLAS_WORKSPACE_CONFIG`, TF32, ...) but **no tool tells you *where* your
pipeline diverges or *which* cause applies to you.**

`seedtrace` is a forensic debugger for nondeterminism. Add two calls to
your training script, run it twice, and get a report that points at the
exact checkpoint and the exact line where the two runs split.

## Demo: a full investigation, start to finish

Real terminal output against the bundled
[`examples/demo_hash_order.py`](examples/demo_hash_order.py) — a script
that *does* seed everything and *still* varies.

**1. Confirm and measure the damage** (exit code 1 makes it a CI gate):

```
$ seedtrace variance -n 3 --seed 42 -- examples/demo_hash_order.py
------------------------------------------------------------------------
seedtrace variance: 3 repeats, seed=42, PYTHONHASHSEED=random
  run 0: 1.71
  run 1: 1.5
  run 2: 1.51
  mean=1.5733333333333333  stdev=0.11846237095944571  min=1.5  max=1.71
  RESULT: metric VARIES under a fixed seed -> nondeterminism confirmed
  stdev 0.11846237095944571 exceeds tolerance 0.0 -> exit 1
```

**2. Point at the exact line:**

```
$ seedtrace audit examples/demo_hash_order.py
------------------------------------------------------------------------
seedtrace audit: examples\demo_hash_order.py
------------------------------------------------------------------------
[HIGH  ] hash-seed-unpinned
    (environment)
    PYTHONHASHSEED is not set; str/bytes hash order is randomized per process
    fix: Export PYTHONHASHSEED=0 (or use `seedtrace run --hashseed 0 -- ...`)

[HIGH  ] set-iteration-order
    examples/demo_hash_order.py:56
    list(...) over a set iterates in PYTHONHASHSEED-dependent order;
    sampling or indexing from it changes every process
    fix: Use sorted(...) or a list/tuple literal, or run with PYTHONHASHSEED=0

summary: 2 high, 0 medium, 1 info
```

**3. Localize the divergence between two recorded runs:**

```
$ seedtrace compare <runA> <runB>
------------------------------------------------------------------------
FIRST DIVERGENCE: checkpoint #0  mark='pick_0'
  RNG streams: all EQUAL -> values diverged without RNG-state drift
  diverged recorded values at positions: [0]
  environment differs -> pythonhashseed: A=1 B=7

WHAT THIS MEANS
  - RNG stream states are IDENTICAL at this checkpoint but recorded
    values differ - the divergence is NOT from an unseeded generator.
    Prime suspects: PYTHONHASHSEED-order-dependent iteration feeding a
    choice/sample, filesystem/network/clock inputs, thread scheduling,
    or GPU atomics.
  - PYTHONHASHSEED differs between runs (1 vs 7) - this alone can
    reorder sets/dicts and change sampling results.
```

That classification is the core trick: **RNG equal + values different**
instantly exonerates your seeding and indicts hash-order iteration,
GPU atomics, or ambient inputs. Search space halved.

**4. Prove the fix:**

```
$ seedtrace run --seed 42 --hashseed 0 -- examples/demo_hash_order.py   # twice
$ seedtrace compare <runA> <runB>
RESULT: runs are identical across all recorded checkpoints.
        8 checkpoint(s) compared, no divergence found.
```

## Install

```bash
# from git (PyPI release imminent - `pip install seedtrace` will work once v0.1.0 is published):
pip install git+https://github.com/realgauravvyas/seedtrace.git
```

Zero runtime dependencies. `numpy` and `torch` streams are fingerprinted
automatically *when present* — nothing is ever force-imported.

## 60-second tour

**1. Instrument (optional but powerful).** Two lines anywhere:

```python
import seedtrace

seedtrace.auto()                      # top of your entry point

seedtrace.mark("epoch_2_end", model.weight)   # at checkpoints you care about
```

Marks fingerprint every RNG stream (`python.random`, `numpy.random`,
`torch.cpu`, `torch.cuda.N`) and digest any attached values — tensors,
arrays, scalars, nested containers — bit-exactly. Skip this step and the
CLI still records both runs; you lose checkpoint resolution, not function.

**2. Run twice under the tracer:**

```bash
seedtrace run --seed 42 -- python train.py
seedtrace run --seed 42 -- python train.py
seedtrace list                        # shows recorded run ids
seedtrace compare <runA> <runB>       # first divergence, explained
```

**3. Or skip instrumentation entirely and measure the damage:**

```bash
seedtrace variance -n 5 --seed 42 -- python train.py --epochs 1
# runs 5 repeats, extracts the metric from stdout, prints mean/stdev
# exit code 1 when stdev exceeds --tol  ->  use it as a CI gate
```

**4. Static audit of a repo (no execution needed):**

```bash
seedtrace audit . --fail-on high      # pre-merge reproducibility check
```

**5. Get the hardening snippet:**

```bash
seedtrace determinism-patch -o determinism_patch.py
```

## How it works

| Piece | What it does |
|---|---|
| `capture` | SHA-256 fingerprints of every RNG stream's raw state; bit-exact digests of tensors/arrays/values |
| `session` | Ordered checkpoint recording flushed to `.seedtrace/runs/<id>/run.json` (plain JSON, diff-able, git-diff-able) |
| `compare` | Aligns two runs by checkpoint index *and* name, finds the **earliest** divergence, and classifies it: RNG-state drift vs. value drift with equal RNG state (the hash-order/GPU-atomics signature) vs. control-flow split |
| `audit` | AST scanner + live-environment checks for the known nondeterminism hazards |
| `patch` | A conservative, commented determinism-hardening snippet you opt into knowingly |

The classification in `compare` is the interesting part: if RNG states
are *equal* but recorded *values* differ, your generator seeding is
innocent — the cause is hash-order iteration, filesystem/network/clock
inputs, thread scheduling, or GPU atomics. That single distinction
halves the search space immediately.

## Hazards the audit knows about

- `list(set(...))` / unsorted `os.listdir` feeding sampling or ordering decisions
- `PYTHONHASHSEED` unpinned (the default!) while str/bytes hashing is load-bearing
- `torch.backends.cudnn.benchmark = True`, TF32 switches
- Known non-deterministic CUDA ops (`scatter_add_`, `index_put_`, `grid_sampler_*`, ...)
- `DataLoader(num_workers>0)` without `worker_init_fn`/`generator`
- Modules that consume randomness without ever seeding it
- `use_deterministic_algorithms` never called in torch-using code (info)

New hazards are one function each — see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Limitations (honest ones)

- Divergence *localization* needs `mark()` calls; without them you get
  audit + variance, not bisection.
- Threaded nondeterminism inside a single op (atomics *between* two marks)
  shows up as "values differ, RNG equal" — a correct but coarse signal.
- Python 3.9+. Notebooks work via the `seedtrace.auto()` API (no CLI wrapper).
- Distributed/multi-process training: only per-process streams are captured;
  cross-rank nondeterminism is on the roadmap.

## Related work

- [PyTorch Reproducibility notes](https://pytorch.org/docs/stable/notes/randomness.html) — the canonical *flags*; `seedtrace` is the *debugger* that tells you which flags you actually need.
- `torch.use_deterministic_algorithms(warn_only=True)` — surfaces op-level warnings at runtime; `seedtrace audit` finds them statically before you burn a GPU hour.
- We are not aware of another standalone tool that bisects run-to-run divergence by RNG-stream fingerprint. If one exists — open an issue and we'll position against it honestly.

## Development

```bash
pip install -e .[dev]
pytest
ruff check src tests examples
```

## License

MIT — see [LICENSE](LICENSE).
