# Contributing to seedtrace

Thanks for being here. seedtrace is small on purpose: zero runtime
dependencies, plain-JSON outputs, and one job — making run-to-run
nondeterminism debuggable.

## Ground rules

- **Zero runtime deps stays zero.** numpy/torch are detected
  opportunistically and must remain optional.
- **Reports are plain ASCII.** They get pasted into bug reports and CI
  logs; no unicode arrows, no ANSI colors in command output.
- **Never break the user's program.** Recording failures are swallowed
  silently (`Session.finalize` returning `None` is a valid outcome).
- **Every hazard or stream type ships with a test** that asserts on its
  stable finding `id` (see `tests/test_audit.py`).

## Adding a hazard check

1. Add a detector in `src/seedtrace/audit.py` emitting a
   `Finding(id="kebab-case-stable-id", severity=..., fix=...)`.
2. Add a case in `tests/test_audit.py` with the offending snippet.
3. Mention it in the README "Hazards the audit knows about" list.

## Adding an RNG stream

Extend `capture.capture_streams()`. Contract: returns
`Dict[str, str]` of `sha256:<hex>` digests; must never raise; missing
libraries are skipped, unreadable-but-present streams digest to
`"sha256:unavailable"`.

## Dev setup

```bash
git clone https://github.com/realgauravvyas/seedtrace.git
cd seedtrace
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
ruff check src tests examples && ruff format src tests examples
```

CI runs the test matrix on Linux/macOS/Windows for Python 3.9–3.13;
keep tests subprocess-isolated where they touch global RNG state so the
suite stays order-independent.

## Proposing bigger changes

Open an issue first. Anything touching the `run.json` format needs a
versioned schema discussion — old run files must stay comparable.
