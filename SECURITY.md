# Security Policy

## Scope

seedtrace runs *your own* scripts as child processes and writes plain
JSON under `.seedtrace/runs/`. It performs no network I/O, has no
runtime dependencies, and never uploads anything.

## What we guarantee

- **No secrets collection.** Run records contain argv, environment
  *flag names* (values only for a fixed allowlist: `PYTHONHASHSEED`,
  `CUBLAS_WORKSPACE_CONFIG`, thread-count and CUDA debugging vars),
  package versions, and a hostname. Tokens, keys, and unrelated
  environment variables are never read into run records.
- **Value digests only.** `mark()` stores SHA-256 digests, not data.
  Digesting a tensor never persists tensor contents.

## Reporting a vulnerability

Email the maintainer (see `pyproject.toml` authors) with "seedtrace
security" in the subject. Please do not open a public issue for live
vulnerabilities. We aim to acknowledge within 5 business days and will
publish a `SECURITY_ADVISORY`-style changelog entry for fixed issues.

## Version support

| Version | Supported |
|---|---|
| 0.1.x | ✅ |
| < 0.1  | ❌ |
