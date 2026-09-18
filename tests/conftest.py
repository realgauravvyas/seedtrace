"""Shared fixtures: make ``src/`` importable in-process and in children."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

SRC = str(Path(__file__).resolve().parents[1] / "src")


@pytest.fixture(autouse=True)
def _add_src_to_path(monkeypatch):
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    env = os.environ.copy()
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONPATH", env["PYTHONPATH"])


@pytest.fixture
def seedtrace_cli():
    """Invoke the CLI in a subprocess (fully isolated from pytest state)."""
    import subprocess

    def run(*argv, cwd=None, env_extra=None):
        env = os.environ.copy()
        env["PYTHONPATH"] = SRC
        env.pop("SEEDTRACE_SEED", None)
        env.pop("SEEDTRACE_RUN_DIR", None)
        env.pop("SEEDTRACE_LABEL", None)
        if env_extra:
            for k, v in env_extra.items():
                if v is None:
                    env.pop(k, None)
                else:
                    env[k] = v
        return subprocess.run(
            [sys.executable, "-m", "seedtrace", *argv],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
        )

    return run
