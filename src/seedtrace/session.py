"""Session recording: environment snapshot + ordered checkpoints.

A seedtrace *session* wraps one execution of a user program. Every call
to :func:`mark` records:

- the fingerprint of all RNG streams at that instant,
- value digests of any objects the user attached to the mark,
- a wall-clock timestamp and monotonic sequence number.

On exit the session is written to ``<runs_dir>/<run_id>/run.json`` and
can be inspected or diffed with the CLI.
"""

from __future__ import annotations

import atexit
import json
import os
import socket
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ._utils import sha_of_object
from .capture import capture_hash_seed, capture_streams, digest_tensor

DEFAULT_RUNS_DIR = os.path.join(".seedtrace", "runs")

ENV_RUN_DIR = "SEEDTRACE_RUN_DIR"
ENV_SEED = "SEEDTRACE_SEED"
ENV_LABEL = "SEEDTRACE_LABEL"


def _package_version() -> str:
    from . import __version__

    return __version__


def _installed_versions() -> Dict[str, str]:
    """Best-effort versions of libraries that commonly affect numerics."""
    versions: Dict[str, str] = {}
    for name in ("numpy", "torch", "scipy", "pandas", "scikit-learn"):
        try:
            mod = __import__(name)
            versions[name] = str(getattr(mod, "__version__", "?"))
        except Exception:
            continue
    return versions


class Session:
    """One recorded execution. Normally accessed via module functions."""

    def __init__(
        self,
        runs_dir: str = DEFAULT_RUNS_DIR,
        label: Optional[str] = None,
        seed: Optional[int] = None,
        record_environment: bool = True,
    ) -> None:
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        self.run_id = f"{ts}-{uuid.uuid4().hex[:8]}"
        self.runs_dir = runs_dir
        self.label = label
        self.seed = seed
        self.started_at = time.time()
        self._marks: List[Dict[str, Any]] = []
        self._finalized = False

        self.environment: Dict[str, Any] = {}
        if record_environment:
            self.environment = {
                "python": sys.version.split()[0],
                "platform": sys.platform,
                "hostname": socket.gethostname(),
                "argv": [os.path.basename(a) if i == 0 else a for i, a in enumerate(sys.argv)],
                "packages": _installed_versions(),
                "pythonhashseed": capture_hash_seed(),
                "env_flags": {
                    k: os.environ.get(k)
                    for k in (
                        "PYTHONHASHSEED",
                        "CUBLAS_WORKSPACE_CONFIG",
                        "MKL_NUM_THREADS",
                        "OMP_NUM_THREADS",
                        "CUDA_LAUNCH_BLOCKING",
                        "DETERMINISTIC_EXTENSIONS",
                    )
                },
            }

    # -- recording -----------------------------------------------------

    def mark(self, name: str, *objects: Any) -> Dict[str, Any]:
        checkpoint: Dict[str, Any] = {
            "index": len(self._marks),
            "name": name,
            "t": round(time.time() - self.started_at, 6),
            "streams": capture_streams(),
        }
        if objects:
            checkpoint["digests"] = [_digest_object(o) for o in objects]
        self._marks.append(checkpoint)
        return checkpoint

    # -- persistence ---------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seedtrace_version": _package_version(),
            "run_id": self.run_id,
            "label": self.label,
            "seed": self.seed,
            "duration_s": round(time.time() - self.started_at, 3),
            "environment": self.environment,
            "marks": self._marks,
        }

    def finalize(self) -> Optional[Path]:
        """Write run.json. Returns the run directory, or None on failure."""
        if self._finalized:
            return None
        self._finalized = True
        try:
            run_dir = Path(self.runs_dir) / self.run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            path = run_dir / "run.json"
            path.write_text(
                json.dumps(self.to_dict(), indent=1, sort_keys=False),
                encoding="utf-8",
            )
            return path
        except OSError:
            # Recording must never crash the user's program.
            return None


def _digest_object(obj: Any) -> str:
    tensor_digest = digest_tensor(obj)
    if tensor_digest is not None:
        return tensor_digest
    try:
        return sha_of_object(obj)
    except Exception:  # pragma: no cover - defensive
        return "sha256:unavailable"


# -- module-level singleton -------------------------------------------

_ACTIVE: Optional[Session] = None


def auto(
    seed: Optional[int] = None,
    runs_dir: Optional[str] = None,
    label: Optional[str] = None,
) -> Session:
    """Start (or return the already-started) recording session.

    If *seed* is given, ``random`` / ``numpy`` / ``torch`` globals are
    seeded with it before the caller's code executes. The session is
    flushed automatically at interpreter exit.
    """
    global _ACTIVE
    if _ACTIVE is not None:
        return _ACTIVE

    if runs_dir is None:
        runs_dir = os.environ.get(ENV_RUN_DIR, DEFAULT_RUNS_DIR)
    if label is None:
        label = os.environ.get(ENV_LABEL) or None
    if seed is None:
        env_seed = os.environ.get(ENV_SEED)
        if env_seed not in (None, ""):
            try:
                seed = int(env_seed)
            except ValueError:
                seed = None

    session = Session(runs_dir=runs_dir, label=label, seed=seed)
    if seed is not None:
        _seed_everything(seed)
    _ACTIVE = session
    atexit.register(session.finalize)
    return session


def _seed_everything(seed: int) -> None:
    import random as _random

    _random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def mark(name: str, *objects: Any) -> Optional[Dict[str, Any]]:
    """Record a checkpoint. No-op (returns None) if auto() was never called."""
    if _ACTIVE is None:
        return None
    return _ACTIVE.mark(name, *objects)


def active_session() -> Optional[Session]:
    return _ACTIVE


def reset() -> None:
    """Drop the active session (used by tests). Does not write files."""
    global _ACTIVE
    if _ACTIVE is not None:
        _ACTIVE._finalized = True  # suppress atexit write
    _ACTIVE = None


def find_run(runs_dir: str, ref: str) -> Optional[Path]:
    """Resolve a run reference: a run dir path, run.json path, or run id."""
    p = Path(ref)
    if p.is_dir() and (p / "run.json").exists():
        return p / "run.json"
    if p.is_file() and p.name == "run.json":
        return p
    candidate = Path(runs_dir) / ref / "run.json"
    if candidate.exists():
        return candidate
    return None


def list_runs(runs_dir: str) -> Sequence[Path]:
    root = Path(runs_dir)
    if not root.is_dir():
        return []
    return sorted(root.glob("*/run.json"))
