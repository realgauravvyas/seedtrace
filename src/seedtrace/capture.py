"""RNG stream fingerprinting.

A "stream" is any independent source of randomness whose state can
diverge between two runs:

- ``python.random``   - the global ``random`` module (Mersenne Twister)
- ``numpy.random``    - the legacy ``np.random`` global (MT19937)
- ``torch.cpu``       - ``torch``'s global CPU generator
- ``torch.cuda.<i>``  - CUDA generator state per device (when initialized)

Every stream is reduced to a short digest of its raw state, so two runs
can be compared cheaply and safely (no pickling of foreign objects).
"""

from __future__ import annotations

import os
import random
from typing import Dict, Optional

from ._utils import sha_hex

STREAM_PYTHON = "python.random"
STREAM_NUMPY = "numpy.random"
STREAM_TORCH_CPU = "torch.cpu"


def _torch_module() -> Optional[object]:
    """Return ``torch`` if already imported or importable, else None."""
    import sys

    mod = sys.modules.get("torch")
    if mod is not None:
        return mod
    try:  # opportunistic: never force a heavy import during normal use
        import importlib.util

        if importlib.util.find_spec("torch") is None:
            return None
    except (ValueError, ModuleNotFoundError, ImportError):
        return None
    try:
        import torch

        return torch
    except Exception:
        return None


def _numpy_module() -> Optional[object]:
    import sys

    mod = sys.modules.get("numpy")
    if mod is not None:
        return mod
    try:
        import numpy

        return numpy
    except Exception:
        return None


def capture_streams() -> Dict[str, str]:
    """Digest of every discoverable RNG stream, keyed by stream name.

    Absent libraries (numpy/torch not installed) are simply skipped.
    A stream that exists but cannot be read is recorded with the digest
    ``"sha256:unavailable"`` so comparisons still notice its presence.
    """
    streams: Dict[str, str] = {}

    try:
        state = random.getstate()
        # state = (version, tuple_of_ints, gauss_next)
        payload = repr(state).encode("utf-8", errors="replace")
        streams[STREAM_PYTHON] = "sha256:" + sha_hex(payload)
    except Exception:
        streams[STREAM_PYTHON] = "sha256:unavailable"

    np = _numpy_module()
    if np is not None:
        try:
            st = np.random.get_state()
            # st = ('MT19937', keys_array, pos, has_gauss, cached_gaussian)
            keys = st[1]
            buf = (
                keys.tobytes()
                if hasattr(keys, "tobytes")
                else repr(keys).encode("ascii", errors="replace")
            )
            extra = repr((int(st[2]), int(st[3]), float(st[4]))).encode("ascii")
            streams[STREAM_NUMPY] = "sha256:" + sha_hex(buf + b"|" + extra)
        except Exception:
            streams[STREAM_NUMPY] = "sha256:unavailable"

    torch = _torch_module()
    if torch is not None:
        try:
            cpu_state = torch.random.get_rng_state()
            streams[STREAM_TORCH_CPU] = "sha256:" + sha_hex(cpu_state.numpy().tobytes())
        except Exception:
            streams[STREAM_TORCH_CPU] = "sha256:unavailable"
        try:
            if torch.cuda.is_initialized():
                for i in range(torch.cuda.device_count()):
                    gen = torch.cuda.default_generators[i]
                    if gen is not None:
                        seed_offset = gen.get_seed()
                        payload = repr(tuple(seed_offset)).encode("ascii")
                        streams[f"torch.cuda.{i}"] = "sha256:" + sha_hex(payload)
        except Exception:
            pass

    return streams


def capture_hash_seed() -> Optional[int]:
    """The PYTHONHASHSEED value for this process, if it was pinned."""
    raw = os.environ.get("PYTHONHASHSEED")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def digest_tensor(obj: object) -> Optional[str]:
    """Digest a torch tensor or numpy array by value + shape + dtype.

    Returns None when *obj* is not an array-like we know how to read.
    """
    torch = _torch_module()
    np = _numpy_module()

    if torch is not None and hasattr(obj, "detach") and hasattr(obj, "cpu"):
        try:
            t = obj.detach().cpu().contiguous()
            meta = repr((tuple(t.shape), str(t.dtype))).encode("ascii")
            return "sha256:" + sha_hex(meta + b"|" + t.numpy().tobytes())
        except Exception:
            return "sha256:unavailable"

    if np is not None and isinstance(obj, np.ndarray):
        try:
            meta = repr((tuple(obj.shape), str(obj.dtype))).encode("ascii")
            return "sha256:" + sha_hex(meta + b"|" + obj.tobytes())
        except Exception:
            return "sha256:unavailable"

    return None
