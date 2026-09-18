"""Small shared helpers: stable hashing and JSON-safe conversions."""

from __future__ import annotations

import hashlib
import math
from typing import Any

DIGEST_PREFIX = "sha256:"


def sha_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_of_object(obj: Any) -> str:
    """Hash an arbitrary object into a stable digest string.

    Handles the cases that matter for reproducibility debugging:
    bytes, numbers (bit-exact via repr, NaN/Inf safe), strings, and
    nested dicts/lists/tuples/sets. Anything else falls back to repr,
    which is best-effort but never raises.
    """
    return DIGEST_PREFIX + sha_hex(_canonical_bytes(obj))


def _canonical_bytes(obj: Any) -> bytes:
    if isinstance(obj, bytes):
        return b"b" + obj
    if isinstance(obj, bytearray):
        return b"b" + bytes(obj)
    if isinstance(obj, bool):
        return b"B" + (b"1" if obj else b"0")
    if isinstance(obj, (int,)):
        return b"i" + repr(obj).encode()
    if isinstance(obj, float):
        if math.isnan(obj):
            return b"f<nan>"
        if math.isinf(obj):
            return b"f<inf>" if obj > 0 else b"f<-inf>"
        return b"f" + repr(obj).encode()
    if isinstance(obj, str):
        return b"s" + obj.encode("utf-8", errors="replace")
    if isinstance(obj, dict):
        items = sorted(
            ((_canonical_bytes(k), _canonical_bytes(v)) for k, v in obj.items()),
            key=lambda kv: kv[0],
        )
        return b"d(" + b"".join(k + b"=" + v + b";" for k, v in items) + b")"
    if isinstance(obj, (list, tuple)):
        return b"l(" + b"".join(_canonical_bytes(x) + b";" for x in obj) + b")"
    if isinstance(obj, (set, frozenset)):
        parts = sorted(_canonical_bytes(x) for x in obj)
        return b"e(" + b"".join(p + b";" for p in parts) + b")"
    # Objects exposing a buffer-protocol state (e.g. numpy arrays) are
    # handled upstream in capture; repr() is the last-resort fallback.
    return b"r" + repr(obj).encode("utf-8", errors="replace")


def short(digest: str, n: int = 12) -> str:
    """Trim a digest for display."""
    if digest.startswith(DIGEST_PREFIX):
        digest = digest[len(DIGEST_PREFIX) :]
    return digest[:n]
