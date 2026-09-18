"""seedtrace: find out why two runs with the same seed gave different results.

Minimal usage::

    import seedtrace

    seedtrace.auto(seed=42)          # once, at the top of your entry point

    for epoch in range(epochs):
        ...
        seedtrace.mark(f"epoch_{epoch}_end", model)   # any number of times

Then::

    seedtrace run --seed 42 -- python train.py     # twice
    seedtrace compare <runA> <runB>                # first divergence
    seedtrace audit train.py                       # the exact hazardous lines

Both runs can also be launched normally (``python train.py``); marks
still record as long as ``auto()`` runs. The CLI wrapper merely adds
convenience (seed/hash-seed pinning, run bookkeeping).
"""

from __future__ import annotations

from .session import Session, active_session, auto, mark, reset

__version__ = "0.1.0"

__all__ = ["auto", "mark", "active_session", "reset", "Session", "__version__"]
