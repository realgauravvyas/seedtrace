"""The classic silent killer: PYTHONHASHSEED-dependent set ordering.

Run this script twice with the SAME seed and it prints a different
"score" every time -- because `random.choice` samples from a list
built out of a *set of strings*, whose iteration order is randomized
per process. No RNG is unseeded; the seed is fine. This is exactly
the bug people lose hours to.

    python examples/demo_hash_order.py
    python examples/demo_hash_order.py           # different result, same seed

Reproduce the diagnosis workflow:

    seedtrace variance -n 3 --seed 42 -- examples/demo_hash_order.py   # stdev > 0
    seedtrace audit examples/demo_hash_order.py                        # flags line
    seedtrace run --seed 42 --hashseed 0 -- examples/demo_hash_order.py  # twice
    seedtrace run --seed 42 --hashseed 0 -- examples/demo_hash_order.py
    seedtrace compare <runA> <runB>                               # identical

Without --hashseed 0, the two `seedtrace run` invocations diverge at
mark "features_built" and `compare` points straight at it.
"""

from __future__ import annotations

import random

import seedtrace

FEATURES = {
    "hue",
    "saturation",
    "luminance",
    "contrast",
    "texture",
    "edges",
    "corners",
    "blob",
    "gradient",
    "entropy",
    "variance",
    "kurtosis",
    "skewness",
    "dominant",
    "palette",
    "histogram",
    "fourier",
    "wavelet",
}

seedtrace.auto(seed=None)
random.seed(42)  # pinned -- and yet the result still varies. That's the point.

selected = []
for i in range(6):
    pool = list(FEATURES)  # iteration order depends on PYTHONHASHSEED!
    pick = random.choice(pool)
    selected.append(pick)
    seedtrace.mark(f"pick_{i}", pick)

seedtrace.mark("features_built", tuple(sorted(selected)))

# A tiny deterministic toy model over the (nondeterministically chosen)
# features, so the final score reflects which features were sampled.
score = sum(len(name) * (idx + 1) for idx, name in enumerate(selected)) / 100.0
seedtrace.mark("final", round(score, 6))
print(f"selected: {selected}")
print(f"final score: {score}")
