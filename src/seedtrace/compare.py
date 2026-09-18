"""Compare two recorded runs and locate the first divergence.

The comparison aligns checkpoints by sequence index *and* name, then
reports the earliest point where the runs differ, which stream(s)
diverged, and (from the environment snapshots) whether the process
configuration itself differed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Divergence:
    """Result of comparing two runs."""

    identical: bool
    first_divergence_index: Optional[int] = None
    mark_name: Optional[str] = None
    diverged_streams: List[str] = field(default_factory=list)
    diverged_digest_positions: List[int] = field(default_factory=list)
    marks_compared: int = 0
    marks_a: int = 0
    marks_b: int = 0
    name_mismatch: bool = False
    env_differences: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    seeds: Dict[str, Optional[int]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identical": self.identical,
            "first_divergence_index": self.first_divergence_index,
            "mark_name": self.mark_name,
            "diverged_streams": self.diverged_streams,
            "diverged_digest_positions": self.diverged_digest_positions,
            "marks_compared": self.marks_compared,
            "marks_a": self.marks_a,
            "marks_b": self.marks_b,
            "name_mismatch": self.name_mismatch,
            "env_differences": self.env_differences,
            "seeds": self.seeds,
            "notes": self.notes,
        }


def load_run(path: str | Path) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "marks" not in data:
        raise ValueError(f"{path} does not look like a seedtrace run file")
    return data


def _streams_differ(a: Dict[str, str], b: Dict[str, str]) -> List[str]:
    keys = sorted(set(a) | set(b))
    return [k for k in keys if a.get(k) != b.get(k)]


def compare_runs(run_a: Dict[str, Any], run_b: Dict[str, Any]) -> Divergence:
    marks_a: List[Dict[str, Any]] = run_a.get("marks", [])
    marks_b: List[Dict[str, Any]] = run_b.get("marks", [])

    result = Divergence(
        identical=True,
        marks_a=len(marks_a),
        marks_b=len(marks_b),
        seeds={"a": run_a.get("seed"), "b": run_b.get("seed")},
    )

    # Environment-level differences explain (or excuse) divergence.
    for key in ("pythonhashseed", "python", "packages"):
        va = run_a.get("environment", {}).get(key)
        vb = run_b.get("environment", {}).get(key)
        if va != vb:
            result.env_differences[key] = {"a": va, "b": vb}
    if run_a.get("seed") != run_b.get("seed"):
        result.env_differences["seed"] = {"a": run_a.get("seed"), "b": run_b.get("seed")}

    n = min(len(marks_a), len(marks_b))
    for i in range(n):
        ma, mb = marks_a[i], marks_b[i]
        result.marks_compared = i + 1
        if ma.get("name") != mb.get("name"):
            result.name_mismatch = True
            result.identical = False
            result.first_divergence_index = i
            result.mark_name = f"{ma.get('name')!r} vs {mb.get('name')!r}"
            result.notes.append(
                "checkpoint names diverged: control flow itself differs between runs "
                "(a branch is consuming randomness conditionally)"
            )
            return result

        streams = _streams_differ(ma.get("streams", {}), mb.get("streams", {}))
        digests = _digest_diff_positions(ma.get("digests", []), mb.get("digests", []))
        if streams or digests:
            result.identical = False
            result.first_divergence_index = i
            result.mark_name = ma.get("name")
            result.diverged_streams = streams
            result.diverged_digest_positions = digests
            _explain(result, streams, digests, i, ma, mb)
            return result

    if len(marks_a) != len(marks_b):
        result.identical = False
        result.first_divergence_index = n
        result.notes.append(
            f"runs recorded the same marks for the first {n} checkpoints, then one "
            f"run ended early ({len(marks_a)} vs {len(marks_b)} marks): control flow "
            "diverged after the last shared checkpoint"
        )
    return result


def _digest_diff_positions(da: List[str], db: List[str]) -> List[int]:
    return [i for i in range(min(len(da), len(db))) if da[i] != db[i]]


def _explain(
    result: Divergence,
    streams: List[str],
    digests: List[int],
    index: int,
    ma: Dict[str, Any],
    mb: Dict[str, Any],
) -> None:
    rng_state_same = not streams
    if rng_state_same and digests:
        result.notes.append(
            "RNG stream states are IDENTICAL at this checkpoint but recorded values "
            "differ - the divergence is NOT from an unseeded generator. Prime suspects: "
            "PYTHONHASHSEED-order-dependent iteration feeding a choice/sample, "
            "filesystem/network/clock inputs, thread scheduling, or GPU atomics."
        )
        hashseed = result.env_differences.get("pythonhashseed")
        if hashseed:
            result.notes.append(
                f"PYTHONHASHSEED differs between runs ({hashseed['a']!r} vs {hashseed['b']!r}) "
                "- this alone can reorder sets/dicts and change sampling results."
            )
    elif streams and not digests:
        result.notes.append(
            "an RNG stream was consumed a different number of times before this "
            "checkpoint (state diverged while recorded values still matched)"
        )
    else:
        result.notes.append(
            "both RNG state and values diverged at this checkpoint; the streams "
            "listed above are the earliest evidence of nondeterminism"
        )
    if index == 0:
        result.notes.append(
            "this is the FIRST checkpoint - randomness diverges before or at the "
            "first mark; audit module-level/import-time randomness"
        )


def classify_stream_note(stream: str) -> str:
    if stream == "python.random":
        return "the built-in random module"
    if stream == "numpy.random":
        return "numpy's legacy global RNG"
    if stream.startswith("torch.cuda"):
        return "PyTorch's CUDA RNG (device-side generator)"
    if stream == "torch.cpu":
        return "PyTorch's CPU RNG"
    return stream
