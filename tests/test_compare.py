"""Tests for checkpoint comparison / divergence bisection."""

from __future__ import annotations

from seedtrace.compare import compare_runs
from seedtrace.session import Session, _digest_object


def _run(marks, seed=42, hashseed=0):
    s = Session(runs_dir="<memory>", seed=seed, record_environment=False)
    s.environment["pythonhashseed"] = hashseed
    for name, *objs in marks:
        s.mark(name, *objs)
    return s.to_dict()


def test_identical_runs_report_identical():
    a = _run([("m0", 1), ("m1", 2.5)])
    b = _run([("m0", 1), ("m1", 2.5)])
    div = compare_runs(a, b)
    assert div.identical
    assert div.first_divergence_index is None
    assert div.marks_compared == 2


def test_finds_first_diverging_stream():
    a = _run([("m0", 1), ("m1", 2)])
    b = _run([("m0", 1), ("m1", 2)])
    b["marks"][1]["streams"]["python.random"] = "sha256:DIFFERENT"
    div = compare_runs(a, b)
    assert not div.identical
    assert div.first_divergence_index == 1
    assert div.mark_name == "m1"
    assert div.diverged_streams == ["python.random"]
    assert not div.diverged_digest_positions
    assert any("consumed a different number of times" in n for n in div.notes)


def test_values_differ_while_streams_equal_explains_hash_order():
    a = _run([("m0", "alpha")])
    b = _run([("m0", "beta")])
    div = compare_runs(a, b)
    assert not div.identical
    assert div.diverged_streams == []
    assert div.diverged_digest_positions == [0]
    assert any("IDENTICAL" in n for n in div.notes)


def test_hashseed_env_difference_surfaced():
    a = _run([("m0", 1)], hashseed=1)
    b = _run([("m0", 1)], hashseed=2)
    div = compare_runs(a, b)
    assert div.first_divergence_index is None  # marks themselves match
    assert "pythonhashseed" in div.env_differences


def test_name_mismatch_detected():
    a = _run([("train", 1)])
    b = _run([("eval", 1)])
    div = compare_runs(a, b)
    assert div.name_mismatch
    assert div.first_divergence_index == 0
    assert any("control flow" in n for n in div.notes)


def test_unequal_mark_counts():
    a = _run([("m0", 1), ("m1", 2)])
    b = _run([("m0", 1)])
    div = compare_runs(a, b)
    assert not div.identical
    assert div.first_divergence_index == 1
    assert any("ended early" in n for n in div.notes)


def test_digest_object_covers_arrays_and_scalars():
    assert _digest_object(3.14) == _digest_object(3.14)
    assert _digest_object(3.14) != _digest_object(3.15)
    assert _digest_object("x").startswith("sha256:")
