"""Session recording + JSON round-trip tests."""

from __future__ import annotations

import json
import random

import seedtrace
from seedtrace.session import Session


def test_session_roundtrip(tmp_path):
    s = Session(runs_dir=str(tmp_path), seed=42, label="unit")
    random.seed(5)
    s.mark("a", 1.5)
    s.mark("b", "text")
    path = s.finalize()
    assert path is not None and path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["run_id"] == s.run_id
    assert data["seed"] == 42
    assert data["label"] == "unit"
    assert [m["name"] for m in data["marks"]] == ["a", "b"]
    assert "python.random" in data["marks"][0]["streams"]


def test_finalize_is_idempotent(tmp_path):
    s = Session(runs_dir=str(tmp_path))
    assert s.finalize() is not None
    assert s.finalize() is None


def test_public_api_singleton_and_reset():
    seedtrace.reset()
    session = seedtrace.auto(runs_dir="<none>", seed=None)
    assert seedtrace.active_session() is session
    checkpoint = seedtrace.mark("x", 1)
    assert checkpoint is not None and checkpoint["name"] == "x"
    seedtrace.reset()
    assert seedtrace.mark("y") is None
    assert seedtrace.active_session() is None


def test_auto_seeds_global_rng():
    seedtrace.reset()
    seedtrace.auto(runs_dir="<none>", seed=777)
    first = random.random()
    seedtrace.reset()
    seedtrace.auto(runs_dir="<none>", seed=777)
    second = random.random()
    seedtrace.reset()
    assert first == second
