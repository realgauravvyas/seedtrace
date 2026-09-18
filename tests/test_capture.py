"""Unit tests for RNG stream capture and object digest stability."""

from __future__ import annotations

import random

import pytest

from seedtrace._utils import sha_of_object
from seedtrace.capture import capture_streams


def test_streams_include_python_random():
    streams = capture_streams()
    assert "python.random" in streams
    assert streams["python.random"].startswith("sha256:")


def test_stream_digest_changes_after_consumption():
    random.seed(1)
    before = capture_streams()["python.random"]
    random.random()
    after = capture_streams()["python.random"]
    assert before != after


def test_same_seed_same_stream_digest():
    random.seed(123)
    a = capture_streams()["python.random"]
    random.seed(123)
    b = capture_streams()["python.random"]
    assert a == b


@pytest.mark.parametrize(
    "obj",
    [1, -7, 0.5, float("nan"), "str", b"bytes", [1, 2], {"a": 1}, (1, "x"), {1, 2, 3}],
)
def test_digest_stable_and_type_separated(obj):
    assert sha_of_object(obj) == sha_of_object(obj)
    # different type tags keep 1 and "1" and True distinct
    assert sha_of_object(1) != sha_of_object("1")
    assert sha_of_object(True) != sha_of_object(1)


def test_dict_order_does_not_change_digest():
    assert sha_of_object({"a": 1, "b": 2}) == sha_of_object({"b": 2, "a": 1})


def test_nan_and_inf_do_not_raise():
    assert sha_of_object(float("inf")) == sha_of_object(float("inf"))
    assert sha_of_object(float("nan")) == sha_of_object(float("nan"))


def test_numpy_stream_present_and_moves():
    np = pytest.importorskip("numpy")
    streams = capture_streams()
    assert "numpy.random" in streams
    np.random.seed(7)
    a = capture_streams()["numpy.random"]
    np.random.rand()
    b = capture_streams()["numpy.random"]
    assert a != b


def test_tensor_digest_via_mark_objects():
    np = pytest.importorskip("numpy")
    from seedtrace.capture import digest_tensor

    arr = np.arange(6, dtype=np.float64)
    d1 = digest_tensor(arr)
    d2 = digest_tensor(np.arange(6, dtype=np.float64))
    d3 = digest_tensor(arr + 1.0)
    assert d1 == d2
    assert d1 != d3
