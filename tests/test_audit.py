"""Tests for the static + environment audit."""

from __future__ import annotations

import os
from unittest import mock

from seedtrace.audit import audit_environment, audit_file, audit_source

HAZARD_SOURCE = """
import os
import random
import torch
from torch.utils.data import DataLoader

random.seed(1)
torch.manual_seed(0)

pool = list({"alpha", "beta", "gamma", "delta"})
pick = random.choice(pool)

files_raw = os.listdir(".")
files_ok = sorted(os.listdir("."))

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

x = torch.zeros(4)
idx = torch.zeros(4, dtype=torch.long)
x.scatter_add_(0, idx, x)

loader = DataLoader([1, 2, 3], num_workers=4, shuffle=True)
"""


def _ids(findings):
    return {f.id for f in findings}


def test_flags_set_order_and_ops(tmp_path):
    p = tmp_path / "haz.py"
    p.write_text(HAZARD_SOURCE, encoding="utf-8")
    report = audit_file(str(p))
    ids = _ids(report.findings)
    assert "set-iteration-order" in ids
    assert "unsorted-listdir" in ids
    assert "cudnn-benchmark-on" in ids
    assert "tf32-enabled" in ids
    assert "nondeterministic-op:scatter_add_" in ids
    assert "dataloader-unseeded-workers" in ids
    # sorted() version must appear exactly once as unflagged: count listdir findings
    listdir_findings = [f for f in report.findings if f.id == "unsorted-listdir"]
    assert len(listdir_findings) == 1


def test_set_bound_variable_flagged(tmp_path):
    src = (
        "import random\n"
        "random.seed(1)\n"
        "VOCAB = {'alpha', 'beta', 'gamma', 'delta'}\n"
        "x = random.choice(list(VOCAB))\n"
    )
    p = tmp_path / "setvar.py"
    p.write_text(src, encoding="utf-8")
    report = audit_file(str(p))
    assert "set-iteration-order" in _ids(report.findings)


def test_unseeded_module_flagged(tmp_path):
    p = tmp_path / "loose.py"
    p.write_text("import random\nx = random.random()\n", encoding="utf-8")
    report = audit_file(str(p))
    assert "unseeded-random-usage" in _ids(report.findings)


def test_clean_module_no_findings(tmp_path):
    p = tmp_path / "clean.py"
    p.write_text(
        "import random\nrandom.seed(7)\nx = random.Random(1).random()\n",
        encoding="utf-8",
    )
    report = audit_file(str(p))
    assert report.findings == []


def test_dataloader_with_worker_init_fn_is_clean(tmp_path):
    src = (
        "import torch\n"
        "from torch.utils.data import DataLoader\n"
        "def winfo(i): pass\n"
        "loader = DataLoader([1], num_workers=2, worker_init_fn=winfo)\n"
        "torch.manual_seed(1)\n"
        "torch.use_deterministic_algorithms(True)\n"
    )
    p = tmp_path / "dl.py"
    p.write_text(src, encoding="utf-8")
    report = audit_file(str(p))
    assert "dataloader-unseeded-workers" not in _ids(report.findings)


def test_audit_source_directory_skips_venv(tmp_path):
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "bad.py").write_text("import os\nos.listdir('.')\n", encoding="utf-8")
    (tmp_path / "good.py").write_text("x = 1\n", encoding="utf-8")
    reports = audit_source(str(tmp_path))
    assert [os.path.basename(r.path) for r in reports] == ["good.py"]


def test_environment_flags_unpinned_hashseed():
    env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
    with mock.patch.dict(os.environ, env, clear=True):
        findings = audit_environment()
    assert "hash-seed-unpinned" in _ids(findings)


def test_environment_ok_when_pinned():
    env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
    env["PYTHONHASHSEED"] = "0"
    with mock.patch.dict(os.environ, env, clear=True):
        findings = audit_environment()
    assert "hash-seed-unpinned" not in _ids(findings)
