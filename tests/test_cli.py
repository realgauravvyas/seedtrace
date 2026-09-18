"""End-to-end CLI tests: real subprocess runs against tiny scripts."""

from __future__ import annotations

import json

from seedtrace.compare import load_run

DETERMINISTIC = """
import random
random.seed(42)
print(f"score: {random.random():.12f}")
"""

HAZARDOUS = """
import random
import seedtrace

seedtrace.auto()
random.seed(42)
pool = list({"one", "two", "three", "four", "five", "six"})
pick = random.choice(pool)
score = sum(len(p) for p in pool[:3]) / 100.0
seedtrace.mark("draw", pick, score)
print(f"picked: {pick}")
print(f"score: {score}")
"""


def test_run_records_a_run_file(seedtrace_cli, tmp_path):
    script = tmp_path / "det.py"
    script.write_text(DETERMINISTIC, encoding="utf-8")
    runs = tmp_path / "runs"
    proc = seedtrace_cli("run", "--seed", "42", "--runs-dir", str(runs), "--", str(script))
    assert proc.returncode == 0, proc.stderr
    files = list(runs.glob("*/run.json"))
    assert len(files) == 1
    data = load_run(files[0])
    assert data["seed"] == 42
    # the recorded program still printed its output
    assert "score:" in proc.stdout


def test_variance_detects_determinism(seedtrace_cli, tmp_path):
    script = tmp_path / "det.py"
    script.write_text(DETERMINISTIC, encoding="utf-8")
    runs = tmp_path / "runs"
    proc = seedtrace_cli(
        "variance",
        "-n",
        "2",
        "--seed",
        "42",
        "--hashseed",
        "0",
        "--runs-dir",
        str(runs),
        "--",
        str(script),
    )
    assert proc.returncode == 0, proc.stderr
    assert "stdev=0.0" in proc.stdout
    assert "deterministic" in proc.stdout


def test_compare_finds_hash_order_divergence(seedtrace_cli, tmp_path):
    script = tmp_path / "haz.py"
    script.write_text(HAZARDOUS, encoding="utf-8")
    runs = tmp_path / "runs"

    p1 = seedtrace_cli(
        "run",
        "--seed",
        "42",
        "--runs-dir",
        str(runs),
        "--",
        str(script),
        env_extra={"PYTHONHASHSEED": "1"},
    )
    p2 = seedtrace_cli(
        "run",
        "--seed",
        "42",
        "--runs-dir",
        str(runs),
        "--",
        str(script),
        env_extra={"PYTHONHASHSEED": "2"},
    )
    assert p1.returncode == 0 and p2.returncode == 0, (p1.stderr, p2.stderr)
    run_dirs = sorted(runs.glob("*/run.json"))
    assert len(run_dirs) == 2

    proc = seedtrace_cli(
        "compare",
        str(run_dirs[0].parent),
        str(run_dirs[1].parent),
        "--runs-dir",
        str(runs),
        "--json",
    )
    assert proc.returncode == 1  # divergence found
    div = json.loads(proc.stdout)
    assert not div["identical"]
    # RNG streams are equal (seed was pinned); either the mark digests or the
    # recorded environment must expose the hash-seed difference.
    assert (
        div["diverged_digest_positions"]
        or div["diverged_streams"]
        or "pythonhashseed" in div["env_differences"]
    )


def test_compare_clean_between_identical_runs(seedtrace_cli, tmp_path):
    script = tmp_path / "det.py"
    script.write_text(DETERMINISTIC, encoding="utf-8")
    runs = tmp_path / "runs"
    for _ in range(2):
        proc = seedtrace_cli(
            "run", "--seed", "42", "--hashseed", "0", "--runs-dir", str(runs), "--", str(script)
        )
        assert proc.returncode == 0, proc.stderr
    a, b = sorted(runs.glob("*/run.json"))
    proc = seedtrace_cli("compare", str(a.parent), str(b.parent), "--runs-dir", str(runs))
    assert proc.returncode == 0, proc.stdout
    assert "identical" in proc.stdout


def test_audit_json_lists_findings(seedtrace_cli, tmp_path):
    script = tmp_path / "haz.py"
    script.write_text(HAZARDOUS, encoding="utf-8")
    proc = seedtrace_cli("audit", str(script), "--json")
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    ids = {f["id"] for file in payload["files"] for f in file["findings"]}
    assert "set-iteration-order" in ids


def test_audit_fail_on_high(seedtrace_cli, tmp_path):
    script = tmp_path / "haz.py"
    script.write_text(HAZARDOUS, encoding="utf-8")
    proc = seedtrace_cli("audit", str(script), "--fail-on", "high")
    assert proc.returncode == 1
    assert "HIGH" in proc.stdout


def test_list_and_patch(seedtrace_cli, tmp_path):
    script = tmp_path / "det.py"
    script.write_text(DETERMINISTIC, encoding="utf-8")
    runs = tmp_path / "runs"
    seedtrace_cli("run", "--seed", "1", "--runs-dir", str(runs), "--", str(script))
    proc = seedtrace_cli("list", "--runs-dir", str(runs))
    assert proc.returncode == 0
    assert "marks=" in proc.stdout

    proc = seedtrace_cli("determinism-patch")
    assert proc.returncode == 0
    assert "PYTHONHASHSEED" in proc.stdout
    assert "cudnn.benchmark" in proc.stdout
