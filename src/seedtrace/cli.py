"""seedtrace command-line interface.

Commands
--------
run                execute a script under recording (pin seed / hash seed)
audit              static + environment determinism scan of your code
compare            locate the first divergence between two recorded runs
list               show recent recorded runs
variance           rerun N times with one seed and measure metric spread
determinism-patch  write a drop-in hardening snippet
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__
from .audit import (
    SEV_HIGH,
    SEV_MEDIUM,
    audit_environment,
    audit_source,
)
from .compare import compare_runs, load_run
from .patch import render_patch
from .report import render_divergence, render_findings
from .session import DEFAULT_RUNS_DIR, find_run, list_runs

FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="seedtrace",
        description="Find out why two runs with the same seed gave different results.",
    )
    p.add_argument("--version", action="version", version=f"seedtrace {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a script with recording enabled")
    run.add_argument("--seed", type=int, default=None, help="seed random/np/torch with this value")
    run.add_argument(
        "--hashseed",
        type=int,
        default=None,
        help="set PYTHONHASHSEED for the child process (0 is the usual choice)",
    )
    run.add_argument("--runs-dir", default=os.environ.get("SEEDTRACE_RUNS_DIR", DEFAULT_RUNS_DIR))
    run.add_argument("--label", default=None, help="free-form tag stored in the run record")
    run.add_argument("script", help="python script to execute")
    run.add_argument("script_args", nargs=argparse.REMAINDER, help="passed through to the script")

    audit = sub.add_parser("audit", help="scan code + environment for determinism hazards")
    audit.add_argument("path", nargs="?", default=".", help="file or directory (default: cwd)")
    audit.add_argument("--json", action="store_true", dest="as_json")
    audit.add_argument(
        "--env-only", action="store_true", help="skip source scanning, check the live process only"
    )
    audit.add_argument(
        "--fail-on",
        choices=["high", "medium", "never"],
        default="never",
        help="exit 1 when findings of this severity or worse exist (for CI)",
    )

    cmp_ = sub.add_parser("compare", help="find the first divergence between two runs")
    cmp_.add_argument("run_a", help="run id or path to a run directory / run.json")
    cmp_.add_argument("run_b")
    cmp_.add_argument("--runs-dir", default=os.environ.get("SEEDTRACE_RUNS_DIR", DEFAULT_RUNS_DIR))
    cmp_.add_argument("--json", action="store_true", dest="as_json")

    lst = sub.add_parser("list", help="list recent recorded runs")
    lst.add_argument("--runs-dir", default=os.environ.get("SEEDTRACE_RUNS_DIR", DEFAULT_RUNS_DIR))
    lst.add_argument("-n", "--max", type=int, default=15)

    var = sub.add_parser(
        "variance",
        help="run the same script N times with one seed; report metric spread",
    )
    var.add_argument("-n", "--repeats", type=int, default=3)
    var.add_argument("--seed", type=int, default=42)
    var.add_argument(
        "--hashseed",
        type=int,
        default=None,
        help="pin PYTHONHASHSEED for every repeat (omit to vary, as python does by default)",
    )
    var.add_argument(
        "--metric",
        default=None,
        help="regex whose LAST match per run is the metric (default: last float on stdout)",
    )
    var.add_argument("--runs-dir", default=os.environ.get("SEEDTRACE_RUNS_DIR", DEFAULT_RUNS_DIR))
    var.add_argument(
        "--tol",
        type=float,
        default=0.0,
        help="max allowed stdev before exiting nonzero (CI gate)",
    )
    var.add_argument("script")
    var.add_argument("script_args", nargs=argparse.REMAINDER)

    patch = sub.add_parser("determinism-patch", help="print a drop-in hardening snippet")
    patch.add_argument("-o", "--out", default=None, help="write to file instead of stdout")

    return p


# -- helpers ------------------------------------------------------------


def _child_env(
    seed: Optional[int],
    hashseed: Optional[int],
    runs_dir: str,
    label: Optional[str],
) -> dict:
    env = os.environ.copy()
    env.pop("SEEDTRACE_SEED", None)
    env.pop("SEEDTRACE_LABEL", None)
    if seed is not None:
        env["SEEDTRACE_SEED"] = str(seed)
    if hashseed is not None:
        env["PYTHONHASHSEED"] = str(hashseed)
    env["SEEDTRACE_RUN_DIR"] = str(Path(runs_dir).resolve())
    if label:
        env["SEEDTRACE_LABEL"] = label
    return env


def _spawn(script: str, script_args: List[str], env: dict) -> "subprocess.CompletedProcess[str]":
    if script_args and script_args[0] == "--":
        script_args = script_args[1:]
    env = dict(env)
    env["SEEDTRACE_SCRIPT"] = str(Path(script).resolve())
    env["SEEDTRACE_ARGV"] = os.pathsep.join(script_args)
    return subprocess.run(
        [sys.executable, "-m", "seedtrace.bootstrap"],
        env=env,
        capture_output=False,
    )


# -- commands -----------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    if not Path(args.script).exists():
        sys.stderr.write(f"seedtrace: script not found: {args.script}\n")
        return 2
    env = _child_env(args.seed, args.hashseed, args.runs_dir, args.label)
    proc = _spawn(args.script, args.script_args or [], env)
    newest = list_runs(args.runs_dir)
    if newest:
        sys.stderr.write(f"\nseedtrace: recorded {newest[-1].parent}\n")
    return proc.returncode


def cmd_audit(args: argparse.Namespace) -> int:
    from .audit import Finding

    findings: List[Finding] = list(audit_environment())
    file_reports = []
    if not args.env_only:
        file_reports = audit_source(args.path)
        for fr in file_reports:
            findings.extend(fr.findings)

    if args.as_json:
        payload = {
            "environment": [f.to_dict() for f in findings if f.file is None],
            "files": [
                {
                    "path": fr.path,
                    "uses": {
                        "random": fr.uses_random,
                        "numpy_random": fr.uses_numpy_random,
                        "torch": fr.uses_torch,
                    },
                    "findings": [f.to_dict() for f in fr.findings],
                }
                for fr in file_reports
            ],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(render_findings(findings, title=f"seedtrace audit: {args.path}"))

    if args.fail_on != "never":
        wanted = {SEV_HIGH} if args.fail_on == "high" else {SEV_HIGH, SEV_MEDIUM}
        if any(f.severity in wanted for f in findings):
            return 1
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    pa = find_run(args.runs_dir, args.run_a)
    pb = find_run(args.runs_dir, args.run_b)
    if pa is None or pb is None:
        missing = args.run_a if pa is None else args.run_b
        sys.stderr.write(f"seedtrace: cannot find run record for {missing!r}\n")
        sys.stderr.write(f"          (looked in {args.runs_dir}; try `seedtrace list`)\n")
        return 2
    run_a, run_b = load_run(pa), load_run(pb)
    div = compare_runs(run_a, run_b)
    if args.as_json:
        print(json.dumps(div.to_dict(), indent=2))
    else:
        print(render_divergence(div, pa.parent.name, pb.parent.name))
    return 0 if div.identical else 1


def cmd_list(args: argparse.Namespace) -> int:
    runs = list_runs(args.runs_dir)[-args.max :]
    if not runs:
        print(f"no recorded runs under {args.runs_dir}")
        return 0
    for path in reversed(runs):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        label = data.get("label") or ""
        seed = data.get("seed")
        n = len(data.get("marks", []))
        print(f"{data.get('run_id', path.parent.name)}  marks={n:<4} seed={seed!s:<6} {label}")
    return 0


def cmd_variance(args: argparse.Namespace) -> int:
    if not Path(args.script).exists():
        sys.stderr.write(f"seedtrace: script not found: {args.script}\n")
        return 2
    script_args = args.script_args or []
    if script_args and script_args[0] == "--":
        script_args = script_args[1:]
    env = _child_env(args.seed, args.hashseed, args.runs_dir, None)
    hashseed_display = env.get("PYTHONHASHSEED", "random")
    metric_re = re.compile(args.metric) if args.metric else None
    values: List[float] = []
    failed = False

    for i in range(args.repeats):
        inner_env = dict(env)
        inner_env["SEEDTRACE_LABEL"] = f"variance-{i}"
        inner_env["SEEDTRACE_SCRIPT"] = str(Path(args.script).resolve())
        inner_env["SEEDTRACE_ARGV"] = os.pathsep.join(script_args)
        proc = subprocess.run(
            [sys.executable, "-m", "seedtrace.bootstrap"],
            env=inner_env,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout or "")
            sys.stderr.write(proc.stderr or "")
            sys.stderr.write(f"repeat {i} failed with exit code {proc.returncode}\n")
            failed = True
            continue
        value = _extract_metric(proc.stdout, metric_re)
        if value is None:
            sys.stderr.write(
                f"repeat {i}: no metric found in stdout; enable a print or fix --metric\n"
            )
            failed = True
            continue
        values.append(value)

    if failed or len(values) < 2:
        return 1
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    stdev = var**0.5

    print("-" * 72)
    print(
        f"seedtrace variance: {args.repeats} repeats, seed={args.seed}, "
        f"PYTHONHASHSEED={hashseed_display}"
    )
    for i, v in enumerate(values):
        print(f"  run {i}: {v!r}")
    print(f"  mean={mean!r}  stdev={stdev!r}  min={min(values)!r}  max={max(values)!r}")
    if stdev == 0.0:
        print("  RESULT: bit-identical metric across repeats -> deterministic (as far as probed)")
        return 0
    print("  RESULT: metric VARIES under a fixed seed -> nondeterminism confirmed")
    if stdev > args.tol:
        print(f"  stdev {stdev} exceeds tolerance {args.tol} -> exit 1")
        return 1
    return 0


def _extract_metric(stdout: str, metric_re: Optional[re.Pattern]) -> Optional[float]:
    if metric_re is not None:
        matches = list(metric_re.finditer(stdout))
        if not matches:
            return None
        m = matches[-1]
        text = m.group(1) if m.groups() else m.group(0)
    else:
        floats = FLOAT_RE.findall(stdout)
        if not floats:
            return None
        text = floats[-1]
    try:
        return float(text)
    except ValueError:
        return None


def cmd_patch(args: argparse.Namespace) -> int:
    text = render_patch()
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(text)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "run": cmd_run,
        "audit": cmd_audit,
        "compare": cmd_compare,
        "list": cmd_list,
        "variance": cmd_variance,
        "determinism-patch": cmd_patch,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
