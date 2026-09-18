"""Static + environmental determinism audit.

Two independent scanners:

- :func:`audit_source` walks Python files with ``ast`` and flags code
  patterns that silently break run-to-run reproducibility (iteration
  over unordered sets, ``os.listdir`` without sorting, non-deterministic
  PyTorch ops, DataLoader workers without seeds, TF32/cudnn.benchmark
  switched on, ...).
- :func:`audit_environment` inspects the current process: environment
  variables and, if torch is importable, the live determinism flags.

Findings carry stable ids so they can be referenced from bug reports
and asserted on in tests.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Set

SEV_HIGH = "high"
SEV_MEDIUM = "medium"
SEV_INFO = "info"

# Ops with known non-deterministic GPU (atomic accumulation) implementations.
NONDETERMINISTIC_TORCH_OPS = {
    "scatter_add_",
    "scatter_reduce_",
    "index_put_",
    "index_add_",
    "gather",  # torch.gather backward is nondeterministic
    "grid_sampler_2d",
    "grid_sampler_3d",
    "ctc_loss",
    "softmax",  # cudnn softmax backward
}

# Files that are never worth scanning.
_SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".seedtrace", "node_modules", ".tox"}


@dataclass
class Finding:
    id: str
    severity: str
    message: str
    fix: str
    file: Optional[str] = None
    line: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "severity": self.severity,
            "message": self.message,
            "fix": self.fix,
            "file": self.file,
            "line": self.line,
        }


@dataclass
class FileReport:
    path: str
    uses_random: bool = False
    uses_numpy_random: bool = False
    uses_torch: bool = False
    seeds_random: bool = False
    seeds_numpy: bool = False
    seeds_torch: bool = False
    uses_torch_deterministic_api: bool = False
    findings: List[Finding] = field(default_factory=list)


def _name_of(node: ast.AST) -> str:
    """Render an attribute chain like ``torch.backends.cudnn.benchmark``."""
    parts: List[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return ".".join(reversed(parts))


class _HazardVisitor(ast.NodeVisitor):
    def __init__(self, filename: str, report: FileReport) -> None:
        self.filename = filename
        self.report = report
        self._lines: Set[tuple] = set()
        self._sorted_calls: Set[int] = set()
        self._set_names: Set[str] = set()  # variables bound to set literals/calls

    def _add(self, finding: Finding, node: ast.AST) -> None:
        finding.file = self.filename
        finding.line = getattr(node, "lineno", None)
        key = (finding.id, finding.line)
        if key in self._lines:
            return
        self._lines.add(key)
        self.report.findings.append(finding)

    # -- tracking ------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name.split(".")[0] == "torch":
                self.report.uses_torch = True
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and node.module.split(".")[0] == "torch":
            self.report.uses_torch = True
        self.generic_visit(node)

    # -- hazards -------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        fname = _name_of(node.func)
        short = fname.split(".")[-1]

        # Remember calls that are explicitly sorted - they neutralize
        # iteration-order hazards like os.listdir().
        if fname == "sorted" and node.args and isinstance(node.args[0], ast.Call):
            self._sorted_calls.add(id(node.args[0]))

        if fname in ("random.seed", "np.random.seed", "numpy.random.seed"):
            if fname == "random.seed":
                self.report.seeds_random = True
            else:
                self.report.seeds_numpy = True
        elif fname in ("torch.manual_seed", "torch.cuda.manual_seed", "torch.cuda.manual_seed_all"):
            self.report.uses_torch = True
            self.report.seeds_torch = True
        elif fname == "torch.use_deterministic_algorithms":
            self.report.uses_torch_deterministic_api = True
        elif short == "seed":
            self.report.seeds_random = True  # Generator.seed() etc.

        if fname.startswith("random."):
            self.report.uses_random = True
        if fname.startswith(("np.random.", "numpy.random.")):
            self.report.uses_numpy_random = True

        # list(set(...)) / list({...literal...}) / list(<set_var>)
        # -> hash-order dependent
        arg0 = node.args[0] if node.args else None
        is_set_expr = bool(arg0) and (
            isinstance(arg0, ast.Set)
            or (isinstance(arg0, ast.Call) and _name_of(arg0.func) == "set")
            or (isinstance(arg0, ast.Name) and arg0.id in self._set_names)
        )
        if fname == "list" and is_set_expr:
            self._add(
                Finding(
                    id="set-iteration-order",
                    severity=SEV_HIGH,
                    message=(
                        "list(...) over a set iterates in PYTHONHASHSEED-dependent "
                        "order; sampling or indexing from it changes every process"
                    ),
                    fix="Use sorted(...) or a list/tuple literal, or run with PYTHONHASHSEED=0",
                ),
                node,
            )

        if fname == "os.listdir" and id(node) not in self._sorted_calls:
            self._add(
                Finding(
                    id="unsorted-listdir",
                    severity=SEV_MEDIUM,
                    message="os.listdir() order is OS/filesystem dependent",
                    fix="Wrap in sorted(...) before consuming the file list",
                ),
                node,
            )

        if fname in NONDETERMINISTIC_TORCH_OPS and fname.split(".")[0] in ("torch", "t"):
            self._add(
                Finding(
                    id=f"nondeterministic-op:{fname}",
                    severity=SEV_HIGH,
                    message=f"torch.{fname} has a known non-deterministic GPU implementation",
                    fix="Enable torch.use_deterministic_algorithms(True) or avoid the op on CUDA",
                ),
                node,
            )
        elif short in NONDETERMINISTIC_TORCH_OPS and short.endswith("_"):
            # tensor method form: x.scatter_add_(...)
            self._add(
                Finding(
                    id=f"nondeterministic-op:{short}",
                    severity=SEV_HIGH,
                    message=f".{short} has a known non-deterministic GPU implementation",
                    fix="Enable torch.use_deterministic_algorithms(True) or avoid the op on CUDA",
                ),
                node,
            )

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        # Remember variables bound to a set: FEATURES = {...} makes
        # a later list(FEATURES) hash-order dependent too.
        value_is_set = isinstance(node.value, ast.Set) or (
            isinstance(node.value, ast.Call) and _name_of(node.value.func) == "set"
        )
        if value_is_set:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self._set_names.add(target.id)
        for target in node.targets:
            tname = _name_of(target)
            value = node.value
            truthy = isinstance(value, ast.Constant) and value.value is True
            if tname == "torch.backends.cudnn.benchmark" and truthy:
                self._add(
                    Finding(
                        id="cudnn-benchmark-on",
                        severity=SEV_HIGH,
                        message="cudnn.benchmark=True picks algorithms by runtime timing; results vary run to run",
                        fix="Set torch.backends.cudnn.benchmark = False for reproducible runs",
                    ),
                    node,
                )
            elif (
                tname
                in (
                    "torch.backends.cuda.matmul.allow_tf32",
                    "torch.backends.cudnn.allow_tf32",
                )
                and truthy
            ):
                self._add(
                    Finding(
                        id="tf32-enabled",
                        severity=SEV_MEDIUM,
                        message=f"{tname}=True reduces float precision and can change results",
                        fix="Set the flag to False when bit-exact reproduction matters",
                    ),
                    node,
                )
        self.generic_visit(node)


def _check_dataloader_calls(tree: ast.Module, report: FileReport, filename: str) -> None:
    """DataLoader(num_workers>0) without worker_init_fn/generator is a classic
    source of per-epoch randomness that global seeding does not control."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fname = _name_of(node.func)
        if not fname.endswith("DataLoader"):
            continue
        kwargs = {kw.arg for kw in node.keywords if kw.arg}
        nw = next((kw.value for kw in node.keywords if kw.arg == "num_workers"), None)
        workers_possible = nw is not None and (not isinstance(nw, ast.Constant) or bool(nw.value))
        if workers_possible and "worker_init_fn" not in kwargs and "generator" not in kwargs:
            report.findings.append(
                Finding(
                    id="dataloader-unseeded-workers",
                    severity=SEV_HIGH,
                    message=(
                        "DataLoader with worker processes inherits no explicit RNG seed; "
                        "worker order and augmentation draws vary between runs"
                    ),
                    fix="Pass worker_init_fn (seed each worker) and/or an explicit torch.Generator",
                    file=filename,
                    line=node.lineno,
                )
            )


def audit_file(path: str) -> FileReport:
    report = FileReport(path=path)
    try:
        source = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return report
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError:
        return report

    visitor = _HazardVisitor(path, report)
    visitor.visit(tree)
    _check_dataloader_calls(tree, report, path)

    if (report.uses_random or report.uses_numpy_random) and not (
        report.seeds_random or report.seeds_numpy
    ):
        report.findings.append(
            Finding(
                id="unseeded-random-usage",
                severity=SEV_MEDIUM,
                message="module consumes randomness but never seeds the RNG",
                fix="Call random.seed(...) / np.random.seed(...) at the start of main",
                file=path,
            )
        )
    if report.uses_torch and not report.seeds_torch:
        report.findings.append(
            Finding(
                id="unseeded-torch",
                severity=SEV_MEDIUM,
                message="module uses torch but never calls torch.manual_seed",
                fix="Call torch.manual_seed(...) (and torch.cuda.manual_seed_all) at startup",
                file=path,
            )
        )
    if report.uses_torch and not report.uses_torch_deterministic_api:
        report.findings.append(
            Finding(
                id="no-deterministic-algorithms",
                severity=SEV_INFO,
                message="torch.use_deterministic_algorithms(True) is never called",
                fix="Add it (with CUBLAS_WORKSPACE_CONFIG set) to make torch error out on nondeterministic ops",
                file=path,
            )
        )
    return report


def audit_source(root: str) -> List[FileReport]:
    """Audit a file or directory tree."""
    p = Path(root)
    files: Iterable[Path]
    if p.is_file():
        files = [p]
    else:
        files = (d for d in p.rglob("*.py") if not (_SKIP_DIRS & set(d.parts)) and d.is_file())
    return [audit_file(str(f)) for f in files]


def audit_environment() -> List[Finding]:
    """Live-process checks that static analysis cannot make."""
    findings: List[Finding] = []
    if os.environ.get("PYTHONHASHSEED") is None:
        findings.append(
            Finding(
                id="hash-seed-unpinned",
                severity=SEV_HIGH,
                message="PYTHONHASHSEED is not set; str/bytes hash order is randomized per process",
                fix="Export PYTHONHASHSEED=0 (or use `seedtrace run --hashseed 0 -- ...`)",
            )
        )
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") is None:
        findings.append(
            Finding(
                id="cublas-workspace-unset",
                severity=SEV_INFO,
                message="CUBLAS_WORKSPACE_CONFIG is unset; required by use_deterministic_algorithms on CUDA",
                fix="Export CUBLAS_WORKSPACE_CONFIG=:4096:8 (or :16:8)",
            )
        )
    try:
        import sys

        if "torch" in sys.modules:
            import torch

            if torch.backends.cudnn.benchmark:
                findings.append(
                    Finding(
                        id="cudnn-benchmark-on",
                        severity=SEV_HIGH,
                        message="torch.backends.cudnn.benchmark is True in this process",
                        fix="Set it to False for reproducible runs",
                    )
                )
            if not torch.are_deterministic_algorithms_enabled():
                findings.append(
                    Finding(
                        id="deterministic-algorithms-off",
                        severity=SEV_MEDIUM,
                        message="torch determinism mode is off; nondeterministic kernels are silently allowed",
                        fix="torch.use_deterministic_algorithms(True) after setting CUBLAS_WORKSPACE_CONFIG",
                    )
                )
    except Exception:
        pass
    return findings
