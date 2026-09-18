"""Human-readable terminal reports.

Plain ASCII on purpose: reports get pasted into bug reports, forums and
CI logs where unicode and ANSI colors misbehave.
"""

from __future__ import annotations

from typing import List, Optional

from .audit import Finding
from .compare import Divergence, classify_stream_note

RULE = "-" * 72


def render_divergence(div: Divergence, name_a: str, name_b: str) -> str:
    lines: List[str] = []
    lines.append(RULE)
    lines.append("seedtrace compare")
    lines.append(f"  run A: {name_a}")
    lines.append(f"  run B: {name_b}")
    lines.append(RULE)

    if div.identical and not div.env_differences:
        lines.append("RESULT: runs are identical across all recorded checkpoints.")
        lines.append(f"        {div.marks_compared} checkpoint(s) compared, no divergence found.")
        return "\n".join(lines)

    if div.first_divergence_index is None:
        lines.append("RESULT: checkpoints match, but the process environments differ:")
        for key, pair in div.env_differences.items():
            lines.append(f"  - {key}: A={pair['a']!r}  B={pair['b']!r}")
        return "\n".join(lines)

    idx = div.first_divergence_index
    label = div.mark_name if div.mark_name is not None else "(name mismatch)"
    lines.append(f"FIRST DIVERGENCE: checkpoint #{idx}  mark={label!r}")
    lines.append(f"  (compared {div.marks_compared} of A={div.marks_a}/B={div.marks_b} marks)")

    if div.diverged_streams:
        lines.append("  diverged RNG streams:")
        for s in div.diverged_streams:
            lines.append(f"    * {s}  <- {classify_stream_note(s)}")
    else:
        lines.append("  RNG streams: all EQUAL -> values diverged without RNG-state drift")
    if div.diverged_digest_positions:
        lines.append(f"  diverged recorded values at positions: {div.diverged_digest_positions}")

    for key, pair in div.env_differences.items():
        lines.append(f"  environment differs -> {key}: A={pair['a']!r} B={pair['b']!r}")

    lines.append("")
    lines.append("WHAT THIS MEANS")
    for note in div.notes:
        wrapped = _wrap(note)
        if wrapped:
            lines.append(f"  - {wrapped[0]}")
            for cont in wrapped[1:]:
                lines.append(f"    {cont}")
        else:
            lines.append("  -")
    lines.append("")
    lines.append("NEXT STEP: run `seedtrace audit <your script>` to list the exact")
    lines.append("code lines that can cause this, and `seedtrace determinism-patch`")
    lines.append("for a drop-in hardening snippet.")
    return "\n".join(lines)


def render_findings(findings: List[Finding], title: Optional[str] = None) -> str:
    lines: List[str] = []
    order = {"high": 0, "medium": 1, "info": 2}
    findings = sorted(findings, key=lambda f: (order.get(f.severity, 3), f.file or "", f.line or 0))
    if title:
        lines.append(RULE)
        lines.append(title)
        lines.append(RULE)
    if not findings:
        lines.append("No determinism hazards found.")
        return "\n".join(lines)
    counts = {"high": 0, "medium": 0, "info": 0}
    for f in findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1
        loc = f.file or "(environment)"
        if f.line:
            loc = f"{loc}:{f.line}"
        lines.append(f"[{f.severity.upper():6}] {f.id}")
        lines.append(f"    {loc}")
        lines.append(f"    {f.message}")
        lines.append(f"    fix: {f.fix}")
        lines.append("")
    lines.append(
        f"summary: {counts.get('high', 0)} high, {counts.get('medium', 0)} medium, "
        f"{counts.get('info', 0)} info"
    )
    return "\n".join(lines)


def _wrap(text: str, width: int = 64) -> List[str]:
    words = text.split()
    out, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        out.append(cur)
    return out
