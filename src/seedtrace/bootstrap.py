"""Child-process entry point used by ``seedtrace run`` / ``seedtrace variance``.

Started as ``python -m seedtrace.bootstrap``, it activates recording and
then executes the user script with ``__main__`` semantics, so the user's
code needs no seedtrace import to still be traced (any seedtrace.mark()
calls inside it attach to the active session automatically).
"""

from __future__ import annotations

import os
import runpy
import sys


def main() -> None:
    script = os.environ.get("SEEDTRACE_SCRIPT")
    if not script:
        sys.stderr.write("seedtrace bootstrap: SEEDTRACE_SCRIPT not set\n")
        raise SystemExit(2)

    user_argv = os.environ.get("SEEDTRACE_ARGV")
    argv = user_argv.split(os.pathsep) if user_argv else []

    import seedtrace

    seed_env = os.environ.get("SEEDTRACE_SEED")
    seed = int(seed_env) if seed_env not in (None, "") else None
    seedtrace.auto(seed=seed)

    sys.argv = [script, *argv]
    # Make the script's own directory importable, exactly like python does.
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit:
        raise
    except BaseException:
        traceback = __import__("traceback").format_exc()
        sys.stderr.write(traceback)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
