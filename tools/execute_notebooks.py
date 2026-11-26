"""Execute every notebook in a fresh Python process and persist its outputs.

Each notebook gets an isolated Jupyter kernel process. This avoids stale kernel
sockets leaking from a previous notebook and makes failures attributable to a
single file in both local runs and CI.
"""
from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
WORKER = Path(__file__).with_name("execute_one_notebook.py")


def main() -> int:
    files = sorted(NOTEBOOKS.glob("*.ipynb"))
    if not files:
        print("No notebooks were found.", file=sys.stderr)
        return 1
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "module://matplotlib_inline.backend_inline")
    failures: list[tuple[str, int]] = []
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        print(f"\nExecuting {relative} in an isolated kernel ...", flush=True)
        try:
            completed = subprocess.run(
                [sys.executable, str(WORKER), relative],
                cwd=ROOT,
                env=env,
                check=False,
                timeout=600,
            )
            return_code = completed.returncode
        except subprocess.TimeoutExpired:
            return_code = 124
        if return_code:
            failures.append((relative, return_code))
            print(f"FAILED: {relative} (exit {return_code})", file=sys.stderr)
        else:
            print(f"PASSED: {relative}", flush=True)
    if failures:
        print("\nNotebook failures:", file=sys.stderr)
        for name, code in failures:
            print(f"- {name}: exit {code}", file=sys.stderr)
        return 1
    print(f"\nAll {len(files)} notebooks executed and saved successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
