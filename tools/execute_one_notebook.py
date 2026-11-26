"""Execute one notebook in an isolated process and persist its outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", help="Notebook path relative to repository root")
    args = parser.parse_args()
    path = (ROOT / args.notebook).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file() or path.suffix != ".ipynb":
        parser.error("notebook must be an existing .ipynb file inside the repository")

    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
        record_timing=True,
    )
    client.execute()
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print(f"Executed and saved {path.relative_to(ROOT)} ({len(notebook.cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
