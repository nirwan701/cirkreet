"""Run the reproducible CirKreety circuit catalog.

Usage from the repository root::

    python examples/run_examples.py --list
    python examples/run_examples.py --example rc_lowpass
    python examples/run_examples.py --example all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.catalog import list_examples, get_example
from examples.notebook_support import run_example


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CirKreety's reproducible analog, semiconductor, digital and IC examples.")
    parser.add_argument("--example", default="all", help="Example ID or 'all' (default).")
    parser.add_argument("--list", action="store_true", help="List available example IDs and descriptions.")
    parser.add_argument("--output-dir", default=str(ROOT / "reports" / "example_catalog"), help="Report output directory.")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plot generation.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        for item in list_examples():
            print(f"{item['id']:24s} [{item['category']}] {item['focus']}")
        return 0

    selected = list_examples() if args.example.strip().lower() == "all" else [get_example(args.example)]
    results = []
    failed = False
    for spec in selected:
        print(f"\n=== {spec['title']} [{spec['id']}] ===")
        result = run_example(
            spec["id"],
            output_root=args.output_dir,
            make_plot=not args.no_plots,
        )
        summary = result.summary()
        analysis_status = summary["analysis_status"]
        missing = [name for name in spec["required_analyses"] if analysis_status.get(name) != "success"]
        passed = summary["status"] == "success" and not missing
        failed = failed or not passed
        print(json.dumps({
            "status": "passed" if passed else "failed",
            "backend": summary["backend"],
            "validation": summary["validation_status"],
            "analyses": analysis_status,
            "required_analysis_failures": missing,
            "warnings": summary["warnings"],
            "errors": summary["errors"],
            "artifacts": result.artifacts,
        }, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
