"""Command-line interface for a complete circuit review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api import CircuitReviewer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cirkreet",
        description="Import a KiCad/SPICE circuit, run supported analyses, and save a JSON report.",
    )
    parser.add_argument("source", help="Input .kicad_sch, .kicad_net, or SPICE-compatible netlist")
    parser.add_argument("--output-dir", "-o", default="reports", help="Directory for JSON and plot outputs")
    parser.add_argument(
        "--analyses",
        default="all",
        help="Comma-separated analysis names, or 'all' (default)",
    )
    parser.add_argument("--kicad-cli", help="Optional path to the kicad-cli executable")
    parser.add_argument("--no-plot", action="store_true", help="Do not save the review plot")
    parser.add_argument("--no-json", action="store_true", help="Do not save the JSON report")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    analyses = "all" if args.analyses.strip().lower() == "all" else [
        item.strip() for item in args.analyses.split(",") if item.strip()
    ]
    reviewer = CircuitReviewer(
        output_dir=Path(args.output_dir),
        plot=not args.no_plot,
        kicad_cli=args.kicad_cli,
    )
    result = reviewer.process_file(
        args.source,
        analyses=analyses,
        save_json=not args.no_json,
    )
    print(json.dumps({"result": result.summary(), "artifacts": result.artifacts}, indent=2))
    return 1 if result.status in {"failed", "partial"} else 0
