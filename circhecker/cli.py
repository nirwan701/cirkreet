"""Command-line and guided interactive interface for CirKreet."""
from __future__ import annotations

import argparse
from pathlib import Path

from .api import CircuitReviewer

ANALYSIS_CHOICES = (
    ("1", "dc_operating_point", "DC operating point — find steady node voltages and branch currents"),
    ("2", "dc_sweep", "DC sweep — vary one independent source"),
    ("3", "ac_sweep", "AC sweep — see gain and phase across frequency"),
    ("4", "transient", "Transient — see how voltages and currents change with time"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cirkreet",
        description="Read a supported KiCad/SPICE circuit, analyse it, and create a clear report.",
    )
    parser.add_argument("source", nargs="?", help="Input .kicad_sch, .kicad_net, or SPICE-compatible netlist")
    parser.add_argument("--interactive", "-i", action="store_true", help="Open the guided circuit-analysis menu")
    parser.add_argument("--output-dir", "-o", default="reports", help="Folder for JSON and plot outputs")
    parser.add_argument(
        "--analyses",
        default="all",
        help="Comma-separated analysis names or 'all' (DC, AC, transient, frequency, transfer function, state space)",
    )
    parser.add_argument("--kicad-cli", help="Optional path to the kicad-cli executable")
    parser.add_argument("--no-plot", action="store_true", help="Do not save a review plot")
    parser.add_argument("--no-json", action="store_true", help="Do not save the JSON report")
    return parser


def _prompt_source(input_fn, output_fn) -> Path:
    while True:
        raw = input_fn("Path to your circuit file: ").strip().strip('"')
        if not raw:
            output_fn("Please enter a file path.")
            continue
        path = Path(raw).expanduser()
        if path.is_file():
            return path
        output_fn(f"I could not find '{path}'. Check the path and try again.")


def _prompt_analyses(input_fn, output_fn) -> list[str]:
    output_fn("\nChoose the analyses you want to run:")
    for number, _key, description in ANALYSIS_CHOICES:
        output_fn(f"  {number}. {description}")
    output_fn("  A. All available analyses")
    output_fn("Press Enter for the student-friendly default: DC, AC, and transient.")

    while True:
        raw = input_fn("Your choice [1,3,4]: ").strip().lower()
        if not raw:
            return ["dc_operating_point", "ac_sweep", "transient"]
        if raw in {"a", "all"}:
            return ["all"]
        tokens = [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
        lookup = {number: key for number, key, _description in ANALYSIS_CHOICES}
        if tokens and all(token in lookup for token in tokens):
            return list(dict.fromkeys(lookup[token] for token in tokens))
        output_fn("Please use numbers 1–4 separated by commas, or choose A for all.")


def run_interactive(input_fn=None, output_fn=print, *, kicad_cli=None) -> int:
    """Run a small guided workflow; injectable I/O keeps it testable."""
    input_fn = input if input_fn is None else input_fn
    output_fn("\nCirKreet — Circuit Analysis")
    output_fn("I will help you choose a circuit file and run the analyses you need.\n")
    source = _prompt_source(input_fn, output_fn)
    analyses = _prompt_analyses(input_fn, output_fn)
    output_dir = input_fn("Where should reports be saved? [reports]: ").strip() or "reports"
    make_plot = input_fn("Save a plot as well? [Y/n]: ").strip().lower() not in {"n", "no"}

    reviewer = CircuitReviewer(output_dir=output_dir, plot=make_plot, kicad_cli=kicad_cli)
    result = reviewer.process_file(source, analyses=analyses, save_json=True, plot=make_plot)
    output_fn("\n" + result.human_summary())
    output_fn("\nFiles created:")
    if result.artifacts:
        for kind, path in result.artifacts.items():
            output_fn(f"  {kind}: {path}")
    else:
        output_fn("  No output files were requested.")
    return 1 if result.status in {"failed", "partial"} else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.interactive or not args.source:
        try:
            return run_interactive(kicad_cli=args.kicad_cli)
        except (KeyboardInterrupt, EOFError):
            print("\nAnalysis cancelled. No files were changed.")
            return 130
        except Exception as exc:
            print(f"Could not complete the circuit analysis: {type(exc).__name__}: {exc}")
            return 2

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
    print(result.human_summary())
    if result.artifacts:
        print("\nFiles created:")
        for kind, path in result.artifacts.items():
            print(f"  {kind}: {path}")
    return 1 if result.status in {"failed", "partial"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
