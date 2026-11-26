"""A small Python-first RC filter lab for students.

Run from the repository root::

    python examples/student_lab.py
    python examples/student_lab.py --no-plot

The circuit is defined in the script so learners can edit R, C, and the
source waveform without first learning a schematic editor.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from circhecker import ACSweepSpec, AnalysisSpec, CircuitReviewer, TransientSpec

RC_LOW_PASS_NETLIST = """\
* Student lab: first-order RC low-pass filter
.title Student RC low-pass filter
V1 input 0 PULSE(0 5 0 1u 1u 5m 10m) AC 1
R1 input output 1k
C1 output 0 100n
.end
"""


def run_student_lab(output_dir: str | Path = "reports/student_lab", *, plot: bool = True):
    """Run DC, AC, and transient analyses and return a ``CircuitResult``."""
    request = AnalysisSpec(
        analyses=("dc_operating_point", "ac_sweep", "transient"),
        ac_sweep=ACSweepSpec(start_Hz=10, stop_Hz=100_000, points=120, scale="log"),
        transient=TransientSpec(
            stop_s=2e-3,
            step_s=2e-6,
            method="trapezoidal",
        ),
        output_node="output",
    )
    reviewer = CircuitReviewer(output_dir=output_dir, plot=plot)
    return reviewer.process_netlist_text(
        RC_LOW_PASS_NETLIST,
        analysis_spec=request,
        save_json=True,
        plot=plot,
        source_name="student_rc_lowpass.cir",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a guided RC low-pass circuit lab.")
    parser.add_argument("--output-dir", default="reports/student_lab", help="Folder for the JSON report and plot")
    parser.add_argument("--no-plot", action="store_true", help="Create the JSON report without a PNG plot")
    args = parser.parse_args(argv)

    result = run_student_lab(args.output_dir, plot=not args.no_plot)
    print(result.human_summary())
    print("\nChange R1, C1, or the source in examples/student_lab.py and run it again.")
    if result.artifacts:
        print("\nFiles created:")
        for kind, path in result.artifacts.items():
            print(f"  {kind}: {path}")
    return 0 if result.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
