# CirKreet

**A Python-first toolkit for learning, analysing, and reviewing electronic circuits.**

CirKreet reads supported SPICE-compatible netlists and KiCad netlists, converts
them into one canonical `CircuitIR`, and runs the analyses supported by its
native numerical engines. Use it from Python, the command line, or the guided
interactive menu. Jupyter notebooks provide lessons and visual demonstrations;
the reusable implementation lives in the `circhecker/` Python package.

> CirKreet is an educational engineering tool, not a complete SPICE replacement
> or a hardware sign-off system. Read the [model limitations](docs/LIMITATIONS.md)
> before using results to make design decisions.

## What you can do

- Import supported KiCad netlists and SPICE-compatible circuit descriptions.
- Work with a shared circuit model containing components, pins, nodes,
  branches, models, parameters, and source information.
- Run DC operating-point and source-sweep analyses, AC sweeps, and transient
  simulations where the topology and device models are supported.
- Inspect numerical data, units, solver diagnostics, warnings, and reports.
- Use the same public Python API in scripts, applications, tests, and notebooks.

## Install

Python 3.10 or newer is required.

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

For the guided notebooks, install the optional notebook tools:

```bash
python -m pip install -e ".[notebooks]"
```

## Your first circuit analysis

### Python API

```python
from circhecker import (
    ACSweepSpec,
    AnalysisSpec,
    CircuitReviewer,
    TransientSpec,
)

request = AnalysisSpec(
    analyses=("dc_operating_point", "ac_sweep", "transient"),
    ac_sweep=ACSweepSpec(start_Hz=10, stop_Hz=100_000, points=120),
    transient=TransientSpec(stop_s=2e-3, step_s=2e-6),
    output_node="out",
)

result = CircuitReviewer(plot=False).process_file(
    "examples/circuits/analog/rc_lowpass.cir",
    analysis_spec=request,
    save_json=False,
)

print(result.human_summary())
print(result.simulation_result.to_dict())
```

### Guided interactive mode

Start CirKreet without arguments:

```bash
cirkreet
```

Or explicitly open the menu:

```bash
cirkreet --interactive
```

It asks for a circuit file, lets you choose the analyses, and saves a readable
summary, JSON report, and plot. You can also run it directly from a clone:

```bash
python -m circhecker --help
```

### Command-line mode

```bash
cirkreet examples/circuits/analog/rc_lowpass.cir \
  --analyses dc_operating_point,ac_sweep,transient \
  --output-dir reports/rc_lowpass
```

Use `--no-plot` to skip the PNG plot or `--no-json` to skip the JSON report.

## The Python data flow

CirKreet keeps the circuit, the analysis request, the execution plan, and the
results separate:

```text
KiCad / SPICE input
       |
       v
   CircuitIR  <---- canonical electrical circuit
       |
       +---- AnalysisSpec <---- your requested calculations
                    |
                    v
                AnalysisIR <---- normalized execution plan
                    |
             Native numerical solvers
                    |
                    v
             SimulationResult <---- results with units and diagnostics
```

- **`CircuitIR`** describes the circuit independently of its source format.
- **`AnalysisSpec`** describes the analyses and settings you request.
- **`AnalysisIR`** records the normalized plan used for that run.
- **`SimulationResult`** provides consistent per-analysis status, messages,
  data, units, and diagnostics.

See [the analysis workflow guide](docs/ANALYSIS_WORKFLOW.md) for details.

## Learn by doing

Start with the [student guide](docs/STUDENT_GUIDE.md), which walks through an
RC low-pass filter and connects its simulated response to the equation
\(f_c = 1/(2\pi RC)\).

Run the Python lab:

```bash
python examples/student_lab.py
```

Explore the circuit catalog:

```bash
python examples/run_examples.py --list
python examples/run_examples.py --example rc_lowpass
python examples/run_examples.py --example series_rlc --no-plots
```

The notebooks in `notebooks/` cover a connected signal chain, analog circuits,
semiconductor amplifiers, transistor-level digital gates, and IC-style
subcircuits. They use the package API; they are learning material rather than
the home of the solver implementation.

## Project structure

```text
circhecker/      Python package: importers, CircuitIR, analysis, solvers, results
examples/        Runnable Python labs and SPICE-compatible circuit files
notebooks/       Guided lessons and plots
tests/           Python unit, numerical-reference, and integration tests
docs/            Architecture, analysis workflow, student guide, limitations
```

## Test the project

```bash
python -m pip install -e ".[test]"
python -m pytest -q --ignore=tests/test_unified.py
python -m pytest -q tests/test_unified.py
```

GitHub Actions runs the test suite on Python 3.10–3.13. Tests include numerical
checks against known circuit behaviour, parser regressions, and end-to-end
workflows.

## Contribute

Bug reports, reproducible circuit examples, numerical reference tests, and
clear documentation are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md)
before opening a pull request.

## License

MIT. See [LICENSE](LICENSE).
