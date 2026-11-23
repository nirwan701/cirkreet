# CirKreety

**A Python toolkit for reading, checking, and analysing electronic circuits.**

CirKreety imports supported KiCad and SPICE-compatible circuit files into one shared representation, checks the extracted circuit structure, runs the analyses supported by its numerical backends, and writes reports that people and other software can inspect.

It is an experimental, educational project. It is not a full SPICE replacement and its results are not a substitute for engineering review or hardware validation.

## What it can do

- **Read circuit files:** supported SPICE-compatible netlists, KiCad XML netlists, and KiCad schematics converted with `kicad-cli`.
- **Create one circuit model:** normalize components, pins, nodes, branches, models, parameters, hierarchy, directives, and source information into `CircuitIR`.
- **Check the extracted design:** report structural issues such as missing references, unresolved models, and floating nodes.
- **Run supported analyses:** DC operating point and sweep, AC sweep, transient analysis, and selected linear frequency and transfer-function analyses.
- **Review results:** include solver diagnostics, available KCL and power checks, signal measurements, plots, and JSON reports.
- **Extend project-specific checks:** add review plugins without changing the numerical solver.

The project includes example circuits and Jupyter notebooks covering analog circuits, semiconductor devices, transistor-level CMOS gates, IC-style subcircuits, and an integrated signal chain.

## How it works

```text
KiCad / SPICE-compatible input
              |
              v
       Import and parse
              |
              v
           CircuitIR
              |
              v
     Structural validation
              |
              v
    Supported analysis engines
              |
              v
       Results and reports
```

`CircuitIR` is the shared boundary between importing a circuit and analysing it. Applications can inspect or modify this representation and pass it to the reviewer without parsing the original file again.

## Requirements

- Python 3.10 or newer
- NumPy, SciPy, Matplotlib, and jsonschema (installed automatically)
- `kicad-cli` only when importing a native `.kicad_sch` schematic

KiCad `.kicad_net` files and the supported SPICE-compatible netlist subset are read directly. LTspice `.asc` drawing files are not parsed; export a SPICE-compatible netlist first.

## Install

From the repository root:

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e .
```

To install the test tools or notebook environment as well:

```bash
python -m pip install -e ".[test]"
python -m pip install -e ".[notebooks]"
```

## Try it

### Run a circuit from the command line

```bash
cirkreet examples/circuits/analog/rc_lowpass.cir \
  --analyses dc_operating_point,ac_sweep \
  --output-dir reports/rc_lowpass
```

The command prints a summary and saves the requested report and plot files under the output directory. Use `--no-plot` or `--no-json` to disable either output. To see all options:

```bash
cirkreet --help
```

### Use it from Python

The installable command and distribution are named `cirkreet`; the Python package is imported as `circhecker`.

```python
from circhecker import CircuitReviewer

reviewer = CircuitReviewer(output_dir="reports/rc_lowpass", plot=False)
result = reviewer.process_file(
    "examples/circuits/analog/rc_lowpass.cir",
    analyses=["dc_operating_point", "ac_sweep", "transient"],
)

print(result.summary())
```

You can also import a circuit first and reuse its `CircuitIR`:

```python
from circhecker import SchematicExtractor, CircuitReviewer

extractor = SchematicExtractor()
circuit = extractor.extract("examples/circuits/analog/rc_lowpass.cir")

reviewer = CircuitReviewer(plot=False)
result = reviewer.process_ir(circuit, analyses=["dc_operating_point", "ac_sweep"])
```

The `capability_profile()` function provides an overview of the implemented component and solver capabilities. Actual support still depends on the circuit topology, model parameters, and selected analysis; check each result status and diagnostic.

## Examples and notebooks

List the runnable examples:

```bash
python examples/run_examples.py --list
```

Run one example or the full catalog:

```bash
python examples/run_examples.py --example rc_lowpass
python examples/run_examples.py --example all
```

The catalog includes RC and RLC networks, rectifiers, diode and transistor circuits, CMOS inverter/NAND/NOR circuits, IC-style macro-models, and a multi-stage signal chain. Each example documents its intended use and model limitations in [`examples/README.md`](examples/README.md).

Five notebooks walk through the same public APIs:

| Notebook | Main subject |
|---|---|
| `01_integrated_signal_chain.ipynb` | Multi-stage circuit analysis |
| `02_analog_circuits.ipynb` | Passive networks and rectifiers |
| `03_semiconductor_amplifiers.ipynb` | Diodes, BJTs, and MOSFETs |
| `04_digital_logic.ipynb` | CMOS gates and ideal logic reference models |
| `05_ic_subcircuits.ipynb` | Hierarchical IC-style macro-models |

Start JupyterLab from the repository root with `jupyter lab`. To execute the notebooks in clean kernels, install the notebook extra and run:

```bash
python tools/execute_notebooks.py
```

## Project layout

```text
circhecker/    Importers, CircuitIR, validation, solvers, and reporting
examples/      Runnable circuits, example runner, and review plugins
notebooks/     Guided Jupyter notebooks
tests/        Automated tests, circuit fixtures, and expected JSON outputs
docs/         Architecture, integration, circuit catalog, and limitations
```

More detail is available in the [architecture guide](docs/ARCHITECTURE.md), [integration guide](docs/INTEGRATION.md), and [circuit catalog](docs/CIRCUIT_CATALOG.md).

## Testing

Install the test extra and run the default test suite:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

The default test configuration excludes tests marked `slow`. Run those separately when doing a full numerical regression check:

```bash
python -m pytest -m slow -q
```

GitHub Actions runs the default test suite on the supported Python matrix for pushes and pull requests.

## Important limitations

This project intentionally supports a defined subset of circuit syntax and device models. In particular:

- Its diode, BJT, MOSFET, and JFET models are compact educational models, not complete vendor SPICE models.
- Some parsed elements and directives are retained in the circuit representation but may not have a numerical backend.
- The digital logic helpers are ideal Boolean/state models; they do not simulate electrical timing or logic thresholds. CMOS gate examples are separate transistor-level electrical simulations.
- It is not a complete KiCad ERC implementation, mixed-signal simulator, parasitic/thermal analysis tool, or hardware sign-off system.

Read [the full limitations and model assumptions](docs/LIMITATIONS.md) before using results to make engineering decisions. A successful import or numerical solve does not certify a circuit.

## Contributing

Contributions are welcome, especially parser tests, numerical reference tests, documentation improvements, and clearly scoped device models. Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Please report security issues privately as described in [SECURITY.md](SECURITY.md).

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
