# Student guide

CirKreet is designed to help you connect circuit theory with numerical
simulation. Start with a simple circuit, change one parameter, and compare the
result with the equation you learned in class.

## 1. Install and run the first lab

From a terminal in the repository folder:

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\\Scripts\\Activate.ps1

python -m pip install -e ".[test]"
python examples/student_lab.py
```

The lab creates a JSON result and a plot in `reports/student_lab/`. It analyses
a first-order RC low-pass filter using DC operating point, AC sweep, and
transient simulation.

## 2. Understand the circuit

The example starts with a 5 V pulse source, a 1 kΩ resistor, and a 100 nF
capacitor. The ideal RC cutoff frequency is

\[
 f_c = \frac{1}{2\pi RC}
\]

For the starter values, the expected cutoff is approximately 1.59 kHz. Use the
AC plot to see where the output magnitude begins to decrease. Use the
transient plot to see how the capacitor voltage responds to the input pulse.

## 3. Change one component value

Open `examples/student_lab.py` and edit one value at a time:

- Change `1k` to `2.2k` in `R1`.
- Change `100n` to `47n` in `C1`.
- Change the pulse high level from `5` to `3.3` volts.

Run the script again and compare the cutoff frequency, rise time, and output
voltage. Write down your prediction before running the simulation.

## 4. Use the Python API in your own program

```python
from circhecker import (
    ACSweepSpec,
    AnalysisSpec,
    CircuitReviewer,
    TransientSpec,
)

netlist = """\
.title My RC filter
V1 in 0 PULSE(0 5 0 1u 1u 1m 2m) AC 1
R1 in out 1k
C1 out 0 100n
.end
"""

request = AnalysisSpec(
    analyses=("dc_operating_point", "ac_sweep", "transient"),
    ac_sweep=ACSweepSpec(start_Hz=10, stop_Hz=100_000, points=120),
    transient=TransientSpec(stop_s=2e-3, step_s=2e-6),
    output_node="out",
)

result = CircuitReviewer(plot=False).process_netlist_text(
    netlist,
    analysis_spec=request,
    save_json=False,
)

print(result.human_summary())
print(result.simulation_result.to_dict())
```

`CircuitIR` describes the electrical circuit. `AnalysisSpec` describes what
you want to calculate. `AnalysisIR` is the normalized execution plan created
from both. `SimulationResult` stores the outcomes in a consistent structure
with units, messages, and solver diagnostics.

## 5. Try another example

```bash
python examples/run_examples.py --list
python examples/run_examples.py --example series_rlc --no-plots
```

Suggested exercises:

1. **Voltage divider:** predict the output voltage, then compare it with the
   DC operating-point result.
2. **RC low-pass:** calculate the cutoff frequency analytically, then locate it
   on the AC response.
3. **Series RLC:** calculate the resonant frequency and compare it with the AC
   sweep.
4. **Diode clamp:** identify when the diode begins conducting in the transient
   waveform.

## 6. Read results responsibly

A successful numerical solve means the selected model and solver completed.
It does not prove that a physical circuit is safe or that every real component
is represented in detail. Read the warning messages and the project's
[model limitations](LIMITATIONS.md), especially before comparing results with
hardware measurements.
