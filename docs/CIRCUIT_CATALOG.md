# Single integrated circuit example

The project has one user-facing circuit demonstration:

```text
../examples/schematics/integrated_ct_supply_oscillator_cs_mosfet.cir
../notebooks/01_integrated_signal_chain.ipynb
```

The circuit is one connected netlist and one extracted `CircuitIR`. Its five stages are:

| Stage | Function | Key devices / values |
|---|---|---|
| 1 | Centre-tapped full-wave rectifier | Two 12 Vrms, 50 Hz half-winding source abstractions; D1/D2 |
| 2 | Reservoir low-pass | Cbulk = 2200 µF; Rbleed = 100 kΩ |
| 3 | Shunt regulator | 680 Ω series resistor; 5.1 V educational Zener model; 1 kΩ load; 470 µF output capacitor |
| 4 | RC relaxation oscillator | 330 kΩ / 10 nF timing network; two voltage-controlled hysteretic switches |
| 5 | NMOS common-source amplifier | Level-1 NMOS; RD = 2.2 kΩ; RS = 470 Ω; oscillator-coupled gate input |

## Images

PNG and SVG schematics are supplied in `../examples/images/`. The full-sheet schematic shows all connected devices on one page; the overview and connectivity map show stage relationships; the four stage images document each subcircuit in greater detail. These drawings accompany the executable netlist; they are not parsed as raster/vector image input.

## Run

```bash
python -m pip install -e ".[test,notebooks]"
jupyter lab
```

Open `../notebooks/01_integrated_signal_chain.ipynb` and run all cells. For the same analysis without Jupyter:

```bash
python examples/run_examples.py
```

## Supported analyses in this topology

| Analysis | Result |
|---|---|
| Nonlinear DC operating point | Supported |
| DC sweep | Supported; the demonstration sweeps the DC value of VSEC_A only, so treat it as a diagnostic source sweep, not balanced AC line regulation |
| Small-signal AC sweep | Supported; linearized at the computed DC operating point through a separate AC test port |
| Nonlinear transient | Supported |
| Linear frequency analysis / transfer function / state space | Not applicable to this nonlinear switching topology |

Read [supported limitations](LIMITATIONS.md) before interpreting the model results as a physical design.
