# Circuit analysis workflow

CirKreet separates the description of a circuit from the request to analyse
it and from the numerical results. This keeps the Python API predictable and
makes reports easier to inspect or use in another program.

## The four public data objects

| Object | What it represents | Main contents |
|---|---|---|
| `CircuitIR` | The electrical circuit | Components, pins, nets, nodes, branches, models and source provenance |
| `AnalysisSpec` | The user's request | Which analyses to run and their settings, expressed in SI units |
| `AnalysisIR` | The normalized execution plan | Circuit ID, requested analyses, effective settings and output selection |
| `SimulationResult` | The numerical outcomes | Per-analysis status, data, units, messages and solver diagnostics |

The importer creates `CircuitIR`. The user supplies an `AnalysisSpec`. CirKreet
combines those into an `AnalysisIR`, then dispatches the requested analyses to
the supported numerical solver. The solver outputs are exposed through the
legacy `CircuitResult.analysis_results` interface and the normalized
`CircuitResult.simulation_result` interface.

```text
SPICE / KiCad input
       |
       v
   CircuitIR  ---------------------- source directives are preserved
       |
       +---- AnalysisSpec (user request)
                     |
                     v
                 AnalysisIR
                     |
          +----------+----------+
          |          |          |
          v          v          v
         DC         AC       Transient
          \          |         /
           \         |        /
            +--------+-------+
                     |
                     v
             SimulationResult
                     |
              JSON / plots / API
```

## Python example

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
print(result.analysis_ir.to_dict())
print(result.simulation_result.to_dict())
```

## Numerical methods

- **DC operating point:** capacitors are open circuits; ideal inductors are
  represented by zero-voltage branch constraints.
- **AC sweep:** linearizes the supported circuit about its operating point and
  solves the complex frequency-domain MNA system at each requested frequency.
- **Transient:** integrates the supported RLC and compact nonlinear-device
  equations using Backward Euler or Trapezoidal integration. Adaptive step
  control is available for supported transient models.
- **DC sweep:** repeatedly solves the DC operating point while changing one
  independent source.

The native sparse solvers use NumPy and SciPy. Solver selection is capability
checked; CirKreet reports unsupported topologies rather than silently treating
an unmodelled element as an ideal element.

## Stable identifiers and reproducibility

`CircuitIR.fingerprint()` returns a SHA-256 identifier based on normalized
electrical content. `AnalysisIR.analysis_id` identifies that circuit together
with its normalized analysis request. These IDs are included in the JSON report
so results can be traced to the circuit and settings that produced them.

## Result contract

Every analysis in `SimulationResult.analyses` uses the same fields:

- `status`: `success`, `failed`, `unsupported`, `not_applicable`, or `skipped`;
- `message`: a readable explanation of the outcome;
- `data`: the numerical result for that analysis;
- `units`: unit labels for public result fields;
- `diagnostics`: solver and execution diagnostics when available.

For compatibility, `CircuitResult.analysis_results` keeps its established
`status` / `reason` / `data` structure. New integrations should use
`CircuitResult.simulation_result`.

## Scope

The supported device and topology coverage is described in
[`LIMITATIONS.md`](LIMITATIONS.md). Always inspect per-analysis status,
warning messages, solver residuals, and model assumptions before relying on a
result.
