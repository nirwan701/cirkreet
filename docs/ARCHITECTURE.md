# CirKreet architecture

## Design intent

CirKreet is a Python-first circuit-analysis toolkit. It imports selected KiCad and SPICE-compatible circuit descriptions into one normalized `CircuitIR`, validates the extracted structure, selects a supported numerical backend, and exports inspectable results.

## Analysis pipeline

```text
KiCad netlist / supported SPICE input
                  |
                  v
              CircuitIR
                  |
                  +------ AnalysisSpec (student/user request)
                               |
                               v
                           AnalysisIR
                               |
                  +------------+------------+
                  |            |            |
                  v            v            v
                DC MNA       AC MNA     Transient MNA
                  \            |            /
                   +-----------+-----------+
                               |
                               v
                       SimulationResult
                               |
                     JSON report / plots
```

`CircuitIR` contains the normalized electrical circuit. `AnalysisSpec`
contains per-run settings and does not rewrite the source directives.
`AnalysisIR` records the effective, normalized execution plan. The native
solver results are exposed through `SimulationResult`, which provides
consistent status, messages, unit labels, and diagnostics.

## Main package areas

| Package | Responsibility |
|---|---|
| `circhecker/importers/` | Detect and load supported KiCad and SPICE input forms |
| `circhecker/parser/` | Parse supported SPICE syntax and normalize numeric units |
| `circhecker/circuit_ir.py` | Shared intermediate representation for components, pins, nodes, models and source metadata |
| `circhecker/validation/` | Structural extraction checks and schema validation |
| `circhecker/analysis/` | Typed `AnalysisSpec` requests and canonical `AnalysisIR` execution plans |
| `circhecker/solver/` | Native sparse DC/AC MNA, nonlinear semiconductor MNA, and linear/nonlinear transient solvers |
| `circhecker/execution/` | Capability registries and execution orchestration |
| `circhecker/engineering/` | Signal measurements and derived engineering metrics |
| `circhecker/digital/` | Ideal Boolean network, truth-table, flip-flop and shift-register reference models |
| `circhecker/reporting/` | Human-readable summaries, normalized simulation results, and report serialization |
| `circhecker/exporters/` | Circuit interchange JSON export |

## Supported analysis families

The implementation includes DC operating point, DC sweep, AC sweep and transient paths, plus selected linear frequency/transfer-function/state-space and signal-measurement functions. Exact element support depends on the backend and topology. The registries in `circhecker/execution/registry.py` and capability decisions in the solver should be consulted before assuming that a parsed element is numerically simulated.

## Data contract

`CircuitIR` is the shared boundary between import and execution. Importers preserve source references, terminal-to-node mappings, model references, raw expressions and warnings. `AnalysisIR` is built from the circuit and a user request; its SHA-256 identifiers connect a result to the electrical content and effective analysis settings. Numerical backends consume the normalized representation and report unsupported elements or parameters rather than silently treating them as ideal devices.

## Extending the system

1. Add or refine a parser/importer mapping into `CircuitIR`.
2. Add a capability declaration and explicit parameter checks.
3. Implement the element's stamping/constitutive equations in the correct solver family.
4. Add unit tests for element behavior and numerical convergence.
5. Add a reproducible circuit fixture and expected-result bounds.
6. Add a runnable example and explain its model fidelity and analysis limits.

The public `circhecker.digital` package contains ideal Boolean network and state-reference models. It is intentionally separate from MNA: `LogicNetwork` evaluates combinational logic with zero delay, while `HC74` and `HC595` model selected state behavior. They do not model voltage thresholds, loading, propagation delay, setup/hold, metastability or power. The `examples/digital/` module only re-exports the public API for backwards-compatible examples. Transistor-level CMOS gates are represented separately as SPICE netlists and run through the nonlinear electrical solver. The project is not a full SPICE replacement. See [LIMITATIONS.md](LIMITATIONS.md) and [IC_MODELING_GUIDE.md](IC_MODELING_GUIDE.md) before treating an example as a vendor model or a design sign-off.
