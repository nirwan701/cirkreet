# Integration guide

CirKreety is organized around small public boundaries so applications can use the parser, intermediate representation, execution engine, digital reference models, and review plugins independently.

## 1. Import and review a circuit

```python
from circhecker import CircuitReviewer, SchematicExtractor

extractor = SchematicExtractor()
circuit = extractor.extract("examples/circuits/analog/rc_lowpass.cir")
structural_report = extractor.validate(circuit)

review = CircuitReviewer(output_dir="reports/my_project", plot=False)
result = review.process_ir(
    circuit,
    analyses=["dc_operating_point", "ac_sweep", "transient"],
    save_json=False,
    plot=False,
)
print(result.summary())
```

This preserves a single extraction step. The same `CircuitIR` can be inspected, annotated by an application, or supplied to `process_ir` without reparsing the source.

## 2. Discover analysis and component support

```python
from circhecker import capability_profile

profile = capability_profile()
for component in profile["components"]:
    print(component)
for solver in profile["solvers"]:
    print(solver)
```

Treat this as an inventory, not a guarantee that every topology or model parameter is supported. The returned analysis record for a particular circuit is authoritative: inspect each status, warning, error, and diagnostic.

## 3. Add project-specific review checks

A review plugin implements `name` and `run(circuit, execution)`. It runs after core analyses and returns a mapping that is placed in `CircuitResult.plugin_results` and the JSON report.

```python
from circhecker import CircuitReviewer

class ExampleProjectRule:
    name = "example_project_rule"

    def run(self, circuit, execution):
        return {
            "status": "reviewed",
            "component_count": len(circuit.components),
            "analysis_backend": execution.get("execution_summary", {}).get("backend"),
        }

result = CircuitReviewer(review_plugins=[ExampleProjectRule()]).process_file(
    "examples/circuits/analog/rc_lowpass.cir",
    analyses=["dc_operating_point"],
    save_json=False,
    plot=False,
)
print(result.plugin_results)
```

Plugins should be deterministic where possible, identify their assumptions, and distinguish warnings from verified violations. They must not silently modify solver outputs. A plugin failure is recorded as a warning and does not invalidate a successful numerical solve.

## 4. Compose functional digital logic

`circhecker.digital.LogicNetwork` is a zero-delay Boolean evaluator for combinational logic. It is separate from the electrical MNA engine.

```python
from circhecker.digital import LogicNetwork

mux = LogicNetwork()
mux.add_gate("not_sel", "NOT", "sel")
mux.add_gate("d0_selected", "AND", "d0", "not_sel")
mux.add_gate("d1_selected", "AND", "d1", "sel")
mux.add_gate("y", "OR", "d0_selected", "d1_selected")
print(mux.truth_table(("d0", "d1", "sel"), ("y",)))
```

Supported gates are `NOT`, `AND`, `NAND`, `OR`, `NOR`, `XOR`, and `XNOR`. `HC74` and `HC595` provide selected ideal state behavior. They do not model electrical voltage levels, loading, propagation delay, setup/hold time, metastability, or switching power. For electrical CMOS behavior, use the corresponding supported transistor-level netlists.

## 5. Add a component or analysis backend

A numerical extension should follow this sequence:

1. Preserve component reference, value, model reference, pin numbers, pin-to-node mapping, source expression, and provenance in `CircuitIR`.
2. Declare support and parameter constraints in the capability/registry layer.
3. Implement the correct constitutive equation or MNA stamping in the appropriate solver family.
4. Return structured numerical results and explicit failure/unsupported reasons.
5. Add analytic reference tests, boundary-condition tests, convergence/failure tests, and at least one reproducible circuit fixture.
6. Add a catalog entry and a notebook cell that uses the public API; document the model fidelity and limitations.

Do not silently replace an unsupported device with an ideal source or generic resistor. Keep manufacturer-specific macro-models distinct from educational abstractions, and cite the source and permitted redistribution terms for any vendor model added to the repository.
