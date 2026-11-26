# Circuit examples

Begin with the runnable [Python RC lab](student_lab.py) and the [student guide](../docs/STUDENT_GUIDE.md). The notebooks are guided lessons; the circuit runner and reusable analysis logic live in the Python package.

The executable circuit catalog lives in `circuits/`; the integrated five-stage reference design remains in `schematics/`. The `.cir` files are authoritative connectivity/model inputs. Images are explanatory drawings and are not parsed as circuit inputs.

Run one example:

```bash
python examples/run_examples.py --example cmos_inverter
```

Run them all:

```bash
python examples/run_examples.py --list
python examples/run_examples.py --example all
```

Each run writes a CircuitIR JSON, an analysis JSON, and (unless `--no-plots` is used) a four-panel review image under `reports/example_catalog/<example-id>/`.

| ID | Circuit source | Intended use |
|---|---|---|
| `rc_lowpass` | `circuits/analog/rc_lowpass.cir` | RC step and AC response |
| `series_rlc` | `circuits/analog/series_rlc.cir` | Resonance and damping |
| `bridge_rectifier` | `circuits/analog/bridge_rectifier.cir` | Rectification and reservoir ripple |
| `diode_clamp` | `circuits/semiconductor/diode_clamp.cir` | Diode limiter waveform |
| `bjt_common_emitter` | `circuits/semiconductor/bjt_common_emitter.cir` | BJT bias and amplification |
| `mosfet_common_source` | `circuits/semiconductor/mosfet_common_source.cir` | NMOS bias and amplification |
| `cmos_inverter` | `circuits/digital/cmos_inverter.cir` | Transistor-level CMOS inverter switching |
| `cmos_nand2` | `circuits/digital/cmos_nand2.cir` | Two-input CMOS NAND with series NMOS / parallel PMOS |
| `cmos_nor2` | `circuits/digital/cmos_nor2.cir` | Two-input CMOS NOR with parallel NMOS / series PMOS |
| `timer_555_style_macro` | `circuits/ic/timer_555_style_macro.cir` | RC timing with hysteretic switch primitives; educational abstraction, not an NE555 model |
| `opamp_subcircuit` | `circuits/ic/idealized_opamp_subcircuit.cir` | Reusable nested IC macro-model |
| `integrated_signal_chain` | `schematics/integrated_ct_supply_oscillator_cs_mosfet.cir` | Integrated multi-stage nonlinear design |

## Model fidelity

- Diodes, BJTs, MOSFETs and JFETs use the solver's documented compact subsets, not complete commercial SPICE models.
- The op-amp example is an idealized controlled-source subcircuit; it is not an LM358/741 vendor macro-model.
- The CMOS inverter, NAND and NOR examples are educational Level-1 NMOS/PMOS simulations. It omits process-corner, threshold, timing, layout, and thermal validation.
- `circhecker.digital` provides a reusable ideal logic-network API and 74HC-style state-reference models; `examples/digital/logic_models.py` re-exports these examples. It does not calculate electrical timing, voltage thresholds, loading, setup/hold, or metastability.
- Native manufacturer chip models should be added only after the required model syntax and device parameters are supported and validated against datasheets/reference simulations.

Never treat a successful numerical solve as design sign-off. Inspect the reported status, warnings, model assumptions, residuals and analysis diagnostics.

## Generated circuit connectivity images

Generate source-grounded pin-to-net maps for every catalog entry:

```bash
python examples/generate_circuit_images.py
```

The generator parses each authoritative `.cir` file through `SchematicExtractor` and writes a PNG and SVG under `examples/images/catalog/`. These are pin-to-net connectivity references, not hand-drawn CAD schematics or inputs to the parser.

| Circuit | PNG | SVG |
|---|---|---|
| RC low-pass | [PNG](images/catalog/rc_lowpass_connectivity.png) | [SVG](images/catalog/rc_lowpass_connectivity.svg) |
| Series RLC | [PNG](images/catalog/series_rlc_connectivity.png) | [SVG](images/catalog/series_rlc_connectivity.svg) |
| Bridge rectifier | [PNG](images/catalog/bridge_rectifier_connectivity.png) | [SVG](images/catalog/bridge_rectifier_connectivity.svg) |
| Diode clamp | [PNG](images/catalog/diode_clamp_connectivity.png) | [SVG](images/catalog/diode_clamp_connectivity.svg) |
| BJT common-emitter | [PNG](images/catalog/bjt_common_emitter_connectivity.png) | [SVG](images/catalog/bjt_common_emitter_connectivity.svg) |
| MOSFET common-source | [PNG](images/catalog/mosfet_common_source_connectivity.png) | [SVG](images/catalog/mosfet_common_source_connectivity.svg) |
| CMOS inverter | [PNG](images/catalog/cmos_inverter_connectivity.png) | [SVG](images/catalog/cmos_inverter_connectivity.svg) |
| CMOS NAND | [PNG](images/catalog/cmos_nand2_connectivity.png) | [SVG](images/catalog/cmos_nand2_connectivity.svg) |
| CMOS NOR | [PNG](images/catalog/cmos_nor2_connectivity.png) | [SVG](images/catalog/cmos_nor2_connectivity.svg) |
| 555-style timer | [PNG](images/catalog/timer_555_style_macro_connectivity.png) | [SVG](images/catalog/timer_555_style_macro_connectivity.svg) |
| Op-amp subcircuit | [PNG](images/catalog/opamp_subcircuit_connectivity.png) | [SVG](images/catalog/opamp_subcircuit_connectivity.svg) |
| Integrated signal chain | [PNG](images/catalog/integrated_signal_chain_connectivity.png) | [SVG](images/catalog/integrated_signal_chain_connectivity.svg) |
