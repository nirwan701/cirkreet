# Supported scope

CirKreety is an educational circuit-review prototype. A successful import confirms that a supported source format was read. A successful solver result confirms that the selected numerical analysis completed; neither certifies a design.

## Import

- KiCad XML `.kicad_net` files are read from explicit component-pin-to-net mappings.
- Native KiCad `.kicad_sch` files are exported through the external `kicad-cli` executable.
- SPICE-compatible files use the parser's documented subset of elements, parameters, expressions, models, and analysis directives.
- LTspice `.asc` graphics are not interpreted. Export a SPICE-compatible netlist first.

## Numerical models

- Linear modified nodal analysis supports the implemented R/L/C, independent source, and selected controlled-source elements.
- DC operating point, DC sweep, AC sweep, transient analysis, and supported derived linear-system analyses are available according to the circuit topology.
- Nonlinear device models are compact approximations for selected diodes (including a piecewise-linear reverse-breakdown BV/RBV branch for Zener use), BJTs, MOSFETs, and JFETs. They are not full SPICE model-library implementations. Zener temperature coefficients and detailed breakdown physics are not implemented.
- Some parsed elements and directives are retained in the report but do not have a numerical execution model. Their analyses are marked unsupported.
- Adaptive transient output may be non-uniformly sampled. Check the returned time vector before applying sample-domain signal processing.

## Verification

- Structural checks cover references, pin records, connectivity evidence, unresolved models, and floating nodes.
- KCL, power balance, and solver diagnostics are included when the selected backend calculates them.
- The circuit-interchange JSON schema validates normalized extraction data; the full analysis report is checked for valid JSON serialization. Neither validates electrical design correctness.
- The public `circhecker.digital` 74HC-style functions are ideal Boolean/state models only; they are not part of the MNA simulation engine and do not calculate electrical timing, propagation delay, thresholds, setup/hold, or load effects. This project does not implement mixed-signal co-simulation or a general-purpose event-driven digital simulator, complete KiCad ERC, full SPICE compatibility, parasitic extraction, thermal analysis, or hardware safety certification.

Always inspect the analysis status, model assumptions, warnings, residuals, and verification checks before using numerical results in a real design.

## Integrated demonstration model boundaries

- The centre-tapped supply is represented by two ideal sinusoidal sources sharing the centre-tap reference. It is not a coupled transformer model; leakage inductance, winding resistance beyond the explicit 2 Ω source resistors, magnetizing current, core saturation and inter-winding capacitance are omitted.
- The shunt regulator uses an educational diode reverse-breakdown BV/RBV approximation. It is not a vendor-qualified Zener model and omits detailed temperature coefficient and breakdown dynamics.
- The relaxation oscillator uses voltage-controlled hysteretic switch primitives. It is not an NE555 model or a general-purpose digital logic simulator.
- The MOSFET amplifier uses the supported Level-1 compact model. It is not a BSIM model and omits electrothermal behavior and many device parasitics.
- The DC sweep varies the DC value of `VSEC_A` only, while `VSEC_B` remains fixed. It is a useful diagnostic sweep for the current model, but it is not a balanced AC line-regulation test.
- The AC sweep uses a dedicated small-signal test source and linearizes around the computed DC operating point. The transient waveform instead uses the oscillator as the large-signal input.
- Frequency analysis, transfer-function and state-space outputs may be reported `not_applicable` when a circuit uses nonlinear switching devices; that status is an intentional capability report.

Treat the example as a software and numerical-analysis demonstration, not a production power supply or amplifier design. Compare any design adaptation with manufacturer datasheets and trusted reference simulations.
