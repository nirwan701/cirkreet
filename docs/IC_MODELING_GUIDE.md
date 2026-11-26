# Device-model assumptions in the integrated example

This notebook uses primitive electrical devices already supported by CirKreety. It does not claim that a circuit built from a Zener and switches is a transistor-level model of a commercial timer/regulator IC.

| Device / stage | CirKreety representation | Scope boundary |
|---|---|---|
| Rectifier diodes D1/D2 | Shockley diode model (`IS`, `N`) | Educational diode I–V approximation; no detailed package parasitics, reverse recovery, thermal drift or vendor corner data |
| Shunt regulator DZ | Diode with `BV` and `RBV` reverse-breakdown parameters | Simplified reverse-breakdown/Zener behavior; not a manufacturer-validated Zener model |
| Relaxation oscillator Sdis/Sout | Voltage-controlled hysteretic switch models | Captures RC ramp and switching for this demonstration; not an NE555 internal model, and not a general digital simulator |
| Amplifier transistor M1 | Level-1 NMOS compact model (`VTO`, `KP`, `LAMBDA`, `CGS`, `CGD`) | Educational MOSFET approximation; not a BSIM or vendor model; no electrothermal coupling or detailed parasitic network |
| R/C/V elements | Ideal lumped resistors, capacitors and independent voltage sources | No component tolerance, self-heating, ESR/ESL, transformer magnetic coupling or PCB parasitics unless explicitly represented |

## Interpreting analyses

- The DC operating point is a static numerical solution. It does not show the oscillator's periodic time-domain activity.
- The DC sweep in the example varies `VSEC_A`'s DC source value only. Since `VSEC_B` remains fixed, this is a one-source diagnostic sweep, not balanced variation of both transformer half-windings.
- The AC sweep is a local small-signal linearization around the computed DC operating point. A dedicated AC test port is capacitively coupled to the MOSFET gate for this analysis.
- The transient analysis uses the oscillator output as the amplifier's large-signal input. The ratio of output AC RMS to gate AC RMS is therefore a large-signal waveform ratio, not the same as the local AC-sweep gain.
- Linear frequency-analysis, transfer-function and state-space modules are not applicable to this nonlinear switching circuit and should remain reported as such.

## Engineering use

Treat results as educational and exploratory. Before using a similar topology in hardware, select manufacturer devices, verify ratings and Zener dissipation, compute transformer and rectifier current margins, assess capacitor ripple current and inrush, validate MOSFET bias and thermal dissipation, and compare against a trusted reference simulator and datasheets.
