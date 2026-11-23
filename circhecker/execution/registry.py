"""Small, explicit capability registries for element and analysis backends."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ComponentCapability:
    prefix: str
    category: str
    terminal_count: int | None
    linear_mna: bool = False
    linear_ac: bool = False
    nonlinear_analog: bool = False
    nonlinear_mna: bool = False
    transient_mna: bool = False
    model_family: str = "ideal / not modelled"
    notes: str = ""


class ComponentRegistry:
    def __init__(self, entries: Iterable[ComponentCapability]):
        self._entries = {entry.prefix.upper(): entry for entry in entries}

    def get(self, prefix: str) -> ComponentCapability:
        return self._entries.get(str(prefix).upper(), ComponentCapability(
            prefix=str(prefix).upper(), category="unsupported", terminal_count=None,
            notes="No component adapter is registered for this element prefix."))

    def profile(self) -> dict:
        return {k: vars(v) for k, v in sorted(self._entries.items())}


class SolverRegistry:
    def __init__(self):
        self._entries = {
            "sparse_mna_dc": {
                "module": "circhecker.solver.mna",
                "elements": ["R", "C(open in DC)", "L(ideal short in DC)", "V", "I", "E", "G", "F", "H"],
                "analyses": ["dc_operating_point"],
                "limitations": "Generalized sparse linear DC MNA for R/C/L/V/I/E/G/F/H; nonlinear devices are handled by a separately gated Newton backend.",
            },
            "sparse_mna_ac": {
                "module": "circhecker.solver.ac",
                "elements": ["R", "C(jwC)", "L(branch-current impedance)", "V", "I", "E", "G", "F", "H"],
                "analyses": ["ac_sweep"],
                "limitations": "Linear AC subset only; direct solve_ac also accepts a single frequency; nonlinear devices use the legacy small-signal adapter.",
            },
            "nonlinear_mna_dc": {
                "module": "circhecker.solver.nonlinear",
                "elements": ["R", "C(open in DC)", "L(ideal short in DC)", "V", "I", "E", "G", "F", "H", "D(Shockley)", "Q(compact Gummel-Poon subset)", "M(Level-1)", "J(Level-1 JFET)"],
                "analyses": ["dc_operating_point"],
                "limitations": "Shared terminal-current/Jacobian Newton DC and operating-point AC for D/Q/M/J compact subsets. No diode series resistance or breakdown, no full SPICE Gummel-Poon/BSIM, no nonlinear transient DAE, and only declared charge terms are included.",
            },
            "mna_transient": {
                "module": "circhecker.solver.transient",
                "elements": ["R", "C(linear)", "L(ideal lumped)", "V(SIN/PULSE/PWL)", "I", "E", "G", "F", "H", "D(Shockley)", "Q(compact BJT subset)", "M(Level-1 subset)", "J(Level-1 JFET)"],
                "analyses": ["transient"],
                "limitations": "Linear RLC plus supported D/Q/M/J compact models; backward Euler and trapezoidal integration; adaptive step doubling; terminal charge derivatives and linear inductor flux. No coupled inductors, nonlinear inductance, UIC/TSTART .tran options, behavioral sources, switches, or nonlinear solver model families beyond declared subsets.",
            },
            "linear_mna": {
                "module": "circhecker.solver.linear.circuitsolver + circhecker.solver.transfer_function.transferfunction",
                "elements": ["R", "C", "L", "V", "I", "E", "G", "Q(linearized)"],
                "analyses": ["dc_operating_point", "dc_sweep", "ac_sweep", "transient", "frequency_analysis", "transfer_function", "state_space"],
                "limitations": "Supported linear subset only; singular descriptor systems may not admit all derived analyses.",
            },
            "nonlinear_analog": {
                "module": "circhecker.solver.analog.analogcircuit",
                "elements": ["R", "C", "V", "D", "Q", "J", "M", "E/OTA", "G", "S"],
                "analyses": ["dc_operating_point", "dc_sweep", "ac_sweep", "transient"],
                "limitations": "Educational models; grounded independent voltage source constraints; capacitor-oriented transient; AC is local small signal.",
            },
        }

    def get(self, name: str) -> dict | None:
        return self._entries.get(name)

    def profile(self) -> dict:
        return {k: dict(v) for k, v in self._entries.items()}


DEFAULT_COMPONENT_REGISTRY = ComponentRegistry([
    ComponentCapability("R", "passive", 2, linear_mna=True, linear_ac=True, nonlinear_analog=True, nonlinear_mna=True, transient_mna=True, model_family="ideal resistor"),
    ComponentCapability("C", "passive", 2, linear_mna=True, linear_ac=True, nonlinear_analog=True, nonlinear_mna=True, transient_mna=True, model_family="ideal capacitor"),
    ComponentCapability("L", "passive", 2, linear_mna=True, linear_ac=True, nonlinear_mna=True, transient_mna=True, model_family="ideal inductor", notes="Ideal lumped inductor; no saturation model."),
    ComponentCapability("V", "source", 2, linear_mna=True, linear_ac=True, nonlinear_analog=True, nonlinear_mna=True, transient_mna=True, model_family="independent voltage source", notes="SIN/PULSE/PWL supported in the new linear transient subset."),
    ComponentCapability("I", "source", 2, linear_mna=True, linear_ac=True, nonlinear_mna=True, transient_mna=True, model_family="independent current source", notes="New transient and nonlinear MNA capability gates check supported source parameters."),
    ComponentCapability("D", "semiconductor", 2, nonlinear_analog=True, nonlinear_mna=True, transient_mna=True, model_family="Shockley diode / piecewise Zener", notes="Unified MNA transient supports Shockley IS/N plus declared depletion/diffusion charge; legacy analog Zener approximation is not used by MNA."),
    ComponentCapability("Q", "semiconductor", 3, nonlinear_analog=True, nonlinear_mna=True, transient_mna=True, model_family="compact Gummel-Poon-inspired BJT subset", notes="Forward/reverse transport, BF/BR, Early modulation and junction/diffusion charge subset; not full SPICE GP."),
    ComponentCapability("J", "semiconductor", 3, nonlinear_analog=True, nonlinear_mna=True, transient_mna=True, model_family="Level-1 square-law JFET", notes="NJF/PJF, ohmic/saturation/cutoff, gate junctions and optional linear CGS/CGD charge in unified MNA transient."),
    ComponentCapability("M", "semiconductor", 4, nonlinear_analog=True, nonlinear_mna=True, transient_mna=True, model_family="Level-1 square-law MOSFET", notes="NMOS/PMOS, body effect, channel-length modulation, body diodes and declared overlap charge subset in unified MNA transient."),
    ComponentCapability("E", "controlled_source", 4, linear_mna=True, linear_ac=True, nonlinear_analog=True, nonlinear_mna=True, transient_mna=True, model_family="VCVS / optional rail-limited OTA"),
    ComponentCapability("G", "controlled_source", 4, linear_mna=True, linear_ac=True, nonlinear_analog=True, nonlinear_mna=True, transient_mna=True, model_family="VCCS"),
    ComponentCapability("F", "controlled_source", 2, linear_mna=True, linear_ac=True, nonlinear_mna=True, transient_mna=True, model_family="CCCS", notes="Generalized DC MNA uses the branch current of a named voltage-source reference."),
    ComponentCapability("H", "controlled_source", 2, linear_mna=True, linear_ac=True, nonlinear_mna=True, transient_mna=True, model_family="CCVS", notes="Generalized DC MNA uses the branch current of a named voltage-source reference."),
    ComponentCapability("S", "switch", 4, nonlinear_analog=True, model_family="hysteretic voltage-controlled switch"),
    ComponentCapability("W", "switch", 3, model_family="current-controlled switch", notes="Parsed but not yet simulated."),
    ComponentCapability("K", "magnetic", 0, model_family="coupled inductors", notes="Coupling record parsed; mutual inductance not assembled in solver."),
    ComponentCapability("B", "behavioral_source", 2, model_family="behavioral source", notes="Preserved but not evaluated."),
    ComponentCapability("T", "transmission_line", 4, model_family="transmission line", notes="Preserved but not evaluated."),
    ComponentCapability("X", "subcircuit", None, model_family="subcircuit instance", notes="File/text parser expands positional instances for the documented nested numeric-parameter subset before solver capability selection."),
])
DEFAULT_SOLVER_REGISTRY = SolverRegistry()
