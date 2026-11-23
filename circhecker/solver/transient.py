"""Sparse linear RLC transient MNA with Backward Euler and Trapezoidal rules.

The transient equations use the exact same DC MNA topology and branch-current
unknowns as the linear DC/AC engines.  Capacitor and inductor histories are
represented by integration-method-specific companion stamps; no transfer
function is used as a substitute for time-domain integration.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import sys
from typing import Any, Callable

import numpy as np
from scipy.sparse import csc_matrix, lil_matrix

from ..circuit_ir import CircuitIR
from ..parser.units import parse_spice_number
from .exceptions import MNAError, NumericalSolveError, UnsupportedElementError
from .matrix import solve_matrix
from .mna import assemble_dc, model_from_circuit_ir


@dataclass(frozen=True)
class StorageElement:
    reference: str
    kind: str
    p: int
    n: int
    value: float
    initial_condition: float | None = None


@dataclass
class TransientSolution:
    data: dict[str, Any]
    status: str = "success"
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "reason": self.reason, "data": self.data}


def _time_roundoff_tolerance(stop_time_s: float, step_count: int) -> float:
    """Estimate accumulated floating-point drift in repeated time updates.

    Fixed and adaptive transient loops advance time by repeated additions.  A
    mathematically exact final grid can therefore leave a femtosecond-scale
    remainder at ``tstop``.  Treat only an IEEE-754 round-off-sized remainder
    as the stop time; this is not a relaxation of the configured integration
    minimum step for a physical interval.
    """
    scale = max(abs(float(stop_time_s)), sys.float_info.min)
    operations = max(1, int(step_count))
    return max(1e-15, 8.0 * sys.float_info.epsilon * operations * scale)


def _source_function(component: dict[str, Any]) -> tuple[Callable[[float], float], list[float]]:
    parameters = component.get("parameters", {})
    dc = float(parameters.get("dc", 0.0))
    waveform = parameters.get("waveform")
    if not waveform:
        return (lambda _t, value=dc: value), []
    kind = str(waveform.get("kind", "")).upper()
    args = [parse_spice_number(value) for value in waveform.get("arguments", [])]
    if kind == "SIN" and len(args) >= 3:
        offset, amplitude, frequency = args[:3]
        delay = args[3] if len(args) > 3 else 0.0
        damping = args[4] if len(args) > 4 else 0.0
        phase = args[5] if len(args) > 5 else 0.0
        if frequency <= 0 or damping < 0:
            raise ValueError("SIN frequency must be positive and damping must be non-negative.")
        def value(t: float) -> float:
            if t < delay:
                return offset
            tau = t - delay
            return offset + amplitude * math.sin(2.0 * math.pi * frequency * tau + math.radians(phase)) * math.exp(-damping * tau)
        return value, [delay]
    if kind == "PULSE" and len(args) >= 2:
        low, high = args[:2]
        delay = args[2] if len(args) > 2 else 0.0
        rise = args[3] if len(args) > 3 else 0.0
        fall = args[4] if len(args) > 4 else 0.0
        width = args[5] if len(args) > 5 else 1e-3
        period = args[6] if len(args) > 6 else max(width + rise + fall, 2e-3)
        if min(rise, fall, width, delay) < 0 or period <= 0:
            raise ValueError("PULSE delay/rise/fall/width must be non-negative and period must be positive.")
        if rise + width + fall > period:
            raise ValueError("PULSE rise + width + fall cannot exceed the period.")
        def value(t: float) -> float:
            elapsed = t - delay
            if elapsed < 0:
                return low
            q = elapsed % period
            if rise > 0 and q < rise:
                return low + (high - low) * q / rise
            if q < rise + width:
                return high
            if fall > 0 and q < rise + width + fall:
                return high - (high - low) * (q - rise - width) / fall
            return low
        # Breakpoints in each cycle are generated lazily by the step controller.
        return value, [delay, rise, fall, width, period]
    if kind == "PWL" and len(args) >= 4 and len(args) % 2 == 0:
        points = list(zip(args[::2], args[1::2]))
        if any(t1 <= t0 for (t0, _), (t1, _) in zip(points, points[1:])):
            raise ValueError("PWL time coordinates must be strictly increasing.")
        def value(t: float) -> float:
            if t <= points[0][0]:
                return points[0][1]
            for (t0, v0), (t1, v1) in zip(points, points[1:]):
                if t <= t1:
                    return v0 + (v1 - v0) * (t - t0) / (t1 - t0)
            return points[-1][1]
        return value, [point[0] for point in points]
    raise UnsupportedElementError(
        f"{component.get('reference', '?')}: transient source waveform {kind or '<empty>'} "
        "is unsupported; implemented waveforms are SIN, PULSE, and PWL."
    )


def _next_breakpoint(current: float, proposed: float, components: list[dict[str, Any]]) -> float:
    target = proposed
    eps = max(1e-15, abs(current) * 1e-12)
    for component in components:
        waveform = component.get("parameters", {}).get("waveform") or {}
        kind = str(waveform.get("kind", "")).upper()
        try:
            args = [parse_spice_number(value) for value in waveform.get("arguments", [])]
        except Exception:
            continue
        candidates: list[float] = []
        if kind == "SIN" and len(args) >= 3:
            delay = args[3] if len(args) > 3 else 0.0
            candidates = [delay]
        elif kind == "PWL":
            candidates = args[::2]
        elif kind == "PULSE" and len(args) >= 2:
            delay = args[2] if len(args) > 2 else 0.0
            rise = args[3] if len(args) > 3 else 0.0
            fall = args[4] if len(args) > 4 else 0.0
            width = args[5] if len(args) > 5 else 1e-3
            period = args[6] if len(args) > 6 else max(width + rise + fall, 2e-3)
            if period > 0:
                cycle = max(0, int(math.floor((current - delay) / period)))
                for count in range(cycle, cycle + 3):
                    base = delay + count * period
                    candidates.extend((base, base + rise, base + rise + width, base + rise + width + fall))
        for point in candidates:
            if current + eps < point < target - eps:
                target = point
    return target


def supports_circuit_ir_transient(ir: CircuitIR) -> tuple[bool, list[str]]:
    """Return the explicit supported scope for linear RLC transient analysis."""
    reasons: list[str] = []
    if not ir.components:
        return False, ["CircuitIR contains no components to simulate"]
    allowed = {
        "R": {"value", "raw_value", "unit"},
        "C": {"value", "raw_value", "unit", "IC"},
        "L": {"value", "raw_value", "unit", "IC"},
        "V": {"dc", "ac_magnitude", "ac_phase_deg", "ac_specified", "waveform"},
        "I": {"dc", "ac_magnitude", "ac_phase_deg", "ac_specified", "waveform"},
        "E": {"gain", "raw_value"},
        "G": {"gain", "raw_value"},
        "F": {"control_source", "gain"},
        "H": {"control_source", "gain"},
    }
    if not any(str(c.get("type", "")).upper() in ("C", "L") for c in ir.components):
        reasons.append("No capacitor or inductor energy-storage element is present.")
    for component in ir.components:
        reference = str(component.get("reference", "?"))
        kind = str(component.get("type", "?")).upper()
        if kind not in allowed:
            reasons.append(f"{reference}: {kind} is outside the linear RLC transient subset")
            continue
        parameters = component.get("parameters", {})
        extra = sorted(set(parameters) - allowed[kind])
        if extra:
            reasons.append(f"{reference}: unsupported transient parameter(s): {', '.join(extra)}")
        if kind in ("C", "L") and float(parameters.get("value", 0.0)) <= 0:
            reasons.append(f"{reference}: {kind} value must be strictly positive for transient integration")
        if kind in ("V", "I") and parameters.get("waveform"):
            waveform = str(parameters["waveform"].get("kind", "")).upper()
            if waveform not in ("SIN", "PULSE", "PWL"):
                reasons.append(f"{reference}: unsupported time-domain waveform {waveform}")
    if not reasons:
        try:
            model_from_circuit_ir(ir)
            for component in ir.components:
                if component["type"] in ("V", "I"):
                    _source_function(component)
        except Exception as exc:
            reasons.append(str(exc))
    return not reasons, reasons


def _stamp_step(base_matrix: csc_matrix, base_rhs: np.ndarray, model,
                capacitors: list[StorageElement], inductors: list[StorageElement],
                previous_x: np.ndarray, node_index: dict[int, int], branch_index: dict[str, int],
                previous_cap_currents: dict[str, float], step_s: float, method: str):
    matrix = lil_matrix(base_matrix, dtype=float)
    rhs = base_rhs.copy()
    trapezoidal = method == "trapezoidal"
    for element in capacitors:
        ip = node_index.get(element.p)
        inn = node_index.get(element.n)
        conductance = (2.0 if trapezoidal else 1.0) * element.value / step_s
        if ip is not None:
            matrix[ip, ip] += conductance
        if inn is not None:
            matrix[inn, inn] += conductance
        if ip is not None and inn is not None:
            matrix[ip, inn] -= conductance
            matrix[inn, ip] -= conductance
        vp = 0.0 if ip is None else float(previous_x[ip])
        vn = 0.0 if inn is None else float(previous_x[inn])
        history = conductance * (vp - vn)
        if trapezoidal:
            history += previous_cap_currents[element.reference]
        if ip is not None:
            rhs[ip] += history
        if inn is not None:
            rhs[inn] -= history
    for element in inductors:
        branch = branch_index[element.reference.casefold()]
        coefficient = (2.0 if trapezoidal else 1.0) * element.value / step_s
        matrix[branch, branch] -= coefficient
        previous_current = float(previous_x[branch])
        history = -coefficient * previous_current
        if trapezoidal:
            ip = node_index.get(element.p)
            inn = node_index.get(element.n)
            previous_voltage = (0.0 if ip is None else float(previous_x[ip])) - (0.0 if inn is None else float(previous_x[inn]))
            history -= previous_voltage
        rhs[branch] += history
    return matrix.tocsc(), rhs


def solve_transient(ir: CircuitIR, *, stop_time_s: float | None = None, max_step_s: float | None = None,
                    method: str = "backward_euler", adaptive: bool = False,
                    relative_tolerance: float = 1e-3, absolute_voltage_tolerance: float = 1e-6,
                    minimum_step_s: float | None = None, maximum_steps: int = 200_000,
                    output_node: int | None = None, output_pair: tuple[int, int] | None = None,
                    absolute_current_tolerance: float = 1e-9,
                    absolute_charge_tolerance_C: float = 1e-15,
                    absolute_charge_current_tolerance_A: float = 1e-9,
                    absolute_flux_tolerance_Wb: float = 1e-12,
                    absolute_energy_tolerance_J: float = 1e-12,
                    maximum_newton_iterations: int = 60,
                    gmin_S: float = 1e-12) -> TransientSolution:
    """Run linear RLC or unified nonlinear semiconductor/RLC transient MNA.

    Nonlinear devices are dispatched to the terminal-charge companion-form
    engine; the established linear-only algorithm is retained unchanged.
    """
    nonlinear_kinds = {"D", "Q", "J", "M"}
    if any(str(component.get("type", "")).upper() in nonlinear_kinds for component in ir.components):
        from .nonlinear_transient import solve_nonlinear_transient
        return solve_nonlinear_transient(
            ir, stop_time_s=stop_time_s, max_step_s=max_step_s, method=method,
            adaptive=adaptive, relative_tolerance=relative_tolerance,
            absolute_voltage_tolerance=absolute_voltage_tolerance,
            absolute_current_tolerance=absolute_current_tolerance,
            absolute_charge_tolerance_C=absolute_charge_tolerance_C,
            absolute_charge_current_tolerance_A=absolute_charge_current_tolerance_A,
            absolute_flux_tolerance_Wb=absolute_flux_tolerance_Wb,
            absolute_energy_tolerance_J=absolute_energy_tolerance_J,
            gmin_S=gmin_S, minimum_step_s=minimum_step_s, maximum_steps=maximum_steps,
            maximum_newton_iterations=maximum_newton_iterations,
            output_node=output_node, output_pair=output_pair,
        )
    normalized_method = method.strip().lower().replace("-", "_")
    if normalized_method not in ("backward_euler", "trapezoidal"):
        raise ValueError("method must be 'backward_euler' or 'trapezoidal'.")
    supported, reasons = supports_circuit_ir_transient(ir)
    if not supported:
        raise UnsupportedElementError("Linear RLC transient MNA does not support this CircuitIR: " + "; ".join(reasons))
    config = ir.analysis_configuration.get("transient", {})
    stop = float(stop_time_s if stop_time_s is not None else config.get("stop_s", 0.0))
    step = float(max_step_s if max_step_s is not None else config.get("step_s", stop / 400.0 if stop > 0 else 0.0))
    if not math.isfinite(stop) or stop <= 0 or not math.isfinite(step) or step <= 0:
        raise ValueError("Transient stop time and maximum step must be finite and positive.")
    if (not math.isfinite(relative_tolerance) or not math.isfinite(absolute_voltage_tolerance)
            or relative_tolerance <= 0 or absolute_voltage_tolerance <= 0):
        raise ValueError("Transient tolerances must be finite and positive.")
    min_step = float(minimum_step_s if minimum_step_s is not None else max(1e-15, step * 1e-6))
    if min_step <= 0 or min_step > step:
        raise ValueError("minimum_step_s must be positive and no larger than max_step_s.")

    model = model_from_circuit_ir(ir)
    components = [c for c in ir.components if c["type"] in ("V", "I")]
    source_functions = {c["reference"].casefold(): _source_function(c)[0] for c in components}
    capacitors: list[StorageElement] = []
    inductors: list[StorageElement] = []
    for component in ir.components:
        if component["type"] not in ("C", "L"):
            continue
        pins = component["pins"]
        p, n = int(pins[0]["node_id"]), int(pins[1]["node_id"])
        params = component.get("parameters", {})
        state = StorageElement(component["reference"], component["type"], p, n,
                               float(params["value"]), float(params["IC"]) if params.get("IC") is not None else None)
        (capacitors if state.kind == "C" else inductors).append(state)

    initial_source_overrides = {
        component["reference"]: source_functions[component["reference"].casefold()](0.0)
        for component in components
    }
    base_matrix, base_rhs, node_index, branch_index, _ = assemble_dc(
        model, source_overrides=initial_source_overrides
    )
    # Build a physically consistent t=0 MNA system.  Inductor ICs replace the
    # inductor's DC zero-voltage equation with a prescribed branch current;
    # capacitor ICs add independent ideal-voltage constraints whose branch
    # currents are the initial capacitor currents.  Explicit .ic node voltages
    # are treated as auxiliary constraints as well.
    class _InitialPotentials:
        def __init__(self, nodes: set[int]):
            self.parent = {node: node for node in nodes}
            self.weight = {node: 0.0 for node in nodes}  # V(node) - V(parent)

        def find(self, node: int) -> tuple[int, float]:
            if node not in self.parent:
                self.parent[node] = node
                self.weight[node] = 0.0
            parent = self.parent[node]
            if parent == node:
                return node, 0.0
            root, parent_potential = self.find(parent)
            self.weight[node] += parent_potential
            self.parent[node] = root
            return root, self.weight[node]

        def add(self, p: int, n: int, value: float, description: str) -> bool:
            rp, vp = self.find(p)
            rn, vn = self.find(n)
            if rp == rn:
                if abs((vp - vn) - value) > 1e-8:
                    raise MNAError(f"{description} conflicts with an existing initial-voltage constraint.")
                return False
            self.parent[rp] = rn
            self.weight[rp] = value - vp + vn
            return True

    initial_potentials = _InitialPotentials({0, *node_index.keys()})
    # Seed the potential graph with independent ideal voltage sources at t=0.
    for component in components:
        if component["type"] != "V":
            continue
        p_node, n_node = (int(pin["node_id"]) for pin in component["pins"][:2])
        initial_potentials.add(
            p_node, n_node, source_functions[component["reference"].casefold()](0.0),
            f"{component['reference']} voltage-source equation",
        )

    extra_constraints: list[dict[str, Any]] = []
    for node_name, voltage in ir.initial_conditions.items():
        node = next((int(item["id"]) for item in ir.nodes
                     if str(item["name"]).casefold() == str(node_name).casefold()), None)
        if node is None:
            raise MNAError(f".ic references unknown node {node_name!r}.")
        prescribed = float(voltage)
        if initial_potentials.add(node, 0, prescribed, f".ic node {node_name!r}"):
            extra_constraints.append({"p": node, "n": 0, "value": prescribed, "kind": "node_ic"})

    # Merge parallel capacitor ICs into one constraint.  The source current is
    # then divided between parallel capacitors in proportion to capacitance.
    capacitor_groups: dict[tuple[int, int], dict[str, Any]] = {}
    for capacitor in capacitors:
        if capacitor.initial_condition is None:
            continue
        if capacitor.p == capacitor.n:
            if abs(float(capacitor.initial_condition)) > 1e-12:
                raise MNAError(f"{capacitor.reference}: a capacitor connected across one node cannot have a non-zero IC.")
            continue
        p_node, n_node = capacitor.p, capacitor.n
        if p_node < n_node:
            key, target, sign = (p_node, n_node), float(capacitor.initial_condition), 1.0
        else:
            key, target, sign = (n_node, p_node), -float(capacitor.initial_condition), -1.0
        group = capacitor_groups.setdefault(key, {"value": target, "members": [], "total_C": 0.0})
        if abs(group["value"] - target) > 1e-8:
            raise MNAError(f"{capacitor.reference}: capacitor IC conflicts with a parallel capacitor IC.")
        group["members"].append((capacitor, sign))
        group["total_C"] += capacitor.value

    for (p_node, n_node), group in capacitor_groups.items():
        if initial_potentials.add(p_node, n_node, group["value"], "capacitor IC"):
            extra_constraints.append({"p": p_node, "n": n_node, "value": group["value"],
                                      "kind": "capacitor_ic", "group": group})

    base_dimension = base_matrix.shape[0]
    augmented_dimension = base_dimension + len(extra_constraints)
    initial_matrix = lil_matrix((augmented_dimension, augmented_dimension), dtype=float)
    initial_matrix[:base_dimension, :base_dimension] = base_matrix
    initial_rhs = np.zeros(augmented_dimension, dtype=float)
    initial_rhs[:base_dimension] = base_rhs

    # Enforce inductor initial currents while retaining all node KCL equations.
    # An unspecified inductor IC is zero; retaining the DC short equation here
    # would incorrectly force zero inductor voltage at t=0 and contradict a
    # non-zero capacitor IC in a parallel RLC network.
    for inductor in inductors:
        branch = branch_index[inductor.reference.casefold()]
        initial_matrix.rows[branch] = [branch]
        initial_matrix.data[branch] = [1.0]
        initial_rhs[branch] = (0.0 if inductor.initial_condition is None
                               else float(inductor.initial_condition))

    constraint_branches: list[int] = []
    for offset, constraint in enumerate(extra_constraints):
        branch = base_dimension + offset
        p_node, n_node = int(constraint["p"]), int(constraint["n"])
        ip, inn = node_index.get(p_node), node_index.get(n_node)
        if ip is not None:
            initial_matrix[ip, branch] += 1.0
            initial_matrix[branch, ip] += 1.0
        if inn is not None:
            initial_matrix[inn, branch] -= 1.0
            initial_matrix[branch, inn] -= 1.0
        initial_rhs[branch] = float(constraint["value"])
        constraint_branches.append(branch)

    initial_solution, _, _, _, _, _, _ = solve_matrix(initial_matrix.tocsc(), initial_rhs)
    x = np.asarray(initial_solution[:base_dimension], dtype=float)
    previous_cap_currents = {element.reference: 0.0 for element in capacitors}
    for constraint, branch in zip(extra_constraints, constraint_branches):
        if constraint["kind"] != "capacitor_ic":
            continue
        group = constraint["group"]
        group_current = float(initial_solution[branch])
        for capacitor, sign in group["members"]:
            previous_cap_currents[capacitor.reference] = (
                sign * group_current * capacitor.value / group["total_C"]
            )

    time_values = [0.0]
    state_values = [x.copy()]
    accepted = rejected = linear_solves = 0
    step_sizes: list[float] = []
    current_step = min(step, stop)
    current_time = 0.0
    names = {int(node["id"]): str(node["name"]) for node in ir.nodes}

    while current_time < stop - _time_roundoff_tolerance(stop, accepted + rejected + 1):
        if accepted + rejected >= maximum_steps:
            raise MNAError(f"Transient integration exceeded maximum_steps={maximum_steps}.")
        proposed = min(current_time + current_step, stop)
        proposed = _next_breakpoint(current_time, proposed, components)
        h = proposed - current_time
        if h < min_step * (1.0 - 1e-12):
            raise MNAError(f"Transient timestep {h:.3e} s fell below minimum_step_s={min_step:.3e} s.")
        overrides = {component["reference"]: source_functions[component["reference"].casefold()](proposed)
                     for component in components}
        _, rhs_base, _, _, _ = assemble_dc(model, source_overrides=overrides)
        primary_matrix, primary_rhs = _stamp_step(base_matrix, rhs_base, model, capacitors, inductors,
                                                  x, node_index, branch_index, previous_cap_currents, h,
                                                  normalized_method)
        primary, _, _, _, _, _, _ = solve_matrix(primary_matrix, primary_rhs)
        linear_solves += 1
        trial = primary
        error_norm = 0.0
        if adaptive:
            embedded_method = "trapezoidal" if normalized_method == "backward_euler" else "backward_euler"
            embedded_matrix, embedded_rhs = _stamp_step(base_matrix, rhs_base, model, capacitors, inductors,
                                                         x, node_index, branch_index, previous_cap_currents, h,
                                                         embedded_method)
            embedded, _, _, _, _, _, _ = solve_matrix(embedded_matrix, embedded_rhs)
            linear_solves += 1
            voltage_error = float(np.linalg.norm(primary[:len(node_index)] - embedded[:len(node_index)], ord=np.inf)) if node_index else 0.0
            voltage_scale = max(absolute_voltage_tolerance,
                                float(np.linalg.norm(primary[:len(node_index)], ord=np.inf)) if node_index else 0.0)
            error_norm = voltage_error / (absolute_voltage_tolerance + relative_tolerance * voltage_scale)
            if error_norm > 1.0:
                if h <= min_step * (1.0 + 1e-12):
                    raise NumericalSolveError(
                        f"Adaptive transient error {error_norm:.3e} remains above one at minimum step {min_step:.3e} s."
                    )
                rejected += 1
                current_step = max(min_step, h * max(0.2, 0.8 / math.sqrt(max(error_norm, 1e-12))))
                continue
            if error_norm < 0.1:
                current_step = min(step, h * 1.5)
            else:
                current_step = h

        # Commit state only after the step has passed all acceptance checks.
        old_x = x
        x = np.asarray(trial, dtype=float)
        for capacitor in capacitors:
            ip, inn = node_index.get(capacitor.p), node_index.get(capacitor.n)
            old_v = (0.0 if ip is None else float(old_x[ip])) - (0.0 if inn is None else float(old_x[inn]))
            new_v = (0.0 if ip is None else float(x[ip])) - (0.0 if inn is None else float(x[inn]))
            if normalized_method == "backward_euler":
                previous_cap_currents[capacitor.reference] = capacitor.value * (new_v - old_v) / h
            else:
                previous_cap_currents[capacitor.reference] = 2.0 * capacitor.value * (new_v - old_v) / h - previous_cap_currents[capacitor.reference]
        current_time = proposed
        time_values.append(current_time)
        state_values.append(x.copy())
        step_sizes.append(h)
        accepted += 1

    final_time_roundoff_s = max(0.0, stop - current_time)
    final_time_snapped = final_time_roundoff_s <= _time_roundoff_tolerance(
        stop, accepted + rejected + 1
    )
    if final_time_snapped:
        # The final state differs from the exact stop instant only by the
        # accumulated floating-point error bounded above.  Snap the timestamp
        # so consumers receive a waveform ending at the requested TSTOP.
        time_values[-1] = stop

    states = np.asarray(state_values, dtype=float)
    node_voltages = {names[node]: states[:, index].tolist() for node, index in node_index.items()}
    branch_currents = {element.reference: states[:, index].tolist()
                       for element in model.elements
                       if element.reference.casefold() in branch_index
                       for index in [branch_index[element.reference.casefold()]]}
    if output_pair is not None:
        output = np.asarray(node_voltages.get(names.get(output_pair[0], ""), np.zeros(len(time_values)))) - np.asarray(
            node_voltages.get(names.get(output_pair[1], ""), np.zeros(len(time_values))))
    elif output_node is not None and output_node in names:
        output = np.asarray(node_voltages[names[output_node]])
    else:
        output = None
    diagnostics = {
        "backend": "circhecker.solver.transient",
        "method": normalized_method,
        "adaptive": bool(adaptive),
        "accepted_steps": int(accepted),
        "rejected_steps": int(rejected),
        "linear_solves": int(linear_solves),
        "minimum_accepted_step_s": float(min(step_sizes)) if step_sizes else None,
        "maximum_accepted_step_s": float(max(step_sizes)) if step_sizes else None,
        "minimum_configured_step_s": min_step,
        "maximum_configured_step_s": step,
        "final_time_snapped_to_stop": bool(final_time_snapped),
        "final_time_roundoff_adjustment_s": float(final_time_roundoff_s),
        "relative_tolerance": float(relative_tolerance),
        "absolute_voltage_tolerance_V": float(absolute_voltage_tolerance),
    }
    data: dict[str, Any] = {
        "time_s": time_values,
        "node_voltages_V": node_voltages,
        "branch_currents_A": branch_currents,
        "integration_method": normalized_method,
        "solver_diagnostics": diagnostics,
        "output_V": output.tolist() if output is not None else None,
        "output_pair": list(output_pair) if output_pair is not None else None,
    }
    return TransientSolution(data)
