"""Nonlinear transient MNA using the shared terminal semiconductor models.

The dynamic equations are assembled as
    G*x + f_device(x) + d(Q_global(x))/dt = b(t)
with ideal inductor branch equations
    v_L - d(Phi(i_L))/dt = 0.
All semiconductor currents, charges and derivatives come from the same
constitutive device instances used by nonlinear DC and AC analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping

import numpy as np
from scipy.sparse import csc_matrix, coo_matrix, diags, lil_matrix

from ..circuit_ir import CircuitIR
from ..parser.units import parse_spice_number
from .exceptions import MNAError, NumericalSolveError, SingularMatrixError, UnsupportedElementError
from .matrix import solve_matrix
from .mna import assemble_dc
from .nonlinear import (
    _build_system,
    _residual_and_jacobian,
    _semiconductors_from_ir,
    supports_circuit_ir_nonlinear_dc,
)


@dataclass(frozen=True)
class _ICConstraint:
    p: int
    n: int
    value: float
    kind: str
    members: tuple[tuple[str, float], ...] = ()


def supports_circuit_ir_nonlinear_transient(ir: CircuitIR) -> tuple[bool, list[str]]:
    """Explicit capability gate for D/Q/J/M plus linear RLC transient MNA."""
    reasons: list[str] = []
    if not any(str(c.get("type", "")).upper() in ("D", "Q", "J", "M") for c in ir.components):
        return False, ["No supported nonlinear semiconductor device is present."]
    if not any(str(c.get("type", "")).upper() in ("C", "L") and
               float(c.get("parameters", {}).get("value", 0.0)) > 0.0 for c in ir.components):
        reasons.append("No positive capacitor or inductor energy-storage element is present.")
    transient_options = ir.analysis_configuration.get("transient", {}).get("unsupported_options", [])
    if transient_options:
        reasons.append("Unsupported .tran option(s): " + ", ".join(map(str, transient_options)) + ".")
    allowed = {"R", "C", "L", "V", "I", "E", "G", "F", "H", "D", "Q", "J", "M"}
    for component in ir.components:
        ref = str(component.get("reference", "?"))
        kind = str(component.get("type", "")).upper()
        if kind not in allowed:
            reasons.append(f"{ref}: {kind} is outside nonlinear transient MNA scope.")
        if kind in ("V", "I"):
            waveform = component.get("parameters", {}).get("waveform")
            if waveform and str(waveform.get("kind", "")).upper() not in ("SIN", "PULSE", "PWL"):
                reasons.append(f"{ref}: unsupported time-domain waveform {waveform.get('kind')!r}.")
    if not reasons:
        ok, nonlinear_reasons = supports_circuit_ir_nonlinear_dc(ir)
        if not ok:
            reasons.extend(nonlinear_reasons)
        try:
            _semiconductors_from_ir(ir)
        except Exception as exc:
            reasons.append(str(exc))
    return not reasons, reasons


def _source_function(component: dict[str, Any]) -> Callable[[float], float]:
    from .transient import _source_function as make_source
    return make_source(component)[0]


def _source_values(components: list[dict[str, Any]], functions: dict[str, Callable[[float], float]],
                   time_s: float) -> dict[str, float]:
    return {
        str(component["reference"]): float(functions[str(component["reference"]).casefold()](time_s))
        for component in components
    }


def _node_voltage(x: np.ndarray, node: int, node_index: dict[int, int]) -> float:
    index = node_index.get(int(node))
    return 0.0 if index is None else float(x[index])


class _PotentialConstraints:
    """Weighted disjoint-set for detecting inconsistent ideal initial voltages."""

    def __init__(self, nodes: set[int]):
        self.parent = {node: node for node in nodes}
        self.weight = {node: 0.0 for node in nodes}

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


def _build_initial_constraints(ir: CircuitIR, components: list[dict[str, Any]],
                                source_functions: dict[str, Callable[[float], float]]) -> tuple[list[_ICConstraint], dict[str, float]]:
    constraints: list[_ICConstraint] = []
    node_map = {str(node["name"]).casefold(): int(node["id"]) for node in ir.nodes}
    potential = _PotentialConstraints({0, *node_map.values()})
    source_values = _source_values(components, source_functions, 0.0)

    # Ideal voltage sources determine the t=0 voltage-difference constraints.
    for component in components:
        if str(component.get("type", "")).upper() != "V":
            continue
        p, n = (int(pin["node_id"]) for pin in component["pins"][:2])
        potential.add(p, n, source_values[str(component["reference"])],
                      f"{component['reference']} voltage-source equation")

    for node_name, raw_voltage in ir.initial_conditions.items():
        key = str(node_name).casefold()
        if key not in node_map:
            raise MNAError(f".ic references unknown node {node_name!r}.")
        voltage = float(raw_voltage)
        if not math.isfinite(voltage):
            raise MNAError(f".ic for node {node_name!r} must be finite.")
        node = node_map[key]
        if potential.add(node, 0, voltage, f".ic node {node_name!r}"):
            constraints.append(_ICConstraint(node, 0, voltage, "node_ic"))

    groups: dict[tuple[int, int], dict[str, Any]] = {}
    for component in ir.components:
        if str(component.get("type", "")).upper() != "C":
            continue
        params = component.get("parameters", {})
        if params.get("IC") is None:
            continue
        p, n = (int(pin["node_id"]) for pin in component["pins"][:2])
        ic = float(params["IC"])
        if not math.isfinite(ic):
            raise MNAError(f"{component['reference']}: capacitor IC must be finite.")
        if p == n:
            if abs(ic) > 1e-12:
                raise MNAError(f"{component['reference']}: capacitor connected across one node cannot have non-zero IC.")
            continue
        if p < n:
            key, target, sign = (p, n), ic, 1.0
        else:
            key, target, sign = (n, p), -ic, -1.0
        group = groups.setdefault(key, {"value": target, "members": [], "total_C": 0.0})
        if abs(group["value"] - target) > 1e-8:
            raise MNAError(f"{component['reference']}: capacitor IC conflicts with a parallel capacitor IC.")
        value = float(params["value"])
        group["members"].append((str(component["reference"]), sign * value))
        group["total_C"] += value

    for (p, n), group in groups.items():
        if potential.add(p, n, float(group["value"]), "capacitor IC"):
            members = tuple((ref, signed_c / float(group["total_C"]))
                            for ref, signed_c in group["members"])
            constraints.append(_ICConstraint(p, n, float(group["value"]), "capacitor_ic", members))
    return constraints, source_values


def _terminal_charge(ir: CircuitIR, model, devices, x: np.ndarray,
                     node_index: dict[int, int]) -> tuple[np.ndarray, csc_matrix, dict[str, Any]]:
    """Return B^T Q, B^T(dQ/dV)B and device evaluations at state x."""
    n = len(node_index)
    dimension = x.size
    q_global = np.zeros(dimension, dtype=float)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    evaluations: dict[str, Any] = {}

    for element in model.elements:
        if element.kind != "C":
            continue
        p, neg = element.nodes[:2]
        ip, inn = node_index.get(p), node_index.get(neg)
        vp = 0.0 if ip is None else float(x[ip])
        vn = 0.0 if inn is None else float(x[inn])
        q = element.value * (vp - vn)
        if ip is not None:
            q_global[ip] += q
            rows.append(ip); cols.append(ip); vals.append(element.value)
        if inn is not None:
            q_global[inn] -= q
            rows.append(inn); cols.append(inn); vals.append(element.value)
        if ip is not None and inn is not None:
            rows.extend((ip, inn)); cols.extend((inn, ip)); vals.extend((-element.value, -element.value))

    for device in devices:
        terminal_voltages = np.array([_node_voltage(x, node, node_index) for node in device.terminals])
        evaluation = device.model.evaluate_terminals(terminal_voltages)
        evaluations[device.reference] = evaluation
        charges = evaluation.terminal_charges_C
        qjac = evaluation.charge_jacobian_F
        if charges is None or qjac is None:
            continue
        for local_row, node in enumerate(device.terminals):
            global_row = node_index.get(node)
            if global_row is None:
                continue
            q_global[global_row] += float(charges[local_row])
            for local_col, col_node in enumerate(device.terminals):
                global_col = node_index.get(col_node)
                if global_col is None:
                    continue
                value = float(qjac[local_row, local_col])
                if value != 0.0:
                    rows.append(global_row); cols.append(global_col); vals.append(value)

    q_jac = coo_matrix((vals, (rows, cols)), shape=(dimension, dimension)).tocsc()
    q_jac.sum_duplicates()
    if not np.all(np.isfinite(q_global)) or not np.all(np.isfinite(q_jac.data)):
        raise NumericalSolveError("Nonlinear transient charge assembly produced non-finite values.")
    return q_global, q_jac, evaluations


def _initial_state(ir: CircuitIR, model, matrix: csc_matrix, rhs: np.ndarray,
                   node_index: dict[int, int], branch_index: dict[str, int],
                   devices, components: list[dict[str, Any]],
                   source_functions: dict[str, Callable[[float], float]]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Solve a nonlinear t=0 constrained MNA system and return state and Qdot history."""
    constraints, _ = _build_initial_constraints(ir, components, source_functions)
    node_map = {str(node["name"]).casefold(): int(node["id"]) for node in ir.nodes}
    dimension = matrix.shape[0]
    inductors = [e for e in model.elements if e.kind == "L"]

    # Build the augmented linear seed.  A declared inductor IC replaces its
    # voltage equation; an unspecified IC retains the DC operating-point equation.
    extra = len(constraints)
    aug_dim = dimension + extra
    seed = lil_matrix((aug_dim, aug_dim), dtype=float)
    seed[:dimension, :dimension] = matrix
    seed_rhs = np.zeros(aug_dim, dtype=float)
    seed_rhs[:dimension] = rhs
    # The linear seed omits semiconductor conductance by construction.  Add a
    # small seed-only GMIN to avoid false singularity at reverse-biased/off
    # junctions; the nonlinear initial-condition residual below is unregularized.
    seed_gmin_S = 1e-12
    for node_row in range(len(node_index)):
        seed[node_row, node_row] += seed_gmin_S
    for element in inductors:
        params = next(c.get("parameters", {}) for c in ir.components
                      if str(c.get("reference", "")).casefold() == element.reference.casefold())
        if params.get("IC") is not None:
            branch = branch_index[element.reference.casefold()]
            seed.rows[branch] = [branch]
            seed.data[branch] = [1.0]
            seed_rhs[branch] = float(params["IC"])

    for offset, constraint in enumerate(constraints):
        branch = dimension + offset
        ip, inn = node_index.get(constraint.p), node_index.get(constraint.n)
        if ip is not None:
            seed[ip, branch] += 1.0
            seed[branch, ip] += 1.0
        if inn is not None:
            seed[inn, branch] -= 1.0
            seed[branch, inn] -= 1.0
        seed_rhs[branch] = constraint.value

    initial_guess, *_ = solve_matrix(seed.tocsc(), seed_rhs)
    y = np.asarray(initial_guess, dtype=float)

    # Give unconstrained diode junctions a physically useful knee seed.  This
    # is only a Newton starting point: the equations solved below are the exact
    # unregularized device equations and all ideal source/IC constraints remain
    # authoritative.
    voltage_constrained_nodes = {
        int(pin["node_id"])
        for component in components if str(component.get("type", "")).upper() == "V"
        for pin in component.get("pins", []) if int(pin["node_id"]) != 0
    }
    for node_name in ir.initial_conditions:
        voltage_constrained_nodes.add(node_map[str(node_name).casefold()])
    for device in devices:
        if device.kind != "D":
            continue
        diode = device.model
        anode, cathode = device.terminals
        ip, inn = node_index.get(anode), node_index.get(cathode)
        vt_n = diode.ideality_factor * diode.thermal_voltage_V
        vcrit = vt_n * math.log(max(vt_n / (math.sqrt(2.0) * diode.effective_saturation_current_A), 1.0))
        if ip is not None and anode not in voltage_constrained_nodes:
            y[ip] = (0.0 if inn is None else y[inn]) + vcrit
        elif inn is not None and cathode not in voltage_constrained_nodes:
            y[inn] = (0.0 if ip is None else y[ip]) - vcrit

    def evaluate(candidate: np.ndarray):
        x = candidate[:dimension]
        residual, base_jacobian, _ = _residual_and_jacobian(matrix, rhs, x, node_index, devices)
        jac = lil_matrix((aug_dim, aug_dim), dtype=float)
        jac[:dimension, :dimension] = base_jacobian
        f = np.zeros(aug_dim, dtype=float)
        f[:dimension] = residual

        # Replace the inductor branch equation only when an explicit L IC exists.
        for element in inductors:
            params = next(c.get("parameters", {}) for c in ir.components
                          if str(c.get("reference", "")).casefold() == element.reference.casefold())
            if params.get("IC") is None:
                continue
            branch = branch_index[element.reference.casefold()]
            f[branch] = candidate[branch] - float(params["IC"])
            jac.rows[branch] = [branch]
            jac.data[branch] = [1.0]

        for offset, constraint in enumerate(constraints):
            branch = dimension + offset
            ip, inn = node_index.get(constraint.p), node_index.get(constraint.n)
            if ip is not None:
                f[ip] += candidate[branch]
                f[branch] += candidate[ip]
                jac[ip, branch] += 1.0
                jac[branch, ip] += 1.0
            if inn is not None:
                f[inn] -= candidate[branch]
                f[branch] -= candidate[inn]
                jac[inn, branch] -= 1.0
                jac[branch, inn] -= 1.0
            f[branch] -= constraint.value
        result = jac.tocsc()
        if not np.all(np.isfinite(f)) or not np.all(np.isfinite(result.data)):
            raise NumericalSolveError("Initial nonlinear transient system is non-finite.")
        return f, result

    y, initial_diagnostics = _newton_solve(
        evaluate, y, dimension, max_iterations=80,
        relative_tolerance=1e-8, current_absolute_tolerance=1e-11,
        voltage_absolute_tolerance=1e-9,
        label="initial-condition MNA",
    )
    x = y[:dimension]
    charge_current = np.zeros(dimension, dtype=float)
    for offset, constraint in enumerate(constraints):
        group_current = float(y[dimension + offset])
        if constraint.kind == "node_ic":
            # The node-voltage constraint is released after t=0.  Its reaction
            # current is therefore the initial total stored-charge current at
            # that node (including nonlinear semiconductor charge connected to
            # it), not a persistent independent source.
            ip, inn = node_index.get(constraint.p), node_index.get(constraint.n)
            if ip is not None:
                charge_current[ip] += group_current
            if inn is not None:
                charge_current[inn] -= group_current
        elif constraint.kind == "capacitor_ic":
            # Current is positive from constraint.p to constraint.n.  Distribute it
            # in proportion to capacitance among parallel initialized capacitors.
            for reference, fraction in constraint.members:
                cap_current = group_current * fraction
                component = next(c for c in ir.components
                                 if str(c.get("reference", "")).casefold() == reference.casefold())
                p, n = (int(pin["node_id"]) for pin in component["pins"][:2])
                ip, inn = node_index.get(p), node_index.get(n)
                if ip is not None:
                    charge_current[ip] += cap_current
                if inn is not None:
                    charge_current[inn] -= cap_current
    return x, charge_current, initial_diagnostics


def _newton_solve(evaluator: Callable[[np.ndarray], tuple[np.ndarray, csc_matrix]],
                  initial: np.ndarray, node_count: int, *,
                  max_iterations: int, relative_tolerance: float,
                  current_absolute_tolerance: float, voltage_absolute_tolerance: float,
                  label: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Reusable damped Newton solve for one transient/initial MNA system."""
    x = np.asarray(initial, dtype=float).copy()
    last_voltage_update = math.inf
    last_current_update = math.inf
    damping_values: list[float] = []
    linear_methods: list[str] = []
    final_residual = math.inf
    for iteration in range(1, max_iterations + 1):
        residual, jacobian = evaluator(x)
        node_norm = float(np.linalg.norm(residual[:node_count], ord=np.inf)) if node_count else 0.0
        branch_norm = float(np.linalg.norm(residual[node_count:], ord=np.inf)) if x.size > node_count else 0.0
        current_scale = max(current_absolute_tolerance,
                            float(np.linalg.norm(x[:node_count], ord=np.inf)) * 1e-3 if node_count else 0.0,
                            1e-9)
        voltage_scale = max(1.0, float(np.linalg.norm(x[:node_count], ord=np.inf)) if node_count else 0.0)
        residual_ok = (node_norm <= current_absolute_tolerance + relative_tolerance * current_scale
                       and branch_norm <= voltage_absolute_tolerance + relative_tolerance * voltage_scale)
        voltage_scale_for_update = max(
            1.0, float(np.linalg.norm(x[:node_count], ord=np.inf)) if node_count else 0.0
        )
        current_scale_for_update = max(
            1e-3, float(np.linalg.norm(x[node_count:], ord=np.inf)) if x.size > node_count else 0.0
        )
        update_ok = (
            last_voltage_update <= voltage_absolute_tolerance + relative_tolerance * voltage_scale_for_update
            and last_current_update <= current_absolute_tolerance + relative_tolerance * current_scale_for_update
        )
        final_residual = max(node_norm / current_scale, branch_norm / voltage_scale)
        if residual_ok and update_ok:
            return x, {"iterations": iteration - 1, "converged": True,
                       "residual_inf_scaled": final_residual,
                       "last_voltage_update_inf": last_voltage_update,
                       "last_current_update_inf": last_current_update,
                       "damping_factors": damping_values,
                       "linear_methods": linear_methods}

        delta, method, *_ = solve_matrix(jacobian, -residual, residual_tolerance=1e-8)
        linear_methods.append(str(method))
        delta = np.asarray(delta, dtype=float)
        if not np.all(np.isfinite(delta)):
            raise NumericalSolveError(f"{label}: Newton correction became non-finite.")

        def merit(candidate: np.ndarray) -> float:
            f, _ = evaluator(candidate)
            node = float(np.linalg.norm(f[:node_count], ord=np.inf)) if node_count else 0.0
            branch = float(np.linalg.norm(f[node_count:], ord=np.inf)) if candidate.size > node_count else 0.0
            return max(node / current_scale, branch / voltage_scale)

        accepted = False
        alpha = 1.0
        for _ in range(26):
            candidate = x + alpha * delta
            try:
                candidate_merit = merit(candidate)
            except (ValueError, FloatingPointError, OverflowError):
                candidate_merit = math.inf
            if math.isfinite(candidate_merit) and (
                candidate_merit <= final_residual * (1.0 + 1e-12) or candidate_merit <= 1e-13
            ):
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            raise NumericalSolveError(f"{label}: damped Newton line search failed at iteration {iteration}.")
        update = candidate - x
        last_voltage_update = float(np.linalg.norm(update[:node_count], ord=np.inf)) if node_count else 0.0
        last_current_update = float(np.linalg.norm(update[node_count:], ord=np.inf)) if update.size > node_count else 0.0
        damping_values.append(alpha)
        x = candidate
    residual, _ = evaluator(x)
    final_norm = float(np.linalg.norm(residual, ord=np.inf)) if residual.size else 0.0
    raise NumericalSolveError(
        f"{label}: Newton failed to converge in {max_iterations} iterations; "
        f"final residual infinity norm={final_norm:.3e}, "
        f"last voltage update={last_voltage_update:.3e} V, "
        f"last current update={last_current_update:.3e} A."
    )


def solve_nonlinear_transient(ir: CircuitIR, *, stop_time_s: float | None = None,
                              max_step_s: float | None = None, method: str = "backward_euler",
                              adaptive: bool = False, relative_tolerance: float = 1e-3,
                              absolute_voltage_tolerance: float = 1e-6,
                              absolute_current_tolerance: float = 1e-9,
                              absolute_charge_tolerance_C: float = 1e-15,
                              absolute_charge_current_tolerance_A: float = 1e-9,
                              absolute_flux_tolerance_Wb: float = 1e-12,
                              absolute_energy_tolerance_J: float = 1e-12,
                              gmin_S: float = 1e-12,
                              minimum_step_s: float | None = None, maximum_steps: int = 200_000,
                              maximum_newton_iterations: int = 60,
                              output_node: int | None = None,
                              output_pair: tuple[int, int] | None = None):
    """Integrate semiconductor D/Q/J/M plus linear RLC using BE or trapezoidal.

    Adaptive control uses step doubling: one full step is compared with two
    half steps.  The two-half-step result is committed when the scaled local
    error estimate is <= 1.  Rejected trials do not mutate charge/flux history.
    """
    from .transient import TransientSolution, _next_breakpoint, _time_roundoff_tolerance
    ok, reasons = supports_circuit_ir_nonlinear_transient(ir)
    if not ok:
        raise UnsupportedElementError("Nonlinear transient MNA does not support this circuit: " + "; ".join(reasons))
    normalized = method.strip().lower().replace("-", "_")
    if normalized not in ("backward_euler", "trapezoidal"):
        raise ValueError("method must be 'backward_euler' or 'trapezoidal'.")
    stop = float(stop_time_s if stop_time_s is not None else ir.analysis_configuration.get("transient", {}).get("stop_s", 0.0))
    step = float(max_step_s if max_step_s is not None else ir.analysis_configuration.get("transient", {}).get("step_s", stop / 400.0 if stop > 0 else 0.0))
    if not math.isfinite(stop) or stop <= 0.0 or not math.isfinite(step) or step <= 0.0:
        raise ValueError("Transient stop time and maximum step must be finite and positive.")
    if not (math.isfinite(relative_tolerance) and relative_tolerance > 0.0):
        raise ValueError("relative_tolerance must be finite and positive.")
    if not (math.isfinite(absolute_voltage_tolerance) and absolute_voltage_tolerance > 0.0):
        raise ValueError("absolute_voltage_tolerance must be finite and positive.")
    if not (math.isfinite(absolute_current_tolerance) and absolute_current_tolerance > 0.0):
        raise ValueError("absolute_current_tolerance must be finite and positive.")
    unit_tolerances = {
        "absolute_charge_tolerance_C": absolute_charge_tolerance_C,
        "absolute_charge_current_tolerance_A": absolute_charge_current_tolerance_A,
        "absolute_flux_tolerance_Wb": absolute_flux_tolerance_Wb,
        "absolute_energy_tolerance_J": absolute_energy_tolerance_J,
    }
    for tolerance_name, tolerance_value in unit_tolerances.items():
        if not (math.isfinite(tolerance_value) and tolerance_value > 0.0):
            raise ValueError(f"{tolerance_name} must be finite and positive.")
    if not (math.isfinite(gmin_S) and gmin_S >= 0.0):
        raise ValueError("gmin_S must be finite and non-negative.")
    minimum = float(minimum_step_s if minimum_step_s is not None else max(1e-15, step * 1e-6))
    if not math.isfinite(minimum) or minimum <= 0.0 or minimum > step:
        raise ValueError("minimum_step_s must be finite, positive and no larger than max_step_s.")
    if maximum_steps < 1 or maximum_newton_iterations < 1:
        raise ValueError("maximum_steps and maximum_newton_iterations must be positive.")

    components = [c for c in ir.components if str(c.get("type", "")).upper() in ("V", "I")]
    source_functions = {str(c["reference"]).casefold(): _source_function(c) for c in components}
    initial_sources = _source_values(components, source_functions, 0.0)
    model, devices, matrix, rhs0, node_index, branch_index, _ = _build_system(ir, initial_sources)
    node_count = len(node_index)
    if gmin_S > 0.0 and node_count:
        # Explicit, diagnosed SPICE-style shunt regularization.  This is part
        # of the transient equations (not an unreported solver-side clipping).
        regularization = np.r_[np.full(node_count, gmin_S),
                               np.zeros(matrix.shape[0] - node_count)]
        matrix = (matrix + diags(regularization, offsets=0,
                                 shape=matrix.shape, format="csc")).tocsc()
    dimension = matrix.shape[0]
    initial_state, initial_charge_current, initial_diagnostics = _initial_state(
        ir, model, matrix, rhs0, node_index, branch_index, devices, components, source_functions
    )
    initial_q, _, _ = _terminal_charge(ir, model, devices, initial_state, node_index)

    capacitors = [e for e in model.elements if e.kind == "C"]
    inductors = [e for e in model.elements if e.kind == "L"]
    time_values = [0.0]
    states = [initial_state.copy()]
    step_sizes: list[float] = []
    error_values: list[float] = []
    error_estimates: list[dict[str, float]] = []
    rejected_error_values: list[float] = []
    accepted_steps = 0
    rejected_steps = 0
    newton_iterations = int(initial_diagnostics["iterations"])
    linear_solves = len(initial_diagnostics.get("linear_methods", []))
    current_time = 0.0
    current_step = min(step, stop)
    x_prev = initial_state.copy()
    q_prev = initial_q.copy()
    iq_prev = initial_charge_current.copy()

    def node_difference(x: np.ndarray, element) -> float:
        return _node_voltage(x, element.nodes[0], node_index) - _node_voltage(x, element.nodes[1], node_index)

    def linear_storage_energy(x_state: np.ndarray) -> float:
        energy = 0.0
        for element in capacitors:
            voltage = node_difference(x_state, element)
            energy += 0.5 * element.value * voltage * voltage
        for element in inductors:
            current = float(x_state[branch_index[element.reference.casefold()]])
            energy += 0.5 * element.value * current * current
        return float(energy)

    def inductor_fluxes(x_state: np.ndarray) -> np.ndarray:
        return np.asarray([element.value * x_state[branch_index[element.reference.casefold()]]
                           for element in inductors], dtype=float)

    def one_step(x_old: np.ndarray, q_old: np.ndarray, iq_old: np.ndarray,
                 h: float, t_new: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        source_rhs = assemble_dc(model, source_overrides=_source_values(components, source_functions, t_new))[1]
        alpha = (1.0 / h) if normalized == "backward_euler" else (2.0 / h)
        method_history = np.zeros(dimension, dtype=float) if normalized == "backward_euler" else iq_old.copy()
        old_flux = {e.reference.casefold(): e.value * x_old[branch_index[e.reference.casefold()]] for e in inductors}
        old_voltage = {e.reference.casefold(): node_difference(x_old, e) for e in inductors}

        def evaluator(candidate: np.ndarray):
            static_residual, static_jacobian, _ = _residual_and_jacobian(
                matrix, source_rhs, candidate, node_index, devices
            )
            q_new, q_jacobian, _ = _terminal_charge(ir, model, devices, candidate, node_index)
            residual = static_residual + alpha * (q_new - q_old) - method_history
            jacobian = lil_matrix(static_jacobian + alpha * q_jacobian)
            for element in inductors:
                key = element.reference.casefold()
                branch = branch_index[key]
                flux_new = element.value * candidate[branch]
                residual[branch] -= alpha * (flux_new - old_flux[key])
                if normalized == "trapezoidal":
                    residual[branch] += old_voltage[key]
                jacobian[branch, branch] -= alpha * element.value
            return residual, jacobian.tocsc()

        candidate, diagnostics = _newton_solve(
            evaluator, x_old, node_count, max_iterations=maximum_newton_iterations,
            relative_tolerance=relative_tolerance,
            current_absolute_tolerance=absolute_current_tolerance,
            voltage_absolute_tolerance=absolute_voltage_tolerance,
            label=f"transient step at t={t_new:.9g} s",
        )
        q_new, _, _ = _terminal_charge(ir, model, devices, candidate, node_index)
        if normalized == "backward_euler":
            iq_new = (q_new - q_old) / h
        else:
            iq_new = 2.0 * (q_new - q_old) / h - iq_old
        return candidate, q_new, iq_new, diagnostics

    while current_time < stop - _time_roundoff_tolerance(stop, accepted_steps + rejected_steps + 1):
        if accepted_steps + rejected_steps >= maximum_steps:
            raise MNAError(f"Nonlinear transient exceeded maximum_steps={maximum_steps}.")
        target = min(current_time + current_step, stop)
        source_components = [c for c in ir.components if str(c.get("type", "")).upper() in ("V", "I")]
        target = _next_breakpoint(current_time, target, source_components)
        h = target - current_time
        if h < minimum * (1.0 - 1e-12):
            raise MNAError(f"Transient timestep {h:.3e} s fell below minimum_step_s={minimum:.3e} s.")

        if adaptive:
            try:
                full_x, full_q, full_iq, full_diag = one_step(x_prev, q_prev, iq_prev, h, target)
                half_h = 0.5 * h
                mid_t = current_time + half_h
                half1_x, half1_q, half1_iq, half1_diag = one_step(
                    x_prev, q_prev, iq_prev, half_h, mid_t
                )
                half2_x, half2_q, half2_iq, half2_diag = one_step(
                    half1_x, half1_q, half1_iq, half_h, target
                )
            except (NumericalSolveError, SingularMatrixError, FloatingPointError, ValueError) as exc:
                rejected_steps += 1
                if h <= minimum * (1.0 + 1e-12):
                    raise NumericalSolveError(
                        f"Adaptive nonlinear transient could not converge at minimum timestep "
                        f"{minimum:.3e} s; last trial failure: {exc}"
                    ) from exc
                current_step = max(minimum, 0.5 * h)
                continue
            order = 1 if normalized == "backward_euler" else 2
            divisor = 2.0 ** order - 1.0
            error = (half2_x - full_x) / divisor
            scale = np.empty(dimension, dtype=float)
            scale[:node_count] = absolute_voltage_tolerance + relative_tolerance * np.maximum(
                np.abs(half2_x[:node_count]), np.abs(x_prev[:node_count])
            )
            if dimension > node_count:
                scale[node_count:] = absolute_current_tolerance + relative_tolerance * np.maximum(
                    np.abs(half2_x[node_count:]), np.abs(x_prev[node_count:])
                )
            state_error_norm = float(np.max(np.abs(error) / scale)) if dimension else 0.0

            # Distinct error channels retain physical units and use matching
            # absolute tolerances. Charge includes explicit capacitors and the
            # shared semiconductor terminal-charge model.
            charge_error = (half2_q - full_q) / divisor
            charge_scale = absolute_charge_tolerance_C + relative_tolerance * np.maximum(
                np.abs(half2_q), np.abs(q_prev)
            )
            charge_error_norm = float(np.max(np.abs(charge_error) / charge_scale)) if charge_error.size else 0.0
            charge_current_error = (half2_iq - full_iq) / divisor
            charge_current_scale = absolute_charge_current_tolerance_A + relative_tolerance * np.maximum(
                np.abs(half2_iq), np.abs(iq_prev)
            )
            charge_current_error_norm = (float(np.max(np.abs(charge_current_error) / charge_current_scale))
                                         if charge_current_error.size else 0.0)

            half_flux = inductor_fluxes(half2_x)
            full_flux = inductor_fluxes(full_x)
            previous_flux = inductor_fluxes(x_prev)
            flux_error = (half_flux - full_flux) / divisor
            flux_scale = absolute_flux_tolerance_Wb + relative_tolerance * np.maximum(
                np.abs(half_flux), np.abs(previous_flux)
            )
            flux_error_norm = float(np.max(np.abs(flux_error) / flux_scale)) if flux_error.size else 0.0

            half_energy = linear_storage_energy(half2_x)
            full_energy = linear_storage_energy(full_x)
            previous_energy = linear_storage_energy(x_prev)
            energy_error_J = abs(half_energy - full_energy) / divisor
            energy_scale = absolute_energy_tolerance_J + relative_tolerance * max(
                abs(half_energy), abs(previous_energy)
            )
            energy_error_norm = energy_error_J / energy_scale

            channels = {
                "state": state_error_norm,
                "charge": charge_error_norm,
                "charge_current": charge_current_error_norm,
                "magnetic_flux": flux_error_norm,
                "linear_storage_energy": energy_error_norm,
            }
            error_norm = max(channels.values())
            all_diags = (full_diag, half1_diag, half2_diag)
            step_newton = sum(d["iterations"] for d in all_diags)
            if error_norm > 1.0:
                rejected_steps += 1
                rejected_error_values.append(error_norm)
                factor = max(0.2, min(0.8, 0.9 * error_norm ** (-1.0 / (order + 1))))
                current_step = max(minimum, h * factor)
                if h <= minimum * (1.0 + 1e-12):
                    raise NumericalSolveError(
                        f"Adaptive nonlinear transient error {error_norm:.3e} exceeds one at minimum timestep."
                    )
                continue
            # Commit both refined half steps, preserving their charge histories.
            current_time = mid_t
            x_prev, q_prev, iq_prev = half1_x, half1_q, half1_iq
            time_values.append(current_time)
            states.append(x_prev.copy())
            step_sizes.append(half_h)
            error_values.append(error_norm)
            error_estimates.append({"time_s": float(current_time), **{f"{key}_normalized": float(value) for key, value in channels.items()}})
            accepted_steps += 1

            current_time = target
            x_prev, q_prev, iq_prev = half2_x, half2_q, half2_iq
            time_values.append(current_time)
            states.append(x_prev.copy())
            step_sizes.append(half_h)
            error_values.append(error_norm)
            error_estimates.append({"time_s": float(current_time), **{f"{key}_normalized": float(value) for key, value in channels.items()}})
            accepted_steps += 1
            newton_iterations += step_newton
            factor = 2.0 if error_norm <= 1e-16 else max(
                0.5, min(2.0, 0.9 * error_norm ** (-1.0 / (order + 1)))
            )
            current_step = min(step, h * factor)
            linear_solves += sum(max(0, d["iterations"]) for d in all_diags)
        else:
            candidate, q_new, iq_new, diagnostics = one_step(x_prev, q_prev, iq_prev, h, target)
            current_time = target
            x_prev, q_prev, iq_prev = candidate, q_new, iq_new
            time_values.append(current_time)
            states.append(x_prev.copy())
            step_sizes.append(h)
            error_values.append(0.0)
            accepted_steps += 1
            newton_iterations += diagnostics["iterations"]
            current_step = step
            linear_solves += diagnostics["iterations"]

    final_time_roundoff_s = max(0.0, stop - current_time)
    final_time_snapped = final_time_roundoff_s <= _time_roundoff_tolerance(
        stop, accepted_steps + rejected_steps + 1
    )
    if final_time_snapped:
        # Do not manufacture a sub-minimum final integration step solely to
        # remove accumulated floating-point drift from the time grid.
        time_values[-1] = stop

    state_array = np.asarray(states, dtype=float)
    node_names = {int(node["id"]): str(node["name"]) for node in ir.nodes}
    node_voltages = {node_names[node]: state_array[:, index].tolist()
                     for node, index in node_index.items()}
    branch_currents = {
        element.reference: state_array[:, branch_index[element.reference.casefold()]].tolist()
        for element in model.elements if element.reference.casefold() in branch_index
    }
    terminal_names = {"D": ("anode", "cathode"), "Q": ("collector", "base", "emitter"),
                      "J": ("drain", "gate", "source"),
                      "M": ("drain", "gate", "source", "bulk")}
    device_terminal_currents: dict[str, dict[str, list[float]]] = {}
    device_terminal_charges: dict[str, dict[str, list[float]]] = {}
    linear_capacitor_charges: dict[str, list[float]] = {}
    inductor_flux_history: dict[str, list[float]] = {}
    for element in model.elements:
        if element.kind == "C":
            linear_capacitor_charges[element.reference] = [
                float(element.value * node_difference(state, element)) for state in state_array
            ]
        elif element.kind == "L":
            index = branch_index[element.reference.casefold()]
            inductor_flux_history[element.reference] = [
                float(element.value * state[index]) for state in state_array
            ]
    for device in devices:
        terminal_series = {name: [] for name in terminal_names[device.kind]}
        charge_series = {name: [] for name in terminal_names[device.kind]}
        for state in state_array:
            terminal_voltages = np.asarray([_node_voltage(state, node, node_index) for node in device.terminals])
            evaluation = device.model.evaluate_terminals(terminal_voltages)
            for terminal_index, terminal_name in enumerate(terminal_names[device.kind]):
                terminal_series[terminal_name].append(float(evaluation.terminal_currents_A[terminal_index]))
                if evaluation.terminal_charges_C is not None:
                    charge_series[terminal_name].append(float(evaluation.terminal_charges_C[terminal_index]))
                else:
                    charge_series[terminal_name].append(0.0)
        device_terminal_currents[device.reference] = terminal_series
        device_terminal_charges[device.reference] = charge_series
    stored_energy = [linear_storage_energy(state) for state in state_array]
    if output_pair is not None:
        positive = node_names.get(int(output_pair[0]))
        negative = node_names.get(int(output_pair[1]))
        if positive is None or negative is None:
            raise MNAError("Requested differential output node is not present in the circuit.")
        output = np.asarray(node_voltages.get(positive, np.zeros(len(time_values)))) - np.asarray(
            node_voltages.get(negative, np.zeros(len(time_values)))
        )
    elif output_node is not None:
        name = node_names.get(int(output_node))
        output = np.asarray(node_voltages.get(name, np.zeros(len(time_values)))) if name else None
    else:
        output = None

    diagnostics = {
        "backend": "circhecker.solver.nonlinear_transient",
        "method": normalized,
        "adaptive": bool(adaptive),
        "adaptive_controller": "step_doubling_full_vs_two_half_steps" if adaptive else "fixed_step",
        "accepted_steps": int(accepted_steps),
        "rejected_steps": int(rejected_steps),
        "initial_newton_iterations": int(initial_diagnostics["iterations"]),
        "initial_residual_inf_scaled": float(initial_diagnostics["residual_inf_scaled"]),
        "newton_iterations": int(newton_iterations),
        "newton_linear_solve_count": int(linear_solves),
        "minimum_accepted_step_s": float(min(step_sizes)) if step_sizes else None,
        "maximum_accepted_step_s": float(max(step_sizes)) if step_sizes else None,
        "minimum_configured_step_s": minimum,
        "maximum_configured_step_s": step,
        "final_time_snapped_to_stop": bool(final_time_snapped),
        "final_time_roundoff_adjustment_s": float(final_time_roundoff_s),
        "relative_tolerance": float(relative_tolerance),
        "absolute_voltage_tolerance_V": float(absolute_voltage_tolerance),
        "absolute_current_tolerance_A": float(absolute_current_tolerance),
        "gmin_S": float(gmin_S),
        "maximum_scaled_local_error": float(max(error_values)) if error_values else 0.0,
        "maximum_rejected_scaled_local_error": float(max(rejected_error_values)) if rejected_error_values else 0.0,
        "maximum_scaled_state_error": float(max((item["state_normalized"] for item in error_estimates), default=0.0)),
        "maximum_scaled_charge_error": float(max((item["charge_normalized"] for item in error_estimates), default=0.0)),
        "maximum_scaled_charge_current_error": float(max((item["charge_current_normalized"] for item in error_estimates), default=0.0)),
        "maximum_scaled_magnetic_flux_error": float(max((item["magnetic_flux_normalized"] for item in error_estimates), default=0.0)),
        "maximum_scaled_linear_storage_energy_error": float(max((item["linear_storage_energy_normalized"] for item in error_estimates), default=0.0)),
        "absolute_charge_tolerance_C": float(absolute_charge_tolerance_C),
        "absolute_charge_current_tolerance_A": float(absolute_charge_current_tolerance_A),
        "absolute_flux_tolerance_Wb": float(absolute_flux_tolerance_Wb),
        "absolute_energy_tolerance_J": float(absolute_energy_tolerance_J),
        "energy_scope": "Explicit linear capacitors and inductors only; excludes semiconductor charge energy, dissipation, and source work.",
        "error_estimate_history": error_estimates,
    }
    data = {
        "time_s": time_values,
        "node_voltages_V": node_voltages,
        "branch_currents_A": branch_currents,
        # Keep the original result key for compatibility, but name the physical
        # quantity explicitly: these are conduction currents only; dQ/dt is a
        # separate dynamic contribution represented in the DAE residual.
        "device_terminal_currents_A": device_terminal_currents,
        "device_terminal_conduction_currents_A": device_terminal_currents,
        "device_terminal_charges_C": device_terminal_charges,
        "linear_capacitor_charges_C": linear_capacitor_charges,
        "inductor_flux_Wb": inductor_flux_history,
        "device_terminal_current_semantics": "Conduction current positive into each named terminal; displacement current dQ/dt is not included in this trace.",
        "stored_linear_energy_J": stored_energy,
        "integration_method": normalized,
        "solver_diagnostics": diagnostics,
        "output_V": output.tolist() if output is not None else None,
        "output_pair": list(output_pair) if output_pair is not None else None,
    }
    return TransientSolution(data)
