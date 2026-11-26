"""Unified sparse nonlinear DC MNA for terminal-based semiconductor models.

Linear branches continue to use :mod:`circhecker.solver.mna` for their stamps.
Each nonlinear semiconductor is evaluated only through the shared terminal
constitutive interface in ``devices.py``; all device residuals and Jacobians
are accumulated into one global Newton system.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.sparse import csc_matrix, coo_matrix, diags

from ..circuit_ir import CircuitIR
from .devices import (
    DeviceEvaluation, GummelPoonBJT, Level1JFET, Level1MOSFET,
    ShockleyDiode, NonlinearDevice,
)
from .exceptions import MNAError, NumericalSolveError, UnsupportedElementError
from .matrix import solve_matrix
from .mna import BRANCH_KINDS, assemble_dc, model_from_circuit_ir


@dataclass(frozen=True)
class NonlinearDeviceInstance:
    reference: str
    kind: str
    terminals: tuple[int, ...]
    model: NonlinearDevice


# Explicit accepted parameter contracts prevent model cards from being
# accidentally accepted while silently ignoring physically significant keys.
_DEVICE_MODEL_KEYS = {
    "D": {"IS", "N", "CJO", "VJ", "M", "FC", "TT", "XTI", "EG", "BV", "RBV"},
    "Q": {"IS", "BF", "BR", "VAF", "VAR", "CJE", "CJC", "VJE", "VJC",
          "MJE", "MJC", "FC", "TF", "TR", "XTI", "EG"},
    "J": {"VTO", "BETA", "LAMBDA", "IS", "CGS", "CGD"},
    "M": {"VTO", "KP", "LAMBDA", "GAMMA", "PHI", "IS", "W", "L",
          "CGSO", "CGDO", "CGBO", "CGS", "CGD", "CGB", "VJ", "M", "FC"},
}
_INSTANCE_KEYS = {
    "D": {"IS", "N"},
    "Q": {"IS", "BF", "BR", "VAF", "VAR", "CJE", "CJC", "VJE", "VJC",
          "MJE", "MJC", "FC", "TF", "TR"},
    "J": {"VTO", "BETA", "LAMBDA", "IS", "CGS", "CGD"},
    "M": {"W", "L"},
}


def _numeric_parameters(reference: str, parameters: Mapping[str, Any], allowed: set[str], family: str) -> dict[str, float]:
    extra = set(parameters) - allowed
    if extra:
        raise UnsupportedElementError(
            f"{reference}: {family} nonlinear MNA subset does not support parameter(s): {', '.join(sorted(extra))}"
        )
    normalized: dict[str, float] = {}
    for key, value in parameters.items():
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise UnsupportedElementError(f"{reference}: model parameter {key} must be numeric.") from exc
        if not math.isfinite(number):
            raise UnsupportedElementError(f"{reference}: model parameter {key} must be finite.")
        normalized[str(key).upper()] = number
    return normalized


def _temperature(ir: CircuitIR) -> tuple[float, float]:
    options = {str(k).upper(): v for k, v in ir.simulation_options.items()}
    temp_c = float(options.get("TEMP", 27.0))
    tnom_c = float(options.get("TNOM", 27.0))
    if not math.isfinite(temp_c) or not math.isfinite(tnom_c) or temp_c <= -273.15 or tnom_c <= -273.15:
        raise MNAError("TEMP and TNOM must be finite temperatures above absolute zero in degrees Celsius.")
    return temp_c + 273.15, tnom_c + 273.15


def _semiconductors_from_ir(ir: CircuitIR) -> list[NonlinearDeviceInstance]:
    temp_k, tnom_k = _temperature(ir)
    devices: list[NonlinearDeviceInstance] = []
    for component in ir.components:
        kind = str(component.get("type", "")).upper()
        if kind not in _DEVICE_MODEL_KEYS:
            continue
        reference = str(component.get("reference", f"{kind}?"))
        pins = component.get("pins", [])
        expected = {"D": 2, "Q": 3, "J": 3, "M": 4}[kind]
        if len(pins) != expected:
            raise MNAError(f"{reference}: {kind} requires {expected} terminals in parser-defined order.")
        terminals = tuple(int(pin["node_id"]) for pin in pins)
        raw_params = dict(component.get("parameters", {}) or {})
        model_params = dict(raw_params.pop("model_parameters", {}) or {})
        model_type = str(raw_params.pop("model_type", "") or "").upper()
        model_ref = raw_params.pop("model_ref", None)
        instance_params_raw = raw_params.pop("instance_parameters", {}) or {}
        raw_params.pop("raw_value", None)
        raw_params.pop("unit", None)
        # Parser-produced M instances carry only model defaults at top level;
        # W/L are preserved in instance_parameters.
        merged = dict(model_params)
        merged.update(raw_params)
        instance_params = dict(instance_params_raw)
        if kind == "D":
            merged.update(instance_params)
            allowed_model = _numeric_parameters(reference, merged, _DEVICE_MODEL_KEYS[kind], "diode")
            allowed_instance = {k: allowed_model[k] for k in _INSTANCE_KEYS[kind] if k in allowed_model}
            model = ShockleyDiode(
                saturation_current_A=allowed_instance.get("IS", 1e-14),
                ideality_factor=allowed_instance.get("N", 1.0),
                temperature_K=temp_k,
                nominal_temperature_K=tnom_k,
                XTI=allowed_model.get("XTI", 3.0), EG_eV=allowed_model.get("EG", 1.11),
                CJO_F=allowed_model.get("CJO", 0.0), VJ_V=allowed_model.get("VJ", 0.7),
                M=allowed_model.get("M", 0.5), FC=allowed_model.get("FC", 0.5),
                TT_s=allowed_model.get("TT", 0.0), BV_V=allowed_model.get("BV"),
                RBV_Ohm=allowed_model.get("RBV", 5.0),
            )
        elif kind == "Q":
            merged.update(instance_params)
            params = _numeric_parameters(reference, merged, _DEVICE_MODEL_KEYS[kind], "BJT")
            polarity = model_type or "NPN"
            if polarity not in ("NPN", "PNP"):
                raise UnsupportedElementError(f"{reference}: BJT model type must be NPN or PNP, found {polarity!r}.")
            model = GummelPoonBJT(
                polarity=polarity, IS_A=params.get("IS", 1e-15), BF=params.get("BF", 100.0),
                BR=params.get("BR", 1.0), VAF_V=params.get("VAF"), VAR_V=params.get("VAR"),
                temperature_K=temp_k, nominal_temperature_K=tnom_k,
                XTI=params.get("XTI", 3.0), EG_eV=params.get("EG", 1.11),
                CJE_F=params.get("CJE", 0.0), CJC_F=params.get("CJC", 0.0),
                VJE_V=params.get("VJE", 0.75), VJC_V=params.get("VJC", 0.75),
                MJE=params.get("MJE", 0.33), MJC=params.get("MJC", 0.33),
                FC=params.get("FC", 0.5), TF_s=params.get("TF", 0.0), TR_s=params.get("TR", 0.0),
            )
        elif kind == "J":
            merged.update(instance_params)
            params = _numeric_parameters(reference, merged, _DEVICE_MODEL_KEYS[kind], "JFET")
            polarity = model_type or "NJF"
            if polarity not in ("NJF", "PJF"):
                raise UnsupportedElementError(f"{reference}: JFET model type must be NJF or PJF, found {polarity!r}.")
            has_vto_assignment = bool(re.search(r"\bVTO\s*=", str(component.get("original_expression", "")), re.I))
            if polarity == "PJF" and "VTO" not in model_params and not has_vto_assignment:
                params["VTO"] = 2.0
            model = Level1JFET(
                polarity=polarity, VTO_V=params.get("VTO", -2.0),
                BETA_A_V2=params.get("BETA", 1e-3), LAMBDA_per_V=params.get("LAMBDA", 0.0),
                IS_A=params.get("IS", 1e-14), temperature_K=temp_k,
                CGS_F=params.get("CGS", 0.0), CGD_F=params.get("CGD", 0.0),
            )
        else:
            # Keep geometry separate from model-card parameters.  Parser
            # defaults are merged only where the model card omitted a value.
            params = _numeric_parameters(reference, merged, _DEVICE_MODEL_KEYS[kind], "MOSFET")
            instance = _numeric_parameters(reference, instance_params, _INSTANCE_KEYS[kind], "MOSFET instance")
            polarity = model_type or "NMOS"
            if polarity not in ("NMOS", "PMOS"):
                raise UnsupportedElementError(f"{reference}: MOSFET model type must be NMOS or PMOS, found {polarity!r}.")
            has_vto_assignment = bool(re.search(r"\bVTO\s*=", str(component.get("original_expression", "")), re.I))
            vto_default = -1.0 if polarity == "PMOS" else 1.0
            if polarity == "PMOS" and "VTO" not in model_params and "VTO" not in instance_params and not has_vto_assignment:
                params["VTO"] = vto_default
            model = Level1MOSFET(
                polarity=polarity, VTO_V=params.get("VTO", vto_default),
                KP_A_V2=params.get("KP", 1e-3), LAMBDA_per_V=params.get("LAMBDA", 0.0),
                GAMMA_sqrtV=params.get("GAMMA", 0.0), PHI_V=params.get("PHI", 0.6),
                IS_A=params.get("IS", 1e-14), temperature_K=temp_k,
                W_m=instance.get("W", params.get("W", 1e-6)),
                L_m=instance.get("L", params.get("L", 1e-6)),
                CGSO_F_per_m=params.get("CGSO", 0.0), CGDO_F_per_m=params.get("CGDO", 0.0),
                CGBO_F_per_m=params.get("CGBO", 0.0), CGS_F=params.get("CGS", 0.0),
                CGD_F=params.get("CGD", 0.0), CGB_F=params.get("CGB", 0.0),
                VJ_V=params.get("VJ", 0.7), M=params.get("M", 0.5), FC=params.get("FC", 0.5),
            )
        devices.append(NonlinearDeviceInstance(reference, kind, terminals, model))
    return devices


# Backward-compatible helper retained for existing tests and callers that
# inspected the original diode instance records.
@dataclass(frozen=True)
class DiodeInstance:
    reference: str
    anode: int
    cathode: int
    model: ShockleyDiode


def _diodes_from_ir(ir: CircuitIR) -> list[DiodeInstance]:
    devices = _semiconductors_from_ir(ir)
    result = []
    for device in devices:
        if device.kind == "D":
            result.append(DiodeInstance(device.reference, device.terminals[0], device.terminals[1], device.model))
    return result


@dataclass
class NonlinearMNASolution:
    node_voltages_V: dict[int, float]
    branch_currents_A: dict[str, float]
    branch_results: dict[str, dict[str, float | str]]
    diagnostics: dict[str, Any]
    verification: dict[str, Any]
    device_results: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_voltages_V": self.node_voltages_V,
            "branch_currents_A": self.branch_currents_A,
            "branch_results": self.branch_results,
            "solver_diagnostics": self.diagnostics,
            "verification": self.verification,
            "device_results": self.device_results,
        }


def supports_circuit_ir_nonlinear_dc(ir: CircuitIR) -> tuple[bool, list[str]]:
    """Capability gate for linear MNA elements plus documented D/Q/J/M models."""
    reasons: list[str] = []
    if not ir.components:
        return False, ["CircuitIR contains no components to simulate"]
    if not any(str(c.get("type", "")).upper() in _DEVICE_MODEL_KEYS for c in ir.components):
        reasons.append("No supported nonlinear semiconductor device is present; use the linear DC MNA backend.")
    try:
        _semiconductors_from_ir(ir)
        linear_ir = deepcopy(ir)
        linear_ir.components = [c for c in linear_ir.components if str(c.get("type", "")).upper() not in _DEVICE_MODEL_KEYS]
        if not linear_ir.components:
            reasons.append("A linear source/reference network is required alongside nonlinear devices.")
        else:
            model_from_circuit_ir(linear_ir)
    except Exception as exc:
        reasons.append(str(exc))
    return not reasons, reasons


def _build_system(ir: CircuitIR, source_overrides: Mapping[str, float] | None):
    devices = _semiconductors_from_ir(ir)
    linear_ir = deepcopy(ir)
    linear_ir.components = [c for c in linear_ir.components if str(c.get("type", "")).upper() not in _DEVICE_MODEL_KEYS]
    model = model_from_circuit_ir(linear_ir)
    matrix, rhs, node_index, branch_index, ignored = assemble_dc(model, source_overrides)
    return model, devices, matrix, rhs, node_index, branch_index, ignored


def _residual_and_jacobian(matrix: csc_matrix, rhs: np.ndarray, x: np.ndarray,
                           node_index: dict[int, int], devices: list[NonlinearDeviceInstance]):
    residual = np.asarray(matrix @ x - rhs, dtype=float).reshape(-1)
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    evaluations: dict[str, DeviceEvaluation] = {}
    for device in devices:
        terminal_voltages = np.array([
            0.0 if node == 0 else float(x[node_index[node]]) for node in device.terminals
        ], dtype=float)
        evaluation = device.model.evaluate_terminals(terminal_voltages)
        evaluations[device.reference] = evaluation
        if not np.all(np.isfinite(evaluation.terminal_currents_A)):
            raise NumericalSolveError(f"{device.reference}: non-finite terminal current evaluation.")
        # Stamp B^T i and B^T (di/dv) B.  Repeated nodes, if present, are
        # intentionally accumulated through COO duplicate summation.
        for local_row, node in enumerate(device.terminals):
            global_row = node_index.get(node)
            if global_row is None:
                continue
            residual[global_row] += float(evaluation.terminal_currents_A[local_row])
            for local_col, col_node in enumerate(device.terminals):
                global_col = node_index.get(col_node)
                if global_col is None:
                    continue
                derivative = float(evaluation.current_jacobian_S[local_row, local_col])
                if derivative != 0.0:
                    rows.append(global_row)
                    cols.append(global_col)
                    values.append(derivative)
    if rows:
        nonlinear_jacobian = coo_matrix((values, (rows, cols)), shape=matrix.shape).tocsc()
        nonlinear_jacobian.sum_duplicates()
        jacobian = (matrix + nonlinear_jacobian).tocsc()
    else:
        jacobian = matrix
    if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(jacobian.data)):
        raise NumericalSolveError("Assembled nonlinear residual or Jacobian contains non-finite values.")
    return residual, jacobian, evaluations


def solve_nonlinear_dc(ir: CircuitIR, *, source_overrides: Mapping[str, float] | None = None,
                       initial_guess: Mapping[int, float] | None = None,
                       max_iterations: int = 400, relative_tolerance: float = 1e-7,
                       voltage_absolute_tolerance: float = 1e-9,
                       current_absolute_tolerance: float = 1e-12,
                       update_voltage_tolerance: float = 1e-9,
                       update_current_tolerance: float = 1e-12,
                       line_search: bool = True,
                       gmin_steps: tuple[float, ...] | list[float] = (),
                       source_steps: tuple[float, ...] | list[float] = ()) -> NonlinearMNASolution:
    """Solve linear branches and all supported D/Q/J/M devices in one Newton system.

    Newton's equation is ``J(x) dx = -F(x)`` with an optional backtracking
    damping factor ``lambda``.  Convergence requires both the mixed-unit KCL /
    branch residual tests and the voltage/current update tests to pass.  Every
    continuation run ends at GMIN=0 S and source scale=1.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least one.")
    tolerances = (relative_tolerance, voltage_absolute_tolerance, current_absolute_tolerance,
                  update_voltage_tolerance, update_current_tolerance)
    if not all(math.isfinite(float(t)) and float(t) > 0.0 for t in tolerances):
        raise ValueError("All nonlinear solver tolerances must be finite and positive.")
    model, devices, matrix, rhs, node_index, branch_index, ignored = _build_system(ir, source_overrides)
    dimension = matrix.shape[0]
    if dimension == 0:
        raise MNAError("Nonlinear MNA system has no unknowns.")
    n_nodes = len(node_index)

    normalized_gmin = [float(value) for value in gmin_steps]
    if any(not math.isfinite(value) or value < 0 for value in normalized_gmin):
        raise ValueError("GMIN stages must be finite and non-negative conductances in siemens.")
    if normalized_gmin and normalized_gmin[-1] != 0.0:
        normalized_gmin.append(0.0)
    if not normalized_gmin:
        normalized_gmin = [0.0]
    normalized_sources = [float(value) for value in source_steps]
    if any(not math.isfinite(value) or value <= 0 or value > 1 for value in normalized_sources):
        raise ValueError("Source stepping values must be finite and in the interval (0, 1].")
    if any(b <= a for a, b in zip(normalized_sources, normalized_sources[1:])):
        raise ValueError("Source stepping values must be strictly increasing.")
    if normalized_sources and normalized_sources[-1] < 1.0:
        normalized_sources.append(1.0)
    if not normalized_sources:
        normalized_sources = [1.0]
    stage_count = max(len(normalized_gmin), len(normalized_sources))
    stages = [(normalized_gmin[min(i, len(normalized_gmin) - 1)],
               normalized_sources[min(i, len(normalized_sources) - 1)]) for i in range(stage_count)]

    x: np.ndarray | None = None
    stage_records: list[dict[str, Any]] = []
    total_iterations = 0
    final_method = "not_run"
    final_condition = None
    final_condition_method = None
    final_warnings: list[str] = []
    final_jacobian = matrix
    final_current_scale = 1.0
    final_voltage_scale = 1.0
    last_delta_v = math.inf
    last_delta_i = math.inf
    final_residual = np.full(dimension, math.inf)
    final_evaluations: dict[str, DeviceEvaluation] = {}
    final_stage_matrix = matrix
    final_stage_rhs = rhs

    voltage_constrained_nodes = {
        int(pin["node_id"])
        for component in ir.components if str(component.get("type", "")).upper() == "V"
        for pin in component.get("pins", []) if int(pin["node_id"]) != 0
    }

    for stage_number, (gmin, source_scale) in enumerate(stages, start=1):
        regularization = diags(
            np.r_[np.full(n_nodes, gmin, dtype=float), np.zeros(dimension - n_nodes, dtype=float)],
            format="csc",
        )
        stage_matrix = (matrix + regularization).tocsc()
        stage_rhs = np.asarray(rhs * source_scale, dtype=float)
        final_stage_matrix, final_stage_rhs = stage_matrix, stage_rhs
        if x is None:
            # Linear seed can itself be singular for an unconstrained node; the
            # same explicit singular-system diagnostics used by Newton apply.
            linear_seed = np.asarray(solve_matrix(stage_matrix, stage_rhs)[0], dtype=float)
            x = linear_seed.copy()
            if initial_guess is not None:
                for node_id, value in initial_guess.items():
                    if int(node_id) in node_index:
                        if not math.isfinite(float(value)):
                            raise ValueError("Initial node-voltage guesses must be finite.")
                        x[node_index[int(node_id)]] = float(value)
            else:
                # A diode knee seed avoids starting at a strongly exponential
                # forward point while preserving ideal source constraints.
                for device in devices:
                    if device.kind != "D":
                        continue
                    diode: ShockleyDiode = device.model  # type: ignore[assignment]
                    anode, cathode = device.terminals
                    ip, inn = node_index.get(anode), node_index.get(cathode)
                    vt_n = diode.ideality_factor * diode.thermal_voltage_V
                    isat = diode.effective_saturation_current_A
                    vcrit = vt_n * math.log(max(vt_n / (math.sqrt(2.0) * isat), 1.0))
                    vp = 0.0 if ip is None else float(x[ip])
                    vn = 0.0 if inn is None else float(x[inn])
                    if vp - vn > vcrit:
                        if ip is not None and anode not in voltage_constrained_nodes:
                            x[ip] = vn + vcrit
                        elif inn is not None and cathode not in voltage_constrained_nodes:
                            x[inn] = vp - vcrit

        last_delta_v = math.inf
        last_delta_i = math.inf
        stage_converged = False
        stage_iteration = 0

        def merit_at(candidate: np.ndarray):
            f, jac, evals = _residual_and_jacobian(stage_matrix, stage_rhs, candidate, node_index, devices)
            node_scale = max(current_absolute_tolerance,
                             float(np.linalg.norm(stage_rhs[:n_nodes], ord=np.inf)) if n_nodes else 0.0,
                             max((float(np.max(np.abs(e.terminal_currents_A))) for e in evals.values()), default=0.0),
                             1e-9)
            branch_scale = max(1.0,
                               float(np.linalg.norm(stage_rhs[n_nodes:], ord=np.inf)) if dimension > n_nodes else 0.0,
                               float(np.linalg.norm(candidate[:n_nodes], ord=np.inf)) if n_nodes else 0.0)
            node_error = float(np.linalg.norm(f[:n_nodes], ord=np.inf)) if n_nodes else 0.0
            branch_error = float(np.linalg.norm(f[n_nodes:], ord=np.inf)) if dimension > n_nodes else 0.0
            return max(node_error / node_scale, branch_error / branch_scale), f, jac, evals, node_scale, branch_scale

        for stage_iteration in range(1, max_iterations + 1):
            merit, residual, jacobian, evaluations, current_scale, voltage_scale = merit_at(x)
            node_residual = float(np.linalg.norm(residual[:n_nodes], ord=np.inf)) if n_nodes else 0.0
            branch_residual = float(np.linalg.norm(residual[n_nodes:], ord=np.inf)) if dimension > n_nodes else 0.0
            residual_ok = (
                node_residual <= current_absolute_tolerance + relative_tolerance * current_scale
                and branch_residual <= voltage_absolute_tolerance + relative_tolerance * voltage_scale
            )
            update_ok = (
                last_delta_v <= update_voltage_tolerance + relative_tolerance * max(
                    1.0, float(np.linalg.norm(x[:n_nodes], ord=np.inf)) if n_nodes else 0.0
                )
                and last_delta_i <= update_current_tolerance + relative_tolerance * max(
                    1e-3, float(np.linalg.norm(x[n_nodes:], ord=np.inf)) if dimension > n_nodes else 0.0
                )
            )
            if residual_ok and update_ok:
                stage_converged = True
                final_jacobian = jacobian
                final_current_scale, final_voltage_scale = current_scale, voltage_scale
                final_residual, final_evaluations = residual, evaluations
                break

            delta, final_method, final_condition, final_condition_method, _, _, solve_warnings = solve_matrix(
                jacobian, -residual, residual_tolerance=1e-8
            )
            final_warnings.extend(w for w in solve_warnings if w not in final_warnings)
            alpha = 1.0
            candidate = x + delta
            if line_search:
                accepted = False
                for _ in range(24):
                    try:
                        candidate_merit = merit_at(candidate)[0]
                    except (ValueError, FloatingPointError, OverflowError):
                        candidate_merit = math.inf
                    if candidate_merit <= merit * (1.0 + 1e-12) or candidate_merit <= 1e-14:
                        accepted = True
                        break
                    alpha *= 0.5
                    candidate = x + alpha * delta
                if not accepted:
                    raise NumericalSolveError(
                        f"Nonlinear Newton line search failed at continuation stage {stage_number} "
                        f"(GMIN={gmin:.3e} S, source scale={source_scale:.3g}, iteration={stage_iteration}, "
                        f"scaled residual={merit:.3e})."
                    )
            update = candidate - x
            if not np.all(np.isfinite(update)) or not np.all(np.isfinite(candidate)):
                raise NumericalSolveError(
                    f"Nonlinear Newton iterate became non-finite at continuation stage {stage_number}, "
                    f"iteration {stage_iteration}."
                )
            last_delta_v = float(np.linalg.norm(update[:n_nodes], ord=np.inf)) if n_nodes else 0.0
            last_delta_i = float(np.linalg.norm(update[n_nodes:], ord=np.inf)) if dimension > n_nodes else 0.0
            x = candidate

        total_iterations += stage_iteration
        final_merit, final_residual, final_jacobian, final_evaluations, final_current_scale, final_voltage_scale = merit_at(x)
        final_node_residual = float(np.linalg.norm(final_residual[:n_nodes], ord=np.inf)) if n_nodes else 0.0
        final_branch_residual = float(np.linalg.norm(final_residual[n_nodes:], ord=np.inf)) if dimension > n_nodes else 0.0
        # Re-evaluate both criteria after the last allowed update.  Hitting the
        # iteration limit is never interpreted as convergence.
        residual_ok = (
            final_node_residual <= current_absolute_tolerance + relative_tolerance * final_current_scale
            and final_branch_residual <= voltage_absolute_tolerance + relative_tolerance * final_voltage_scale
        )
        update_ok = (
            last_delta_v <= update_voltage_tolerance + relative_tolerance * max(
                1.0, float(np.linalg.norm(x[:n_nodes], ord=np.inf)) if n_nodes else 0.0
            )
            and last_delta_i <= update_current_tolerance + relative_tolerance * max(
                1e-3, float(np.linalg.norm(x[n_nodes:], ord=np.inf)) if dimension > n_nodes else 0.0
            )
        )
        stage_converged = stage_converged or (residual_ok and update_ok)
        stage_records.append({
            "stage": stage_number, "gmin_S": gmin, "source_scale": source_scale,
            "iterations": stage_iteration, "converged": stage_converged,
            "scaled_residual": float(final_merit), "residual_current_inf_A": final_node_residual,
            "residual_voltage_inf_V": final_branch_residual,
        })
        if not stage_converged:
            raise NumericalSolveError(
                f"Nonlinear Newton did not converge in {max_iterations} iterations at continuation stage "
                f"{stage_number}; GMIN={gmin:.3e} S, source scale={source_scale:.3g}, "
                f"scaled residual={final_merit:.3e}, last dV={last_delta_v:.3e} V, "
                f"last dI={last_delta_i:.3e} A."
            )

    assert x is not None
    if not stage_records or not stage_records[-1]["converged"]:
        raise NumericalSolveError("Nonlinear MNA final physical stage did not converge.")
    if stages[-1] != (0.0, 1.0):
        raise NumericalSolveError("Nonlinear MNA continuation did not finish at the physical circuit (GMIN=0, source=1).")

    voltages = {0: 0.0}
    for node_id, index in node_index.items():
        voltages[node_id] = float(x[index])
    branch_currents = {element.reference: float(x[index]) for element in model.elements
                       if element.reference.casefold() in branch_index
                       for index in [branch_index[element.reference.casefold()]]}
    currents_by_key = {key: float(x[index]) for key, index in branch_index.items()}
    normalized_overrides = {str(k).casefold(): float(v) for k, v in (source_overrides or {}).items()}
    kcl = {node_id: 0.0 for node_id in model.non_ground_nodes}
    branch_results: dict[str, dict[str, float | str]] = {}
    for element in model.elements:
        kind, p, n = element.kind, element.nodes[0], element.nodes[1]
        vp, vn = voltages.get(p, 0.0), voltages.get(n, 0.0)
        voltage = vp - vn
        if kind == "R": current = voltage / element.value
        elif kind == "I": current = normalized_overrides.get(element.reference.casefold(), element.value)
        elif kind in ("V", "L", "E", "H"): current = branch_currents[element.reference]
        elif kind == "G": current = element.value * (voltages.get(element.nodes[2], 0.0) - voltages.get(element.nodes[3], 0.0))
        elif kind == "F": current = element.value * currents_by_key[str(element.control_source).casefold()]
        elif kind == "C": current = 0.0
        else: continue
        if p != 0: kcl[p] += current
        if n != 0: kcl[n] -= current
        branch_results[element.reference] = {"voltage_V": float(voltage), "current_A": float(current),
                                             "power_W": float(voltage * current), "model": kind}

    device_results: dict[str, dict[str, Any]] = {}
    for device in devices:
        evaluation = final_evaluations[device.reference]
        terminal_voltages = [voltages.get(node, 0.0) for node in device.terminals]
        for node, current in zip(device.terminals, evaluation.terminal_currents_A):
            if node != 0:
                kcl[node] += float(current)
        entry: dict[str, Any] = {
            "model": type(device.model).__name__, "terminals": list(device.terminals),
            "terminal_voltages_V": [float(v) for v in terminal_voltages],
            "terminal_currents_A": [float(i) for i in evaluation.terminal_currents_A],
            "current_jacobian_S": evaluation.current_jacobian_S.tolist(),
            **{key: float(value) if isinstance(value, (int, float, np.floating)) else value
               for key, value in evaluation.operating_point.items()},
        }
        if evaluation.terminal_charges_C is not None:
            entry["terminal_charges_C"] = [float(q) for q in evaluation.terminal_charges_C]
            entry["charge_jacobian_F"] = evaluation.charge_jacobian_F.tolist()
        device_results[device.reference] = entry
        if device.kind == "D":
            voltage = float(terminal_voltages[0] - terminal_voltages[1])
            current = float(evaluation.terminal_currents_A[0])
            branch_results[device.reference] = {"voltage_V": voltage, "current_A": current,
                                                "power_W": voltage * current, "model": "Shockley diode"}

    kcl_inf = max((abs(value) for value in kcl.values()), default=0.0)
    final_node_residual = float(np.linalg.norm(final_residual[:n_nodes], ord=np.inf)) if n_nodes else 0.0
    final_branch_residual = float(np.linalg.norm(final_residual[n_nodes:], ord=np.inf)) if dimension > n_nodes else 0.0
    diagnostics = {
        "backend": "circhecker.solver.nonlinear",
        "method": final_method,
        "matrix_dimension": int(final_stage_matrix.shape[0]),
        "matrix_nnz": int(final_jacobian.nnz),
        "condition_estimate": final_condition,
        "condition_estimate_method": final_condition_method,
        "iterations": int(total_iterations),
        "continuation_stages": stage_records,
        "converged": True,
        "residual_current_inf_A": final_node_residual,
        "residual_voltage_inf_V": final_branch_residual,
        "relative_tolerance": float(relative_tolerance),
        "voltage_absolute_tolerance_V": float(voltage_absolute_tolerance),
        "current_absolute_tolerance_A": float(current_absolute_tolerance),
        "update_voltage_tolerance_V": float(update_voltage_tolerance),
        "update_current_tolerance_A": float(update_current_tolerance),
        "last_voltage_update_V": float(last_delta_v),
        "last_current_update_A": float(last_delta_i),
        "line_search_enabled": bool(line_search),
        "convergence_assistance": {
            "gmin_stepping": {"attempted": bool(gmin_steps), "stages_S": normalized_gmin if gmin_steps else []},
            "source_stepping": {"attempted": bool(source_steps), "scales": normalized_sources if source_steps else []},
        },
        "device_count": len(devices),
        "device_models": {device.reference: type(device.model).__name__ for device in devices},
        "warnings": final_warnings + (["Capacitors are open and ideal inductors are shorted in DC."] if ignored else []),
    }
    current_tol = max(1e-10, 1e-7 * max(final_current_scale, 1e-12))
    return NonlinearMNASolution(
        voltages, branch_currents, branch_results, diagnostics,
        {"kcl": {"max_residual_A": float(kcl_inf), "is_balanced": bool(kcl_inf <= current_tol),
                 "tolerance_A": float(current_tol), "node_residuals_A": kcl},
         "newton_residual": {"current_inf_A": final_node_residual, "voltage_inf_V": final_branch_residual}},
        device_results,
    )


def assemble_nonlinear_ac(ir: CircuitIR, operating_point: NonlinearMNASolution | Mapping[str, Any],
                           frequency_hz: float, *,
                           source_phasors: Mapping[str, complex] | None = None):
    """Assemble AC MNA about a nonlinear DC operating point.

    The linear AC stamps are reused unchanged.  Each nonlinear device adds
    ``di/dv + j*omega*dq/dv`` at its operating point, so DC and AC consume the
    same terminal equations and derivatives.  Device models without a charge
    law contribute conductance only.
    """
    from .ac import _source_phasors, assemble_ac

    frequency = float(frequency_hz)
    if not math.isfinite(frequency) or frequency < 0.0:
        raise ValueError("AC frequency must be finite and non-negative.")
    node_voltages = (operating_point.node_voltages_V
                     if isinstance(operating_point, NonlinearMNASolution)
                     else operating_point.get("node_voltages_V"))
    if not isinstance(node_voltages, Mapping):
        raise ValueError("A nonlinear DC operating-point mapping with node_voltages_V is required.")
    voltages_by_id = {int(node): float(value) for node, value in node_voltages.items()}
    if any(not math.isfinite(value) for value in voltages_by_id.values()):
        raise ValueError("Nonlinear operating-point voltages must be finite.")
    devices = _semiconductors_from_ir(ir)
    linear_ir = deepcopy(ir)
    linear_ir.components = [c for c in linear_ir.components if str(c.get("type", "")).upper() not in _DEVICE_MODEL_KEYS]
    model = model_from_circuit_ir(linear_ir)
    effective_sources, _ = _source_phasors(model.elements, source_phasors)
    matrix, rhs, node_index, branch_index = assemble_ac(model, frequency, effective_sources)
    omega = 2.0 * math.pi * frequency
    rows: list[int] = []
    cols: list[int] = []
    values: list[complex] = []
    for device in devices:
        terminal_values = np.asarray([voltages_by_id.get(node, 0.0) for node in device.terminals], dtype=float)
        evaluation = device.model.evaluate_terminals(terminal_values)
        local_admittance = evaluation.current_jacobian_S.astype(complex)
        if evaluation.charge_jacobian_F is not None:
            local_admittance = local_admittance + 1j * omega * evaluation.charge_jacobian_F
        for local_row, row_node in enumerate(device.terminals):
            row = node_index.get(row_node)
            if row is None:
                continue
            for local_col, col_node in enumerate(device.terminals):
                col = node_index.get(col_node)
                if col is None:
                    continue
                value = complex(local_admittance[local_row, local_col])
                if value != 0j:
                    rows.append(row)
                    cols.append(col)
                    values.append(value)
    if rows:
        device_matrix = coo_matrix((values, (rows, cols)), shape=matrix.shape, dtype=complex).tocsc()
        device_matrix.sum_duplicates()
        matrix = (matrix + device_matrix).tocsc()
    return matrix, rhs, node_index, branch_index


def solve_nonlinear_ac(ir: CircuitIR, *, operating_point: NonlinearMNASolution | Mapping[str, Any] | None = None,
                       frequencies_hz: Sequence[float] | None = None,
                       source_phasors: Mapping[str, complex] | None = None,
                       dense_threshold: int = 64, residual_tolerance: float = 1e-9) -> dict[str, Any]:
    """Solve small-signal AC using the nonlinear device Jacobians at DC OP."""
    from .ac import frequency_sweep

    op = operating_point if operating_point is not None else solve_nonlinear_dc(ir)
    config = ir.analysis_configuration.get("ac_sweep", {})
    frequencies = frequency_sweep(config, frequencies_hz=frequencies_hz)
    node_names = {int(node["id"]): str(node["name"]) for node in ir.nodes}
    node_names.setdefault(0, "0")
    node_values: dict[str, list[complex | None]] = {name: [] for name in node_names.values()}
    branch_values: dict[str, list[complex | None]] = {}
    diagnostics: list[dict[str, Any]] = []
    for frequency in frequencies:
        try:
            matrix, rhs, node_index, branch_index = assemble_nonlinear_ac(
                ir, op, float(frequency), source_phasors=source_phasors
            )
            solution, method, condition, condition_method, residual, relative, warnings = solve_matrix(
                matrix, rhs, dense_threshold=dense_threshold, residual_tolerance=residual_tolerance
            )
            volts = {0: 0j}
            volts.update({node: complex(solution[index]) for node, index in node_index.items()})
            for node_id, name in node_names.items():
                node_values[name].append(volts.get(node_id, 0j))
            for reference, index in branch_index.items():
                branch_values.setdefault(reference, [None] * (len(diagnostics) + 1))
                branch_values[reference][-1] = complex(solution[index])
            diagnostics.append({"frequency_Hz": float(frequency), "status": "success", "method": method,
                                "matrix_dimension": int(matrix.shape[0]), "matrix_nnz": int(matrix.nnz),
                                "residual_inf": float(residual), "relative_residual": float(relative),
                                "condition_estimate": condition,
                                "condition_estimate_method": condition_method, "warnings": warnings})
        except Exception as exc:
            for values in node_values.values():
                values.append(None)
            for values in branch_values.values():
                values.append(None)
            diagnostics.append({"frequency_Hz": float(frequency), "status": "failed",
                                "error": f"{type(exc).__name__}: {exc}"})
    failed = [item for item in diagnostics if item["status"] != "success"]
    return {
        "frequency_Hz": [float(value) for value in frequencies],
        "node_voltages_V": node_values,
        "branch_currents_A": branch_values,
        "frequency_diagnostics": diagnostics,
        "solver_diagnostics": {"backend": "circhecker.solver.nonlinear_ac",
                                "frequency_count": int(len(frequencies)),
                                "failed_frequency_count": int(len(failed)),
                                "converged": not failed},
        "operating_point_source": "supplied" if operating_point is not None else "solved internally",
    }
