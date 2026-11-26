"""Generalized sparse Modified Nodal Analysis for linear DC circuits.

The implementation solves the DC specialization of ``G x + C dx/dt + f(x) = b``.
Capacitors are open circuits in DC; ideal inductors add a zero-voltage branch
constraint and a branch-current unknown. AC and transient formulations are
intentionally not inferred from this DC-only solver.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..circuit_ir import CircuitIR
from .diagnostics import SolverDiagnostics
from .elements import CircuitModel, Element
from .exceptions import MNAError, UnsupportedElementError
from .matrix import solve_matrix
from .stamping import DCElementStamper, SparseStampBuilder

SUPPORTED_DC = {"R", "C", "L", "V", "I", "E", "G", "F", "H"}
BRANCH_KINDS = {"V", "L", "E", "H"}


@dataclass
class MNASolution:
    node_voltages_V: dict[int, float]
    branch_currents_A: dict[str, float]
    branch_results: dict[str, dict[str, float | str]]
    diagnostics: SolverDiagnostics
    verification: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_voltages_V": self.node_voltages_V,
            "branch_currents_A": self.branch_currents_A,
            "branch_results": self.branch_results,
            "solver_diagnostics": self.diagnostics.to_dict(),
            "verification": self.verification,
        }


def model_from_circuit_ir(ir: CircuitIR) -> CircuitModel:
    """Translate the existing CircuitIR into a strict linear-DC solver model."""
    supported, reasons = supports_circuit_ir_dc(ir)
    if not supported:
        raise UnsupportedElementError("The generalized linear-DC MNA backend cannot model this CircuitIR: " + "; ".join(reasons))
    node_names = {int(node["id"]): str(node["name"]) for node in ir.nodes}
    if 0 not in node_names:
        node_names[0] = "0"
    elements: list[Element] = []
    unsupported: list[str] = []
    for component in ir.components:
        ref = str(component["reference"])
        kind = str(component["type"]).upper()
        if kind not in SUPPORTED_DC:
            unsupported.append(f"{ref} ({kind})")
            continue
        pins = component.get("pins", [])
        nodes = tuple(int(pin["node_id"]) for pin in pins)
        parameters = component.get("parameters", {})
        if kind in ("R", "C", "L", "V", "I"):
            required = 2
        elif kind in ("E", "G"):
            required = 4
        else:
            required = 2
        if len(nodes) != required:
            raise MNAError(f"{ref}: {kind} requires {required} terminal node(s); found {len(nodes)}.")
        if kind in ("R", "C", "L"):
            value = float(parameters.get("value", 0.0))
        elif kind in ("V", "I"):
            value = float(parameters.get("dc", 0.0))
        elif kind in ("E", "G"):
            value = float(parameters.get("gain", 0.0))
        else:
            value = float(parameters.get("gain", 0.0))
        control_source = parameters.get("control_source") if kind in ("F", "H") else None
        ac_magnitude = float(parameters.get("ac_magnitude", 0.0)) if kind in ("V", "I") else 0.0
        ac_phase = np.deg2rad(float(parameters.get("ac_phase_deg", 0.0))) if kind in ("V", "I") else 0.0
        ac_value = complex(ac_magnitude * np.exp(1j * ac_phase)) if kind in ("V", "I") else 0j
        ac_specified = bool(parameters.get("ac_specified", False)) if kind in ("V", "I") else False
        if kind in ("F", "H") and not control_source:
            raise MNAError(f"{ref}: {kind} requires a named controlling voltage-source branch.")
        if not np.isfinite(value):
            raise MNAError(f"{ref}: non-finite parameter value is not permitted.")
        if kind == "R" and value <= 0:
            raise MNAError(f"{ref}: resistance must be strictly positive.")
        if kind in ("C", "L") and value < 0:
            raise MNAError(f"{ref}: {kind} value must be non-negative.")
        elements.append(Element(ref, kind, nodes, value, str(control_source) if control_source else None, ac_value, ac_specified))
    if unsupported:
        raise UnsupportedElementError(
            "The generalized linear-DC MNA backend does not yet support: " + ", ".join(unsupported)
        )
    return CircuitModel(node_names=node_names, elements=tuple(elements))


def assemble_dc(model: CircuitModel, source_overrides: Mapping[str, float] | None = None):
    """Assemble sparse DC MNA matrix and return its unknown-index maps."""
    overrides = {str(k).casefold(): float(v) for k, v in (source_overrides or {}).items()}
    source_kinds = {element.reference.casefold(): element.kind for element in model.elements}
    for reference, value in overrides.items():
        if reference not in source_kinds or source_kinds[reference] not in ("V", "I"):
            raise MNAError(f"Source override {reference!r} does not identify an independent V/I source.")
        if not np.isfinite(value):
            raise MNAError(f"Source override for {reference!r} must be finite.")
    node_ids = model.non_ground_nodes
    node_index = {node_id: i for i, node_id in enumerate(node_ids)}
    branch_elements = [e for e in model.elements if e.kind in BRANCH_KINDS]
    branch_index = {element.reference.casefold(): len(node_ids) + i
                    for i, element in enumerate(branch_elements)}
    dimension = len(node_ids) + len(branch_elements)
    builder = SparseStampBuilder(dimension)
    stamper = DCElementStamper(
        builder=builder,
        node_index=node_index,
        branch_index=branch_index,
        source_overrides=overrides,
    )
    stamper.stamp_all(model.elements)
    return builder.matrix(), builder.rhs, node_index, branch_index, stamper.ignored_dc_elements


def solve_dc(model_or_ir: CircuitModel | CircuitIR, *, source_overrides: Mapping[str, float] | None = None,
             dense_threshold: int = 64, residual_tolerance: float = 1e-9) -> MNASolution:
    """Solve a linear DC operating point from a ``CircuitIR`` or ``CircuitModel``."""
    model = model_from_circuit_ir(model_or_ir) if isinstance(model_or_ir, CircuitIR) else model_or_ir
    matrix, rhs, node_index, branch_index, ignored_dc = assemble_dc(model, source_overrides)
    solution, method, condition, condition_method, residual, relative, solve_warnings = solve_matrix(
        matrix, rhs, dense_threshold=dense_threshold, residual_tolerance=residual_tolerance
    )
    voltages = {0: 0.0}
    for node_id, index in node_index.items():
        voltages[node_id] = float(solution[index])
    normalized_overrides = {str(key).casefold(): float(value) for key, value in (source_overrides or {}).items()}
    currents_by_key = {reference: float(solution[index]) for reference, index in branch_index.items()}
    currents = {element.reference: currents_by_key[element.reference.casefold()]
                for element in model.elements if element.kind in BRANCH_KINDS}

    def voltage(node: int) -> float:
        return voltages.get(node, 0.0)

    terminal_currents: dict[str, tuple[int, int, float, float]] = {}
    branch_results: dict[str, dict[str, float | str]] = {}
    kcl: dict[int, float] = {node_id: 0.0 for node_id in model.non_ground_nodes}
    for element in model.elements:
        kind, p, n = element.kind, element.nodes[0], element.nodes[1]
        v = voltage(p) - voltage(n)
        if kind == "R":
            i = v / element.value
        elif kind == "I":
            i = normalized_overrides.get(element.reference.casefold(), element.value)
        elif kind in ("V", "L", "E", "H"):
            i = currents[element.reference]
        elif kind == "G":
            i = element.value * (voltage(element.nodes[2]) - voltage(element.nodes[3]))
        elif kind == "F":
            control = str(element.control_source).casefold()
            i = element.value * currents_by_key[control]
        else:  # capacitor open in DC
            i = 0.0
        terminal_currents[element.reference] = (p, n, i, v)
        if p != 0:
            kcl[p] += i
        if n != 0:
            kcl[n] -= i
        branch_results[element.reference] = {
            "voltage_V": float(v), "current_A": float(i), "power_W": float(v * i),
            "model": kind,
        }
    kcl_inf = max((abs(value) for value in kcl.values()), default=0.0)
    current_scale = max((abs(item[2]) for item in terminal_currents.values()), default=0.0)
    kcl_tolerance = max(1e-12, 1e-7 * current_scale)
    diagnostics = SolverDiagnostics(
        backend="circhecker.solver.mna", method=method, matrix_dimension=matrix.shape[0],
        matrix_nnz=int(matrix.nnz), residual_inf=float(residual),
        relative_residual=float(relative), condition_estimate=condition,
        condition_estimate_method=condition_method, converged=True,
        warnings=solve_warnings + (["Ideal capacitors are open circuits in DC analysis: " + ", ".join(ignored_dc)] if ignored_dc else []),
        ignored_dc_elements=ignored_dc,
    )
    return MNASolution(
        node_voltages_V=voltages, branch_currents_A=currents,
        branch_results=branch_results, diagnostics=diagnostics,
        verification={"kcl": {"max_residual_A": float(kcl_inf),
                               "is_balanced": bool(kcl_inf <= kcl_tolerance),
                               "tolerance_A": float(kcl_tolerance),
                               "node_residuals_A": kcl}},
    )


def supports_circuit_ir_dc(ir: CircuitIR) -> tuple[bool, list[str]]:
    """Return an explicit capability decision without attempting a numerical solve.

    The capability decision checks both element family and parameter keys, so a
    parameter that would affect the physical model is not silently discarded.
    """
    reasons: list[str] = []
    allowed_parameters = {
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
    references = {str(c.get("reference", "")).casefold(): c for c in ir.components}
    if not ir.components:
        reasons.append("CircuitIR contains no components to simulate")
    for component in ir.components:
        reference = str(component.get("reference", "?"))
        kind = str(component.get("type", "?")).upper()
        if kind not in SUPPORTED_DC:
            reasons.append(f"{reference}: {kind} is not supported by generalized linear-DC MNA")
            continue
        parameters = component.get("parameters", {})
        extra = sorted(set(parameters) - allowed_parameters[kind])
        if extra:
            reasons.append(f"{reference}: unsupported {kind} parameter(s) would be ignored: {', '.join(extra)}")
        if kind in ("F", "H"):
            control = str(parameters.get("control_source", "")).casefold()
            controlled = references.get(control)
            if controlled is None:
                reasons.append(f"{reference}: controlling voltage-source reference {parameters.get('control_source')!r} was not found")
            elif str(controlled.get("type", "")).upper() != "V":
                reasons.append(f"{reference}: control reference {parameters.get('control_source')!r} is not an independent voltage source")
    return not reasons, reasons
