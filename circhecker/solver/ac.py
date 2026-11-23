"""Generalized sparse linear AC MNA analysis.

This module solves ``A(j*w) X = B`` for supported linear RLC and controlled-
source circuits. It reuses the CircuitIR adapter, branch-current convention,
sparse matrix solver, and numerical diagnostics from the DC MNA engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log10
from typing import Any, Iterable, Mapping

import numpy as np

from ..circuit_ir import CircuitIR
from .elements import CircuitModel, Element
from .exceptions import MNAError, UnsupportedElementError
from .matrix import solve_matrix
from .mna import BRANCH_KINDS, model_from_circuit_ir, supports_circuit_ir_dc
from .stamping import ACElementStamper, SparseStampBuilder


@dataclass
class ACAnalysisSolution:
    """Frequency-domain response and per-frequency numerical diagnostics."""

    data: dict[str, Any]
    status: str
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "reason": self.reason, "data": self.data}


def frequency_sweep(config: Mapping[str, Any] | None = None, *,
                    frequencies_hz: Iterable[float] | None = None) -> np.ndarray:
    """Create an explicitly configured linear, decade, or octave AC sweep.

    For ``dec`` and ``oct``, ``points`` means points per decade or octave.
    For ``lin``, it means the total number of points, including both endpoints.
    """
    if frequencies_hz is not None:
        values = np.asarray(list(frequencies_hz), dtype=float)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("At least one AC frequency must be supplied.")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("AC frequencies must be finite and non-negative.")
        if values.size > 1 and np.any(np.diff(values) <= 0):
            raise ValueError("AC frequencies must be strictly increasing.")
        return values

    cfg = dict(config or {})
    start = float(cfg.get("start_Hz", 1.0))
    stop = float(cfg.get("stop_Hz", 1e6))
    points = int(cfg.get("points", 160))
    scale = str(cfg.get("scale", "log")).lower()
    if not np.isfinite(start) or not np.isfinite(stop) or start < 0 or stop < start:
        raise ValueError("AC sweep requires finite frequencies with 0 <= start <= stop.")
    if points < 1:
        raise ValueError("AC sweep points must be a positive integer.")
    if start == stop:
        return np.asarray([start], dtype=float)
    if scale == "lin":
        if points < 2:
            raise ValueError("A linear AC sweep requires at least two points.")
        return np.linspace(start, stop, points, dtype=float)
    if start <= 0:
        raise ValueError("Logarithmic AC sweeps require a strictly positive start frequency.")
    if scale == "log":
        if points < 2:
            raise ValueError("A logarithmic AC sweep requires at least two points.")
        return np.geomspace(start, stop, points, dtype=float)
    if scale == "dec":
        count = max(2, int(ceil(points * log10(stop / start))) + 1)
    elif scale == "oct":
        count = max(2, int(ceil(points * np.log2(stop / start))) + 1)
    else:
        raise ValueError(f"Unsupported AC sweep scale {scale!r}; expected dec, oct, or lin.")
    return np.geomspace(start, stop, count, dtype=float)


def supports_circuit_ir_ac(ir: CircuitIR) -> tuple[bool, list[str]]:
    """Return whether the current linear MNA subset can represent this AC circuit."""
    supported, reasons = supports_circuit_ir_dc(ir)
    if not supported:
        return False, ["AC MNA currently supports the same linear element/parameter subset as DC MNA: " + r
                       for r in reasons]
    if not any(str(c.get("type", "")).upper() in ("V", "I") for c in ir.components):
        return False, ["AC MNA requires at least one independent voltage or current source."]
    return True, []


def _source_phasors(elements: tuple[Element, ...],
                    overrides: Mapping[str, complex] | None) -> tuple[dict[str, complex], bool]:
    sources = [e for e in elements if e.kind in ("V", "I")]
    if not sources:
        raise UnsupportedElementError("AC analysis requires at least one independent V or I source.")
    explicit = any(e.ac_specified for e in sources)
    phasors: dict[str, complex] = {}
    if explicit:
        phasors = {e.reference.casefold(): (complex(e.ac_value) if e.ac_specified else 0j)
                   for e in sources}
        for reference, value in phasors.items():
            if not (np.isfinite(value.real) and np.isfinite(value.imag)):
                raise MNAError(f"AC source {reference!r} has a non-finite magnitude or phase.")
        implicit = False
    else:
        # Preserve the previous CirKreety convention: absent AC cards, use the
        # first independent source as a unit small-signal excitation.
        chosen = next((e for e in sources if e.kind == "V"), sources[0])
        phasors = {e.reference.casefold(): (1.0 + 0j if e is chosen else 0j) for e in sources}
        implicit = True
    source_refs = {e.reference.casefold() for e in sources}
    for reference, value in (overrides or {}).items():
        key = str(reference).casefold()
        if key not in source_refs:
            raise MNAError(f"AC source override {reference!r} does not identify an independent V/I source.")
        z = complex(value)
        if not (np.isfinite(z.real) and np.isfinite(z.imag)):
            raise MNAError(f"AC source override for {reference!r} must be finite.")
        phasors[key] = z
    if not any(abs(value) > 0 for value in phasors.values()):
        raise MNAError("All independent AC source phasors are zero; no circuit excitation is defined.")
    return phasors, implicit


def assemble_ac(model: CircuitModel, frequency_hz: float,
                source_phasors: Mapping[str, complex] | None = None):
    """Assemble the complex sparse MNA matrix at one frequency."""
    frequency = float(frequency_hz)
    if not np.isfinite(frequency) or frequency < 0:
        raise ValueError("AC frequency must be finite and non-negative.")
    node_ids = model.non_ground_nodes
    node_index = {node_id: i for i, node_id in enumerate(node_ids)}
    branch_elements = [element for element in model.elements if element.kind in BRANCH_KINDS]
    branch_index = {element.reference.casefold(): len(node_ids) + i
                    for i, element in enumerate(branch_elements)}
    dimension = len(node_ids) + len(branch_elements)
    builder = SparseStampBuilder(dimension, dtype=complex)
    stamper = ACElementStamper(
        builder=builder,
        node_index=node_index,
        branch_index=branch_index,
        source_phasors={str(k).casefold(): complex(v) for k, v in (source_phasors or {}).items()},
        omega=2.0 * np.pi * frequency,
    )
    stamper.stamp_all(model.elements)
    return builder.matrix(), builder.rhs, node_index, branch_index


def solve_ac(model_or_ir: CircuitModel | CircuitIR,
             frequencies_hz: Iterable[float] | None = None, *,
             sweep_config: Mapping[str, Any] | None = None,
             source_overrides: Mapping[str, complex] | None = None,
             output_nodes: tuple[int, int] | None = None,
             dense_threshold: int = 64,
             residual_tolerance: float = 1e-9) -> ACAnalysisSolution:
    """Solve a single-frequency or multi-frequency linear AC MNA problem.

    ``source_overrides`` are complex peak phasors in SI units. If the netlist
    contains any explicit ``AC`` source specifications, unspecified sources
    are zero. Otherwise the first independent voltage source (or current source
    if no voltage source exists) receives a unit phasor for legacy compatibility.
    """
    model = model_from_circuit_ir(model_or_ir) if isinstance(model_or_ir, CircuitIR) else model_or_ir
    frequencies = frequency_sweep(sweep_config, frequencies_hz=frequencies_hz)
    source_phasors, implicit_excitation = _source_phasors(model.elements, source_overrides)
    active_sources = {key: value for key, value in source_phasors.items() if abs(value) > 0}
    reference_source = next(iter(active_sources)) if len(active_sources) == 1 else None
    reference_source_kind = next((e.kind for e in model.elements
                                  if reference_source is not None and e.reference.casefold() == reference_source), None)
    transfer_units = "V/V" if reference_source_kind == "V" else "V/A (ohm)" if reference_source_kind == "I" else None
    names = {int(k): str(v) for k, v in model.node_names.items()}
    names.setdefault(0, "0")
    if output_nodes is None:
        output_nodes = (max(model.non_ground_nodes, default=0), 0)
    out_pos, out_neg = (int(output_nodes[0]), int(output_nodes[1]))
    if out_pos not in names or out_neg not in names:
        raise MNAError(f"AC output node pair {output_nodes!r} is not present in the circuit model.")

    node_values: dict[str, list[complex | None]] = {names[node]: [] for node in sorted(names)}
    branch_refs = [e.reference for e in model.elements if e.kind in BRANCH_KINDS]
    branch_values: dict[str, list[complex | None]] = {ref: [] for ref in branch_refs}
    component_values: dict[str, list[complex | None]] = {e.reference: [] for e in model.elements}
    output_values: list[complex | None] = []
    transfer_values: list[complex | None] = []
    point_diagnostics: list[dict[str, Any]] = []
    methods: set[str] = set()
    dimensions: list[int] = []
    nonzeros: list[int] = []
    residuals: list[float | None] = []
    relative_residuals: list[float | None] = []

    for frequency in frequencies:
        try:
            matrix, rhs, node_index, branch_index = assemble_ac(model, float(frequency), source_phasors)
            solution, method, condition, condition_method, residual, relative, warnings = solve_matrix(
                matrix, rhs, dense_threshold=dense_threshold,
                residual_tolerance=residual_tolerance,
            )
            methods.add(method)
            dimensions.append(int(matrix.shape[0]))
            nonzeros.append(int(matrix.nnz))
            residuals.append(float(residual))
            relative_residuals.append(float(relative))
            volts = {0: 0j}
            volts.update({node: complex(solution[index]) for node, index in node_index.items()})
            branches = {reference: complex(solution[index]) for reference, index in branch_index.items()}

            def v(node: int) -> complex:
                return volts.get(node, 0j)

            components: dict[str, complex] = {}
            for element in model.elements:
                p, n = element.nodes[:2]
                voltage = v(p) - v(n)
                if element.kind == "R":
                    current = voltage / element.value
                elif element.kind == "C":
                    current = 1j * 2.0 * np.pi * float(frequency) * element.value * voltage
                elif element.kind in ("V", "L", "E", "H"):
                    current = branches[element.reference.casefold()]
                elif element.kind == "I":
                    current = source_phasors.get(element.reference.casefold(), 0j)
                elif element.kind == "G":
                    current = element.value * (v(element.nodes[2]) - v(element.nodes[3]))
                elif element.kind == "F":
                    control = str(element.control_source).casefold()
                    if control not in branches:
                        raise UnsupportedElementError(
                            f"{element.reference}: controlling source {element.control_source!r} "
                            "has no branch-current unknown."
                        )
                    current = element.value * branches[control]
                else:  # pragma: no cover - guarded by model construction
                    raise UnsupportedElementError(f"No AC current evaluation for {element.kind}.")
                components[element.reference] = complex(current)

            out = v(out_pos) - v(out_neg)
            transfer = (out / active_sources[reference_source]) if reference_source else None
            for node, value in volts.items():
                node_values[names[node]].append(value)
            for ref in branch_refs:
                branch_values[ref].append(branches[ref.casefold()])
            for ref in component_values:
                component_values[ref].append(components[ref])
            output_values.append(out)
            transfer_values.append(transfer)
            point_diagnostics.append({
                "frequency_Hz": float(frequency), "status": "success", "method": method,
                "matrix_dimension": int(matrix.shape[0]), "matrix_nnz": int(matrix.nnz),
                "residual_inf": float(residual), "relative_residual": float(relative),
                "condition_estimate": condition,
                "condition_estimate_method": condition_method,
                "warnings": warnings,
            })
        except Exception as exc:
            dimensions.append(0)
            nonzeros.append(0)
            residuals.append(None)
            relative_residuals.append(None)
            for values in node_values.values():
                values.append(None)
            for values in branch_values.values():
                values.append(None)
            for values in component_values.values():
                values.append(None)
            output_values.append(None)
            transfer_values.append(None)
            point_diagnostics.append({
                "frequency_Hz": float(frequency), "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            })

    failed = [item for item in point_diagnostics if item["status"] != "success"]
    # If there is exactly one active input source, magnitude and phase describe
    # its transfer response. For multiple sources they describe output voltage.
    if reference_source is not None:
        response_values = np.asarray([complex(x) if x is not None else np.nan + 1j*np.nan
                                      for x in transfer_values], dtype=complex)
        magnitude_reference = f"transfer from {reference_source} ({transfer_units})"
    else:
        response_values = np.asarray([complex(x) if x is not None else np.nan + 1j*np.nan
                                      for x in output_values], dtype=complex)
        magnitude_reference = "output voltage (dBV); multiple independent AC sources"
    magnitudes = np.abs(response_values)
    phases = np.degrees(np.angle(response_values))
    magnitude_db = 20.0 * np.log10(np.maximum(magnitudes, 1e-300))
    valid = np.asarray([item["status"] == "success" for item in point_diagnostics], dtype=bool)
    magnitude_list = [float(magnitudes[i]) if valid[i] else None for i in range(len(frequencies))]
    magnitude_db_list = [float(magnitude_db[i]) if valid[i] else None for i in range(len(frequencies))]
    phase_list = [float(phases[i]) if valid[i] else None for i in range(len(frequencies))]

    data = {
        "frequency_Hz": [float(x) for x in frequencies],
        "node_voltages_V": node_values,
        "branch_currents_A": branch_values,
        "component_currents_A": component_values,
        "output_nodes": [names[out_pos], names[out_neg]],
        "output_voltage_V": output_values,
        "output_magnitude_V": [float(abs(x)) if x is not None else None for x in output_values],
        "output_phase_deg": [float(np.degrees(np.angle(x))) if x is not None else None for x in output_values],
        "transfer_function": transfer_values if reference_source else None,
        "magnitude": magnitude_list,
        "magnitude_dB": magnitude_db_list,
        "phase_deg": phase_list,
        "magnitude_reference": magnitude_reference,
        "source_phasors": {key: value for key, value in source_phasors.items()},
        "reference_source": reference_source,
        "reference_source_kind": reference_source_kind,
        "transfer_units": transfer_units,
        "implicit_unit_excitation": implicit_excitation,
        "frequency_diagnostics": point_diagnostics,
        "solver_diagnostics": {
            "backend": "circhecker.solver.ac",
            "method": sorted(methods) if methods else [],
            "frequency_count": int(len(frequencies)),
            "matrix_dimensions": dimensions,
            "matrix_nonzeros": nonzeros,
            "residual_inf": residuals,
            "relative_residual": relative_residuals,
            "failed_frequency_count": int(len(failed)),
            "failed_frequencies_Hz": [item["frequency_Hz"] for item in failed],
            "converged": not failed,
        },
    }
    status = "failed" if failed else "success"
    reason = (f"{len(failed)} of {len(frequencies)} AC frequency point(s) failed; see frequency_diagnostics." if failed else None)
    return ACAnalysisSolution(data=data, status=status, reason=reason)
