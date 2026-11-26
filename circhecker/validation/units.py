"""Dimensional consistency checks for normalized CircuitIR component values."""
from __future__ import annotations

import math
from typing import Any

from ..parser.units import UNIT_DIMENSIONS, check_component_units

_EXPECTED_DIMENSION = {
    "R": "resistance",
    "C": "capacitance",
    "L": "inductance",
    "V": "voltage",
    "I": "current",
    "G": "conductance",
}


def _dimension(unit: Any) -> str | None:
    if unit is None:
        return None
    raw = str(unit).strip()
    # Siemens and seconds share the printed letter but differ by case.
    if raw == "S":
        return "conductance"
    token = raw.lower().replace("ω", "ohm").replace("Ω", "ohm")
    token = token.replace("μ", "µ")
    aliases = {
        "ohms": "ohm", "farads": "farad", "henries": "henry",
        "volts": "volt", "amps": "amp", "amperes": "amp",
        "seconds": "sec", "second": "sec", "hertz": "hz",
        "siemens": "siemens",
    }
    token = aliases.get(token, token)
    # SPICE component metadata uses F/H/V/A and ohm. The lower-case lookup is
    # intentional: the value's component type supplies the meaning of the unit.
    return UNIT_DIMENSIONS.get(token)


def validate_circuit_units(circuit: Any) -> dict[str, Any]:
    """Check declared component dimensions and physically invalid passive values.

    This is a metadata consistency check, not full symbolic dimensional
    analysis of arbitrary behavioural expressions or device-model equations.
    Missing unit metadata is counted as skipped rather than assumed correct.
    """
    diagnostics: list[dict[str, Any]] = []
    checked = 0
    skipped = 0
    for component in circuit.components:
        kind = str(component.get("type", "")).upper()
        expected = _EXPECTED_DIMENSION.get(kind)
        if expected is None:
            skipped += 1
            continue
        params = component.get("parameters", {})
        if not isinstance(params, dict):
            diagnostics.append({
                "code": "INVALID_COMPONENT_PARAMETERS",
                "component": component.get("reference"),
                "message": "Component parameters must be a mapping before unit validation.",
            })
            continue
        unit = params.get("unit") or component.get("units")
        actual = _dimension(unit)
        if unit is None:
            skipped += 1
        elif actual is None:
            skipped += 1
            diagnostics.append({
                "code": "UNKNOWN_COMPONENT_UNIT",
                "component": component.get("reference"),
                "unit": str(unit),
                "message": "The declared unit is not recognized; dimensional consistency was not checked.",
            })
        else:
            checked += 1
            if actual != expected:
                diagnostics.append({
                    "code": "COMPONENT_UNIT_DIMENSION_MISMATCH",
                    "component": component.get("reference"),
                    "component_type": kind,
                    "unit": str(unit),
                    "expected_dimension": expected,
                    "actual_dimension": actual,
                    "message": f"{kind} component unit has dimension {actual!r}; expected {expected!r}.",
                })
        value = params.get("value")
        if value is not None:
            try:
                numeric = float(value)
            except (TypeError, ValueError, OverflowError):
                diagnostics.append({
                    "code": "NON_NUMERIC_COMPONENT_VALUE",
                    "component": component.get("reference"),
                    "message": "Normalized component value is not numeric.",
                })
                continue
            if not math.isfinite(numeric):
                diagnostics.append({
                    "code": "NONFINITE_COMPONENT_VALUE",
                    "component": component.get("reference"),
                    "message": "Normalized component value must be finite.",
                })
            for warning in check_component_units(kind, numeric):
                diagnostics.append({
                    "code": "PHYSICALLY_INVALID_COMPONENT_VALUE",
                    "component": component.get("reference"),
                    "message": warning,
                })

    errors = [item for item in diagnostics if item["code"] in {
        "COMPONENT_UNIT_DIMENSION_MISMATCH", "NON_NUMERIC_COMPONENT_VALUE", "NONFINITE_COMPONENT_VALUE",
        "INVALID_COMPONENT_PARAMETERS",
    }]
    status = "error" if errors else "warning" if diagnostics else "complete" if skipped == 0 else "partial"
    return {
        "status": status,
        "scope": "declared component-unit consistency and passive-value checks; no symbolic expression dimensional analysis",
        "checked_components": checked,
        "skipped_components": skipped,
        "diagnostics": diagnostics,
    }
