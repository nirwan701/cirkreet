"""Example post-simulation resistor power-rating check.

This plugin is intentionally opt-in: the user supplies the actual continuous
power ratings from the chosen part datasheets. It checks DC operating-point
power only and does not claim thermal, pulse, PCB, or derating compliance.
"""
from __future__ import annotations

from typing import Any, Mapping

from circhecker.circuit_ir import CircuitIR


class ResistorPowerRatingPlugin:
    name = "resistor_power_rating_dc"

    def __init__(self, ratings_w: Mapping[str, float], safety_fraction: float = 1.0):
        if not 0 < float(safety_fraction) <= 1:
            raise ValueError("safety_fraction must be in the interval (0, 1].")
        self.ratings_w = {str(ref).casefold(): float(value) for ref, value in ratings_w.items()}
        if any(value <= 0 for value in self.ratings_w.values()):
            raise ValueError("Every resistor power rating must be positive.")
        self.safety_fraction = float(safety_fraction)

    def run(self, circuit: CircuitIR, execution: Mapping[str, Any]) -> Mapping[str, Any]:
        components = execution.get("numerical_metrics", {}).get("component_electrical_metrics", {}).get("components", {})
        circuit_resistors = {str(item.get("reference", "")).casefold(): item
                             for item in circuit.components if str(item.get("type", "")).upper() == "R"}
        checks = []
        for ref_key, rating_w in self.ratings_w.items():
            component = circuit_resistors.get(ref_key)
            actual = next((value for name, value in components.items() if str(name).casefold() == ref_key), None)
            if component is None:
                checks.append({"reference": ref_key, "status": "unknown_component", "rating_W": rating_w})
                continue
            if not isinstance(actual, dict) or actual.get("power_W") is None:
                checks.append({"reference": component.get("reference"), "status": "not_evaluated",
                               "rating_W": rating_w, "reason": "No DC operating-point power metric is available."})
                continue
            power_w = abs(float(actual["power_W"]))
            allowed_w = rating_w * self.safety_fraction
            checks.append({
                "reference": component.get("reference"),
                "status": "within_limit" if power_w <= allowed_w else "over_limit",
                "dc_power_W": power_w,
                "datasheet_rating_W": rating_w,
                "configured_limit_W": allowed_w,
                "utilization_fraction": power_w / rating_w,
            })
        return {
            "scope": "DC operating-point resistor dissipation only; transient pulse, thermal, PCB and derating checks are not assessed",
            "safety_fraction": self.safety_fraction,
            "checks": checks,
        }
