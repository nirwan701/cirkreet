"""Stable, unit-labelled simulation result objects for applications and reports."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SIMULATION_RESULT_SCHEMA_VERSION = "1.0.0"

# Units are attached to public data paths, not inferred from numeric values.
ANALYSIS_UNITS: dict[str, dict[str, str]] = {
    "dc_operating_point": {
        "node_voltages_V": "V",
        "branch_currents_A": "A",
        "branch_results.voltage_V": "V",
        "branch_results.current_A": "A",
        "branch_results.power_W": "W",
    },
    "dc_sweep": {"source_value": "V or A (independent-source dependent)", "output_V": "V"},
    "ac_sweep": {"frequency_Hz": "Hz", "magnitude_dB": "dB", "phase_deg": "degree"},
    "transient": {
        "time_s": "s", "node_voltages_V": "V", "branch_currents_A": "A",
        "output_V": "V", "inputs": "V",
    },
    "frequency_analysis": {"frequency_Hz": "Hz"},
    "transfer_function": {"frequency_Hz": "Hz"},
    "state_space": {"A": "1/s", "B": "input-dependent", "C": "output-dependent", "D": "input/output-dependent"},
}


def _normalise_analysis_data(name: str, value: Any) -> Any:
    """Add stable field names while retaining backend-specific original data."""
    if not isinstance(value, Mapping):
        return value
    data = dict(value)
    if name == "dc_sweep":
        if "source_value" not in data:
            for key in ("source_V", "sweep", "source_A"):
                if key in data:
                    data["source_value"] = data[key]
                    break
        if "output_V" not in data:
            for key in ("voltage", "output"):
                if key in data:
                    data["output_V"] = data[key]
                    break
    elif name == "ac_sweep":
        if "frequency_Hz" not in data and "frequency" in data:
            data["frequency_Hz"] = data["frequency"]
        if "magnitude_dB" not in data and "magnitude" in data:
            magnitude = np.asarray(data["magnitude"])
            data["magnitude_dB"] = 20.0 * np.log10(np.maximum(np.abs(magnitude), 1e-300))
        if "phase_deg" not in data and "phase_degrees" in data:
            data["phase_deg"] = data["phase_degrees"]
    elif name == "transient":
        if "time_s" not in data and "time" in data:
            data["time_s"] = data["time"]
        if "output_V" not in data and "outputs" in data:
            output = np.asarray(data["outputs"])
            data["output_V"] = output[:, 0] if output.ndim > 1 else output
    return data


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.complexfloating, complex)):
        number = complex(value)
        return {
            "real": _json_safe(number.real),
            "imag": _json_safe(number.imag),
            "magnitude": _json_safe(abs(number)),
            "phase_deg": _json_safe(math.degrees(math.atan2(number.imag, number.real))),
        }
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return str(value)


@dataclass(frozen=True)
class AnalysisResult:
    """One analysis outcome in a consistent, unit-labelled shape."""

    name: str
    status: str
    message: str
    data: Any
    units: dict[str, str]
    diagnostics: dict[str, Any]

    @classmethod
    def from_mapping(cls, name: str, value: Mapping[str, Any]) -> "AnalysisResult":
        status = str(value.get("status", "unknown"))
        reason = value.get("reason")
        if status == "success":
            message = "Analysis completed successfully."
        elif status == "skipped":
            message = str(reason or "This analysis was not requested.")
        elif status == "not_applicable":
            message = str(reason or "This analysis does not apply to this circuit.")
        elif status == "unsupported":
            message = str(reason or "This circuit is outside the supported model scope for this analysis.")
        elif status == "failed":
            message = str(reason or "The numerical solver could not complete this analysis.")
        else:
            message = str(reason or f"The analysis returned status '{status}'.")

        data = _normalise_analysis_data(name, value.get("data"))
        diagnostics: dict[str, Any] = {}
        if isinstance(data, Mapping):
            for key in ("solver_diagnostics", "diagnostics", "verification"):
                if key in data:
                    diagnostics[key] = data[key]
        if value.get("diagnostics") is not None:
            diagnostics["execution"] = value["diagnostics"]
        return cls(
            name=str(name), status=status, message=message, data=data,
            units=dict(ANALYSIS_UNITS.get(name, {})), diagnostics=diagnostics,
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_safe({
            "status": self.status,
            "message": self.message,
            "data": self.data,
            "units": self.units,
            "diagnostics": self.diagnostics,
        })


@dataclass(frozen=True)
class SimulationResult:
    """Top-level machine-readable result, linked to CircuitIR and AnalysisIR."""

    circuit_id: str
    analysis_id: str | None
    status: str
    backend: str | None
    requested_analyses: tuple[str, ...]
    analyses: dict[str, AnalysisResult]
    metrics: dict[str, Any]
    checks: dict[str, Any]
    warnings: list[Any]
    errors: list[Any]
    schema_version: str = SIMULATION_RESULT_SCHEMA_VERSION

    @classmethod
    def from_execution(
        cls,
        circuit,
        execution: Mapping[str, Any],
        *,
        analysis_id: str | None = None,
    ) -> "SimulationResult":
        results = execution.get("analysis_results", {})
        analysis_records = {
            str(name): AnalysisResult.from_mapping(str(name), value)
            for name, value in results.items() if isinstance(value, Mapping)
        }
        summary = execution.get("execution_summary", {})
        requested = tuple(summary.get("requested_analyses", ()))
        successful = [name for name in requested if analysis_records.get(name, AnalysisResult(name, "unknown", "", None, {}, {})).status == "success"]
        failed = [name for name in requested if analysis_records.get(name, AnalysisResult(name, "unknown", "", None, {}, {})).status == "failed"]
        has_errors = bool(execution.get("errors"))
        if has_errors or failed:
            status = "partial" if successful else "failed"
        elif successful:
            status = "success"
        else:
            status = "unsupported"
        return cls(
            circuit_id=circuit.fingerprint(),
            analysis_id=analysis_id,
            status=status,
            backend=summary.get("backend"),
            requested_analyses=requested,
            analyses=analysis_records,
            metrics=dict(execution.get("numerical_metrics", {})),
            checks=dict(execution.get("engineering_checks", {})),
            warnings=list(execution.get("warnings", [])),
            errors=list(execution.get("errors", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_safe({
            "schema": "SimulationResult",
            "schema_version": self.schema_version,
            "circuit_id": self.circuit_id,
            "analysis_id": self.analysis_id,
            "status": self.status,
            "backend": self.backend,
            "requested_analyses": list(self.requested_analyses),
            "analyses": {name: result.to_dict() for name, result in self.analyses.items()},
            "metrics": self.metrics,
            "checks": self.checks,
            "warnings": self.warnings,
            "errors": self.errors,
        })

    def to_json(self, path: str | Path | None = None, *, indent: int = 2) -> str:
        """Serialize this simulation result and optionally save it atomically."""
        encoded = json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, allow_nan=False)
        if path is not None:
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_text(encoded, encoding="utf-8")
            temporary.replace(destination)
        return encoded

    def summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "backend": self.backend,
            "requested": len(self.requested_analyses),
            "successful": sum(self.analyses.get(name) is not None and self.analyses[name].status == "success" for name in self.requested_analyses),
            "unsupported": sum(self.analyses.get(name) is not None and self.analyses[name].status == "unsupported" for name in self.requested_analyses),
            "failed": sum(self.analyses.get(name) is not None and self.analyses[name].status == "failed" for name in self.requested_analyses),
        }
