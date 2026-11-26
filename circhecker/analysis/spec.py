"""Typed, format-independent requests for circuit analyses.

Analysis configuration belongs to a user's analysis request, not to the
source-derived CircuitIR. Directives parsed from SPICE remain preserved in
CircuitIR; an AnalysisSpec can override them for an individual run without
mutating the imported circuit.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

ANALYSIS_NAMES = frozenset({
    "dc_operating_point", "dc_sweep", "ac_sweep", "transient",
    "frequency_analysis", "transfer_function", "state_space",
})


def _finite(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


@dataclass(frozen=True)
class DCSweepSpec:
    """One-dimensional sweep of an independent voltage/current source."""

    source: str
    start: float
    stop: float
    step: float

    def __post_init__(self) -> None:
        if not str(self.source).strip():
            raise ValueError("DC sweep source reference must not be empty.")
        start, stop, step = (_finite("DC sweep start", self.start),
                             _finite("DC sweep stop", self.stop),
                             _finite("DC sweep step", self.step))
        if step == 0 or (stop > start and step < 0) or (stop < start and step > 0):
            raise ValueError("DC sweep step must be non-zero and point from start toward stop.")
        object.__setattr__(self, "source", str(self.source).strip())
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)
        object.__setattr__(self, "step", step)

    def to_config(self) -> dict[str, Any]:
        return {"source": self.source, "start": self.start, "stop": self.stop, "step": self.step}


@dataclass(frozen=True)
class ACSweepSpec:
    """Frequency sweep in SI units (Hz)."""

    start_Hz: float = 1.0
    stop_Hz: float = 1e6
    points: int = 160
    scale: str = "log"

    def __post_init__(self) -> None:
        start = _finite("AC sweep start_Hz", self.start_Hz)
        stop = _finite("AC sweep stop_Hz", self.stop_Hz)
        scale = str(self.scale).lower()
        raw_points = self.points
        points = int(raw_points)
        if isinstance(raw_points, bool) or points != raw_points:
            raise ValueError("AC sweep points must be an integer.")
        if start < 0 or stop < start:
            raise ValueError("AC sweep requires 0 <= start_Hz <= stop_Hz.")
        if scale not in {"lin", "log", "dec", "oct"}:
            raise ValueError("AC sweep scale must be one of 'lin', 'log', 'dec', or 'oct'.")
        minimum_points = 2 if scale in {"lin", "log"} and start != stop else 1
        if points < minimum_points:
            raise ValueError(f"AC sweep points must be >= {minimum_points} for scale {scale!r}.")
        if scale != "lin" and start <= 0 and start != stop:
            raise ValueError("Logarithmic, decade, and octave sweeps require start_Hz > 0.")
        object.__setattr__(self, "start_Hz", start)
        object.__setattr__(self, "stop_Hz", stop)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "scale", scale)

    def to_config(self) -> dict[str, Any]:
        return {"start_Hz": self.start_Hz, "stop_Hz": self.stop_Hz,
                "points": self.points, "scale": self.scale}


@dataclass(frozen=True)
class TransientSpec:
    """Transient integration controls; all time values are seconds."""

    stop_s: float
    step_s: float
    method: str = "backward_euler"
    adaptive: bool = False
    relative_tolerance: float = 1e-3
    absolute_voltage_tolerance: float = 1e-6
    absolute_current_tolerance: float = 1e-9
    gmin_S: float = 1e-12
    minimum_step_s: float | None = None
    maximum_steps: int = 200_000
    maximum_newton_iterations: int = 60

    def __post_init__(self) -> None:
        stop = _finite("Transient stop_s", self.stop_s)
        step = _finite("Transient step_s", self.step_s)
        method = str(self.method).lower().replace("-", "_")
        if stop <= 0 or step <= 0 or step > stop:
            raise ValueError("Transient requires 0 < step_s <= stop_s.")
        if method not in {"backward_euler", "trapezoidal"}:
            raise ValueError("Transient method must be 'backward_euler' or 'trapezoidal'.")
        reltol = _finite("Transient relative_tolerance", self.relative_tolerance)
        vabstol = _finite("Transient absolute_voltage_tolerance", self.absolute_voltage_tolerance)
        iabstol = _finite("Transient absolute_current_tolerance", self.absolute_current_tolerance)
        gmin = _finite("Transient gmin_S", self.gmin_S)
        if min(reltol, vabstol, iabstol) <= 0 or gmin < 0:
            raise ValueError("Transient tolerances must be positive and gmin_S must be non-negative.")
        minimum = None if self.minimum_step_s is None else _finite("Transient minimum_step_s", self.minimum_step_s)
        if minimum is not None and (minimum <= 0 or minimum > step):
            raise ValueError("Transient minimum_step_s must be positive and no larger than step_s.")
        if not isinstance(self.adaptive, bool):
            raise ValueError("Transient adaptive must be a boolean.")
        max_steps = int(self.maximum_steps)
        max_newton = int(self.maximum_newton_iterations)
        if (max_steps != self.maximum_steps or max_newton != self.maximum_newton_iterations
                or max_steps < 1 or max_newton < 1):
            raise ValueError("Transient maximum_steps and maximum_newton_iterations must be positive integers.")
        object.__setattr__(self, "stop_s", stop)
        object.__setattr__(self, "step_s", step)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "relative_tolerance", reltol)
        object.__setattr__(self, "absolute_voltage_tolerance", vabstol)
        object.__setattr__(self, "absolute_current_tolerance", iabstol)
        object.__setattr__(self, "gmin_S", gmin)
        object.__setattr__(self, "minimum_step_s", minimum)
        object.__setattr__(self, "maximum_steps", max_steps)
        object.__setattr__(self, "maximum_newton_iterations", max_newton)

    def to_config(self) -> dict[str, Any]:
        return {"stop_s": self.stop_s, "step_s": self.step_s, "adaptive": bool(self.adaptive)}

    def to_simulation_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {
            "METHOD": "trap" if self.method == "trapezoidal" else "backward_euler",
            "ADAPTIVE": int(bool(self.adaptive)),
            "RELTOL": self.relative_tolerance,
            "VABSTOL": self.absolute_voltage_tolerance,
            "IABSTOL": self.absolute_current_tolerance,
            "GMIN": self.gmin_S,
            "MAXSTEPS": self.maximum_steps,
            "ITL1": self.maximum_newton_iterations,
        }
        if self.minimum_step_s is not None:
            options["TMIN"] = self.minimum_step_s
        return options


@dataclass(frozen=True)
class AnalysisSpec:
    """Complete analysis request for one CircuitIR execution.

    `analysis_configuration` parsed from the input file remains part of the
    source record. Values supplied here take precedence for this run only.
    """

    analyses: tuple[str, ...] | Iterable[str] | str = ("dc_operating_point",)
    dc_sweep: DCSweepSpec | None = None
    ac_sweep: ACSweepSpec | None = None
    transient: TransientSpec | None = None
    output_node: str | int | None = None
    output_negative_node: str | int | None = None

    def __post_init__(self) -> None:
        raw = self.analyses
        if isinstance(raw, str):
            normalized = (raw,) if raw == "all" else (raw,)
        else:
            normalized = tuple(str(item) for item in raw)
        normalized = tuple(dict.fromkeys(item.strip().lower() for item in normalized))
        if not normalized:
            raise ValueError("AnalysisSpec must request at least one analysis.")
        if "all" in normalized:
            if len(normalized) != 1:
                raise ValueError("'all' cannot be combined with individual analysis names.")
            normalized = tuple(sorted(ANALYSIS_NAMES))
        unknown = set(normalized) - ANALYSIS_NAMES
        if unknown:
            raise ValueError(f"Unknown analysis name(s): {', '.join(sorted(unknown))}.")
        if self.dc_sweep is not None and "dc_sweep" not in normalized:
            raise ValueError("dc_sweep configuration requires 'dc_sweep' in analyses.")
        if self.ac_sweep is not None and "ac_sweep" not in normalized:
            raise ValueError("ac_sweep configuration requires 'ac_sweep' in analyses.")
        if self.transient is not None and "transient" not in normalized:
            raise ValueError("transient configuration requires 'transient' in analyses.")
        if self.output_node is not None and not str(self.output_node).strip():
            raise ValueError("output_node must not be empty.")
        if self.output_negative_node is not None and not str(self.output_negative_node).strip():
            raise ValueError("output_negative_node must not be empty.")
        if self.output_negative_node is not None and self.output_node is None:
            raise ValueError("output_node must be supplied when output_negative_node is set.")
        object.__setattr__(self, "analyses", normalized)

    def configuration_overrides(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return source-independent analysis and solver-option overlays."""
        config: dict[str, Any] = {}
        options: dict[str, Any] = {}
        if self.dc_sweep is not None:
            config["dc_sweep"] = self.dc_sweep.to_config()
        if self.ac_sweep is not None:
            config["ac_sweep"] = self.ac_sweep.to_config()
        if self.transient is not None:
            config["transient"] = self.transient.to_config()
            options.update(self.transient.to_simulation_options())
        if self.output_node is not None:
            config["output_node"] = str(self.output_node)
        if self.output_negative_node is not None:
            config["output_negative_node"] = str(self.output_negative_node)
        return config, options

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "AnalysisSpec":
        """Construct an AnalysisSpec from a plain mapping for JSON/YAML clients."""
        data = dict(values)
        if isinstance(data.get("dc_sweep"), Mapping):
            data["dc_sweep"] = DCSweepSpec(**data["dc_sweep"])
        if isinstance(data.get("ac_sweep"), Mapping):
            data["ac_sweep"] = ACSweepSpec(**data["ac_sweep"])
        if isinstance(data.get("transient"), Mapping):
            data["transient"] = TransientSpec(**data["transient"])
        return cls(**data)
