"""Canonical, source-independent execution plan for circuit analyses.

``CircuitIR`` describes the electrical circuit. ``AnalysisSpec`` describes a
user's requested analyses. ``AnalysisIR`` is the normalized execution plan
created from those two inputs; it records exactly what the solver was asked
to run without modifying the circuit source model.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..circuit_ir import CircuitIR
from .spec import ANALYSIS_NAMES, AnalysisSpec

ANALYSIS_IR_SCHEMA_VERSION = "1.0.0"


def _normalise_analyses(analyses: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(analyses, str):
        values = (analyses.strip().lower(),)
    else:
        values = tuple(str(item).strip().lower() for item in analyses)
    values = tuple(dict.fromkeys(item for item in values if item))
    if not values:
        # An empty plan is useful for parse-and-validate workflows that only
        # want CircuitIR without launching a numerical solver.
        return ()
    if "all" in values:
        if len(values) != 1:
            raise ValueError("'all' cannot be combined with individual analysis names.")
        return tuple(sorted(ANALYSIS_NAMES))
    unknown = set(values) - ANALYSIS_NAMES
    if unknown:
        raise ValueError(f"Unknown analysis name(s): {', '.join(sorted(unknown))}.")
    return values


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AnalysisIR:
    """Normalized analysis execution plan tied to a canonical circuit ID.

    The plan is deliberately JSON-friendly so it can be saved, exchanged with
    another tool, or inspected by a student without importing solver classes.
    All values use SI units unless a field name explicitly says otherwise.
    """

    circuit_id: str
    analyses: tuple[str, ...]
    analysis_configuration: dict[str, Any]
    source_analysis_configuration: dict[str, Any]
    simulation_options: dict[str, Any]
    outputs: dict[str, Any]
    schema_version: str = ANALYSIS_IR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.circuit_id or len(self.circuit_id) != 64 or any(ch not in "0123456789abcdef" for ch in self.circuit_id):
            raise ValueError("AnalysisIR requires a lowercase SHA-256 circuit_id.")
        if self.schema_version != ANALYSIS_IR_SCHEMA_VERSION:
            raise ValueError(f"Unsupported AnalysisIR schema version: {self.schema_version}.")
        normalized = _normalise_analyses(self.analyses)
        object.__setattr__(self, "analyses", normalized)
        object.__setattr__(self, "analysis_configuration", deepcopy(self.analysis_configuration))
        object.__setattr__(self, "source_analysis_configuration", deepcopy(self.source_analysis_configuration))
        object.__setattr__(self, "simulation_options", deepcopy(self.simulation_options))
        object.__setattr__(self, "outputs", deepcopy(self.outputs))

    @classmethod
    def from_request(
        cls,
        circuit: CircuitIR,
        analyses: str | Iterable[str] = "all",
        analysis_spec: AnalysisSpec | None = None,
    ) -> "AnalysisIR":
        """Build the effective execution plan without changing ``circuit``."""
        if not isinstance(circuit, CircuitIR):
            raise TypeError("AnalysisIR.from_request expects a CircuitIR instance.")
        if analysis_spec is not None and not isinstance(analysis_spec, AnalysisSpec):
            raise TypeError("analysis_spec must be an AnalysisSpec instance.")

        requested = analysis_spec.analyses if analysis_spec is not None else _normalise_analyses(analyses)
        source_configuration = deepcopy(circuit.analysis_configuration)
        effective_configuration = deepcopy(source_configuration)
        simulation_options = deepcopy(circuit.simulation_options)
        outputs: dict[str, Any] = {}

        if analysis_spec is not None:
            configuration_overrides, option_overrides = analysis_spec.configuration_overrides()
            effective_configuration.update(configuration_overrides)
            simulation_options.update(option_overrides)
            outputs = {
                "positive_node": None if analysis_spec.output_node is None else str(analysis_spec.output_node),
                "negative_node": None if analysis_spec.output_negative_node is None else str(analysis_spec.output_negative_node),
            }
        else:
            outputs = {
                "positive_node": effective_configuration.get("output_node"),
                "negative_node": effective_configuration.get("output_negative_node"),
            }

        return cls(
            circuit_id=circuit.fingerprint(),
            analyses=tuple(requested),
            analysis_configuration=effective_configuration,
            source_analysis_configuration=source_configuration,
            simulation_options=simulation_options,
            outputs=outputs,
        )

    @property
    def analysis_id(self) -> str:
        """Stable identifier for this request and circuit pairing."""
        return _canonical_hash(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": "AnalysisIR",
            "schema_version": self.schema_version,
            "circuit_id": self.circuit_id,
            "analyses": list(self.analyses),
            "analysis_configuration": deepcopy(self.analysis_configuration),
            "source_analysis_configuration": deepcopy(self.source_analysis_configuration),
            "simulation_options": deepcopy(self.simulation_options),
            "outputs": deepcopy(self.outputs),
        }
        if include_id:
            payload["analysis_id"] = self.analysis_id
        return payload

    def to_json(self, path: str | Path | None = None, *, indent: int = 2) -> str:
        """Serialize the plan as JSON and optionally write it atomically."""
        encoded = json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, allow_nan=False)
        if path is not None:
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_text(encoded, encoding="utf-8")
            temporary.replace(destination)
        return encoded

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AnalysisIR":
        """Load a previously serialized plan after checking its schema name."""
        if value.get("schema") not in (None, "AnalysisIR"):
            raise ValueError("The supplied document is not an AnalysisIR document.")
        if value.get("schema_version", ANALYSIS_IR_SCHEMA_VERSION) != ANALYSIS_IR_SCHEMA_VERSION:
            raise ValueError(f"Unsupported AnalysisIR schema version: {value.get('schema_version')}.")
        plan = cls(
            circuit_id=str(value.get("circuit_id", "")),
            analyses=tuple(value.get("analyses", ())),
            analysis_configuration=dict(value.get("analysis_configuration", {})),
            source_analysis_configuration=dict(value.get("source_analysis_configuration", {})),
            simulation_options=dict(value.get("simulation_options", {})),
            outputs=dict(value.get("outputs", {})),
            schema_version=str(value.get("schema_version", ANALYSIS_IR_SCHEMA_VERSION)),
        )
        supplied_id = value.get("analysis_id")
        if supplied_id is not None and supplied_id != plan.analysis_id:
            raise ValueError("AnalysisIR analysis_id does not match its contents.")
        return plan
