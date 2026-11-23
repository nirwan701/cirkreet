"""Versioned, format-neutral circuit intermediate representation."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any

SCHEMA_VERSION = "1.0.0"

@dataclass
class CircuitIR:
    metadata: dict[str, Any] = field(default_factory=lambda: {"schema": "CircuitIR", "schema_version": SCHEMA_VERSION})
    source_information: dict[str, Any] = field(default_factory=dict)
    components: list[dict[str, Any]] = field(default_factory=list)
    terminals: list[dict[str, Any]] = field(default_factory=list)
    nodes: list[dict[str, Any]] = field(default_factory=list)
    branches: list[dict[str, Any]] = field(default_factory=list)
    wires: list[dict[str, Any]] = field(default_factory=list)
    labels: list[dict[str, Any]] = field(default_factory=list)
    ports: list[dict[str, Any]] = field(default_factory=list)
    subcircuits: dict[str, Any] = field(default_factory=dict)
    hierarchy: list[dict[str, Any]] = field(default_factory=list)
    models: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    expressions: list[dict[str, Any]] = field(default_factory=list)
    directives: list[dict[str, Any]] = field(default_factory=list)
    simulation_options: dict[str, Any] = field(default_factory=dict)
    initial_conditions: dict[str, float] = field(default_factory=dict)
    analysis_configuration: dict[str, Any] = field(default_factory=dict)
    validation_results: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def node_map(self) -> dict[str, int]:
        return {str(n["name"]): int(n["id"]) for n in self.nodes}

    def component(self, reference: str) -> dict[str, Any]:
        for comp in self.components:
            if comp["reference"].casefold() == reference.casefold():
                return comp
        raise KeyError(reference)
