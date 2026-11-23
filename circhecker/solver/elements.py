"""Typed element records accepted by the generalized DC MNA assembler."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True)
class Element:
    """A two-terminal or controlled element in the circuit's node numbering."""

    reference: str
    kind: str
    nodes: tuple[int, ...]
    value: float = 0.0
    control_source: str | None = None
    ac_value: complex = 0j
    ac_specified: bool = False


@dataclass(frozen=True)
class CircuitModel:
    """Solver-neutral linear circuit model; node ID 0 is the reference node."""

    node_names: dict[int, str]
    elements: tuple[Element, ...]

    @property
    def non_ground_nodes(self) -> tuple[int, ...]:
        return tuple(sorted(node_id for node_id in self.node_names if node_id != 0))


ElementKind: TypeAlias = str
