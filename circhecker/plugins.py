"""Extension interfaces for project-specific circuit review logic.

Review plugins are deliberately post-simulation extensions: they consume the
normalized CircuitIR and the completed execution record, then return a
JSON-serializable mapping. They do not replace or silently modify solver
models. A plugin failure is reported as a warning and does not erase a valid
core simulation result.
"""
from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

from .circuit_ir import CircuitIR


@runtime_checkable
class ReviewPlugin(Protocol):
    """Protocol implemented by optional post-simulation review plugins.

    Implementations must expose a stable, non-empty ``name`` and a
    ``run(circuit, execution)`` method that returns a mapping. Values should
    be JSON-compatible; the report serializer will normalize NumPy scalars
    and arrays but plugins should avoid custom object instances.
    """

    name: str

    def run(
        self,
        circuit: CircuitIR,
        execution: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...
