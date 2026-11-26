"""Public, read-only capability inventory for integrations and tooling."""
from __future__ import annotations

from .execution.engine import CircuitExecutionEngine
from .execution.registry import DEFAULT_COMPONENT_REGISTRY, DEFAULT_SOLVER_REGISTRY


def capability_profile() -> dict:
    """Return a JSON-compatible snapshot of registered components and solvers.

    This is an advisory capability inventory, not a promise that every element
    combination is simulatable. Backend-level topology and parameter checks
    remain authoritative for an individual circuit.
    """
    return {
        "analysis_keys": list(CircuitExecutionEngine.ANALYSIS_KEYS),
        "components": DEFAULT_COMPONENT_REGISTRY.profile(),
        "solvers": DEFAULT_SOLVER_REGISTRY.profile(),
        "scope_note": (
            "Capabilities are conditional on topology, parameters, and the selected backend. "
            "Inspect per-circuit analysis status and diagnostics before using results."
        ),
    }
