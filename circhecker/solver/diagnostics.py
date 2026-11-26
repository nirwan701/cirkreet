"""Structured numerical diagnostics for MNA solves."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SolverDiagnostics:
    backend: str
    method: str
    matrix_dimension: int
    matrix_nnz: int
    residual_inf: float | None
    relative_residual: float | None
    condition_estimate: float | None
    condition_estimate_method: str | None
    converged: bool
    warnings: list[str] = field(default_factory=list)
    ignored_dc_elements: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
