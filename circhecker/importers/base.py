"""Common contracts for extraction-only importers."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

IMPORTER_VERSION = "1.0.0"

class ImporterError(RuntimeError):
    """A source could not be imported without guessing its circuit structure."""

@dataclass
class ImportResult:
    circuit: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
