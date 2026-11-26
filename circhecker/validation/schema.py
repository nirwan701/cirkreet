"""Validation helpers for the external CirKreetyCircuit JSON contract."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

class CircuitSchemaValidationError(RuntimeError):
    """A JSON document does not comply with its declared CirKreety schema."""

def validate_circuit_json(document: dict[str, Any], schema_path: str | Path) -> None:
    try:
        import jsonschema
    except ImportError as exc:
        raise CircuitSchemaValidationError("JSON Schema validation requires the 'jsonschema' package.") from exc
    path = Path(schema_path)
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema).validate(document)
    except Exception as exc:
        raise CircuitSchemaValidationError(f"Schema validation failed against {path}: {exc}") from exc
