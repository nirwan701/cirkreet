"""Extraction-only adapter around the existing SPICE parser."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ..parser.spice import parse_spice_file, parse_spice_text
from .base import ImportResult, IMPORTER_VERSION


class SpiceImporter:
    """Normalize metadata on the existing parser's CircuitIR without simulating."""

    def import_file(self, path: str | Path) -> ImportResult:
        source = Path(path).expanduser().resolve()
        ir = parse_spice_file(source)
        try:
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
        except OSError:
            digest = None
        self._annotate(ir)
        ir.source_information.update({
            "format": "SPICE-compatible netlist",
            "path": str(source),
            "filename": source.name,
            "sha256": digest,
            "simulation_executed": False,
        })
        ir.provenance.update({
            "parser": "CirKreety SpiceImporter",
            "importer_version": IMPORTER_VERSION,
            "connectivity_basis": "explicit positional node tokens in SPICE element statements",
        })
        metadata = {
            "format": "spice", "filename": source.name, "path": str(source),
            "sha256": digest, "simulation_executed": False,
        }
        return ImportResult(ir, metadata, _diagnostics(ir))

    def import_text(self, text: str, source_name: str = "<memory>") -> ImportResult:
        ir = parse_spice_text(text, source_name=source_name)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self._annotate(ir)
        ir.source_information.update({
            "format": "SPICE-compatible netlist", "path": source_name,
            "filename": Path(source_name).name, "sha256": digest,
            "simulation_executed": False,
        })
        ir.provenance.update({
            "parser": "CirKreety SpiceImporter",
            "importer_version": IMPORTER_VERSION,
            "connectivity_basis": "explicit positional node tokens in SPICE element statements",
        })
        metadata = {
            "format": "spice", "filename": Path(source_name).name,
            "sha256": digest, "simulation_executed": False,
        }
        return ImportResult(ir, metadata, _diagnostics(ir))

    @staticmethod
    def _annotate(ir) -> None:
        """Add public extraction evidence without changing parser/solver equations."""
        for component in ir.components:
            params: dict[str, Any] = component.get("parameters", {})
            if not isinstance(params, dict):
                params = {}
            if "value" not in component:
                if "value" in params:
                    component["value"] = params.get("value")
                elif "dc" in params:
                    component["value"] = params.get("dc")
                elif "gain" in params:
                    component["value"] = params.get("gain")
                else:
                    component["value"] = None
            if "value_raw" not in component:
                component["value_raw"] = params.get("raw_value")
                if component["value_raw"] is None and component.get("model_ref"):
                    component["value_raw"] = str(component["model_ref"])
            component.setdefault("normalized_value", component.get("value") if isinstance(component.get("value"), (int, float)) else None)
            component.setdefault("symbol_type", component.get("type", "unknown"))
            component.setdefault("lib_id", component.get("model_ref"))
            component.setdefault("pin_inventory_complete", bool(component.get("pins")))
            component.setdefault("extraction_status", component.get("validation_status", "parsed"))
            component.setdefault("source_references", {})
            component["source_references"].update({
                "source_line": component.get("source_line"),
                "raw_expression": component.get("original_expression"),
                "hierarchy": component.get("hierarchy", []),
            })
            for pin in component.get("pins", []):
                pin["connectivity_status"] = "connected" if pin.get("node_id") is not None else "unknown"
                pin.setdefault("source_evidence", {
                    "kind": "SPICE positional node token",
                    "source_line": component.get("source_line"),
                    "raw_expression": component.get("original_expression"),
                })
                # Keep CircuitIR.terminals consistent with normalized component pins.
                for terminal in ir.terminals:
                    if (str(terminal.get("component", "")).casefold() == str(component.get("reference", "")).casefold()
                            and str(terminal.get("number", "")) == str(pin.get("number", ""))):
                        terminal.update({
                            "connectivity_status": pin["connectivity_status"],
                            "source_evidence": pin["source_evidence"],
                        })


def _diagnostics(ir):
    return list(ir.validation_results.get("errors", [])) + list(ir.validation_results.get("warnings", []))
