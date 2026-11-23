"""Deterministic CirKreetyCircuit 1.0.0 JSON export and schema validation."""
from __future__ import annotations
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

SCHEMA_NAME = "CirKreetyCircuit"
SCHEMA_VERSION = "1.0.0"
SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "circuit.schema.json"

class CircuitJSONExportError(RuntimeError):
    pass

class CircuitJSONExporter:
    def __init__(self, schema_path: str | Path | None = None):
        self.schema_path = Path(schema_path) if schema_path else SCHEMA_PATH

    def to_dict(self, circuit) -> dict[str, Any]:
        nodes = {int(node["id"]): node for node in circuit.nodes if "id" in node}
        terminals = {(str(t.get("component", "")).casefold(), str(t.get("number", ""))): t for t in circuit.terminals}
        components = []
        for comp in circuit.components:
            ref = str(comp.get("reference", ""))
            params = comp.get("parameters", {}) if isinstance(comp.get("parameters", {}), dict) else {}
            raw_value = comp.get("value_raw")
            if raw_value is None:
                raw_value = params.get("raw_value")
            if raw_value is None:
                raw_value = comp.get("value")
            normalized_value = comp.get("value", params.get("value", params.get("dc", params.get("gain"))))
            pins = []
            for pin in comp.get("pins", []):
                pin_no = str(pin.get("number", ""))
                terminal = terminals.get((ref.casefold(), pin_no), {})
                node_id = pin.get("node_id", terminal.get("node_id"))
                node = nodes.get(int(node_id)) if node_id is not None else None
                status = pin.get("connectivity_status", terminal.get("connectivity_status"))
                if status is None:
                    status = "connected" if node_id is not None else "unknown"
                pins.append({
                    "number": pin_no, "name": str(pin.get("name", pin_no)), "type": pin.get("type"),
                    "net_id": f"net_{int(node_id):04d}" if node_id is not None else None,
                    "net_name": pin.get("node_name") or (node.get("name") if node else None),
                    "connectivity_status": status,
                    "source_evidence": pin.get("source_evidence", terminal.get("source_evidence")),
                })
            components.append({
                "id": str(comp.get("id", f"cmp_{len(components)+1:04d}")),
                "reference": ref, "symbol_type": str(comp.get("symbol_type") or comp.get("lib_id") or comp.get("type", "?")),
                "value": _json_safe(normalized_value),
                "value_normalized": _json_safe(comp.get("normalized_value", normalized_value if isinstance(normalized_value, (int, float)) else None)),
                "value_raw": str(raw_value) if raw_value is not None else None,
                "library_id": comp.get("lib_id") or comp.get("library_id") or comp.get("model_ref"),
                "properties": _json_safe(comp.get("properties", params)),
                "footprint": comp.get("footprint"), "datasheet": comp.get("datasheet"),
                "pins": pins,
                "source_references": _json_safe(comp.get("source_references", {"source_line": comp.get("source_line"), "hierarchy": comp.get("hierarchy", [])})),
                "extraction_status": str(comp.get("extraction_status", comp.get("validation_status", "extracted"))),
                "pin_inventory_complete": bool(comp.get("pin_inventory_complete", bool(comp.get("pins")))),
            })
        nets = []
        source_net_records = circuit.source_information.get("net_records", []) if circuit.source_information else []
        records_by_node = {}
        for record in source_net_records:
            records_by_node.setdefault(int(record.get("node_id", 0)), []).append(record)
        for node in circuit.nodes:
            node_id = int(node["id"])
            raw_records = records_by_node.get(node_id, [])
            connected_to_node = any(
                pin.get("node_id") is not None and int(pin.get("node_id")) == node_id
                and pin.get("connectivity_status", "connected") == "connected"
                for comp in circuit.components for pin in comp.get("pins", [])
            )
            # Node 0 is a solver convention; do not export it as a source net
            # unless the source actually contained that reference connection.
            if node.get("source_present") is False and not raw_records and not connected_to_node:
                continue
            source_names = [str(record.get("name")) for record in raw_records if record.get("name")]
            name = source_names[0] if source_names else str(node.get("name", "0" if node_id == 0 else f"N${node_id}"))
            aliases = list(dict.fromkeys([str(a) for a in node.get("aliases", [])] + source_names[1:]))
            aliases = [a for a in aliases if a != name]
            connected_pins = []
            for comp in components:
                for pin in comp["pins"]:
                    if pin["net_id"] == f"net_{node_id:04d}" and pin["connectivity_status"] == "connected":
                        connected_pins.append({"reference": comp["reference"], "pin": pin["number"],
                                               "pin_name": pin["name"], "source_evidence": pin.get("source_evidence")})
            names_lower = {name.casefold(), *(alias.casefold() for alias in aliases)}
            reference_class = _reference_class(name, aliases)
            hierarchy = _hierarchy(name)
            if raw_records and raw_records[0].get("hierarchy"):
                hierarchy = raw_records[0]["hierarchy"]
            nets.append({"id": f"net_{node_id:04d}", "name": name, "aliases": aliases,
                         "connections": connected_pins, "reference_class": reference_class,
                         "hierarchy": hierarchy})
        unconnected = []
        for comp in components:
            for pin in comp["pins"]:
                if pin["connectivity_status"] != "connected":
                    unconnected.append({"reference": comp["reference"], **pin})
        source_info = circuit.source_information or {}
        source = {
            "format": str(source_info.get("format", "unknown")),
            "filename": str(source_info.get("filename") or Path(str(source_info.get("path", "circuit"))).name),
            "original_schematic": source_info.get("original_schematic"),
            "design_date": source_info.get("design_date"),
            "sha256": source_info.get("sha256"),
            "original_schematic_sha256": source_info.get("original_schematic_sha256"),
            "netlist_sha256": source_info.get("netlist_sha256"),
            "variant": source_info.get("variant"), "kicad_version": source_info.get("kicad_version"),
        }
        validation = circuit.validation_results or {}
        errors = list(validation.get("errors", [])); warnings = list(validation.get("warnings", []))
        status = "error" if errors else "warning" if warnings else "success"
        conn_states = [pin["connectivity_status"] for comp in components for pin in comp["pins"]]
        conn_status = "invalid" if errors else "partial" if any(x in {"unknown", "unconnected"} for x in conn_states) or any(not c["pin_inventory_complete"] for c in components) else "complete"
        output = {
            "schema": SCHEMA_NAME, "schema_version": SCHEMA_VERSION,
            "source": source, "components": components, "nets": nets,
            "unconnected_pins": unconnected,
            "models": _json_safe(circuit.models), "parameters": _json_safe(circuit.parameters),
            "subcircuits": _json_safe(circuit.subcircuits),
            "ports": _json_safe(circuit.ports), "labels": _json_safe(circuit.labels),
            "wires": _json_safe(circuit.wires), "expressions": _json_safe(circuit.expressions),
            "hierarchy": _json_safe(circuit.hierarchy), "directives": _json_safe(circuit.directives),
            "validation": {"status": status, "connectivity_status": conn_status,
                           "errors": _json_safe(errors), "warnings": _json_safe(warnings),
                           "electrical_design_assessed": False,
                           "scope": "Structural extraction only. No electrical design review or simulation was performed."},
            "provenance": {"parser": str(circuit.provenance.get("parser", "CirKreety SchematicExtractor")),
                           "importer_version": circuit.provenance.get("importer_version", "1.0.0"),
                           "connectivity_basis": str(circuit.provenance.get("connectivity_basis", "explicit source netlist node mapping")),
                           **_json_safe(circuit.provenance)},
        }
        _ensure_finite(output)
        return output

    def validate(self, document: dict[str, Any]) -> None:
        from ..validation.schema import validate_circuit_json, CircuitSchemaValidationError
        try:
            validate_circuit_json(document, self.schema_path)
        except CircuitSchemaValidationError as exc:
            raise CircuitJSONExportError(f"CirKreetyCircuit {SCHEMA_VERSION} validation failed: {exc}") from exc

    def export(self, circuit, path: str | Path) -> dict[str, Any]:
        document = self.to_dict(circuit)
        self.validate(document)
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=destination.parent,
                                             prefix=f".{destination.name}.", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(document, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
            temporary.replace(destination)
        except (OSError, TypeError, ValueError) as exc:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise CircuitJSONExportError(f"Could not safely write circuit JSON to {destination}: {exc}") from exc
        return document

def _reference_class(name: str, aliases: list[str]) -> str:
    """Conservatively classify only explicit common reference/power net names.

    A leading slash in a KiCad hierarchical net name is a sheet path, not
    evidence that the net is a power rail.
    """
    tokens = {str(name).casefold(), *(str(alias).casefold() for alias in aliases)}
    if tokens & {"0", "gnd", "/gnd", "agnd", "dgnd", "pgnd", "earth", "chassis_gnd"}:
        return "ground"
    tails = {token.rsplit("/", 1)[-1] for token in tokens}
    common_rails = {
        "vcc", "vdd", "vss", "vee", "vccio", "vddio", "vbus", "vbat", "vbatt",
        "+5v", "-5v", "+3v3", "+3.3v", "+1v8", "+1.8v", "+12v", "-12v",
        "+24v", "-24v", "+48v", "-48v",
    }
    return "power_or_reference" if tails & common_rails else "signal"


def _hierarchy(name: str):
    return [part for part in name.split("/")[:-1] if part] if name.startswith("/") else []

def _json_safe(value):
    if isinstance(value, dict): return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_json_safe(v) for v in value]
    if value is None or isinstance(value, (str, bool, int)): return value
    if isinstance(value, float):
        if not math.isfinite(value): raise ValueError("Non-finite numbers are not permitted in JSON exports.")
        return value
    if hasattr(value, "item"): return _json_safe(value.item())
    return str(value)

def _ensure_finite(value):
    if isinstance(value, dict):
        for item in value.values(): _ensure_finite(item)
    elif isinstance(value, list):
        for item in value: _ensure_finite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("CircuitIR contains NaN or Infinity; JSON export aborted.")
