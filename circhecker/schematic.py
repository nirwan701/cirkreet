"""Public extraction-only API for KiCad and SPICE circuit sources."""
from __future__ import annotations
from pathlib import Path
from typing import Any

from .importers.base import ImportResult, ImporterError
from .importers.detection import detect_format
from .importers.kicad_xml import KiCadXMLImporter
from .importers.kicad_cli import KiCadCLIImporter
from .importers.spice import SpiceImporter
from .exporters.circuit_json import CircuitJSONExporter
from .validation.extraction import validate_extraction

class SchematicExtractor:
    """Extract source connectivity to CircuitIR without running circuit simulation.

    Supported direct inputs: KiCad XML ``.kicad_net`` and SPICE-compatible
    ``.cir/.net/.sp/.spi/.spice/.ckt/.txt``. Native ``.kicad_sch`` is handled
    by KiCad CLI export. LTspice ``.asc`` drawing interpretation is not enabled.
    """
    def __init__(self, *, kicad_cli: str | Path | None = None, timeout: float = 120.0,
                 schema_path: str | Path | None = None):
        self.kicad_xml = KiCadXMLImporter()
        self.kicad_cli = KiCadCLIImporter(kicad_cli, timeout=timeout)
        self.spice = SpiceImporter()
        self.json_exporter = CircuitJSONExporter(schema_path)
        self.last_result: ImportResult | None = None

    def extract(self, source: str | Path, *, variant: str | None = None,
                generated_netlist_path: str | Path | None = None,
                preserve_generated_netlist: bool = False,
                ai_provenance: dict[str, Any] | None = None):
        """Return CircuitIR from a file; this method never invokes the solver."""
        # Do not expose stale diagnostics/metadata when a later import fails.
        self.last_result = None
        fmt = detect_format(source)
        if fmt == "kicad_xml":
            result = self.kicad_xml.import_file(source, variant=variant, ai_provenance=ai_provenance)
        elif fmt == "kicad_schematic":
            result = self.kicad_cli.import_file(source, variant=variant,
                output_path=generated_netlist_path, preserve_netlist=preserve_generated_netlist,
                ai_provenance=ai_provenance)
        elif fmt == "spice":
            result = self.spice.import_file(source)
            if ai_provenance:
                result.circuit.provenance["external_generation"] = dict(ai_provenance)
        elif fmt == "ltspice_asc":
            raise ImporterError("Native LTspice .asc drawing parsing is not supported. Export a SPICE-compatible .net/.cir netlist from LTspice and import that file. CirKreety does not infer wiring from drawing coordinates.")
        else:  # pragma: no cover - guarded by detect_format
            raise ImporterError(f"Unsupported source format: {fmt}")
        validate_extraction(result.circuit)
        # Explicit extraction-only metadata is preserved in CircuitIR as well as
        # returned through extract_with_metadata().
        result.circuit.source_information["simulation_executed"] = False
        result.metadata["simulation_executed"] = False
        result.diagnostics = _diagnostics(result.circuit)
        self.last_result = result
        return result.circuit

    def extract_with_metadata(self, source: str | Path, **kwargs) -> ImportResult:
        circuit = self.extract(source, **kwargs)
        assert self.last_result is not None
        self.last_result.circuit = circuit
        self.last_result.diagnostics = (list(circuit.validation_results.get("errors", [])) +
                                        list(circuit.validation_results.get("warnings", [])))
        return self.last_result

    def import_kicad_xml(self, path: str | Path, *, ai_provenance: dict[str, Any] | None = None):
        result = self.kicad_xml.import_file(path, ai_provenance=ai_provenance)
        validate_extraction(result.circuit)
        result.circuit.source_information["simulation_executed"] = False
        result.metadata["simulation_executed"] = False
        result.diagnostics = _diagnostics(result.circuit)
        self.last_result = result
        return result.circuit

    def import_spice(self, path: str | Path):
        result = self.spice.import_file(path)
        validate_extraction(result.circuit)
        result.circuit.source_information["simulation_executed"] = False
        result.metadata["simulation_executed"] = False
        result.diagnostics = _diagnostics(result.circuit)
        self.last_result = result
        return result.circuit

    def import_spice_text(self, text: str, source_name: str = "<memory>"):
        result = self.spice.import_text(text, source_name=source_name)
        validate_extraction(result.circuit)
        result.circuit.source_information["simulation_executed"] = False
        result.metadata["simulation_executed"] = False
        result.diagnostics = _diagnostics(result.circuit)
        self.last_result = result
        return result.circuit

    def validate(self, circuit):
        result = validate_extraction(circuit)
        if self.last_result is not None and self.last_result.circuit is circuit:
            self.last_result.diagnostics = _diagnostics(circuit)
        return result

    def export_json(self, circuit, path: str | Path):
        return self.json_exporter.export(circuit, path)

    def to_json_dict(self, circuit):
        document = self.json_exporter.to_dict(circuit)
        self.json_exporter.validate(document)
        return document

    @property
    def diagnostics(self) -> list[dict[str, Any]]:
        return list(self.last_result.diagnostics) if self.last_result else []

    @property
    def import_metadata(self) -> dict[str, Any]:
        return dict(self.last_result.metadata) if self.last_result else {}

__all__ = ["SchematicExtractor", "ImportResult", "ImporterError"]


def _diagnostics(circuit):
    return list(circuit.validation_results.get("errors", [])) + list(circuit.validation_results.get("warnings", []))
