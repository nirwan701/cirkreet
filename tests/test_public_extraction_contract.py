"""Public import, package-layout and extraction-only contract regressions."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from circhecker import CircuitIR, CircuitReviewer, SchematicExtractor
from circhecker.execution.engine import CircuitExecutionEngine
from circhecker.importers.base import ImporterError

ROOT = Path(__file__).parent


def test_public_api_is_exposed_from_circhecker():
    from circhecker import CircuitIR as PublicCircuitIR
    from circhecker import CircuitReviewer as PublicReviewer
    from circhecker import SchematicExtractor as PublicExtractor

    assert PublicCircuitIR is CircuitIR
    assert PublicReviewer is CircuitReviewer
    assert PublicExtractor is SchematicExtractor


def test_extract_does_not_invoke_simulation_backend(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("extraction-only import called simulation")

    monkeypatch.setattr(CircuitExecutionEngine, "execute", forbidden)
    source = ROOT / "fixtures" / "kicad" / "simple_rc.kicad_net"
    circuit = SchematicExtractor().extract(source)
    assert isinstance(circuit, CircuitIR)
    assert circuit.source_information["simulation_executed"] is False


def test_synthetic_solver_ground_is_not_exported_as_source_connectivity(tmp_path):
    source = (ROOT / "fixtures" / "kicad" / "simple_rc.kicad_net").read_text(encoding="utf-8")
    source = source.replace(
        '<net code="3" name="GND"><node ref="C1" pin="2" pinfunction="~" pintype="passive"/></net>',
        "",
    )
    path = tmp_path / "no_ground.kicad_net"
    path.write_text(source, encoding="utf-8")
    extractor = SchematicExtractor()
    circuit = extractor.extract(path)
    document = extractor.to_json_dict(circuit)

    assert not any(net["name"] in {"0", "GND"} for net in document["nets"])
    c1_pin2 = next(pin for pin in document["components"] if pin["reference"] == "C1")["pins"][1]
    assert c1_pin2["connectivity_status"] == "unconnected"
    assert c1_pin2["net_id"] is None


def test_spice_extraction_keeps_normalized_and_raw_values():
    extractor = SchematicExtractor()
    circuit = extractor.import_spice_text(".title values\nR1 IN 0 10k\n.end", "values.cir")
    component = circuit.component("R1")
    assert component["value"] == pytest.approx(10_000.0)
    assert component["value_raw"] == "10k"
    pin = component["pins"][0]
    assert pin["connectivity_status"] == "connected"
    assert pin["source_evidence"]["kind"] == "SPICE positional node token"
    output = extractor.to_json_dict(circuit)
    record = next(item for item in output["components"] if item["reference"] == "R1")
    assert record["value"] == pytest.approx(10_000.0)
    assert record["value_raw"] == "10k"


def test_failed_native_import_clears_old_metadata(tmp_path):
    extractor = SchematicExtractor(kicad_cli="/missing/kicad-cli")
    extractor.extract(ROOT / "fixtures" / "kicad" / "simple_rc.kicad_net")
    assert extractor.import_metadata
    schematic = tmp_path / "missing_cli.kicad_sch"
    schematic.write_text("(kicad_sch)", encoding="utf-8")
    with pytest.raises(ImporterError):
        extractor.extract(schematic)
    assert extractor.import_metadata == {}
    assert extractor.diagnostics == []


def test_json_schema_identity_and_nonassessment_contract():
    extractor = SchematicExtractor()
    circuit = extractor.extract(ROOT / "fixtures" / "kicad" / "simple_rc.kicad_net")
    document = extractor.to_json_dict(circuit)
    assert document["schema"] == "CirKreetyCircuit"
    assert document["schema_version"] == "1.0.0"
    assert document["validation"]["electrical_design_assessed"] is False
    assert document["provenance"]["connectivity_basis"].startswith("explicit KiCad XML")


def test_export_is_atomic_and_keeps_prior_file_on_bad_document(tmp_path, monkeypatch):
    extractor = SchematicExtractor()
    circuit = extractor.extract(ROOT / "fixtures" / "kicad" / "simple_rc.kicad_net")
    target = tmp_path / "nested" / "out.json"
    target.parent.mkdir()
    target.write_text('{"prior":true}\n', encoding="utf-8")

    original_validate = extractor.json_exporter.validate
    monkeypatch.setattr(extractor.json_exporter, "validate", lambda doc: (_ for _ in ()).throw(ValueError("bad schema")))
    with pytest.raises(ValueError):
        extractor.export_json(circuit, target)
    assert json.loads(target.read_text(encoding="utf-8")) == {"prior": True}
    monkeypatch.setattr(extractor.json_exporter, "validate", original_validate)
