from pathlib import Path
import json
import os
import stat
import sys
import textwrap

import pytest

from circhecker import SchematicExtractor, CircuitReviewer, CircuitIR
from circhecker.importers.base import ImporterError
from circhecker.importers.kicad_cli import KiCadCLIImporter
from circhecker.exporters.circuit_json import CircuitJSONExportError

ROOT = Path(__file__).parent
KICAD = ROOT / "fixtures" / "kicad"
SPICE = ROOT / "fixtures" / "spice"


def test_simple_kicad_xml_pin_to_net_mapping_and_json(tmp_path):
    extractor = SchematicExtractor()
    ir = extractor.extract(KICAD / "simple_rc.kicad_net")
    assert isinstance(ir, CircuitIR)
    assert ir.source_information["simulation_executed"] is False
    assert [c["reference"] for c in ir.components] == ["R1", "C1"]
    assert ir.component("R1")["pins"][0]["node_name"] == "/IN"
    assert ir.component("C1")["pins"][1]["node_id"] == 0
    output = tmp_path / "design.json"
    doc = extractor.export_json(ir, output)
    assert output.exists()
    assert json.loads(output.read_text()) == doc
    assert doc["schema"] == "CirKreetyCircuit"
    assert doc["schema_version"] == "1.0.0"
    assert doc["source"]["design_date"] == "2026-11-21"
    assert doc["validation"]["electrical_design_assessed"] is False
    assert next(n for n in doc["nets"] if n["name"] == "GND")["reference_class"] == "ground"
    assert next(c for c in doc["components"] if c["reference"] == "R1")["value_raw"] == "10k"


def test_pin_inventory_no_connect_unconnected_and_unknown_symbol():
    extractor = SchematicExtractor()
    ir = extractor.extract(KICAD / "complex_features.kicad_net")
    components = {c["reference"]: c for c in ir.components}
    assert components["D1"]["pins"][1]["connectivity_status"] == "no_connect"
    u1_pins = {p["number"]: p for p in components["U1"]["pins"]}
    assert u1_pins["6"]["connectivity_status"] == "unconnected"
    assert u1_pins["7"]["connectivity_status"] == "unconnected"
    xu2 = components["XU2"]
    assert xu2["pin_inventory_complete"] is False
    assert xu2["pins"][0]["connectivity_status"] == "connected"
    assert any(w["code"] == "UNKNOWN_OR_INCOMPLETE_LIBRARY_SYMBOL" for w in ir.validation_results["warnings"])
    assert any(w["code"] == "UNCONNECTED_PIN" for w in ir.validation_results["warnings"])


def test_mixed_device_inventory_sheet_hierarchy_and_nets():
    ir = SchematicExtractor().extract(KICAD / "complex_features.kicad_net")
    refs = {c["reference"]: c for c in ir.components}
    assert {"R1", "C1", "D1", "Q1", "M1", "U1", "XU2"} == set(refs)
    assert len(refs["Q1"]["pins"]) == 3
    assert len(refs["M1"]["pins"]) == 4
    assert len(refs["U1"]["pins"]) == 7
    assert refs["Q1"]["source_references"]["sheet_path"][0]["names"] == "/Power"
    assert [sheet["name"] for sheet in ir.hierarchy] == ["/", "/Power"]
    assert any(n["name"] == "/GLOBAL/CTRL" for n in ir.nodes)
    assert any(n["name"] == "/Power/VCC" for n in ir.nodes)
    exported = SchematicExtractor().to_json_dict(ir)
    assert [sheet["name"] for sheet in exported["hierarchy"]] == ["/", "/Power"]
    assert exported["source"]["design_date"] == "2026-11-21"
    net_classes = {n["name"]: n["reference_class"] for n in exported["nets"]}
    assert net_classes["/Main/IN"] == "signal"
    assert net_classes["/Power/VCC"] == "power_or_reference"


def test_duplicate_net_name_emits_diagnostic_and_preserves_distinct_nodes():
    ir = SchematicExtractor().extract(KICAD / "complex_features.kicad_net")
    hits = [w for w in ir.validation_results["warnings"] if w["code"] == "DUPLICATE_NET_NAME"]
    assert hits
    ids = [n["id"] for n in ir.nodes if n["name"] == "/Main/IN"]
    assert len(ids) == 2 and ids[0] != ids[1]


def test_invalid_connectivity_is_not_silently_dropped(tmp_path):
    text = (KICAD / "simple_rc.kicad_net").read_text()
    text = text.replace('<node ref="R1" pin="1" pinfunction="~" pintype="passive"/>',
                        '<node ref="R1" pin="1" pinfunction="~" pintype="passive"/><node ref="Ghost" pin="1"/>')
    path = tmp_path / "bad.kicad_net"
    path.write_text(text)
    ir = SchematicExtractor().extract(path)
    assert any(e["code"] == "CONNECTIVITY_UNKNOWN_COMPONENT" for e in ir.validation_results["errors"])
    assert ir.validation_results["status"] == "error"


def test_pin_mapped_to_two_nets_is_error(tmp_path):
    text = (KICAD / "simple_rc.kicad_net").read_text()
    text = text.replace('<net code="3" name="GND">', '<net code="3" name="GND"><node ref="R1" pin="1"/>')
    path = tmp_path / "double_pin.kicad_net"
    path.write_text(text)
    ir = SchematicExtractor().extract(path)
    assert any(e["code"] == "PIN_ASSIGNED_TO_MULTIPLE_NETS" for e in ir.validation_results["errors"])


def test_duplicate_component_references_are_reported(tmp_path):
    text = (KICAD / "simple_rc.kicad_net").read_text()
    text = text.replace('<comp ref="C1">', '<comp ref="R1">')
    path = tmp_path / "duplicate_refs.kicad_net"
    path.write_text(text)
    ir = SchematicExtractor().extract(path)
    assert any(e["code"] == "DUPLICATE_REFERENCE" for e in ir.validation_results["errors"])


def test_spice_extraction_reuses_parser_and_does_not_simulate():
    extractor = SchematicExtractor()
    ir = extractor.extract(SPICE / "hierarchy_example.cir")
    assert ir.source_information["simulation_executed"] is False
    assert ir.models["DTEST"]["type"].startswith("D")
    assert ir.hierarchy
    assert any(c["reference"].startswith("R_") for c in ir.components)
    assert ir.provenance["connectivity_basis"].startswith("explicit positional")


def test_spice_text_extraction_api():
    ir = SchematicExtractor().import_spice_text(".title test\nR1 in 0 1k\n.end", "inline.cir")
    assert ir.component("R1")["pins"][0]["node_name"] == "in"


def test_ltspice_asc_is_explicitly_not_supported(tmp_path):
    asc = tmp_path / "sample.asc"
    asc.write_text("Version 4\nSHEET 1 880 680\n")
    with pytest.raises(ImporterError, match="Native LTspice"):
        SchematicExtractor().extract(asc)


def test_circuit_reviewer_processes_kicad_xml_netlist(tmp_path):
    result = CircuitReviewer(output_dir=tmp_path, plot=False).process_file(
        KICAD / "simple_rc.kicad_net", analyses=[], save_json=False, plot=False
    )
    assert isinstance(result.circuit_ir, CircuitIR)
    assert result.circuit_ir.component("C1")["pins"][0]["node_name"] == "/OUT"


def test_exporter_rejects_nonfinite_values():
    extractor = SchematicExtractor()
    ir = extractor.import_spice_text(".title bad\nR1 a 0 1k\n.end")
    ir.parameters["BAD"] = float("nan")
    with pytest.raises((ValueError, CircuitJSONExportError)):
        extractor.to_json_dict(ir)


def test_ai_generated_input_provenance_is_only_recorded_when_supplied():
    extractor = SchematicExtractor()
    ir = extractor.extract(KICAD / "ai_generated_design.kicad_net",
                           ai_provenance={"generator": "external fixture", "model": "caller-supplied"})
    assert ir.provenance["external_generation"]["generator"] == "external fixture"
    ir2 = extractor.extract(KICAD / "ai_generated_design.kicad_net")
    assert "external_generation" not in ir2.provenance


def test_kicad_cli_adapter_uses_arg_list_variant_and_preserves_output(tmp_path):
    xml_text = (KICAD / "simple_rc.kicad_net").read_text()
    output_text = tmp_path / "fake_kicad_cli.py"
    output_text.write_text(textwrap.dedent(f'''\
        #!/usr/bin/env python3
        import pathlib, sys
        if sys.argv[1:] == ["--version"]:
            print("8.0.0")
            raise SystemExit(0)
        args = sys.argv[1:]
        out = pathlib.Path(args[args.index("--output") + 1])
        out.write_text({xml_text!r})
        print("exported")
    '''))
    output_text.chmod(output_text.stat().st_mode | stat.S_IXUSR)
    sch = tmp_path / "design.kicad_sch"
    sch.write_text("(kicad_sch)")
    generated = tmp_path / "out" / "design.kicad_net"
    result = KiCadCLIImporter(output_text, timeout=10).import_file(
        sch, variant="Proto", output_path=generated, preserve_netlist=True)
    assert generated.exists()
    assert result.metadata["kicad_version"] == "8.0.0"
    assert result.metadata["variant"] == "Proto"
    assert result.metadata["generated_netlist_preserved"] is True
    assert result.circuit.component("R1")
    # Matches KiCad's documented CLI form and uses an argv list, not shell text.
    command = result.metadata["cli_command"]
    assert command[1:4] == ["sch", "export", "netlist"]
    assert command[command.index("--format") + 1] == "kicadxml"
    assert command[command.index("--variant") + 1] == "Proto"
    assert command[-1] == str(sch.resolve())
    assert "--output" in command



def test_public_extractor_routes_native_schematic_through_kicad_cli(tmp_path):
    xml_text = (KICAD / "simple_rc.kicad_net").read_text()
    executable = tmp_path / "kicad-cli-test.py"
    executable.write_text(textwrap.dedent(f'''\
        #!/usr/bin/env python3
        import pathlib, sys
        if sys.argv[1:] == ["--version"]:
            print("9.0.1")
            raise SystemExit(0)
        args = sys.argv[1:]
        out = pathlib.Path(args[args.index("--output") + 1])
        out.write_text({xml_text!r})
    '''))
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    schematic = tmp_path / "design.kicad_sch"
    schematic.write_text("(kicad_sch)")
    output = tmp_path / "saved" / "design.kicad_net"
    extractor = SchematicExtractor(kicad_cli=executable)
    result = extractor.extract(schematic, variant="Prototype", generated_netlist_path=output,
                               preserve_generated_netlist=True)
    assert result.component("R1") is not None
    assert output.is_file()
    assert result.source_information["simulation_executed"] is False
    assert extractor.import_metadata["kicad_version"] == "9.0.1"
    assert extractor.import_metadata["variant"] == "Prototype"

def test_kicad_cli_unavailable_fails_cleanly(tmp_path):
    sch = tmp_path / "design.kicad_sch"
    sch.write_text("(kicad_sch)")
    with pytest.raises(ImporterError, match="kicad-cli was not found"):
        KiCadCLIImporter("/definitely/missing/kicad-cli").import_file(sch)


def test_kicad_port_records_survive_circuit_ir_and_json_export(tmp_path):
    source = tmp_path / "with_ports.kicad_net"
    source.write_text('''<?xml version="1.0"?>
<export><components><comp ref="R1"><value>1k</value><libsource lib="Device" part="R"/></comp></components>
<nets><net code="1" name="/IN"><node ref="R1" pin="1"/></net></nets>
<ports><port name="/IN" type="input"><pin num="1"/></port></ports></export>''', encoding="utf-8")
    extractor = SchematicExtractor()
    circuit = extractor.extract(source)
    document = extractor.to_json_dict(circuit)
    assert document["ports"] == [{"attributes": {"name": "/IN", "type": "input"},
                                  "children": [{"tag": "pin", "attributes": {"num": "1"}, "text": None}]}]
    assert document["validation"]["electrical_design_assessed"] is False
