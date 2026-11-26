from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pytest

from circhecker import CircuitReviewer, SCHEMA_VERSION
from circhecker.importers.base import ImporterError
from circhecker.parser.units import parse_spice_number

ROOT = Path(__file__).resolve().parents[1]
NETLISTS = ROOT / "tests" / "fixtures"
DIODE_FIXTURE = ROOT / "tests" / "fixtures" / "diode_forward_reverse.cir"


def test_spice_suffixes_and_parameter_expression():
    assert parse_spice_number("1m") == pytest.approx(1e-3)
    assert parse_spice_number("2MEG") == pytest.approx(2e6)
    assert parse_spice_number("3.3kOhm") == pytest.approx(3300)
    assert parse_spice_number("{RBASE*2}", {"RBASE": 4700}) == pytest.approx(9400)


def test_ir_multiterminal_pins_are_preserved():
    text = """.title device pin test\nVCC vcc 0 12\nQ1 c b e QN\nM1 d g s 0 MN\nE1 out 0 ctrl 0 5\n.model QN NPN (BF=100)\n.model MN NMOS (VTO=1 KP=1m)\n.end\n"""
    reviewer = CircuitReviewer(output_dir="/tmp/circhecker-test-reports", plot=False)
    ir = reviewer.process_netlist_text(text, analyses=[], save_json=False, plot=False).circuit_ir
    assert ir.metadata["schema_version"] == SCHEMA_VERSION
    pin_counts = {c["reference"]: len(c["pins"]) for c in ir.components}
    assert pin_counts["Q1"] == 3
    assert pin_counts["M1"] == 4
    assert pin_counts["E1"] == 4


def test_linear_voltage_divider_matches_analytic_reference(tmp_path):
    text = """.title divider regression\nV1 in 0 10\nR1 in out 1k\nR2 out 0 1k\n.end\n"""
    result = CircuitReviewer(output_dir=tmp_path, plot=True).process_netlist_text(text, save_json=True)
    assert result.analysis_results["dc_operating_point"]["status"] == "success"
    volts = result.analysis_results["dc_operating_point"]["data"]["node_voltages_V"]
    out_id = next(n["id"] for n in result.circuit_ir.nodes if n["name"] == "out")
    assert float(np.real(volts[out_id])) == pytest.approx(5.0, abs=1e-8)
    assert result.analysis_results["ac_sweep"]["status"] == "success"
    assert result.analysis_results["transient"]["status"] == "not_applicable"
    report = json.loads((tmp_path / "divider_regression_analysis.json").read_text())
    assert report["circuit_summary"]["status"] == "success"
    assert report["analysis_results"]["transient"]["status"] == "not_applicable"
    assert (tmp_path / "divider_regression_analysis.png").is_file()
    assert report["artifacts"]["plot"].endswith("divider_regression_analysis.png")


def test_rc_cutoff_and_waveform_transient(tmp_path):
    result = CircuitReviewer(output_dir=tmp_path, plot=False).process_file(NETLISTS / "06_rc_low_pass.cir", save_json=False)
    ac = result.analysis_results["ac_sweep"]["data"]
    freq = np.asarray(ac.get("frequency", ac.get("frequency_Hz")), dtype=float)
    mag = np.asarray(ac["magnitude_dB"], dtype=float)
    assert result.analysis_results["ac_sweep"]["status"] == "success"
    # The expected -3 dB corner is about 15.9 Hz; the coarse logarithmic sweep brackets it.
    near = np.argmin(np.abs(mag + 3.0103))
    assert 5 < freq[near] < 40
    assert result.analysis_results["transient"]["status"] == "success"


def test_nonlinear_diode_operating_point_kcl(tmp_path):
    result = CircuitReviewer(output_dir=tmp_path, plot=False).process_file(DIODE_FIXTURE, save_json=False)
    assert result.summary()["backend"] == "nonlinear_analog"
    op = result.numerical_metrics["component_electrical_metrics"]
    assert op["kcl"]["is_balanced"]
    assert op["kcl"]["max_residual_A"] < 1e-6


def test_one_integrated_example_runs_and_exports_full_analysis(tmp_path):
    examples = ROOT / "examples" / "schematics"
    files = sorted(examples.glob("*.cir"))
    assert [path.name for path in files] == ["integrated_ct_supply_oscillator_cs_mosfet.cir"]
    source = files[0]
    result = CircuitReviewer(output_dir=tmp_path, plot=False).process_file(source, save_json=True, plot=False)
    assert result.status == "success"
    assert result.analysis_results["dc_operating_point"]["status"] == "success"
    assert result.analysis_results["dc_sweep"]["status"] == "success"
    assert result.analysis_results["ac_sweep"]["status"] == "success"
    assert result.analysis_results["transient"]["status"] == "success"
    assert (tmp_path / "integrated_ct_supply_oscillator_cs_mosfet_analysis.json").exists()


def test_complete_notebook_suite_covers_the_public_workflows():
    import nbformat
    paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
    expected = [
        "01_integrated_signal_chain.ipynb",
        "02_analog_circuits.ipynb",
        "03_semiconductor_amplifiers.ipynb",
        "04_digital_logic.ipynb",
        "05_ic_subcircuits.ipynb",
    ]
    assert [path.name for path in paths] == expected
    all_code = []
    all_markdown = []
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        all_code.extend(cell.source for cell in notebook.cells if cell.cell_type == "code")
        all_markdown.extend(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        assert code_cells, path.name
        assert all(cell.execution_count is not None for cell in code_cells), path.name
        assert any(
            output.output_type in {"display_data", "execute_result"}
            and "image/png" in output.get("data", {})
            for cell in code_cells for output in cell.get("outputs", [])
        ), f"No embedded plot output in {path.name}"
        assert all(
            not any(output.output_type == "error" for output in cell.get("outputs", []))
            for cell in code_cells
        )
    code = "\n".join(all_code)
    markdown = "\n".join(all_markdown).lower()
    for api in ("CircuitReviewer", "SchematicExtractor", "SignalAnalyzer", "run_example"):
        assert api in code
    for topic in ("analog", "digital", "semiconductor", "common-source", "subcircuit"):
        assert topic in markdown
    assert "integrated_circuit_analysis.json" in markdown


def test_catalog_entries_have_existing_circuit_sources_and_valid_analysis_lists():
    from examples.catalog import list_examples
    examples = list_examples()
    assert len(examples) >= 10
    identifiers = [item["id"] for item in examples]
    assert len(identifiers) == len(set(identifiers))
    for item in examples:
        assert Path(item["absolute_path"]).is_file(), item["id"]
        assert item["required_analyses"]


def test_kicad_xml_netlist_has_explicit_pin_connectivity(tmp_path):
    path = tmp_path / "minimal.kicad_net"
    path.write_text('''<?xml version="1.0"?><export><components><comp ref="R1"><value>1k</value><libsource><lib>Device</lib><part>R</part></libsource></comp></components><nets><net name="GND"><node ref="R1" pin="2" pinfunction="K"/></net><net name="/IN"><node ref="R1" pin="1" pinfunction="A"/></net></nets></export>''', encoding="utf-8")
    result = CircuitReviewer(output_dir=tmp_path, plot=False).process_schematic(path, analyses=[], save_json=False, plot=False)
    assert result.circuit_ir.source_information["format"] == "KiCad XML netlist"
    pins = result.circuit_ir.component("R1")["pins"]
    assert len(pins) == 2
    assert {p["number"] for p in pins} == {"1", "2"}


def test_native_kicad_schematic_fails_explicitly(tmp_path):
    path = tmp_path / "demo.kicad_sch"
    path.write_text("(kicad_sch (version 20231120))", encoding="utf-8")
    with pytest.raises(ImporterError, match="kicad-cli was not found"):
        CircuitReviewer().process_schematic(path)


def test_jfet_negative_vds_is_flagged_as_physical_inconsistency(tmp_path):
    text = """* Reversed-polarity n-channel JFET bias regression
VDD vdd 0 5
R1 vdd s 1k
R2 d 0 1k
VGG g 0 5
J1 d g s JMOD
.model JMOD NJF (VTO=-4 BETA=5m)
.end
"""
    result = CircuitReviewer(output_dir=tmp_path, plot=False).process_netlist_text(text, save_json=False)
    device = result.numerical_metrics["component_electrical_metrics"]["components"]["J1"]
    assert device["VDS_V"] < 0
    assert device["validation_status"] == "physically_inconsistent_drain_source_polarity"
    assert any(w["code"] == "JFET_NEGATIVE_VDS" for w in result.warnings)


def test_exp_waveform_is_explicitly_unsupported_for_transient(tmp_path):
    text = """* EXP source must not be treated as a DC value
V1 in 0 EXP(0 5 1m 1u 1u 0 0)
R1 in out 1k
D1 out 0 DM
.model DM D (IS=1e-14 N=1)
.tran 1u 5m
.end
"""
    result = CircuitReviewer(output_dir=tmp_path, plot=False).process_netlist_text(text, save_json=False)
    transient = result.analysis_results["transient"]
    assert transient["status"] == "unsupported"
    assert "EXP" in transient["reason"]


def test_process_file_accepts_kicad_xml_netlist(tmp_path):
    path = tmp_path / "dispatch.kicad_net"
    path.write_text('''<?xml version="1.0"?><export><components><comp ref="R1"><value>1k</value></comp></components><nets><net name="GND"><node ref="R1" pin="2"/></net><net name="/IN"><node ref="R1" pin="1"/></net></nets></export>''', encoding="utf-8")
    result = CircuitReviewer(output_dir=tmp_path, plot=False).process_file(path, analyses=[], save_json=False, plot=False)
    assert result.circuit_ir.source_information["format"] == "KiCad XML netlist"
    assert len(result.circuit_ir.component("R1")["pins"]) == 2


def test_linear_rlc_does_not_report_irrelevant_nonlinear_adapter_warning():
    result = CircuitReviewer(plot=False).process_file(
        ROOT / "examples" / "circuits" / "analog" / "series_rlc.cir",
        analyses=["dc_operating_point", "ac_sweep"],
        save_json=False,
        plot=False,
    )
    assert result.status == "success"
    assert not any(item.get("code") == "ADAPTER_WARNING" for item in result.warnings if isinstance(item, dict))
