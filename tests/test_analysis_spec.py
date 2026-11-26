from __future__ import annotations

import json

import numpy as np
import pytest

from circhecker import (
    ACSweepSpec,
    AnalysisSpec,
    CircuitReviewer,
    DCSweepSpec,
    TransientSpec,
)
from circhecker.schematic import SchematicExtractor
from circhecker.validation.units import validate_circuit_units


def test_analysis_spec_rejects_invalid_or_inconsistent_requests():
    with pytest.raises(ValueError, match="Unknown analysis"):
        AnalysisSpec(analyses=("dc_operating_point", "imaginary_analysis"))
    with pytest.raises(ValueError, match="requires 'ac_sweep'"):
        AnalysisSpec(analyses=("dc_operating_point",), ac_sweep=ACSweepSpec())
    with pytest.raises(ValueError, match="step must be non-zero"):
        DCSweepSpec(source="V1", start=0, stop=5, step=0)
    with pytest.raises(ValueError, match="start_Hz > 0"):
        ACSweepSpec(start_Hz=0, stop_Hz=1e3, scale="log")
    with pytest.raises(ValueError, match="step_s <= stop_s"):
        TransientSpec(stop_s=1e-3, step_s=2e-3)
    with pytest.raises(ValueError, match="output_node must be supplied"):
        AnalysisSpec(analyses=("ac_sweep",), output_negative_node="return")


def test_analysis_spec_overrides_source_transient_without_mutating_circuit(tmp_path):
    netlist = """.title explicit analysis request
V1 in 0 PULSE(0 1 0 1u 1u 100u 200u)
R1 in out 1k
C1 out 0 1u
.tran 10u 100u
.end
"""
    circuit = SchematicExtractor().import_spice_text(netlist)
    source_configuration = dict(circuit.analysis_configuration["transient"])
    spec = AnalysisSpec(
        analyses=("transient",),
        transient=TransientSpec(stop_s=200e-6, step_s=5e-6, method="trapezoidal"),
        output_node="out",
    )
    result = CircuitReviewer(output_dir=tmp_path, plot=False).process_ir(
        circuit, save_json=True, plot=False, analysis_spec=spec,
    )

    assert result.analysis_results["transient"]["status"] == "success"
    time = np.asarray(result.analysis_results["transient"]["data"]["time_s"])
    assert time[-1] == pytest.approx(200e-6)
    assert circuit.analysis_configuration["transient"] == source_configuration
    assert result.analysis_configuration["transient"]["stop_s"] == pytest.approx(200e-6)
    assert result.simulation_options["METHOD"] == "trap"
    assert result.execution_summary["requested_analyses"] == ["transient"]
    report_path = tmp_path / "explicit_analysis_request_analysis.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["analysis_configuration"]["transient"]["stop_s"] == pytest.approx(200e-6)
    assert report["source_analysis_configuration"]["transient"] == source_configuration


def test_analysis_spec_ac_sweep_is_used_for_one_execution(tmp_path):
    netlist = """.title explicit ac request
V1 in 0 AC 1
R1 in out 1k
C1 out 0 1u
.end
"""
    result = CircuitReviewer(output_dir=tmp_path, plot=False).process_netlist_text(
        netlist,
        save_json=False,
        plot=False,
        analysis_spec=AnalysisSpec(
            analyses=("ac_sweep",),
            ac_sweep=ACSweepSpec(start_Hz=10, stop_Hz=1e4, points=7, scale="log"),
            output_node="out",
        ),
    )
    assert result.analysis_results["ac_sweep"]["status"] == "success"
    freq = result.analysis_results["ac_sweep"]["data"].get("frequency_Hz",
                                                            result.analysis_results["ac_sweep"]["data"].get("frequency"))
    assert len(freq) == 7
    assert float(freq[0]) == pytest.approx(10)
    assert float(freq[-1]) == pytest.approx(1e4)


def test_unit_validation_checks_declared_dimensions_and_passive_values():
    circuit = SchematicExtractor().import_spice_text(
        ".title unit test\nR1 in 0 10k\nC1 in 0 1u\n.end\n"
    )
    assert validate_circuit_units(circuit)["status"] == "complete"
    circuit.component("R1")["parameters"]["unit"] = "F"
    circuit.component("C1")["parameters"]["value"] = 0.0
    report = validate_circuit_units(circuit)
    codes = {item["code"] for item in report["diagnostics"]}
    assert report["status"] == "error"
    assert "COMPONENT_UNIT_DIMENSION_MISMATCH" in codes
    assert "PHYSICALLY_INVALID_COMPONENT_VALUE" in codes


def test_analysis_spec_from_mapping_builds_typed_nested_specs():
    spec = AnalysisSpec.from_mapping({
        "analyses": ["dc_operating_point", "ac_sweep"],
        "ac_sweep": {"start_Hz": 5, "stop_Hz": 500, "points": 11, "scale": "log"},
    })
    assert isinstance(spec.ac_sweep, ACSweepSpec)
    assert spec.analyses == ("dc_operating_point", "ac_sweep")


def test_dc_sweep_spec_drives_dc_sweep_and_preserves_normalized_results():
    netlist = ".title typed dc sweep\nV1 in 0 5\nR1 in out 1k\nR2 out 0 1k\n.end\n"
    result = CircuitReviewer(plot=False).process_netlist_text(
        netlist,
        save_json=False,
        plot=False,
        analysis_spec=AnalysisSpec(
            analyses=("dc_sweep",),
            dc_sweep=DCSweepSpec(source="V1", start=0, stop=10, step=1),
            output_node="out",
        ),
    )
    state = result.analysis_results["dc_sweep"]
    assert state["status"] == "success"
    assert state["data"]["sweep"].tolist() == pytest.approx(list(range(11)))
    assert state["data"]["voltage"].tolist() == pytest.approx([n / 2 for n in range(11)])
    assert state["reason"] is None


def test_reviewer_surfaces_unit_dimension_errors_in_validation():
    circuit = SchematicExtractor().import_spice_text(".title bad units\nR1 in 0 10k\n.end\n")
    circuit.component("R1")["parameters"]["unit"] = "F"
    result = CircuitReviewer(plot=False).process_ir(circuit, analyses=(), save_json=False, plot=False)
    assert result.circuit_ir.validation_results["unit_check"]["status"] == "error"
    assert any(item["code"] == "COMPONENT_UNIT_DIMENSION_MISMATCH" for item in result.errors)
    assert result.summary()["validation_status"] == "error"


def test_explicit_dc_sweep_source_does_not_silently_fall_back_to_another_source():
    netlist = ".title invalid sweep source\nV1 in 0 5\nR1 in out 1k\nR2 out 0 1k\n.end\n"
    result = CircuitReviewer(plot=False).process_netlist_text(
        netlist,
        save_json=False,
        plot=False,
        analysis_spec=AnalysisSpec(
            analyses=("dc_sweep",),
            dc_sweep=DCSweepSpec(source="V9", start=0, stop=10, step=1),
            output_node="out",
        ),
    )
    assert result.analysis_results["dc_sweep"]["status"] == "unsupported"
    assert "No compatible" in result.analysis_results["dc_sweep"]["reason"]
