from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from circhecker import (
    ACSweepSpec,
    AnalysisIR,
    AnalysisSpec,
    CircuitReviewer,
    SimulationResult,
    TransientSpec,
)
from circhecker.schematic import SchematicExtractor
from circhecker.reporting.simulation import AnalysisResult


def _circuit():
    return SchematicExtractor().import_spice_text(
        ".title AnalysisIR example\n"
        "V1 in 0 PULSE(0 5 0 1u 1u 1m 2m) AC 1\n"
        "R1 in out 1k\n"
        "C1 out 0 100n\n"
        ".tran 10u 2m\n.end\n",
        source_name="analysis_ir_example.cir",
    )


def test_circuit_fingerprint_ignores_file_location_and_title():
    parser = SchematicExtractor()
    first = parser.import_spice_text(".title first\nR1 a 0 1k\n.end\n", source_name="a.cir")
    second = parser.import_spice_text(".title second\nR1 a 0 1k\n.end\n", source_name="b.cir")
    assert first.fingerprint() == second.fingerprint()
    second.component("R1")["parameters"]["value"] = 2200.0
    assert first.fingerprint() != second.fingerprint()


def test_analysis_ir_is_separate_from_source_and_round_trips(tmp_path):
    circuit = _circuit()
    source_transient = dict(circuit.analysis_configuration["transient"])
    request = AnalysisSpec(
        analyses=("dc_operating_point", "ac_sweep", "transient"),
        ac_sweep=ACSweepSpec(start_Hz=20, stop_Hz=20_000, points=31),
        transient=TransientSpec(stop_s=1e-3, step_s=2e-6, method="trapezoidal"),
        output_node="out",
    )
    plan = AnalysisIR.from_request(circuit, analysis_spec=request)

    assert plan.circuit_id == circuit.fingerprint()
    assert plan.outputs["positive_node"] == "out"
    assert plan.analysis_configuration["transient"]["stop_s"] == pytest.approx(1e-3)
    assert plan.source_analysis_configuration["transient"] == source_transient
    assert circuit.analysis_configuration["transient"] == source_transient

    document = plan.to_dict()
    restored = AnalysisIR.from_mapping(json.loads(json.dumps(document)))
    assert restored.analysis_id == plan.analysis_id
    assert restored.to_dict() == document
    assert json.loads(plan.to_json()) == document
    plan_path = tmp_path / "analysis_ir.json"
    plan.to_json(plan_path)
    assert json.loads(plan_path.read_text(encoding="utf-8")) == document
    from jsonschema import validate
    schema_path = Path(__file__).parents[1] / "circhecker" / "schemas" / "analysis_ir.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validate(document, schema)


def test_analysis_ir_rejects_tampered_identifier_and_invalid_names():
    circuit = _circuit()
    with pytest.raises(ValueError, match="Unknown analysis"):
        AnalysisIR.from_request(circuit, analyses=("dc_operating_point", "not_real"))
    document = AnalysisIR.from_request(circuit, analyses="dc_operating_point").to_dict()
    document["analysis_configuration"]["dc_sweep"] = {"source": "V1"}
    with pytest.raises(ValueError, match="analysis_id does not match"):
        AnalysisIR.from_mapping(document)


def test_analysis_ir_rejects_a_plan_for_another_circuit():
    first = _circuit()
    other = SchematicExtractor().import_spice_text(".title other\nR9 a 0 2k\n.end\n")
    plan = AnalysisIR.from_request(first, analyses=("dc_operating_point",))
    with pytest.raises(ValueError, match="different CircuitIR fingerprint"):
        from circhecker.execution.engine import CircuitExecutionEngine
        CircuitExecutionEngine(other, analysis_ir=plan)


def test_simulation_result_has_stable_shape_units_and_diagnostics(tmp_path):
    result = CircuitReviewer(output_dir=tmp_path, plot=False).process_ir(
        _circuit(),
        analysis_spec=AnalysisSpec(
            analyses=("dc_operating_point", "ac_sweep", "transient"),
            ac_sweep=ACSweepSpec(start_Hz=10, stop_Hz=100_000, points=40),
            transient=TransientSpec(stop_s=1e-3, step_s=5e-6),
            output_node="out",
        ),
        save_json=True,
        plot=False,
    )

    assert isinstance(result.analysis_ir, AnalysisIR)
    assert isinstance(result.simulation_result, SimulationResult)
    document = result.to_dict()
    simulation = document["simulation_result"]
    assert simulation["schema"] == "SimulationResult"
    from jsonschema import validate
    schema_path = Path(__file__).parents[1] / "circhecker" / "schemas" / "simulation_result.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validate(simulation, schema)
    assert simulation["circuit_id"] == result.circuit_ir.fingerprint()
    assert simulation["analysis_id"] == result.analysis_ir.analysis_id
    assert simulation["analyses"]["ac_sweep"]["units"]["frequency_Hz"] == "Hz"
    assert simulation["analyses"]["transient"]["units"]["time_s"] == "s"
    assert simulation["analyses"]["dc_operating_point"]["status"] == "success"
    assert "[OK] DC operating point" in result.human_summary()

    report_path = result.artifacts["json"]
    saved = json.loads(Path(report_path).read_text(encoding="utf-8"))
    assert saved["analysis_ir"]["analysis_id"] == result.analysis_ir.analysis_id
    assert saved["simulation_result"]["schema_version"] == "1.0.0"
    simulation_path = tmp_path / "simulation_result.json"
    result.simulation_result.to_json(simulation_path)
    assert json.loads(simulation_path.read_text(encoding="utf-8"))["analysis_id"] == result.analysis_ir.analysis_id
    assert np.isfinite(result.analysis_results["ac_sweep"]["data"]["frequency_Hz"]).all()


def test_empty_analysis_plan_supports_parse_and_validate_only():
    circuit = SchematicExtractor().import_spice_text(".title parse only\nR1 in 0 1k\n.end\n")
    result = CircuitReviewer(plot=False).process_ir(circuit, analyses=[], save_json=False, plot=False)
    assert result.analysis_ir.analyses == ()
    assert result.status == "unsupported"
    assert result.simulation_result.requested_analyses == ()


def test_standard_analysis_data_adds_stable_names_for_legacy_backend_fields():
    ac = AnalysisResult.from_mapping("ac_sweep", {
        "status": "success",
        "data": {
            "frequency": np.asarray([10.0, 100.0]),
            "magnitude": np.asarray([1.0, 0.5]),
            "phase_degrees": np.asarray([0.0, -45.0]),
        },
    })
    assert ac.data["frequency_Hz"].tolist() == [10.0, 100.0]
    assert ac.data["magnitude_dB"][0] == pytest.approx(0.0)
    assert ac.data["phase_deg"].tolist() == [0.0, -45.0]

    transient = AnalysisResult.from_mapping("transient", {
        "status": "success",
        "data": {
            "time": np.asarray([0.0, 1e-3]),
            "outputs": np.asarray([[0.0], [5.0]]),
        },
    })
    assert transient.data["time_s"].tolist() == [0.0, 1e-3]
    assert transient.data["output_V"].tolist() == [0.0, 5.0]
