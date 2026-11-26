from pathlib import Path

import numpy as np

from circhecker import CircuitReviewer, SchematicExtractor

ROOT = Path(__file__).resolve().parents[1]


def test_kicad_xml_values_bridge_into_numerical_transient(tmp_path):
    source = ROOT / "tests" / "fixtures" / "rc_lowpass_pulse.kicad_net"
    circuit = SchematicExtractor().extract(source)

    parameters = {item["reference"]: item["parameters"] for item in circuit.components}
    assert parameters["R1"]["value"] == 1000.0
    assert parameters["C1"]["value"] == 1e-6
    assert parameters["V1"]["waveform"]["kind"] == "PULSE"

    circuit.analysis_configuration["transient"] = {"step_s": 1e-5, "stop_s": 0.01}
    result = CircuitReviewer(output_dir=tmp_path, plot=False).process_ir(
        circuit, analyses="transient", save_json=True, plot=False
    )
    assert result.analysis_results["transient"]["status"] == "success"
    data = result.analysis_results["transient"]["data"]
    assert len(data["time_s"]) > 100
    assert np.ptp(data["output_V"]) > 1.0
    assert (tmp_path / "rc_lowpass_pulse_analysis.json").exists()
