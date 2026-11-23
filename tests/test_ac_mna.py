"""Independent analytical tests for the generalized complex sparse AC MNA engine."""
from __future__ import annotations

import numpy as np
import pytest

from circhecker import CircuitReviewer
from circhecker.parser.spice import parse_spice_text
from circhecker.solver import frequency_sweep, solve_ac


def _solve(netlist: str, frequency: float, output: tuple[str, str] = ("out", "0")):
    ir = parse_spice_text(netlist)
    names = {node["name"]: node["id"] for node in ir.nodes}
    result = solve_ac(ir, frequencies_hz=[frequency], output_nodes=(names[output[0]], names[output[1]]))
    assert result.status == "success", result.reason
    return ir, result.data


def _node(ir, data, name: str) -> complex:
    node_id = next(n["id"] for n in ir.nodes if n["name"] == name)
    return complex(data["node_voltages_V"][str(name)][0] if str(name) in data["node_voltages_V"]
                   else data["node_voltages_V"][name][0])


def test_rc_lowpass_cutoff_magnitude_and_phase():
    rc = 1e3 * 1e-6
    fc = 1.0 / (2.0 * np.pi * rc)
    text = "V1 in 0 AC 1\nR1 in out 1k\nC1 out 0 1u\n.end"
    ir, data = _solve(text, fc)
    expected = 1.0 / (1.0 + 1j)
    assert _node(ir, data, "out") == pytest.approx(expected, rel=1e-10)
    assert data["magnitude"][0] == pytest.approx(1 / np.sqrt(2), rel=1e-10)
    assert data["phase_deg"][0] == pytest.approx(-45.0, abs=1e-8)
    assert data["solver_diagnostics"]["failed_frequency_count"] == 0


def test_rc_highpass_cutoff_magnitude_and_phase():
    fc = 1.0 / (2.0 * np.pi * 1e3 * 1e-6)
    text = "V1 in 0 AC 1\nC1 in out 1u\nR1 out 0 1k\n.end"
    ir, data = _solve(text, fc)
    assert _node(ir, data, "out") == pytest.approx(0.5 + 0.5j, rel=1e-10)
    assert data["magnitude"][0] == pytest.approx(1 / np.sqrt(2), rel=1e-10)
    assert data["phase_deg"][0] == pytest.approx(45.0, abs=1e-8)


def test_rl_inductive_output_cutoff_magnitude_and_phase():
    r, inductance = 1e3, 1e-3
    fc = r / (2.0 * np.pi * inductance)
    text = f"V1 in 0 AC 1\nR1 in out {r}\nL1 out 0 {inductance}\n.end"
    ir, data = _solve(text, fc)
    assert _node(ir, data, "out") == pytest.approx(0.5 + 0.5j, rel=1e-10)
    assert data["branch_currents_A"]["L1"][0] == pytest.approx(0.0005 - 0.0005j, rel=1e-10)


def test_series_rlc_resonance_and_voltage_source_branch_current():
    resistance, inductance, capacitance = 10.0, 10e-3, 10e-6
    f0 = 1.0 / (2.0 * np.pi * np.sqrt(inductance * capacitance))
    text = (
        "V1 in 0 AC 1\n"
        f"R1 in n1 {resistance}\nL1 n1 n2 {inductance}\nC1 n2 0 {capacitance}\n.end"
    )
    _, data = _solve(text, f0, output=("in", "0"))
    assert abs(data["branch_currents_A"]["V1"][0]) == pytest.approx(1 / resistance, rel=1e-9)
    assert abs(data["branch_currents_A"]["L1"][0]) == pytest.approx(1 / resistance, rel=1e-9)


def test_parallel_rlc_resonance_source_current_cancellation():
    inductance, capacitance, resistance = 10e-3, 10e-6, 1e3
    f0 = 1.0 / (2.0 * np.pi * np.sqrt(inductance * capacitance))
    text = (
        "V1 in 0 AC 1\n"
        f"R1 in 0 {resistance}\nL1 in 0 {inductance}\nC1 in 0 {capacitance}\n.end"
    )
    _, data = _solve(text, f0, output=("in", "0"))
    assert data["branch_currents_A"]["V1"][0] == pytest.approx(-1e-3 + 0j, rel=1e-9)


def test_floating_voltage_source_and_complex_branch_current():
    text = "V1 a b AC 1\nR1 a 0 1k\nR2 b 0 1k\n.end"
    ir = parse_spice_text(text)
    ids = {n["name"]: n["id"] for n in ir.nodes}
    result = solve_ac(ir, [1000.0], output_nodes=(ids["a"], ids["b"]))
    assert result.status == "success"
    va = result.data["node_voltages_V"]["a"][0]
    vb = result.data["node_voltages_V"]["b"][0]
    assert va == pytest.approx(0.5 + 0j)
    assert vb == pytest.approx(-0.5 + 0j)
    assert result.data["branch_currents_A"]["V1"][0] == pytest.approx(-0.5e-3 + 0j)


def test_vcvs_and_vccs_complex_control_stamps():
    vcvs = "V1 cp 0 AC 1\nE1 out 0 cp 0 2\nR1 out 0 1k\n.end"
    ir, data = _solve(vcvs, 100.0)
    assert _node(ir, data, "out") == pytest.approx(2 + 0j)

    vccs = "V1 cp 0 AC 1\nG1 out 0 cp 0 1m\nR1 out 0 1k\n.end"
    ir, data = _solve(vccs, 100.0)
    assert _node(ir, data, "out") == pytest.approx(-1 + 0j)


def test_cccs_and_ccvs_use_explicit_voltage_source_branch_sensing():
    cccs = (
        "V1 in 0 AC 1\nR1 in sense 1k\nVsen sense 0 0\n"
        "F1 out 0 Vsen 2\nR2 out 0 1k\n.end"
    )
    ir, data = _solve(cccs, 100.0)
    assert _node(ir, data, "out") == pytest.approx(-2 + 0j)
    assert data["branch_currents_A"]["Vsen"][0] == pytest.approx(1e-3 + 0j)

    ccvs = (
        "V1 in 0 AC 1\nR1 in sense 1k\nVsen sense 0 0\n"
        "H1 out 0 Vsen 1k\nR2 out 0 1k\n.end"
    )
    ir, data = _solve(ccvs, 100.0)
    assert _node(ir, data, "out") == pytest.approx(1 + 0j)


def test_multiple_explicit_independent_ac_sources_are_superposed():
    text = (
        "V1 in1 0 AC 1 0\nV2 in2 0 AC 2 90\n"
        "R1 in1 out 1k\nR2 in2 out 1k\nR3 out 0 1k\n.end"
    )
    ir, data = _solve(text, 100.0)
    expected = (1.0 + 2.0j) / 3.0
    assert _node(ir, data, "out") == pytest.approx(expected)
    assert data["reference_source"] is None
    assert data["transfer_function"] is None
    assert data["magnitude_reference"].startswith("output voltage")


def test_frequency_sweep_lin_dec_and_octave_conventions():
    assert frequency_sweep({"scale": "lin", "points": 3, "start_Hz": 1, "stop_Hz": 9}).tolist() == [1, 5, 9]
    dec = frequency_sweep({"scale": "dec", "points": 10, "start_Hz": 1, "stop_Hz": 100})
    assert len(dec) == 21
    assert dec[0] == pytest.approx(1.0) and dec[-1] == pytest.approx(100.0)
    octaves = frequency_sweep({"scale": "oct", "points": 4, "start_Hz": 1, "stop_Hz": 16})
    assert len(octaves) == 17
    with pytest.raises(ValueError, match="strictly positive"):
        frequency_sweep({"scale": "dec", "points": 10, "start_Hz": 0, "stop_Hz": 100})


def test_singular_ac_frequency_point_is_explicitly_failed():
    text = "V1 out 0 AC 1\nV2 out 0 AC 2\nR1 out 0 1k\n.end"
    ir = parse_spice_text(text)
    out = next(n["id"] for n in ir.nodes if n["name"] == "out")
    result = solve_ac(ir, [100.0], output_nodes=(out, 0))
    assert result.status == "failed"
    assert result.data["solver_diagnostics"]["failed_frequency_count"] == 1
    assert result.data["solver_diagnostics"]["failed_frequencies_Hz"] == [100.0]
    assert result.data["frequency_diagnostics"][0]["status"] == "failed"


def test_public_circuit_reviewer_uses_generalized_ac_mna(tmp_path):
    text = (
        "V1 in 0 AC 1\nR1 in out 1k\nC1 out 0 1u\n"
        ".ac lin 3 0 159.15494309189535\n.end"
    )
    result = CircuitReviewer(output_dir=tmp_path, plot=False).process_netlist_text(
        text, analyses=["ac_sweep"], save_json=False,
    )
    ac = result.analysis_results["ac_sweep"]
    assert ac["status"] == "success", ac["reason"]
    assert ac["data"]["solver_diagnostics"]["backend"] == "circhecker.solver.ac"
    assert len(ac["data"]["frequency_Hz"]) == 3
    assert ac["data"]["frequency_diagnostics"][-1]["status"] == "success"


def test_ac_result_is_valid_json_with_complex_values(tmp_path):
    import json
    text = "V1 in 0 AC 1 30\nR1 in out 1k\nC1 out 0 1u\n.end"
    result = CircuitReviewer(output_dir=tmp_path, plot=False).process_netlist_text(
        text, analyses=["ac_sweep"], save_json=True,
    )
    path = next(tmp_path.glob("*_analysis.json"))
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["analysis_results"]["ac_sweep"]["status"] == "success"
    value = report["analysis_results"]["ac_sweep"]["data"]["node_voltages_V"]["out"][0]
    assert set(value) >= {"real", "imag", "magnitude", "phase_deg"}


def test_independent_ac_current_source_has_correct_injection_polarity():
    text = "I1 out 0 AC 1m\nR1 out 0 1k\n.end"
    ir, data = _solve(text, 50.0)
    assert _node(ir, data, "out") == pytest.approx(-1 + 0j)
    assert data["component_currents_A"]["I1"][0] == pytest.approx(1e-3 + 0j)
    assert data["transfer_function"][0] == pytest.approx(-1000 + 0j)


def test_large_complex_ac_system_uses_sparse_solver():
    lines = []
    for index in range(70):
        lines.extend((f"I{index} n{index} 0 AC 1m", f"R{index} n{index} 0 1k"))
    result = solve_ac(parse_spice_text("\n".join(lines) + "\n.end"), [100.0])
    assert result.status == "success"
    diagnostics = result.data["frequency_diagnostics"][0]
    assert diagnostics["method"] == "scipy_sparse_lu"
    assert diagnostics["matrix_dimension"] == 70
    assert diagnostics["matrix_nnz"] == 70
    assert result.data["node_voltages_V"]["n0"][0] == pytest.approx(-1 + 0j)


def test_ac_sweep_preserves_good_frequency_points_when_one_point_fails():
    # A lossless series LC branch has a singular ideal resonance at omega=1 rad/s.
    inductance, capacitance = 1.0, 1.0
    f0 = 1.0 / (2.0 * np.pi)
    text = (
        "V1 in 0 AC 1\n"
        f"L1 in mid {inductance}\nC1 mid 0 {capacitance}\n.end"
    )
    ir = parse_spice_text(text)
    out = next(n["id"] for n in ir.nodes if n["name"] == "in")
    result = solve_ac(ir, [0.5*f0, f0, 2.0*f0], output_nodes=(out, 0))
    assert result.status == "failed"
    statuses = [x["status"] for x in result.data["frequency_diagnostics"]]
    assert statuses.count("success") == 2
    assert statuses.count("failed") == 1
    assert result.data["output_voltage_V"][0] is not None
    assert result.data["output_voltage_V"][1] is None
    assert result.data["output_voltage_V"][2] is not None


def test_ac_controlled_sources_accept_non_grounded_differential_control_nodes():
    vcvs = (
        "V1 cp 0 AC 1\nV2 cn 0 AC 0.25\n"
        "E1 out 0 cp cn 2\nR1 out 0 1k\n.end"
    )
    ir, data = _solve(vcvs, 100.0)
    assert _node(ir, data, "out") == pytest.approx(1.5 + 0j)

    vccs = (
        "V1 cp 0 AC 1\nV2 cn 0 AC 0.25\n"
        "G1 out 0 cp cn 1m\nR1 out 0 1k\n.end"
    )
    ir, data = _solve(vccs, 100.0)
    assert _node(ir, data, "out") == pytest.approx(-0.75 + 0j)


def test_public_api_does_not_hide_singular_ac_frequency_failure(tmp_path):
    text = (
        "V1 out 0 AC 1\nV2 out 0 AC 2\nR1 out 0 1k\n"
        ".ac lin 1 100 100\n.end"
    )
    result = CircuitReviewer(output_dir=tmp_path, plot=False).process_netlist_text(
        text, analyses=["ac_sweep"], save_json=False,
    )
    assert result.analysis_results["ac_sweep"]["status"] == "failed"
    assert result.status == "failed"
    assert any(error.get("code") == "AC_MNA_FREQUENCY_FAILURE" for error in result.errors)
