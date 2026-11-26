"""Analytical and topology regressions for generalized sparse DC MNA."""
from __future__ import annotations

import numpy as np
import pytest

from circhecker.parser.spice import parse_spice_text
from circhecker.solver import (
    SingularMatrixError,
    UnsupportedElementError,
    solve_dc,
    supports_circuit_ir_dc,
)


def _solve(text: str):
    return solve_dc(parse_spice_text(text))


def _node(result, ir_text: str, name: str) -> float:
    ir = parse_spice_text(ir_text)
    node_id = next(n["id"] for n in ir.nodes if n["name"] == name)
    return result.node_voltages_V[node_id]


def test_sparse_mna_voltage_divider_and_branch_current():
    text = "V1 in 0 10\nR1 in out 1k\nR2 out 0 1k\n.end"
    result = _solve(text)
    assert result.node_voltages_V[2] == pytest.approx(5.0)
    assert result.branch_currents_A["V1"] == pytest.approx(-5e-3)
    assert result.diagnostics.converged
    assert result.diagnostics.relative_residual < 1e-12
    assert result.verification["kcl"]["is_balanced"]


def test_sparse_mna_floating_voltage_source_and_non_grounded_return():
    text = "V1 a b 5\nR1 a 0 1k\nR2 b 0 1k\n.end"
    ir = parse_spice_text(text)
    result = solve_dc(ir)
    node_ids = {n["name"]: n["id"] for n in ir.nodes}
    assert result.node_voltages_V[node_ids["a"]] == pytest.approx(2.5)
    assert result.node_voltages_V[node_ids["b"]] == pytest.approx(-2.5)
    assert result.branch_currents_A["V1"] == pytest.approx(-2.5e-3)


def test_sparse_mna_vcvs_accepts_differential_control_terminals():
    text = "V1 cp 0 3\nV2 cn 0 1\nE1 out 0 cp cn 2\nR1 out 0 1k\n.end"
    ir = parse_spice_text(text)
    result = solve_dc(ir)
    output_id = next(n["id"] for n in ir.nodes if n["name"] == "out")
    assert result.node_voltages_V[output_id] == pytest.approx(4.0)


def test_sparse_mna_vccs_accepts_differential_control_terminals():
    text = "V1 cp 0 3\nV2 cn 0 1\nG1 out 0 cp cn 1m\nR1 out 0 1k\n.end"
    ir = parse_spice_text(text)
    result = solve_dc(ir)
    output_id = next(n["id"] for n in ir.nodes if n["name"] == "out")
    assert result.node_voltages_V[output_id] == pytest.approx(-2.0)


def test_parser_and_solver_support_cccs_with_named_branch_current():
    text = (
        "V1 in 0 5\nR1 in sense 1k\nVsen sense 0 0\n"
        "F1 out 0 Vsen 2\nR2 out 0 1k\n.end"
    )
    ir = parse_spice_text(text)
    cccs = ir.component("F1")
    assert len(cccs["pins"]) == 2  # The control source is not a circuit node.
    assert cccs["parameters"]["control_source"] == "Vsen"
    out_id = next(n["id"] for n in ir.nodes if n["name"] == "out")
    result = solve_dc(ir)
    assert result.node_voltages_V[out_id] == pytest.approx(-10.0)
    assert result.branch_currents_A["Vsen"] == pytest.approx(5e-3)


def test_parser_and_solver_support_ccvs_with_named_branch_current():
    text = (
        "V1 in 0 5\nR1 in sense 1k\nVsen sense 0 0\n"
        "H1 out 0 Vsen 1k\nR2 out 0 1k\n.end"
    )
    ir = parse_spice_text(text)
    ccvs = ir.component("H1")
    assert len(ccvs["pins"]) == 2
    assert ccvs["parameters"]["control_source"] == "Vsen"
    out_id = next(n["id"] for n in ir.nodes if n["name"] == "out")
    result = solve_dc(ir)
    assert result.node_voltages_V[out_id] == pytest.approx(5.0)


def test_ideal_inductor_is_a_dc_short_with_branch_current():
    text = "V1 in 0 5\nR1 in out 1k\nL1 out 0 10m\n.end"
    ir = parse_spice_text(text)
    out_id = next(n["id"] for n in ir.nodes if n["name"] == "out")
    result = solve_dc(ir)
    assert result.node_voltages_V[out_id] == pytest.approx(0.0, abs=1e-12)
    assert result.branch_currents_A["L1"] == pytest.approx(5e-3)


def test_capacitor_is_open_in_dc_and_reported_as_such():
    text = "V1 in 0 5\nR1 in out 1k\nR2 out 0 1k\nC1 out 0 1u\n.end"
    result = _solve(text)
    assert "C1" in result.diagnostics.ignored_dc_elements
    assert any("open circuits in DC" in item for item in result.diagnostics.warnings)
    assert result.node_voltages_V[2] == pytest.approx(2.5)


def test_sparse_factorization_is_used_above_dense_threshold():
    lines = []
    for index in range(70):
        node = f"n{index}"
        lines.extend((f"I{index} {node} 0 1m", f"R{index} {node} 0 1k"))
    result = _solve("\n".join(lines) + "\n.end")
    assert result.diagnostics.matrix_dimension == 70
    assert result.diagnostics.method == "scipy_sparse_lu"
    assert result.diagnostics.matrix_nnz == 70
    assert result.diagnostics.condition_estimate == pytest.approx(1.0)
    assert "onenormest" in result.diagnostics.condition_estimate_method
    assert np.allclose(list(result.node_voltages_V.values())[1:], -1.0)


def test_singular_contradictory_voltage_sources_are_not_reported_as_success():
    text = "V1 out 0 5\nV2 out 0 10\nR1 out 0 1k\n.end"
    with pytest.raises(SingularMatrixError, match="rank deficient"):
        _solve(text)


def test_unsupported_components_are_explicitly_rejected():
    ir = parse_spice_text("V1 in 0 5\nD1 in 0 DM\n.model DM D (IS=1e-14)\n.end")
    supported, reasons = supports_circuit_ir_dc(ir)
    assert not supported
    assert any("D1" in reason for reason in reasons)
    with pytest.raises(UnsupportedElementError, match="D1"):
        solve_dc(ir)


def test_source_overrides_are_case_insensitive_and_reassemble_rhs():
    ir = parse_spice_text("Vdrive in 0 5\nR1 in out 1k\nR2 out 0 1k\n.end")
    result = solve_dc(ir, source_overrides={"vDRIVE": 20.0})
    out_id = next(n["id"] for n in ir.nodes if n["name"] == "out")
    assert result.node_voltages_V[out_id] == pytest.approx(10.0)


def test_public_circuit_reviewer_uses_new_sparse_mna_for_linear_dc(tmp_path):
    from circhecker import CircuitReviewer

    result = CircuitReviewer(output_dir=tmp_path, plot=False).process_netlist_text(
        "V1 in 0 10\nR1 in out 1k\nR2 out 0 1k\n.end",
        analyses=["dc_operating_point"], save_json=False,
    )
    assert result.execution_summary["backend"] == "sparse_mna_dc"
    assert result.analysis_results["dc_operating_point"]["status"] == "success"
    assert result.analysis_results["dc_operating_point"]["data"]["solver_diagnostics"]["backend"] == "circhecker.solver.mna"


def test_public_circuit_reviewer_routes_cccs_to_sparse_mna(tmp_path):
    from circhecker import CircuitReviewer

    text = (
        "V1 in 0 5\nR1 in sense 1k\nVsen sense 0 0\n"
        "F1 out 0 Vsen 2\nR2 out 0 1k\n.end"
    )
    result = CircuitReviewer(output_dir=tmp_path, plot=False).process_netlist_text(
        text, analyses=["dc_operating_point"], save_json=False,
    )
    assert result.execution_summary["backend"] == "sparse_mna_dc"
    assert result.analysis_results["dc_operating_point"]["status"] == "success"


def test_floating_resistor_current_source_network_is_reported_singular():
    text = "R1 a b 1k\nI1 a b 1m\n.end"
    with pytest.raises(SingularMatrixError):
        _solve(text)


def test_material_parameter_outside_model_subset_is_not_silently_ignored():
    ir = parse_spice_text("V1 in 0 5\nR1 in 0 1k TC1=0.001\n.end")
    supported, reasons = supports_circuit_ir_dc(ir)
    assert not supported
    assert any("TC1" in reason for reason in reasons)
    with pytest.raises(UnsupportedElementError, match="TC1"):
        solve_dc(ir)


def test_invalid_source_override_is_rejected():
    ir = parse_spice_text("V1 in 0 5\nR1 in 0 1k\n.end")
    with pytest.raises(Exception, match="does not identify"):
        solve_dc(ir, source_overrides={"R1": 2.0})


def test_empty_circuit_is_not_claimed_as_simulatable():
    ir = parse_spice_text("* empty circuit\n.end")
    supported, reasons = supports_circuit_ir_dc(ir)
    assert not supported
    assert any("no components" in reason for reason in reasons)


def test_public_api_does_not_claim_success_when_parameter_has_no_adapter(tmp_path):
    from circhecker import CircuitReviewer

    result = CircuitReviewer(output_dir=tmp_path, plot=False).process_netlist_text(
        "V1 in 0 5\nR1 in 0 1k TC1=0.001\n.end",
        analyses=["dc_operating_point"], save_json=False,
    )
    assert result.analysis_results["dc_operating_point"]["status"] == "unsupported"
    assert any("TC1" in str(warning) for warning in result.warnings)


def test_sparse_mna_diagnostics_are_preserved_as_json_numbers(tmp_path):
    import json
    from circhecker import CircuitReviewer

    result = CircuitReviewer(output_dir=tmp_path, plot=False).process_netlist_text(
        "V1 in 0 10\nR1 in out 1k\nR2 out 0 1k\n.end",
        analyses=["dc_operating_point"], save_json=True,
    )
    report_path = tmp_path / "Imported_circuit_analysis.json"
    if not report_path.exists():
        # The fallback is determined by the project naming sanitizer; locate the only report.
        report_path = next(tmp_path.glob("*_analysis.json"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    diagnostics = report["analysis_results"]["dc_operating_point"]["data"]["solver_diagnostics"]
    assert isinstance(diagnostics["matrix_dimension"], int)
    assert isinstance(diagnostics["matrix_nnz"], int)
    assert isinstance(diagnostics["relative_residual"], float)
    assert isinstance(diagnostics["condition_estimate"], float)


def test_public_api_reports_singular_circuit_as_failed_not_success(tmp_path):
    from circhecker import CircuitReviewer

    result = CircuitReviewer(output_dir=tmp_path, plot=False).process_netlist_text(
        "V1 out 0 5\nV2 out 0 10\nR1 out 0 1k\n.end",
        analyses=["dc_operating_point"], save_json=False,
    )
    assert result.analysis_results["dc_operating_point"]["status"] == "failed"
    assert result.status == "failed"
    assert "rank deficient" in result.analysis_results["dc_operating_point"]["reason"]
