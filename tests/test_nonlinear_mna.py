"""Independent analytical tests for the sparse nonlinear MNA backend."""
from __future__ import annotations

import math

import pytest
from scipy.optimize import brentq

from circhecker import CircuitReviewer
from circhecker.parser.spice import parse_spice_text
from circhecker.solver import ShockleyDiode, solve_nonlinear_dc, supports_circuit_ir_nonlinear_dc


def _thermal_voltage(temp_k: float = 300.15) -> float:
    return 1.380649e-23 * temp_k / 1.602176634e-19


def test_shockley_diode_current_and_derivative_match_independent_equations():
    diode = ShockleyDiode(2e-14, 1.7, 315.0)
    voltage = 0.48
    vt_n = 1.7 * _thermal_voltage(315.0)
    expected_i = 2e-14 * math.expm1(voltage / vt_n)
    expected_g = 2e-14 * math.exp(voltage / vt_n) / vt_n
    evaluation = diode.evaluate(voltage)
    assert evaluation.current_A == pytest.approx(expected_i, rel=1e-13)
    assert evaluation.conductance_S == pytest.approx(expected_g, rel=1e-13)
    step = 1e-7
    finite_difference = (diode.current_from_voltage(voltage + step) -
                         diode.current_from_voltage(voltage - step)) / (2 * step)
    assert evaluation.conductance_S == pytest.approx(finite_difference, rel=1e-7)


def test_shockley_diode_reverse_current_and_exponential_limiting_are_finite():
    diode = ShockleyDiode(1e-14, 1.0, 300.15)
    assert diode.current_from_voltage(-2.0) == pytest.approx(-1e-14, rel=1e-12)
    assert math.isfinite(diode.current_from_voltage(100.0))
    assert math.isfinite(diode.differential_conductance(100.0))


def test_nonlinear_mna_series_diode_matches_independent_brent_root():
    text = """V1 in 0 5
R1 in out 1k
D1 out 0 DM
.model DM D (IS=1e-14 N=1)
.end
"""
    ir = parse_spice_text(text)
    supported, reasons = supports_circuit_ir_nonlinear_dc(ir)
    assert supported, reasons
    result = solve_nonlinear_dc(ir)
    thermal_voltage = _thermal_voltage()
    expected_voltage = brentq(
        lambda v: (5.0 - v) / 1000.0 - 1e-14 * math.expm1(v / thermal_voltage),
        0.0, 1.2,
    )
    assert result.node_voltages_V[2] == pytest.approx(expected_voltage, abs=1e-10)
    diode = result.device_results["D1"]
    assert diode["current_A"] == pytest.approx((5.0 - expected_voltage) / 1000.0, rel=1e-9)
    assert result.diagnostics["converged"] is True
    assert result.diagnostics["iterations"] < 20
    assert result.verification["kcl"]["max_residual_A"] < 1e-10
    assert result.verification["kcl"]["is_balanced"]


def test_nonlinear_mna_reverse_biased_diode_matches_shockley_reference():
    text = """V1 in 0 -5
R1 in out 1k
D1 out 0 DM
.model DM D (IS=1e-14 N=1)
.end
"""
    result = solve_nonlinear_dc(parse_spice_text(text))
    expected_voltage = -5.0 + 1e-11  # reverse current is approximately -IS
    assert result.node_voltages_V[2] == pytest.approx(expected_voltage, abs=2e-9)
    assert result.device_results["D1"]["current_A"] == pytest.approx(-1e-14, rel=1e-5)


def test_nonlinear_mna_public_api_routes_explicit_dc_request(tmp_path):
    text = """V1 in 0 5
R1 in out 1k
D1 out 0 DM
.model DM D (IS=1e-14 N=1)
.end
"""
    result = CircuitReviewer(output_dir=tmp_path, plot=False).process_netlist_text(
        text, analyses=["dc_operating_point"], save_json=False,
    )
    assert result.execution_summary["backend"] == "nonlinear_mna_dc"
    assert result.analysis_results["dc_operating_point"]["status"] == "success"
    assert result.analysis_results["dc_operating_point"]["data"]["solver_diagnostics"]["backend"] == "circhecker.solver.nonlinear"


def test_nonlinear_mna_rejects_unimplemented_diode_series_resistance():
    text = """V1 in 0 5
R1 in out 1k
D1 out 0 DM RS=10
.model DM D (IS=1e-14 N=1)
.end
"""
    supported, reasons = supports_circuit_ir_nonlinear_dc(parse_spice_text(text))
    assert not supported
    assert any("RS" in reason for reason in reasons)


def test_nonlinear_mna_gmin_and_source_stepping_are_recorded_and_finish_unmodified():
    text = """V1 in 0 5
R1 in out 1k
D1 out 0 DM
.model DM D (IS=1e-14 N=1)
.end
"""
    result = solve_nonlinear_dc(parse_spice_text(text),
                                gmin_steps=(1e-3, 1e-6), source_steps=(0.1, 0.5))
    assistance = result.diagnostics["convergence_assistance"]
    assert assistance["gmin_stepping"]["attempted"] is True
    assert assistance["source_stepping"]["attempted"] is True
    assert len(result.diagnostics["continuation_stages"]) == 3
    assert result.diagnostics["continuation_stages"][-1]["gmin_S"] == 0.0
    assert result.diagnostics["continuation_stages"][-1]["source_scale"] == 1.0
    expected_voltage = brentq(
        lambda v: (5.0 - v) / 1000.0 - 1e-14 * math.expm1(v / _thermal_voltage()),
        0.0, 1.2,
    )
    assert result.node_voltages_V[2] == pytest.approx(expected_voltage, abs=1e-9)


def test_shockley_exponential_limiter_derivative_matches_its_current_law():
    diode = ShockleyDiode(1e-14, 1.0, 300.15)
    for voltage in (1.1, -2.0):
        step = 1e-7
        finite_difference = (diode.current_from_voltage(voltage + step) -
                             diode.current_from_voltage(voltage - step)) / (2 * step)
        assert diode.differential_conductance(voltage) == pytest.approx(
            finite_difference, rel=2e-5, abs=1e-20
        )


def test_spice_temperature_option_is_degrees_celsius():
    ir = parse_spice_text(
        """V1 in 0 5
R1 in out 1k
D1 out 0 DM
.model DM D (IS=1e-14 N=1)
.options TEMP=100
.end
"""
    )
    result = solve_nonlinear_dc(ir)
    expected_thermal_voltage = _thermal_voltage(373.15)
    assert result.device_results["D1"]["thermal_voltage_V"] == pytest.approx(
        expected_thermal_voltage, rel=1e-12
    )
