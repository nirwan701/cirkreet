"""Independent analytical tests for sparse RLC time-domain MNA."""
from __future__ import annotations

import math

import numpy as np
import pytest

from circhecker import CircuitReviewer
from circhecker.parser.spice import parse_spice_text
from circhecker.solver import solve_transient, supports_circuit_ir_transient


def _ramp_then_hold_rc(time_s: float, resistance: float, capacitance: float,
                       final_voltage: float, rise_time: float) -> float:
    tau = resistance * capacitance
    slope = final_voltage / rise_time
    at_rise_end = slope * (rise_time - tau * (1.0 - math.exp(-rise_time / tau)))
    if time_s <= rise_time:
        return slope * (time_s - tau * (1.0 - math.exp(-time_s / tau)))
    return at_rise_end + (final_voltage - at_rise_end) * (1.0 - math.exp(-(time_s - rise_time) / tau))


def test_rc_backward_euler_matches_analytical_ramp_and_hold_response():
    text = """V1 in 0 PULSE(0 5 0 100u 100u 10m 20m)
R1 in out 1k
C1 out 0 1u
.tran 10u 2m
.end
"""
    ir = parse_spice_text(text)
    assert supports_circuit_ir_transient(ir)[0]
    result = solve_transient(ir, method="backward_euler", output_node=2)
    time_s = result.data["time_s"][-1]
    expected = _ramp_then_hold_rc(time_s, 1e3, 1e-6, 5.0, 100e-6)
    assert result.data["output_V"][-1] == pytest.approx(expected, abs=5e-3)
    assert result.data["solver_diagnostics"]["accepted_steps"] == len(result.data["time_s"]) - 1
    assert result.data["integration_method"] == "backward_euler"


def test_rc_trapezoidal_matches_analytical_ramp_and_hold_response_more_tightly():
    text = """V1 in 0 PULSE(0 5 0 100u 100u 10m 20m)
R1 in out 1k
C1 out 0 1u
.tran 10u 2m
.end
"""
    result = solve_transient(parse_spice_text(text), method="trapezoidal", output_node=2)
    expected = _ramp_then_hold_rc(2e-3, 1e3, 1e-6, 5.0, 100e-6)
    assert result.data["output_V"][-1] == pytest.approx(expected, abs=2e-5)


def test_rl_trapezoidal_branch_current_matches_analytical_ramp_and_hold():
    text = """V1 in 0 PULSE(0 5 0 1u 1u 10m 20m)
R1 in mid 1k
L1 mid 0 10m
.tran 0.5u 100u
.end
"""
    result = solve_transient(parse_spice_text(text), method="trapezoidal")
    tau = 10e-3 / 1e3
    rise = 1e-6
    slope = 5.0 / rise
    current_at_rise = slope / 1e3 * (rise - tau * (1.0 - math.exp(-rise / tau)))
    expected = 5.0 / 1e3 + (current_at_rise - 5.0 / 1e3) * math.exp(-(100e-6 - rise) / tau)
    assert result.data["branch_currents_A"]["L1"][-1] == pytest.approx(expected, abs=2e-5)


def test_adaptive_transient_acceptance_and_rejection_do_not_corrupt_history():
    text = """V1 in 0 PULSE(0 5 0 100u 100u 10m 20m)
R1 in out 1k
C1 out 0 1u
.tran 100u 2m
.end
"""
    result = solve_transient(parse_spice_text(text), method="trapezoidal", adaptive=True,
                             relative_tolerance=1e-5, absolute_voltage_tolerance=1e-8,
                             output_node=2)
    diagnostics = result.data["solver_diagnostics"]
    assert diagnostics["accepted_steps"] > 0
    assert diagnostics["rejected_steps"] > 0
    assert diagnostics["linear_solves"] >= 2 * diagnostics["accepted_steps"]
    assert np.all(np.diff(result.data["time_s"]) > 0)
    assert result.data["time_s"][-1] == pytest.approx(2e-3)
    assert result.data["output_V"][-1] == pytest.approx(
        _ramp_then_hold_rc(2e-3, 1e3, 1e-6, 5.0, 100e-6), abs=2e-3
    )


def test_pwl_source_breakpoints_are_retained_in_transient_time_vector():
    text = """V1 in 0 PWL(0 0 1m 1 2m 0)
R1 in out 1k
C1 out 0 1u
.tran 0.7m 2m
.end
"""
    result = solve_transient(parse_spice_text(text))
    times = result.data["time_s"]
    assert any(abs(value - 1e-3) < 1e-15 for value in times)
    assert any(abs(value - 2e-3) < 1e-15 for value in times)


def test_public_api_routes_explicit_transient_request_to_mna(tmp_path):
    text = """V1 in 0 PULSE(0 5 0 100u 100u 10m 20m)
R1 in out 1k
C1 out 0 1u
.options METHOD=TRAP
.tran 10u 2m
.end
"""
    result = CircuitReviewer(output_dir=tmp_path, plot=False).process_netlist_text(
        text, analyses=["transient"], save_json=False,
    )
    assert result.execution_summary["backend"] == "mna_transient"
    assert result.analysis_results["transient"]["status"] == "success"
    assert result.analysis_results["transient"]["data"]["integration_method"] == "trapezoidal"
    assert result.analysis_results["transient"]["data"]["solver_diagnostics"]["backend"] == "circhecker.solver.transient"


def test_nonstorage_network_is_not_claimed_as_transient_supported():
    ir = parse_spice_text("V1 in 0 5\nR1 in out 1k\n.end")
    supported, reasons = supports_circuit_ir_transient(ir)
    assert not supported
    assert any("energy-storage" in reason for reason in reasons)


def test_parallel_rlc_natural_response_matches_analytical_solution():
    text = """R1 out 0 100
L1 out 0 10m
C1 out 0 1u IC=1
.tran 1u 1m
.end
"""
    result = solve_transient(parse_spice_text(text), method="trapezoidal", output_node=1)
    time_s = 200e-6
    alpha = 1.0 / (2.0 * 100.0 * 1e-6)
    omega_d = math.sqrt(1.0 / (10e-3 * 1e-6) - alpha**2)
    expected = math.exp(-alpha * time_s) * (math.cos(omega_d * time_s) -
                                                  (alpha / omega_d) * math.sin(omega_d * time_s))
    index = int(np.argmin(np.abs(np.asarray(result.data["time_s"]) - time_s)))
    assert result.data["output_V"][index] == pytest.approx(expected, abs=2e-3)


def test_transient_initial_operating_point_uses_waveform_value_at_zero():
    text = """V1 in 0 PWL(0 2 1m 2)
R1 in out 1k
C1 out 0 1u
.tran 10u 100u
.end
"""
    result = solve_transient(parse_spice_text(text), method="trapezoidal", output_node=2)
    assert result.data["output_V"][0] == pytest.approx(2.0, abs=1e-10)


def test_conflicting_parallel_capacitor_initial_conditions_are_rejected():
    text = """R1 out 0 1k
C1 out 0 1u IC=1
C2 out 0 2u IC=2
.tran 1u 10u
.end
"""
    with pytest.raises(Exception, match="capacitor IC conflicts"):
        solve_transient(parse_spice_text(text))


def test_consistent_parallel_capacitor_initial_conditions_are_accepted():
    text = """R1 out 0 1k
C1 out 0 1u IC=1
C2 out 0 2u IC=1
.tran 1u 10u
.end
"""
    result = solve_transient(parse_spice_text(text), method="trapezoidal", output_node=1)
    assert result.data["output_V"][0] == pytest.approx(1.0, abs=1e-12)


def test_public_api_enables_adaptive_timestep_through_options(tmp_path):
    text = """V1 in 0 PULSE(0 5 0 100u 100u 10m 20m)
R1 in out 1k
C1 out 0 1u
.options METHOD=TRAP ADAPTIVE=1 RELTOL=1e-5 ABSTOL=1e-8
.tran 100u 2m
.end
"""
    result = CircuitReviewer(output_dir=tmp_path, plot=False).process_netlist_text(
        text, analyses=["transient"], save_json=False
    )
    assert result.analysis_results["transient"]["status"] == "success"
    diagnostics = result.analysis_results["transient"]["data"]["solver_diagnostics"]
    assert diagnostics["adaptive"] is True
    assert diagnostics["rejected_steps"] > 0


def test_inductor_initial_current_decays_from_declared_initial_state():
    text = """R1 out 0 100
L1 out 0 10m IC=1
.tran 1u 300u
.end
"""
    result = solve_transient(parse_spice_text(text), method="trapezoidal")
    time_s = 100e-6
    index = int(np.argmin(np.abs(np.asarray(result.data["time_s"]) - time_s)))
    expected_current = math.exp(-time_s / (10e-3 / 100.0))
    assert result.data["time_s"][index] == pytest.approx(time_s, abs=1e-15)
    assert result.data["branch_currents_A"]["L1"][index] == pytest.approx(expected_current, abs=3e-4)


def test_capacitor_initial_condition_includes_consistent_initial_current():
    text = """R1 out 0 1k
C1 out 0 1u IC=1
.tran 1u 1m
.end
"""
    result = solve_transient(parse_spice_text(text), method="trapezoidal", output_node=1)
    index = int(np.argmin(np.abs(np.asarray(result.data["time_s"]) - 1e-3)))
    expected = math.exp(-1.0)
    assert result.data["output_V"][index] == pytest.approx(expected, abs=1e-7)


def test_node_initial_condition_is_enforced_in_initial_mna_state():
    text = """R1 out 0 1k
C1 out 0 1u
.ic V(out)=1
.tran 1u 10u
.end
"""
    result = solve_transient(parse_spice_text(text), method="trapezoidal", output_node=1)
    assert result.data["output_V"][0] == pytest.approx(1.0, abs=1e-12)


def test_node_initial_condition_conflict_with_ideal_voltage_source_is_rejected():
    text = """V1 out 0 2
R1 out 0 1k
C1 out 0 1u
.ic V(out)=1
.tran 1u 10u
.end
"""
    with pytest.raises(Exception, match="conflicts with an existing initial-voltage constraint"):
        solve_transient(parse_spice_text(text), method="trapezoidal")


def test_trapezoidal_timestep_refinement_converges_to_rc_reference():
    text = """V1 in 0 PULSE(0 1 0 1n 1n 10m 20m)
R1 in out 1k
C1 out 0 1u
.tran 10u 5m
.end
"""
    ir = parse_spice_text(text)
    exact_time = 1e-3
    exact_voltage = 1.0 - math.exp(-exact_time / 1e-3)
    errors = []
    for maximum_step in (100e-6, 50e-6, 25e-6, 12.5e-6):
        result = solve_transient(
            ir, method="trapezoidal", max_step_s=maximum_step, output_node=2,
        )
        times = np.asarray(result.data["time_s"])
        values = np.asarray(result.data["output_V"])
        sample_index = int(np.argmin(np.abs(times - exact_time)))
        assert times[sample_index] == pytest.approx(exact_time + 1e-9, abs=1e-12)
        errors.append(abs(float(values[sample_index]) - exact_voltage))
    assert all(later < earlier for earlier, later in zip(errors, errors[1:]))
    # Halving h in trapezoidal integration should approach its second-order regime.
    assert errors[0] / errors[1] > 3.5
    assert errors[-1] < 6e-6
