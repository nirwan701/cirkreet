"""Nonlinear transient tests against independent ODE and device-family regressions."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from circhecker import CircuitReviewer
from circhecker.parser.spice import parse_spice_text
from circhecker.solver import solve_transient, supports_circuit_ir_nonlinear_transient


def _junction_capacitance(v: float, c0: float, vj: float, grading: float, fc: float) -> float:
    """Independent scalar derivative of the declared depletion-charge law."""
    if v <= fc * vj:
        return c0 * (1.0 - v / vj) ** (-grading)
    return c0 * (1.0 - fc) ** (-grading)


def test_diode_rc_with_nonlinear_junction_charge_matches_independent_ode():
    netlist = """V1 src 0 DC 5
R1 src out 1k
D1 out 0 DM
C1 out 0 1u IC=0
.model DM D (IS=1e-14 N=1 CJO=10n VJ=0.7 M=0.5 FC=0.5)
.tran 10u 2m
.end
"""
    ir = parse_spice_text(netlist)
    assert supports_circuit_ir_nonlinear_transient(ir) == (True, [])
    solution = solve_transient(
        ir, method="trapezoidal", adaptive=True, relative_tolerance=1e-5,
        absolute_voltage_tolerance=1e-8, output_node=2,
    )
    time = np.asarray(solution.data["time_s"], dtype=float)
    actual = np.asarray(solution.data["output_V"], dtype=float)

    # Independent ODE reference: Cload + dQjunction/dV, with Shockley current.
    vt = 1.380649e-23 * 300.15 / 1.602176634e-19

    def rhs(_t, state):
        voltage = float(state[0])
        diode_current = 1e-14 * (math.exp(voltage / vt) - 1.0)
        total_capacitance = 1e-6 + _junction_capacitance(
            voltage, 10e-9, 0.7, 0.5, 0.5
        )
        return [((5.0 - voltage) / 1e3 - diode_current) / total_capacitance]

    reference = solve_ivp(
        rhs, (0.0, 2e-3), [0.0], t_eval=time,
        method="DOP853", rtol=1e-11, atol=1e-13,
    )
    assert reference.success
    assert actual[0] == pytest.approx(0.0, abs=1e-12)
    assert np.max(np.abs(actual - reference.y[0])) < 5e-5
    diagnostics = solution.data["solver_diagnostics"]
    assert diagnostics["backend"] == "circhecker.solver.nonlinear_transient"
    assert diagnostics["adaptive_controller"] == "step_doubling_full_vs_two_half_steps"
    assert diagnostics["accepted_steps"] > 0
    assert diagnostics["rejected_steps"] > 0


def test_nonlinear_capacitor_initial_condition_is_enforced():
    netlist = """R1 out 0 1k
D1 out 0 DM
C1 out 0 1u IC=0.7
.model DM D (IS=1e-14 N=1 CJO=2n)
.tran 1u 10u
.end
"""
    solution = solve_transient(parse_spice_text(netlist), method="backward_euler", output_node=1)
    assert solution.data["output_V"][0] == pytest.approx(0.7, abs=1e-10)
    assert solution.data["solver_diagnostics"]["backend"] == "circhecker.solver.nonlinear_transient"


@pytest.mark.parametrize(
    "netlist, output_node, expected_swing",
    [
        (
            """VCC vcc 0 5
VB drive 0 PULSE(0 0.8 10u 1u 1u 20u 60u)
RB drive b 10k
RC vcc c 1k
Q1 c b 0 QN
CLOAD c 0 1n
.model QN NPN (IS=1e-15 BF=100 BR=1 VAF=100 CJE=2p CJC=1p TF=1n)
.tran 2u 40u
.end
""",
            4, 0.2,
        ),
        (
            """VDD vdd 0 5
VG g 0 PULSE(0 3 10u 1u 1u 20u 60u)
RD vdd d 1k
M1 d g 0 0 MN W=10u L=1u
CLOAD d 0 1n
.model MN NMOS (VTO=1 KP=1m LAMBDA=0.02 CGS=2p CGD=1p CGB=0.5p)
.tran 2u 40u
.end
""",
            3, 1.0,
        ),
        (
            """VDD vdd 0 5
VG g 0 PULSE(0 -5 10u 1u 1u 20u 60u)
RD vdd d 1k
J1 d g 0 JN
CLOAD d 0 1n
.model JN NJF (VTO=-2 BETA=1m LAMBDA=0.01 CGS=2p CGD=1p)
.tran 2u 40u
.end
""",
            3, 1.0,
        ),
    ],
    ids=["bjt-switching", "mosfet-switching", "jfet-switching"],
)
def test_transistor_families_share_nonlinear_transient_engine(netlist, output_node, expected_swing):
    ir = parse_spice_text(netlist)
    solution = solve_transient(ir, method="backward_euler", adaptive=False, output_node=output_node)
    waveform = np.asarray(solution.data["output_V"], dtype=float)
    assert solution.data["solver_diagnostics"]["backend"] == "circhecker.solver.nonlinear_transient"
    assert np.all(np.isfinite(waveform))
    assert np.ptp(waveform) > expected_swing


def test_public_api_routes_semiconductor_circuit_to_charge_based_transient(tmp_path):
    netlist = """V1 src 0 DC 5
R1 src out 1k
D1 out 0 DM
C1 out 0 1u IC=0
.model DM D (IS=1e-14 N=1 CJO=10n)
.tran 10u 300u
.end
"""
    result = CircuitReviewer(output_dir=tmp_path, plot=False).process_netlist_text(
        netlist, analyses=["transient"], save_json=False
    )
    transient = result.analysis_results["transient"]
    assert result.execution_summary["backend"] == "mna_transient"
    assert transient["status"] == "success"
    assert transient["data"]["solver_diagnostics"]["backend"] == "circhecker.solver.nonlinear_transient"


def test_diode_rl_inductor_flux_matches_independent_current_ode():
    netlist = """V1 src 0 PULSE(2.6745 5 20u 1u 1u 2m 4m)
R1 src n 1k
L1 n out 10m IC=2m
D1 out 0 DM
.model DM D (IS=1e-14 N=1)
.tran 2u 100u
.end
"""
    solution = solve_transient(
        parse_spice_text(netlist), method="trapezoidal", adaptive=True,
        relative_tolerance=1e-5, absolute_current_tolerance=1e-9,
        absolute_voltage_tolerance=1e-8,
    )
    time = np.asarray(solution.data["time_s"], dtype=float)
    actual = np.asarray(solution.data["branch_currents_A"]["L1"], dtype=float)

    # Independent ODE for L di/dt + R i + Vd(i) = Vs(t); no solver device
    # evaluation or derivative code is reused in this reference.
    vt = 1.380649e-23 * 300.15 / 1.602176634e-19

    def source_voltage(t: float) -> float:
        if t < 20e-6:
            return 2.6745
        if t < 21e-6:
            return 2.6745 + (5.0 - 2.6745) * (t - 20e-6) / 1e-6
        return 5.0

    def rhs(t: float, state):
        current = max(float(state[0]), -0.999999999e-14)
        diode_voltage = vt * math.log1p(current / 1e-14)
        return [(source_voltage(t) - 1e3 * current - diode_voltage) / 0.01]

    reference = solve_ivp(
        rhs, (0.0, 100e-6), [2e-3], t_eval=time,
        method="DOP853", rtol=1e-11, atol=1e-13,
    )
    assert reference.success
    assert actual[0] == pytest.approx(2e-3, abs=1e-12)
    assert np.max(np.abs(actual - reference.y[0])) < 2e-7
    diag = solution.data["solver_diagnostics"]
    assert diag["rejected_steps"] > 0
    assert diag["minimum_accepted_step_s"] < diag["maximum_configured_step_s"]


def test_conflicting_parallel_capacitor_initial_conditions_are_rejected():
    netlist = """R1 out 0 1k
D1 out 0 DM
C1 out 0 1u IC=1
C2 out 0 2u IC=2
.model DM D (IS=1e-14)
.tran 1u 10u
.end
"""
    with pytest.raises(Exception, match="parallel capacitor IC"):
        solve_transient(parse_spice_text(netlist), method="backward_euler")


def test_unsupported_uic_option_is_reported_and_gated():
    ir = parse_spice_text("""R1 out 0 1k
D1 out 0 DM
C1 out 0 1u
.model DM D (IS=1e-14)
.tran 1u 10u UIC
.end
""")
    assert any(w.get("code") == "UNSUPPORTED_TRAN_OPTION" for w in ir.validation_results["warnings"])
    ok, reasons = supports_circuit_ir_nonlinear_transient(ir)
    assert not ok
    assert any("UIC" in reason for reason in reasons)


def test_full_wave_bridge_rectifier_matches_independent_charge_storage_ode():
    from scipy.optimize import brentq

    netlist = """VSEC drive 0 SIN(0 10 50)
RS drive a 50
D1 a vp DM
D2 0 vp DM
D3 vn a DM
D4 vn 0 DM
RLOAD vp vn 1k
CLOAD vp vn 47u IC=0
.model DM D (IS=1e-14 N=1)
.tran 100u 10m
.end
"""
    ir = parse_spice_text(netlist)
    positive = next(n["id"] for n in ir.nodes if n["name"] == "vp")
    negative = next(n["id"] for n in ir.nodes if n["name"] == "vn")
    solution = solve_transient(
        ir, method="trapezoidal", adaptive=True, relative_tolerance=1e-3,
        absolute_voltage_tolerance=1e-5, absolute_current_tolerance=1e-7,
        output_pair=(positive, negative),
    )
    time = np.asarray(solution.data["time_s"], dtype=float)
    actual = np.asarray(solution.data["output_V"], dtype=float)

    # Independent averaged full-wave bridge equation.  A conducting path has
    # two Shockley junctions and the 50-ohm source resistance; the bridge
    # current is found by a scalar root, then integrated as C*dV/dt=I-V/R.
    vt = 1.380649e-23 * 300.15 / 1.602176634e-19

    def rhs(t: float, state):
        source = abs(10.0 * math.sin(2.0 * math.pi * 50.0 * t))
        voltage = max(0.0, float(state[0]))
        available = max(0.0, source - voltage)
        if available == 0.0:
            bridge_current = 0.0
        else:
            bridge_current = brentq(
                lambda current: 50.0 * current + 2.0 * vt * math.log1p(current / 1e-14) - available,
                0.0, available / 50.0, xtol=1e-14,
            )
        return [(bridge_current - voltage / 1e3) / 47e-6]

    reference = solve_ivp(
        rhs, (0.0, 10e-3), [0.0], t_eval=time,
        method="Radau", rtol=1e-10, atol=1e-12,
    )
    assert reference.success
    assert np.all(np.isfinite(actual))
    assert np.max(np.abs(actual - reference.y[0])) < 5e-4
    diagnostics = solution.data["solver_diagnostics"]
    assert diagnostics["adaptive_controller"] == "step_doubling_full_vs_two_half_steps"
    assert diagnostics["gmin_S"] == pytest.approx(1e-12)


def test_backward_euler_diode_rc_matches_independent_ode():
    netlist = """V1 src 0 DC 5
R1 src out 1k
D1 out 0 DM
C1 out 0 1u IC=0
.model DM D (IS=1e-14 N=1 CJO=10n VJ=0.7 M=0.5 FC=0.5)
.tran 10u 2m
.end
"""
    ir = parse_spice_text(netlist)
    output = next(n["id"] for n in ir.nodes if n["name"] == "out")
    solution = solve_transient(
        ir, method="backward_euler", adaptive=True, relative_tolerance=1e-5,
        absolute_voltage_tolerance=1e-8, output_node=output,
    )
    time = np.asarray(solution.data["time_s"], dtype=float)
    actual = np.asarray(solution.data["output_V"], dtype=float)
    vt = 1.380649e-23 * 300.15 / 1.602176634e-19

    def capacitance(v: float) -> float:
        return 10e-9 * (1.0 - v / 0.7) ** -0.5 if v <= 0.35 else 10e-9 * (0.5 ** -0.5)

    def rhs(_t, state):
        voltage = float(state[0])
        diode_current = 1e-14 * math.expm1(voltage / vt)
        return [((5.0 - voltage) / 1e3 - diode_current) / (1e-6 + capacitance(voltage))]

    reference = solve_ivp(rhs, (0.0, 2e-3), [0.0], t_eval=time,
                          method="DOP853", rtol=1e-11, atol=1e-13)
    assert reference.success
    assert np.max(np.abs(actual - reference.y[0])) < 4e-4
    assert solution.data["solver_diagnostics"]["method"] == "backward_euler"


def test_level1_mosfet_switching_matches_independent_scalar_ode():
    netlist = """VDD vdd 0 5
VG g 0 PULSE(0 3 10u 1u 1u 20u 60u)
RD vdd d 1k
M1 d g 0 0 MN W=10u L=1u
CLOAD d 0 1n IC=5
.model MN NMOS (VTO=1 KP=1m LAMBDA=0.02)
.tran 2u 40u
.end
"""
    ir = parse_spice_text(netlist)
    output = next(n["id"] for n in ir.nodes if n["name"] == "d")
    solution = solve_transient(
        ir, method="trapezoidal", adaptive=True, relative_tolerance=1e-4,
        absolute_voltage_tolerance=1e-7, absolute_current_tolerance=1e-9,
        output_node=output,
    )
    time = np.asarray(solution.data["time_s"], dtype=float)
    actual = np.asarray(solution.data["output_V"], dtype=float)

    def gate(t: float) -> float:
        if t < 10e-6: return 0.0
        if t < 11e-6: return 3.0 * (t - 10e-6) / 1e-6
        if t < 31e-6: return 3.0
        if t < 32e-6: return 3.0 * (32e-6 - t) / 1e-6
        return 0.0

    def rhs(t: float, state):
        drain = float(state[0])
        overdrive = gate(t) - 1.0
        vds = max(drain, 0.0)
        current = 0.0
        if overdrive > 0.0:
            beta_eff = 1e-3 * 10.0  # KP * W/L, independently resolved from the netlist.
            if vds < overdrive:
                base = beta_eff * (overdrive * vds - 0.5 * vds * vds)
            else:
                base = 0.5 * beta_eff * overdrive * overdrive
            current = base * (1.0 + 0.02 * vds)
        return [((5.0 - drain) / 1e3 - current) / 1e-9]

    reference = solve_ivp(rhs, (0.0, 40e-6), [5.0], t_eval=time,
                          method="DOP853", rtol=1e-11, atol=1e-12)
    assert reference.success
    assert np.max(np.abs(actual - reference.y[0])) < 1e-3


@pytest.mark.slow
def test_jfet_switching_matches_independent_scalar_ode():
    from scipy.optimize import brentq

    netlist = """VDD vdd 0 5
VG g 0 PULSE(0 -5 10u 1u 1u 20u 60u)
RD vdd d 1k
J1 d g 0 JN
CLOAD d 0 1n
.model JN NJF (VTO=-2 BETA=1m LAMBDA=0.01)
.tran 2u 40u
.end
"""
    ir = parse_spice_text(netlist)
    output = next(n["id"] for n in ir.nodes if n["name"] == "d")
    solution = solve_transient(
        ir, method="trapezoidal", adaptive=True, relative_tolerance=1e-4,
        absolute_voltage_tolerance=1e-7, absolute_current_tolerance=1e-9,
        output_node=output,
    )
    time = np.asarray(solution.data["time_s"], dtype=float)
    actual = np.asarray(solution.data["output_V"], dtype=float)

    def gate(t: float) -> float:
        if t < 10e-6: return 0.0
        if t < 11e-6: return -5.0 * (t - 10e-6) / 1e-6
        if t < 31e-6: return -5.0
        if t < 32e-6: return -5.0 * (32e-6 - t) / 1e-6
        return 0.0

    def channel_current(vg: float, drain: float) -> float:
        overdrive = vg - (-2.0)
        vds = max(drain, 0.0)
        if overdrive <= 0.0:
            return 0.0
        if vds < overdrive:
            base = 2.0 * 1e-3 * (overdrive * vds - 0.5 * vds * vds)
        else:
            base = 1e-3 * overdrive * overdrive
        return base * (1.0 + 0.01 * vds)

    initial = brentq(lambda v: (5.0 - v) / 1e3 - channel_current(0.0, v), 0.0, 5.0)

    def rhs(t: float, state):
        drain = float(state[0])
        return [((5.0 - drain) / 1e3 - channel_current(gate(t), drain)) / 1e-9]

    reference = solve_ivp(rhs, (0.0, 40e-6), [initial], t_eval=time,
                          method="DOP853", rtol=1e-11, atol=1e-12)
    assert reference.success
    assert np.max(np.abs(actual - reference.y[0])) < 1e-3


@pytest.mark.slow
def test_bjt_switching_matches_independent_scalar_ode():
    netlist = """VCC vcc 0 5
VB b 0 PULSE(0 0.8 10u 1u 1u 20u 60u)
RC vcc c 1k
Q1 c b 0 QN
CLOAD c 0 1n IC=5
.model QN NPN (IS=1e-15 BF=100 BR=1 VAF=100 VAR=100 CJE=0 CJC=0 TF=0 TR=0)
.tran 2u 40u
.end
"""
    ir = parse_spice_text(netlist)
    output = next(n["id"] for n in ir.nodes if n["name"] == "c")
    solution = solve_transient(
        ir, method="trapezoidal", adaptive=True, relative_tolerance=1e-4,
        absolute_voltage_tolerance=1e-7, absolute_current_tolerance=1e-9,
        output_node=output, maximum_newton_iterations=100,
    )
    time = np.asarray(solution.data["time_s"], dtype=float)
    actual = np.asarray(solution.data["output_V"], dtype=float)
    vt = 1.380649e-23 * 300.15 / 1.602176634e-19
    alpha_f = 100.0 / 101.0

    def gate(t: float) -> float:
        if t < 10e-6: return 0.0
        if t < 11e-6: return 0.8 * (t - 10e-6) / 1e-6
        if t < 31e-6: return 0.8
        if t < 32e-6: return 0.8 * (32e-6 - t) / 1e-6
        return 0.0

    def shockley(voltage: float) -> float:
        exponent = voltage / vt
        if exponent < -80.0: exp_value = math.exp(-80.0)
        elif exponent > 40.0: exp_value = math.exp(40.0) * (1.0 + exponent - 40.0)
        else: exp_value = math.exp(exponent)
        return 1e-15 * (exp_value - 1.0)

    def rhs(t: float, state):
        collector = float(state[0])
        base = gate(t)
        forward = shockley(base) * (1.0 + collector / 100.0)
        reverse = shockley(base - collector) * (1.0 - collector / 100.0)
        collector_current = alpha_f * forward - reverse
        return [((5.0 - collector) / 1e3 - collector_current) / 1e-9]

    reference = solve_ivp(
        rhs, (0.0, 40e-6), [5.0], t_eval=time,
        method="DOP853", rtol=1e-10, atol=1e-12, max_step=0.05e-6,
    )
    assert reference.success
    assert np.max(np.abs(actual - reference.y[0])) < 1e-3


def test_node_ic_constraint_seeds_consistent_charge_history_for_trapezoidal():
    netlist = """R1 out 0 1k
C1 out 0 1u
D1 out 0 DM
.model DM D (IS=1e-14 N=1 CJO=10n)
.ic V(out)=0.7
.tran 1u 20u
.end
"""
    ir = parse_spice_text(netlist)
    output = next(n["id"] for n in ir.nodes if n["name"] == "out")
    solution = solve_transient(
        ir, method="trapezoidal", adaptive=False, output_node=output,
        absolute_voltage_tolerance=1e-8,
    )
    waveform = np.asarray(solution.data["output_V"], dtype=float)
    assert waveform[0] == pytest.approx(0.7, abs=1e-10)
    assert np.all(np.isfinite(waveform))
    assert waveform[-1] < waveform[0]


def test_adaptive_error_channels_and_device_terminal_current_outputs_are_recorded():
    netlist = """V1 src 0 PULSE(0 1 0 10u 10u 1m 2m)
R1 src n 100
L1 n out 1m
D1 out 0 DM
C1 out 0 1u IC=0
.model DM D (IS=1e-14 N=1 CJO=2n TT=1n)
.tran 20u 500u
.end
"""
    solution = solve_transient(
        parse_spice_text(netlist), method="trapezoidal", adaptive=True,
        relative_tolerance=1e-4, absolute_voltage_tolerance=1e-7,
    )
    data = solution.data
    diagnostics = data["solver_diagnostics"]
    assert diagnostics["maximum_scaled_state_error"] <= 1.0
    assert diagnostics["maximum_scaled_charge_error"] <= 1.0
    assert diagnostics["maximum_scaled_charge_current_error"] <= 1.0
    assert diagnostics["maximum_scaled_magnetic_flux_error"] <= 1.0
    assert diagnostics["maximum_scaled_magnetic_flux_error"] > 0.0
    assert diagnostics["maximum_scaled_linear_storage_energy_error"] <= 1.0
    assert diagnostics["maximum_scaled_local_error"] <= 1.0
    assert len(diagnostics["error_estimate_history"]) > 0
    assert len(data["stored_linear_energy_J"]) == len(data["time_s"])
    assert np.all(np.asarray(data["stored_linear_energy_J"]) >= 0.0)
    assert len(data["linear_capacitor_charges_C"]["C1"]) == len(data["time_s"])
    assert len(data["inductor_flux_Wb"]["L1"]) == len(data["time_s"])
    np.testing.assert_allclose(
        np.asarray(data["inductor_flux_Wb"]["L1"]),
        1e-3 * np.asarray(data["branch_currents_A"]["L1"]),
        rtol=1e-12, atol=1e-18,
    )
    diode = data["device_terminal_currents_A"]["D1"]
    assert data["device_terminal_conduction_currents_A"] == data["device_terminal_currents_A"]
    assert "Conduction current" in data["device_terminal_current_semantics"]
    assert set(diode) == {"anode", "cathode"}
    terminal_sum = np.asarray(diode["anode"]) + np.asarray(diode["cathode"])
    assert np.max(np.abs(terminal_sum)) < 1e-18
    diode_charge = data["device_terminal_charges_C"]["D1"]
    assert np.max(np.abs(np.asarray(diode_charge["anode"]) + np.asarray(diode_charge["cathode"]))) < 1e-22


@pytest.mark.parametrize("tolerance_name", [
    "absolute_charge_tolerance_C",
    "absolute_charge_current_tolerance_A",
    "absolute_flux_tolerance_Wb",
    "absolute_energy_tolerance_J",
])
def test_adaptive_dimensioned_error_tolerances_reject_nonpositive_values(tolerance_name):
    netlist = """V1 src 0 DC 1
R1 src out 1k
D1 out 0 DM
C1 out 0 1n
.model DM D (IS=1e-14 N=1 CJO=1p)
.tran 1u 10u
.end
"""
    with pytest.raises(ValueError, match=tolerance_name):
        solve_transient(parse_spice_text(netlist), adaptive=True, **{tolerance_name: 0.0})
