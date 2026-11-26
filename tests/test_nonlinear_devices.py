"""Independent derivative and physical-consistency tests for D/Q/M/J devices."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from circhecker.parser.spice import parse_spice_text
from circhecker.solver.devices import (
    GummelPoonBJT, Level1JFET, Level1MOSFET, ShockleyDiode,
)
from circhecker.solver.nonlinear import (
    assemble_nonlinear_ac, solve_nonlinear_ac, solve_nonlinear_dc,
    supports_circuit_ir_nonlinear_dc,
)


def _finite_difference_jacobian(model, voltages, attribute, step=1e-6):
    values = np.asarray(voltages, dtype=float)
    base = model.evaluate_terminals(values)
    result = np.zeros_like(getattr(base, attribute), dtype=float)
    for column in range(values.size):
        plus, minus = values.copy(), values.copy()
        plus[column] += step
        minus[column] -= step
        vp = getattr(model.evaluate_terminals(plus), "terminal_currents_A" if attribute == "current_jacobian_S" else "terminal_charges_C")
        vm = getattr(model.evaluate_terminals(minus), "terminal_currents_A" if attribute == "current_jacobian_S" else "terminal_charges_C")
        result[:, column] = (vp - vm) / (2.0 * step)
    return result


def _assert_terminal_derivatives(model, voltages, *, current_rtol=3e-5, charge_rtol=3e-5):
    evaluation = model.evaluate_terminals(voltages)
    current_fd = _finite_difference_jacobian(model, voltages, "current_jacobian_S")
    np.testing.assert_allclose(evaluation.current_jacobian_S, current_fd, rtol=current_rtol, atol=2e-12)
    assert abs(float(np.sum(evaluation.terminal_currents_A))) < 1e-10
    if evaluation.terminal_charges_C is not None:
        charge_fd = _finite_difference_jacobian(model, voltages, "charge_jacobian_F", step=1e-5)
        np.testing.assert_allclose(evaluation.charge_jacobian_F, charge_fd, rtol=charge_rtol, atol=2e-16)
        assert abs(float(np.sum(evaluation.terminal_charges_C))) < 1e-18


def test_diode_terminal_interface_derivative_and_charge_derivative():
    diode = ShockleyDiode(2e-14, 1.2, 310.15, CJO_F=2e-12, VJ_V=0.7, M=0.4, TT_s=2e-9)
    _assert_terminal_derivatives(diode, [0.52, 0.0])
    for voltage in (-1.0, 0.0, 0.3, 0.8, 1.2):
        assert np.isfinite(diode.current_from_voltage(voltage))
        assert diode.differential_conductance(voltage) >= 0.0


def test_zener_reverse_breakdown_branch_is_continuous_and_has_correct_derivative():
    diode = ShockleyDiode(1e-12, 1.0, 300.15, BV_V=5.1, RBV_Ohm=8.0)
    at_knee = diode.evaluate(-5.1)
    beyond_knee = diode.evaluate(-5.2)
    assert at_knee.current_A == pytest.approx(-1e-12, rel=1e-6)
    assert beyond_knee.current_A == pytest.approx(-0.0125, rel=1e-8)
    assert beyond_knee.conductance_S == pytest.approx(1.0 / 8.0)
    _assert_terminal_derivatives(diode, [0.0, 5.2], current_rtol=1e-8)


def test_zener_reverse_breakdown_participates_in_nonlinear_dc_mna():
    ir, solution = _solve_netlist("""V1 vin 0 DC 12
R1 vin vout 330
RLOAD vout 0 1k
DZ 0 vout DZM
.model DZM D (IS=1e-12 N=1 BV=5.1 RBV=8)
.end
""")
    node_id = ir.node_map["vout"]
    assert solution.node_voltages_V[node_id] == pytest.approx(5.2225, abs=0.01)
    assert solution.device_results["DZ"]["current_A"] < 0
    assert solution.verification["kcl"]["is_balanced"]


def test_bjt_gummel_poon_subset_derivatives_and_current_conservation():
    bjt = GummelPoonBJT(
        "NPN", IS_A=2e-15, BF=120, BR=2, VAF_V=90, VAR_V=35,
        CJE_F=3e-12, CJC_F=1e-12, TF_s=0.8e-9, TR_s=0.2e-9,
    )
    _assert_terminal_derivatives(bjt, [4.5, 0.68, 0.0], current_rtol=5e-5, charge_rtol=5e-5)
    active = bjt.evaluate_terminals([5.0, 0.68, 0.0])
    assert active.operating_point["region"] == "forward_active"
    assert active.operating_point["IC_A"] > 0
    assert active.operating_point["IB_A"] > 0
    assert active.operating_point["IE_A"] < 0


def test_pnp_bjt_polarity_is_consistent():
    bjt = GummelPoonBJT("PNP", IS_A=1e-15, BF=80, BR=1)
    evaluation = bjt.evaluate_terminals([-5.0, -0.68, 0.0])
    assert evaluation.terminal_currents_A[0] < 0
    assert evaluation.terminal_currents_A[1] < 0
    assert evaluation.terminal_currents_A[2] > 0
    assert abs(float(evaluation.terminal_currents_A.sum())) < 1e-12


def test_level1_mosfet_derivatives_body_effect_and_charge_conservation():
    model = Level1MOSFET(
        "NMOS", VTO_V=1.0, KP_A_V2=1e-3, LAMBDA_per_V=0.02,
        GAMMA_sqrtV=0.35, PHI_V=0.7, CGS_F=1.2e-12, CGD_F=0.7e-12, CGB_F=0.2e-12,
    )
    _assert_terminal_derivatives(model, [4.0, 3.0, 0.4, 0.0])
    evaluation = model.evaluate_terminals([4.0, 3.0, 0.4, 0.0])
    assert evaluation.operating_point["region"] == "saturation"
    cutoff = model.evaluate_terminals([1.0, 0.0, 0.0, 0.0])
    assert cutoff.operating_point["region"] == "cutoff"
    linear = model.evaluate_terminals([0.2, 3.0, 0.0, 0.0])
    assert linear.operating_point["region"] == "linear"


def test_pmos_polarity_and_jfet_derivatives_regions():
    pmos = Level1MOSFET("PMOS", VTO_V=-1.0, KP_A_V2=1e-3, CGS_F=1e-12)
    pe = pmos.evaluate_terminals([-4.0, -3.0, 0.0, 0.0])
    assert pe.terminal_currents_A[0] < 0
    assert abs(float(pe.terminal_currents_A.sum())) < 1e-12

    jfet = Level1JFET("NJF", VTO_V=-2.0, BETA_A_V2=1e-3,
                      LAMBDA_per_V=0.01, CGS_F=1e-12, CGD_F=0.5e-12)
    _assert_terminal_derivatives(jfet, [3.0, -0.5, 0.0])
    assert jfet.evaluate_terminals([3.0, -0.5, 0.0]).operating_point["region"] == "saturation"
    assert jfet.evaluate_terminals([0.2, -0.5, 0.0]).operating_point["region"] == "ohmic"
    assert jfet.evaluate_terminals([3.0, -3.0, 0.0]).operating_point["region"] == "cutoff"


def _solve_netlist(text):
    ir = parse_spice_text(text)
    supported, reasons = supports_circuit_ir_nonlinear_dc(ir)
    assert supported, reasons
    return ir, solve_nonlinear_dc(ir)


def test_global_nonlinear_mna_solves_bjt_and_conserves_kcl():
    ir, solution = _solve_netlist("""VCC vcc 0 5
RC vcc c 1k
RB vcc b 100k
Q1 c b 0 QN
.model QN NPN (IS=1e-15 BF=100 BR=1 VAF=100 CJE=2p CJC=1p TF=1n)
.end
""")
    assert solution.device_results["Q1"]["region"] in ("forward_active", "saturation")
    assert abs(sum(solution.device_results["Q1"]["terminal_currents_A"])) < 1e-10
    assert solution.verification["kcl"]["is_balanced"]


def test_global_nonlinear_mna_solves_mosfet_and_jfet():
    _, mos = _solve_netlist("""VDD dd 0 5
RD dd d 1k
VG g 0 3
M1 d g 0 0 MN W=1u L=1u
.model MN NMOS (VTO=1 KP=1m LAMBDA=0.02 GAMMA=0.3 PHI=0.6)
.end
""")
    assert mos.device_results["M1"]["region"] in ("linear", "saturation")
    assert mos.verification["kcl"]["is_balanced"]

    _, jfet = _solve_netlist("""VDD dd 0 5
RD dd d 1k
VG g 0 -1
J1 d g 0 JN
.model JN NJF (VTO=-2 BETA=1m LAMBDA=0.01)
.end
""")
    assert jfet.device_results["J1"]["region"] == "saturation"
    assert jfet.verification["kcl"]["is_balanced"]


def test_nonlinear_ac_uses_operating_point_jacobian_and_charge_terms():
    ir = parse_spice_text("""VIN in 0 DC 5 AC 1
R1 in out 1k
D1 out 0 DM
.model DM D (IS=1e-14 N=1 CJO=1n VJ=0.7 M=0.5)
.end
""")
    op = solve_nonlinear_dc(ir)
    matrix, rhs, node_index, _ = assemble_nonlinear_ac(ir, op, 1e3)
    assert matrix.dtype.kind == "c"
    assert np.any(np.abs(matrix.data.imag) > 0.0)
    assert matrix.shape[0] == len(node_index) + 1  # voltage-source branch unknown
    ac = solve_nonlinear_ac(ir, operating_point=op, frequencies_hz=[0.0, 1e3, 1e5])
    assert ac["solver_diagnostics"]["converged"]
    assert len(ac["frequency_diagnostics"]) == 3
    assert all(item["status"] == "success" for item in ac["frequency_diagnostics"])


def test_nonlinear_solver_never_reports_convergence_at_iteration_limit():
    ir = parse_spice_text("""V1 in 0 5
R1 in out 1k
D1 out 0 DM
.model DM D (IS=1e-14 N=1)
.end
""")
    with pytest.raises(Exception, match="did not converge|line search"):
        solve_nonlinear_dc(ir, max_iterations=1)


def test_unsupported_physical_parameters_are_rejected_not_ignored():
    ir = parse_spice_text("""V1 in 0 5
R1 in out 1k
D1 out 0 DM RS=10
.model DM D (IS=1e-14 N=1)
.end
""")
    supported, reasons = supports_circuit_ir_nonlinear_dc(ir)
    assert not supported
    assert any("RS" in reason for reason in reasons)


def test_polarity_specific_model_defaults_for_pmos_and_pjf():
    _, pmos_solution = _solve_netlist("""VDD vdd 0 -5
RD vdd d 1k
VG g 0 -3
M1 d g 0 0 MP W=1u L=1u
.model MP PMOS (KP=1m)
.end
""")
    assert pmos_solution.device_results["M1"]["VTH_internal_V"] > 0
    assert pmos_solution.device_results["M1"]["region"] in ("linear", "saturation")

    _, pjfet_solution = _solve_netlist("""VDD dd 0 -5
RD dd d 1k
VG g 0 1
J1 d g 0 JP
.model JP PJF (BETA=1m)
.end
""")
    assert pjfet_solution.device_results["J1"]["pinch_off_V"] < 0
    assert pjfet_solution.device_results["J1"]["region"] in ("ohmic", "saturation")


def test_public_api_routes_shared_nonlinear_dc_and_ac(tmp_path):
    from circhecker import CircuitReviewer

    netlist = """VIN in 0 DC 5 AC 1
R1 in out 1k
D1 out 0 DM
.model DM D (IS=1e-14 N=1 CJO=1n)
.ac dec 3 1k 10k
.end
"""
    reviewer = CircuitReviewer(output_dir=tmp_path, plot=False)
    combined = reviewer.process_netlist_text(
        netlist, analyses=["dc_operating_point", "ac_sweep"], save_json=False,
    )
    assert combined.execution_summary["backend"] == "nonlinear_mna"
    assert combined.analysis_results["dc_operating_point"]["status"] == "success"
    assert combined.analysis_results["ac_sweep"]["status"] == "success"
    assert combined.analysis_results["ac_sweep"]["data"]["interpretation"].startswith("Small-signal nonlinear MNA")

    dc_only = reviewer.process_netlist_text(netlist, analyses=["dc_operating_point"], save_json=False)
    assert dc_only.execution_summary["backend"] == "nonlinear_mna_dc"



def test_bjt_operating_point_matches_independent_scalar_reference():
    import math
    from circhecker.solver.devices import BOLTZMANN_J_PER_K, ELEMENTARY_CHARGE_C

    ir, solution = _solve_netlist("""VCC vcc 0 5
RC vcc c 1k
RB vcc b 100k
Q1 c b 0 QN
.model QN NPN (IS=1e-15 BF=100 BR=1 VAF=100)
.end
""")
    vt = BOLTZMANN_J_PER_K * 300.15 / ELEMENTARY_CHARGE_C
    alpha = 100.0 / 101.0
    expected_vb = brentq(
        lambda value: (5.0 - value) / 100e3
        - (1.0 - alpha) * 1e-15 * math.expm1(value / vt) * (1.0 + value / 100.0),
        0.5, 0.9,
    )
    expected_ic = alpha * 1e-15 * math.expm1(expected_vb / vt) * (1.0 + expected_vb / 100.0)
    assert solution.node_voltages_V[3] == pytest.approx(expected_vb, abs=2e-9)
    assert solution.device_results["Q1"]["IC_A"] == pytest.approx(expected_ic, rel=2e-7)


def test_mosfet_and_jfet_operating_points_match_independent_scalar_references():
    _, mos = _solve_netlist("""VDD dd 0 5
RD dd d 1k
VG g 0 3
M1 d g 0 0 MN W=1u L=1u
.model MN NMOS (VTO=1 KP=1m LAMBDA=0.02)
.end
""")
    expected_mos_vd = brentq(
        lambda value: (5.0 - value) / 1e3 - 0.5 * 1e-3 * (3.0 - 1.0) ** 2 * (1.0 + 0.02 * value),
        2.0, 5.0,
    )
    assert mos.node_voltages_V[2] == pytest.approx(expected_mos_vd, abs=2e-9)

    _, jfet = _solve_netlist("""VDD dd 0 5
RD dd d 1k
VG g 0 -1
J1 d g 0 JN
.model JN NJF (VTO=-2 BETA=1m LAMBDA=0.01)
.end
""")
    expected_jfet_vd = brentq(
        lambda value: (5.0 - value) / 1e3 - 1e-3 * (1.0 - 1e-2 * 0.0) ** 2 * (1.0 + 0.01 * value),
        1.0, 5.0,
    )
    assert jfet.node_voltages_V[2] == pytest.approx(expected_jfet_vd, abs=2e-9)


def test_nonlinear_ac_diode_matches_closed_form_small_signal_reference():
    ir = parse_spice_text("""VIN in 0 DC 5 AC 1
R1 in out 1k
D1 out 0 DM
.model DM D (IS=1e-14 N=1 CJO=1n VJ=0.7 M=0.5)
.end
""")
    op = solve_nonlinear_dc(ir)
    ac = solve_nonlinear_ac(ir, operating_point=op, frequencies_hz=[1e3, 1e5])
    vout = ac["node_voltages_V"]["out"]
    from circhecker.solver.devices import ShockleyDiode
    diode = ShockleyDiode(1e-14, 1.0, 300.15, nominal_temperature_K=300.15, CJO_F=1e-9)
    voltage = op.node_voltages_V[2]
    evaluation = diode.evaluate_terminals([voltage, 0.0])
    conductance = evaluation.current_jacobian_S[0, 0]
    capacitance = evaluation.charge_jacobian_F[0, 0]
    for actual, frequency in zip(vout, (1e3, 1e5)):
        expected = 1.0 / (1.0 + 1e3 * (conductance + 1j * 2.0 * np.pi * frequency * capacitance))
        assert actual == pytest.approx(expected, rel=2e-8, abs=1e-10)



def test_pnp_pmos_and_pjfet_analytic_derivatives_and_power_consistency():
    reverse_polarity_models = [
        (GummelPoonBJT("PNP", IS_A=2e-15, BF=120, BR=2, VAF_V=90, VAR_V=35,
                       CJE_F=3e-12, CJC_F=1e-12, TF_s=0.8e-9, TR_s=0.2e-9),
         [-4.5, -0.68, 0.0]),
        (Level1MOSFET("PMOS", VTO_V=-1.0, KP_A_V2=1e-3, LAMBDA_per_V=0.02,
                      GAMMA_sqrtV=0.35, PHI_V=0.7, CGS_F=1.2e-12,
                      CGD_F=0.7e-12, CGB_F=0.2e-12),
         [-4.0, -3.0, -0.4, 0.0]),
        (Level1JFET("PJF", VTO_V=2.0, BETA_A_V2=1e-3, LAMBDA_per_V=0.01,
                    CGS_F=1e-12, CGD_F=0.5e-12),
         [-3.0, 0.5, 0.0]),
    ]
    for model, voltages in reverse_polarity_models:
        _assert_terminal_derivatives(model, voltages, current_rtol=6e-5, charge_rtol=6e-5)
        evaluation = model.evaluate_terminals(voltages)
        terminal_power = float(np.dot(np.asarray(voltages, dtype=float), evaluation.terminal_currents_A))
        assert terminal_power >= -1e-12
        assert abs(float(evaluation.terminal_currents_A.sum())) < 1e-12


@pytest.mark.parametrize("netlist,reference,input_terminal,output_terminal,frequency", [
    ("""VCC vcc 0 5
RC vcc c 1k
VB b 0 DC 0.68 AC 1
Q1 c b 0 QN
.model QN NPN (IS=1e-15 BF=100 BR=1 VAF=100 CJE=2p CJC=1p TF=1n)
.end
""", "Q1", 1, 0, 1e3),
    ("""VDD dd 0 5
RD dd d 1k
VG g 0 DC 3 AC 1
M1 d g 0 0 MN W=1u L=1u
.model MN NMOS (VTO=1 KP=1m LAMBDA=0.02 CGS=2p CGD=1p CGB=0.5p)
.end
""", "M1", 1, 0, 1e3),
    ("""VDD dd 0 5
RD dd d 1k
VG g 0 DC -1 AC 1
J1 d g 0 JN
.model JN NJF (VTO=-2 BETA=1m LAMBDA=0.01 CGS=2p CGD=1p)
.end
""", "J1", 1, 0, 1e3),
])
def test_transistor_small_signal_ac_matches_terminal_linearization(
    netlist, reference, input_terminal, output_terminal, frequency,
):
    ir = parse_spice_text(netlist)
    op = solve_nonlinear_dc(ir)
    result = solve_nonlinear_ac(ir, operating_point=op, frequencies_hz=[frequency])
    assert result["solver_diagnostics"]["converged"]
    device = op.device_results[reference]
    conductance = np.asarray(device["current_jacobian_S"], dtype=float)
    capacitance = np.asarray(device["charge_jacobian_F"], dtype=float)
    admittance = conductance + 2j * np.pi * frequency * capacitance
    resistance = 1e3
    expected = -admittance[output_terminal, input_terminal] / (
        1.0 / resistance + admittance[output_terminal, output_terminal]
    )
    output_node = "c" if reference == "Q1" else "d"
    actual = result["node_voltages_V"][output_node][0]
    assert actual == pytest.approx(expected, rel=2e-8, abs=2e-10)


@pytest.mark.parametrize("netlist", [
    """VIN in 0 DC 5 AC 1
R1 in out 1k
D1 out 0 DM
.model DM D (IS=1e-14 N=1 CJO=1n)
.ac lin 2 1k 2k
.end
""",
    """VCC vcc 0 DC 5 AC 0
VB b 0 DC 0.68 AC 1
RC vcc c 1k
Q1 c b 0 QN
.model QN NPN (IS=1e-15 BF=100 BR=1 VAF=100 CJE=2p CJC=1p TF=1n)
.ac lin 2 1k 2k
.end
""",
    """VDD dd 0 DC 5 AC 0
VG g 0 DC 3 AC 1
RD dd d 1k
M1 d g 0 0 MN W=1u L=1u
.model MN NMOS (VTO=1 KP=1m LAMBDA=0.02 CGS=2p CGD=1p CGB=0.5p)
.ac lin 2 1k 2k
.end
""",
    """VDD dd 0 DC 5 AC 0
VG g 0 DC -1 AC 1
RD dd d 1k
J1 d g 0 JN
.model JN NJF (VTO=-2 BETA=1m LAMBDA=0.01 CGS=2p CGD=1p)
.ac lin 2 1k 2k
.end
""",
])
def test_public_api_routes_all_supported_nonlinear_families_to_shared_dc_ac(tmp_path, netlist):
    from circhecker import CircuitReviewer

    result = CircuitReviewer(output_dir=tmp_path, plot=False).process_netlist_text(
        netlist, analyses=["dc_operating_point", "ac_sweep"], save_json=False,
    )
    assert result.execution_summary["backend"] == "nonlinear_mna"
    assert result.analysis_results["dc_operating_point"]["status"] == "success"
    assert result.analysis_results["ac_sweep"]["status"] == "success"


def test_bjt_cutoff_saturation_and_reverse_active_regions():
    bjt = GummelPoonBJT(
        "NPN", IS_A=2e-15, BF=120, BR=2, VAF_V=90, VAR_V=35,
        CJE_F=3e-12, CJC_F=1e-12, TF_s=0.8e-9, TR_s=0.2e-9,
    )
    assert bjt.evaluate_terminals([0.0, 0.0, 0.0]).operating_point["region"] == "cutoff"
    assert bjt.evaluate_terminals([0.5, 0.7, 0.0]).operating_point["region"] == "saturation"
    reverse = bjt.evaluate_terminals([0.0, 0.68, 5.0])
    assert reverse.operating_point["region"] == "reverse_active"
    _assert_terminal_derivatives(bjt, [0.0, 0.68, 5.0], current_rtol=5e-5, charge_rtol=5e-5)
    assert abs(float(reverse.terminal_currents_A.sum())) < 1e-12


def test_bjt_temperature_scaling_is_explicit_and_finite():
    ambient = GummelPoonBJT(
        "NPN", IS_A=1e-15, temperature_K=300.15, nominal_temperature_K=300.15,
    )
    hot = GummelPoonBJT(
        "NPN", IS_A=1e-15, temperature_K=373.15, nominal_temperature_K=300.15,
    )
    assert hot.thermal_voltage_V > ambient.thermal_voltage_V
    assert hot.effective_saturation_current_A > ambient.effective_saturation_current_A
    assert np.isfinite(hot.evaluate_terminals([5.0, 0.68, 0.0]).terminal_currents_A).all()


def test_level1_mosfet_channel_current_scales_with_instance_width_over_length():
    unit = Level1MOSFET("NMOS", VTO_V=1.0, KP_A_V2=1e-3, W_m=1e-6, L_m=1e-6)
    wider = Level1MOSFET("NMOS", VTO_V=1.0, KP_A_V2=1e-3, W_m=4e-6, L_m=1e-6)
    narrow = Level1MOSFET("NMOS", VTO_V=1.0, KP_A_V2=1e-3, W_m=1e-6, L_m=2e-6)
    bias = [4.0, 3.0, 0.0, 0.0]
    i_unit = float(unit.evaluate_terminals(bias).terminal_currents_A[0])
    i_wide = float(wider.evaluate_terminals(bias).terminal_currents_A[0])
    i_narrow = float(narrow.evaluate_terminals(bias).terminal_currents_A[0])
    assert i_wide == pytest.approx(4.0 * i_unit, rel=1e-12)
    assert i_narrow == pytest.approx(0.5 * i_unit, rel=1e-12)
    _assert_terminal_derivatives(wider, bias)
