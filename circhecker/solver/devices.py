"""Constitutive semiconductor device models for the nonlinear MNA engine.

Terminal convention
-------------------
Every device uses the parser's canonical terminal order.  Terminal current is
positive *into* each device terminal.  A charge-conserving device satisfies
``sum(i_terminal) == 0`` and ``sum(q_terminal) == 0``.  The returned Jacobians
are derivatives with respect to the terminal voltages in that same order.

The models here are deliberately compact, documented subsets.  They are not
claimed to be full SPICE implementations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Protocol, Sequence

import numpy as np

BOLTZMANN_J_PER_K = 1.380649e-23
ELEMENTARY_CHARGE_C = 1.602176634e-19
BOLTZMANN_EV_PER_K = 8.617333262145e-5


@dataclass(frozen=True)
class DeviceEvaluation:
    """Terminal constitutive values and local derivatives.

    ``current_jacobian_S[i, j]`` is d(I_i)/d(V_j), and
    ``charge_jacobian_F[i, j]`` is d(Q_i)/d(V_j).  Charges and their Jacobian
    are ``None`` when a model does not implement charge storage.
    """

    terminal_currents_A: np.ndarray
    current_jacobian_S: np.ndarray
    terminal_charges_C: np.ndarray | None = None
    charge_jacobian_F: np.ndarray | None = None
    operating_point: Mapping[str, float | str | bool] = field(default_factory=dict)
    convergence_info: Mapping[str, float | str | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        currents = np.asarray(self.terminal_currents_A, dtype=float)
        jacobian = np.asarray(self.current_jacobian_S, dtype=float)
        if currents.ndim != 1 or jacobian.shape != (currents.size, currents.size):
            raise ValueError("Device current vector and Jacobian dimensions do not match.")
        if not np.all(np.isfinite(currents)) or not np.all(np.isfinite(jacobian)):
            raise ValueError("Device current evaluation contains non-finite values.")
        object.__setattr__(self, "terminal_currents_A", currents)
        object.__setattr__(self, "current_jacobian_S", jacobian)
        if self.terminal_charges_C is not None:
            charges = np.asarray(self.terminal_charges_C, dtype=float)
            qjac = np.asarray(self.charge_jacobian_F, dtype=float)
            if charges.shape != currents.shape or qjac.shape != jacobian.shape:
                raise ValueError("Device charge vector and Jacobian dimensions do not match.")
            if not np.all(np.isfinite(charges)) or not np.all(np.isfinite(qjac)):
                raise ValueError("Device charge evaluation contains non-finite values.")
            object.__setattr__(self, "terminal_charges_C", charges)
            object.__setattr__(self, "charge_jacobian_F", qjac)
        elif self.charge_jacobian_F is not None:
            raise ValueError("A charge Jacobian cannot be supplied without terminal charges.")


class NonlinearDevice(Protocol):
    """Pure terminal-voltage device evaluation contract."""

    terminal_count: int

    def evaluate_terminals(self, terminal_voltages_V: Sequence[float]) -> DeviceEvaluation:
        """Evaluate terminal currents, derivatives and optional charges."""


def _finite_positive(name: str, value: float, *, allow_zero: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result) or (result < 0 if allow_zero else result <= 0):
        bound = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {bound}.")
    return result


def _limited_exponential(voltage_V: float, thermal_voltage_V: float,
                         saturation_current_A: float,
                         min_exponent: float = -80.0,
                         max_exponent: float = 40.0) -> tuple[float, float, float]:
    """Return current, derivative, and reported exponent for a C1 upper limit.

    Below the lower bound the reverse current is held at its asymptotic value;
    above the upper bound the exponential is continued with its tangent.  The
    derivative returned is exactly the derivative of this piecewise current.
    """
    exponent = float(voltage_V) / float(thermal_voltage_V)
    if exponent < min_exponent:
        exp_value, derivative_factor, reported = math.exp(min_exponent), 0.0, min_exponent
    elif exponent > max_exponent:
        boundary = math.exp(max_exponent)
        exp_value, derivative_factor, reported = boundary * (1.0 + exponent - max_exponent), boundary, max_exponent
    else:
        exp_value, derivative_factor, reported = math.exp(exponent), math.exp(exponent), exponent
    current = saturation_current_A * (exp_value - 1.0)
    conductance = saturation_current_A * derivative_factor / thermal_voltage_V
    if not math.isfinite(current) or not math.isfinite(conductance):
        raise ValueError("Semiconductor exponential evaluation exceeded finite numerical range.")
    return current, conductance, reported


def _junction_charge(voltage_V: float, capacitance_F: float, potential_V: float,
                     grading_coefficient: float, forward_coefficient: float = 0.5) -> tuple[float, float]:
    """Abrupt/graded depletion charge with a linear forward-bias continuation.

    Integrates C(V)=C0*(1-V/Phi)^(-m) up to FC*Phi. Above that point,
    charge is continued linearly with the capacitance at the boundary.  This
    avoids evaluating a fractional power outside the depletion-model domain.
    """
    if capacitance_F == 0.0:
        return 0.0, 0.0
    if not (0.0 <= grading_coefficient < 1.0):
        raise ValueError("Junction grading coefficient must be in [0, 1).")
    if not (0.0 < forward_coefficient < 1.0):
        raise ValueError("Junction forward coefficient must be in (0, 1).")
    v_limit = forward_coefficient * potential_V
    if voltage_V <= v_limit:
        base = 1.0 - voltage_V / potential_V
        if base <= 0.0:
            raise ValueError("Junction depletion model reached an invalid voltage domain.")
        cap = capacitance_F * base ** (-grading_coefficient)
        if abs(1.0 - grading_coefficient) < 1e-12:
            charge = -capacitance_F * potential_V * math.log(base)
        else:
            charge = (capacitance_F * potential_V / (1.0 - grading_coefficient)
                      * (1.0 - base ** (1.0 - grading_coefficient)))
        return charge, cap
    base_fc = 1.0 - forward_coefficient
    cap_fc = capacitance_F * base_fc ** (-grading_coefficient)
    if abs(1.0 - grading_coefficient) < 1e-12:
        q_fc = -capacitance_F * potential_V * math.log(base_fc)
    else:
        q_fc = (capacitance_F * potential_V / (1.0 - grading_coefficient)
                * (1.0 - base_fc ** (1.0 - grading_coefficient)))
    return q_fc + cap_fc * (voltage_V - v_limit), cap_fc


@dataclass(frozen=True)
class DiodeEvaluation:
    current_A: float
    conductance_S: float
    voltage_V: float
    limited_exponent: float


@dataclass(frozen=True)
class ShockleyDiode:
    """Temperature-aware Shockley diode with optional charge storage.

    Conduction is ``IS(T) * (exp(V/(N*VT))-1)`` with a bounded exponential.
    ``nominal_temperature_K=None`` preserves the original CirKreety behaviour
    where IS is treated as already resolved at the requested temperature.  If
    supplied, the standard XTI/EG temperature scaling is applied.

    Optional ``CJO``, ``VJ``, ``M``, ``FC`` and ``TT`` parameters add depletion
    plus transit-time diffusion charge. Optional ``BV`` and ``RBV`` add a
    piecewise-linear reverse-breakdown branch for educational Zener analysis;
    this is not the complete SPICE diode breakdown formulation. Series
    resistance is not included.
    """

    saturation_current_A: float = 1e-14
    ideality_factor: float = 1.0
    temperature_K: float = 300.15
    min_exponent: float = -80.0
    max_exponent: float = 40.0
    nominal_temperature_K: float | None = None
    XTI: float = 3.0
    EG_eV: float = 1.11
    CJO_F: float = 0.0
    VJ_V: float = 0.7
    M: float = 0.5
    FC: float = 0.5
    TT_s: float = 0.0
    BV_V: float | None = None
    RBV_Ohm: float = 5.0

    terminal_count: int = 2

    def __post_init__(self) -> None:
        _finite_positive("Diode IS", self.saturation_current_A)
        _finite_positive("Diode ideality factor", self.ideality_factor)
        _finite_positive("Diode temperature", self.temperature_K)
        _finite_positive("Diode CJO", self.CJO_F, allow_zero=True)
        _finite_positive("Diode VJ", self.VJ_V)
        _finite_positive("Diode TT", self.TT_s, allow_zero=True)
        _finite_positive("Diode RBV", self.RBV_Ohm)
        if self.BV_V is not None:
            _finite_positive("Diode BV", self.BV_V)
        if not (0.0 <= self.M < 1.0):
            raise ValueError("Diode M must be in [0, 1).")
        if not (0.0 < self.FC < 1.0):
            raise ValueError("Diode FC must be in (0, 1).")
        if self.min_exponent >= self.max_exponent:
            raise ValueError("Diode exponent limits must be ordered.")
        if self.nominal_temperature_K is not None:
            _finite_positive("Diode nominal temperature", self.nominal_temperature_K)
        if not math.isfinite(float(self.XTI)) or not math.isfinite(float(self.EG_eV)):
            raise ValueError("Diode XTI and EG must be finite.")

    @property
    def thermal_voltage_V(self) -> float:
        return BOLTZMANN_J_PER_K * self.temperature_K / ELEMENTARY_CHARGE_C

    @property
    def effective_saturation_current_A(self) -> float:
        if self.nominal_temperature_K is None:
            return self.saturation_current_A
        t, tn = self.temperature_K, self.nominal_temperature_K
        exponent = -self.EG_eV / BOLTZMANN_EV_PER_K * (1.0 / t - 1.0 / tn)
        value = self.saturation_current_A * (t / tn) ** self.XTI * math.exp(exponent)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("Temperature-adjusted diode saturation current is invalid.")
        return value

    def evaluate(self, voltage_V: float) -> DiodeEvaluation:
        voltage = float(voltage_V)
        if not math.isfinite(voltage):
            raise ValueError("Diode terminal voltage must be finite.")
        vt_n = self.ideality_factor * self.thermal_voltage_V
        if self.BV_V is not None and voltage < -self.BV_V:
            # Continuous piecewise-linear Zener breakdown branch. Current is
            # defined positive from anode to cathode, so reverse breakdown is
            # negative for anode-to-cathode voltage below -BV.
            current = -self.effective_saturation_current_A + (voltage + self.BV_V) / self.RBV_Ohm
            conductance = 1.0 / self.RBV_Ohm
            reported = self.min_exponent
        else:
            current, conductance, reported = _limited_exponential(
                voltage, vt_n, self.effective_saturation_current_A,
                self.min_exponent, self.max_exponent,
            )
        return DiodeEvaluation(float(current), float(conductance), voltage, float(reported))

    def evaluate_terminals(self, terminal_voltages_V: Sequence[float]) -> DeviceEvaluation:
        values = np.asarray(terminal_voltages_V, dtype=float)
        if values.shape != (2,) or not np.all(np.isfinite(values)):
            raise ValueError("Diode requires two finite terminal voltages in (anode, cathode) order.")
        voltage = float(values[0] - values[1])
        evaluation = self.evaluate(voltage)
        g = evaluation.conductance_S
        currents = np.array([evaluation.current_A, -evaluation.current_A])
        jac = np.array([[g, -g], [-g, g]])
        qj, cj = _junction_charge(voltage, self.CJO_F, self.VJ_V, self.M, self.FC)
        qdiff = self.TT_s * evaluation.current_A
        cdiff = self.TT_s * g
        charge = qj + qdiff
        capacitance = cj + cdiff
        return DeviceEvaluation(
            currents, jac, np.array([charge, -charge]),
            np.array([[capacitance, -capacitance], [-capacitance, capacitance]]),
            {"voltage_V": voltage, "current_A": evaluation.current_A,
             "differential_conductance_S": g, "thermal_voltage_V": self.thermal_voltage_V,
             "saturation_current_A": self.effective_saturation_current_A,
             "limited_exponent": evaluation.limited_exponent},
        )

    def current_from_voltage(self, voltage_V: float) -> float:
        return self.evaluate(voltage_V).current_A

    def differential_conductance(self, voltage_V: float) -> float:
        return self.evaluate(voltage_V).conductance_S


@dataclass(frozen=True)
class GummelPoonBJT:
    """Compact Gummel-Poon-inspired BJT subset (NPN/PNP).

    Implements Ebers-Moll forward/reverse transport, BF/BR base-current
    partition, optional forward/reverse Early modulation (VAF/VAR), base-emitter
    and base-collector depletion capacitance, and TF/TR diffusion charge.
    High-injection roll-off, internal resistances, depletion high-current
    corrections, recombination/leakage currents and many SPICE GP parameters
    are deliberately unsupported.
    Terminal order is ``(collector, base, emitter)``.
    """

    polarity: str = "NPN"
    IS_A: float = 1e-15
    BF: float = 100.0
    BR: float = 1.0
    VAF_V: float | None = None
    VAR_V: float | None = None
    temperature_K: float = 300.15
    nominal_temperature_K: float | None = None
    XTI: float = 3.0
    EG_eV: float = 1.11
    CJE_F: float = 0.0
    CJC_F: float = 0.0
    VJE_V: float = 0.75
    VJC_V: float = 0.75
    MJE: float = 0.33
    MJC: float = 0.33
    FC: float = 0.5
    TF_s: float = 0.0
    TR_s: float = 0.0

    terminal_count: int = 3

    def __post_init__(self) -> None:
        if str(self.polarity).upper() not in ("NPN", "PNP"):
            raise ValueError("BJT polarity must be NPN or PNP.")
        for name, value in (("IS", self.IS_A), ("BF", self.BF), ("BR", self.BR),
                            ("temperature", self.temperature_K), ("VJE", self.VJE_V), ("VJC", self.VJC_V)):
            _finite_positive(name, value)
        for name, value in (("CJE", self.CJE_F), ("CJC", self.CJC_F), ("TF", self.TF_s), ("TR", self.TR_s)):
            _finite_positive(name, value, allow_zero=True)
        for name, value in (("MJE", self.MJE), ("MJC", self.MJC)):
            if not (0.0 <= value < 1.0):
                raise ValueError(f"{name} must be in [0, 1).")
        for name, value in (("VAF", self.VAF_V), ("VAR", self.VAR_V)):
            if value is not None:
                _finite_positive(name, value)
        if self.nominal_temperature_K is not None:
            _finite_positive("BJT nominal temperature", self.nominal_temperature_K)

    @property
    def thermal_voltage_V(self) -> float:
        return BOLTZMANN_J_PER_K * self.temperature_K / ELEMENTARY_CHARGE_C

    @property
    def effective_saturation_current_A(self) -> float:
        if self.nominal_temperature_K is None:
            return self.IS_A
        t, tn = self.temperature_K, self.nominal_temperature_K
        value = self.IS_A * (t / tn) ** self.XTI * math.exp(
            -self.EG_eV / BOLTZMANN_EV_PER_K * (1.0 / t - 1.0 / tn)
        )
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Temperature-adjusted BJT saturation current is invalid.")
        return value

    def evaluate_terminals(self, terminal_voltages_V: Sequence[float]) -> DeviceEvaluation:
        v = np.asarray(terminal_voltages_V, dtype=float)
        if v.shape != (3,) or not np.all(np.isfinite(v)):
            raise ValueError("BJT requires finite (collector, base, emitter) terminal voltages.")
        sign = 1.0 if self.polarity.upper() == "NPN" else -1.0
        vc, vb, ve = sign * v
        vbe, vbc, vce = vb - ve, vb - vc, vc - ve
        vt = self.thermal_voltage_V
        isat = self.effective_saturation_current_A
        f0, gf0, _ = _limited_exponential(vbe, vt, isat)
        r0, gr0, _ = _limited_exponential(vbc, vt, isat)
        sf = 1.0 + (vce / self.VAF_V if self.VAF_V else 0.0)
        sr = 1.0 - (vce / self.VAR_V if self.VAR_V else 0.0)
        if sf <= 0.0 or sr <= 0.0:
            raise ValueError("BJT Early-effect operating point is outside the positive base-charge domain.")
        fwd, rev = f0 * sf, r0 * sr
        df_dbe, df_dce = gf0 * sf, (f0 / self.VAF_V if self.VAF_V else 0.0)
        dr_dbc, dr_dce = gr0 * sr, (-r0 / self.VAR_V if self.VAR_V else 0.0)
        alpha_f, alpha_r = self.BF / (self.BF + 1.0), self.BR / (self.BR + 1.0)
        inorm = np.array([alpha_f * fwd - rev,
                          (1.0 - alpha_f) * fwd + (1.0 - alpha_r) * rev,
                          -fwd + alpha_r * rev])
        d_idv = np.array([
            [alpha_f * df_dbe, -dr_dbc, alpha_f * df_dce - dr_dce],
            [(1.0 - alpha_f) * df_dbe, (1.0 - alpha_r) * dr_dbc,
             (1.0 - alpha_f) * df_dce + (1.0 - alpha_r) * dr_dce],
            [-df_dbe, alpha_r * dr_dbc, -df_dce + alpha_r * dr_dce],
        ])
        voltage_map = np.array([[0.0, 1.0, -1.0], [-1.0, 1.0, 0.0], [1.0, 0.0, -1.0]])
        currents = sign * inorm
        jac = d_idv @ voltage_map

        qbe_dep, cbe_dep = _junction_charge(vbe, self.CJE_F, self.VJE_V, self.MJE, self.FC)
        qbc_dep, cbc_dep = _junction_charge(vbc, self.CJC_F, self.VJC_V, self.MJC, self.FC)
        qbe = qbe_dep + self.TF_s * f0
        qbc = qbc_dep + self.TR_s * r0
        dqbe = cbe_dep + self.TF_s * gf0
        dqbc = cbc_dep + self.TR_s * gr0
        qnorm = np.array([-qbc, qbe + qbc, -qbe])
        qmap = np.array([[0.0, -1.0], [1.0, 1.0], [-1.0, 0.0]])
        qjac = qmap @ np.diag([dqbe, dqbc]) @ np.array([[0.0, 1.0, -1.0], [-1.0, 1.0, 0.0]])
        return DeviceEvaluation(
            currents, jac, sign * qnorm, qjac,
            {"VBE_V": vbe, "VBC_V": vbc, "VCE_V": vce, "IC_A": float(currents[0]),
             "IB_A": float(currents[1]), "IE_A": float(currents[2]),
             "forward_transport_A": fwd, "reverse_transport_A": rev,
             "region": "forward_active" if vbe > 0.0 and vbc <= 0.0 else
                       "reverse_active" if vbc > 0.0 and vbe <= 0.0 else
                       "saturation" if vbe > 0.0 and vbc > 0.0 else "cutoff"},
        )


@dataclass(frozen=True)
class Level1MOSFET:
    """Square-law Level-1 MOSFET subset with body effect and charge caps.

    Terminal order is ``(drain, gate, source, bulk)``.  The model includes
    cutoff, triode and saturation regions, body-effect threshold shift, channel
    length modulation, body-junction Shockley currents and optional linear
    overlap capacitances CGSO/CGDO/CGBO (scaled by W).  It is not BSIM and does
    not include mobility degradation, velocity saturation, subthreshold
    conduction, detailed nonlinear channel charge or drain/source resistances.
    """

    polarity: str = "NMOS"
    VTO_V: float = 1.0
    KP_A_V2: float = 1e-3
    LAMBDA_per_V: float = 0.0
    GAMMA_sqrtV: float = 0.0
    PHI_V: float = 0.6
    IS_A: float = 1e-14
    temperature_K: float = 300.15
    W_m: float = 1e-6
    L_m: float = 1e-6
    CGSO_F_per_m: float = 0.0
    CGDO_F_per_m: float = 0.0
    CGBO_F_per_m: float = 0.0
    CGS_F: float = 0.0
    CGD_F: float = 0.0
    CGB_F: float = 0.0
    VJ_V: float = 0.7
    M: float = 0.5
    FC: float = 0.5

    terminal_count: int = 4

    def __post_init__(self) -> None:
        if str(self.polarity).upper() not in ("NMOS", "PMOS"):
            raise ValueError("MOSFET polarity must be NMOS or PMOS.")
        for name, value in (("KP", self.KP_A_V2), ("PHI", self.PHI_V), ("IS", self.IS_A),
                            ("temperature", self.temperature_K), ("W", self.W_m), ("L", self.L_m),
                            ("VJ", self.VJ_V)):
            _finite_positive(name, value)
        for name, value in (("GAMMA", self.GAMMA_sqrtV), ("LAMBDA", self.LAMBDA_per_V),
                            ("CGSO", self.CGSO_F_per_m), ("CGDO", self.CGDO_F_per_m),
                            ("CGBO", self.CGBO_F_per_m), ("CGS", self.CGS_F),
                            ("CGD", self.CGD_F), ("CGB", self.CGB_F)):
            _finite_positive(name, value, allow_zero=True)
        if not (0.0 <= self.M < 1.0):
            raise ValueError("MOSFET junction grading M must be in [0, 1).")

    def evaluate_terminals(self, terminal_voltages_V: Sequence[float]) -> DeviceEvaluation:
        v = np.asarray(terminal_voltages_V, dtype=float)
        if v.shape != (4,) or not np.all(np.isfinite(v)):
            raise ValueError("MOSFET requires finite (drain, gate, source, bulk) voltages.")
        sign = 1.0 if self.polarity.upper() == "NMOS" else -1.0
        u = sign * v
        vd, vg, vs, vb = u
        hi, lo = (0, 2) if vd >= vs else (2, 0)
        vds = float(u[hi] - u[lo])
        vgs = float(vg - u[lo])
        vsb = float(u[lo] - vb)
        body_argument = self.PHI_V + vsb
        if body_argument <= 0.0:
            raise ValueError("MOSFET body-effect square-root domain PHI+VSB must be positive.")
        vth = sign * self.VTO_V + self.GAMMA_sqrtV * (math.sqrt(body_argument) - math.sqrt(self.PHI_V))
        dvth_dvsb = self.GAMMA_sqrtV / (2.0 * math.sqrt(body_argument))
        overdrive = vgs - vth
        ids = 0.0
        did_overdrive = 0.0
        did_vds = 0.0
        region = "cutoff"
        kp_effective = self.KP_A_V2 * self.W_m / self.L_m
        if overdrive > 0.0:
            if vds < overdrive:
                base = kp_effective * (overdrive * vds - 0.5 * vds * vds)
                dbase_over = kp_effective * vds
                dbase_vds = kp_effective * (overdrive - vds)
                region = "linear"
            else:
                base = 0.5 * kp_effective * overdrive * overdrive
                dbase_over = kp_effective * overdrive
                dbase_vds = 0.0
                region = "saturation"
            factor = 1.0 + self.LAMBDA_per_V * vds
            if factor <= 0.0:
                raise ValueError("MOSFET channel-length modulation produced a non-positive current factor.")
            ids = base * factor
            did_overdrive = dbase_over * factor
            did_vds = dbase_vds * factor + base * self.LAMBDA_per_V
        grad_over = np.zeros(4)
        grad_over[1] = 1.0
        grad_over[lo] = -1.0 - dvth_dvsb
        grad_over[3] = dvth_dvsb
        grad_vds = np.zeros(4)
        grad_vds[hi], grad_vds[lo] = 1.0, -1.0
        grad_ids = did_overdrive * grad_over + did_vds * grad_vds
        inorm = np.zeros(4)
        jnorm = np.zeros((4, 4))
        inorm[hi], inorm[lo] = ids, -ids
        jnorm[hi, :] = grad_ids
        jnorm[lo, :] = -grad_ids

        # Intrinsic body diodes: bulk -> drain and bulk -> source in the
        # n-channel-equivalent coordinates.  Their currents conserve KCL.
        vt = BOLTZMANN_J_PER_K * self.temperature_K / ELEMENTARY_CHARGE_C
        for channel_terminal in (0, 2):
            current, conductance, _ = _limited_exponential(
                float(u[3] - u[channel_terminal]), vt, self.IS_A
            )
            inorm[3] += current
            inorm[channel_terminal] -= current
            jnorm[3, 3] += conductance
            jnorm[3, channel_terminal] -= conductance
            jnorm[channel_terminal, 3] -= conductance
            jnorm[channel_terminal, channel_terminal] += conductance

        cgs = self.CGS_F if self.CGS_F else self.CGSO_F_per_m * self.W_m
        cgd = self.CGD_F if self.CGD_F else self.CGDO_F_per_m * self.W_m
        cgb = self.CGB_F if self.CGB_F else self.CGBO_F_per_m * self.W_m
        qnorm = np.zeros(4)
        qjac = np.zeros((4, 4))
        for other, cap in ((2, cgs), (0, cgd), (3, cgb)):
            delta = float(u[1] - u[other])
            q = cap * delta
            qnorm[1] += q
            qnorm[other] -= q
            qjac[1, 1] += cap
            qjac[1, other] -= cap
            qjac[other, 1] -= cap
            qjac[other, other] += cap
        currents = sign * inorm
        # d(sign*I(sign*V))/dV = dI/du.
        return DeviceEvaluation(
            currents, jnorm, sign * qnorm, qjac,
            {"VGS_internal_V": vgs, "VDS_internal_V": vds, "VSB_internal_V": vsb,
             "VTH_internal_V": vth, "overdrive_V": overdrive,
             "channel_current_A": sign * ids, "region": region,
             "body_diode_current_D_A": sign * (inorm[0] - (ids if hi == 0 else -ids)),
             "body_diode_current_S_A": sign * (inorm[2] - (-ids if lo == 2 else ids))},
        )


@dataclass(frozen=True)
class Level1JFET:
    """Square-law JFET subset with reverse/forward gate-junction current.

    Terminal order is ``(drain, gate, source)``.  Channel equations use the
    Shockley/Level-1 pinch-off square law with an ohmic region, saturation and
    optional channel-length modulation.  Two gate-channel junctions use the
    bounded Shockley law.  Optional CGS/CGD are linear charge storage.  Gate
    avalanche, gate-current breakdown and detailed SPICE JFET corrections are
    unsupported.
    """

    polarity: str = "NJF"
    VTO_V: float = -2.0
    BETA_A_V2: float = 1e-3
    LAMBDA_per_V: float = 0.0
    IS_A: float = 1e-14
    temperature_K: float = 300.15
    CGS_F: float = 0.0
    CGD_F: float = 0.0

    terminal_count: int = 3

    def __post_init__(self) -> None:
        if str(self.polarity).upper() not in ("NJF", "PJF"):
            raise ValueError("JFET polarity must be NJF or PJF.")
        _finite_positive("JFET BETA", self.BETA_A_V2)
        _finite_positive("JFET IS", self.IS_A)
        _finite_positive("JFET temperature", self.temperature_K)
        _finite_positive("JFET CGS", self.CGS_F, allow_zero=True)
        _finite_positive("JFET CGD", self.CGD_F, allow_zero=True)
        _finite_positive("JFET LAMBDA", self.LAMBDA_per_V, allow_zero=True)

    def evaluate_terminals(self, terminal_voltages_V: Sequence[float]) -> DeviceEvaluation:
        v = np.asarray(terminal_voltages_V, dtype=float)
        if v.shape != (3,) or not np.all(np.isfinite(v)):
            raise ValueError("JFET requires finite (drain, gate, source) voltages.")
        sign = 1.0 if self.polarity.upper() == "NJF" else -1.0
        u = sign * v
        hi, lo = (0, 2) if u[0] >= u[2] else (2, 0)
        vds = float(u[hi] - u[lo])
        vgs = float(u[1] - u[lo])
        vto = sign * self.VTO_V
        overdrive = vgs - vto
        ids = 0.0
        d_over = 0.0
        d_vds = 0.0
        region = "cutoff"
        if overdrive > 0.0:
            if vds < overdrive:
                base = 2.0 * self.BETA_A_V2 * (overdrive * vds - 0.5 * vds * vds)
                d_over = 2.0 * self.BETA_A_V2 * vds
                d_vds = 2.0 * self.BETA_A_V2 * (overdrive - vds)
                region = "ohmic"
            else:
                base = self.BETA_A_V2 * overdrive * overdrive
                d_over = 2.0 * self.BETA_A_V2 * overdrive
                d_vds = 0.0
                region = "saturation"
            factor = 1.0 + self.LAMBDA_per_V * vds
            if factor <= 0.0:
                raise ValueError("JFET channel-length modulation produced a non-positive current factor.")
            ids = base * factor
            d_over *= factor
            d_vds = d_vds * factor + base * self.LAMBDA_per_V
        grad_over = np.zeros(3)
        grad_over[1], grad_over[lo] = 1.0, -1.0
        grad_vds = np.zeros(3)
        grad_vds[hi], grad_vds[lo] = 1.0, -1.0
        grad_ids = d_over * grad_over + d_vds * grad_vds
        inorm = np.zeros(3)
        jnorm = np.zeros((3, 3))
        inorm[hi], inorm[lo] = ids, -ids
        jnorm[hi, :], jnorm[lo, :] = grad_ids, -grad_ids

        # Gate p-n junction currents are oriented from gate into each channel
        # terminal (NJF convention).  This gives the expected small reverse
        # leakage for ordinary negative VGS operation.
        vt = BOLTZMANN_J_PER_K * self.temperature_K / ELEMENTARY_CHARGE_C
        for channel_terminal in (0, 2):
            current, conductance, _ = _limited_exponential(
                float(u[1] - u[channel_terminal]), vt, self.IS_A
            )
            inorm[1] += current
            inorm[channel_terminal] -= current
            jnorm[1, 1] += conductance
            jnorm[1, channel_terminal] -= conductance
            jnorm[channel_terminal, 1] -= conductance
            jnorm[channel_terminal, channel_terminal] += conductance

        qnorm = np.zeros(3)
        qjac = np.zeros((3, 3))
        for channel_terminal, cap in ((2, self.CGS_F), (0, self.CGD_F)):
            q = cap * float(u[1] - u[channel_terminal])
            qnorm[1] += q
            qnorm[channel_terminal] -= q
            qjac[1, 1] += cap
            qjac[1, channel_terminal] -= cap
            qjac[channel_terminal, 1] -= cap
            qjac[channel_terminal, channel_terminal] += cap
        return DeviceEvaluation(
            sign * inorm, jnorm, sign * qnorm, qjac,
            {"VGS_internal_V": vgs, "VDS_internal_V": vds, "pinch_off_V": vto,
             "overdrive_V": overdrive, "channel_current_A": sign * ids, "region": region},
        )
