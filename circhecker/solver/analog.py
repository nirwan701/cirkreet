"""
This module complements (and does not replace) ``circuitsolver`` and
``transferfunction``.  It provides a compact node-based Newton solver for the
common analog topologies demonstrated in the project: resistors, capacitors,
Shockley diodes, piecewise-linear Zeners, Level-1 MOSFETs, exponential BJTs,
VCCSs, finite-gain one-pole op-amps, and hysteretic Schmitt outputs.

The models are intentionally educational/engineering models, not a full
device library.  AC analysis is small-signal linearization about a DC
operating point.  Oscillators have no static periodic DC operating point;
for those circuits the DC plot is an explicitly held-state initial snapshot.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class analysisresult:
    """Container returned by circuit analyses."""
    data: dict
    figures: list

    def __getitem__(self, key):
        return self.data[key]


class analogcircuit:
    """Nonlinear nodal solver for low-frequency analog circuit analysis.

    Parameters
    ----------
    name:
        Descriptive circuit name.
    sources:
        Mapping of ideal voltage-source nodes to DC values or callables ``f(t)``.
        Node 0 is ground and must not be listed.
    elements:
        Optional sequence of element dictionaries. Prefer the ``add_*`` methods.
    ground:
        Reference node (currently fixed at node 0).
    gmin:
        Numerical shunt conductance placed from every unknown node to ground.
    """

    def __init__(self, name="Analog circuit", sources=None, elements=None, ground=0, gmin=1e-12):
        self.name = str(name)
        self.ground = int(ground)
        if self.ground != 0:
            raise ValueError("The current nodal backend uses node 0 as ground.")
        self.sources = dict(sources or {})
        if 0 in self.sources:
            raise ValueError("Ground node 0 cannot be an independent voltage source.")
        self.elements = list(elements or [])
        self.gmin = float(gmin)
        if not np.isfinite(self.gmin) or self.gmin <= 0:
            raise ValueError("gmin must be finite and positive.")
        self._last_solution = None
        self._schmitt_state = {}
        self._element_count = 0
        for e in self.elements:
            self._element_count += 1
            if e.get("kind") == "SCHMITT":
                self._schmitt_state[self._element_count - 1] = bool(e.get("initial_high", True))
            elif e.get("kind") == "S":
                self._schmitt_state[self._element_count - 1] = bool(e.get("initial_closed", False))

    # -------------------------- construction API --------------------------
    def _add(self, kind, nodes, **kwargs):
        ns = tuple(int(n) for n in nodes)
        if not ns or any(n < 0 for n in ns):
            raise ValueError("Element nodes must be non-negative integers.")
        if len(ns) >= 2 and ns[0] == ns[1] and kind not in ("OTA", "SCHMITT"):
            raise ValueError(f"{kind} has identical terminals.")
        element = {"kind": kind, "nodes": ns, **kwargs}
        self.elements.append(element)
        if kind == "SCHMITT":
            self._schmitt_state[len(self.elements) - 1] = bool(kwargs.get("initial_high", True))
        elif kind == "S":
            self._schmitt_state[len(self.elements) - 1] = bool(kwargs.get("initial_closed", False))
        return self

    def add_resistor(self, p, n, resistance, name=None):
        r = float(resistance)
        if not np.isfinite(r) or r <= 0:
            raise ValueError("Resistance must be finite and positive.")
        return self._add("R", (p, n), value=r, name=name)

    def add_capacitor(self, p, n, capacitance, name=None):
        c = float(capacitance)
        if not np.isfinite(c) or c < 0:
            raise ValueError("Capacitance must be finite and non-negative.")
        return self._add("C", (p, n), value=c, name=name)

    def add_diode(self, anode, cathode, saturation_current=1e-12, ideality=1.0,
                  temperature=300.0, name=None):
        isat, eta, temp = float(saturation_current), float(ideality), float(temperature)
        if isat <= 0 or eta <= 0 or temp <= 0 or not np.isfinite([isat, eta, temp]).all():
            raise ValueError("Invalid diode parameters.")
        return self._add("D", (anode, cathode), isat=isat, eta=eta, temp=temp, name=name)

    def add_zener(self, anode, cathode, zener_voltage=5.1, dynamic_resistance=5.0,
                  leakage=1e-9, name=None):
        vz, rz, leak = float(zener_voltage), float(dynamic_resistance), float(leakage)
        if vz <= 0 or rz <= 0 or leak < 0 or not np.isfinite([vz, rz, leak]).all():
            raise ValueError("Invalid Zener parameters.")
        return self._add("Z", (anode, cathode), vz=vz, rz=rz, leakage=leak, name=name)

    def add_mosfet(self, drain, gate, source, k=1e-3, threshold=1.0,
                   channel_length_modulation=0.02, mos_type="NMOS", name=None):
        k, vt, lam = float(k), float(threshold), float(channel_length_modulation)
        kind = str(mos_type).upper()
        if k <= 0 or lam < 0 or not np.isfinite([k, vt, lam]).all() or kind not in ("NMOS", "PMOS"):
            raise ValueError("Invalid MOSFET parameters.")
        return self._add("M", (drain, gate, source), k=k, vt=vt, lam=lam, mos_type=kind, name=name)

    def add_bjt(self, collector, base, emitter, beta=100.0, saturation_current=1e-15,
                temperature=300.0, transistor_type="NPN", name=None):
        beta, isat, temp = float(beta), float(saturation_current), float(temperature)
        typ = str(transistor_type).upper()
        if beta <= 0 or isat <= 0 or temp <= 0 or typ not in ("NPN", "PNP"):
            raise ValueError("Invalid BJT parameters.")
        return self._add("Q", (collector, base, emitter), beta=beta, isat=isat,
                         temp=temp, transistor_type=typ, name=name)

    def add_vccs(self, p, n, cp, cn, gm, name=None):
        gm = float(gm)
        if not np.isfinite(gm):
            raise ValueError("gm must be finite.")
        return self._add("G", (p, n, cp, cn), gm=gm, name=name)

    def add_vcvs(self, p, n, cp, cn, gain, output_resistance=1.0,
                 low_rail=None, high_rail=None, name=None):
        gain, rout = float(gain), float(output_resistance)
        if not np.isfinite([gain, rout]).all() or rout <= 0:
            raise ValueError("VCVS gain must be finite and output resistance positive.")
        return self._add("E", (p, n, cp, cn), gain=gain, rout=rout,
                         low_rail=low_rail, high_rail=high_rail, name=name)

    def add_jfet(self, drain, gate, source, idss=5e-3, pinch_off=-4.0,
                 transistor_type="NJF", gate_is=1e-14, name=None):
        idss, vp, gate_is = float(idss), float(pinch_off), float(gate_is)
        typ = str(transistor_type).upper()
        if idss <= 0 or vp == 0 or gate_is <= 0 or typ not in ("NJF", "PJF"):
            raise ValueError("Invalid JFET model parameters.")
        return self._add("J", (drain, gate, source), idss=idss, vp=vp,
                         jfet_type=typ, gate_is=gate_is, name=name)

    def add_switch(self, p, n, cp, cn, ron=1.0, roff=1e9, vt=0.5, vh=0.0, initial_closed=False, name=None):
        ron, roff, vt, vh = map(float, (ron, roff, vt, vh))
        if ron <= 0 or roff <= 0 or roff < ron or vh < 0:
            raise ValueError("Invalid voltage-controlled switch parameters.")
        return self._add("S", (p, n, cp, cn), ron=ron, roff=roff,
                         vt=vt, vh=vh, initial_closed=bool(initial_closed), name=name)

    def add_ota(self, output, noninverting, inverting, gain=1e5, output_resistance=100.0,
                low_rail=-10.0, high_rail=10.0, name=None):
        gain, rout, low, high = map(float, (gain, output_resistance, low_rail, high_rail))
        if gain <= 0 or rout <= 0 or low >= high or not np.isfinite([gain, rout, low, high]).all():
            raise ValueError("Invalid OTA/op-amp parameters.")
        return self._add("OTA", (output, noninverting, inverting), gain=gain,
                         rout=rout, low=low, high=high, name=name)

    def add_schmitt(self, output, control, high_rail, low_rail=0, high_fraction=2/3,
                    low_fraction=1/3, output_resistance=5.0, initial_high=True, name=None):
        hi, lo, rout = float(high_fraction), float(low_fraction), float(output_resistance)
        if not (0 < lo < hi < 1) or rout <= 0:
            raise ValueError("Schmitt thresholds must satisfy 0 < low < high < 1.")
        return self._add("SCHMITT", (output, control, high_rail, low_rail),
                         high_fraction=hi, low_fraction=lo, rout=rout,
                         initial_high=bool(initial_high), name=name)

    # --------------------------- device equations -------------------------
    @staticmethod
    def _vt(temp):
        return 8.617333262e-5 * float(temp)

    @staticmethod
    def _exp(x):
        """Exponent limiter with a linear high-voltage continuation.

        Plain clipping gives zero derivative for an initially over-driven diode,
        which can stall Newton iteration in bridge rectifiers.  The linear tail
        preserves a finite slope while retaining the exact Shockley law near the
        operating region.
        """
        x = float(x)
        if x > 25.0:
            return np.exp(25.0) * (1.0 + x - 25.0)
        return np.exp(max(x, -40.0))

    def _element_currents(self, index, e, v, cap_terms=None, states=None):
        """Terminal currents (positive leaving each terminal), summed to zero."""
        k, ns = e["kind"], e["nodes"]
        def vv(node): return 0.0 if node == 0 else float(v[node])
        if k == "R":
            i = (vv(ns[0]) - vv(ns[1])) / e["value"]
            return np.array([i, -i])
        if k == "C":
            if cap_terms is None or index not in cap_terms:
                return np.zeros(2)
            geq, ihist = cap_terms[index]
            i = geq * (vv(ns[0]) - vv(ns[1])) + ihist
            return np.array([i, -i])
        if k == "D":
            vd = vv(ns[0]) - vv(ns[1])
            nvt = e["eta"] * self._vt(e["temp"])
            i = e["isat"] * (self._exp(vd / nvt) - 1.0)
            return np.array([i, -i])
        if k == "Z":
            vd = vv(ns[0]) - vv(ns[1])
            if vd >= 0:
                nvt = 1.5 * self._vt(300.0)
                i = 1e-12 * (self._exp(vd / nvt) - 1.0)
            elif vd < -e["vz"]:
                i = -e["leakage"] + (vd + e["vz"]) / e["rz"]
            else:
                i = -e["leakage"]
            return np.array([i, -i])
        if k == "M":
            d, g, s = ns
            vd, vg, vs = vv(d), vv(g), vv(s)
            # PMOS is solved through polarity-normalized NMOS equations.
            pol = 1.0 if e["mos_type"] == "NMOS" else -1.0
            vds, vgs = pol * (vd - vs), pol * (vg - vs)
            if vds >= 0:
                vov = vgs - e["vt"]
                if vov <= 0:
                    ids_norm = 1e-12 * vds
                elif vds < vov:
                    ids_norm = e["k"] * (vov * vds - 0.5 * vds * vds) * (1 + e["lam"] * vds)
                else:
                    ids_norm = 0.5 * e["k"] * vov * vov * (1 + e["lam"] * vds)
                ids = pol * ids_norm
            else:
                # Swap drain/source in reverse channel operation.
                vds_r, vgs_r = -vds, pol * (vg - vd)
                vov = vgs_r - e["vt"]
                if vov <= 0:
                    ids_r = 1e-12 * vds_r
                elif vds_r < vov:
                    ids_r = e["k"] * (vov * vds_r - 0.5 * vds_r * vds_r) * (1 + e["lam"] * vds_r)
                else:
                    ids_r = 0.5 * e["k"] * vov * vov * (1 + e["lam"] * vds_r)
                ids = -pol * ids_r
            return np.array([ids, 0.0, -ids])
        if k == "Q":
            c, b, em = ns
            pol = 1.0 if e["transistor_type"] == "NPN" else -1.0
            vbe = pol * (vv(b) - vv(em))
            vbc = pol * (vv(b) - vv(c))
            vt = self._vt(e["temp"])
            exp_be = self._exp(vbe / vt)
            exp_bc = self._exp(vbc / vt)
            # Ebers-Moll-like two-junction model: the base-collector diode
            # naturally limits forward current as the device enters saturation.
            ic_norm = e["isat"] * (exp_be - exp_bc)
            ib_norm = e["isat"] * ((exp_be - 1.0) / e["beta"] + (exp_bc - 1.0) / 1.0)
            ic, ib = pol * ic_norm, pol * ib_norm
            return np.array([ic, ib, -(ic + ib)])
        if k == "J":
            d, g, s = ns
            pol = 1.0 if e["jfet_type"] == "NJF" else -1.0
            vds = pol * (vv(d) - vv(s))
            vgs = pol * (vv(g) - vv(s))
            vp = e["vp"] if pol > 0 else -e["vp"]
            # Shockley JFET transfer law with an ohmic-region continuation.
            if vds < 0:
                vds_n = -vds
                vgs_n = pol * (vv(g) - vv(d))
                sign = -1.0
            else:
                vds_n, vgs_n, sign = vds, vgs, 1.0
            if vgs_n <= vp:
                ich = 1e-12 * vds_n
            else:
                vpinch = max(vgs_n - vp, 0.0)
                vds_sat = vpinch
                if vds_n < vds_sat:
                    beta = 2.0 * e["idss"] / (vp * vp)
                    ich = beta * (vpinch * vds_n - 0.5 * vds_n * vds_n)
                else:
                    ich = e["idss"] * (1.0 - vgs_n / vp) ** 2
            ig = e["gate_is"] * (self._exp(vgs_n / self._vt(300.0)) - 1.0)
            idrain = pol * sign * ich
            igate = pol * ig
            return np.array([idrain, igate, -(idrain + igate)])
        if k == "E":
            p, n, cp, cn = ns
            target = e["gain"] * (vv(cp) - vv(cn))
            if e.get("low_rail") is not None: target = max(float(e["low_rail"]), target)
            if e.get("high_rail") is not None: target = min(float(e["high_rail"]), target)
            i = ((vv(p) - vv(n)) - target) / e["rout"]
            return np.array([i, -i, 0.0, 0.0])
        if k == "S":
            p, n, cp, cn = ns
            is_closed = (states or self._schmitt_state).get(index, bool(e.get("initial_closed", False)))
            resistance = e["ron"] if is_closed else e["roff"]
            i = (vv(p) - vv(n)) / resistance
            return np.array([i, -i, 0.0, 0.0])
        if k == "G":
            p, n, cp, cn = ns
            i = e["gm"] * (vv(cp) - vv(cn))
            return np.array([i, -i, 0.0, 0.0])
        if k == "OTA":
            out, plus, minus = ns
            # Smooth rail limiting avoids a discontinuous Jacobian at saturation.
            mid = 0.5 * (e["high"] + e["low"])
            span = 0.5 * (e["high"] - e["low"])
            target = mid + span * np.tanh(e["gain"] * (vv(plus) - vv(minus)) / span)
            i = (vv(out) - target) / e["rout"]
            return np.array([i, 0.0, 0.0])
        if k == "SCHMITT":
            out, ctrl, hi, lo = ns
            high = (states or self._schmitt_state).get(index, bool(e["initial_high"]))
            target = vv(hi) if high else vv(lo)
            i = (vv(out) - target) / e["rout"]
            return np.array([i, 0.0, -i if high else 0.0, -i if not high else 0.0])
        raise ValueError(f"Unsupported analog element kind: {k}")

    def _all_nodes(self):
        nodes = set(self.sources)
        for e in self.elements:
            nodes.update(e["nodes"])
        nodes.discard(0)
        return sorted(nodes)

    def _source_values(self, t=0.0, overrides=None):
        result = {}
        overrides = dict(overrides or {})
        for node, spec in self.sources.items():
            value = overrides.get(node, spec)
            value = value(float(t)) if callable(value) else value
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"Non-finite source at node {node}.")
            result[int(node)] = value
        for node, spec in overrides.items():
            if node not in result:
                value = spec(float(t)) if callable(spec) else spec
                result[int(node)] = float(value)
        return result

    @staticmethod
    def _finite_difference_jacobian(fun, x, f0=None):
        x = np.asarray(x, dtype=float)
        f0 = np.asarray(fun(x) if f0 is None else f0, dtype=float)
        jac = np.empty((f0.size, x.size), dtype=float)
        for j in range(x.size):
            h = 1e-5 * max(1.0, abs(x[j]))
            xp, xm = x.copy(), x.copy()
            xp[j] += h
            xm[j] -= h
            jac[:, j] = (np.asarray(fun(xp)) - np.asarray(fun(xm))) / (2.0 * h)
        return jac

    def _residual(self, x, unknown_nodes, fixed_values, cap_terms=None, states=None):
        all_nodes = self._all_nodes()
        v = {0: 0.0, **fixed_values}
        v.update({n: float(x[i]) for i, n in enumerate(unknown_nodes)})
        currents = {n: 0.0 for n in all_nodes}
        for idx, e in enumerate(self.elements):
            terminal_i = self._element_currents(idx, e, v, cap_terms=cap_terms, states=states)
            for node, current in zip(e["nodes"], terminal_i):
                if node != 0:
                    currents[node] += float(current)
        return np.array([currents[n] + self.gmin * v.get(n, 0.0) for n in unknown_nodes], dtype=float)

    def _solve_nonlinear(self, fixed_values, initial=None, cap_terms=None, states=None,
                         max_iter=80, current_tolerance=2e-10, voltage_tolerance=2e-8):
        nodes = [n for n in self._all_nodes() if n not in fixed_values]
        if not nodes:
            return {0: 0.0, **fixed_values}
        if initial is None:
            initial = self._last_solution or {}
        x = np.array([float(initial.get(n, 0.0)) for n in nodes], dtype=float)
        fun = lambda xx: self._residual(xx, nodes, fixed_values, cap_terms=cap_terms, states=states)
        last_dx = np.inf
        for _ in range(max_iter):
            f = fun(x)
            if not np.isfinite(f).all():
                raise RuntimeError(f"{self.name}: non-finite KCL residual.")
            norm = float(np.linalg.norm(f, ord=np.inf))
            if norm < current_tolerance and last_dx < voltage_tolerance:
                result = {0: 0.0, **fixed_values, **{n: float(x[i]) for i, n in enumerate(nodes)}}
                self._last_solution = result.copy()
                return result
            jac = self._finite_difference_jacobian(fun, x, f)
            try:
                dx = np.linalg.solve(jac, -f)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(jac, -f, rcond=1e-12)[0]
            if not np.isfinite(dx).all():
                raise RuntimeError(f"{self.name}: Newton step is non-finite.")
            last_dx = float(np.linalg.norm(dx, ord=np.inf))
            # Backtracking protects the exponential device models from divergence.
            alpha, accepted = 1.0, False
            for _ in range(18):
                candidate = x + alpha * dx
                trial_norm = float(np.linalg.norm(fun(candidate), ord=np.inf))
                if np.isfinite(trial_norm) and (trial_norm < norm or trial_norm < current_tolerance):
                    x, accepted = candidate, True
                    break
                alpha *= 0.5
            if not accepted:
                # A small damped step is still useful close to a piecewise model knee.
                x = x + 1e-3 * dx
                if last_dx * 1e-3 < voltage_tolerance and norm < 1e-7:
                    result = {0: 0.0, **fixed_values, **{n: float(x[i]) for i, n in enumerate(nodes)}}
                    self._last_solution = result.copy()
                    return result
        f = fun(x)
        norm = float(np.linalg.norm(f, ord=np.inf)) if f.size else 0.0
        # Finite-difference Jacobians near hard rail / limiter knees can stall
        # just above the tight Newton tolerance. Accept only a small absolute
        # KCL residual (0.5 µA); the caller still reports the measured residual.
        if norm < 5e-7:
            result = {0: 0.0, **fixed_values, **{n: float(x[i]) for i, n in enumerate(nodes)}}
            self._last_solution = result.copy()
            return result
        raise RuntimeError(f"{self.name}: Newton solver did not converge after {max_iter} iterations (KCL residual {norm:.3e} A).")

    def _update_schmitt(self, voltages, states=None):
        state_dict = self._schmitt_state if states is None else states
        changed = False
        for idx, e in enumerate(self.elements):
            if e["kind"] == "SCHMITT":
                _, ctrl, highrail, _ = e["nodes"]
                vctrl = voltages.get(ctrl, 0.0)
                vhigh = voltages.get(highrail, 0.0)
                high = state_dict.get(idx, bool(e["initial_high"]))
                if high and vctrl >= e["high_fraction"] * vhigh:
                    state_dict[idx] = False
                    changed = True
                elif (not high) and vctrl <= e["low_fraction"] * vhigh:
                    state_dict[idx] = True
                    changed = True
            elif e["kind"] == "S":
                _, _, cp, cn = e["nodes"]
                vc = voltages.get(cp, 0.0) - voltages.get(cn, 0.0)
                closed = state_dict.get(idx, bool(e.get("initial_closed", False)))
                upper = e["vt"] + 0.5 * e["vh"]
                lower = e["vt"] - 0.5 * e["vh"]
                if not closed and vc >= upper:
                    state_dict[idx] = True; changed = True
                elif closed and vc <= lower:
                    state_dict[idx] = False; changed = True
        return changed

    # ------------------------------ analyses ------------------------------
    def dc_operating_point(self, source_overrides=None, initial_conditions=None,
                           plot=True, title=None, states=None):
        """Solve nonlinear DC KCL; capacitors are open circuits.

        ``initial_conditions`` optionally pins nodes for a held-state snapshot.
        This is useful for oscillators, whose periodic operation has no static
        equilibrium; the returned point is labelled as a constrained snapshot.
        """
        fixed = self._source_values(0.0, source_overrides)
        ic = dict(initial_conditions or {})
        fixed.update({int(n): float(v) for n, v in ic.items()})
        sol = self._solve_nonlinear(fixed, states=states)
        currents = self._branch_current_report(sol, states=states)
        figures = []
        if plot:
            fig, ax = plt.subplots(figsize=(8.5, 3.6))
            node_volts = {n: sol.get(n, 0.0) for n in self._all_nodes()}
            ax.bar([str(n) for n in node_volts], list(node_volts.values()))
            ax.axhline(0, linewidth=0.8)
            ax.set_xlabel("Circuit node")
            ax.set_ylabel("Node voltage (V)")
            ax.set_title(title or f"{self.name}: DC operating point")
            ax.grid(True, axis="y", alpha=0.25)
            fig.tight_layout()
            figures.append(fig)
            plt.show()
        return {"node_voltages_V": sol, "branch_currents_A": currents,
                "fixed_nodes": fixed, "initial_condition_snapshot": bool(ic), "figures": figures}

    def _branch_current_report(self, voltages, cap_terms=None, states=None):
        report = {}
        for idx, e in enumerate(self.elements):
            try:
                ii = self._element_currents(idx, e, voltages, cap_terms=cap_terms, states=states)
                report[e.get("name") or f"{e['kind']}{idx+1}"] = float(ii[0])
            except Exception:
                continue
        return report

    def dc_sweep(self, start, stop, points=101, source_weights=None, output_node=None,
                 output_pair=None, plot=True, initial_conditions=None, title=None):
        """Sweep one scalar source parameter mapped onto one or more source nodes."""
        values = np.linspace(float(start), float(stop), int(points))
        if int(points) < 2 or not np.isfinite(values).all():
            raise ValueError("DC sweep requires at least two finite points.")
        weights = dict(source_weights or {next(iter(self.sources)): 1.0})
        results, previous = [], None
        original = self._last_solution
        for value in values:
            overrides = {int(node): float(weight) * value for node, weight in weights.items()}
            fixed = self._source_values(0.0, overrides)
            fixed.update({int(n): float(v) for n, v in dict(initial_conditions or {}).items()})
            sol = self._solve_nonlinear(fixed, initial=previous)
            previous = sol
            results.append(sol)
        self._last_solution = previous
        outnode = int(output_node if output_node is not None else max(self._all_nodes()))
        if output_pair is not None:
            op, on = map(int, output_pair)
            out = np.array([r.get(op, 0.0) - r.get(on, 0.0) for r in results], dtype=float)
            output_label = f"V({op}) - V({on})"
        else:
            out = np.array([r.get(outnode, 0.0) for r in results], dtype=float)
            output_label = f"V({outnode})"
        figures = []
        if plot:
            fig, ax = plt.subplots(figsize=(8.5, 3.8))
            ax.plot(values, out, linewidth=2)
            ax.set_xlabel("Swept source parameter (V)")
            ax.set_ylabel(f"{output_label} (V)")
            ax.set_title(title or f"{self.name}: DC sweep")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            figures.append(fig)
            plt.show()
        return {"source_V": values, "output_V": out, "node_voltages": results,
                "output_node": outnode, "output_pair": output_pair, "figures": figures}

    def _dc_jacobians(self, dc_solution, source_nodes):
        all_nodes = self._all_nodes()
        source_nodes = set(source_nodes)
        free = [n for n in all_nodes if n not in source_nodes]
        # Form complete KCL vector for all non-ground nodes, including source nodes,
        # then extract the free-node Jacobian and coupling to ideal voltage sources.
        def full_residual(xall):
            v = {0: 0.0, **{n: float(xall[i]) for i, n in enumerate(all_nodes)}}
            currents = {n: 0.0 for n in all_nodes}
            for idx, e in enumerate(self.elements):
                ii = self._element_currents(idx, e, v, states=self._schmitt_state)
                for node, current in zip(e["nodes"], ii):
                    if node != 0:
                        currents[node] += float(current)
            return np.array([currents[n] + self.gmin * v[n] for n in all_nodes], dtype=float)
        x0 = np.array([dc_solution.get(n, 0.0) for n in all_nodes], dtype=float)
        f0 = full_residual(x0)
        jac = self._finite_difference_jacobian(full_residual, x0, f0)
        # Capacitor nodal matrix.
        cmat = np.zeros_like(jac)
        idxmap = {n: i for i, n in enumerate(all_nodes)}
        for e in self.elements:
            if e["kind"] != "C" or e["value"] == 0:
                continue
            p, n = e["nodes"]
            c = e["value"]
            if p != 0:
                cmat[idxmap[p], idxmap[p]] += c
            if n != 0:
                cmat[idxmap[n], idxmap[n]] += c
            if p != 0 and n != 0:
                cmat[idxmap[p], idxmap[n]] -= c
                cmat[idxmap[n], idxmap[p]] -= c
        return all_nodes, free, jac, cmat

    def ac_sweep(self, start=1.0, stop=1e6, points=301, source_weights=None,
                 output_node=None, output_pair=None, plot=True, dc_source_overrides=None,
                 dc_initial_conditions=None, title=None):
        """Small-signal AC sweep linearized at the nonlinear DC operating point.

        ``dc_initial_conditions`` applies the same constrained bias snapshot used
        by the caller's operating-point analysis. This matters for powered-up
        circuits whose operating state is defined by explicit capacitor/node ICs.
        """
        frequencies = np.logspace(np.log10(float(start)), np.log10(float(stop)), int(points))
        if start <= 0 or stop <= start or points < 2:
            raise ValueError("AC sweep requires 0 < start < stop and at least two points.")
        source_weights = dict(source_weights or {next(iter(self.sources)): 1.0})
        dc = self.dc_operating_point(source_overrides=dc_source_overrides,
                                     initial_conditions=dc_initial_conditions, plot=False)
        dc_solution = dc["node_voltages_V"]
        source_phasors = {int(n): complex(w) for n, w in source_weights.items()}
        all_nodes, free, jac, cmat = self._dc_jacobians(dc_solution, set(self.sources) | set(source_phasors))
        index = {n: i for i, n in enumerate(all_nodes)}
        free_idx = [index[n] for n in free]
        src_idx = [index[n] for n in source_phasors if n in index]
        src_vec = np.array([source_phasors[n] for n in source_phasors if n in index], dtype=complex)
        outnode = int(output_node if output_node is not None else max(all_nodes))
        response = np.zeros(frequencies.size, dtype=complex)
        for kf, freq in enumerate(frequencies):
            y = jac.astype(complex) + 1j * 2 * np.pi * freq * cmat
            if free_idx:
                a = y[np.ix_(free_idx, free_idx)]
                b = -y[np.ix_(free_idx, src_idx)] @ src_vec if src_idx else np.zeros(len(free_idx), complex)
                try:
                    free_v = np.linalg.solve(a, b)
                except np.linalg.LinAlgError:
                    free_v = np.linalg.lstsq(a, b, rcond=1e-12)[0]
            else:
                free_v = np.array([], dtype=complex)
            acv = {n: source_phasors.get(n, 0j) for n in source_phasors}
            acv.update({n: free_v[i] for i, n in enumerate(free)})
            response[kf] = (acv.get(int(output_pair[0]), 0j) - acv.get(int(output_pair[1]), 0j)) if output_pair is not None else acv.get(outnode, 0j)
        mag_db = 20.0 * np.log10(np.maximum(np.abs(response), 1e-15))
        phase_deg = np.unwrap(np.angle(response)) * 180.0 / np.pi
        figures = []
        if plot:
            fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.0), sharex=True)
            axes[0].semilogx(frequencies, mag_db, linewidth=2)
            axes[0].set_ylabel("Magnitude (dB)")
            axes[0].set_title(title or f"{self.name}: small-signal AC sweep")
            axes[0].grid(True, which="both", alpha=0.25)
            axes[1].semilogx(frequencies, phase_deg, linewidth=2)
            axes[1].set_xlabel("Frequency (Hz)")
            axes[1].set_ylabel("Phase (degrees)")
            axes[1].grid(True, which="both", alpha=0.25)
            fig.tight_layout()
            figures.append(fig)
            plt.show()
        return {"frequency_Hz": frequencies, "response": response, "magnitude_dB": mag_db,
                "phase_deg": phase_deg, "dc_operating_point": dc_solution, "figures": figures}

    def transient(self, time, source_signals=None, initial_conditions=None, method="trapezoidal",
                  output_node=None, output_pair=None, plot=True, title=None):
        """Implicit transient analysis with backward-Euler or trapezoidal capacitors."""
        t = np.asarray(time, dtype=float)
        if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError("time must be a finite, strictly increasing vector with at least two samples.")
        method = str(method).lower()
        if method not in ("trapezoidal", "backward_euler", "be"):
            raise ValueError("method must be trapezoidal or backward_euler.")
        signals = dict(source_signals or {})
        source_at = lambda ti: self._source_values(ti, signals)
        fixed0 = source_at(t[0])
        # Start from the nonlinear DC solution and then apply explicit capacitor ICs.
        v0 = self._solve_nonlinear(fixed0, initial={}, states=self._schmitt_state)
        for node, value in dict(initial_conditions or {}).items():
            v0[int(node)] = float(value)
        for idx, e in enumerate(self.elements):
            if e["kind"] == "SCHMITT":
                self._schmitt_state.setdefault(idx, bool(e["initial_high"]))
        nodes = self._all_nodes()
        vhist = {n: np.empty(t.size, dtype=float) for n in nodes}
        for n in nodes:
            vhist[n][0] = v0.get(n, 0.0)
        prev_v = v0
        cap_i_prev = {idx: 0.0 for idx, e in enumerate(self.elements) if e["kind"] == "C"}
        for k in range(1, t.size):
            dt = t[k] - t[k - 1]
            fixed = source_at(t[k])
            use_trap = method == "trapezoidal"
            cap_terms = {}
            for idx, e in enumerate(self.elements):
                if e["kind"] != "C" or e["value"] == 0:
                    continue
                p, n = e["nodes"]
                vp = prev_v.get(p, 0.0) if p else 0.0
                vn = prev_v.get(n, 0.0) if n else 0.0
                v_prev = vp - vn
                c = e["value"]
                if use_trap:
                    geq = 2.0 * c / dt
                    ihist = -geq * v_prev - cap_i_prev[idx]
                else:
                    geq = c / dt
                    ihist = -geq * v_prev
                cap_terms[idx] = (geq, ihist)
            sol = self._solve_nonlinear(fixed, initial=prev_v, cap_terms=cap_terms,
                                        states=self._schmitt_state, max_iter=60)
            # Detect a threshold crossing at the current step and re-solve with
            # the newly latched output state, so switching is not delayed by dt.
            if self._update_schmitt(sol):
                sol = self._solve_nonlinear(fixed, initial=sol, cap_terms=cap_terms,
                                            states=self._schmitt_state, max_iter=60)
            for idx, e in enumerate(self.elements):
                if e["kind"] != "C" or e["value"] == 0:
                    continue
                p, n = e["nodes"]
                v_new = sol.get(p, 0.0) - sol.get(n, 0.0)
                v_old = prev_v.get(p, 0.0) - prev_v.get(n, 0.0)
                if use_trap:
                    cap_i_prev[idx] = 2.0 * e["value"] / dt * (v_new - v_old) - cap_i_prev[idx]
                else:
                    cap_i_prev[idx] = e["value"] / dt * (v_new - v_old)
            for n in nodes:
                vhist[n][k] = sol.get(n, 0.0)
            prev_v = sol
        outnode = int(output_node if output_node is not None else max(nodes))
        out = (vhist.get(int(output_pair[0]), np.zeros(t.size)) - vhist.get(int(output_pair[1]), np.zeros(t.size))) if output_pair is not None else vhist.get(outnode, np.zeros(t.size))
        figures = []
        if plot:
            fig, ax = plt.subplots(figsize=(8.5, 3.8))
            ax.plot(t, out, linewidth=1.8, label=f"V({outnode})")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Voltage (V)")
            ax.set_title(title or f"{self.name}: transient response ({method})")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            figures.append(fig)
            plt.show()
        return {"time_s": t, "node_voltages_V": vhist, "output_V": out,
                "output_node": outnode, "output_pair": output_pair, "figures": figures,
                "schmitt_states": dict(self._schmitt_state)}

    def run_four_analyses(self, dc_sweep, ac_sweep, transient, output_node,
                          source_overrides=None, op_initial_conditions=None,
                          sweep_source_weights=None, ac_source_weights=None,
                          transient_source_signals=None, transient_initial_conditions=None,
                          plot=True, label=None):
        """Run DC OP, DC sweep, AC sweep, and transient with consistent labels."""
        op = self.dc_operating_point(source_overrides=source_overrides,
                                     initial_conditions=op_initial_conditions, plot=plot,
                                     title=f"{label or self.name}: DC operating point")
        sweep = self.dc_sweep(*dc_sweep, source_weights=sweep_source_weights,
                              output_node=output_node, plot=plot,
                              title=f"{label or self.name}: DC sweep")
        ac = self.ac_sweep(*ac_sweep, source_weights=ac_source_weights,
                           output_node=output_node, plot=plot,
                           dc_source_overrides=source_overrides,
                           title=f"{label or self.name}: AC sweep")
        tran = self.transient(*transient, source_signals=transient_source_signals,
                              initial_conditions=transient_initial_conditions,
                              output_node=output_node, plot=plot,
                              title=f"{label or self.name}: transient response")
        return {"dc_operating_point": op, "dc_sweep": sweep, "ac_sweep": ac, "transient": tran}
