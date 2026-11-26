import numpy as np

class circuitsolver:
    """Linear MNA solver for DC, AC, verification, and state-space models."""

    TYPES = "RIVCLGEOQ"

    def __init__(self, netlist):
        self.netlist = netlist
        self.node_voltages = {}
        self.branch_currents = {}
        self._terminal_currents = {}
        self._signature = None
        self._mode = None
        self._frequency = None
        self._linear_residual = None

    @staticmethod
    def _finite(x):
        try:
            return bool(np.isfinite(np.asarray(x)).all())
        except (TypeError, ValueError, OverflowError):
            return False

    @staticmethod
    def _tol(tolerance):
        if isinstance(tolerance, (bool, np.bool_)):
            raise ValueError("Invalid tolerance.")
        try:
            t = float(tolerance)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Invalid tolerance.") from exc
        if not np.isfinite(t) or t < 0:
            raise ValueError("Tolerance must be finite and nonnegative.")
        return t

    @staticmethod
    def _node(n):
        return (not isinstance(n, (bool, np.bool_)) and isinstance(n, (int, np.integer)) and n >= 0)

    def _parts(self):
        if not isinstance(self.netlist, dict) or not self.netlist:
            raise ValueError("Netlist must be a non-empty dictionary.")

        parts = []
        for name, entry in self.netlist.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Invalid component name.")
            k = name[0].upper()
            if k not in self.TYPES:
                raise ValueError(f"Unsupported component: {name}")

            try:
                e = tuple(entry)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name}: invalid component entry.") from exc

            nn = 3 if k == "Q" else 4 if k in "GEO" else 2
            nv = 3 if k == "Q" else 1
            if len(e) != nn + nv:
                raise ValueError(f"{name}: expected {nn + nv} entries, got {len(e)}.")

            nodes = e[:nn]
            if not all(self._node(n) for n in nodes):
                raise ValueError(f"{name}: nodes must be nonnegative integers.")
            if nodes[0] == nodes[1]:
                raise ValueError(f"{name}: identical output terminals.")
            if k == "Q" and len(set(nodes)) != 3:
                raise ValueError(f"{name}: transistor terminals must be distinct.")

            try:
                vals = np.asarray(e[nn:], dtype=complex)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name}: invalid component values.") from exc

            if vals.shape != (nv,) or not self._finite(vals):
                raise ValueError(f"{name}: values must be finite scalars.")
            if k not in "IV" and np.any(vals.imag != 0):
                raise ValueError(f"{name}: parameters must be real.")

            vals = (vals.astype(complex) if k in "IV" else vals.real.astype(float))

            if k == "R" and vals[0] <= 0:
                raise ValueError(f"{name}: resistance must be positive.")
            if k in "CL" and vals[0] < 0:
                raise ValueError(f"{name}: {k} must be nonnegative.")
            if k == "Q" and (vals[0] < 0 or np.any(vals[1:] <= 0)):
                raise ValueError(f"{name}: require gm >= 0, rpi > 0, ro > 0.")

            parts.append((name, k, tuple(int(n) for n in nodes), vals))
        return parts

    @staticmethod
    def _nodes(parts):
        """
        Extract all non-ground circuit nodes.
        """    
        return sorted({node for _, _, nodes, _ in parts for node in nodes if node != 0})
        
    @staticmethod
    def _branch_set(parts, mode):
        return [p for p in parts if p[1] in "VEO" or (p[1] == "L" and mode == "dc")]

    @staticmethod
    def _relative_residual(M, x, z):
        with np.errstate(over="ignore", invalid="ignore"):
            r = M @ x - z
            num = np.linalg.norm(r, np.inf)
            den = (np.linalg.norm(M, np.inf) * np.linalg.norm(x, np.inf) + np.linalg.norm(z, np.inf))
        if not np.isfinite(num) or not np.isfinite(den):
            return float("inf")
        if den == 0:
            return 0.0 if num == 0 else float("inf")
        return float(num / den)

    def solve(self, mode="dc", frequency=0.0):
        """Solve a linear DC or single-frequency AC circuit."""
        if not isinstance(mode, str) or mode.lower() not in ("dc", "ac"):
            raise ValueError("mode must be 'dc' or 'ac'.")
        mode = mode.lower()

        try:
            f = float(frequency)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Invalid frequency.") from exc
        if (not np.isfinite(f) or f < 0 or (mode == "ac" and f <= 0)):
            raise ValueError("Invalid frequency.")

        parts = self._parts()
        if mode == "dc":
            for name, k, _, v in parts:
                if k in "IV" and np.any(v.imag != 0):
                    raise ValueError(f"{name}: DC source values must be real.")

        nodes = self._nodes(parts)
        if not nodes:
            raise ValueError("No non-ground nodes.")

        branches = self._branch_set(parts, mode)
        idx = {n: i for i, n in enumerate(nodes)}
        N = len(nodes)
        M = np.zeros((N + len(branches), N + len(branches)),dtype=complex)
        z = np.zeros(N + len(branches), dtype=complex)
        bi = {p[0]: N + j for j, p in enumerate(branches)}
        w = 2 * np.pi * f

        def stamp(row, node, value):
            if node:
                M[row, idx[node]] += value

        def admittance(a, b, y):
            if a:
                M[idx[a], idx[a]] += y
            if b:
                M[idx[b], idx[b]] += y
            if a and b:
                M[idx[a], idx[b]] -= y
                M[idx[b], idx[a]] -= y

        def current_source(a, b, current):
            if a:
                z[idx[a]] -= current
            if b:
                z[idx[b]] += current

        for name, k, ns, v in parts:
            a, b = ns[:2]

            if k == "R":
                admittance(a, b, 1 / v[0])
            elif k == "I":
                current_source(a, b, v[0])
            elif k == "C":
                if mode == "ac":
                    admittance(a, b, 1j * w * v[0])
            elif k == "L":
                if mode == "dc":
                    pass
                else:
                    if v[0] == 0:
                        raise ValueError(f"{name}: zero inductance is an ideal short.")
                    admittance(a, b, 1 / (1j * w * v[0]))
            elif k == "G":
                cp, cn = ns[2:]
                for n, s in ((a, 1), (b, -1)):
                    if n:
                        stamp(idx[n], cp, s * v[0])
                        stamp(idx[n], cn, -s * v[0])
            elif k == "Q":
                c, base, e = ns
                gm, rpi, ro = v
                admittance(base, e, 1 / rpi)
                admittance(c, e, 1 / ro)
                for n, s in ((c, 1), (e, -1)):
                    if n:
                        stamp(idx[n], base, s * gm)
                        stamp(idx[n], e, -s * gm)

        for name, k, ns, v in branches:
            row = bi[name]
            a, b = ns[:2]
            for n, s in ((a, 1), (b, -1)):
                if n:
                    M[idx[n], row] += s
                    M[row, idx[n]] += s

            if k == "V":
                z[row] = v[0]
            elif k == "L":
                z[row] = 0
            else:
                cp, cn = ns[2:]
                stamp(row, cp, -v[0])
                stamp(row, cn, v[0])

        if not self._finite(M) or not self._finite(z):
            raise ValueError("Non-finite circuit equations.")
        try:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                sol = np.linalg.solve(M, z)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Singular circuit: no unique solution.") from exc
        if not self._finite(sol):
            raise ValueError("Non-finite circuit solution.")

        residual = self._relative_residual(M, sol, z)
        volts = {0: 0j}
        volts.update({n: sol[i] for n, i in idx.items()})
        terminals, results = {}, {}

        for name, k, ns, v in parts:
            a, b = ns[:2]
            voltage = volts[a] - volts[b]

            if k == "R":
                current = voltage / v[0]
            elif k == "I":
                current = complex(v[0])
            elif k in "VEO" or (k == "L" and mode == "dc"):
                current = sol[bi[name]]
            elif k == "C":
                current = (0j if mode == "dc" else 1j * w * v[0] * voltage)
            elif k == "L":
                current = voltage / (1j * w * v[0])
            elif k == "G":
                current = v[0] * (volts[ns[2]] - volts[ns[3]])
            elif k == "Q":
                c, base, e = ns
                gm, rpi, ro = v
                ib = (volts[base] - volts[e]) / rpi
                ic = (gm * (volts[base] - volts[e]) + (volts[c] - volts[e]) / ro)
                ie = -(ic + ib)
                terminals[name] = {c: ic, base: ib, e: ie}
                results[name] = {"collector_A": ic, "base_A": ib, "emitter_A": ie}
                continue
            else:
                raise RuntimeError(f"Unhandled component type: {k}")

            terminals[name] = {a: current, b: -current}
            power = voltage * np.conj(current)
            real_power = (power.real if mode == "ac" else voltage.real * current.real)
            results[name] = {"voltage_V": voltage, "current_A": current, "power_W": float(real_power), "complex_power_VA": power}

        if not all(self._finite(list(r.values())) for r in results.values()):
            raise ValueError("Non-finite component results.")

        self.node_voltages = volts
        self.branch_currents = results
        self._terminal_currents = terminals
        self._signature = (mode, f, repr(self.netlist))
        self._mode, self._frequency = mode, f
        self._linear_residual = residual

        return {"mode": mode, "frequency_Hz": f if mode == "ac" else 0.0, "node_voltages_V": volts.copy(), "branch_results": {n: r.copy() for n, r in results.items()}, "linear_residual": residual,}

    def solve_dc(self):
        return self.solve("dc")

    def solve_ac(self, frequency):
        return self.solve("ac", frequency)

    def check_kcl(self, tolerance=1e-9):
        tol = self._tol(tolerance)
        if self._mode is None:
            raise ValueError("Run solve() first.")

        residual = {n: 0j for n in self.node_voltages if n != 0}
        for terms in self._terminal_currents.values():
            for n, current in terms.items():
                if n != 0:
                    residual[n] += current

        maximum = max((abs(v) for v in residual.values()), default=0.0)
        return {"node_residuals_A": residual, "max_residual_A": float(maximum), "is_balanced": bool(np.isfinite(maximum) and maximum <= tol)}

    def check_kvl(self, loop, tolerance=1e-9):
        tol = self._tol(tolerance)
        if self._mode is None:
            raise ValueError("Run solve() first.")
        try:
            loop = list(loop)
        except (TypeError, ValueError) as exc:
            raise ValueError("Loop must be iterable.") from exc

        terminals = {name: ns[:2] for name, k, ns, _ in self._parts() if k != "Q"}
        total, start, current_node = 0j, None, None

        for item in loop:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError(f"Invalid loop entry: {item!r}")
            name, direction = item
            if (not isinstance(name, str) or name not in terminals or isinstance(direction, (bool, np.bool_)) or not isinstance(direction, (int, np.integer)) or direction not in (-1, 1)):
                raise ValueError(f"Invalid loop entry: {item!r}")

            a, b = terminals[name]
            u, v = (a, b) if direction == 1 else (b, a)
            if current_node is not None and u != current_node:
                raise ValueError(f"Disconnected loop at {name}.")
            if start is None:
                start = u
            current_node = v
            total += direction * (self.node_voltages[a] - self.node_voltages[b])

        if not loop or current_node != start:
            raise ValueError("Loop must be non-empty and closed.")

        return {"net_voltage_V": total, "is_balanced": bool(np.isfinite(abs(total)) and abs(total) <= tol)}

    def verify(self, loops=None, tolerance=1e-9):
        tol = self._tol(tolerance)
        kcl = self.check_kcl(tol)
        try:
            loops = [] if loops is None else list(loops)
        except (TypeError, ValueError) as exc:
            raise ValueError("loops must be iterable.") from exc

        kvl = [self.check_kvl(loop, tol) for loop in loops]
        return {"mode": self._mode, "kcl": kcl, "kvl": kvl, "passed": bool(kcl["is_balanced"] and all(x["is_balanced"] for x in kvl))}

    def verify_dc(self, loops=None, tolerance=1e-9):
        if self._mode != "dc":
            raise ValueError("Run solve_dc() first.")
        return self.verify(loops, tolerance)

    def verify_ac(self, loops=None, tolerance=1e-9):
        if self._mode != "ac":
            raise ValueError("Run solve_ac(frequency) first.")
        return self.verify(loops, tolerance)

    def state_space(self, inputs=None):
        """Build E*x_dot + G*x = B*u + b0.

        Inductor equation: L*dI/dt - Va + Vb = 0.
        Unlisted independent sources are zeroed.
        """
        parts = self._parts()
        by_name = {n: (k, ns, v) for n, k, ns, v in parts}

        if inputs is None:
            inputs = [n for n, k, _, _ in parts if k in "IV"]
        else:
            if isinstance(inputs, str):
                raise ValueError("inputs must be an iterable, not a string.")
            try:
                inputs = list(inputs)
            except (TypeError, ValueError) as exc:
                raise ValueError("Invalid inputs.") from exc

        if any(not isinstance(n, str) for n in inputs):
            raise ValueError("Input names must be strings.")
        if len(set(inputs)) != len(inputs):
            raise ValueError("Duplicate input names.")
        for n in inputs:
            if n not in by_name or by_name[n][0] not in "IV":
                raise ValueError(f"{n!r}: input must name an independent current or voltage source.")

        nodes = self._nodes(parts)
        if not nodes:
            raise ValueError("No non-ground nodes.")

        branches = [p for p in parts if p[1] in "VELO"]
        nn, nb = len(nodes), len(branches)
        size, ni = nn + nb, len(inputs)
        idx = {n: i for i, n in enumerate(nodes)}
        bi = {p[0]: nn + j for j, p in enumerate(branches)}
        ui = {n: j for j, n in enumerate(inputs)}
        E = np.zeros((size, size), dtype=float)
        G = np.zeros((size, size), dtype=float)
        B = np.zeros((size, ni), dtype=float)
        b0 = np.zeros(size, dtype=float)

        def stamp_node(M, row, node, value):
            if node:
                M[row, idx[node]] += value

        def admittance(M, a, b, value):
            if a:
                M[idx[a], idx[a]] += value
            if b:
                M[idx[b], idx[b]] += value
            if a and b:
                M[idx[a], idx[b]] -= value
                M[idx[b], idx[a]] -= value

        def branch(name, a, b, voltage_sign=1.0):
            row = bi[name]
            if a:
                G[idx[a], row] += 1.0
                G[row, idx[a]] += voltage_sign
            if b:
                G[idx[b], row] -= 1.0
                G[row, idx[b]] -= voltage_sign

        for name, k, ns, v in parts:
            a, b = ns[:2]

            if k == "R":
                admittance(G, a, b, 1 / v[0])
            elif k == "C":
                admittance(E, a, b, v[0])
            elif k == "L":
                branch(name, a, b, -1.0)
                E[bi[name], bi[name]] += v[0]
            elif k == "V":
                branch(name, a, b)
            elif k in "EO":
                branch(name, a, b)
                cp, cn = ns[2:]
                stamp_node(G, bi[name], cp, -v[0])
                stamp_node(G, bi[name], cn, v[0])
            elif k == "G":
                cp, cn = ns[2:]
                for n, s in ((a, 1.0), (b, -1.0)):
                    if n:
                        stamp_node(G, idx[n], cp, s * v[0])
                        stamp_node(G, idx[n], cn, -s * v[0])
            elif k == "Q":
                c, base, e = ns
                gm, rpi, ro = v
                admittance(G, base, e, 1 / rpi)
                admittance(G, c, e, 1 / ro)
                for n, s in ((c, 1.0), (e, -1.0)):
                    if n:
                        stamp_node(G, idx[n], base, s * gm)
                        stamp_node(G, idx[n], e, -s * gm)
            elif k == "I":
                pass

        for name in inputs:
            k, ns, _ = by_name[name]
            a, b = ns[:2]
            col = ui[name]

            if k == "I":
                if a:
                    B[idx[a], col] -= 1.0
                if b:
                    B[idx[b], col] += 1.0
            else:
                B[bi[name], col] = 1.0

        if not all(self._finite(M) for M in (E, G, B, b0)):
            raise ValueError("Non-finite state-space matrices.")

        unknowns = ([f"V({n})" for n in nodes] + [f"I({p[0]})" for p in branches])

        try:
            rank_E = int(np.linalg.matrix_rank(E))
            condition_E = float(np.linalg.cond(E))
        except np.linalg.LinAlgError:
            rank_E, condition_E = None, float("inf")

        result = {"E": E, "G": G, "B": B, "b0": b0, "inputs": inputs.copy(), "unknowns": unknowns, "equation": "E @ x_dot + G @ x = B @ u + b0", "A": None, "B_explicit": None, "rank_E": rank_E, "condition_E": condition_E, "explicit_available": False}

        if rank_E == size:
            try:
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    A = -np.linalg.solve(E, G)
                    B_explicit = np.linalg.solve(E, B)
                if self._finite(A) and self._finite(B_explicit):
                    result["A"] = A
                    result["B_explicit"] = B_explicit
                    result["explicit_available"] = True
            except np.linalg.LinAlgError:
                pass

        return result