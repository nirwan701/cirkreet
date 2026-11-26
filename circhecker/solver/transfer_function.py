import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig, svdvals

class _transferfunctioncore:
    """
    Generalized MNA transfer-function engine.

    Descriptor equations:

        E dx/dt + Gx = Bu + b0

        y = Cx + Du

    Transfer function:

        H(s) = C(sE + G)^-1 B + D
    """

    def __init__(self, circuit, inputs, outputs, ports=None, tolerance=1e-9, condition_limit=1e14):
        self.circuit = circuit
        self.inputs = list(inputs)
        self.ports = dict(ports or {})
        self.tolerance = float(tolerance)
        self.condition_limit = float(condition_limit)

        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("Invalid tolerance.")

        if not np.isfinite(self.condition_limit) or self.condition_limit <= 1:
            raise ValueError("Invalid condition limit.")

        if not self.inputs:
            raise ValueError("At least one input is required.")

        if not outputs:
            raise ValueError("At least one output is required.")

        # Validate inputs
        for name in self.inputs:
            if (not isinstance(name, str) or not name or name not in circuit.netlist):
                raise ValueError(f"Invalid input source: {name}")

            if name[0].upper() not in ("V", "I"):
                raise ValueError(f"Input {name} must be a voltage or current source.")

        if len(set(self.inputs)) != len(self.inputs):
            raise ValueError("Duplicate inputs.")

        self.outputs = [self.ports.get(name, name) for name in outputs]

        # Obtain state-space model
        self.model = circuit.state_space(inputs=self.inputs)

        if not isinstance(self.model, dict):
            raise ValueError("Invalid state-space model.")

        for key in ("E", "G", "B", "unknowns"):
            if key not in self.model:
                raise ValueError(f"Missing model field: {key}")

        self.E = self._matrix(self.model["E"], "E")
        self.G = self._matrix(self.model["G"], "G")
        self.B = self._matrix(self.model["B"], "B")
        self.unknowns = list(self.model["unknowns"])
        n = self.E.shape[0]

        if n == 0 or self.E.shape != (n, n):
            raise ValueError("E must be a non-empty square matrix.")

        if self.G.shape != (n, n):
            raise ValueError("Invalid G dimensions.")

        if self.B.shape != (n, len(self.inputs)):
            raise ValueError("Invalid B dimensions.")

        if len(self.unknowns) != n:
            raise ValueError("Invalid unknowns dimensions.")

        if len(set(self.unknowns)) != n:
            raise ValueError("Duplicate unknowns.")

        # Build unknown indices
        self.node_index = {}
        self.branch_index = {}

        for i, name in enumerate(self.unknowns):
            if not isinstance(name, str):
                raise ValueError("Unknown names must be strings.")

            if name.startswith("V(") and name.endswith(")"):
                try:
                    node = int(name[2:-1])
                except ValueError as exc:
                    raise ValueError(f"Invalid node unknown: {name}") from exc

                self.node_index[node] = i

            elif name.startswith("I(") and name.endswith(")"):
                branch = name[2:-1]
                if not branch:
                    raise ValueError(f"Invalid branch unknown: {name}")

                self.branch_index[branch] = i

        # Build output matrix
        self.C = np.zeros((len(self.outputs), n), dtype=complex)

        for i, node in enumerate(self.outputs):
            if (isinstance(node, (bool, np.bool_)) or not isinstance(node, (int, np.integer)) or node < 0):
                raise ValueError(f"Invalid output node: {node}")

            if node != 0:
                if node not in self.node_index:
                    raise ValueError(f"Output node {node} does not exist.")

                self.C[i, self.node_index[node]] = 1.0

        # Direct feedthrough
        self.D = self._matrix(self.model.get("D", np.zeros((len(self.outputs), len(self.inputs)))), "D")

        if self.D.shape != (len(self.outputs), len(self.inputs)):
            raise ValueError("Invalid D dimensions.")

        # Independent source vector
        self.b0 = np.zeros(n, dtype=complex)
        input_set = set(self.inputs)

        for name, entry in circuit.netlist.items():
            if (not isinstance(name, str) or not name or name in input_set):
                continue

            kind = name[0].upper()

            if kind not in ("I", "V"):
                continue

            if len(entry) < 3:
                raise ValueError(f"Invalid source entry: {name}")

            a, b = entry[:2]

            try:
                value = float(np.real(entry[-1]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid source value: {name}") from exc

            if not np.isfinite(value):
                raise ValueError(f"Non-finite source value: {name}")

            if kind == "I":
                if a != 0:
                    if a not in self.node_index:
                        raise ValueError(f"Unknown node {a} in {name}.")

                    self.b0[self.node_index[a]] -= value

                if b != 0:
                    if b not in self.node_index:
                        raise ValueError(f"Unknown node {b} in {name}.")

                    self.b0[self.node_index[b]] += value

            else:
                if name not in self.branch_index:
                    raise ValueError(f"Missing voltage branch: {name}")

                self.b0[self.branch_index[name]] += value

    # Matrix validation
    @staticmethod
    def _matrix(value, name):
        try:
            result = np.asarray(value, dtype=complex)

        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid matrix {name}.") from exc

        if result.ndim != 2:
            raise ValueError(f"{name} must be two-dimensional.")

        if not np.isfinite(result).all():
            raise ValueError(f"{name} contains non-finite values.")

        return result
        
    # Input validation
    def _input(self, value):
        if np.isscalar(value):
            if len(self.inputs) != 1:
                raise ValueError("Provide one value per input.")

            value = [value]

        try:
            value = np.asarray(value, dtype=float)

        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid input vector.") from exc

        if value.shape != (len(self.inputs),):
            raise ValueError("Invalid input dimensions.")

        if not np.isfinite(value).all():
            raise ValueError("Non-finite input.")

        return value

    # Numerical linear solver
    def _solve(self, matrix, rhs, context="system"):
        if not np.isfinite(matrix).all():
            raise ValueError(f"Non-finite matrix in {context}.")

        try:
            singular_values = svdvals(matrix)

            if (len(singular_values) == 0 or singular_values[-1] == 0):
                raise ValueError(f"Singular matrix in {context}.")

            condition = (singular_values[0] / singular_values[-1])

            if not np.isfinite(condition):
                raise ValueError(f"Singular matrix in {context}.")

            if condition > self.condition_limit:
                raise ValueError(f"Ill-conditioned matrix in {context}. Condition number: {condition:.3e}")

            result = np.linalg.solve(matrix, rhs)

        except np.linalg.LinAlgError as exc:
            raise ValueError(f"Unable to solve {context}.") from exc

        if not np.isfinite(result).all():
            raise ValueError(f"Non-finite result in {context}.")

        return result

    # Transfer function
    def H(self, s):
        try:
            s = complex(s)

        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid complex frequency.") from exc

        if not np.isfinite(s):
            raise ValueError("Non-finite complex frequency.")

        matrix = s * self.E + self.G
        X = self._solve(matrix, self.B, context=f"transfer function at s={s}")
        result = self.C @ X + self.D

        if not np.isfinite(result).all():
            raise ValueError("Non-finite transfer function.")

        return result

    # DC gain
    def dc_gain(self):
        return self.H(0j)

    # AC response
    def ac(self, frequency):
        try:
            frequency = float(frequency)

        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid frequency.") from exc

        if (not np.isfinite(frequency) or frequency < 0):
            raise ValueError("Frequency must be finite and non-negative.")

        s = 1j * 2 * np.pi * frequency

        return self.H(s)
        
    # Generalized pole analysis
    def pole_analysis(self):
        try:
            homogeneous = eig(-self.G, self.E, right=False, homogeneous_eigvals=True)

        except Exception as exc:
            raise ValueError("Generalized eigenvalue calculation failed.") from exc

        alpha = homogeneous[0]
        beta = homogeneous[1]
        finite = []
        infinite = []
        indeterminate = []

        for a, b in zip(alpha, beta):
            if abs(a) == 0 and abs(b) == 0:
                indeterminate.append(complex(np.nan, np.nan))

            elif abs(b) == 0:
                infinite.append(complex(np.inf, 0))

            else:
                value = a / b
                if np.isfinite(value):
                    finite.append(value)

                else:
                    indeterminate.append(complex(np.nan, np.nan))

        return {"finite": np.asarray(finite, dtype=complex), "infinite": np.asarray(infinite, dtype=complex), "indeterminate": np.asarray(indeterminate, dtype=complex), "regular_candidate": (len(indeterminate) == 0)}

    # Pole accessor
    def poles(self):
        result = self.pole_analysis()

        return np.concatenate([result["finite"], result["infinite"], result["indeterminate"]])

    # Stability analysis
    def stability(self):
        p = self.poles()
        finite = p[np.isfinite(p)]

        if len(finite) == 0:
            return {"classification": "indeterminate", "stable": False, "unstable": False, "marginally_stable": False, "marginal_candidate": False, "indeterminate": True, "poles": p}

        tolerance = 1e-9
        real_parts = np.real(finite)
        unstable = np.any(real_parts > tolerance)
        imaginary_axis = np.any(np.abs(real_parts) <= tolerance)
        stable = np.all(real_parts < -tolerance)
        marginal_candidate = (not stable and not unstable and imaginary_axis)

        if unstable:
            classification = "unstable"

        elif stable:
            classification = "asymptotically_stable"

        elif marginal_candidate:
            classification = "marginal_candidate"

        else:
            classification = "indeterminate"

        return {"classification": classification, "stable": bool(stable), "unstable": bool(unstable), "marginally_stable": False, "marginal_candidate": bool(marginal_candidate), "indeterminate": bool(classification == "indeterminate"), "poles": p}
            
    # Numerical diagnostics
    def diagnostics(self):
        try:
            pole_data = self.pole_analysis()
            condition_E = np.linalg.cond(self.E)
            condition_G = np.linalg.cond(self.G)

            return {"dimension": self.E.shape[0], "inputs": len(self.inputs), "outputs": len(self.outputs), "rank_E": np.linalg.matrix_rank(self.E), "rank_G": np.linalg.matrix_rank(self.G), "condition_E": condition_E, "condition_G": condition_G, "finite_poles": len(pole_data["finite"]), "infinite_poles": len(pole_data["infinite"]), "indeterminate_poles": len(pole_data["indeterminate"]), "regular_candidate": (pole_data["regular_candidate"]), "poles": self.poles()}

        except Exception as exc:
            raise ValueError("Numerical diagnostics failed.") from exc

# Backwards-compatible alias for the historical class name used below.
_TransferFunctionCore = _transferfunctioncore

class transferfunction(_TransferFunctionCore):
    """
    Complete generalized MNA circuit analysis engine.
    """
    # DC operating point
    def dc_operating_point(self, plot=True):
        result = self.circuit.solve_dc()

        if not isinstance(result, dict):
            raise ValueError("Invalid DC operating point result.")

        nodes = result.get("node_voltages_V")

        if not isinstance(nodes, dict):
            raise ValueError("Missing node_voltages_V.")

        for node, value in nodes.items():
            if not np.isfinite(value):
                raise ValueError(f"Non-finite voltage at node {node}.")

        if plot:
            names = list(nodes.keys())
            values = np.asarray([np.real(v) for v in nodes.values()], dtype=float)
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.bar([str(n) for n in names], values, color="steelblue")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Node")
            ax.set_ylabel("Voltage (V)")
            ax.set_title("DC Operating Point")
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.show()

        return result

    # DC sweep
    def dc_sweep(self, source, start, stop, points=100, output=0, plot=True):
        if (not isinstance(source, str) or source not in self.circuit.netlist):
            raise ValueError("Unknown sweep source.")

        if source[0].upper() not in ("V", "I"):
            raise ValueError("Source must be a voltage or current source.")

        try:
            start = float(start)
            stop = float(stop)

        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid sweep range.") from exc

        if (not np.isfinite(start) or not np.isfinite(stop) or stop < start):
            raise ValueError("Invalid sweep range.")

        if (isinstance(points, (bool, np.bool_)) or not isinstance(points, (int, np.integer)) or points < 2):
            raise ValueError("Points must be an integer >= 2.")

        if isinstance(output, str):
            if output not in self.ports:
                raise ValueError("Unknown output port.")

            output = self.ports[output]

        if (isinstance(output, (bool, np.bool_)) or not isinstance(output, (int, np.integer)) or output not in self.outputs):
            raise ValueError("Output is not configured.")

        original = self.circuit.netlist[source]

        if len(original) < 3:
            raise ValueError("Invalid source netlist entry.")

        sweep = np.linspace(start, stop, points)
        values = []
        primary_error = None
        restoration_error = None

        try:
            for value in sweep:
                self.circuit.netlist[source] = (*original[:-1], float(value))
                result = self.circuit.solve_dc()

                if not isinstance(result, dict):
                    raise ValueError("Invalid DC sweep result.")

                nodes = result.get("node_voltages_V")

                if not isinstance(nodes, dict):
                    raise ValueError("Missing node voltages.")

                voltage = (0.0 if output == 0 else nodes[output])
                voltage = float(np.real(voltage))

                if not np.isfinite(voltage):
                    raise ValueError(f"Non-finite result at {value}.")

                values.append(voltage)

        except Exception as exc:
            primary_error = exc

        finally:
            self.circuit.netlist[source] = original

            try:
                self.circuit.solve_dc()

            except Exception as exc:
                restoration_error = exc

        if primary_error is not None:
            if restoration_error is not None:
                raise RuntimeError("DC sweep failed and original operating point restoration also failed.") from primary_error

            raise primary_error

        if restoration_error is not None:
            raise RuntimeError("Source restored, but original DC operating point could not be recalculated.") from restoration_error

        values = np.asarray(values, dtype=float)

        if plot:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(sweep, values, linewidth=2)
            ax.set_xlabel(source)
            ax.set_ylabel(f"V({output}) [V]")
            ax.set_title("DC Sweep")
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

        return {"sweep": sweep, "voltage": values}
            
    def ac_sweep(self, start=1.0, stop=1e6, points=500, input_index=0, output_index=0, plot=True):
        try:
            start = float(start)
            stop = float(stop)

        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid AC frequencies.") from exc

        if (not np.isfinite(start) or not np.isfinite(stop) or start <= 0 or stop <= start):
            raise ValueError("Invalid AC sweep range.")

        if (isinstance(points, (bool, np.bool_)) or not isinstance(points, (int, np.integer)) or points < 2):
            raise ValueError("Points must be an integer >= 2.")

        if (isinstance(input_index, (bool, np.bool_)) or not isinstance(input_index, (int, np.integer)) or not 0 <= input_index < len(self.inputs)):
            raise ValueError("Invalid input index.")

        if (isinstance(output_index, (bool, np.bool_)) or not isinstance(output_index, (int, np.integer)) or not 0 <= output_index < len(self.outputs)):
            raise ValueError("Invalid output index.")

        frequency = np.logspace(np.log10(start), np.log10(stop), points)
        response = np.empty(points, dtype=complex)

        for k, f in enumerate(frequency):
            response[k] = self.ac(f)[output_index, input_index]

        magnitude = np.abs(response)
        db = 20 * np.log10(np.maximum(magnitude, 1e-300))

        phase = (np.unwrap(np.angle(response)) * 180 / np.pi)

        result = {"frequency": frequency, "H": response, "magnitude": magnitude, "magnitude_dB": db, "phase": phase}

        if plot:
            fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            ax[0].semilogx(frequency, db, linewidth=2)
            ax[0].set_ylabel("Magnitude (dB)")
            ax[0].set_title("AC Sweep / Bode Plot")
            ax[0].grid(True, which="both", alpha=0.3)
            ax[1].semilogx(frequency, phase, color="tab:orange", linewidth=2)
            ax[1].set_xlabel("Frequency (Hz)")
            ax[1].set_ylabel("Phase (degrees)")
            ax[1].grid(True, which="both", alpha=0.3)
            plt.tight_layout()
            plt.show()

        return result    
        
    def transient(self, time, signal, initial=None, method="trapezoidal", plot=True, check_consistency=True):
        # Time validation
        try:
            time = np.asarray(time, dtype=float)

        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid time vector.") from exc

        if (time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0)):
            raise ValueError("Time must be finite and strictly increasing.")

        if not callable(signal):
            raise TypeError("Signal must be a callable function.")

        if method not in ("backward_euler", "trapezoidal"):
            raise ValueError("Method must be backward_euler or trapezoidal.")

        n = self.E.shape[0]

        # Initial state
        if initial is None:
            result = self.circuit.solve_dc()

            if not isinstance(result, dict):
                raise ValueError("Invalid initial DC solution.")

            nodes = result.get("node_voltages_V", {})
            branches = result.get("branch_results", {})
            x = np.zeros(n, dtype=complex)

            for node, index in self.node_index.items():
                if node not in nodes:
                    raise ValueError(f"Missing initial voltage for node {node}.")

                x[index] = nodes[node]

            for name, index in self.branch_index.items():
                if name not in branches:
                    raise ValueError(f"Missing initial current for branch {name}.")

                if "current_A" not in branches[name]:
                    raise ValueError(f"Missing current_A for branch {name}.")

                x[index] = branches[name]["current_A"]

        else:
            try:
                x = np.asarray(initial, dtype=complex)

            except (TypeError, ValueError) as exc:
                raise ValueError("Invalid initial state.") from exc

            if x.shape != (n,):
                raise ValueError("Incorrect initial state dimensions.")

            x = x.copy()

        if not np.isfinite(x).all():
            raise ValueError("Non-finite initial state.")

        # Input history
        U = np.empty((len(time), len(self.inputs)), dtype=float)

        for k, t in enumerate(time):
            U[k] = self._input(signal(t))

        # Algebraic consistency
        if check_consistency:
            row_norm = np.linalg.norm(self.E, axis=1)
            scale = max(np.linalg.norm(self.E), np.linalg.norm(self.G), 1.0)
            algebraic = (row_norm <= self.tolerance * scale)

            if np.any(algebraic):
                residual = (self.G @ x - self.B @ U[0] - self.b0)
                error = np.linalg.norm(residual[algebraic])
                reference = max(np.linalg.norm((self.G @ x)[algebraic]), np.linalg.norm((self.B @ U[0] + self.b0)[algebraic]), 1.0)

                if error > self.tolerance * reference:
                    raise ValueError("Initial state violates algebraic circuit constraints. Provide a consistent initial state.")

        # Storage
        states = np.zeros((len(time), n), dtype=complex)
        outputs = np.zeros((len(time), len(self.outputs)), dtype=complex)
        states[0] = x
        outputs[0] = (self.C @ x + self.D @ U[0])

        # Time integration
        for k in range(1, len(time)):
            dt = time[k] - time[k - 1]

            if method == "backward_euler":
                matrix = (self.E / dt + self.G)
                rhs = (self.E @ x / dt + self.B @ U[k] + self.b0)

            else:
                matrix = (self.E / dt + self.G / 2)
                rhs = (self.E @ x / dt - self.G @ x / 2 + self.B @ (U[k] + U[k - 1]) / 2 + self.b0)

            x = self._solve(matrix, rhs, context=f"transient at t={time[k]}")
            states[k] = x
            outputs[k] = (self.C @ x + self.D @ U[k])

        # Remove numerical imaginary parts
        if np.max(np.abs(np.imag(states))) < 1e-12:
            states = np.real(states)

        if np.max(np.abs(np.imag(outputs))) < 1e-12:
            outputs = np.real(outputs)

        # Plotting
        if plot:
            fig, ax = plt.subplots(figsize=(10, 5))

            for i, node in enumerate(self.outputs):
                ax.plot(time, outputs[:, i], linewidth=2, label=f"V({node})")

            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Voltage (V)")
            ax.set_title(f"Transient Analysis: {method}")
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

        return {"time": time, "states": states, "outputs": outputs, "inputs": U, "method": method}    
        
    def report(self):
        print("\n")
        print("=" * 65)
        print("             CIRCUIT ANALYSIS REPORT")
        print("=" * 65)

        # Circuit information
        print("\nCIRCUIT INFORMATION")
        print("Inputs:", self.inputs)
        print("Outputs:", self.outputs)
        print("State variables:", self.unknowns)
        print("System dimension:", self.E.shape[0])

        # State-space matrices
        matrices = {"E": self.E, "G": self.G, "B": self.B, "C": self.C, "D": self.D, "b0": self.b0}
        print("\n" + "-" * 65)
        print("STATE-SPACE MATRICES")
        print("-" * 65)

        for name, matrix in matrices.items():
            print(f"\n{name} =")
            print(np.array2string(matrix, precision=6, suppress_small=True))

        # DC gain
        print("\n" + "-" * 65)
        print("DC GAIN")
        print("-" * 65)

        try:
            print(self.dc_gain())

        except ValueError as exc:
            print("Unavailable:", exc)

        # Generalized poles
        print("\n" + "-" * 65)
        print("GENERALIZED POLES")
        print("-" * 65)

        try:
            pole_data = self.pole_analysis()

            for i, pole in enumerate(pole_data["finite"]):
                print(f"Finite pole {i + 1}: {pole.real:.6g}{pole.imag:+.6g}j")

            for i, pole in enumerate(pole_data["infinite"]):
                print(f"Infinite pole {i + 1}: Infinity")

            for i, pole in enumerate(pole_data["indeterminate"]):
                print(f"Indeterminate pole {i + 1}")

        except ValueError as exc:
            print("Unavailable:", exc)

        # Stability
        print("\n" + "-" * 65)
        print("STABILITY ANALYSIS")
        print("-" * 65)

        try:
            result = self.stability()

            for key, value in result.items():
                if key != "poles":
                    print(f"{key}: {value}")

            if result.get("marginal_candidate"):
                print("\nWarning: Imaginary-axis poles require a semisimplicity test.")

        except ValueError as exc:
            print("Unavailable:", exc)

        # Numerical diagnostics
        print("\n" + "-" * 65)
        print("NUMERICAL DIAGNOSTICS")
        print("-" * 65)

        try:
            result = self.diagnostics()

            for key, value in result.items():
                if key != "poles":
                    print(f"{key}: {value}")

        except ValueError as exc:
            print("Unavailable:", exc)

        # End report
        print("\n" + "=" * 65)
        print("              END OF REPORT")
        print("=" * 65)