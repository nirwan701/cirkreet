from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csc_matrix, diags

from circhecker.parser.spice import parse_spice_text
from circhecker.solver.exceptions import NumericalSolveError, SingularMatrixError
from circhecker.solver.matrix import solve_matrix
from circhecker.solver.mna import solve_dc


def test_small_si_scale_system_uses_scale_invariant_backward_error():
    matrix = csc_matrix(np.diag([1e-15, 2e-15, 5e-15]))
    expected = np.asarray([2.0, -3.0, 0.5])
    rhs = matrix @ expected
    solution, method, condition, _, residual, backward_error, warnings = solve_matrix(matrix, rhs)
    assert np.allclose(solution, expected, rtol=1e-12, atol=1e-12)
    assert backward_error < 1e-12
    assert residual < 1e-27
    assert method == "dense_lu"


def test_sparse_complex_system_uses_sparse_factorization_and_residual_check():
    n = 90
    main = (4.0 + 1.0j) * np.ones(n)
    off = (-0.5 + 0.2j) * np.ones(n-1)
    matrix = diags([off, main, off], [-1, 0, 1], format="csc", dtype=complex)
    expected = np.linspace(-1.0, 1.0, n) + 1j*np.linspace(0.2, 0.8, n)
    rhs = matrix @ expected
    solution, method, condition, condition_method, residual, backward_error, warnings = solve_matrix(matrix, rhs)
    assert method == "scipy_sparse_lu"
    assert condition is not None and condition >= 1
    assert condition_method.startswith("scipy.sparse")
    assert np.allclose(solution, expected, rtol=1e-11, atol=1e-11)
    assert backward_error < 1e-12


def test_exactly_singular_dense_and_sparse_systems_fail_explicitly():
    with pytest.raises(SingularMatrixError):
        solve_matrix(csc_matrix([[1.0, 2.0], [2.0, 4.0]]), np.asarray([1.0, 2.0]))
    sparse_singular = diags([np.ones(70), np.ones(70)], [0, 1], shape=(70, 70), format="csc")
    sparse_singular = sparse_singular - diags([np.ones(70)], [0], shape=(70, 70), format="csc")
    with pytest.raises(SingularMatrixError):
        solve_matrix(sparse_singular, np.ones(70))


def test_matrix_solver_rejects_nonfinite_and_inconsistent_inputs():
    with pytest.raises(NumericalSolveError, match="finite"):
        solve_matrix(csc_matrix([[1.0, np.nan], [0.0, 1.0]]), np.ones(2))
    with pytest.raises(NumericalSolveError, match="right-hand side"):
        solve_matrix(csc_matrix(np.eye(2)), np.ones(3))
    with pytest.raises(NumericalSolveError, match="square"):
        solve_matrix(csc_matrix(np.ones((2, 3))), np.ones(2))


def test_ill_conditioned_but_nonsingular_matrix_is_flagged_not_silently_accepted():
    matrix = csc_matrix(np.diag([1.0, 1e-13]))
    rhs = np.asarray([1.0, 1e-13])
    solution, _, condition, _, _, backward_error, warnings = solve_matrix(matrix, rhs)
    assert condition == pytest.approx(1e13, rel=1e-5)
    assert any("ill-conditioned" in item for item in warnings)
    assert np.allclose(solution, [1.0, 1.0])
    assert backward_error < 1e-12


def test_high_dynamic_range_resistor_divider_remains_finite_and_diagnostic():
    ir = parse_spice_text("""V1 in 0 1
R1 in out 1u
R2 out 0 1G
.end
""")
    solution = solve_dc(ir)
    output_id = next(node["id"] for node in ir.nodes if node["name"] == "out")
    assert np.isfinite(solution.node_voltages_V[output_id])
    assert solution.node_voltages_V[output_id] == pytest.approx(1.0, abs=1e-8)
    assert solution.diagnostics.relative_residual < 1e-9
