"""Linear-system solution policy for assembled sparse MNA matrices."""
from __future__ import annotations

import warnings
import math

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator, MatrixRankWarning, norm as sparse_norm, onenormest, splu

from .exceptions import NumericalSolveError, SingularMatrixError


def solve_matrix(matrix: csc_matrix, rhs: np.ndarray, dense_threshold: int = 64,
                 residual_tolerance: float = 1e-9):
    """Solve ``A x = b`` with a dense small-system path and sparse LU otherwise."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise NumericalSolveError("MNA matrix must be square.")
    n = matrix.shape[0]
    rhs = np.asarray(rhs)
    if rhs.ndim != 1 or rhs.size != n:
        raise NumericalSolveError(f"MNA right-hand side must have shape ({n},).")
    if not np.all(np.isfinite(matrix.data)) or not np.all(np.isfinite(rhs)):
        raise NumericalSolveError("MNA matrix and right-hand side must contain only finite values.")
    if not np.isfinite(residual_tolerance) or residual_tolerance <= 0:
        raise ValueError("residual_tolerance must be finite and positive.")
    condition: float | None = None
    warning_messages: list[str] = []
    if n == 0:
        return np.zeros(0), "empty", condition, None, 0.0, 0.0, warning_messages

    condition_method: str | None = None
    if n <= dense_threshold:
        dense = matrix.toarray()
        rank = int(np.linalg.matrix_rank(dense))
        if rank < n:
            raise SingularMatrixError(
                f"MNA matrix is rank deficient ({rank}/{n}); check floating nodes, "
                "contradictory ideal sources, and missing reference paths."
            )
        condition = float(np.linalg.cond(dense))
        condition_method = "numpy.linalg.cond (2-norm)"
        if not np.isfinite(condition):
            raise SingularMatrixError("MNA matrix has a non-finite condition estimate.")
        if condition > 1e12:
            warning_messages.append(
                f"MNA matrix is ill-conditioned (estimated condition number {condition:.3e})."
            )
        try:
            solution = np.linalg.solve(dense, rhs)
        except np.linalg.LinAlgError as exc:
            raise SingularMatrixError(f"Dense MNA solve failed: {exc}") from exc
        method = "dense_lu"
    else:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", MatrixRankWarning)
                factor = splu(matrix)
                solution = factor.solve(rhs)
            if any(issubclass(item.category, MatrixRankWarning) for item in caught):
                raise SingularMatrixError("Sparse MNA factorization reported a rank warning.")
            try:
                inverse_operator = LinearOperator(
                    matrix.shape,
                    matvec=factor.solve,
                    rmatvec=lambda vector: factor.solve(vector, trans="H"),
                    dtype=matrix.dtype,
                )
                condition = float(sparse_norm(matrix, ord=1) * onenormest(inverse_operator))
                condition_method = "scipy.sparse.linalg.onenormest (estimated 1-norm)"
                if not np.isfinite(condition):
                    condition = None
                elif condition > 1e12:
                    warning_messages.append(
                        f"Sparse MNA matrix is ill-conditioned (estimated 1-norm condition number {condition:.3e})."
                    )
            except Exception as estimate_error:  # solve remains usable if an estimate cannot be formed
                warning_messages.append(f"Sparse condition estimate unavailable: {estimate_error}")
        except RuntimeError as exc:
            raise SingularMatrixError(
                "Sparse MNA factorization failed; the circuit may be singular or underdetermined."
            ) from exc
        method = "scipy_sparse_lu"

    if not np.all(np.isfinite(solution)):
        raise NumericalSolveError("MNA solution contains NaN or infinite values.")
    residual = np.asarray(matrix @ solution - rhs)
    residual_inf = float(np.linalg.norm(residual, ord=np.inf)) if n else 0.0
    matrix_inf = float(np.max(np.asarray(abs(matrix).sum(axis=1)))) if n else 0.0
    denominator = (matrix_inf * float(np.linalg.norm(solution, ord=np.inf))
                   + float(np.linalg.norm(rhs, ord=np.inf)))
    # Normwise backward error is scale invariant.  Flooring this denominator
    # at 1.0 incorrectly makes small-SI-unit systems appear more accurate.
    relative_residual = (0.0 if denominator == 0.0 and residual_inf == 0.0
                         else math.inf if denominator == 0.0
                         else residual_inf / denominator)
    if not np.isfinite(relative_residual) or relative_residual > residual_tolerance:
        raise NumericalSolveError(
            f"MNA residual check failed (relative residual {relative_residual:.3e}, "
            f"tolerance {residual_tolerance:.3e})."
        )
    return solution, method, condition, condition_method, residual_inf, relative_residual, warning_messages
