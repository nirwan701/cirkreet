"""Typed failures raised by the generalized MNA solver."""


class MNAError(RuntimeError):
    """Base class for MNA assembly and numerical errors."""


class UnsupportedElementError(MNAError):
    """Raised when the selected analysis cannot model an element."""


class SingularMatrixError(MNAError):
    """Raised when the circuit equations are singular or underdetermined."""


class NumericalSolveError(MNAError):
    """Raised when a numerical result is non-finite or fails residual checks."""
