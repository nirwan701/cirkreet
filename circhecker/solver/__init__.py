"""Generalized MNA solver interfaces."""
from .exceptions import MNAError, NumericalSolveError, SingularMatrixError, UnsupportedElementError
from .mna import MNASolution, assemble_dc, model_from_circuit_ir, solve_dc, supports_circuit_ir_dc
from .ac import ACAnalysisSolution, assemble_ac, frequency_sweep, solve_ac, supports_circuit_ir_ac
from .devices import (DiodeEvaluation, ShockleyDiode, DeviceEvaluation, NonlinearDevice,
                      GummelPoonBJT, Level1MOSFET, Level1JFET)
from .nonlinear import (NonlinearMNASolution, solve_nonlinear_dc, supports_circuit_ir_nonlinear_dc,
                        assemble_nonlinear_ac, solve_nonlinear_ac)
from .transient import TransientSolution, solve_transient, supports_circuit_ir_transient
from .nonlinear_transient import (solve_nonlinear_transient,
                                  supports_circuit_ir_nonlinear_transient)

__all__ = [
    "MNAError", "MNASolution", "NumericalSolveError", "SingularMatrixError",
    "UnsupportedElementError", "assemble_dc", "model_from_circuit_ir", "solve_dc",
    "supports_circuit_ir_dc", "ACAnalysisSolution", "assemble_ac", "frequency_sweep",
    "solve_ac", "supports_circuit_ir_ac", "DiodeEvaluation", "ShockleyDiode",
    "DeviceEvaluation", "NonlinearDevice", "GummelPoonBJT", "Level1MOSFET", "Level1JFET",
    "NonlinearMNASolution", "solve_nonlinear_dc", "supports_circuit_ir_nonlinear_dc",
    "assemble_nonlinear_ac", "solve_nonlinear_ac",
    "TransientSolution", "solve_transient", "supports_circuit_ir_transient",
    "solve_nonlinear_transient", "supports_circuit_ir_nonlinear_transient",
]
