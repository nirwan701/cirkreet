"""SPICE engineering-number parsing and lightweight dimensional checks."""
from __future__ import annotations
import re

# SPICE: M/m means milli; MEG is the unambiguous mega suffix.
_SUFFIX = {
    "t": 1e12, "g": 1e9, "meg": 1e6, "k": 1e3, "mil": 25.4e-6,
    "m": 1e-3, "u": 1e-6, "µ": 1e-6, "n": 1e-9,
    "p": 1e-12, "f": 1e-15,
}
_NUM = re.compile(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)([a-zA-Zµμ]*)$")

UNIT_DIMENSIONS = {
    "ohm": "resistance", "r": "resistance", "f": "capacitance", "farad": "capacitance",
    "h": "inductance", "henry": "inductance", "v": "voltage", "volt": "voltage",
    "a": "current", "amp": "current", "w": "power", "hz": "frequency",
    "s": "time", "sec": "time", "siemens": "conductance", "sma": "conductance",
    "c": "charge", "j": "energy", "k": "temperature", "kelvin": "temperature",
}

UNIT_FOR_KIND = {
    "R": "ohm", "C": "F", "L": "H", "V": "V", "I": "A",
    "D": "A", "Q": "A", "J": "A", "M": "A", "G": "S", "E": "V",
}


def parse_spice_number(value: str | float | int, params: dict | None = None) -> float:
    """Parse SPICE numeric literals. Trailing unit text is ignored after suffix."""
    if isinstance(value, (int, float)):
        return float(value)
    raw = str(value).strip().strip("{}")
    if params and raw in params:
        return float(params[raw])
    # A conservative arithmetic expression subset, with names resolved from .param.
    if params and any(ch in raw for ch in "+-*/()"):
        import ast, operator
        ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
               ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
               ast.UAdd: operator.pos}
        tree = ast.parse(raw, mode="eval")
        def ev(node):
            if isinstance(node, ast.Expression): return ev(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)): return float(node.value)
            if isinstance(node, ast.Name) and node.id in params: return float(params[node.id])
            if isinstance(node, ast.BinOp) and type(node.op) in ops: return ops[type(node.op)](ev(node.left), ev(node.right))
            if isinstance(node, ast.UnaryOp) and type(node.op) in ops: return ops[type(node.op)](ev(node.operand))
            raise ValueError(f"Unsupported expression {raw!r}")
        out = float(ev(tree))
        if out != out or abs(out) == float("inf"): raise ValueError(f"Non-finite value {raw!r}")
        return out
    m = _NUM.match(raw)
    if not m:
        raise ValueError(f"Invalid SPICE number or unsupported expression: {value!r}")
    number, suffix = m.groups()
    suffix = suffix.lower().replace("μ", "µ")
    scale = 1.0
    if suffix:
        # Longest recognised leading suffix wins; text after it is a unit label.
        for key in sorted(_SUFFIX, key=len, reverse=True):
            if suffix.startswith(key):
                scale = _SUFFIX[key]
                break
        else:
            # Permit a trailing unit word when no engineering prefix is present.
            if suffix not in {"ohm", "v", "volt", "a", "amp", "f", "farad", "h", "henry", "hz", "s", "sec", "w", "j", "c", "k", "kelvin"}:
                raise ValueError(f"Unknown engineering suffix in {value!r}")
    return float(number) * scale


def engineering_value(value: float, unit: str) -> dict:
    return {"value": float(value), "unit": unit}


def check_component_units(kind: str, value: float) -> list[str]:
    warnings = []
    if kind == "R" and value <= 0: warnings.append("Resistance must be positive.")
    if kind in ("C", "L") and value <= 0: warnings.append(f"{kind} value must be positive.")
    return warnings
