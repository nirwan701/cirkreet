"""SPICE engineering-number parsing and lightweight dimensional checks."""
from __future__ import annotations
import math
import re

# SPICE: M/m means milli; MEG is the unambiguous mega suffix.
_SUFFIX = {
    "t": 1e12, "g": 1e9, "meg": 1e6, "k": 1e3, "mil": 25.4e-6,
    "m": 1e-3, "u": 1e-6, "µ": 1e-6, "n": 1e-9,
    "p": 1e-12, "f": 1e-15,
}
_NUM = re.compile(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)([a-zA-Zµμ]*)$")
_EXPRESSION_SUFFIX = re.compile(
    r"(?<![A-Za-z0-9_.])(?P<number>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?P<suffix>meg|mil|[tgkmunpf])(?![A-Za-z_])",
    re.IGNORECASE,
)

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
    """Parse a SPICE number or a small, safe .param arithmetic expression.

    Parameter identifiers are case-insensitive, like SPICE node and device
    identifiers. Engineering suffixes follow SPICE conventions (``m`` is
    milli; use ``MEG`` for mega).
    """
    if isinstance(value, (int, float)):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"Non-finite SPICE value {value!r}")
        return result

    raw = str(value).strip().strip("{}")
    normalized_params = {str(key).casefold(): float(val) for key, val in (params or {}).items()}
    direct = normalized_params.get(raw.casefold())
    if direct is not None:
        if not math.isfinite(direct):
            raise ValueError(f"Non-finite parameter value {raw!r}")
        return direct

    # Only arithmetic syntax is interpreted. Calls, attributes, indexing and
    # every other AST form are rejected; no Python code is ever evaluated.
    if any(ch in raw for ch in "+-*/()"):
        if len(raw) > 256:
            raise ValueError("SPICE arithmetic expression exceeds the 256-character limit")
        import ast
        import operator

        # Convert SPICE engineering suffixes before parsing the safe arithmetic
        # expression (Python itself does not understand literals such as 2.2k).
        def replace_suffix(match: re.Match[str]) -> str:
            number = float(match.group("number"))
            factor = _SUFFIX[match.group("suffix").lower()]
            return repr(number * factor)

        expression = _EXPRESSION_SUFFIX.sub(replace_suffix, raw)
        operations = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }

        try:
            tree = ast.parse(expression, mode="eval")
            if sum(1 for _ in ast.walk(tree)) > 64:
                raise ValueError("SPICE arithmetic expression is too complex")

            def finite(value: float) -> float:
                result = float(value)
                if not math.isfinite(result):
                    raise ValueError(f"Non-finite intermediate result in {raw!r}")
                return result

            def evaluate(node):
                if isinstance(node, ast.Expression):
                    return evaluate(node.body)
                if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                    return finite(node.value)
                if isinstance(node, ast.Name):
                    key = node.id.casefold()
                    if key not in normalized_params:
                        raise ValueError(f"Unknown parameter {node.id!r}")
                    return finite(normalized_params[key])
                if isinstance(node, ast.BinOp) and type(node.op) in operations:
                    left, right = evaluate(node.left), evaluate(node.right)
                    if isinstance(node.op, ast.Pow) and abs(right) > 100:
                        raise ValueError("SPICE arithmetic exponent magnitude must not exceed 100")
                    return finite(operations[type(node.op)](left, right))
                if isinstance(node, ast.UnaryOp) and type(node.op) in operations:
                    return finite(operations[type(node.op)](evaluate(node.operand)))
                raise ValueError(f"Unsupported expression {raw!r}")

            result = finite(evaluate(tree))
        except (SyntaxError, OverflowError, ZeroDivisionError) as exc:
            raise ValueError(f"Invalid SPICE expression {raw!r}: {exc}") from exc
        if not math.isfinite(result):
            raise ValueError(f"Non-finite value {raw!r}")
        return result

    match = _NUM.match(raw)
    if not match:
        raise ValueError(f"Invalid SPICE number or unsupported expression: {value!r}")
    number, suffix = match.groups()
    suffix = suffix.lower().replace("μ", "µ")
    scale = 1.0
    if suffix:
        # Longest recognised leading suffix wins; following text may be a unit.
        for key in sorted(_SUFFIX, key=len, reverse=True):
            if suffix.startswith(key):
                scale = _SUFFIX[key]
                break
        else:
            allowed_units = {"ohm", "v", "volt", "a", "amp", "f", "farad", "h", "henry",
                             "hz", "s", "sec", "w", "j", "c", "k", "kelvin"}
            if suffix not in allowed_units:
                raise ValueError(f"Unknown engineering suffix in {value!r}")
    result = float(number) * scale
    if not math.isfinite(result):
        raise ValueError(f"Non-finite value {value!r}")
    return result


def engineering_value(value: float, unit: str) -> dict:
    return {"value": float(value), "unit": unit}


def check_component_units(kind: str, value: float) -> list[str]:
    warnings = []
    if kind == "R" and value <= 0: warnings.append("Resistance must be positive.")
    if kind in ("C", "L") and value <= 0: warnings.append(f"{kind} value must be positive.")
    return warnings
