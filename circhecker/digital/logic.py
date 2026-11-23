"""Digital functional reference models that complement the electrical solver.

The combinational network is an ideal, zero-delay Boolean evaluator. HC74 and
HC595 are ideal state models. None of these calculate analogue voltage,
propagation delay, fan-out, setup/hold margins, metastability, or power. For
transistor-level electrical switching, use supported MOSFET netlists with the
analogue solver instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable, Mapping


def _bit(value: int | bool, name: str = "logic value") -> int:
    if value not in (0, 1, False, True):
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")
    return int(value)


def inv(a: int | bool) -> int:
    return 1 - _bit(a, "a")


def nand2(a: int | bool, b: int | bool) -> int:
    return 1 - (_bit(a, "a") & _bit(b, "b"))


def and2(a: int | bool, b: int | bool) -> int:
    return _bit(a, "a") & _bit(b, "b")


def or2(a: int | bool, b: int | bool) -> int:
    return _bit(a, "a") | _bit(b, "b")


def nor2(a: int | bool, b: int | bool) -> int:
    return 1 - (_bit(a, "a") | _bit(b, "b"))


def xor2(a: int | bool, b: int | bool) -> int:
    return _bit(a, "a") ^ _bit(b, "b")


def xnor2(a: int | bool, b: int | bool) -> int:
    return 1 - xor2(a, b)


_GATE_FUNCTIONS: dict[str, tuple[int, Callable[..., int]]] = {
    "NOT": (1, inv),
    "AND": (2, and2),
    "NAND": (2, nand2),
    "OR": (2, or2),
    "NOR": (2, nor2),
    "XOR": (2, xor2),
    "XNOR": (2, xnor2),
}


class LogicNetwork:
    """Small, deterministic combinational logic network evaluator.

    Add gates in any order; evaluation resolves dependencies topologically.
    A wire name may refer either to a supplied primary input or to a gate
    output. Cycles, duplicate output names, missing inputs and invalid bits
    are rejected with actionable errors.
    """

    def __init__(self) -> None:
        self._gates: dict[str, tuple[str, tuple[str, ...]]] = {}

    @property
    def gates(self) -> Mapping[str, tuple[str, tuple[str, ...]]]:
        return dict(self._gates)

    def add_gate(self, output: str, operation: str, *inputs: str) -> "LogicNetwork":
        output = str(output).strip()
        operation = str(operation).strip().upper()
        input_names = tuple(str(name).strip() for name in inputs)
        if not output:
            raise ValueError("Gate output name must not be empty.")
        if output in self._gates:
            raise ValueError(f"A gate already drives output {output!r}.")
        if operation not in _GATE_FUNCTIONS:
            raise ValueError(f"Unsupported gate {operation!r}; choose from {', '.join(_GATE_FUNCTIONS)}.")
        arity, _ = _GATE_FUNCTIONS[operation]
        if len(input_names) != arity or any(not name for name in input_names):
            raise ValueError(f"{operation} requires {arity} named input(s).")
        if output in input_names:
            raise ValueError(f"Gate {output!r} cannot directly use itself as an input.")
        self._gates[output] = (operation, input_names)
        return self

    def evaluate(self, inputs: Mapping[str, int | bool]) -> dict[str, int]:
        values = {str(name): _bit(value, str(name)) for name, value in inputs.items()}
        overlap = set(values).intersection(self._gates)
        if overlap:
            raise ValueError(f"Primary inputs cannot also be gate outputs: {sorted(overlap)}")
        pending = dict(self._gates)
        while pending:
            progressed = False
            for output, (operation, names) in list(pending.items()):
                if all(name in values for name in names):
                    _, function = _GATE_FUNCTIONS[operation]
                    values[output] = function(*(values[name] for name in names))
                    del pending[output]
                    progressed = True
            if not progressed:
                unresolved = {name: [wire for wire in wires if wire not in values]
                              for name, (_, wires) in pending.items()}
                raise ValueError(
                    "Logic network has unresolved inputs or a combinational cycle: "
                    f"{unresolved}"
                )
        return values

    def truth_table(
        self,
        input_names: Iterable[str],
        output_names: Iterable[str] | None = None,
    ) -> list[dict[str, int]]:
        names = tuple(str(name) for name in input_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("Provide one or more unique primary input names.")
        outputs = tuple(output_names) if output_names is not None else tuple(self._gates)
        missing = set(outputs).difference(self._gates)
        if missing:
            raise KeyError(f"Unknown logic output(s): {sorted(missing)}")
        rows: list[dict[str, int]] = []
        for bits in product((0, 1), repeat=len(names)):
            values = self.evaluate(dict(zip(names, bits)))
            rows.append({**dict(zip(names, bits)), **{name: values[name] for name in outputs}})
        return rows


@dataclass
class HC74:
    """Ideal positive-edge-triggered 74HC74 D flip-flop functional model."""

    q: int = 0
    previous_clock: int = 0

    def __post_init__(self) -> None:
        self.q = _bit(self.q, "q")
        self.previous_clock = _bit(self.previous_clock, "previous_clock")

    @property
    def q_bar(self) -> int:
        return 1 - self.q

    def step(
        self, clock: int | bool, d: int | bool,
        clear_n: int | bool = 1, preset_n: int | bool = 1,
    ) -> tuple[int, int]:
        clk = _bit(clock, "clock")
        data = _bit(d, "d")
        clr = _bit(clear_n, "clear_n")
        pre = _bit(preset_n, "preset_n")
        if clr == 0 and pre == 0:
            raise ValueError("74HC74 clear_n and preset_n must not both be low")
        if clr == 0:
            self.q = 0
        elif pre == 0:
            self.q = 1
        elif self.previous_clock == 0 and clk == 1:
            self.q = data
        self.previous_clock = clk
        return self.q, self.q_bar


@dataclass
class HC595:
    """Ideal 8-bit 74HC595-style shift register with a separate output latch."""

    shift_register: int = 0
    output_register: int = 0
    previous_shift_clock: int = 0
    previous_latch_clock: int = 0

    def __post_init__(self) -> None:
        self.shift_register = int(self.shift_register)
        self.output_register = int(self.output_register)
        if not 0 <= self.shift_register <= 0xFF or not 0 <= self.output_register <= 0xFF:
            raise ValueError("74HC595 register values must fit in 8 bits.")
        self.previous_shift_clock = _bit(self.previous_shift_clock, "previous_shift_clock")
        self.previous_latch_clock = _bit(self.previous_latch_clock, "previous_latch_clock")

    @staticmethod
    def _bits(value: int) -> tuple[int, ...]:
        return tuple((value >> bit) & 1 for bit in range(7, -1, -1))

    @property
    def outputs(self) -> tuple[int, ...]:
        return self._bits(self.output_register)

    def step(
        self, shift_clock: int | bool, serial_data: int | bool,
        latch_clock: int | bool, srclr_n: int | bool = 1,
        oe_n: int | bool = 0,
    ) -> tuple[int | None, ...]:
        sclk = _bit(shift_clock, "shift_clock")
        data = _bit(serial_data, "serial_data")
        lclk = _bit(latch_clock, "latch_clock")
        clear = _bit(srclr_n, "srclr_n")
        output_disable = _bit(oe_n, "oe_n")
        if clear == 0:
            self.shift_register = 0
        elif self.previous_shift_clock == 0 and sclk == 1:
            self.shift_register = ((self.shift_register << 1) | data) & 0xFF
        if self.previous_latch_clock == 0 and lclk == 1:
            self.output_register = self.shift_register
        self.previous_shift_clock = sclk
        self.previous_latch_clock = lclk
        if output_disable:
            return (None,) * 8
        return self.outputs
