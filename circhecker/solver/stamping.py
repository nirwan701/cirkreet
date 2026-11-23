"""Sparse MNA matrix stamping primitives."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix

from .elements import Element
from .exceptions import UnsupportedElementError


@dataclass
class SparseStampBuilder:
    """Accumulate sparse matrix coefficients and the right-hand side."""

    dimension: int
    rows: list[int] = field(default_factory=list)
    columns: list[int] = field(default_factory=list)
    values: list[complex] = field(default_factory=list)
    dtype: object = float
    rhs: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.rhs = np.zeros(self.dimension, dtype=self.dtype)

    def add(self, row: int, column: int, value: float | complex) -> None:
        if value != 0:
            self.rows.append(int(row))
            self.columns.append(int(column))
            self.values.append(value)

    def add_rhs(self, row: int, value: float | complex) -> None:
        self.rhs[int(row)] += value

    def matrix(self) -> csc_matrix:
        dtype = np.result_type(self.dtype, *self.values) if self.values else self.dtype
        matrix = coo_matrix(
            (np.asarray(self.values, dtype=dtype), (self.rows, self.columns)),
            shape=(self.dimension, self.dimension),
        ).tocsc()
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        return matrix


def stamp_conductance(builder: SparseStampBuilder, p: int | None, n: int | None,
                      conductance: float) -> None:
    """Stamp a passive conductance between nodes p and n (None means ground)."""
    if p is not None:
        builder.add(p, p, conductance)
    if n is not None:
        builder.add(n, n, conductance)
    if p is not None and n is not None:
        builder.add(p, n, -conductance)
        builder.add(n, p, -conductance)


def stamp_branch_current(builder: SparseStampBuilder, p: int | None, n: int | None,
                         branch_index: int) -> None:
    """Stamp a branch-current unknown into the two terminal KCL equations."""
    if p is not None:
        builder.add(p, branch_index, 1.0)
        builder.add(branch_index, p, 1.0)
    if n is not None:
        builder.add(n, branch_index, -1.0)
        builder.add(branch_index, n, -1.0)


@dataclass
class DCElementStamper:
    """Element-level DC stamper separated from model translation and solving.

    New linear element types can extend this interface without changing the
    matrix factorization or CircuitIR adapter.
    """

    builder: SparseStampBuilder
    node_index: dict[int, int]
    branch_index: dict[str, int]
    source_overrides: dict[str, float]
    ignored_dc_elements: list[str] = field(default_factory=list)

    def _node(self, node: int) -> int | None:
        return None if node == 0 else self.node_index[node]

    def _current_source_rhs(self, p: int, n: int, current: float) -> None:
        if p != 0:
            self.builder.add_rhs(self._node(p), -current)  # type: ignore[arg-type]
        if n != 0:
            self.builder.add_rhs(self._node(n), current)  # type: ignore[arg-type]

    def stamp(self, element: Element) -> None:
        kind, nodes = element.kind, element.nodes
        p, n = nodes[:2]
        ip, inn = self._node(p), self._node(n)
        if kind == "R":
            stamp_conductance(self.builder, ip, inn, 1.0 / element.value)
        elif kind == "C":
            self.ignored_dc_elements.append(element.reference)
        elif kind in ("V", "L", "E", "H"):
            branch = self.branch_index[element.reference.casefold()]
            stamp_branch_current(self.builder, ip, inn, branch)
            if kind == "V":
                self.builder.add_rhs(branch, self.source_overrides.get(element.reference.casefold(), element.value))
            elif kind == "L":
                # Ideal inductor in DC: its voltage is constrained to zero.
                self.builder.add_rhs(branch, 0.0)
            elif kind == "E":
                cp, cn = self._node(nodes[2]), self._node(nodes[3])
                if cp is not None:
                    self.builder.add(branch, cp, -element.value)
                if cn is not None:
                    self.builder.add(branch, cn, element.value)
            else:  # CCVS: V(p,n) = transresistance * I(control)
                control = element.control_source.casefold()
                if control not in self.branch_index:
                    raise UnsupportedElementError(
                        f"{element.reference}: controlling source {element.control_source!r} "
                        "has no branch-current unknown."
                    )
                self.builder.add(branch, self.branch_index[control], -element.value)
        elif kind == "I":
            current = self.source_overrides.get(element.reference.casefold(), element.value)
            self._current_source_rhs(p, n, current)
        elif kind == "G":  # VCCS: I(p->n) = gm * (V(cp) - V(cn))
            cp, cn = self._node(nodes[2]), self._node(nodes[3])
            if ip is not None and cp is not None:
                self.builder.add(ip, cp, element.value)
            if ip is not None and cn is not None:
                self.builder.add(ip, cn, -element.value)
            if inn is not None and cp is not None:
                self.builder.add(inn, cp, -element.value)
            if inn is not None and cn is not None:
                self.builder.add(inn, cn, element.value)
        elif kind == "F":  # CCCS: I(p->n) = gain * I(control branch)
            control = element.control_source.casefold()
            if control not in self.branch_index:
                raise UnsupportedElementError(
                    f"{element.reference}: controlling source {element.control_source!r} "
                    "has no branch-current unknown."
                )
            control_index = self.branch_index[control]
            if ip is not None:
                self.builder.add(ip, control_index, element.value)
            if inn is not None:
                self.builder.add(inn, control_index, -element.value)
        else:  # pragma: no cover - guarded by the IR capability check
            raise UnsupportedElementError(f"No DC stamping rule is registered for element type {kind}.")

    def stamp_all(self, elements: Iterable[Element]) -> None:
        for element in elements:
            self.stamp(element)


@dataclass
class ACElementStamper:
    """Stamp linear RLC and controlled elements into a complex AC MNA matrix.

    The branch-current convention matches :class:`DCElementStamper`: positive
    current flows from the first terminal to the second.  Inductor branch rows
    use ``V(p)-V(n)-j*w*L*I=0``.
    """

    builder: SparseStampBuilder
    node_index: dict[int, int]
    branch_index: dict[str, int]
    source_phasors: dict[str, complex]
    omega: float

    def _node(self, node: int) -> int | None:
        return None if node == 0 else self.node_index[node]

    def _current_source_rhs(self, p: int, n: int, current: complex) -> None:
        if p != 0:
            self.builder.add_rhs(self._node(p), -current)  # type: ignore[arg-type]
        if n != 0:
            self.builder.add_rhs(self._node(n), current)  # type: ignore[arg-type]

    def stamp(self, element: Element) -> None:
        kind, nodes = element.kind, element.nodes
        p, n = nodes[:2]
        ip, inn = self._node(p), self._node(n)
        if kind == "R":
            stamp_conductance(self.builder, ip, inn, 1.0 / element.value)
        elif kind == "C":
            stamp_conductance(self.builder, ip, inn, 1j * self.omega * element.value)
        elif kind in ("V", "L", "E", "H"):
            branch = self.branch_index[element.reference.casefold()]
            stamp_branch_current(self.builder, ip, inn, branch)
            if kind == "V":
                self.builder.add_rhs(branch, self.source_phasors.get(element.reference.casefold(), 0j))
            elif kind == "L":
                self.builder.add(branch, branch, -1j * self.omega * element.value)
            elif kind == "E":
                cp, cn = self._node(nodes[2]), self._node(nodes[3])
                if cp is not None:
                    self.builder.add(branch, cp, -element.value)
                if cn is not None:
                    self.builder.add(branch, cn, element.value)
            else:  # CCVS: V(p,n) = transresistance * I(control)
                control = str(element.control_source).casefold()
                if control not in self.branch_index:
                    raise UnsupportedElementError(
                        f"{element.reference}: controlling source {element.control_source!r} "
                        "has no branch-current unknown."
                    )
                self.builder.add(branch, self.branch_index[control], -element.value)
        elif kind == "I":
            current = self.source_phasors.get(element.reference.casefold(), 0j)
            self._current_source_rhs(p, n, current)
        elif kind == "G":
            cp, cn = self._node(nodes[2]), self._node(nodes[3])
            if ip is not None and cp is not None:
                self.builder.add(ip, cp, element.value)
            if ip is not None and cn is not None:
                self.builder.add(ip, cn, -element.value)
            if inn is not None and cp is not None:
                self.builder.add(inn, cp, -element.value)
            if inn is not None and cn is not None:
                self.builder.add(inn, cn, element.value)
        elif kind == "F":
            control = str(element.control_source).casefold()
            if control not in self.branch_index:
                raise UnsupportedElementError(
                    f"{element.reference}: controlling source {element.control_source!r} "
                    "has no branch-current unknown."
                )
            control_index = self.branch_index[control]
            if ip is not None:
                self.builder.add(ip, control_index, element.value)
            if inn is not None:
                self.builder.add(inn, control_index, -element.value)
        else:  # pragma: no cover - guarded by adapter capability checks
            raise UnsupportedElementError(f"No AC stamping rule is registered for element type {kind}.")

    def stamp_all(self, elements: Iterable[Element]) -> None:
        for element in elements:
            self.stamp(element)
