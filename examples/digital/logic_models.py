"""Backward-compatible example imports for the public digital reference models."""
from circhecker.digital import (
    HC74, HC595, LogicNetwork, and2, inv, nand2, nor2, or2, xnor2, xor2,
)


def demo() -> None:
    print("74HC00-style NAND2 truth table")
    print("A B | Y")
    for a in (0, 1):
        for b in (0, 1):
            print(f"{a} {b} | {nand2(a, b)}")
    ff = HC74()
    print("\n74HC74: clock a 1 into D")
    print("Q, Qbar =", ff.step(clock=0, d=1))
    print("Q, Qbar =", ff.step(clock=1, d=1))
    sr = HC595()
    for bit in (1, 0, 1, 0, 0, 1, 0, 1):
        sr.step(0, bit, 0)
        sr.step(1, bit, 0)
        sr.step(0, bit, 0)
    print("\n74HC595 output:", sr.step(0, 0, 1))


if __name__ == "__main__":
    demo()
