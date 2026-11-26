from circhecker.digital import HC74, HC595, LogicNetwork, and2, inv, nand2, nor2, or2, xor2


def test_basic_logic_gate_truth_tables():
    assert [inv(a) for a in (0, 1)] == [1, 0]
    assert [nand2(a, b) for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))] == [1, 1, 1, 0]
    assert [and2(a, b) for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))] == [0, 0, 0, 1]
    assert [or2(a, b) for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))] == [0, 1, 1, 1]
    assert [xor2(a, b) for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))] == [0, 1, 1, 0]
    assert [nor2(a, b) for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))] == [1, 0, 0, 0]


def test_hc74_positive_edge_and_async_controls():
    ff = HC74()
    assert ff.step(0, 1) == (0, 1)  # No rising edge.
    assert ff.step(1, 1) == (1, 0)  # Rising edge captures D.
    assert ff.step(1, 0) == (1, 0)  # No new edge, holds state.
    assert ff.step(0, 0, clear_n=0) == (0, 1)
    assert ff.step(0, 0, preset_n=0) == (1, 0)


def test_hc74_rejects_simultaneous_async_controls():
    ff = HC74()
    try:
        ff.step(0, 0, clear_n=0, preset_n=0)
    except ValueError as exc:
        assert "must not both be low" in str(exc)
    else:
        raise AssertionError("Expected invalid asynchronous control combination")


def test_hc595_shift_then_latch_and_output_enable():
    sr = HC595()
    for bit in (1, 0, 1, 0, 0, 1, 0, 1):
        sr.step(0, bit, 0)
        sr.step(1, bit, 0)
        sr.step(0, bit, 0)
    assert sr.outputs == (0, 0, 0, 0, 0, 0, 0, 0)  # Output latch is separate.
    assert sr.step(0, 0, 1) == (1, 0, 1, 0, 0, 1, 0, 1)
    assert sr.step(0, 0, 1, oe_n=1) == (None,) * 8
    assert sr.step(0, 0, 0, srclr_n=0) == (1, 0, 1, 0, 0, 1, 0, 1)  # Latch holds.


def test_logic_network_builds_and_evaluates_a_two_to_one_mux():
    mux = LogicNetwork()
    mux.add_gate("not_sel", "NOT", "sel")
    mux.add_gate("left", "AND", "d0", "not_sel")
    mux.add_gate("right", "AND", "d1", "sel")
    mux.add_gate("y", "OR", "left", "right")
    rows = mux.truth_table(("d0", "d1", "sel"), ("y",))
    assert [row["y"] for row in rows] == [0, 0, 0, 1, 1, 0, 1, 1]
    assert mux.evaluate({"d0": 1, "d1": 0, "sel": 0})["y"] == 1
    assert mux.evaluate({"d0": 1, "d1": 0, "sel": 1})["y"] == 0


def test_logic_network_rejects_unknown_wires_cycles_and_invalid_gate_arity():
    network = LogicNetwork().add_gate("a_out", "NOT", "missing")
    try:
        network.evaluate({"input": 1})
    except ValueError as exc:
        assert "unresolved" in str(exc)
    else:
        raise AssertionError("Expected unresolved input error")

    cyclic = LogicNetwork().add_gate("a", "NOT", "b").add_gate("b", "NOT", "a")
    try:
        cyclic.evaluate({})
    except ValueError as exc:
        assert "cycle" in str(exc)
    else:
        raise AssertionError("Expected cycle error")

    try:
        LogicNetwork().add_gate("y", "AND", "a")
    except ValueError as exc:
        assert "requires 2" in str(exc)
    else:
        raise AssertionError("Expected arity error")
