from __future__ import annotations

import pytest

from circhecker.parser.spice import parse_spice_text
from circhecker.parser.units import parse_spice_number


def _components(ir):
    return {component["reference"]: component for component in ir.components}


def test_node_names_are_case_insensitive_and_original_aliases_are_kept():
    ir = parse_spice_text("""R1 OUT 0 1k
R2 out 0 2k
.end
""")

    components = _components(ir)
    first_node = components["R1"]["pins"][0]["node_id"]
    second_node = components["R2"]["pins"][0]["node_id"]

    assert first_node == second_node
    node = next(item for item in ir.nodes if item["id"] == first_node)
    assert node["name"] == "OUT"
    assert node["aliases"] == ["out"]


def test_parameter_names_are_case_insensitive():
    ir = parse_spice_text(""".param RBASE=1k
R1 IN 0 {rbase}
.end
""")

    assert _components(ir)["R1"]["parameters"]["value"] == pytest.approx(1000.0)
    assert not ir.validation_results.get("errors")


def test_inline_comments_and_continuations_do_not_change_values():
    ir = parse_spice_text(""".title \"A; B\"
R1 IN OUT 1k ; first line comment
+ TC1=0.001 $ continuation comment
R2 OUT 0 2k
.end
R3 OUT 0 9k
""")

    components = _components(ir)
    assert set(components) == {"R1", "R2"}
    assert components["R1"]["parameters"]["value"] == pytest.approx(1000.0)
    assert components["R1"]["parameters"]["TC1"] == pytest.approx(0.001)
    assert ir.metadata["title"] == '"A; B"'


def test_invalid_value_keeps_component_and_electrical_connectivity():
    ir = parse_spice_text("""V1 IN 0 5
R1 IN OUT {NOT_DEFINED}
R2 OUT 0 1k
.end
""")

    components = _components(ir)
    assert "R1" in components
    assert components["R1"]["validation_status"] == "value_parse_error"
    assert len(components["R1"]["pins"]) == 2
    assert components["R1"]["pins"][0]["node_id"] == next(
        pin["node_id"] for pin in components["V1"]["pins"] if pin["name"] == "+"
    )
    assert components["R1"]["pins"][1]["node_id"] == components["R2"]["pins"][0]["node_id"]
    assert any(error["code"] == "VALUE_PARSE" and error["component"] == "R1"
               for error in ir.validation_results["errors"])


def test_parameter_expressions_support_engineering_suffixes_and_sequential_values():
    ir = parse_spice_text(""".param RBASE=1k RLOAD={RBASE*2} CLOAD={1/(2k)}
R1 IN 0 {rload}
C1 IN 0 {CLOAD}
.end
""")

    components = _components(ir)
    assert components["R1"]["parameters"]["value"] == pytest.approx(2000.0)
    assert components["C1"]["parameters"]["value"] == pytest.approx(0.0005)
    assert not ir.validation_results.get("errors")


def test_non_finite_spice_values_are_rejected():
    with pytest.raises(ValueError, match="Non-finite"):
        parse_spice_number("1e999")


def test_orphan_continuation_is_reported():
    with pytest.raises(ValueError, match="no preceding statement"):
        parse_spice_text("+ 1k\nR1 IN 0 1k\n.end\n")


def test_current_controlled_switch_is_structurally_parsed():
    ir = parse_spice_text("""VCTRL CTRL 0 0
W1 OUT 0 VCTRL SWMOD
.model SWMOD CSW (RON=1 ROFF=1MEG)
.end
""")

    switch = _components(ir)["W1"]
    assert switch["category"] == "switch"
    assert [pin["name"] for pin in switch["pins"]] == ["+", "-"]
    assert len(switch["pins"]) == 2
    assert not any(error.get("component") == "W1" for error in ir.validation_results.get("errors", []))


def test_spice_arithmetic_expression_rejects_unbounded_exponents():
    from circhecker.parser.units import parse_spice_number

    with pytest.raises(ValueError, match="exponent"):
        parse_spice_number("2**10000")


def test_spice_arithmetic_expression_rejects_excessive_length():
    from circhecker.parser.units import parse_spice_number

    with pytest.raises(ValueError, match="256-character"):
        parse_spice_number("1+" + "1" * 300)
