from __future__ import annotations

import pytest

from circhecker import CircuitReviewer
from circhecker.parser.spice import parse_spice_file, parse_spice_text


def _component_map(ir):
    return {item["reference"]: item for item in ir.components}


def test_subcircuit_instance_expands_and_parameter_override_reaches_dc_solver(tmp_path):
    netlist = tmp_path / "divider.cir"
    netlist.write_text(
        """* Parameterized divider subcircuit
.SUBCKT DIV IN OUT PARAMS: R=1k
R1 IN OUT {R}
R2 OUT 0 {R}
.ENDS DIV
V1 source 0 10
X1 source out DIV R=2k
.op
.end
""",
        encoding="utf-8",
    )
    result = CircuitReviewer(output_dir=tmp_path, plot=False).process_file(
        netlist, analyses=["dc_operating_point"], save_json=False
    )
    assert result.analysis_results["dc_operating_point"]["status"] == "success"
    out_id = result.circuit_ir.node_map["out"]
    volts = result.analysis_results["dc_operating_point"]["data"]["node_voltages_V"]
    assert volts[out_id] == pytest.approx(5.0, abs=1e-10)
    components = _component_map(result.circuit_ir)
    assert components["R_X1_1"]["parameters"]["value"] == pytest.approx(2000.0)
    assert components["R_X1_2"]["parameters"]["value"] == pytest.approx(2000.0)
    assert result.circuit_ir.hierarchy[0]["subcircuit"] == "DIV"


def test_subcircuit_instances_have_isolated_internal_nodes_and_references():
    ir = parse_spice_text(
        """.SUBCKT DIV IN OUT
R1 IN MID 1k
R2 MID OUT 1k
.ENDS DIV
V1 in 0 10
XLEFT in left DIV
XRIGHT in right DIV
RLOADL left 0 1k
RLOADR right 0 1k
.end
"""
    )
    components = _component_map(ir)
    assert "R_XLEFT_1" in components and "R_XRIGHT_1" in components
    left_mid = {pin["node_name"] for pin in components["R_XLEFT_1"]["pins"]}
    right_mid = {pin["node_name"] for pin in components["R_XRIGHT_1"]["pins"]}
    assert "XLEFT__MID" in left_mid
    assert "XRIGHT__MID" in right_mid
    assert (left_mid - {"in"}).isdisjoint(right_mid - {"in"})
    assert ir.provenance["subcircuit_expansion"]["instance_count"] == 2


def test_nested_subcircuits_expand_recursively_with_hierarchy_paths():
    ir = parse_spice_text(
        """.SUBCKT DIV IN OUT PARAMS: R=1k
R1 IN MID {R}
R2 MID OUT {R}
.ENDS DIV
.SUBCKT TOP IN OUT
XINNER IN OUT DIV R=3k
.ENDS TOP
V1 in 0 12
XTOP in out TOP
.end
"""
    )
    components = _component_map(ir)
    assert "R_XTOP_XINNER_1" in components
    assert "R_XTOP_XINNER_2" in components
    assert components["R_XTOP_XINNER_1"]["parameters"]["value"] == pytest.approx(3000.0)
    assert components["R_XTOP_XINNER_2"]["parameters"]["value"] == pytest.approx(3000.0)
    assert any(item["depth"] == 2 for item in ir.hierarchy)
    assert components["R_XTOP_XINNER_1"]["hierarchy"] == ["XTOP", "XINNER"]


def test_recursive_subcircuit_is_rejected_with_instance_chain():
    with pytest.raises(ValueError, match="Recursive subcircuit instantiation"):
        parse_spice_text(
            """.SUBCKT LOOP A B
X1 A B LOOP
.ENDS LOOP
V1 in 0 1
XTOP in 0 LOOP
.end
"""
        )


def test_include_and_named_library_section_are_loaded_safely(tmp_path):
    (tmp_path / "models.lib").write_text(
        ".lib FAST\n.model DM D (IS=1e-10 N=2)\n.endl FAST\n"
        ".lib ACCURATE\n.model DM D (IS=2e-14 N=1)\n.endl ACCURATE\n",
        encoding="utf-8",
    )
    netlist = tmp_path / "top.cir"
    netlist.write_text(
        '.lib "models.lib" ACCURATE\nV1 in 0 5\nD1 in 0 DM\n.end\n',
        encoding="utf-8",
    )
    ir = parse_spice_file(netlist)
    assert ir.models["DM"]["parameters"]["IS"] == pytest.approx(2e-14)
    assert ir.models["DM"]["parameters"]["N"] == pytest.approx(1.0)
    assert len(ir.source_information["included_files"]) == 1
    assert ir.source_information["included_files"][0]["section"] == "ACCURATE"


def test_include_cycles_and_directory_escape_are_rejected(tmp_path):
    first = tmp_path / "first.cir"
    second = tmp_path / "second.cir"
    first.write_text('.include "second.cir"\n', encoding="utf-8")
    second.write_text('.include "first.cir"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Circular SPICE include"):
        parse_spice_file(first)

    outside = tmp_path.parent / "outside_model.lib"
    outside.write_text(".model DM D (IS=1e-14)\n", encoding="utf-8")
    escaping = tmp_path / "escape.cir"
    escaping.write_text(f'.include "../{outside.name}"\n', encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="escapes the input directory"):
            parse_spice_file(escaping)
    finally:
        outside.unlink(missing_ok=True)


def test_registry_reports_new_opt_in_solvers_and_subcircuit_expansion():
    from circhecker.execution.registry import DEFAULT_COMPONENT_REGISTRY, DEFAULT_SOLVER_REGISTRY

    solvers = DEFAULT_SOLVER_REGISTRY.profile()
    assert "nonlinear_mna_dc" in solvers
    assert "mna_transient" in solvers
    components = DEFAULT_COMPONENT_REGISTRY.profile()
    assert components["D"]["nonlinear_mna"] is True
    assert components["C"]["transient_mna"] is True
    assert "expands positional instances" in components["X"]["notes"]
