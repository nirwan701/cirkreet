from __future__ import annotations

import json
from pathlib import Path

import pytest

from circhecker import CircuitReviewer, capability_profile
from examples.plugins.resistor_power_rating import ResistorPowerRatingPlugin

ROOT = Path(__file__).resolve().parents[1]
INTEGRATED = ROOT / "examples" / "schematics" / "integrated_ct_supply_oscillator_cs_mosfet.cir"


def test_public_capability_profile_is_json_serializable_and_advisory():
    profile = capability_profile()
    assert "M" in profile["components"]
    assert "nonlinear_mna_dc" in profile["solvers"]
    assert "topology" in profile["scope_note"].lower() or "conditional" in profile["scope_note"].lower()
    json.dumps(profile)


def test_review_plugin_results_are_included_in_json_report(tmp_path):
    plugin = ResistorPowerRatingPlugin({"Rseries": 0.25}, safety_fraction=0.5)
    result = CircuitReviewer(output_dir=tmp_path, plot=False, review_plugins=[plugin]).process_file(
        INTEGRATED, analyses=["dc_operating_point"], save_json=True, plot=False
    )
    record = result.plugin_results[plugin.name]
    assert record["status"] == "success"
    check = record["data"]["checks"][0]
    assert check["reference"] == "Rseries"
    assert check["status"] in {"within_limit", "over_limit"}
    document = json.loads((tmp_path / f"{INTEGRATED.stem}_analysis.json").read_text())
    assert document["plugin_results"][plugin.name]["status"] == "success"


def test_review_plugin_failure_is_visible_but_does_not_corrupt_core_analysis(tmp_path):
    class BrokenPlugin:
        name = "broken"

        def run(self, circuit, execution):
            raise RuntimeError("deliberate test failure")

    result = CircuitReviewer(output_dir=tmp_path, plot=False, review_plugins=[BrokenPlugin()]).process_file(
        INTEGRATED, analyses=["dc_operating_point"], save_json=False, plot=False
    )
    assert result.status == "success"
    assert result.plugin_results["broken"]["status"] == "failed"
    assert any(item.get("code") == "REVIEW_PLUGIN_FAILED" for item in result.warnings)


def test_duplicate_plugin_names_are_rejected():
    class Plugin:
        name = "same"

        def run(self, circuit, execution):
            return {}

    with pytest.raises(ValueError, match="unique"):
        CircuitReviewer(review_plugins=[Plugin(), Plugin()])
