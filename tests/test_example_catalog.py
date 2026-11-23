"""Every public circuit example must parse, validate and run its declared analyses."""
from __future__ import annotations

import pytest

from circhecker import CircuitReviewer, SchematicExtractor
from examples.catalog import list_examples


# The integrated design already has its own end-to-end regression in
# test_unified.py; keep this catalog contract focused on individual examples.
@pytest.mark.parametrize(
    "spec",
    [item for item in list_examples() if item["id"] != "integrated_signal_chain"],
    ids=lambda spec: spec["id"],
)
def test_catalog_circuit_runs_declared_analyses(spec, tmp_path):
    source = spec["absolute_path"]
    extractor = SchematicExtractor()
    circuit = extractor.extract(source)
    structural = extractor.validate(circuit)
    assert not structural.get("errors"), structural.get("errors")

    result = CircuitReviewer(output_dir=tmp_path / spec["id"], plot=False).process_ir(
        circuit,
        analyses=spec["required_analyses"],
        save_json=False,
        plot=False,
    )
    summary = result.summary()
    assert summary["status"] == "success", summary
    for analysis in spec["required_analyses"]:
        assert result.analysis_results[analysis]["status"] == "success", (
            spec["id"], analysis, result.analysis_results[analysis]
        )


def test_every_catalog_entry_exposes_existing_connectivity_images():
    for spec in list_examples():
        assert spec["connectivity_png"].endswith(f"{spec['id']}_connectivity.png")
        assert spec["connectivity_svg"].endswith(f"{spec['id']}_connectivity.svg")
        assert __import__("pathlib").Path(spec["connectivity_png"]).is_file(), spec["id"]
        assert __import__("pathlib").Path(spec["connectivity_svg"]).is_file(), spec["id"]
