from pathlib import Path
import json
from circhecker import SchematicExtractor

ROOT = Path(__file__).parent


def test_kicad_simple_rc_matches_golden():
    extractor = SchematicExtractor()
    circuit = extractor.extract(ROOT / "fixtures/kicad/simple_rc.kicad_net")
    actual = extractor.to_json_dict(circuit)
    expected = json.loads((ROOT / "golden/kicad_simple_rc.json").read_text())
    assert actual == expected


def test_kicad_complex_features_matches_golden():
    extractor = SchematicExtractor()
    circuit = extractor.extract(ROOT / "fixtures/kicad/complex_features.kicad_net")
    actual = extractor.to_json_dict(circuit)
    expected = json.loads((ROOT / "golden/kicad_complex_features.json").read_text())
    assert actual == expected


def test_spice_hierarchy_matches_golden():
    extractor = SchematicExtractor()
    circuit = extractor.extract(ROOT / "fixtures/spice/hierarchy_example.cir")
    actual = extractor.to_json_dict(circuit)
    expected = json.loads((ROOT / "golden/spice_hierarchy_example.json").read_text())
    assert actual == expected
