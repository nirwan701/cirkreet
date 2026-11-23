"""Generate source-grounded CircuitIR connectivity maps for every catalog entry.

These drawings are pin-to-net maps, not replacements for a CAD schematic.
They are generated from each authoritative netlist through SchematicExtractor.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt

from circhecker import SchematicExtractor
from examples.catalog import list_examples


OUTPUT = ROOT / "examples" / "images" / "catalog"


def _display_value(component: dict) -> str:
    params = component.get("parameters", {})
    raw = params.get("raw_value") or component.get("value_raw")
    if raw:
        return str(raw)
    model = component.get("model_ref")
    if model:
        return f"model {model}"
    if "dc" in params:
        return f"DC {params['dc']} V/A"
    return str(component.get("original_expression", "")).strip()[:70]


def render(spec: dict, extractor: SchematicExtractor) -> tuple[Path, Path]:
    circuit = extractor.extract(spec["absolute_path"])
    components = circuit.components
    OUTPUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for component in components:
        pins = component.get("pins", [])
        assignments = "   |   ".join(
            f"{pin.get('name') or pin.get('number', '?')} → {pin.get('node_name', '?')}"
            for pin in pins
        )
        rows.append([
            str(component.get("reference", "?")),
            str(component.get("type", "?")),
            _display_value(component),
            assignments,
        ])

    height = max(3.4, 1.9 + 0.34 * len(rows))
    fig, ax = plt.subplots(figsize=(15, height))
    ax.axis("off")
    ax.set_title(
        f"{spec['title']} — CircuitIR pin-to-net map\n"
        f"{len(components)} components · {len(circuit.nodes) - 1} non-ground nodes · "
        f"{spec['category']}",
        fontsize=15,
        weight="bold",
        pad=18,
    )
    table = ax.table(
        cellText=rows,
        colLabels=["Reference", "Type", "Value / model", "Pin → node connectivity"],
        colWidths=[0.12, 0.07, 0.18, 0.63],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5 if len(rows) < 20 else 7.2)
    table.scale(1.0, 1.5)
    fig.text(
        0.02, 0.015,
        "Generated from the parsed CircuitIR. Node 0 is ground. "
        "This is a connectivity reference, not a graphical CAD schematic; "
        "the .cir source remains authoritative.",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout(rect=(0.01, 0.04, 0.99, 0.96))
    png = OUTPUT / f"{spec['id']}_connectivity.png"
    svg = OUTPUT / f"{spec['id']}_connectivity.svg"
    fig.savefig(png, dpi=170, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return png, svg


def main() -> int:
    extractor = SchematicExtractor()
    for spec in list_examples():
        png, svg = render(spec, extractor)
        print(f"{spec['id']:24s} {png.relative_to(ROOT)}  {svg.relative_to(ROOT)}")
    print(f"Generated connectivity maps for {len(list_examples())} circuit examples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
