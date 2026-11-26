"""Conservative source-format detection."""
from pathlib import Path

SPICE_EXTENSIONS = {".cir", ".net", ".sp", ".spi", ".spice", ".ckt", ".txt"}


def detect_format(source: str | Path) -> str:
    suffix = Path(source).suffix.lower()
    if suffix == ".kicad_sch":
        return "kicad_schematic"
    if suffix in {".kicad_net", ".xml"}:
        return "kicad_xml"
    if suffix in SPICE_EXTENSIONS:
        return "spice"
    if suffix == ".asc":
        return "ltspice_asc"
    raise ValueError(f"Unsupported source extension {suffix!r}. Supported: KiCad .kicad_sch/.kicad_net and SPICE {', '.join(sorted(SPICE_EXTENSIONS))}.")
