"""Central catalog for reproducible CirKreety circuit examples."""
from __future__ import annotations

from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

_EXAMPLES: tuple[dict[str, Any], ...] = (
    {"id": "rc_lowpass", "title": "RC low-pass filter", "category": "analog",
     "path": "examples/circuits/analog/rc_lowpass.cir",
     "focus": "Step response, cutoff frequency, magnitude and phase.",
     "required_analyses": ["dc_operating_point", "ac_sweep", "transient"]},
    {"id": "series_rlc", "title": "Series RLC resonant network", "category": "analog",
     "path": "examples/circuits/analog/series_rlc.cir",
     "focus": "Resonance, bandwidth, damping and time-domain response.",
     "required_analyses": ["dc_operating_point", "ac_sweep", "transient"]},
    {"id": "bridge_rectifier", "title": "Full-wave bridge rectifier", "category": "analog",
     "path": "examples/circuits/analog/bridge_rectifier.cir",
     "focus": "Nonlinear diode conduction, reservoir ripple and output statistics.",
     "required_analyses": ["dc_operating_point", "transient"]},
    {"id": "diode_clamp", "title": "Diode limiter", "category": "semiconductor",
     "path": "examples/circuits/semiconductor/diode_clamp.cir",
     "focus": "Diode conduction and clipped transient waveform.",
     "required_analyses": ["dc_operating_point", "transient"]},
    {"id": "bjt_common_emitter", "title": "BJT common-emitter amplifier", "category": "semiconductor",
     "path": "examples/circuits/semiconductor/bjt_common_emitter.cir",
     "focus": "Bias point, device currents, AC response and transient behavior.",
     "required_analyses": ["dc_operating_point", "ac_sweep", "transient"]},
    {"id": "mosfet_common_source", "title": "NMOS common-source amplifier", "category": "semiconductor",
     "path": "examples/circuits/semiconductor/mosfet_common_source.cir",
     "focus": "MOSFET bias, small-signal response and pulse-driven output.",
     "required_analyses": ["dc_operating_point", "ac_sweep", "transient"]},
    {"id": "cmos_inverter", "title": "CMOS inverter", "category": "digital-electrical",
     "path": "examples/circuits/digital/cmos_inverter.cir",
     "focus": "Transistor-level NMOS/PMOS switching with capacitive output load.",
     "required_analyses": ["dc_operating_point", "transient"]},
    {"id": "cmos_nand2", "title": "CMOS two-input NAND gate", "category": "digital-electrical",
     "path": "examples/circuits/digital/cmos_nand2.cir",
     "focus": "Complementary MOS series/parallel networks and loaded switching.",
     "required_analyses": ["dc_operating_point", "transient"]},
    {"id": "cmos_nor2", "title": "CMOS two-input NOR gate", "category": "digital-electrical",
     "path": "examples/circuits/digital/cmos_nor2.cir",
     "focus": "Complementary MOS pull-up/pull-down topology and loaded switching.",
     "required_analyses": ["dc_operating_point", "transient"]},
    {"id": "timer_555_style_macro", "title": "555-style timer macro block", "category": "ic-macro-model",
     "path": "examples/circuits/ic/timer_555_style_macro.cir",
     "focus": "Nested subcircuit, RC charge/discharge timing and hysteretic switch output.",
     "required_analyses": ["dc_operating_point", "transient"]},
    {"id": "opamp_subcircuit", "title": "Nested op-amp subcircuit macro-model", "category": "ic-macro-model",
     "path": "examples/circuits/ic/idealized_opamp_subcircuit.cir",
     "focus": "Hierarchical subcircuit expansion and controlled-source AC response.",
     "required_analyses": ["dc_operating_point", "ac_sweep"]},
    {"id": "integrated_signal_chain", "title": "Integrated rectifier, regulator, oscillator and MOSFET chain", "category": "complex",
     "path": "examples/schematics/integrated_ct_supply_oscillator_cs_mosfet.cir",
     "focus": "One connected multi-stage nonlinear circuit using the full review pipeline.",
     "required_analyses": ["dc_operating_point", "dc_sweep", "ac_sweep", "transient"]},
)


def _public_spec(item: dict[str, Any]) -> dict[str, Any]:
    """Return a detached catalog record with source and image artifact paths."""
    image_stem = f"{item['id']}_connectivity"
    return dict(
        item,
        absolute_path=str(ROOT / item["path"]),
        connectivity_png=str(ROOT / "examples" / "images" / "catalog" / f"{image_stem}.png"),
        connectivity_svg=str(ROOT / "examples" / "images" / "catalog" / f"{image_stem}.svg"),
    )


def list_examples(category: str | None = None) -> list[dict[str, Any]]:
    """Return copies of example metadata, optionally filtered by category."""
    selected = _EXAMPLES if category is None else tuple(
        item for item in _EXAMPLES if item["category"] == category
    )
    return [_public_spec(item) for item in selected]


def get_example(example_id: str) -> dict[str, Any]:
    """Resolve an example by its stable identifier."""
    for item in _EXAMPLES:
        if item["id"].casefold() == str(example_id).casefold():
            return _public_spec(item)
    valid = ", ".join(item["id"] for item in _EXAMPLES)
    raise KeyError(f"Unknown example {example_id!r}. Available examples: {valid}")
