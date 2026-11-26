"""Shared, thin notebook helpers built only on public CirKreety APIs."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from circhecker import CircuitReviewer, SchematicExtractor, SignalAnalyzer
from examples.catalog import ROOT, get_example


def run_example(
    example_id: str,
    *,
    analyses: str | list[str] = "all",
    output_root: str | Path | None = None,
    make_plot: bool = True,
):
    """Import once to CircuitIR, review that same IR, and export artifacts."""
    spec = get_example(example_id)
    source = Path(spec["absolute_path"])
    if not source.is_file():
        raise FileNotFoundError(source)

    output_base = Path(output_root) if output_root is not None else ROOT / "reports" / "notebook_examples"
    destination = output_base / example_id
    destination.mkdir(parents=True, exist_ok=True)

    extractor = SchematicExtractor()
    circuit = extractor.extract(source)
    validation = extractor.validate(circuit)
    if validation.get("errors"):
        raise ValueError(f"CircuitIR validation failed for {example_id}: {validation['errors']}")

    reviewer = CircuitReviewer(output_dir=destination, plot=False)
    result = reviewer.process_ir(circuit, analyses=analyses, save_json=False, plot=False)
    ir_path = destination / "circuit_ir.json"
    analysis_path = destination / "analysis.json"
    plot_path = destination / "review.png" if make_plot else None

    extractor.export_json(circuit, ir_path)
    if make_plot:
        result.plot_all(show=False, path=plot_path)
    def display_path(path: Path) -> str:
        try:
            return path.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            return str(path.resolve())

    result.artifacts.update({
        "example_id": example_id,
        "input_netlist": display_path(source),
        "circuit_ir_json": display_path(ir_path),
        "plot": display_path(plot_path) if plot_path else None,
        "analysis_json": display_path(analysis_path),
    })
    result.to_json(analysis_path)
    return result


def node_voltage(result, node_name: str, *, analysis: str = "dc_operating_point") -> float:
    """Read a named node's DC value from a completed analysis."""
    record = result.analysis_results[analysis]
    if record.get("status") != "success":
        raise ValueError(f"{analysis} did not succeed: {record.get('reason')}")
    node_id = result.circuit_ir.node_map[node_name]
    values = record["data"].get("node_voltages_V", {})
    value = values.get(node_id, values.get(str(node_id)))
    if value is None:
        raise KeyError(f"No voltage for node {node_name!r} in {analysis} output.")
    return float(np.real(value))


def component_metric(result, reference: str) -> dict[str, Any]:
    """Return a component metric record when the selected backend provides it."""
    metrics = result.numerical_metrics.get("component_electrical_metrics", {}).get("components", {})
    for name, record in metrics.items():
        if str(name).casefold() == str(reference).casefold():
            return dict(record)
    raise KeyError(f"No electrical metrics were reported for component {reference!r}.")


def ac_response(result):
    """Return frequency, gain in dB and phase in degrees from the AC result."""
    record = result.analysis_results.get("ac_sweep", {})
    if record.get("status") != "success":
        raise ValueError(f"AC sweep did not succeed: {record.get('reason')}")
    data = record["data"]
    frequency = np.asarray(data.get("frequency_Hz", data.get("frequency")), dtype=float)
    if data.get("magnitude_dB") is not None:
        magnitude_db = np.asarray(data["magnitude_dB"], dtype=float)
    elif data.get("magnitude") is not None:
        magnitude_db = 20.0 * np.log10(np.maximum(np.asarray(data["magnitude"], dtype=float), 1e-300))
    elif data.get("response") is not None:
        magnitude_db = 20.0 * np.log10(np.maximum(np.abs(np.asarray(data["response"])), 1e-300))
    else:
        raise KeyError("AC result has no magnitude response field.")
    phase = data.get("phase_deg", data.get("phase"))
    phase_deg = None if phase is None else np.asarray(phase, dtype=float)
    if phase_deg is not None and data.get("phase_deg") is None:
        phase_deg = np.degrees(phase_deg)
    return frequency, magnitude_db, phase_deg


def transient_trace(result, *, node_name: str | None = None, output: str = "output"):
    """Return a time vector and waveform for a node or selected backend output."""
    record = result.analysis_results.get("transient", {})
    if record.get("status") != "success":
        raise ValueError(f"Transient analysis did not succeed: {record.get('reason')}")
    data = record["data"]

    if "time_s" in data:
        time = np.asarray(data["time_s"], dtype=float)
    elif "time" in data:
        time = np.asarray(data["time"], dtype=float)
    else:
        raise KeyError("Transient result does not contain a time vector.")

    if node_name is not None and data.get("node_voltages_V") is not None:
        node_id = result.circuit_ir.node_map[node_name]
        values = data["node_voltages_V"].get(node_id, data["node_voltages_V"].get(str(node_id)))
        if values is None:
            raise KeyError(f"Transient result has no samples for node {node_name!r}.")
        return time, np.asarray(values, dtype=float)

    if output == "input" and data.get("inputs") is not None:
        array = np.asarray(data["inputs"], dtype=float)
    elif data.get("output_V") is not None:
        array = np.asarray(data["output_V"], dtype=float)
    elif data.get("outputs") is not None:
        array = np.asarray(data["outputs"], dtype=float)
    else:
        raise KeyError("Transient result does not contain an output waveform.")
    return time, array[:, 0] if array.ndim > 1 else array


def measure_signal(time_s, values, name: str = "signal"):
    """Measure a sampled waveform with CirKreety's SignalAnalyzer."""
    return SignalAnalyzer().analyze(time_s, values, name=name, unit="V", window="hann")
