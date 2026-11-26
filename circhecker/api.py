"""Public entry point for circuit import, analysis, plotting, and JSON reports."""
from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable
from collections.abc import Mapping

from .analysis import AnalysisIR, AnalysisSpec
from .circuit_ir import CircuitIR
from .execution.engine import CircuitExecutionEngine
from .importers.base import ImporterError
from .reporting.result import CircuitResult
from .schematic import SchematicExtractor
from .plugins import ReviewPlugin
from .validation.extraction import validate_extraction
from .validation.units import validate_circuit_units


class CircuitReviewer:
    """Run the complete CirKreety review pipeline on one circuit source.

    Supported inputs are native KiCad schematics (through ``kicad-cli``),
    KiCad XML netlists, and the documented SPICE-compatible netlist subset.
    """

    def __init__(
        self,
        output_dir: str | Path = "reports",
        plot: bool = True,
        kicad_cli: str | Path | None = None,
        review_plugins: Iterable[ReviewPlugin] = (),
    ):
        self.output_dir = Path(output_dir)
        self.default_plot = bool(plot)
        self.extractor = SchematicExtractor(kicad_cli=kicad_cli)
        self.review_plugins = tuple(review_plugins)
        plugin_names = [str(getattr(plugin, "name", "")).strip() for plugin in self.review_plugins]
        if any(not name for name in plugin_names):
            raise ValueError("Every review plugin must expose a non-empty 'name'.")
        if len({name.casefold() for name in plugin_names}) != len(plugin_names):
            raise ValueError("Review plugin names must be unique (case-insensitive).")
        for plugin in self.review_plugins:
            if not callable(getattr(plugin, "run", None)):
                raise TypeError(f"Review plugin {plugin.name!r} must implement run(circuit, execution).")

    def process(
        self,
        source: str | Path,
        analyses: str | Iterable[str] = "all",
        save_json: bool = True,
        plot: bool | None = None,
        *,
        analysis_spec: AnalysisSpec | None = None,
    ) -> CircuitResult:
        """Process a file path or SPICE netlist text."""
        if isinstance(source, Path):
            return self.process_file(source, analyses, save_json, plot, analysis_spec=analysis_spec)

        text = str(source)
        path = Path(text)
        if "\n" not in text and path.is_file():
            return self.process_file(path, analyses, save_json, plot, analysis_spec=analysis_spec)

        if "\n" in text or re.match(
            r"(?is)^\s*(?:\*|//|\.(?:title|param|model|subckt|ends|op|dc|ac|tran|tf|four|meas|ic|nodeset|options|save|probe|include|lib|end)\b|[rclvidqjmeghfswkbtx]\w*\s)",
            text,
        ):
            return self.process_netlist_text(text, analyses, save_json, plot, analysis_spec=analysis_spec)
        return self.process_file(path, analyses, save_json, plot, analysis_spec=analysis_spec)

    def process_file(
        self,
        path: str | Path,
        analyses: str | Iterable[str] = "all",
        save_json: bool = True,
        plot: bool | None = None,
        *,
        analysis_spec: AnalysisSpec | None = None,
    ) -> CircuitResult:
        """Import and review a KiCad or SPICE-compatible circuit file."""
        circuit = self.extractor.extract(path)
        return self._execute(circuit, analyses, save_json, plot, analysis_spec)

    def process_netlist_text(
        self,
        netlist_text: str,
        analyses: str | Iterable[str] = "all",
        save_json: bool = True,
        plot: bool | None = None,
        source_name: str = "<memory>",
        *,
        analysis_spec: AnalysisSpec | None = None,
    ) -> CircuitResult:
        """Review a SPICE-compatible netlist held in a string."""
        circuit = self.extractor.import_spice_text(netlist_text, source_name=source_name)
        return self._execute(circuit, analyses, save_json, plot, analysis_spec)

    def process_ir(
        self,
        circuit: CircuitIR,
        analyses: str | Iterable[str] = "all",
        save_json: bool = True,
        plot: bool | None = None,
        *,
        analysis_spec: AnalysisSpec | None = None,
    ) -> CircuitResult:
        """Review an already parsed :class:`CircuitIR`."""
        if not isinstance(circuit, CircuitIR):
            raise TypeError("process_ir expects a CircuitIR returned by SchematicExtractor.")
        return self._execute(circuit, analyses, save_json, plot, analysis_spec)

    def process_schematic(
        self,
        path: str | Path,
        analyses: str | Iterable[str] | None = "all",
        save_json: bool = True,
        plot: bool | None = None,
        *,
        analysis_spec: AnalysisSpec | None = None,
    ):
        """Explicit KiCad entry point retained for notebook and script clarity."""
        if Path(path).suffix.lower() not in {".kicad_sch", ".kicad_net"}:
            raise ImporterError("KiCad inputs must be .kicad_sch or .kicad_net files.")
        circuit = self.extractor.extract(path)
        if analyses is None:
            if analysis_spec is not None:
                raise ValueError("analysis_spec cannot be supplied when analyses=None requests extraction only.")
            return circuit
        return self._execute(circuit, analyses, save_json, plot, analysis_spec)

    def _execute(self, circuit: CircuitIR, analyses, save_json: bool, plot: bool | None, analysis_spec: AnalysisSpec | None = None):
        if not circuit.metadata.get("title"):
            source_label = circuit.source_information.get("filename") or circuit.source_information.get("path")
            circuit.metadata["title"] = Path(str(source_label)).stem if source_label else "Untitled circuit"
        self._validate(circuit)
        if analysis_spec is not None and not isinstance(analysis_spec, AnalysisSpec):
            raise TypeError("analysis_spec must be an AnalysisSpec instance.")
        if analysis_spec is not None:
            # The explicit request object takes precedence over source directives
            # and the legacy `analyses` argument for this one execution.
            analyses = analysis_spec.analyses
        analysis_ir = AnalysisIR.from_request(circuit, analyses=analyses, analysis_spec=analysis_spec)
        execution = CircuitExecutionEngine(circuit, analysis_ir=analysis_ir).execute()
        plugin_results = {}
        for plugin in self.review_plugins:
            plugin_name = str(plugin.name).strip()
            try:
                plugin_output = plugin.run(circuit, execution)
                if not isinstance(plugin_output, Mapping):
                    raise TypeError("plugin.run() must return a mapping")
                plugin_results[plugin_name] = {"status": "success", "data": dict(plugin_output)}
            except Exception as exc:
                plugin_results[plugin_name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                execution["warnings"].append({
                    "code": "REVIEW_PLUGIN_FAILED",
                    "plugin": plugin_name,
                    "message": f"Review plugin failed without changing core analysis status: {type(exc).__name__}: {exc}",
                })
        execution["plugin_results"] = plugin_results
        result = CircuitResult(circuit, execution, analysis_ir=analysis_ir)

        make_plot = self.default_plot if plot is None else bool(plot)
        source_path = circuit.source_information.get("path", "")
        if str(source_path).startswith("<") or not source_path:
            source_path = circuit.metadata.get("title", "circuit")
        stem = self._safe_stem(source_path)

        json_path = self.output_dir / f"{stem}_analysis.json" if save_json else None
        plot_path = self.output_dir / f"{stem}_analysis.png" if make_plot else None
        if json_path is not None:
            result.artifacts["json"] = str(json_path)
        if plot_path is not None:
            result.artifacts["plot"] = str(plot_path)
            result.plot_all(show=False, path=plot_path)
        if json_path is not None:
            result.to_json(json_path)
        return result

    @staticmethod
    def _safe_stem(value: str | Path) -> str:
        stem = Path(str(value)).stem or "circuit"
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)

    @staticmethod
    def _validate(circuit: CircuitIR) -> None:
        """Add circuit-level checks to the structural extraction report."""
        validation = validate_extraction(circuit)
        warnings = list(validation.get("warnings", []))
        errors = list(validation.get("errors", []))
        known_warning_codes = {item.get("code") for item in warnings}

        for component in circuit.components:
            model_ref = component.get("model_ref")
            if model_ref and model_ref.upper() not in circuit.models and "UNRESOLVED_MODEL" not in known_warning_codes:
                warnings.append({
                    "code": "UNRESOLVED_MODEL",
                    "component": component.get("reference"),
                    "model": model_ref,
                    "message": "Referenced model is not defined in the source; simulation may be unsupported or use an adapter default.",
                })
                known_warning_codes.add("UNRESOLVED_MODEL")

        # Check whether each node belongs to a connected component containing ground.
        adjacency: dict[int, set[int]] = {0: set()}
        for component in circuit.components:
            node_ids = [int(pin.get("node_id", 0)) for pin in component.get("pins", [])]
            for node_id in node_ids:
                adjacency.setdefault(node_id, set()).update(other for other in node_ids if other != node_id)

        reached = {0}
        stack = [0]
        while stack:
            current = stack.pop()
            for neighbour in adjacency.get(current, ()):
                if neighbour not in reached:
                    reached.add(neighbour)
                    stack.append(neighbour)

        floating = [
            node.get("name", str(node.get("id")))
            for node in circuit.nodes
            if int(node.get("id", 0)) != 0 and int(node["id"]) not in reached
        ]
        if floating:
            warnings.append({
                "code": "FLOATING_NODES",
                "nodes": floating,
                "message": "These nodes are not connected to reference node 0; analysis may be singular.",
            })

        unit_check = validate_circuit_units(circuit)
        unit_error_codes = {
            "COMPONENT_UNIT_DIMENSION_MISMATCH", "NON_NUMERIC_COMPONENT_VALUE",
            "NONFINITE_COMPONENT_VALUE", "INVALID_COMPONENT_PARAMETERS",
        }
        for diagnostic in unit_check["diagnostics"]:
            record = {**diagnostic, "scope": "component unit/value validation"}
            (errors if diagnostic["code"] in unit_error_codes else warnings).append(record)
        validation.update({
            "status": "error" if errors else "warning" if warnings else "success",
            "errors": errors,
            "warnings": warnings,
            "unit_check": unit_check,
            "connectivity": {"node_count": max(0, len(circuit.nodes) - 1), "floating_nodes": floating},
        })


__all__ = ["CircuitReviewer"]
