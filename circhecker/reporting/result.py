"""Public result object, plot generation, and machine-readable JSON reporting."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .simulation import SimulationResult


def json_safe(x):
    if isinstance(x, dict): return {str(k):json_safe(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [json_safe(v) for v in x]
    if isinstance(x, np.ndarray): return json_safe(x.tolist())
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating, float)):
        v=float(x); return v if math.isfinite(v) else None
    if isinstance(x, complex):
        return {"real":json_safe(x.real),"imag":json_safe(x.imag),"magnitude":json_safe(abs(x)),"phase_deg":json_safe(math.degrees(math.atan2(x.imag,x.real)))}
    if isinstance(x, (np.complexfloating,)): return json_safe(complex(x))
    if isinstance(x, (str,int,bool)) or x is None: return x
    if hasattr(x,"item"):
        try:return json_safe(x.item())
        except Exception:pass
    return str(x)


class CircuitResult:
    def __init__(self, circuit_ir, execution, source_text=None, analysis_ir=None):
        self.circuit_ir=circuit_ir
        self.analysis_ir = analysis_ir if analysis_ir is not None else execution.get("analysis_ir")
        self.analysis_results=execution["analysis_results"]
        self.module_execution=execution["module_execution"]
        self.numerical_metrics=execution["numerical_metrics"]
        self.engineering_checks=execution["engineering_checks"]
        self.warnings=execution["warnings"]
        self.errors=execution["errors"]
        self.execution_summary=execution["execution_summary"]
        self.analysis_configuration=execution.get("analysis_configuration", dict(circuit_ir.analysis_configuration))
        self.simulation_options=execution.get("simulation_options", dict(circuit_ir.simulation_options))
        self.plugin_results=execution.get("plugin_results", {})
        self.source_text=source_text
        self.artifacts={}
        analysis_id = getattr(self.analysis_ir, "analysis_id", None)
        if analysis_id is None and isinstance(self.analysis_ir, dict):
            analysis_id = self.analysis_ir.get("analysis_id")
        self.simulation_result = SimulationResult.from_execution(
            circuit_ir, execution, analysis_id=analysis_id
        )

    @property
    def status(self):
        if self.errors or self.execution_summary.get("failed_analyses"): return "partial" if self.execution_summary.get("successful_analyses") else "failed"
        return "success" if self.execution_summary.get("successful_analyses") else "unsupported"

    def summary(self):
        kcl = self.engineering_checks.get("kcl", {})
        kcl_status = ("passed" if kcl.get("is_balanced") is True else
                      "failed" if kcl.get("is_balanced") is False else "not_available")
        return {
            "title": self.circuit_ir.metadata.get("title", "Imported circuit"),
            "circuit_id": self.circuit_ir.fingerprint(),
            "analysis_id": getattr(self.analysis_ir, "analysis_id", None),
            "status": self.status,
            "validation_status": self.circuit_ir.validation_results.get("status", "unknown"),
            "kcl_check": kcl_status,
            "backend": self.execution_summary.get("backend"),
            "analysis_status": {key: value["status"] for key, value in self.analysis_results.items()},
            "component_count": len(self.circuit_ir.components),
            "node_count": max(0, len(self.circuit_ir.nodes) - 1),
            "warnings": len(self.warnings),
            "errors": len(self.errors),
        }

    def human_summary(self):
        """Return a readable summary suitable for students and CLI users."""
        title = self.circuit_ir.metadata.get("title", "Imported circuit")
        status_labels = {
            "success": "completed successfully",
            "partial": "completed with some issues",
            "failed": "could not be completed",
            "unsupported": "not supported by the available analysis engines",
        }
        lines = [f"Circuit: {title}", f"Overall result: {status_labels.get(self.status, self.status)}"]
        backend = self.execution_summary.get("backend")
        if backend:
            backend_names = {
                "sparse_mna_dc": "native sparse MNA",
                "nonlinear_mna": "native nonlinear MNA",
                "nonlinear_mna_dc": "native nonlinear MNA",
                "mna_transient": "native transient MNA",
                "linear_mna": "native linear circuit solver",
                "nonlinear_analog": "native nonlinear circuit solver",
                "unsupported": "no compatible numerical engine",
            }
            lines.append(f"Analysis engine: {backend_names.get(backend, backend.replace('_', ' '))}")
        lines.append("Analyses:")
        requested = self.execution_summary.get("requested_analyses", [])
        if not requested:
            lines.append("  No analyses were requested.")
        readable_names = {
            "dc_operating_point": "DC operating point",
            "dc_sweep": "DC sweep",
            "ac_sweep": "AC sweep",
            "transient": "Transient",
            "frequency_analysis": "Frequency analysis",
            "transfer_function": "Transfer function",
            "state_space": "State-space analysis",
        }
        for name in requested:
            state = self.analysis_results.get(name, {})
            result_status = state.get("status", "unknown")
            readable_name = readable_names.get(name, name.replace("_", " ").capitalize())
            if result_status == "success":
                lines.append(f"  [OK] {readable_name}: completed")
            elif result_status in {"unsupported", "not_applicable", "skipped"}:
                reason = state.get("reason") or "This analysis was not run."
                lines.append(f"  [INFO] {readable_name}: {reason}")
            else:
                reason = state.get("reason") or "The solver reported an error."
                lines.append(f"  [ISSUE] {readable_name}: {reason}")
        lines.append(
            f"Circuit contents: {len(self.circuit_ir.components)} components, "
            f"{max(0, len(self.circuit_ir.nodes) - 1)} non-ground nodes."
        )
        if self.warnings:
            lines.append(f"Notes: {len(self.warnings)} warning(s). Review them in the JSON report.")
        if self.errors:
            lines.append(f"Errors: {len(self.errors)}. Review them in the JSON report.")
        lines.append("Reminder: numerical results depend on the supported device models and circuit assumptions.")
        return "\n".join(lines)

    def to_dict(self):
        ir = self.circuit_ir.to_dict()
        return json_safe({
            "metadata": ir["metadata"],
            "source": ir["source_information"],
            "circuit_summary": self.summary(),
            "components": ir["components"],
            "nodes": ir["nodes"],
            "branches": ir["branches"],
            "models": ir["models"],
            "parameters": ir["parameters"],
            "schematic_connectivity": {
                "terminals": ir["terminals"], "wires": ir["wires"],
                "labels": ir["labels"], "ports": ir["ports"],
            },
            "validation": ir["validation_results"],
            "module_execution": self.module_execution,
            "analysis_results": self.analysis_results,
            "simulation_result": self.simulation_result.to_dict(),
            "analysis_ir": self.analysis_ir.to_dict() if hasattr(self.analysis_ir, "to_dict") else self.analysis_ir,
            "numerical_metrics": self.numerical_metrics,
            "engineering_checks": self.engineering_checks,
            "warnings": self.warnings,
            "errors": self.errors,
            "provenance": ir["provenance"],
            "execution_summary": self.execution_summary,
            "analysis_configuration": json_safe(self.analysis_configuration),
            "source_analysis_configuration": ir["analysis_configuration"],
            "simulation_options": json_safe(self.simulation_options),
            "plugin_results": self.plugin_results,
            "artifacts": self.artifacts,
            "model_assumptions": {
                "device_models": "Educational nonlinear models (Shockley diode, piecewise-linear Zener, compact BJT, Level-1 MOSFET/JFET); linear MNA for supported passive and controlled-source networks.",
                "not_a_full_spice_replacement": True,
            },
        })

    def to_json(self, path=None, indent=2):
        obj=self.to_dict()
        encoded=json.dumps(obj,indent=indent,ensure_ascii=False,allow_nan=False)
        # Validate generated JSON before writing.
        json.loads(encoded)
        if path is not None:
            p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
            tmp=p.with_suffix(p.suffix+".tmp"); tmp.write_text(encoded,encoding="utf-8"); tmp.replace(p)
        return encoded

    def plot_workbench(self, show=True, figsize=(13, 10), path=None, window_s=0.01):
        """Plot the key waveforms and sweeps of a combined circuit workbench.

        The plot is node-name aware and uses only data already present in this
        :class:`CircuitResult`. The transient panels zoom to the last
        ``window_s`` seconds (except the complete power-conversion panel).
        """
        transient = self.analysis_results.get("transient", {})
        if transient.get("status") != "success" or not transient.get("data"):
            raise ValueError("A successful transient result is required for the workbench plot.")
        data = transient["data"]
        time = np.asarray(data.get("time_s", data.get("time")), dtype=float)
        node_data = data.get("node_voltages_V", {})
        node_ids = {str(node.get("name")): node.get("id") for node in self.circuit_ir.nodes}

        def signal(name):
            node_id = node_ids.get(name)
            if node_id is None:
                return None
            values = node_data.get(node_id, node_data.get(str(node_id)))
            return None if values is None else np.asarray(values, dtype=float)

        start = max(float(time[0]), float(time[-1]) - max(float(window_s), 0.0))
        mask = time >= start
        tw = time[mask]
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle(self.circuit_ir.metadata.get("title", "CirKreety workbench"), fontsize=14)

        # 1) Transformer secondary, rectified reservoir and regulated rail.
        ax = axes[0, 0]
        for name, label in (("sec_a", "Secondary A"), ("sec_b", "Secondary B"),
                            ("rect_raw", "Filtered rectifier rail"), ("vreg", "Shunt-regulated rail")):
            values = signal(name)
            if values is not None and len(values) == len(time):
                ax.plot(time, values, linewidth=1.0, label=label)
        ax.set_title("1. Centre-tapped rectifier and filter")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Voltage (V)"); ax.legend(fontsize=7); ax.grid(True, alpha=0.25)

        # 2) Zoomed regulator output ripple.
        ax = axes[0, 1]
        values = signal("vreg")
        if values is not None and len(values) == len(time):
            selected = values[mask]
            ax.plot(tw, selected, linewidth=1.2, label="V(vreg)")
            span = float(np.ptp(selected))
            pad = max(span * 0.25, 1e-5)
            ax.set_ylim(float(np.min(selected)) - pad, float(np.max(selected)) + pad)
            ax.set_title(f"2. Shunt regulator ripple ({span * 1e3:.2f} mV p-p)")
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "V(vreg) not found", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("2. Shunt regulator")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Voltage (V)"); ax.grid(True, alpha=0.25)

        # 3) RC hysteretic relaxation oscillator.
        ax = axes[1, 0]
        for name, label in (("osc_cap", "Timing capacitor"), ("osc_out", "Switch output")):
            values = signal(name)
            if values is not None and len(values) == len(time):
                ax.plot(tw, values[mask], linewidth=1.0, label=label)
        ax.set_title("3. RC relaxation oscillator")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Voltage (V)"); ax.legend(fontsize=7); ax.grid(True, alpha=0.25)

        # 4) Oscillator-driven common-source MOSFET amplifier. Plot the AC
        # component at the gate and the AC-coupled output, not the separate
        # small-signal AC test source (which is DC-zero in transient analysis).
        ax = axes[1, 1]
        gate_signal, output_signal = signal("gate"), signal("amp_out")
        if gate_signal is not None and output_signal is not None and len(gate_signal) == len(time) and len(output_signal) == len(time):
            gate_window, out_window = gate_signal[mask], output_signal[mask]
            gate_ac = gate_window - float(np.mean(gate_window))
            out_ac = out_window - float(np.mean(out_window))
            ax.plot(tw, gate_ac, linewidth=1.0, label="MOSFET gate AC component")
            ax.plot(tw, out_ac, linewidth=1.1, label="AC-coupled drain output")
            input_rms = float(np.sqrt(np.mean(gate_ac ** 2)))
            output_rms = float(np.sqrt(np.mean(out_ac ** 2)))
            gain = output_rms / input_rms if input_rms > 0 else float("nan")
            ax.set_title(f"4. Common-source MOSFET (large-signal AC RMS ratio {gain:.2f} V/V)")
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "MOSFET gate/output traces not found", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("4. Common-source MOSFET")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("AC voltage (V)"); ax.grid(True, alpha=0.25)

        # 5) DC line-regulation sweep.
        ax = axes[2, 0]
        sweep = self.analysis_results.get("dc_sweep", {})
        if sweep.get("status") == "success" and sweep.get("data"):
            sweep_data = sweep["data"]
            source = sweep_data.get("source_V", sweep_data.get("sweep"))
            output = sweep_data.get("output_V", sweep_data.get("voltage"))
            if source is not None and output is not None:
                ax.plot(source, output, linewidth=1.4)
                ax.set_xlabel("VSEC_A DC sweep (V)"); ax.set_ylabel("VREG (V)")
            else:
                ax.text(0.5, 0.5, "Sweep data unavailable", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "DC sweep not requested", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("5. Single-source DC sweep (diagnostic)"); ax.grid(True, alpha=0.25)

        # 6) AC small-signal response.
        ax = axes[2, 1]
        ac = self.analysis_results.get("ac_sweep", {})
        if ac.get("status") == "success" and ac.get("data"):
            ac_data = ac["data"]
            frequency = ac_data.get("frequency_Hz", ac_data.get("frequency"))
            magnitude = ac_data.get("magnitude_dB")
            if frequency is not None and magnitude is not None:
                ax.semilogx(frequency, magnitude, linewidth=1.2, label="Magnitude")
                phase = ac_data.get("phase_deg", ac_data.get("phase"))
                if phase is not None:
                    phase = np.asarray(phase, dtype=float)
                    if ac_data.get("phase_deg") is None:
                        phase = np.degrees(phase)
                    phase_ax = ax.twinx()
                    phase_ax.semilogx(frequency, phase, linestyle="--", alpha=0.7, label="Phase")
                    phase_ax.set_ylabel("Phase (degrees)")
                    lines, labels = ax.get_legend_handles_labels()
                    lines2, labels2 = phase_ax.get_legend_handles_labels()
                    ax.legend(lines + lines2, labels + labels2, fontsize=7, loc="best")
                ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Gain (dB)")
            else:
                ax.text(0.5, 0.5, "AC response unavailable", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "AC sweep not requested", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("6. Small-signal AC response"); ax.grid(True, which="both", alpha=0.25)

        for axis in axes.flat:
            axis.tick_params(axis="both", labelsize=8)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        if path is not None:
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(destination, dpi=160, bbox_inches="tight")
        if show:
            plt.show()
        elif path is not None:
            plt.close(fig)
        return fig

    def plot_all(self, show=True, figsize=(13,9), path=None):
        """Create the four-panel DC, sweep, AC, and transient review plot."""
        fig,axs=plt.subplots(2,2,figsize=figsize)
        fig.suptitle(self.circuit_ir.metadata.get("title","Circuit analysis"),fontsize=14)
        ax=axs[0,0]; r=self.analysis_results["dc_operating_point"]
        if r["status"]=="success" and r["data"]:
            data=r["data"]
            nodes=data.get("node_voltages_V",{})
            if nodes:
                ns=list(nodes.keys()); vs=[float(np.real(v)) for v in nodes.values()]
                ax.bar([str(n) for n in ns],vs); ax.set_ylabel("Voltage (V)")
            else: ax.text(.5,.5,"No node voltages in result",ha="center",va="center",transform=ax.transAxes)
        else: ax.text(.5,.5,f"{r['status'].upper()}\n{r.get('reason') or ''}",ha="center",va="center",transform=ax.transAxes,wrap=True)
        ax.set_title("1. DC operating point"); ax.grid(True,axis="y",alpha=.25)
        ax=axs[0,1]; r=self.analysis_results["dc_sweep"]
        if r["status"]=="success" and r["data"]:
            d=r["data"]
            x=d.get("source_V",d.get("sweep")); y=d.get("output_V",d.get("voltage"))
            if x is not None and y is not None: ax.plot(x,y); ax.set_xlabel("Sweep value (V/A)"); ax.set_ylabel("Output (V)")
            else: ax.text(.5,.5,"Sweep data unavailable",ha="center",va="center",transform=ax.transAxes)
        else: ax.text(.5,.5,f"{r['status'].upper()}\n{r.get('reason') or ''}",ha="center",va="center",transform=ax.transAxes,wrap=True)
        ax.set_title("2. DC sweep"); ax.grid(True,alpha=.25)
        ax=axs[1,0]; r=self.analysis_results["ac_sweep"]
        if r["status"]=="success" and r["data"]:
            d=r["data"]; f=d.get("frequency_Hz",d.get("frequency")); mag=d.get("magnitude_dB")
            if mag is None and "magnitude" in d: mag=20*np.log10(np.maximum(d["magnitude"],1e-300))
            if f is not None and mag is not None:
                ax.semilogx(f,mag,label="Magnitude")
                phase=d.get("phase_deg",d.get("phase"))
                if phase is not None:
                    phase=np.asarray(phase,dtype=float)
                    # Legacy transferfunction returns radians; analog AC reports degrees.
                    if d.get("phase_deg") is None: phase=np.degrees(phase)
                    axp=ax.twinx(); axp.semilogx(f,phase,linestyle="--",alpha=.7,label="Phase")
                    axp.set_ylabel("Phase (degrees)")
                    lines,labels=ax.get_legend_handles_labels(); lines2,labels2=axp.get_legend_handles_labels()
                    ax.legend(lines+lines2,labels+labels2,fontsize=7,loc="best")
                ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Magnitude (dB)")
            else: ax.text(.5,.5,"AC response data unavailable",ha="center",va="center",transform=ax.transAxes)
        else: ax.text(.5,.5,f"{r['status'].upper()}\n{r.get('reason') or ''}",ha="center",va="center",transform=ax.transAxes,wrap=True)
        ax.set_title("3. AC sweep (small signal where nonlinear)"); ax.grid(True,which="both",alpha=.25)
        ax=axs[1,1]; r=self.analysis_results["transient"]
        if r["status"]=="success" and r["data"]:
            d=r["data"]; t=d.get("time_s",d.get("time")); out=d.get("output_V")
            if out is None and "outputs" in d:
                arr=np.asarray(d["outputs"]); out=arr[:,0] if arr.ndim>1 else arr
            if t is not None and out is not None:
                ax.plot(t,out,label="Output")
                if d.get("inputs") is not None:
                    inp=np.asarray(d["inputs"])
                    if inp.ndim>1: inp=inp[:,0]
                    if len(inp)==len(t): ax.plot(t,inp,alpha=.65,linestyle="--",label="Input")
                elif d.get("node_voltages_V"):
                    # Nonlinear transient data exports voltage arrays by node.
                    for node,values in d["node_voltages_V"].items():
                        values=np.asarray(values)
                        if values.ndim==1 and len(values)==len(t) and not np.allclose(values,out):
                            ax.plot(t,values,alpha=.45,linewidth=.8,label=f"V({node})")
                ax.set_xlabel("Time (s)"); ax.set_ylabel("Voltage (V)"); ax.legend(fontsize=7)
            else: ax.text(.5,.5,"Transient waveform unavailable",ha="center",va="center",transform=ax.transAxes)
        else: ax.text(.5,.5,f"{r['status'].upper()}\n{r.get('reason') or ''}",ha="center",va="center",transform=ax.transAxes,wrap=True)
        ax.set_title("4. Transient response"); ax.grid(True,alpha=.25)
        for a in axs.flat: a.tick_params(axis="both",labelsize=8)
        fig.tight_layout(rect=[0,0,1,.96])
        if path is not None:
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(destination, dpi=160, bbox_inches="tight")
        if show:
            plt.show()
        elif path is not None:
            plt.close(fig)
        return fig
