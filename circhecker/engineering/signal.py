"""Reproducible time- and frequency-domain analysis for sampled signals.

The API deliberately distinguishes sample-domain measurements from spectral
measurements. Uniform sampling is required for the FFT/PSD path; adaptive
transient records are analysed in the time domain as-is and can be explicitly
resampled with linear interpolation when requested.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import get_window


_EPS = np.finfo(float).eps


def _vector(values: Any, name: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional numeric sequence.") from exc
    if result.ndim != 1 or result.size < 3:
        raise ValueError(f"{name} must contain at least three one-dimensional samples.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _uniform_step(time: np.ndarray, *, rtol: float = 1e-7, atol: float | None = None) -> float | None:
    differences = np.diff(time)
    if np.any(differences <= 0):
        raise ValueError("time values must be strictly increasing with no duplicates.")
    step = float(np.median(differences))
    threshold = max(float(atol or 0.0), abs(step) * rtol)
    return step if np.max(np.abs(differences - step)) <= threshold else None


def _crossing_times(time: np.ndarray, signal: np.ndarray, level: float, direction: str) -> np.ndarray:
    shifted = signal - level
    if direction == "rising":
        indices = np.flatnonzero((shifted[:-1] < 0) & (shifted[1:] >= 0))
    elif direction == "falling":
        indices = np.flatnonzero((shifted[:-1] > 0) & (shifted[1:] <= 0))
    else:
        raise ValueError("direction must be rising or falling.")
    result = []
    for index in indices:
        dy = float(signal[index + 1] - signal[index])
        if dy == 0.0:
            result.append(float(time[index]))
        else:
            fraction = (level - float(signal[index])) / dy
            result.append(float(time[index] + fraction * (time[index + 1] - time[index])))
    return np.asarray(result, dtype=float)


def _time_metrics(time: np.ndarray, signal: np.ndarray, *, settling_fraction: float,
                  edge_thresholds: tuple[float, float]) -> dict[str, Any]:
    uniform = _uniform_step(time) is not None
    # For uniformly sampled records, use sample statistics. For adaptive records,
    # use time-integral-weighted statistics so densely sampled intervals do not
    # receive disproportionate statistical weight.
    if uniform:
        mean = float(np.mean(signal))
        mean_square = float(np.mean(signal * signal))
    else:
        duration = float(time[-1] - time[0])
        mean = float(trapezoid(signal, time) / duration)
        mean_square = float(trapezoid(signal * signal, time) / duration)
    rms = math.sqrt(max(0.0, mean_square))
    peak = float(np.max(np.abs(signal)))
    minimum, maximum = float(np.min(signal)), float(np.max(signal))
    peak_to_peak = maximum - minimum
    ac_rms = math.sqrt(max(0.0, rms * rms - mean * mean))

    # Estimate frequency using mean-removed positive-going zero crossings.
    centered = signal - mean
    crossings = _crossing_times(time, centered, 0.0, "rising")
    periods = np.diff(crossings)
    periods = periods[periods > 0]
    zc_frequency = float(1.0 / np.mean(periods)) if periods.size else None

    initial = float(signal[0])
    tail_count = max(2, int(math.ceil(signal.size * 0.05)))
    final = float(np.mean(signal[-tail_count:]))
    excursion = final - initial
    measurement_range = max(float(np.ptp(signal)), abs(excursion), _EPS)
    tolerance = settling_fraction * max(abs(excursion), measurement_range * 1e-3, _EPS)
    settling_time = None
    in_band = np.abs(signal - final) <= tolerance
    tail_ripple = float(np.ptp(signal[-tail_count:]))
    step_like = (abs(excursion) > max(0.1 * peak_to_peak, 1e-12) and
                 tail_ripple <= max(settling_fraction * max(abs(excursion), peak_to_peak), 1e-12))
    if step_like and np.any(in_band):
        candidates = np.flatnonzero(in_band)
        for index in candidates:
            if np.all(in_band[index:]):
                settling_time = float(time[index] - time[0])
                break

    low_fraction, high_fraction = edge_thresholds
    if not step_like:
        rise_time = fall_time = overshoot = undershoot = None
    elif excursion >= 0:
        low_level = initial + low_fraction * excursion
        high_level = initial + high_fraction * excursion
        low_cross = _crossing_times(time, signal, low_level, "rising")
        high_cross = _crossing_times(time, signal, high_level, "rising")
        rise_time = (float(high_cross[0] - low_cross[0])
                     if low_cross.size and high_cross.size and high_cross[0] >= low_cross[0] else None)
        fall_time = None
        overshoot = max(0.0, maximum - final) / max(abs(excursion), _EPS) * 100.0
        undershoot = max(0.0, final - minimum) / max(abs(excursion), _EPS) * 100.0
    else:
        high_level = initial + low_fraction * excursion
        low_level = initial + high_fraction * excursion
        high_cross = _crossing_times(time, signal, high_level, "falling")
        low_cross = _crossing_times(time, signal, low_level, "falling")
        fall_time = (float(low_cross[0] - high_cross[0])
                     if high_cross.size and low_cross.size and low_cross[0] >= high_cross[0] else None)
        rise_time = None
        overshoot = max(0.0, final - minimum) / max(abs(excursion), _EPS) * 100.0
        undershoot = max(0.0, maximum - final) / max(abs(excursion), _EPS) * 100.0

    return {
        "mean": mean,
        "rms": rms,
        "ac_rms": ac_rms,
        "peak": peak,
        "minimum": minimum,
        "maximum": maximum,
        "peak_to_peak": peak_to_peak,
        "crest_factor": peak / rms if rms > 0 else None,
        "dc_offset": mean,
        "ripple_peak_to_peak": peak_to_peak,
        "zero_crossing_frequency_hz": zc_frequency,
        "settling_time_s": settling_time,
        "rise_time_s": rise_time,
        "fall_time_s": fall_time,
        "overshoot_percent": None if overshoot is None else float(overshoot),
        "undershoot_percent": None if undershoot is None else float(undershoot),
        "initial_value": initial,
        "final_value_estimate": final,
        "settling_band_fraction": float(settling_fraction),
        "edge_thresholds_fraction": [float(low_fraction), float(high_fraction)],
    }


@dataclass
class SignalAnalysisReport:
    """JSON-serializable signal-analysis result."""
    metrics: dict[str, Any]
    sampling: dict[str, Any]
    spectrum: dict[str, Any] | None
    harmonics: dict[str, Any] | None
    metadata: dict[str, Any]

    def __getattr__(self, name: str) -> Any:
        # Convenient read access (report.rms, report.dominant_frequency_hz).
        if name in self.metrics:
            return self.metrics[name]
        if self.spectrum and name in self.spectrum:
            return self.spectrum[name]
        raise AttributeError(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "metrics": self.metrics,
            "sampling": self.sampling,
            "spectrum": self.spectrum,
            "harmonics": self.harmonics,
            "metadata": self.metadata,
        }

    def to_json(self, path: str | Path | None = None, *, indent: int = 2) -> str:
        encoded = json.dumps(self.to_dict(), indent=indent, allow_nan=False, sort_keys=True)
        json.loads(encoded)
        if path is not None:
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_text(encoded + "\n", encoding="utf-8")
            temporary.replace(destination)
        return encoded


class SignalAnalyzer:
    """Analyse time records and sampled frequency responses.

    ``analyze`` never assumes adaptive solver output is uniformly sampled. A
    spectrum is produced only for uniform samples unless the caller explicitly
    enables linear resampling. The reported FFT grid spacing is distinguished
    from the true record-limited frequency resolution.
    """

    WINDOWS = ("boxcar", "hann", "hamming", "blackman", "flattop")

    def analyze(
        self,
        time: Any,
        signal: Any,
        sample_rate: float | None = None,
        *,
        fundamental_hz: float | None = None,
        window: str = "hann",
        nfft: int | None = None,
        resample_nonuniform: bool = False,
        name: str | None = None,
        unit: str | None = None,
        settling_fraction: float = 0.02,
        edge_thresholds: tuple[float, float] = (0.1, 0.9),
        expected_max_frequency_hz: float | None = None,
    ) -> SignalAnalysisReport:
        t = _vector(time, "time")
        x = _vector(signal, "signal")
        if t.size != x.size:
            raise ValueError("time and signal must contain the same number of samples.")
        if np.any(t < 0) or np.any(np.diff(t) <= 0):
            raise ValueError("time must be non-negative and strictly increasing.")
        if not (0.0 < settling_fraction < 1.0):
            raise ValueError("settling_fraction must be between zero and one.")
        if (len(edge_thresholds) != 2 or not 0 < edge_thresholds[0] < edge_thresholds[1] < 1):
            raise ValueError("edge_thresholds must be two increasing fractions between zero and one.")
        if fundamental_hz is not None and (not math.isfinite(fundamental_hz) or fundamental_hz <= 0):
            raise ValueError("fundamental_hz must be finite and positive.")
        if expected_max_frequency_hz is not None and (not math.isfinite(expected_max_frequency_hz)
                                                       or expected_max_frequency_hz <= 0):
            raise ValueError("expected_max_frequency_hz must be finite and positive.")

        if sample_rate is not None:
            requested_rate = float(sample_rate)
            if not math.isfinite(requested_rate) or requested_rate <= 0:
                raise ValueError("sample_rate must be finite and positive.")
        original_step = _uniform_step(t)
        resampling = {"performed": False, "method": None, "assumption": None}
        if original_step is None and resample_nonuniform:
            target_rate = float(sample_rate) if sample_rate is not None else 1.0 / float(np.median(np.diff(t)))
            if not math.isfinite(target_rate) or target_rate <= 0:
                raise ValueError("sample_rate must be finite and positive when resampling.")
            count = int(math.floor((t[-1] - t[0]) * target_rate)) + 1
            if count < 3:
                raise ValueError("Resampling would produce fewer than three samples.")
            uniform_time = t[0] + np.arange(count, dtype=float) / target_rate
            x = np.interp(uniform_time, t, x)
            t = uniform_time
            original_step = 1.0 / target_rate
            resampling = {
                "performed": True,
                "method": "linear interpolation onto a uniform grid",
                "assumption": "The signal is sufficiently smooth between accepted adaptive solver points; interpolation does not restore unresolved bandwidth.",
            }
        elif original_step is None:
            resampling = {
                "performed": False,
                "method": None,
                "assumption": "Input time points are nonuniform; FFT/PSD omitted. Set resample_nonuniform=True to request documented linear interpolation.",
            }

        metrics = _time_metrics(t, x, settling_fraction=settling_fraction, edge_thresholds=edge_thresholds)
        sampling: dict[str, Any] = {
            "sample_count": int(t.size),
            "start_time_s": float(t[0]),
            "stop_time_s": float(t[-1]),
            "record_duration_s": float(t[-1] - t[0]),
            "uniform_sampling": original_step is not None,
            "sample_rate_hz": None if original_step is None else 1.0 / original_step,
            "nyquist_frequency_hz": None if original_step is None else 0.5 / original_step,
            "resampling": resampling,
        }
        if sample_rate is not None and original_step is not None:
            requested_rate = float(sample_rate)
            if not math.isfinite(requested_rate) or requested_rate <= 0:
                raise ValueError("sample_rate must be finite and positive.")
            measured_rate = 1.0 / original_step
            if not math.isclose(requested_rate, measured_rate, rel_tol=1e-5, abs_tol=1e-10):
                raise ValueError(f"sample_rate={requested_rate:g} Hz conflicts with time spacing ({measured_rate:g} Hz).")
            sampling["sample_rate_source"] = "explicit, verified against time vector"
        elif original_step is not None:
            sampling["sample_rate_source"] = "derived from time vector"
        else:
            sampling["sample_rate_source"] = "not uniform; FFT unavailable"

        spectrum = None
        harmonics = None
        if original_step is not None:
            fs = 1.0 / original_step
            n = int(x.size)
            if nfft is None:
                transform_length = n
            else:
                if isinstance(nfft, bool) or int(nfft) != nfft or int(nfft) < n:
                    raise ValueError("nfft must be an integer greater than or equal to the sample count.")
                transform_length = int(nfft)
            normalized_window = str(window).casefold()
            if normalized_window not in self.WINDOWS:
                raise ValueError(f"window must be one of {', '.join(self.WINDOWS)}.")
            w = get_window(normalized_window, n, fftbins=True)
            coherent_sum = float(np.sum(w))
            window_power = float(np.sum(w * w))
            if coherent_sum <= 0 or window_power <= 0:
                raise ValueError("Selected window has invalid normalization.")
            transformed = np.fft.rfft(x * w, n=transform_length)
            frequencies = np.fft.rfftfreq(transform_length, d=original_step)
            amplitude = np.abs(transformed) / coherent_sum
            if amplitude.size > 1:
                amplitude[1:-1 if transform_length % 2 == 0 else None] *= 2.0
            psd = (np.abs(transformed) ** 2) / (fs * window_power)
            if psd.size > 1:
                psd[1:-1 if transform_length % 2 == 0 else None] *= 2.0
            phase = np.angle(transformed)
            non_dc = amplitude.copy()
            non_dc[0] = 0.0
            dominant_index = int(np.argmax(non_dc))
            has_ac_content = bool(np.max(non_dc) > _EPS * max(1.0, float(np.max(np.abs(x)))))
            dominant_frequency = float(frequencies[dominant_index]) if has_ac_content else None
            estimated_fundamental = (float(fundamental_hz) if fundamental_hz is not None else dominant_frequency)
            coherent_cycles = estimated_fundamental * n / fs if estimated_fundamental is not None else None
            coherent = (None if coherent_cycles is None else
                        abs(coherent_cycles - round(coherent_cycles)) <= 1e-8 * max(1.0, abs(coherent_cycles)))
            nyquist = fs / 2.0
            aliasing: dict[str, Any]
            if expected_max_frequency_hz is None:
                aliasing = {"assessed": False, "risk": None,
                            "reason": "Signal bandwidth is unknown; supply expected_max_frequency_hz for a Nyquist check."}
            else:
                risk = expected_max_frequency_hz >= nyquist
                aliasing = {"assessed": True, "risk": bool(risk),
                            "expected_max_frequency_hz": float(expected_max_frequency_hz),
                            "nyquist_frequency_hz": nyquist,
                            "reason": "Expected bandwidth reaches/exceeds Nyquist." if risk else "Expected bandwidth is below Nyquist."}
            sampling.update({
                "fft_length": transform_length,
                "true_frequency_resolution_hz": fs / n,
                "fft_bin_spacing_hz": fs / transform_length,
                "zero_padding_used": transform_length > n,
                "zero_padding_note": "Zero padding interpolates the displayed FFT grid; it does not improve physical frequency resolution." if transform_length > n else None,
                "window": normalized_window,
                "window_coherent_gain": coherent_sum / n,
                "window_power_gain": window_power / n,
                "coherent_sampling_for_fundamental": coherent,
                "spectral_leakage_risk": bool(coherent is False and normalized_window == "boxcar"),
                "aliasing": aliasing,
            })
            spectrum = {
                "frequency_hz": frequencies.tolist(),
                "amplitude": amplitude.tolist(),
                "phase_rad": phase.tolist(),
                "phase_deg": np.degrees(phase).tolist(),
                "power_spectral_density_per_hz": psd.tolist(),
                "dominant_frequency_hz": dominant_frequency,
                "dominant_amplitude": float(amplitude[dominant_index]) if has_ac_content else 0.0,
                "fundamental_frequency_hz": estimated_fundamental,
                "fundamental_bin_frequency_hz": float(frequencies[int(np.argmin(np.abs(frequencies - estimated_fundamental)))]) if estimated_fundamental else None,
                "fundamental_phase_deg": float(np.degrees(phase[int(np.argmin(np.abs(frequencies - estimated_fundamental)))]) if estimated_fundamental else 0.0),
            }
            if estimated_fundamental is not None and estimated_fundamental > 0:
                fundamental_index = int(np.argmin(np.abs(frequencies - estimated_fundamental)))
                fundamental_amplitude = float(amplitude[fundamental_index])
                harmonic_list = []
                harmonic_square_sum = 0.0
                nyquist_harmonic = int(math.floor(nyquist / estimated_fundamental))
                for order in range(1, nyquist_harmonic + 1):
                    target = order * estimated_fundamental
                    index = int(np.argmin(np.abs(frequencies - target)))
                    measured = float(amplitude[index])
                    harmonic_list.append({
                        "order": order,
                        "target_frequency_hz": target,
                        "bin_frequency_hz": float(frequencies[index]),
                        "amplitude": measured,
                    })
                    if order >= 2:
                        harmonic_square_sum += measured * measured
                thd = (math.sqrt(harmonic_square_sum) / fundamental_amplitude * 100.0
                       if fundamental_amplitude > 0 else None)
                harmonics = {
                    "fundamental_frequency_hz": estimated_fundamental,
                    "fundamental_amplitude": fundamental_amplitude,
                    "components": harmonic_list,
                    "thd_percent": thd,
                    "definition": "sqrt(sum of squared harmonic amplitudes from 2 through Nyquist) / fundamental amplitude * 100; nearest FFT bins are used.",
                }
                metrics["thd_percent"] = thd
            else:
                metrics["thd_percent"] = None
        else:
            sampling["spectral_analysis_status"] = "omitted_nonuniform_samples"
            metrics["thd_percent"] = None

        metadata = {"name": name, "unit": unit, "measurement_definitions": {
            "mean": "Arithmetic sample mean for uniform records; time-integral mean for nonuniform records.",
            "rms": "Root-mean-square over the recorded samples (time-integral weighted for nonuniform records).",
            "peak": "Maximum absolute sample magnitude.",
            "ripple_peak_to_peak": "Maximum minus minimum over the complete record; DC offset does not change this difference.",
            "settling_time_s": "First time after which all remaining samples stay within settling_fraction of the estimated final value; measured from record start.",
            "rise_time_s": f"First {edge_thresholds[0]:.0%}-to-{edge_thresholds[1]:.0%} crossing interval for an increasing record.",
            "fall_time_s": f"First {edge_thresholds[1]:.0%}-to-{edge_thresholds[0]:.0%} crossing interval for a decreasing record.",
            "zero_crossing_frequency_hz": "Mean spacing of positive-going zero crossings after subtracting the record mean.",
            "phase": "FFT coefficient phase in radians/degrees relative to the first sample of the record and the selected window.",
        }}
        return SignalAnalysisReport(metrics, sampling, spectrum, harmonics, metadata)

    def analyze_ac(self, frequency_hz: Any, response: Any, *, name: str | None = None,
                   unit: str | None = None) -> dict[str, Any]:
        """Summarize complex small-signal AC response without treating it as time data."""
        f = _vector(frequency_hz, "frequency_hz")
        try:
            h = np.asarray(response, dtype=complex)
        except (TypeError, ValueError) as exc:
            raise ValueError("response must be a one-dimensional complex sequence.") from exc
        if h.ndim != 1 or h.size != f.size or not np.isfinite(h).all():
            raise ValueError("response must be finite, one-dimensional, and match frequency_hz.")
        if np.any(f < 0) or np.any(np.diff(f) <= 0):
            raise ValueError("frequency_hz must be non-negative and strictly increasing.")
        magnitude = np.abs(h)
        phase = np.unwrap(np.angle(h))
        peak_index = int(np.argmax(magnitude))
        return {
            "kind": "ac_frequency_response",
            "name": name,
            "unit": unit,
            "frequency_hz": f.tolist(),
            "response_real": h.real.tolist(),
            "response_imag": h.imag.tolist(),
            "magnitude": magnitude.tolist(),
            "magnitude_db": (20.0 * np.log10(np.maximum(magnitude, np.finfo(float).tiny))).tolist(),
            "phase_rad": phase.tolist(),
            "phase_deg": np.degrees(phase).tolist(),
            "peak_magnitude": float(magnitude[peak_index]),
            "peak_frequency_hz": float(f[peak_index]),
            "assumption": "Input is complex AC response data; it is not uniformly sampled time-domain data.",
        }

    def analyze_transient(self, circuit_result: Any, *, node: str | None = None,
                          branch: str | None = None, device: str | None = None,
                          terminal: str | None = None, name: str | None = None,
                          window: str = "hann", resample_nonuniform: bool = False,
                          **kwargs: Any) -> SignalAnalysisReport:
        """Analyse a node-voltage or branch-current trace from CircuitResult."""
        selected = sum(value is not None for value in (node, branch, device))
        if selected != 1:
            raise ValueError("Specify exactly one of node, branch, or device.")
        if device is not None and not terminal:
            raise ValueError("terminal is required when analysing a device current.")
        if device is None and terminal is not None:
            raise ValueError("terminal can only be used together with device.")
        analyses = getattr(circuit_result, "analysis_results", None)
        if analyses is None and isinstance(circuit_result, Mapping):
            analyses = circuit_result.get("analysis_results", circuit_result)
        if not isinstance(analyses, Mapping):
            raise ValueError("circuit_result does not expose analysis_results.")
        transient_entry = analyses.get("transient", {})
        if transient_entry.get("status") != "success":
            raise ValueError(f"Transient analysis is not successful: {transient_entry.get('reason')}")
        data = transient_entry.get("data") or {}
        time = data.get("time_s", data.get("time"))
        if time is None:
            raise ValueError("Transient result does not contain time_s/time samples.")
        if node is not None:
            node_values = data.get("node_voltages_V") or {}
            node_key = str(node)
            if node_key not in node_values:
                circuit_ir = getattr(circuit_result, "circuit_ir", None)
                if circuit_ir is not None:
                    for node_record in getattr(circuit_ir, "nodes", []):
                        if str(node_record.get("id")) == node_key:
                            node_key = str(node_record.get("name"))
                            break
            values = node_values.get(node_key)
            if values is None:
                # The legacy linear state-space adapter exports only its selected
                # output trace. Use it only when the requested node identity matches.
                output_node = data.get("output_node")
                if output_node is not None and str(output_node).casefold() == node_key.casefold():
                    values = data.get("output_V")
                    if values is None and data.get("outputs") is not None:
                        output_array = np.asarray(data["outputs"])
                        if output_array.ndim == 1:
                            values = output_array
                        elif output_array.ndim == 2 and output_array.shape[1] == 1:
                            values = output_array[:, 0]
                if values is None:
                    raise KeyError(
                        f"Node {node!r} is not present in transient node_voltages_V "
                        f"(available: {', '.join(node_values)}). The selected legacy output "
                        f"trace is only available for node {output_node!r}."
                    )
            return self.analyze(time, values, window=window, resample_nonuniform=resample_nonuniform,
                                name=name or f"V({node_key})", unit="V", **kwargs)
        values = (data.get("branch_currents_A") or {}).get(str(branch))
        if values is None:
            # Branch keys may be normalized case-insensitively by solver output.
            mapping = data.get("branch_currents_A") or {}
            match = next((v for k, v in mapping.items() if str(k).casefold() == str(branch).casefold()), None)
            values = match
        if device is not None:
            device_map = (data.get("device_terminal_conduction_currents_A")
                          or data.get("device_terminal_currents_A") or {})
            device_key = next((key for key in device_map if str(key).casefold() == str(device).casefold()), None)
            if device_key is None:
                raise KeyError(f"Device {device!r} is not present in transient device_terminal_currents_A.")
            terminal_map = device_map[device_key]
            terminal_key = next((key for key in terminal_map if str(key).casefold() == str(terminal).casefold()), None)
            if terminal_key is None:
                raise KeyError(f"Terminal {terminal!r} is unavailable for device {device!r}.")
            return self.analyze(time, terminal_map[terminal_key], window=window,
                                resample_nonuniform=resample_nonuniform,
                                name=name or f"I({device_key}.{terminal_key})", unit="A", **kwargs)
        if values is None:
            raise KeyError(f"Branch {branch!r} is not present in transient branch_currents_A.")
        return self.analyze(time, values, window=window, resample_nonuniform=resample_nonuniform,
                            name=name or f"I({branch})", unit="A", **kwargs)
