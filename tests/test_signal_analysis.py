from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import sawtooth

from circhecker import CircuitReviewer, SignalAnalyzer
from circhecker.parser.spice import parse_spice_text
from circhecker.solver.transient import _next_breakpoint, _source_function


@pytest.fixture
def analyzer():
    return SignalAnalyzer()


def _record(frequency=50.0, sample_rate=4096.0, duration=1.0):
    count = int(sample_rate * duration)
    time = np.arange(count, dtype=float) / sample_rate
    return time, frequency, sample_rate


def test_pure_sine_metrics_spectrum_and_phase(analyzer):
    time, f0, fs = _record()
    phase = 0.37
    values = 2.5 * np.sin(2 * np.pi * f0 * time + phase)
    report = analyzer.analyze(time, values, fs, fundamental_hz=f0, window="boxcar")
    assert report.mean == pytest.approx(0.0, abs=1e-12)
    assert report.rms == pytest.approx(2.5 / math.sqrt(2), abs=1e-12)
    assert report.peak == pytest.approx(2.5, abs=3e-6)
    assert report.spectrum["dominant_frequency_hz"] == pytest.approx(f0)
    assert report.spectrum["dominant_amplitude"] == pytest.approx(2.5, rel=1e-10)
    assert report.spectrum["fundamental_phase_deg"] == pytest.approx(
        math.degrees(phase - math.pi / 2), abs=1e-8
    )
    assert report.harmonics["thd_percent"] == pytest.approx(0.0, abs=1e-8)
    assert np.sum(report.spectrum["power_spectral_density_per_hz"]) * fs / time.size == pytest.approx(
        report.rms**2, rel=1e-12
    )


def test_sine_with_dc_offset_and_fft_dc_bin(analyzer):
    time, f0, fs = _record()
    values = 3.2 + 1.25 * np.sin(2 * np.pi * f0 * time)
    report = analyzer.analyze(time, values, fs, fundamental_hz=f0, window="boxcar")
    assert report.dc_offset == pytest.approx(3.2, abs=1e-12)
    assert report.spectrum["amplitude"][0] == pytest.approx(3.2, abs=1e-12)
    assert report.spectrum["dominant_amplitude"] == pytest.approx(1.25, rel=1e-10)


def test_square_wave_harmonics_and_thd(analyzer):
    time, f0, fs = _record()
    values = np.zeros_like(time)
    odd_orders = list(range(1, 32, 2))
    for order in odd_orders:
        values += (4.0 / (np.pi * order)) * np.sin(2 * np.pi * order * f0 * time)
    report = analyzer.analyze(time, values, fs, fundamental_hz=f0, window="boxcar")
    expected = math.sqrt(sum((1.0/order)**2 for order in odd_orders[1:])) * 100
    assert report.harmonics["thd_percent"] == pytest.approx(expected, abs=1e-8)


def test_triangle_sawtooth_and_two_tone_dominant_frequencies(analyzer):
    time, f0, fs = _record()
    triangle = sawtooth(2 * np.pi * f0 * time, width=0.5)
    tri = analyzer.analyze(time, triangle, fs, fundamental_hz=f0, window="boxcar")
    assert tri.spectrum["dominant_frequency_hz"] == pytest.approx(f0)
    saw = analyzer.analyze(time, sawtooth(2 * np.pi * f0 * time), fs, window="boxcar")
    assert saw.spectrum["dominant_frequency_hz"] == pytest.approx(f0)
    two_tone = np.sin(2*np.pi*40*time) + 0.4*np.sin(2*np.pi*125*time)
    two = analyzer.analyze(time, two_tone, fs, window="boxcar")
    assert two.spectrum["dominant_frequency_hz"] == pytest.approx(40.0)


def test_harmonic_rich_noise_am_and_fm_signals_are_finite(analyzer):
    time, f0, fs = _record()
    harmonic = (np.sin(2*np.pi*f0*time) + 0.3*np.sin(2*np.pi*2*f0*time)
                + 0.12*np.sin(2*np.pi*3*f0*time))
    report = analyzer.analyze(time, harmonic, fs, fundamental_hz=f0)
    assert 0 < report.harmonics["thd_percent"] < 40
    rng = np.random.default_rng(2026)
    noisy = np.sin(2*np.pi*f0*time) + 0.05*rng.normal(size=time.size)
    assert np.isfinite(analyzer.analyze(time, noisy, fs).spectrum["dominant_frequency_hz"])
    carrier, modulation = 250.0, 5.0
    am = (1 + 0.5*np.cos(2*np.pi*modulation*time))*np.cos(2*np.pi*carrier*time)
    fm = np.sin(2*np.pi*carrier*time + 2*np.sin(2*np.pi*modulation*time))
    assert np.isfinite(analyzer.analyze(time, am, fs).spectrum["dominant_frequency_hz"])
    assert np.isfinite(analyzer.analyze(time, fm, fs).spectrum["dominant_frequency_hz"])


def test_window_coherent_gain_zero_padding_and_nyquist_diagnostics(analyzer):
    time, f0, fs = _record()
    values = np.sin(2*np.pi*f0*time)
    report = analyzer.analyze(time, values, fs, fundamental_hz=f0,
                              window="hann", nfft=8192,
                              expected_max_frequency_hz=fs*0.6)
    assert report.sampling["zero_padding_used"] is True
    assert report.sampling["true_frequency_resolution_hz"] == pytest.approx(1.0)
    assert report.sampling["fft_bin_spacing_hz"] == pytest.approx(0.5)
    assert "does not improve physical frequency resolution" in report.sampling["zero_padding_note"]
    assert report.sampling["aliasing"]["risk"] is True


def test_nonuniform_samples_omit_fft_unless_explicitly_resampled(analyzer):
    base = np.linspace(0, 1, 1001)
    time = base + 1e-4*np.sin(np.arange(base.size))
    time[0] = 0.0
    values = np.sin(2*np.pi*10*time)
    report = analyzer.analyze(time, values)
    assert report.spectrum is None
    assert report.sampling["uniform_sampling"] is False
    resampled = analyzer.analyze(time, values, resample_nonuniform=True)
    assert resampled.spectrum is not None
    assert resampled.sampling["resampling"]["performed"] is True


def test_rc_low_high_pass_and_rlc_resonance_reference_signals(analyzer):
    time, f0, fs = _record()
    omega = 2*np.pi*f0
    tau = 1/(2*np.pi*100.0)
    source = np.sin(omega*time)
    low_gain = 1/math.sqrt(1+(omega*tau)**2)
    low_phase = -math.atan(omega*tau)
    low = low_gain*np.sin(omega*time + low_phase)
    low_report = analyzer.analyze(time, low, fs, fundamental_hz=f0, window="boxcar")
    assert low_report.spectrum["dominant_amplitude"] == pytest.approx(low_gain, rel=1e-7)
    high_gain = omega*tau/math.sqrt(1+(omega*tau)**2)
    high_phase = math.atan(1/(omega*tau))
    high = high_gain*np.sin(omega*time + high_phase)
    high_report = analyzer.analyze(time, high, fs, fundamental_hz=f0, window="boxcar")
    assert high_report.spectrum["dominant_amplitude"] == pytest.approx(high_gain, rel=1e-7)
    # Series-RLC current maximum is at the analytic resonance frequency.
    l_value, c_value, resistance = 10e-3, 1e-6, 10.0
    resonance = 1/(2*np.pi*math.sqrt(l_value*c_value))
    frequencies = np.linspace(100, 3000, 1000)
    omega_sweep = 2*np.pi*frequencies
    current = 1/np.sqrt(resistance**2 + (omega_sweep*l_value - 1/(omega_sweep*c_value))**2)
    peak_frequency = float(frequencies[np.argmax(current)])
    assert abs(peak_frequency-resonance) < 3.0


def test_switching_waveform_statistics_and_json_output(analyzer, tmp_path):
    time, f0, fs = _record(sample_rate=10000, duration=0.2)
    values = (np.mod(time, 1/f0) < 0.5/f0).astype(float)*5.0
    report = analyzer.analyze(time, values, fs, fundamental_hz=f0)
    assert report.minimum == 0
    assert report.maximum == 5
    assert report.ripple_peak_to_peak == 5
    path = tmp_path / "reports" / "signal_report.json"
    encoded = report.to_json(path)
    assert json.loads(encoded)["schema_version"] == "1.0"
    assert json.loads(path.read_text())["sampling"]["uniform_sampling"] is True


def test_ac_frequency_response_analysis(analyzer):
    frequency = np.geomspace(10, 10000, 100)
    response = 1/(1+1j*frequency/1000)
    report = analyzer.analyze_ac(frequency, response, name="V(out)/V(in)")
    assert report["kind"] == "ac_frequency_response"
    assert report["magnitude"][0] > report["magnitude"][-1]
    assert report["phase_deg"][-1] < report["phase_deg"][0]


def test_analyzer_consumes_actual_transient_result(tmp_path):
    text = """V1 in 0 PULSE(0 5 0 1u 1u 5m 10m)
R1 in out 1k
C1 out 0 1u
.tran 10u 2m
.end
"""
    result = CircuitReviewer(output_dir=tmp_path, plot=False).process_netlist_text(
        text, analyses=["transient"], save_json=False
    )
    node_id = str(next(node["id"] for node in result.circuit_ir.nodes if node["name"] == "out"))
    report = SignalAnalyzer().analyze_transient(result, node=node_id)
    assert report.metadata["name"] == "V(out)"
    assert report.sampling["sample_count"] > 100


def test_signal_analyzer_reads_selected_output_from_full_linear_review(tmp_path):
    source = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "06_rc_low_pass.cir"
    result = CircuitReviewer(output_dir=tmp_path, plot=False).process_file(source, save_json=False)
    transient = result.analysis_results["transient"]
    assert transient["status"] == "success"
    assert transient["data"]["output_node"] == "out"

    report = SignalAnalyzer().analyze_transient(result, node="out")
    assert report.metadata["name"] == "V(out)"
    assert report.sampling["sample_count"] == len(transient["data"]["time"])
    assert np.isfinite(report.rms)


def test_signal_statistics_measure_step_metrics(analyzer):
    time = np.linspace(0, 1, 1001)
    values = np.where(time < 0.1, 0.0, 1.0 - np.exp(-(time-0.1)/0.05))
    values = np.where(time < 0.1, 0.0, values)
    report = analyzer.analyze(time, values)
    assert report.rise_time_s is not None
    assert report.rise_time_s == pytest.approx(0.05*math.log(9), abs=0.002)
    assert report.settling_time_s is not None


def test_spice_sin_pulse_pwl_source_values_and_breakpoints():
    sin, sin_breaks = _source_function({"reference": "V1", "parameters": {
        "waveform": {"kind": "SIN", "arguments": [1, 2, 10, 0.1, 0, 90]}}})
    assert sin(0) == pytest.approx(1.0)  # pre-delay output is Voff
    assert sin(0.1) == pytest.approx(3.0)  # phase is applied when the sine starts
    assert sin_breaks == [0.1]
    sin_component = {"parameters": {"waveform": {"kind": "SIN", "arguments": [1, 2, 10, 0.1, 0, 90]}}}
    assert _next_breakpoint(0.0, 0.5, [sin_component]) == pytest.approx(0.1)
    pulse, pulse_breaks = _source_function({"reference": "V2", "parameters": {
        "waveform": {"kind": "PULSE", "arguments": [0, 5, 1, 0.1, 0.2, 0.5, 2]}}})
    assert pulse(0.9) == 0
    assert pulse(1.05) == pytest.approx(2.5)
    assert pulse(1.2) == 5
    assert pulse(1.8) == 0
    assert pulse_breaks == [1, 0.1, 0.2, 0.5, 2]
    pwl, _ = _source_function({"reference": "V3", "parameters": {
        "waveform": {"kind": "PWL", "arguments": [0, 0, 1, 2, 2, 0]}}})
    assert pwl(-1) == 0
    assert pwl(0.5) == pytest.approx(1)
    assert pwl(3) == 0


def test_invalid_sample_vectors_and_sampling_rate_are_rejected(analyzer):
    time = np.arange(10, dtype=float)
    values = np.sin(time)
    with pytest.raises(ValueError, match="strictly increasing"):
        analyzer.analyze([0, 1, 1, 2], [0, 1, 0, 1])
    with pytest.raises(ValueError, match="conflicts with time spacing"):
        analyzer.analyze(time, values, sample_rate=10)


def test_signal_analyzer_accepts_semiconductor_terminal_current(tmp_path):
    text = """V1 src 0 PULSE(0 1 0 1u 1u 1m 2m)
R1 src out 1k
D1 out 0 DM
C1 out 0 1n
.model DM D (IS=1e-14 N=1 CJO=1p)
.tran 1u 100u
.end
"""
    result = CircuitReviewer(output_dir=tmp_path, plot=False).process_netlist_text(
        text, analyses=["transient"], save_json=False
    )
    report = SignalAnalyzer().analyze_transient(result, device="D1", terminal="anode")
    assert report.metadata["name"] == "I(D1.anode)"
    assert report.metadata["unit"] == "A"
    assert len(report.metrics) > 5


def test_dc_only_signal_has_no_spurious_dominant_ac_frequency(analyzer):
    time = np.arange(1024, dtype=float) / 1024.0
    report = analyzer.analyze(time, np.full(time.size, 2.75), window="boxcar")
    assert report.spectrum["dominant_frequency_hz"] is None
    assert report.spectrum["dominant_amplitude"] == 0.0
    assert report.harmonics is None
    assert report.metrics["thd_percent"] is None
