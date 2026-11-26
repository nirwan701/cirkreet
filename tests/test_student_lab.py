from __future__ import annotations

from examples.student_lab import run_student_lab


def test_student_rc_lab_runs_from_python_without_plotting(tmp_path):
    result = run_student_lab(tmp_path, plot=False)
    assert result.status == "success"
    assert result.analysis_results["dc_operating_point"]["status"] == "success"
    assert result.analysis_results["ac_sweep"]["status"] == "success"
    assert result.analysis_results["transient"]["status"] == "success"
    assert (tmp_path / "student_rc_lowpass_analysis.json").is_file()
