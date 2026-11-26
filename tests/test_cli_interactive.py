from __future__ import annotations

from pathlib import Path

from circhecker.cli import main, run_interactive


def test_interactive_cli_runs_a_circuit_with_student_defaults(tmp_path):
    source = tmp_path / "divider.cir"
    source.write_text(
        ".title CLI voltage divider\nV1 in 0 10\nR1 in out 1k\nR2 out 0 1k\n.end\n",
        encoding="utf-8",
    )
    answers = iter([str(source), "1", str(tmp_path / "reports"), "n"])
    output = []
    code = run_interactive(input_fn=lambda _prompt: next(answers), output_fn=output.append)

    assert code == 0
    joined = "\n".join(output)
    assert "Circuit Analysis" in joined
    assert "DC operating point: completed" in joined
    assert "divider_analysis.json" in joined
    assert (tmp_path / "reports" / "divider_analysis.json").is_file()


def test_cli_without_source_opens_interactive_mode(monkeypatch, tmp_path):
    source = tmp_path / "resistor.cir"
    source.write_text(".title resistor\nV1 in 0 5\nR1 in 0 1k\n.end\n", encoding="utf-8")
    answers = iter([str(source), "1", str(tmp_path / "reports2"), "n"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    assert main([]) == 0


def test_cli_help_does_not_require_a_circuit_file(capsys):
    import pytest
    with pytest.raises(SystemExit) as stopped:
        main(["--help"])
    assert stopped.value.code == 0
    assert "--interactive" in capsys.readouterr().out
