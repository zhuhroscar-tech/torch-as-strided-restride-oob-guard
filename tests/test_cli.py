"""Tests for the CLI entry point: argument parsing, --version, --json,
--no-color, and exit codes -- independent of whether torch is installed.
Mirrors the test_cli.py pattern already used across the fleet, with full
branch coverage for every status-line path from the start (learned from
the fleet's own history of a coverage gap being found later)."""
from __future__ import annotations

import json
import runpy
import sys

import pytest

from torch_as_strided_restride_oob_guard import core
from torch_as_strided_restride_oob_guard.cli import main


def _fake_report(**overrides):
    report = {
        "torch_version": "9.9.9-fake",
        "cases": [],
        "any_divergence_bug": False,
        "guard_fully_correct": True,
    }
    report.update(overrides)
    return report


def test_version_flag(capsys):
    code = main(["--version"])
    out = capsys.readouterr().out
    assert code == 0
    assert "torch-as-strided-restride-oob-guard" in out


def test_json_output_is_valid_json_and_reports_guard_status(capsys):
    torch = pytest.importorskip("torch")
    code = main(["--json"])
    out = capsys.readouterr().out
    report = json.loads(out)
    assert "torch_version" in report
    assert report["torch_version"] == torch.__version__
    assert "guard_fully_correct" in report
    assert code in (0, 1)


def test_json_exit_code_matches_guard_fully_correct(capsys):
    pytest.importorskip("torch")
    code = main(["--json"])
    out = capsys.readouterr().out
    report = json.loads(out)
    assert code == (0 if report["guard_fully_correct"] else 1)


def test_text_output_no_color_has_no_ansi_escapes(capsys):
    pytest.importorskip("torch")
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "\x1b[" not in out


def test_text_output_reports_calls_section(capsys):
    pytest.importorskip("torch")
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "calls (index" in out


def test_torch_unavailable_json_mode_reports_error_and_exit_2(monkeypatch, capsys):
    def _raise(*args, **kwargs):
        raise core.TorchUnavailableError("torch is required for diagnosis")

    monkeypatch.setattr(core, "diagnose", _raise)
    code = main(["--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload == {"error": "torch is required for diagnosis"}
    assert code == 2


def test_torch_unavailable_text_mode_reports_fail_headline_and_exit_2(monkeypatch, capsys):
    def _raise(*args, **kwargs):
        raise core.TorchUnavailableError("torch is required for diagnosis")

    monkeypatch.setattr(core, "diagnose", _raise)
    code = main(["--no-color"])
    out = capsys.readouterr().out
    assert "torch unavailable: torch is required for diagnosis" in out
    assert "[X]" in out
    assert code == 2


def test_no_divergence_bug_prints_info_line(monkeypatch, capsys):
    monkeypatch.setattr(core, "diagnose", lambda: _fake_report())
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "no as_strided restride divergence reproduced on this host's installed torch build" in out
    assert "restride divergence reproduced on this host" not in out.replace(
        "no as_strided restride divergence reproduced on this host's installed torch build", ""
    )


def test_divergence_bug_prints_warn_line(monkeypatch, capsys):
    monkeypatch.setattr(core, "diagnose", lambda: _fake_report(any_divergence_bug=True))
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "as_strided restride divergence reproduced on this host" in out


def test_guard_mismatch_prints_fail_line_and_exit_1(monkeypatch, capsys):
    monkeypatch.setattr(core, "diagnose", lambda: _fake_report(guard_fully_correct=False))
    code = main(["--no-color"])
    out = capsys.readouterr().out
    assert "guard did NOT match eager on at least one call" in out
    assert "matches eager on every call" not in out
    assert code == 1


def test_calls_section_renders_compiled_error_when_present(monkeypatch, capsys):
    fake = _fake_report(
        any_divergence_bug=True,
        cases=[
            {
                "call_index": 0,
                "x_values": (7.0, 9.0),
                "eager_value": [[7.0, 9.0]],
                "compiled_ok": False,
                "compiled_value": None,
                "compiled_error": "RuntimeError: setStorage out of bounds",
                "divergence_bug": True,
                "guard_matches_eager": True,
            }
        ],
    )
    monkeypatch.setattr(core, "diagnose", lambda: fake)
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "RuntimeError: setStorage out of bounds" in out
    assert "DIVERGENCE" in out
    assert "guard-ok" in out


def test_module_entry_point_runs_main_and_exits_with_its_code(monkeypatch):
    monkeypatch.setattr(core, "diagnose", lambda: _fake_report(guard_fully_correct=False))
    monkeypatch.setattr(sys, "argv", ["torch-as-strided-restride-oob-guard", "--no-color"])
    monkeypatch.delitem(sys.modules, "torch_as_strided_restride_oob_guard.cli", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("torch_as_strided_restride_oob_guard.cli", run_name="__main__")
    assert exc_info.value.code == 1
