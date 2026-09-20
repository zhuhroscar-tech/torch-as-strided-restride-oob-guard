"""Command-line interface: run the from-scratch diagnosis of the
torch.compile(backend="inductor") as_strided storage-span
miscomputation bug against the currently installed torch build, using
the shared semantic-color design system.
"""
from __future__ import annotations

import argparse
import json
import sys

from .style import print_fields, resolve_style, section, status_headline


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="torch-as-strided-restride-oob-guard",
        description=(
            "Diagnose whether the currently installed torch build's "
            "torch.compile(backend='inductor') miscomputes the storage "
            "span for torch.as_strided views built from a repeated + "
            "sliced buffer (a real weight-tying/broadcast/sliding-window "
            "pattern), producing a spurious out-of-bounds error or a "
            "silently wrong result where eager is correct and "
            "deterministic -- and verify the safe_as_strided() guard "
            "matches eager under compilation. Never trusts a cached or "
            "previously-reported result, always re-runs the repro on "
            "THIS host's actual installed torch version."
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of text")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI color even on a TTY")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__

        print(f"torch-as-strided-restride-oob-guard {__version__}")
        return 0

    from .core import TorchUnavailableError, diagnose

    try:
        report = diagnose()
    except TorchUnavailableError as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, indent=2))
        else:
            style = resolve_style(no_color_flag=args.no_color)
            print(status_headline(style, "fail", f"torch unavailable: {exc}"))
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["guard_fully_correct"] else 1

    style = resolve_style(no_color_flag=args.no_color)
    print_fields([("torch version", report["torch_version"])])

    if report["any_divergence_bug"]:
        print(status_headline(style, "warn", "as_strided restride divergence reproduced on this host"))
    else:
        print(status_headline(style, "info", "no as_strided restride divergence reproduced on this host's installed torch build"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "safe_as_strided() matches eager on every call, including under torch.compile"))
    else:
        print(status_headline(style, "fail", "guard did NOT match eager on at least one call"))

    section("calls (index -> input values -> eager vs compiled vs guard)")
    for c in report["cases"]:
        flag = "DIVERGENCE" if c["divergence_bug"] else "ok"
        guard_flag = "guard-ok" if c["guard_matches_eager"] else "GUARD-FAILED"
        compiled_desc = c["compiled_error"] if not c["compiled_ok"] else str(c["compiled_value"])
        print_fields(
            [
                (
                    f"call {c['call_index']}",
                    f"x={c['x_values']!s:16s} eager={c['eager_value']}  "
                    f"compiled={compiled_desc}  {flag:10s}  {guard_flag}",
                )
            ]
        )

    return 0 if report["guard_fully_correct"] else 1


if __name__ == "__main__":
    sys.exit(main())
