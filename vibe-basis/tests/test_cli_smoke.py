"""CLI smoke tests for vb.

Verifies:
* ``vb --version`` prints something and exits zero.
* ``vb parse crystal14 <successful.out>`` exits zero and prints
  the parsed energy.
* ``vb parse crystal14 <failed.out>`` exits non-zero and prints
  the failure mode to stderr.

Uses Click's CliRunner — no real subprocess, no PATH lookup of
the installed ``vb`` script. The pyproject's ``[project.scripts]``
entry-point is what makes ``vb`` available after install; here we
just exercise the underlying function.
"""

from __future__ import annotations

import textwrap

from click.testing import CliRunner

from vibe_basis.cli import main


CONVERGED_OUT = textwrap.dedent("""\
     CYCLE   5 TOTAL ENERGY(HF)(AU)(   5)        -2.7468175400E+02 DE-9.5E-09

     == SCF ENDED - CONVERGENCE ON ENERGY      E(AU) -2.7468175400E+02

     EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
""")

FAILED_OUT = textwrap.dedent("""\
     CYCLE 200 TOTAL ENERGY(HF)(AU)( 200)        -2.7468175000E+02 DE-1.0E-05

     == SCF ENDED - TOO MANY CYCLES      E(AU) -2.7468175000E+02

     EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
""")


def test_version_prints_and_exits_zero():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "vb" in result.output


def test_parse_converged_output(tmp_path):
    out_path = tmp_path / "converged.out"
    out_path.write_text(CONVERGED_OUT)

    runner = CliRunner()
    result = runner.invoke(main, ["parse", "crystal14", str(out_path)])
    assert result.exit_code == 0
    assert "HF" in result.output
    assert "-274.6817540000" in result.output
    assert "cycle 5" in result.output


def test_parse_failed_output_exits_non_zero(tmp_path):
    out_path = tmp_path / "failed.out"
    out_path.write_text(FAILED_OUT)

    runner = CliRunner()
    result = runner.invoke(main, ["parse", "crystal14", str(out_path)])
    assert result.exit_code == 2
    # The failure message goes to stderr — CliRunner merges by default.
    assert "FAILED" in result.output
    assert "non_converged" in result.output


def test_parse_rejects_unknown_backend(tmp_path):
    out_path = tmp_path / "any.out"
    out_path.write_text("anything")
    runner = CliRunner()
    result = runner.invoke(main, ["parse", "orca", str(out_path)])
    # Click rejects the choice before the function runs.
    assert result.exit_code != 0
    assert "orca" in result.output.lower() or "invalid" in result.output.lower()
