"""Tests for ``vibeqc.naming.report.write_iupac_name``.

This is the first writer extracted from ``runner.py`` onto the ambient
output channel. It is the proof that a module can contribute a line to
the ``.out`` file without being handed a file object.
"""

from __future__ import annotations

import io

import pytest

from vibeqc import Atom, Molecule
from vibeqc.naming.report import write_iupac_name
from vibeqc.output import OutputChannel


def _water() -> Molecule:
    return Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.98]),
            Atom(1, [0.0, -1.43, -0.98]),
        ]
    )


def test_writes_the_iupac_line_into_the_active_channel():
    buf = io.StringIO()
    with OutputChannel.to_stream(buf):
        assert write_iupac_name(_water()) is True
    assert buf.getvalue() == "  IUPAC name: water\n"


def test_line_format_matches_what_runner_used_to_emit():
    # Pinned separately from the value: two leading spaces, one trailing
    # newline. The .out format is a contract (docs/output_files.md).
    buf = io.StringIO()
    with OutputChannel.to_stream(buf):
        write_iupac_name(_water())
    line = buf.getvalue()
    assert line.startswith("  IUPAC name: ")
    assert line.endswith("\n")
    assert line.count("\n") == 1


def test_is_a_no_op_without_a_channel_and_does_not_raise(capsys):
    # The whole point of the channel: a writer is callable anywhere.
    assert write_iupac_name(_water()) is True
    assert capsys.readouterr().out == ""


def test_naming_failure_is_swallowed_and_reported_as_false(monkeypatch):
    import vibeqc_naming

    def _boom(*args, **kwargs):
        raise RuntimeError("perception failed")

    monkeypatch.setattr(vibeqc_naming, "name_from_atoms", _boom)

    buf = io.StringIO()
    with OutputChannel.to_stream(buf):
        # A job must never die because its molecule could not be named.
        assert write_iupac_name(_water()) is False
    assert buf.getvalue() == ""


def test_geometry_is_converted_from_bohr_to_angstrom(monkeypatch):
    import vibeqc_naming

    captured: list = []

    def _spy(atoms, **kwargs):
        captured.append(atoms)
        return "spied"

    monkeypatch.setattr(vibeqc_naming, "name_from_atoms", _spy)

    with OutputChannel.to_stream(io.StringIO()):
        write_iupac_name(_water())

    (atoms,) = captured
    # O sits at the origin; H at y = 1.43 bohr = 0.7567 angstrom. Passing
    # bohr straight through would name the molecule wrongly, or not at all.
    assert atoms[0] == (8, 0.0, 0.0, 0.0)
    assert atoms[1][2] == pytest.approx(1.43 * 0.529177210903)
