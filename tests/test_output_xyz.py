"""``vibeqc.output.write_xyz`` — final-geometry XYZ writer.

Pins the contract documented in
``docs/design_output_module.md § Phase O3 — .xyz``:

  1. ``write_xyz(stem, molecule)`` writes ``{stem}.xyz`` with the
     canonical 4-line-per-atom XYZ layout: count + comment + per-atom
     ``<symbol> <x> <y> <z>`` rows. Positions are converted from bohr
     (vibe-qc's internal unit) to Ångström using CODATA 2018 a₀.
  2. The default comment line is a vibe-qc provenance string.
  3. Passing ``energy_ha=`` appends ``energy=<value>`` to the comment
     line so ASE / Open Babel can recover the SCF result.
  3b. ``energy_method=`` and ``method=`` append ``energy_method=<m>``
     and ``method=<m>`` so the file names what that energy is. Without
     them a correlated job's .xyz carried its mean-field reference's
     energy under a bare ``energy=`` and was byte-identical to the
     reference job's file (#360).
  4. Out-of-range atomic numbers fall through to ``"X"`` (the XYZ-
     convention unknown-element placeholder) rather than raising —
     a finished SCF must never lose its geometry to a stray Z.
  5. ``format_xyz(...)`` returns the same content as a string.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from vibeqc.output import write_xyz
from vibeqc.output.formats.xyz import (
    BOHR_TO_ANGSTROM,
    _symbol,
    format_xyz,
)


# Duck-typed stand-ins so the test suite doesn't need the C++ core
# importable (the parent venv currently has a stale .so missing
# EEQOptions; my pure-Python writer works fine against any object
# that exposes ``.atoms`` / ``.Z`` / ``.xyz``).
@dataclass
class _Atom:
    Z: int
    xyz: tuple[float, float, float]


@dataclass
class _Mol:
    atoms: list[_Atom]


def _h2o_bohr() -> _Mol:
    # Geometry in bohr — exactly the H₂O reference used in the
    # docstring examples. The actual numbers don't matter for the
    # writer-contract tests; they're just deterministic.
    return _Mol([
        _Atom(8, (0.0, 0.0, 0.224)),
        _Atom(1, (0.0,  1.430, -0.896)),
        _Atom(1, (0.0, -1.430, -0.896)),
    ])


# ---------------------------------------------------------------------- #
# Layout
# ---------------------------------------------------------------------- #

def test_write_xyz_target_is_stem_dot_xyz(tmp_path: Path) -> None:
    stem = tmp_path / "h2o"
    target = write_xyz(stem, _h2o_bohr())
    assert target == stem.with_suffix(".xyz")
    assert target.is_file()


def test_xyz_first_line_is_atom_count(tmp_path: Path) -> None:
    text = format_xyz(_h2o_bohr())
    assert text.splitlines()[0] == "3"


def test_xyz_per_atom_rows_use_symbols(tmp_path: Path) -> None:
    text = format_xyz(_h2o_bohr())
    lines = text.splitlines()
    # comment + 3 atom rows
    assert len(lines) == 5
    assert lines[2].split()[0] == "O"
    assert lines[3].split()[0] == "H"
    assert lines[4].split()[0] == "H"


def test_xyz_positions_in_angstrom(tmp_path: Path) -> None:
    mol = _Mol([_Atom(8, (1.0, 0.0, 0.0))])
    text = format_xyz(mol)
    sym, x, y, z = text.splitlines()[2].split()
    assert sym == "O"
    assert float(x) == pytest.approx(BOHR_TO_ANGSTROM, abs=5e-11)
    assert float(y) == 0.0
    assert float(z) == 0.0


# ---------------------------------------------------------------------- #
# Comment line / energy embedding
# ---------------------------------------------------------------------- #

def test_default_comment_mentions_vibe_qc(tmp_path: Path) -> None:
    text = format_xyz(_h2o_bohr())
    assert "vibe-qc" in text.splitlines()[1].lower()


def test_custom_comment_overrides_default(tmp_path: Path) -> None:
    text = format_xyz(_h2o_bohr(), comment="manually authored geometry")
    assert text.splitlines()[1] == "manually authored geometry"


def test_energy_appended_to_comment(tmp_path: Path) -> None:
    text = format_xyz(_h2o_bohr(), energy_ha=-76.0098765432)
    comment = text.splitlines()[1]
    assert "energy=" in comment
    # The numeric value is rendered with 10 decimals.
    assert "-76.0098765432" in comment


def test_energy_appended_to_custom_comment(tmp_path: Path) -> None:
    text = format_xyz(
        _h2o_bohr(),
        comment="My H₂O run",
        energy_ha=-76.123,
    )
    comment = text.splitlines()[1]
    assert comment.startswith("My H₂O run")
    assert "energy=-76.1230000000" in comment


def test_energy_method_names_what_the_energy_is() -> None:
    """The tag that closes #360: ``energy=`` alone said nothing about
    which method produced it."""
    text = format_xyz(
        _h2o_bohr(), energy_ha=-76.123, energy_method="rhf", method="mp2"
    )
    comment = text.splitlines()[1]
    assert "energy=-76.1230000000" in comment
    assert "energy_method=rhf" in comment
    assert "method=mp2" in comment


def test_method_tags_distinguish_a_correlated_job_from_its_reference() -> None:
    """Same geometry, same energy, different job: the files must differ.

    An MP2 job's .xyz was byte-identical to an RHF job's, because the
    energy really is the RHF one. Naming the energy's method alone does
    not fix that -- both would say ``energy_method=rhf`` -- so the job's
    own method is tagged too.
    """
    mp2 = format_xyz(
        _h2o_bohr(), energy_ha=-76.123, energy_method="rhf", method="mp2"
    )
    rhf = format_xyz(
        _h2o_bohr(), energy_ha=-76.123, energy_method="rhf", method="rhf"
    )
    assert mp2 != rhf
    # ... and they differ only in the method tag, not in the geometry.
    assert mp2.splitlines()[2:] == rhf.splitlines()[2:]


def test_energy_method_is_omitted_without_an_energy() -> None:
    """No number, nothing to label."""
    comment = format_xyz(_h2o_bohr(), energy_method="rhf").splitlines()[1]
    assert "energy" not in comment


def test_method_tag_normalises_a_custom_comment_without_an_energy() -> None:
    """The trailing-period strip used to live inside the energy branch,
    so a method tag with no energy would have appended to unstripped
    text."""
    comment = format_xyz(
        _h2o_bohr(), comment="My run.", method="mp2"
    ).splitlines()[1]
    assert comment == "My run method=mp2"


def test_untagged_output_is_unchanged() -> None:
    """Callers that pass neither tag keep the exact legacy comment, so
    this adds information without moving anyone's existing files."""
    comment = format_xyz(_h2o_bohr(), energy_ha=-76.123).splitlines()[1]
    assert comment.endswith("energy=-76.1230000000")
    assert "method" not in comment


# ---------------------------------------------------------------------- #
# Symbol resolution
# ---------------------------------------------------------------------- #

def test_symbol_table_covers_h_through_rn() -> None:
    # def2-* covers H–Rn — every Z up to 86 must resolve.
    for z in range(1, 87):
        sym = _symbol(z)
        assert sym not in ("X", ""), \
            f"Z={z} fell through to placeholder: {sym!r}"


def test_unknown_z_falls_through_to_x() -> None:
    assert _symbol(200) == "X"
    assert _symbol(-1) == "X"


def test_ghost_atom_z_zero_renders_as_x(tmp_path: Path) -> None:
    """Z=0 is the conventional "ghost atom" placeholder — used by
    counterpoise-style basis-set superposition workarounds. We render
    it as ``X`` so the .xyz reads cleanly and doesn't crash."""
    mol = _Mol([_Atom(0, (0.0, 0.0, 0.0))])
    text = format_xyz(mol)
    assert text.splitlines()[2].split()[0] == "X"


# ---------------------------------------------------------------------- #
# Trailing newline / round-trip-friendliness
# ---------------------------------------------------------------------- #

def test_file_ends_with_newline(tmp_path: Path) -> None:
    """ASE's xyz reader, OpenBabel, and friends all parse files without
    a trailing newline, but emitting one is the universal convention."""
    stem = tmp_path / "h2o"
    write_xyz(stem, _h2o_bohr())
    body = stem.with_suffix(".xyz").read_text(encoding="utf-8")
    assert body.endswith("\n")
