"""CIF writer — Phase D2.

Pins the contract for :func:`vibeqc.output.write_cif`:

  1. ``write_cif(stem, system)`` writes ``{stem}.cif``.
  2. The file is a complete CIF 1.1 document with ``data_`` block
     header, cell parameters, P 1 symmetry stanza, identity
     symmetry op, and an ``_atom_site_*`` loop.
  3. Cell parameters are correctly derived from the lattice (column
     vectors → ``a``/``b``/``c`` lengths in Å, ``α``/``β``/``γ`` in
     degrees, ``_cell_volume`` in Å³).
  4. Fractional coordinates round-trip through inv(lattice) @ cart.
  5. Per-element site labels are 1-indexed (``Mg1`` / ``O1`` /
     ``Mg2`` …) so multiple atoms of the same element are uniquely
     labelled.
  6. ``OutputPlan.from_run_job_kwargs(..., write_cif=True)`` declares
     the matching ``[[plan.files]]`` row.
  7. ``default_dispatcher()`` exposes the ``(geometry, cif)`` pair.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from vibeqc.output import (
    OutputPlan,
    default_dispatcher,
    write_cif,
)
from vibeqc.output.formats.cif import format_cif


@dataclass
class _Atom:
    Z: int
    xyz: tuple


@dataclass
class _Sys:
    lattice: np.ndarray
    unit_cell: list


def _mgo_cubic() -> _Sys:
    """Rocksalt-style cubic MgO, a=4.21 Å."""
    a_A = 4.21
    a_bohr = a_A / 0.529177210903
    lattice = np.diag([a_bohr, a_bohr, a_bohr])
    atoms = [
        _Atom(12, (0.0, 0.0, 0.0)),
        _Atom(8, (a_bohr / 2, a_bohr / 2, a_bohr / 2)),
    ]
    return _Sys(lattice, atoms)


# ---------------------------------------------------------------------- #
# Layout
# ---------------------------------------------------------------------- #

def test_write_cif_returns_stem_dot_cif(tmp_path: Path) -> None:
    target = write_cif(tmp_path / "mgo", _mgo_cubic())
    assert target == (tmp_path / "mgo").with_suffix(".cif")
    assert target.is_file()


def test_cif_starts_with_data_block() -> None:
    text = format_cif(_mgo_cubic())
    # Find the data_ line after the comment header.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()
             and not ln.lstrip().startswith("#")]
    assert lines[0].startswith("data_"), f"first non-comment: {lines[0]!r}"


def test_cif_contains_cell_parameter_keys() -> None:
    text = format_cif(_mgo_cubic())
    for key in (
        "_cell_length_a",
        "_cell_length_b",
        "_cell_length_c",
        "_cell_angle_alpha",
        "_cell_angle_beta",
        "_cell_angle_gamma",
        "_cell_volume",
    ):
        assert key in text, f"missing {key!r}"


def test_cif_p1_symmetry() -> None:
    text = format_cif(_mgo_cubic())
    assert "_symmetry_space_group_name_H-M" in text
    assert "'P 1'" in text
    assert "_symmetry_Int_Tables_number" in text
    assert "1 'x, y, z'" in text  # identity symmetry op


def test_cif_atom_site_loop() -> None:
    text = format_cif(_mgo_cubic())
    for key in (
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
    ):
        assert key in text, f"missing {key!r}"


# ---------------------------------------------------------------------- #
# Cell parameters
# ---------------------------------------------------------------------- #

def test_cubic_cell_parameters_correct() -> None:
    text = format_cif(_mgo_cubic())
    # MgO at a=4.21 Å — extract the cell-length lines.
    lines = {}
    for ln in text.splitlines():
        for key in ("_cell_length_a", "_cell_length_b",
                     "_cell_length_c", "_cell_angle_alpha",
                     "_cell_angle_beta", "_cell_angle_gamma",
                     "_cell_volume"):
            if ln.strip().startswith(key):
                lines[key] = float(ln.split(None, 1)[1])
    assert lines["_cell_length_a"] == pytest.approx(4.21, rel=1e-9)
    assert lines["_cell_length_b"] == pytest.approx(4.21, rel=1e-9)
    assert lines["_cell_length_c"] == pytest.approx(4.21, rel=1e-9)
    assert lines["_cell_angle_alpha"] == pytest.approx(90.0, abs=1e-6)
    assert lines["_cell_angle_beta"] == pytest.approx(90.0, abs=1e-6)
    assert lines["_cell_angle_gamma"] == pytest.approx(90.0, abs=1e-6)
    assert lines["_cell_volume"] == pytest.approx(4.21 ** 3, rel=1e-9)


def test_non_orthogonal_cell_angles() -> None:
    """A monoclinic-style cell with β = 100° should report it."""
    a_A, b_A, c_A, beta_deg = 5.0, 6.0, 7.0, 100.0
    a_bohr = a_A / 0.529177210903
    b_bohr = b_A / 0.529177210903
    c_bohr = c_A / 0.529177210903
    # Construct lattice with α=γ=90° and β=100°.
    lattice = np.array([
        [a_bohr, 0.0,
         c_bohr * math.cos(math.radians(beta_deg))],
        [0.0, b_bohr, 0.0],
        [0.0, 0.0,
         c_bohr * math.sin(math.radians(beta_deg))],
    ])  # columns = lattice vectors
    sys = _Sys(lattice, [_Atom(6, (0.0, 0.0, 0.0))])
    text = format_cif(sys)
    for ln in text.splitlines():
        if ln.strip().startswith("_cell_angle_beta"):
            beta_out = float(ln.split(None, 1)[1])
            break
    else:
        raise AssertionError("no _cell_angle_beta found")
    assert beta_out == pytest.approx(100.0, abs=1e-6)


# ---------------------------------------------------------------------- #
# Fractional coords + labels
# ---------------------------------------------------------------------- #

def test_fractional_coords_at_half_for_o_atom() -> None:
    text = format_cif(_mgo_cubic())
    # The O atom is at (½, ½, ½) in fractional coords.
    lines = [ln for ln in text.splitlines() if ln.startswith("O1")]
    assert len(lines) == 1
    parts = lines[0].split()
    # parts = ['O1', 'O', '0.5000000000', '0.5000000000', '0.5000000000', '1.0']
    assert parts[0] == "O1"
    assert parts[1] == "O"
    assert float(parts[2]) == pytest.approx(0.5, abs=1e-9)
    assert float(parts[3]) == pytest.approx(0.5, abs=1e-9)
    assert float(parts[4]) == pytest.approx(0.5, abs=1e-9)
    assert parts[5] == "1.0"


def test_per_element_labels_are_1_indexed() -> None:
    """Two Mg + two O atoms should produce Mg1 / Mg2 / O1 / O2 labels."""
    a_bohr = 5.0
    sys = _Sys(
        np.eye(3) * a_bohr,
        [
            _Atom(12, (0.0, 0.0, 0.0)),
            _Atom(12, (a_bohr / 2, a_bohr / 2, 0.0)),
            _Atom(8, (a_bohr / 2, 0.0, 0.0)),
            _Atom(8, (0.0, a_bohr / 2, 0.0)),
        ],
    )
    text = format_cif(sys)
    labels = [
        ln.split()[0]
        for ln in text.splitlines()
        if ln and ln.split() and ln.split()[0] in ("Mg1", "Mg2", "O1", "O2")
    ]
    assert labels == ["Mg1", "Mg2", "O1", "O2"]


# ---------------------------------------------------------------------- #
# OutputPlan integration
# ---------------------------------------------------------------------- #

def test_plan_declares_cif_when_kwarg_true(tmp_path: Path) -> None:
    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "mgo",
        method="RKS", basis="pob-tzvp", functional="PBE",
        job_kind="periodic_scf",
        write_cif=True,
    )
    paths = {str(f.path) for f in plan.files}
    assert str(tmp_path / "mgo.cif") in paths
    # And the right role / format.
    cif_files = [f for f in plan.files if f.format == "cif"]
    assert len(cif_files) == 1
    assert cif_files[0].role == "geometry"


def test_plan_no_cif_when_kwarg_false(tmp_path: Path) -> None:
    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "mgo",
        method="RHF", basis="sto-3g", functional=None,
    )
    cif_files = [f for f in plan.files if f.format == "cif"]
    assert cif_files == []


# ---------------------------------------------------------------------- #
# Dispatcher integration
# ---------------------------------------------------------------------- #

def test_default_dispatcher_has_cif_writer() -> None:
    d = default_dispatcher()
    fn = d.get_writer("geometry", "cif")
    assert fn is not None
    assert callable(fn)


def test_default_dispatcher_emits_cif_end_to_end(tmp_path: Path) -> None:
    """End-to-end: dispatch a (geometry, cif) planned file through
    the default dispatcher; verify the file is written + parseable."""
    from vibeqc.output import PlannedFile
    d = default_dispatcher()
    pf = PlannedFile(
        role="geometry", path=tmp_path / "mgo.cif",
        format="cif",  # type: ignore[arg-type]
        always=True, description="",
    )
    out = d.dispatch_planned_file(
        pf, stem=tmp_path / "mgo", system=_mgo_cubic(),
    )
    assert out == tmp_path / "mgo.cif"
    body = out.read_text()
    assert "data_vibeqc" in body
    assert "_cell_length_a" in body
    assert "Mg1" in body
    assert "O1" in body
