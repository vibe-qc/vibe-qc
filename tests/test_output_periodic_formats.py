"""Periodic-format writers — Extended XYZ, POSCAR.

Pins the contract for the writers introduced in Phase O5a:

  1. ``write_extended_xyz(stem, system)`` writes ``{stem}.xyz`` with
     the lattice encoded in the comment line via the ASE-convention
     ``Lattice="..."``  / ``Properties=...`` / ``pbc="T T T"`` tags.
     Atom block is plain XYZ (vanilla XYZ readers still load it).
  2. ``write_poscar(stem, system)`` writes ``{stem}.POSCAR`` in
     VASP-5 layout: comment + scale + 3 lattice rows + element-symbol
     line + per-species counts + ``Direct`` + fractional coords.
  3. Both round-trip via simple readers (the existing
     :func:`vibeqc.poscar.read_poscar` for the POSCAR case is left
     to integration tests; here we assert the file shape).
  4. ``OutputPlan.from_run_job_kwargs`` with ``job_kind="periodic_scf"``
     plus the periodic-format kwargs declares the expected
     ``[[plan.files]]`` rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vibeqc.output import (
    OutputPlan,
    write_extended_xyz,
    write_poscar,
)
from vibeqc.output.formats.extended_xyz import format_extended_xyz
from vibeqc.output.formats.poscar import format_poscar


# Duck-typed stand-ins so tests don't need the C++ core importable.
@dataclass
class _Atom:
    Z: int
    xyz: tuple


@dataclass
class _Sys:
    lattice: np.ndarray
    unit_cell: list


def _mgo_cubic() -> _Sys:
    """Cubic MgO: 4.21 Å lattice, Mg at (0,0,0), O at (½,½,½)."""
    a_A = 4.21
    a_bohr = a_A / 0.529177210903
    lattice = np.array([
        [a_bohr, 0.0, 0.0],
        [0.0, a_bohr, 0.0],
        [0.0, 0.0, a_bohr],
    ]).T  # columns = vectors
    atoms = [
        _Atom(12, (0.0, 0.0, 0.0)),
        _Atom(8, (a_bohr / 2, a_bohr / 2, a_bohr / 2)),
    ]
    return _Sys(lattice, atoms)


# ---------------------------------------------------------------------- #
# Extended XYZ
# ---------------------------------------------------------------------- #

def test_extended_xyz_writes_target_path(tmp_path: Path) -> None:
    target = write_extended_xyz(tmp_path / "mgo", _mgo_cubic())
    assert target == (tmp_path / "mgo").with_suffix(".xyz")
    assert target.is_file()


def test_extended_xyz_first_line_is_atom_count() -> None:
    text = format_extended_xyz(_mgo_cubic())
    assert text.splitlines()[0] == "2"


def test_extended_xyz_comment_carries_lattice() -> None:
    text = format_extended_xyz(_mgo_cubic())
    comment = text.splitlines()[1]
    assert 'Lattice="' in comment
    # The lattice scalar is 4.21 Å on each axis.
    assert "4.2100000000" in comment


def test_extended_xyz_comment_carries_pbc_and_properties() -> None:
    text = format_extended_xyz(_mgo_cubic())
    comment = text.splitlines()[1]
    assert 'pbc="T T T"' in comment
    assert "Properties=species:S:1:pos:R:3" in comment


def test_extended_xyz_energy_in_comment_when_given() -> None:
    text = format_extended_xyz(_mgo_cubic(), energy_ha=-275.123456789)
    comment = text.splitlines()[1]
    assert "energy=-275.1234567890" in comment


def test_extended_xyz_atoms_in_angstrom() -> None:
    text = format_extended_xyz(_mgo_cubic())
    lines = text.splitlines()
    # Atom 1: Mg at origin
    sym, x, y, z = lines[2].split()
    assert sym == "Mg"
    assert float(x) == pytest.approx(0.0, abs=1e-12)
    # Atom 2: O at (a/2, a/2, a/2). a=4.21 Å so a/2=2.105.
    sym, x, y, z = lines[3].split()
    assert sym == "O"
    assert float(x) == pytest.approx(2.105, rel=1e-5)


# ---------------------------------------------------------------------- #
# POSCAR
# ---------------------------------------------------------------------- #

def test_poscar_writes_target_path(tmp_path: Path) -> None:
    target = write_poscar(tmp_path / "mgo", _mgo_cubic())
    assert target == (tmp_path / "mgo").with_suffix(".POSCAR")
    assert target.is_file()


def test_poscar_structure_matches_vasp_5_layout() -> None:
    text = format_poscar(_mgo_cubic())
    lines = text.splitlines()
    assert lines[1] == "1.0"             # scale
    # Lines 2-4: lattice vectors (Å)
    a = lines[2].split()
    assert float(a[0]) == pytest.approx(4.21, rel=1e-9)
    # Line 5: element symbols
    assert lines[5].split() == ["Mg", "O"]
    # Line 6: per-species counts
    assert lines[6].split() == ["1", "1"]
    # Line 7: coords mode
    assert lines[7].strip() == "Direct"
    # Lines 8-9: fractional coords
    mg_frac = [float(c) for c in lines[8].split()]
    o_frac = [float(c) for c in lines[9].split()]
    assert mg_frac == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
    assert o_frac == pytest.approx([0.5, 0.5, 0.5], abs=1e-9)


def test_poscar_groups_species_by_first_occurrence() -> None:
    """Two Mg atoms then one O should produce the species line
    ``Mg O`` with counts ``2 1``, not ``Mg O Mg`` with ``1 1 1``."""
    a_bohr = 5.0
    sys = _Sys(
        np.eye(3) * a_bohr,
        [
            _Atom(12, (0.0, 0.0, 0.0)),
            _Atom(8,  (0.5, 0.5, 0.5)),
            _Atom(12, (0.25, 0.25, 0.25)),  # second Mg, out of order
        ],
    )
    text = format_poscar(sys)
    lines = text.splitlines()
    assert lines[5].split() == ["Mg", "O"]
    assert lines[6].split() == ["2", "1"]


def test_poscar_fractional_coords_use_inverse_lattice() -> None:
    """Off-diagonal lattices must still produce correct fractional
    coords — sanity-check with a sheared cell."""
    # A non-orthogonal lattice: small shear in xy.
    a_bohr = 4.0
    lattice = np.array([
        [a_bohr, 0.5, 0.0],
        [0.0, a_bohr, 0.0],
        [0.0, 0.0, a_bohr],
    ]).T  # columns = vectors
    # Place an atom at lattice * (0.25, 0.5, 0.0) - body-of-cell.
    target_frac = np.array([0.25, 0.5, 0.0])
    target_cart = lattice @ target_frac
    sys = _Sys(lattice, [_Atom(6, tuple(target_cart))])
    text = format_poscar(sys)
    coords_line = text.splitlines()[8]
    frac = [float(c) for c in coords_line.split()]
    assert frac == pytest.approx([0.25, 0.5, 0.0], abs=1e-12)


# ---------------------------------------------------------------------- #
# OutputPlan periodic mode
# ---------------------------------------------------------------------- #

def test_periodic_plan_declares_xyz_poscar_xsf(tmp_path: Path) -> None:
    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "mgo",
        method="RHF", basis="sto-3g", functional=None,
        job_kind="periodic_scf",
        write_poscar=True, write_xsf_structure=True,
    )
    paths = {str(f.path) for f in plan.files}
    stem = tmp_path / "mgo"
    assert str(stem.with_suffix(".xyz"))    in paths
    assert str(stem.with_suffix(".POSCAR")) in paths
    assert str(stem.with_suffix(".xsf"))    in paths


def test_periodic_plan_declares_density_xsf_separately(
    tmp_path: Path,
) -> None:
    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "mgo",
        method="RKS", basis="pob-tzvp", functional="PBE",
        job_kind="periodic_scf",
        write_density_xsf=True,
    )
    density_paths = [str(f.path) for f in plan.files
                     if f.role == "density"]
    assert (tmp_path / "mgo.xsf").as_posix() in (
        p.replace("\\", "/") for p in density_paths
    )


def test_periodic_plan_avoids_structure_density_xsf_collision(
    tmp_path: Path,
) -> None:
    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "mgo",
        method="RKS",
        basis="pob-tzvp",
        functional="PBE",
        job_kind="periodic_scf",
        write_xsf_structure=True,
        write_density_xsf=True,
    )
    xsf_files = [f for f in plan.files if f.format == "xsf"]
    assert [(f.role, f.path.name) for f in xsf_files] == [
        ("geometry", "mgo.structure.xsf"),
        ("density", "mgo.xsf"),
    ]


def test_periodic_plan_job_kind_recorded(tmp_path: Path) -> None:
    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "mgo",
        method="RHF", basis="sto-3g", functional=None,
        job_kind="periodic_scf",
    )
    assert plan.job_kind == "periodic_scf"


def test_periodic_runner_retemplates_short_lattice_density_for_output(
    monkeypatch,
) -> None:
    """A longer output template keeps known cells and zero-fills new ones."""
    import vibeqc._vibeqc_core as core
    from vibeqc.periodic_runner import _density_lattice_set_for_output

    home = SimpleNamespace(index=(0, 0, 0))
    outer = SimpleNamespace(index=(-1, 0, 0))
    density = SimpleNamespace(
        cells=[home],
        blocks=[np.array([[2.0]])],
        set_block=lambda index, block: None,
    )
    target_blocks = [np.full((1, 1), np.nan), np.full((1, 1), np.nan)]
    target = SimpleNamespace(
        nbf=1,
        cells=[outer, home],
        blocks=target_blocks,
        set_block=lambda index, block: target_blocks.__setitem__(index, block),
    )
    monkeypatch.setattr(core, "compute_overlap_lattice", lambda *args: target)
    result = SimpleNamespace(density=density)

    actual = _density_lattice_set_for_output(object(), object(), result, object())

    assert actual is target
    np.testing.assert_array_equal(target_blocks[0], np.zeros((1, 1)))
    np.testing.assert_array_equal(target_blocks[1], np.array([[2.0]]))
    np.testing.assert_array_equal(density.blocks[0], np.array([[2.0]]))


def test_periodic_runner_does_not_truncate_lattice_density_for_output(
    monkeypatch,
) -> None:
    """A shorter output template must not silently discard SCF cells."""
    import vibeqc._vibeqc_core as core
    from vibeqc.periodic_runner import _density_lattice_set_for_output

    home = SimpleNamespace(index=(0, 0, 0))
    outer = SimpleNamespace(index=(-1, 0, 0))
    density = SimpleNamespace(
        cells=[home, outer],
        blocks=[np.array([[2.0]]), np.array([[0.25]])],
        set_block=lambda index, block: None,
    )
    target_blocks = [np.zeros((1, 1))]
    target = SimpleNamespace(
        nbf=1,
        cells=[home],
        blocks=target_blocks,
        set_block=lambda index, block: target_blocks.__setitem__(index, block),
    )
    monkeypatch.setattr(core, "compute_overlap_lattice", lambda *args: target)

    with pytest.raises(ValueError, match="omits converged lattice cells"):
        _density_lattice_set_for_output(
            object(),
            object(),
            SimpleNamespace(density=density),
            object(),
        )


def _one_h_lattice_density_cells():
    import vibeqc as vq
    from vibeqc import _vibeqc_core as core

    lattice = np.eye(3) * 3.0
    system = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(1, [0.0, 0.0, 0.0])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    cells = list(core.direct_lattice_cells(system, 3.0 + 1.0e-8))
    by_key = {
        tuple(int(value) for value in np.asarray(cell.index)): cell
        for cell in cells
    }
    return system, basis, by_key


def test_exact_gamma_density_artifact_preserves_returned_cells_and_order():
    """The grid path copies the converged R set, never an overlap template."""
    from vibeqc import _vibeqc_core as core
    from vibeqc.periodic_runner import (
        _exact_lattice_density_set_for_grid_artifact,
    )

    system, basis, cells = _one_h_lattice_density_cells()
    source = core.make_lattice_matrix_set(
        int(basis.nbasis),
        [cells[(1, 0, 0)], cells[(0, 0, 0)], cells[(-1, 0, 0)]],
        [np.array([[0.125]]), np.array([[0.75]]), np.array([[0.25]])],
    )

    actual = _exact_lattice_density_set_for_grid_artifact(
        basis,
        system,
        source,
        label="restricted Gamma density",
    )

    assert actual is not source
    assert [tuple(int(v) for v in cell.index) for cell in actual.cells] == [
        (1, 0, 0),
        (0, 0, 0),
        (-1, 0, 0),
    ]
    np.testing.assert_allclose(
        np.asarray(actual.blocks),
        np.asarray([[[0.125]], [[0.75]], [[0.25]]]),
    )
    for copied, returned in zip(actual.cells, source.cells):
        np.testing.assert_array_equal(copied.index, returned.index)
        np.testing.assert_allclose(copied.r_cart, returned.r_cart, atol=0.0)

    # The exact artifact helper must not mutate the SCF result while forming
    # its validated copy.
    actual.set_block(0, np.array([[9.0]]))
    np.testing.assert_allclose(source.blocks[0], np.array([[0.125]]))


def test_exact_density_artifact_rejects_surrogate_rank2_density():
    """A rank-2 matrix cannot supply the translated blocks of an exact grid."""
    from vibeqc.periodic_runner import (
        _exact_lattice_density_set_for_grid_artifact,
    )

    system, basis, _ = _one_h_lattice_density_cells()
    with pytest.raises(ValueError, match="will not reconstruct"):
        _exact_lattice_density_set_for_grid_artifact(
            basis,
            system,
            np.array([[1.0]]),
            label="unsupported periodic density",
        )


def test_true_multik_density_artifact_rejects_per_k_matrix_list():
    """The artifact dispatcher must not inverse-fold a surrogate density."""
    from vibeqc.periodic_runner import _exact_periodic_density_grid_artifact

    system, basis, _ = _one_h_lattice_density_cells()
    result = SimpleNamespace(
        density=[
            np.array([[0.75]], dtype=np.complex128),
            np.array([[0.25]], dtype=np.complex128),
        ]
    )

    with pytest.raises(ValueError, match="refusing to reconstruct one from per-k"):
        _exact_periodic_density_grid_artifact(
            basis,
            system,
            result,
            lattice_bohr=np.asarray(system.lattice, dtype=float),
            grid_shape=(2, 2, 2),
            spacing_bohr=1.5,
            gamma_only=False,
            expected_electrons=1.0,
        )


@pytest.mark.parametrize(
    ("cells", "blocks", "message"),
    [
        (
            [SimpleNamespace(index=(0, 0, 0), r_cart=(0.0, 0.0, 0.0))],
            [],
            "1 cells but 0 blocks",
        ),
        (
            [
                SimpleNamespace(index=(0, 0, 0), r_cart=(0.0, 0.0, 0.0)),
                SimpleNamespace(index=(0, 0, 0), r_cart=(0.0, 0.0, 0.0)),
            ],
            [np.array([[1.0]]), np.array([[0.5]])],
            "duplicate lattice cell",
        ),
        (
            [SimpleNamespace(index=(1, 0, 0), r_cart=(0.0, 0.0, 0.0))],
            [np.array([[0.5]])],
            "Cartesian translation",
        ),
    ],
)
def test_exact_density_artifact_rejects_malformed_lattice_sets(
    cells,
    blocks,
    message,
):
    """Malformed cell metadata fails before a misleading grid is emitted."""
    from vibeqc.periodic_runner import (
        _exact_lattice_density_set_for_grid_artifact,
    )

    system, basis, _ = _one_h_lattice_density_cells()
    malformed = SimpleNamespace(
        nbf=int(basis.nbasis),
        cells=cells,
        blocks=blocks,
        set_block=lambda _index, _block: None,
    )
    with pytest.raises(ValueError, match=message):
        _exact_lattice_density_set_for_grid_artifact(
            basis,
            system,
            malformed,
            label="malformed periodic density",
        )


def test_exact_unrestricted_density_rejects_mismatched_spin_cell_sets():
    """Alpha and beta must describe one identical returned lattice domain."""
    from vibeqc import _vibeqc_core as core
    from vibeqc.periodic_runner import (
        _sum_exact_lattice_density_sets_for_grid_artifact,
    )

    _system, basis, cells = _one_h_lattice_density_cells()
    alpha = core.make_lattice_matrix_set(
        int(basis.nbasis),
        [cells[(0, 0, 0)], cells[(1, 0, 0)]],
        [np.array([[0.6]]), np.array([[0.1]])],
    )
    beta = core.make_lattice_matrix_set(
        int(basis.nbasis),
        [cells[(0, 0, 0)]],
        [np.array([[0.4]])],
    )

    with pytest.raises(ValueError, match="different lattice-cell keys"):
        _sum_exact_lattice_density_sets_for_grid_artifact(
            basis,
            alpha,
            beta,
            label="unrestricted density",
        )
