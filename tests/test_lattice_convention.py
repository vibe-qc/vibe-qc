"""Lattice convention lock — vibe-qc stores lattice vectors as COLUMNS.

``PeriodicSystem.lattice[:, i]`` is the i-th lattice vector ``a_i`` (NOT
row ``i``); ``reciprocal_lattice()[:, i]`` is ``b_i``. This is the
convention the C++ core uses for the real-space image sum and the k-point
Cartesian map, and that every periodic-structure writer (XSF / CIF /
POSCAR / QVF) and the GPW plane-wave grid follow.

This file exists to make that convention *impossible to misread*. Several
tests below use a deliberately **non-orthogonal, non-transpose-symmetric**
cell (``lattice != lattice.T``), so a row/column flip in either the core
or a writer turns them red. ``test_fixture_distinguishes_rows_from_columns``
guards against the cell silently becoming symmetric — the exact trap that
let a 2026-05-31 audit misread the convention.

Decisive proof of the convention (asserted below):

* ``direct_lattice_cells(sys)`` returns ``cell.r_cart == lattice @ index``
  — the (1, 0, 0) image shift is **column 0**, not row 0. This is the
  lattice sum the periodic SCF actually evaluates.
* ``KPoints`` gives ``kpoints_cart == reciprocal_lattice() @ frac``.

History: a 2026-05-31 audit briefly concluded the opposite ("lattice is
ROWS; the writers emit the transpose") by feeding a *row-major* lattice
into the column-major API and reading the result back — a circular test
whose two "proofs" hold under both readings. ``2*pi*(L^-1).T`` is dual to
``L`` whether you read both as rows or both as columns, and a hexagonal
cell has ``|a1| == |a2|`` under whichever axis you happened to enter the
vectors on. Do **not** "fix" the writers to rows; that silently corrupts
output for non-orthogonal cells. See the ``lattice-convention-columns``
chat memory and ``tests/test_periodic_lattice_families.py``.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
import vibeqc as vq
from vibeqc.output.formats.cif import format_cif
from vibeqc.output.formats.poscar import format_poscar

_BOHR_TO_ANG = 0.529177210903


def _skew_cell():
    """A non-orthogonal cell whose lattice matrix is NOT symmetric, so the
    column reading (correct) and the row reading (wrong) give different
    vectors. Columns are ``a1, a2, a3`` per the vibe-qc convention."""
    lattice = np.column_stack([
        [8.0, 2.0, 0.0],   # a1 — sheared into y
        [0.0, 8.0, 0.0],   # a2
        [0.0, 0.0, 9.0],   # a3
    ])
    sys = vq.PeriodicSystem(3, lattice, [vq.Atom(1, [0.0, 0.0, 0.0])])
    return sys, np.asarray(sys.lattice, dtype=float)


def test_fixture_distinguishes_rows_from_columns():
    """Guard: the fixture must tell columns from rows, else every assertion
    in this file is vacuous (the trap that hid the convention)."""
    _, L = _skew_cell()
    assert not np.allclose(L, L.T), "skew cell must be non-symmetric"


def test_direct_lattice_cells_uses_columns():
    """The core image sum: ``r_cart(index) == lattice @ index`` so the
    lattice vectors are the **columns**, ``a_i = lattice[:, i]``."""
    sys, L = _skew_cell()
    by_idx = {
        tuple(int(x) for x in np.asarray(c.index).reshape(3)):
            np.asarray(c.r_cart, dtype=float)
        for c in vq.direct_lattice_cells(sys, 12.0)
    }
    shift_100 = by_idx[(1, 0, 0)]
    np.testing.assert_allclose(shift_100, L[:, 0], atol=1e-12)         # column 0
    np.testing.assert_allclose(shift_100, L @ np.array([1.0, 0.0, 0.0]), atol=1e-12)
    assert not np.allclose(shift_100, L[0, :]), "must NOT be row 0"


def test_reciprocal_columns_are_dual_to_lattice_columns():
    """``a_i . b_j = 2 pi delta_ij`` with ``a_i = L[:, i]`` and
    ``b_j = B[:, j]`` => ``L.T @ B == 2 pi I``."""
    sys, L = _skew_cell()
    B = np.asarray(sys.reciprocal_lattice(), dtype=float)
    np.testing.assert_allclose(L.T @ B, 2.0 * math.pi * np.eye(3), atol=1e-12)


def test_kpoints_cart_is_reciprocal_columns_at_frac():
    """``k_cart == B @ k_frac`` (column reading of the reciprocal lattice)."""
    sys, _ = _skew_cell()
    kp = vq.KPoints.gamma_centred(sys, (2, 1, 1))
    B = np.asarray(sys.reciprocal_lattice(), dtype=float)
    frac = np.asarray(kp.kpoints_frac, dtype=float)
    np.testing.assert_allclose(kp.kpoints_cart, (B @ frac.T).T, atol=1e-12)


def test_xsf_primvec_lines_are_lattice_columns(tmp_path):
    """Each XSF ``PRIMVEC`` line is one lattice vector = a column (in A).
    Fails if the writer is flipped to emit rows on a skew cell."""
    sys, L = _skew_cell()
    p = vq.write_xsf_structure(tmp_path / "s.xsf", sys)
    lines = p.read_text().splitlines()
    primvec = np.array([[float(x) for x in lines[i].split()] for i in (2, 3, 4)])
    np.testing.assert_allclose(primvec, (L * _BOHR_TO_ANG).T, atol=1e-8)


def test_poscar_lattice_lines_are_columns():
    """POSCAR writes one lattice vector per line = a column (in A)."""
    sys, L = _skew_cell()
    lines = format_poscar(sys).splitlines()
    latt = np.array([[float(x) for x in lines[i].split()] for i in (2, 3, 4)])
    np.testing.assert_allclose(latt, (L * _BOHR_TO_ANG).T, atol=1e-8)


def test_cif_recovers_hexagonal_cell_parameters():
    """A column-built hexagonal cell must come back as a == b, gamma = 120 deg.
    A row/column flip turns this into a non-hexagonal metric."""
    a, c = 5.0, 12.0
    lattice = np.column_stack([
        [a, 0.0, 0.0],
        [-a / 2.0, a * math.sqrt(3.0) / 2.0, 0.0],
        [0.0, 0.0, c],
    ])
    sys = vq.PeriodicSystem(3, lattice, [vq.Atom(13, [0.0, 0.0, 0.0])])
    vals = {}
    for ln in format_cif(sys).splitlines():
        for key in ("_cell_length_a", "_cell_length_b", "_cell_angle_gamma"):
            if ln.strip().startswith(key):
                vals[key] = float(ln.split()[1])
    assert vals["_cell_length_a"] == pytest.approx(vals["_cell_length_b"], rel=1e-9)
    assert vals["_cell_angle_gamma"] == pytest.approx(120.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Issue #445: the row/column orientation was silently accepted either way.
# A transposed lattice is a *different valid cell*, so nothing geometric can
# refuse it in general; the fix is a declared orientation (``lattice_vectors=``)
# plus orientation-SENSITIVE cell parameters where the volume is blind.
# ---------------------------------------------------------------------------


def _hbn_vectors_bohr():
    """Monolayer h-BN: a = 2.504 Angstrom, gamma = 60 deg, B-N 1.446 Angstrom
    (the literature values quoted in #445). Returned as the three vectors."""
    a = 2.504 / _BOHR_TO_ANG
    a1 = np.array([a, 0.0, 0.0])
    a2 = np.array([a / 2.0, a * math.sqrt(3.0) / 2.0, 0.0])
    a3 = np.array([0.0, 0.0, 15.0 / _BOHR_TO_ANG])
    atoms = [vq.Atom(5, [0.0, 0.0, 0.0]), vq.Atom(7, list((a1 + a2) / 3.0))]
    return (a1, a2, a3), atoms


def test_lattice_vectors_keyword_stacks_columns():
    """``PeriodicSystem(dim, lattice_vectors=[a1, a2, a3], ...)`` puts a_i in
    column i -- identical to the column-form matrix and different from the
    row-fed one on a non-symmetric cell."""
    (a1, a2, a3), atoms = _hbn_vectors_bohr()
    by_vectors = vq.PeriodicSystem(3, lattice_vectors=[a1, a2, a3], unit_cell=atoms)
    by_columns = vq.PeriodicSystem(3, np.column_stack([a1, a2, a3]), atoms)
    by_rows = vq.PeriodicSystem(3, np.vstack([a1, a2, a3]), atoms)
    Lv = np.asarray(by_vectors.lattice, dtype=float)
    assert np.allclose(Lv[:, 0], a1) and np.allclose(Lv[:, 1], a2)
    assert np.allclose(Lv, np.asarray(by_columns.lattice, dtype=float))
    assert not np.allclose(Lv, np.asarray(by_rows.lattice, dtype=float))
    assert np.allclose(Lv, vq.lattice_from_vectors(a1, a2, a3))
    # The keyword form keeps the rest of the constructor contract.
    charged = vq.PeriodicSystem(
        3, lattice_vectors=[a1, a2, a3], unit_cell=atoms, charge=1,
        multiplicity=2,
    )
    assert charged.charge == 1 and charged.multiplicity == 2


@pytest.mark.parametrize("n_vectors", [2, 4])
def test_lattice_vectors_keyword_refuses_wrong_count(n_vectors):
    (a1, a2, a3), atoms = _hbn_vectors_bohr()
    vecs = [a1, a2, a3, a1][:n_vectors]
    with pytest.raises((ValueError, TypeError), match="exactly three"):
        vq.PeriodicSystem(3, lattice_vectors=vecs, unit_cell=atoms)


def test_lattice_vectors_keyword_reads_a_2d_array_as_a_sequence_of_vectors():
    """``lattice_vectors=`` is a *sequence of vectors*, so a 2-D array given
    there is iterated row by row -- the row-major ``[[a1], [a2], [a3]]``
    literal every textbook writes lands in the engine's columns. That is the
    whole point of the keyword: the caller names vectors, never a matrix
    orientation. Pinned so the reading cannot drift."""
    (a1, a2, a3), atoms = _hbn_vectors_bohr()
    system = vq.PeriodicSystem(
        3, lattice_vectors=np.vstack([a1, a2, a3]), unit_cell=atoms
    )
    L = np.asarray(system.lattice, dtype=float)
    assert np.allclose(L, np.column_stack([a1, a2, a3]))
    assert np.allclose(L[:, 1], a2)


def test_cell_parameters_see_the_transpose_where_volume_cannot():
    """The #445 blind spot, pinned: det(L) == det(L.T) so the volume of a
    transposed lattice is bit-identical, while lengths / angles / the nearest
    interatomic distance move. On h-BN the column reading reproduces the
    literature cell and the row reading does not."""
    (a1, a2, a3), atoms = _hbn_vectors_bohr()
    L = vq.lattice_from_vectors(a1, a2, a3)
    cols = vq.cell_parameters(L)
    rows = vq.cell_parameters(L.T)
    assert cols.volume == pytest.approx(rows.volume, rel=1e-12)
    assert cols.a == pytest.approx(2.504 / _BOHR_TO_ANG, rel=1e-12)
    assert cols.b == pytest.approx(2.504 / _BOHR_TO_ANG, rel=1e-12)
    assert cols.gamma == pytest.approx(60.0, abs=1e-9)
    assert cols.alpha == pytest.approx(90.0, abs=1e-9)
    assert not (
        np.allclose(cols.lengths, rows.lengths) and np.allclose(cols.angles, rows.angles)
    ), "a non-symmetric lattice must give different parameters when transposed"
    system_cols = vq.PeriodicSystem(3, lattice_vectors=[a1, a2, a3], unit_cell=atoms)
    system_rows = vq.PeriodicSystem(3, L.T, atoms)
    nn_cols = vq.nearest_neighbour_distance(system_cols)
    nn_rows = vq.nearest_neighbour_distance(system_rows)
    assert nn_cols.distance_angstrom == pytest.approx(1.4457, abs=2e-4)
    assert {nn_cols.atom_i, nn_cols.atom_j} == {0, 1}
    assert nn_rows.distance_angstrom != pytest.approx(nn_cols.distance_angstrom, abs=1e-3)
    # The system accessor reads the same lattice as the bare matrix.
    assert vq.cell_parameters(system_cols) == cols


def test_cell_parameters_symmetric_lattice_is_orientation_invariant():
    """Cubic / FCC lattices are symmetric matrices, which is exactly why the
    three #445 instances all hid on low-symmetry cells."""
    h = 3.979
    fcc = np.array([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]])
    assert vq.cell_parameters(fcc) == vq.cell_parameters(fcc.T)
    assert vq.cell_parameters(fcc).alpha == pytest.approx(60.0, abs=1e-9)


def test_nearest_neighbour_distance_uses_images_and_periodic_axes_only():
    """One atom per cell reports its own nearest image; a 2-D cell must not
    scan images along the synthesized third axis."""
    sys_bulk = vq.PeriodicSystem(3, np.diag([4.0, 5.0, 6.0]), [vq.Atom(1, [0, 0, 0])])
    nn = vq.nearest_neighbour_distance(sys_bulk)
    assert nn.distance_bohr == pytest.approx(4.0)
    assert nn.image in ((1, 0, 0), (-1, 0, 0))
    sys_2d = vq.PeriodicSystem(2, np.diag([4.0, 5.0, 1.0]), [vq.Atom(1, [0, 0, 0])])
    nn2 = vq.nearest_neighbour_distance(sys_2d)
    assert nn2.distance_bohr == pytest.approx(4.0)
    assert nn2.image[2] == 0


def test_periodic_out_reports_orientation_sensitive_cell_parameters(tmp_path):
    """The .out carries |a_i|, angles and the nearest pair (#445 ask 3): a
    row-fed h-BN cell prints different numbers from the column-fed one even
    though the volume line is identical."""
    from vibeqc.periodic_runner import _system_summary

    (a1, a2, a3), atoms = _hbn_vectors_bohr()
    L = vq.lattice_from_vectors(a1, a2, a3)
    text_cols = _system_summary(vq.PeriodicSystem(2, L, atoms))
    text_rows = _system_summary(vq.PeriodicSystem(2, L.T, atoms))
    assert "Cell parameters" in text_cols
    assert "gamma = 60.0000 deg" in text_cols
    assert "nearest pair   = 2.7319" in text_cols  # 1.4457 Angstrom B-N
    assert "1.4457 Angstrom" in text_cols
    assert "gamma = 60.0000 deg" not in text_rows
    vol_line = [ln for ln in text_cols.splitlines() if "periodic area" in ln]
    assert vol_line and vol_line[0] in text_rows.splitlines()


# ---------------------------------------------------------------------------
# GitLab #128 -- the nearest-neighbour check truncated its own image search.
#
# The 2026-09-04 VERIFY_FAILED rejected the original fix on this: the helper
# that the issue offers as the reusable reviewer check, and that the periodic
# ".out" prints, scanned a fixed +-1 shell over the basis exactly as supplied.
# No fixed shell is sufficient for an arbitrary full-rank basis -- the shortest
# translation of a sheared cell needs coefficients larger than one -- so on the
# verifier's own counterexample it returned a distance 2.24x too large, and
# every ".out" for such a cell printed that. The verifier asked for a provably
# sufficient search rather than a wider shell, which is what Minkowski
# reduction gives: in a reduced basis the closest image is within one cell of
# the rounded fractional coordinate for dimension <= 3.
# ---------------------------------------------------------------------------


def _sheared_2d_cell():
    """The verifier's counterexample: a1=(4,0,0), a2=(7,1,0) as COLUMNS.

    Shortest translation is -2*a1 + a2 = (-1, 1, 0), length sqrt(2); a +-1
    shell can only reach (-1, 1, 0) in coefficients, i.e. (3, 1, 0), sqrt(10).
    """
    L = np.array([[4.0, 7.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 20.0]])
    return L, vq.PeriodicSystem(2, L, [vq.Atom(1, [0.0, 0.0, 0.0])])


def test_nearest_neighbour_is_exact_on_a_non_reduced_cell():
    """The default call must find -2*a1 + a2, not the best +-1 image.

    Pre-fix: 3.1622776602 bohr at image (-1, 1, 0); ``image_range=2`` was
    needed to get the true sqrt(2), which is what proved the truncation.
    """
    L, system = _sheared_2d_cell()
    nn = vq.nearest_neighbour_distance(system)
    assert nn.distance_bohr == pytest.approx(math.sqrt(2.0), abs=1e-12)
    # The image is reported in the caller's own basis and must reproduce the
    # distance there -- it must not leak the internally reduced basis.
    shift = L @ np.asarray(nn.image, dtype=float)
    assert float(np.linalg.norm(shift)) == pytest.approx(math.sqrt(2.0), abs=1e-12)
    assert nn.image[2] == 0, "a 2-D cell must not translate along the third axis"


def test_nearest_neighbour_image_range_cannot_narrow_the_search():
    """``image_range`` is advisory: honouring a narrowing request is the bug.

    Every value, including the old default of 1, must give the exact answer.
    """
    _, system = _sheared_2d_cell()
    exact = math.sqrt(2.0)
    for image_range in (0, 1, 2, 3):
        nn = vq.nearest_neighbour_distance(system, image_range=image_range)
        assert nn.distance_bohr == pytest.approx(exact, abs=1e-12), (
            f"image_range={image_range} returned {nn.distance_bohr}"
        )


def test_nearest_neighbour_is_invariant_under_a_change_of_basis():
    """The same crystal written in two valid bases is the same crystal.

    ``a2 -> a2 + 2*a1`` is unimodular, so it describes an identical lattice.
    The reduced writing is within a +-1 shell and the skewed one is not, so
    before the fix the two bases reported different distances for the same
    crystal -- which is how the defect stayed invisible wherever cells happen
    to be given in reduced form.
    """
    a1 = np.array([4.0, 0.0, 0.0])
    a2 = np.array([-1.0, 1.0, 0.0])
    a3 = np.array([0.0, 0.0, 20.0])
    atoms = [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [1.5, 0.25, 0.0])]
    reduced = vq.PeriodicSystem(2, vq.lattice_from_vectors(a1, a2, a3), atoms)
    skewed = vq.PeriodicSystem(
        2, vq.lattice_from_vectors(a1, a2 + 2.0 * a1, a3), atoms
    )
    d_reduced = vq.nearest_neighbour_distance(reduced).distance_bohr
    d_skewed = vq.nearest_neighbour_distance(skewed).distance_bohr
    assert d_skewed == pytest.approx(d_reduced, abs=1e-12), (
        f"the same crystal gave {d_reduced} in a reduced basis and "
        f"{d_skewed} in a skewed one"
    )


def test_nearest_neighbour_is_exact_on_a_sheared_3d_cell():
    """Three dimensions, non-reduced basis, two atoms.

    Checked against an exhaustive +-6 search over the same basis.
    """
    # a1 = (4,0,0), a2 = (7,1,0), a3 = (0,0,5) as COLUMNS: the shortest
    # translation is -2*a1 + a2 = (-1,1,0), coefficients outside any +-1
    # shell.  The second atom sits just off that image of the first, so the
    # true closest contact is an image pair rather than an intra-cell one.
    L = np.array([[4.0, 7.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 5.0]])
    atoms = [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [-0.9, 1.0, 0.1])]
    system = vq.PeriodicSystem(3, L, atoms)
    nn = vq.nearest_neighbour_distance(system)

    positions = [np.asarray(a.xyz, dtype=float) for a in atoms]
    reference = min(
        float(np.linalg.norm(positions[j] + L @ np.asarray(n, float) - positions[i]))
        for n in itertools.product(range(-6, 7), repeat=3)
        for i in range(2)
        for j in range(i, 2)
        if not (i == j and n == (0, 0, 0))
    )
    assert nn.distance_bohr == pytest.approx(reference, abs=1e-12)
    shift = L @ np.asarray(nn.image, dtype=float)
    assert float(
        np.linalg.norm(positions[nn.atom_j] + shift - positions[nn.atom_i])
    ) == pytest.approx(nn.distance_bohr, abs=1e-12)


def test_periodic_out_reports_the_exact_nearest_pair(tmp_path):
    """The ".out" prevention mechanism must carry the true distance.

    Pre-fix this line read ``3.162278 bohr (1.6734 Angstrom)`` for this cell,
    a factor 2.24 too large -- the reusable literature check the issue asks
    for, printing a number a reviewer could not compare to anything.
    """
    from vibeqc.periodic_runner import _system_summary

    _, system = _sheared_2d_cell()
    line = [
        ln for ln in _system_summary(system).splitlines() if "nearest pair" in ln
    ]
    assert line, "the .out must carry a nearest-pair line"
    assert "1.414214 bohr" in line[0], line[0]
    assert "0.7484 Angstrom" in line[0], line[0]


# ---------------------------------------------------------------------------
# GitLab #128 ask 1, second half: a DECLARED crystal system refuses a lattice
# that contradicts it.
#
# Nothing geometric can reject a transpose on its own -- L and L.T are both
# valid lattices, and det L == det L.T so the volume is blind. Only a
# declaration can. These pin what the declaration does catch (a transposed
# hexagonal or rhombohedral cell, which is every instance the issue records)
# and, just as importantly, what it does not, so nobody mistakes a cubic
# declaration for an orientation check.
# ---------------------------------------------------------------------------


def _lattice_from_cellpar(a, b, c, alpha, beta, gamma):
    """Build a COLUMN-convention lattice from cell parameters (degrees).

    The textbook construction yields the three vectors; they are stacked into
    columns here through the declared-orientation factory, so the helper
    itself cannot reintroduce the very transpose these tests are about.
    Round-tripped against cell_parameters by the test below.
    """
    ca, cb, cg = (math.cos(math.radians(x)) for x in (alpha, beta, gamma))
    sg = math.sin(math.radians(gamma))
    a1 = [a, 0.0, 0.0]
    a2 = [b * cg, b * sg, 0.0]
    cx = c * cb
    cy = c * (ca - cb * cg) / sg
    a3 = [cx, cy, math.sqrt(max(c * c - cx * cx - cy * cy, 0.0))]
    return vq.lattice_from_vectors(a1, a2, a3)


def test_cellpar_helper_round_trips_through_cell_parameters():
    """Guard the fixture: if the helper were transposed, every declaration
    test below would be testing the wrong matrix."""
    want = (4.0, 5.0, 6.0, 80.0, 103.0, 95.0)
    got = vq.cell_parameters(_lattice_from_cellpar(*want))
    assert got.lengths == pytest.approx(want[:3], rel=1e-12)
    assert got.angles == pytest.approx(want[3:], abs=1e-9)


def test_declared_hexagonal_refuses_the_row_fed_cell():
    """The #128 instance: h-BN vectors fed as ROWS.

    Measured in the issue at 2.7996 / 2.1685 Angstrom and 63.435 deg, against
    the literature 2.504 / 2.504 and 60. The column reading is accepted and
    the row reading refused, on the same matrix.
    """
    (a1, a2, a3), atoms = _hbn_vectors_bohr()
    L = vq.lattice_from_vectors(a1, a2, a3)
    columns = vq.PeriodicSystem(2, L, atoms)
    rows = vq.PeriodicSystem(2, L.T, atoms)

    params = vq.check_crystal_system(columns, "hexagonal")
    assert params.a == pytest.approx(params.b, rel=1e-12)
    assert params.gamma == pytest.approx(60.0, abs=1e-9)

    with pytest.raises(vq.LatticeDeclarationError) as excinfo:
        vq.check_crystal_system(rows, "hexagonal")
    message = str(excinfo.value)
    # The refusal has to be actionable: name the failed condition, the
    # measured numbers, and that the other reading would have passed.
    assert "|a1| = |a2|" in message
    assert "gamma = 120 or 60 deg" in message
    assert "63.4349" in message
    assert "TRANSPOSE of this matrix does satisfy the declaration" in message
    assert "lattice_vectors=" in message
    assert "NOT transposed for you" in message


def test_declared_system_is_checked_on_the_periodic_axes_only():
    """A 2-D sheet's third axis is synthesized vacuum, not a lattice vector.

    Constraining |a3| would refuse every correct slab in the tree, so a plane
    declaration constrains |a1|, |a2| and the angle between them and nothing
    else -- mirroring what the periodic .out reports.
    """
    a = 4.0
    a1 = np.array([a, 0.0, 0.0])
    a2 = np.array([a / 2.0, a * math.sqrt(3.0) / 2.0, 0.0])
    atoms = [vq.Atom(6, [0.0, 0.0, 0.0])]
    for vacuum in (15.0, 30.0, 60.0):
        sheet = vq.PeriodicSystem(
            2,
            vq.lattice_from_vectors(a1, a2, np.array([0.0, 0.0, vacuum])),
            atoms,
        )
        vq.check_crystal_system(sheet, "hexagonal")  # must not raise


def test_declared_system_accepts_both_hexagonal_gammas():
    """gamma = 60 and gamma = 120 with |a1| = |a2| are the same lattice.

    The change of basis is unimodular, and this repository's own h-BN and
    graphene fixtures use 60, so a 120-only predicate would refuse vibe-qc's
    own reference cells.
    """
    a = 4.0
    for gamma in (60.0, 120.0):
        rad = math.radians(gamma)
        L = vq.lattice_from_vectors(
            [a, 0.0, 0.0],
            [a * math.cos(rad), a * math.sin(rad), 0.0],
            [0.0, 0.0, 30.0],
        )
        vq.check_crystal_system(L, "hexagonal", dim=2)  # must not raise


def test_declared_cubic_accepts_the_primitive_settings():
    """An FCC primitive cell is cubic with 60 deg angles; BCC gives 109.47.

    Requiring 90 deg would refuse the most common real input -- the tutorial
    feeds an FCC primitive cell -- so cubic asks for |a1| = |a2| = |a3| and
    alpha = beta = gamma, which the conventional, F and I settings all meet.
    """
    a = 5.0
    conventional = np.diag([a, a, a])
    fcc = vq.lattice_from_vectors(
        [0.0, a / 2, a / 2], [a / 2, 0.0, a / 2], [a / 2, a / 2, 0.0]
    )
    bcc = vq.lattice_from_vectors(
        [-a / 2, a / 2, a / 2], [a / 2, -a / 2, a / 2], [a / 2, a / 2, -a / 2]
    )
    for lattice in (conventional, fcc, bcc):
        vq.check_crystal_system(lattice, "cubic", dim=3)  # must not raise
    # ... and a cell that is genuinely not cubic is still refused.
    with pytest.raises(vq.LatticeDeclarationError):
        vq.check_crystal_system(np.diag([4.0, 5.0, 6.0]), "cubic", dim=3)
    # Equal angles alone are not enough: a general rhombohedral cell has
    # a = b = c and alpha = beta = gamma too, and is not cubic.
    with pytest.raises(vq.LatticeDeclarationError):
        vq.check_crystal_system(
            _lattice_from_cellpar(4.0, 4.0, 4.0, 70.0, 70.0, 70.0),
            "cubic",
            dim=3,
        )


def test_declared_system_enforces_equalities_not_conventions():
    """Higher symmetry must pass a lower declaration.

    ``a != c`` for tetragonal and ``beta != 90`` for monoclinic are
    conventions for choosing a cell, not requirements on one; enforcing them
    would refuse a cubic cell declared orthorhombic, which is not an error.
    """
    cubic = np.diag([5.0, 5.0, 5.0])
    for declared in ("orthorhombic", "tetragonal", "monoclinic", "triclinic"):
        vq.check_crystal_system(cubic, declared, dim=3)  # must not raise


def test_declared_monoclinic_admits_any_single_unique_axis():
    """ITA's standard setting is unique-axis b; unique-axis c is tabulated
    alongside it and common in older literature, so a bare declaration takes
    either."""
    unique_b = _lattice_from_cellpar(4.0, 5.0, 6.0, 90.0, 103.0, 90.0)
    unique_c = _lattice_from_cellpar(4.0, 5.0, 6.0, 90.0, 90.0, 103.0)
    for lattice in (unique_b, unique_c):
        vq.check_crystal_system(lattice, "monoclinic", dim=3)
    # Two non-90 angles is not a single unique axis.
    triclinic = _lattice_from_cellpar(4.0, 5.0, 6.0, 80.0, 103.0, 95.0)
    with pytest.raises(vq.LatticeDeclarationError):
        vq.check_crystal_system(triclinic, "monoclinic", dim=3)


def test_trigonal_admits_either_lattice_it_can_sit_on():
    """Trigonal has no lattice of its own: a trigonal space group sits on a
    hexagonal or a rhombohedral lattice, so the alias takes both."""
    hexagonal = _lattice_from_cellpar(4.0, 4.0, 9.0, 90.0, 90.0, 120.0)
    rhombohedral = _lattice_from_cellpar(4.0, 4.0, 4.0, 70.0, 70.0, 70.0)
    for lattice in (hexagonal, rhombohedral):
        vq.check_crystal_system(lattice, "trigonal", dim=3)
    with pytest.raises(vq.LatticeDeclarationError):
        vq.check_crystal_system(np.diag([4.0, 5.0, 6.0]), "trigonal", dim=3)


def test_declaration_refuses_a_category_error_rather_than_validating_nothing():
    """A declaration that cannot be tested must fail loudly, not pass."""
    sheet = vq.PeriodicSystem(2, np.diag([4.0, 4.0, 30.0]), [vq.Atom(6, [0, 0, 0])])
    with pytest.raises(ValueError, match="cannot describe a 2-D cell"):
        vq.check_crystal_system(sheet, "cubic")
    chain = vq.PeriodicSystem(1, np.diag([4.0, 30.0, 30.0]), [vq.Atom(1, [0, 0, 0])])
    with pytest.raises(ValueError, match="not meaningful for a 1-D cell"):
        vq.check_crystal_system(chain, "hexagonal")
    with pytest.raises(ValueError, match="unknown crystal system"):
        vq.check_crystal_system(np.diag([4.0, 4.0, 4.0]), "hexagonial", dim=3)


def test_lattice_from_vectors_declares_at_the_orientation_free_factory():
    """The declaration is available where the docs tell users to build cells."""
    a = 4.0
    a1 = [a, 0.0, 0.0]
    a2 = [a / 2.0, a * math.sqrt(3.0) / 2.0, 0.0]
    a3 = [0.0, 0.0, 30.0]
    L = vq.lattice_from_vectors(a1, a2, a3, crystal_system="hexagonal", dim=2)
    assert np.allclose(L[:, 0], a1)
    # Feeding the same three vectors as rows is the mistake #128 is about.
    with pytest.raises(vq.LatticeDeclarationError):
        vq.lattice_from_vectors(
            *np.asarray([a1, a2, a3]).T, crystal_system="hexagonal", dim=2
        )


# ---------------------------------------------------------------------------
# GitLab #128 ask 1: "...or space group".
#
# A crystal-system declaration is about the LATTICE and is tested on the metric
# alone. A space-group declaration is about the whole STRUCTURE and is tested
# by handing lattice and basis to spglib, so it catches what no metric can: an
# atom on the wrong site, a mistranscribed fractional coordinate, a basis that
# broke the symmetry the caller believed the structure had.
#
# It is complementary, not stronger. Measured with the Cartesian positions held
# fixed and only the lattice transposed -- the actual shape of every instance
# in this issue -- hexagonal goes P-6m2 (187) -> Pm (6) and is caught, while
# monoclinic P2/m and triclinic P-1 keep their groups and are not. Neither
# check sees those; the tests below pin that honestly rather than implying a
# guarantee that does not hold.
# ---------------------------------------------------------------------------


def _hbn_system(lattice_is_rows: bool = False):
    (a1, a2, a3), atoms = _hbn_vectors_bohr()
    L = vq.lattice_from_vectors(a1, a2, a3)
    return vq.PeriodicSystem(3, L.T if lattice_is_rows else L, atoms)


def test_declared_space_group_accepts_the_structure_that_has_it():
    pytest.importorskip("spglib")
    detected = vq.check_space_group(_hbn_system(), "P-6m2")
    assert detected.number == 187
    # The same declaration by International Tables number.
    assert vq.check_space_group(_hbn_system(), 187).number == 187


def test_declared_space_group_refuses_the_row_fed_cell():
    """The #128 instance again, caught through the structure this time."""
    pytest.importorskip("spglib")
    with pytest.raises(vq.LatticeDeclarationError) as excinfo:
        vq.check_space_group(_hbn_system(lattice_is_rows=True), "P-6m2")
    message = str(excinfo.value)
    assert "detected : 6 (Pm)" in message
    assert "TRANSPOSE of this lattice does satisfy the declaration" in message
    assert "lattice_vectors=" in message
    assert "NOT transposed for you" in message


def test_declared_space_group_reads_a_symbol_loosely_and_a_number_exactly():
    pytest.importorskip("spglib")
    system = vq.PeriodicSystem(
        3, np.diag([5.4, 5.4, 5.4]), [vq.Atom(14, [0.0, 0.0, 0.0])]
    )
    for spelling in ("Pm-3m", "p m -3 m", "  PM-3M  "):
        vq.check_space_group(system, spelling)
    vq.check_space_group(system, 221)
    with pytest.raises(vq.LatticeDeclarationError):
        vq.check_space_group(system, 227)
    with pytest.raises(ValueError, match="1 to 230"):
        vq.check_space_group(system, 999)


def test_declared_space_group_needs_a_structure_and_three_dimensions():
    """A space group is a property of the structure, not of a lattice, and a
    slab's would depend on how much vacuum was synthesized."""
    pytest.importorskip("spglib")
    with pytest.raises(TypeError, match="not a bare lattice matrix"):
        vq.check_space_group(np.diag([5.0, 5.0, 5.0]), 221)
    sheet = vq.PeriodicSystem(2, np.diag([4.0, 4.0, 30.0]), [vq.Atom(6, [0, 0, 0])])
    with pytest.raises(ValueError, match="only 3-D cells"):
        vq.check_space_group(sheet, 191)


def test_declared_space_group_honours_symprec():
    """spglib's tolerance changes the answer, so it is exposed, not hidden.

    A cell 0.1% off cubic reads as P4/mmm at the 1e-4 default and Pm-3m at
    1e-2; a declaration that is right only within a looser tolerance has to
    say so rather than silently passing.
    """
    pytest.importorskip("spglib")
    system = vq.PeriodicSystem(
        3, np.diag([5.4, 5.4 * 1.001, 5.4]), [vq.Atom(14, [0.0, 0.0, 0.0])]
    )
    assert vq.check_space_group(system, 123).number == 123
    with pytest.raises(vq.LatticeDeclarationError):
        vq.check_space_group(system, 221)
    assert vq.check_space_group(system, 221, symprec=1.0e-2).number == 221


def test_space_group_and_crystal_system_are_complementary_not_redundant():
    """The honest limit, pinned so nobody upgrades the claim by accident.

    An exactly hexagonal LATTICE carrying a symmetry-breaking basis is a
    legitimate cell: the metric declaration holds and the detected space group
    is far lower. That is exactly why a metric check must not be driven off a
    detected space-group number.
    """
    pytest.importorskip("spglib")
    (a1, a2, a3), _ = _hbn_vectors_bohr()
    L = vq.lattice_from_vectors(a1, a2, a3)
    # A basis that breaks the hexagonal symmetry without touching the lattice.
    atoms = [vq.Atom(5, [0.0, 0.0, 0.0]), vq.Atom(7, list(0.31 * a1 + 0.17 * a2))]
    system = vq.PeriodicSystem(3, L, atoms)

    # The lattice is still hexagonal, and the metric declaration holds.
    vq.check_crystal_system(system, "hexagonal")
    # The structure is not: spglib reports Pm, whose crystal system is
    # monoclinic, for a cell whose lattice is exactly hexagonal.
    detected = vq.check_space_group(system, "Pm")
    assert detected.number == 6
    # Declaring the lattice's own holohedry is therefore refused, which is
    # correct -- and is precisely why a metric check must not be driven off a
    # detected space-group number: doing so would call this lattice
    # monoclinic and refuse the hexagonal declaration that legitimately holds.
    with pytest.raises(vq.LatticeDeclarationError):
        vq.check_space_group(system, 191)
