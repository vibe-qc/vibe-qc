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
