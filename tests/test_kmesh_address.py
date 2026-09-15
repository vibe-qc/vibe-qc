"""Exact integer addressing on a regular k-mesh.

``cpp/include/vibeqc/kmesh_address.hpp`` is the layer the periodic
correlated methods will do their crystal-momentum bookkeeping on, so its
whole value is that it is *exact*: every claim below is checked as an
identity between integers, never as a float comparison with a tolerance.
Where a float does appear it is because a consumer sees that float --
``bloch.cpp`` builds its k-points from ``fractional_at`` -- and then the
assertion is bitwise equality, not closeness.

The Python surface exercised here (``vibeqc._vibeqc_core._RegularKMesh``)
is a diagnostic binding, not an API: it exists because this repository's
change-safety inventory is pytest and an exact-integer C++ module with no
Python caller would otherwise ship untested. It is underscore-prefixed and
not re-exported from ``vibeqc``; nothing outside this file should use it.
"""

from __future__ import annotations

import math
import resource
import sys
from fractions import Fraction
from itertools import product

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import _vibeqc_core as core


# The project reference mesh: Gamma-centred, unshifted (CLAUDE.md, and
# python/vibeqc/periodic_runner.py's KMESH_CONVENTION_GAMMA).
REFERENCE_MESH = (8, 8, 8)
REFERENCE_SHIFT = (0, 0, 0)

# Meshes the module has to get right, spanning the four corners of the
# (even/odd) x (unshifted/shifted) grid plus the degenerate 1x1x1 case and
# a fully mixed one.
CASES = [
    ((1, 1, 1), (0, 0, 0)),
    ((1, 1, 1), (1, 1, 1)),
    ((2, 3, 4), (0, 0, 0)),
    ((2, 2, 2), (1, 1, 1)),
    ((4, 4, 2), (1, 1, 1)),
    ((3, 3, 3), (1, 1, 1)),
    ((5, 3, 1), (1, 1, 0)),
    ((2, 3, 4), (1, 0, 1)),
    ((1, 4, 7), (0, 1, 1)),
]


def _divisions(mesh):
    """(m0, m1, m2) in the module's last-axis-fast order."""
    return list(product(range(mesh[0]), range(mesh[1]), range(mesh[2])))


def _legacy_fractional(mesh, shift, m):
    """The pre-refactor bloch.cpp expression, transcribed exactly.

    ``k_frac[d] = (m_d + 0.5 * is_shift[d]) / mesh[d]`` in C++ doubles.
    Python floats are IEEE-754 doubles with correctly rounded ``/`` and
    ``*``, so this reproduces the old C++ value bit for bit.
    """
    return [(m[d] + 0.5 * shift[d]) / mesh[d] for d in range(3)]


def _peak_rss_bytes():
    """ru_maxrss is bytes on macOS and kilobytes on Linux."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw if sys.platform == "darwin" else raw * 1024


def _exact_fraction(doubled, modulus):
    return [Fraction(int(doubled[d]), int(modulus[d])) for d in range(3)]


# --------------------------------------------------------------------------
# Descriptor validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mesh",
    [(0, 1, 1), (1, 0, 1), (1, 1, 0), (-1, 2, 2), (2, 2, -8)],
)
def test_rejects_non_positive_divisions(mesh):
    with pytest.raises(RuntimeError, match="strictly positive divisions"):
        core._RegularKMesh(list(mesh))


@pytest.mark.parametrize(
    "shift",
    [(2, 0, 0), (0, -1, 0), (0, 0, 2), (1, 1, 3)],
)
def test_rejects_shift_outside_zero_or_one(shift):
    with pytest.raises(RuntimeError, match="must be 0 or 1"):
        core._RegularKMesh([2, 2, 2], list(shift))


def test_rejects_divisions_that_would_overflow_the_doubled_modulus():
    """The per-axis bound exists so 2*N still fits an int."""
    limit = core._RegularKMesh.max_divisions
    core._RegularKMesh([limit, 1, 1])  # the bound itself is admissible
    for axis in range(3):
        mesh = [1, 1, 1]
        mesh[axis] = limit + 1
        with pytest.raises(RuntimeError, match="divisions per axis"):
            core._RegularKMesh(mesh)


def test_point_count_overflow_is_refused_without_allocating():
    """A mesh whose product overflows must be rejected as arithmetic.

    10^27 points cannot be counted in a ``std::size_t``, let alone stored.
    The failure has to come from the checked multiply, while the mesh is
    still three integers -- not from a container that tried to size itself
    and died, and not from a wrapped count that silently names a *small*
    mesh. Peak RSS is the witness: an implementation that reached for
    memory first would move it by gigabytes or abort.
    """
    before = _peak_rss_bytes()
    for mesh in ([10**9, 10**9, 10**9],
                 [2 * 10**8, 10**9, 10**9],
                 [10**9, 5 * 10**8, 10**9]):
        with pytest.raises(RuntimeError, match="std::size_t can count"):
            core._RegularKMesh(mesh)
    grew = _peak_rss_bytes() - before
    assert grew < 64 * 1024 * 1024, (
        f"rejecting an oversized mesh grew peak RSS by {grew} bytes; the "
        f"check is supposed to happen before anything allocates"
    )


def test_descriptor_metadata():
    mesh = core._RegularKMesh([2, 3, 4], [1, 0, 1])
    assert tuple(mesh.mesh) == (2, 3, 4)
    assert tuple(mesh.is_shift) == (1, 0, 1)
    assert tuple(mesh.doubled_modulus) == (4, 6, 8)
    assert len(mesh) == 24


# --------------------------------------------------------------------------
# index <-> address bijection, and the ordering contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mesh,shift", CASES)
def test_address_index_round_trip_is_a_bijection(mesh, shift):
    grid = core._RegularKMesh(list(mesh), list(shift))
    n_k = mesh[0] * mesh[1] * mesh[2]
    assert len(grid) == n_k

    seen = set()
    for index, m in enumerate(_divisions(mesh)):
        doubled = tuple(grid.address(index))
        # a_d = 2 m_d + s_d, so the linear index really is last-axis-fast.
        assert doubled == tuple(2 * m[d] + shift[d] for d in range(3))
        assert grid.index(list(doubled)) == index
        assert all(0 <= doubled[d] < 2 * mesh[d] for d in range(3))
        seen.add(doubled)
    assert len(seen) == n_k


@pytest.mark.parametrize("mesh,shift", CASES)
def test_transfer_address_index_round_trip_is_a_bijection(mesh, shift):
    grid = core._RegularKMesh(list(mesh), list(shift))
    seen = set()
    for index, m in enumerate(_divisions(mesh)):
        doubled = tuple(grid.transfer_address(index))
        # The transfer group is the zero-shift mesh whatever is_shift is.
        assert doubled == tuple(2 * m[d] for d in range(3))
        assert grid.index_of_transfer(list(doubled)) == index
        seen.add(doubled)
    assert len(seen) == mesh[0] * mesh[1] * mesh[2]


def test_index_rejects_out_of_range_and_unreduced_addresses():
    grid = core._RegularKMesh([2, 3, 4])
    with pytest.raises(RuntimeError, match="outside the 24 points"):
        grid.address(24)
    with pytest.raises(RuntimeError, match="not reduced"):
        grid.index([4, 0, 0])
    with pytest.raises(RuntimeError, match="not reduced"):
        grid.index([0, -2, 0])


def test_the_two_address_groups_do_not_substitute_for_one_another():
    """The runtime half of the k / transfer-q separation.

    The compile-time half is a pair of ``static_assert``s in
    kmesh_address.cpp: the two address types do not convert. What is
    visible from here is the parity that makes the separation matter -- on
    a shifted mesh a k-point is odd and a transfer momentum is even, so
    handing one to the other's lookup names a point off the grid.
    """
    shifted = core._RegularKMesh([2, 2, 2], [1, 1, 1])
    k = list(shifted.address(0))
    q = list(shifted.transfer_address(0))
    assert k == [1, 1, 1] and q == [0, 0, 0]

    with pytest.raises(RuntimeError, match="parity"):
        shifted.index(q)
    with pytest.raises(RuntimeError, match="is odd on axis"):
        shifted.index_of_transfer(k)

    # On an unshifted mesh the two groups coincide numerically, which is
    # exactly why the distinction has to be carried by the type and not
    # inferred from the values.
    plain = core._RegularKMesh([2, 2, 2])
    assert list(plain.address(3)) == list(plain.transfer_address(3))


# --------------------------------------------------------------------------
# Fractional coordinates, and byte-neutrality against the old bloch.cpp
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mesh,shift", CASES)
def test_fractional_at_matches_the_legacy_formula_bitwise(mesh, shift):
    """(2m + s) / (2N) is the same double as (m + s/2) / N.

    Numerator and denominator are both exactly twice the old ones, so the
    real quotient is unchanged; every operand is exactly representable and
    IEEE division is correctly rounded. This is the assertion that lets
    bloch.cpp be rerouted through RegularKMesh with no reference value
    moving anywhere in the tree, so it is checked with ``==`` and not with
    a tolerance.
    """
    grid = core._RegularKMesh(list(mesh), list(shift))
    for index, m in enumerate(_divisions(mesh)):
        got = np.asarray(grid.fractional_at(index), dtype=float)
        want = np.asarray(_legacy_fractional(mesh, shift, m), dtype=float)
        assert got.tolist() == want.tolist(), (
            f"mesh {mesh} shift {shift} point {m}: {got!r} != {want!r}"
        )
        # And the same value reached through the address form.
        via_address = np.asarray(
            grid.fractional(list(grid.address(index))), dtype=float
        )
        assert via_address.tolist() == want.tolist()


def test_bloch_mesh_matches_legacy_formula_bitwise():
    """The shipped k-point sequence: same order, same bits.

    A cubic cell keeps the reciprocal lattice exactly diagonal, so
    ``B @ k_frac`` is a single scaling per axis and reproduces bit for bit
    on either side of the language boundary; any summation-order freedom in
    the matrix-vector product is multiplied by exact zeros.
    """
    system = vq.PeriodicSystem(
        3, np.eye(3) * 6.0, [vq.Atom(1, [0.0, 0.0, 0.0])]
    )
    b_mat = np.asarray(system.reciprocal_lattice(), dtype=float)
    assert np.count_nonzero(b_mat - np.diag(np.diag(b_mat))) == 0, (
        f"this test needs an exactly diagonal reciprocal lattice to compare "
        f"bit for bit across the language boundary; got {b_mat!r}"
    )
    scale = np.diag(b_mat)

    for mesh, shift in [((1, 1, 1), (0, 0, 0)),
                        ((2, 3, 4), (0, 0, 0)),
                        ((2, 2, 2), (1, 1, 1)),
                        ((3, 3, 3), (1, 1, 1)),
                        ((4, 3, 2), (1, 0, 1))]:
        got = np.asarray(
            vq.monkhorst_pack(system, list(mesh), list(shift)).kpoints,
            dtype=float,
        )
        # B is diagonal, so B @ k_frac is one multiply per axis on either
        # side: the summation-order freedom of a matrix-vector product is
        # multiplied by exact zeros and cannot change a bit.
        want = np.asarray(
            [_legacy_fractional(mesh, shift, m) for m in _divisions(mesh)],
            dtype=float,
        ) * scale
        assert got.shape == want.shape
        np.testing.assert_array_equal(got, want)


def test_monkhorst_pack_now_refuses_a_shift_it_only_used_to_document():
    """bloch.hpp always said ``is_shift[i] in {0, 1}``; now it holds.

    The Python layer already screened this (kpoints._shift_tuple_for_system),
    so no caller in the tree changes behaviour -- but the C++ entry point
    used to build a silently different grid, (m + 1)/N for a shift of 2.
    """
    system = vq.PeriodicSystem(
        3, np.eye(3) * 6.0, [vq.Atom(1, [0.0, 0.0, 0.0])]
    )
    with pytest.raises(RuntimeError, match="must be 0 or 1"):
        core.monkhorst_pack(system, [2, 2, 2], [2, 0, 0], False)


# --------------------------------------------------------------------------
# The Monkhorst-Pack convention itself: the half step is the EVEN case
# --------------------------------------------------------------------------


def _classical_mp_set(q):
    """Monkhorst & Pack 1976, Eq. (3): u_r = (2r - q - 1) / (2q)."""
    return {Fraction(2 * r - q - 1, 2 * q) for r in range(1, q + 1)}


def _folded_axis_set(n, shift):
    """This module's axis-``d`` coordinates folded into [-1/2, 1/2)."""
    out = set()
    for m in range(n):
        f = Fraction(2 * m + shift, 2 * n)
        out.add(f - math.floor(f + Fraction(1, 2)))
    return out


@pytest.mark.parametrize("q", [1, 3, 5, 7, 9])
def test_odd_meshes_are_classical_monkhorst_pack_without_a_shift(q):
    """For odd q the classical set already contains Gamma.

    This is the fact the corrected comments in bloch.hpp and crystal.hpp
    now state: an unshifted odd mesh *is* the classical Monkhorst-Pack
    mesh, and shifting it by half a step moves it off.
    """
    assert _folded_axis_set(q, 0) == _classical_mp_set(q)
    assert _folded_axis_set(q, 1).isdisjoint(_classical_mp_set(q))
    assert Fraction(0) in _folded_axis_set(q, 0)


@pytest.mark.parametrize("q", [2, 4, 6, 8])
def test_even_meshes_need_the_half_step_to_be_classical_monkhorst_pack(q):
    """For even q the classical set excludes Gamma; the shift supplies it.

    The half-step offset is the even-mesh case, which is what the replaced
    comments had backwards.
    """
    assert _folded_axis_set(q, 1) == _classical_mp_set(q)
    assert _folded_axis_set(q, 0).isdisjoint(_classical_mp_set(q))
    assert Fraction(0) not in _classical_mp_set(q)
    assert Fraction(0) in _folded_axis_set(q, 0)


# --------------------------------------------------------------------------
# Exact modular algebra
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mesh,shift", CASES)
def test_negation_closes_exactly_and_reports_its_wrap(mesh, shift):
    grid = core._RegularKMesh(list(mesh), list(shift))
    modulus = [2 * n for n in mesh]
    for index in range(len(grid)):
        a = list(grid.address(index))
        reduced, wrap = grid.negate(a)
        reduced = [int(x) for x in reduced]
        wrap = [int(x) for x in wrap]
        # -a_d = reduced_d + M_d * G_d, exactly.
        assert [-a[d] for d in range(3)] == [
            reduced[d] + modulus[d] * wrap[d] for d in range(3)
        ]
        # The partner is on the same grid: parity survives negation.
        assert [reduced[d] % 2 for d in range(3)] == list(shift)
        assert grid.negate_index(index) == grid.index(reduced)
        # Involution.
        assert list(grid.negate(reduced)[0]) == a


@pytest.mark.parametrize("mesh,shift", CASES)
def test_transfer_momentum_closes_exactly_over_every_pair(mesh, shift):
    grid = core._RegularKMesh(list(mesh), list(shift))
    modulus = [2 * n for n in mesh]
    n_k = len(grid)
    for ki in range(n_k):
        reached = set()
        ai = [int(x) for x in grid.address(ki)]
        for kj in range(n_k):
            aj = [int(x) for x in grid.address(kj)]
            q, wrap = grid.transfer(ai, aj)
            q = [int(x) for x in q]
            wrap = [int(x) for x in wrap]
            # k_j - k_i = q + G, exactly, in doubled integers.
            assert [aj[d] - ai[d] for d in range(3)] == [
                q[d] + modulus[d] * wrap[d] for d in range(3)
            ]
            # A transfer momentum is even on every axis, whatever the shift.
            assert all(v % 2 == 0 for v in q)
            assert grid.transfer_index(ki, kj) == grid.index_of_transfer(q)
            reached.add(tuple(q))
        # Every k_i reaches the whole transfer group exactly once.
        assert len(reached) == n_k


@pytest.mark.parametrize("mesh,shift", CASES)
def test_adding_a_transfer_inverts_the_pair_difference_exactly(mesh, shift):
    """For each ``(k_i, q)``, recover the unique ``k_j`` and exact wrap."""

    grid = core._RegularKMesh(list(mesh), list(shift))
    modulus = [2 * n for n in mesh]
    for ki, q_index in product(range(len(grid)), repeat=2):
        ai = [int(value) for value in grid.address(ki)]
        q = [int(value) for value in grid.transfer_address(q_index)]
        aj, wrap = grid.add_transfer(ai, q)
        aj = [int(value) for value in aj]
        wrap = [int(value) for value in wrap]

        assert [ai[d] + q[d] for d in range(3)] == [
            aj[d] + modulus[d] * wrap[d] for d in range(3)
        ]
        kj = grid.add_transfer_index(ki, q_index)
        assert kj == grid.index(aj)
        assert grid.transfer_index(ki, kj) == q_index


@pytest.mark.parametrize(
    "mesh,shift",
    [((1, 1, 1), (0, 0, 0)),
     ((1, 1, 1), (1, 1, 1)),
     ((2, 2, 2), (0, 0, 0)),
     ((2, 2, 2), (1, 1, 1)),
     ((3, 1, 2), (1, 0, 1)),
     ((1, 2, 3), (0, 1, 1))],
)
def test_conserved_fourth_index_closes_exhaustively(mesh, shift):
    """k_i - k_a + k_j = k_b + G, checked over every triple.

    McClain, Sun, Chan & Berkelbach, J. Chem. Theory Comput. 13, 1209
    (2017), text below Eq. (26): the doubles amplitude conserves crystal
    momentum up to a reciprocal-lattice vector, which fixes the fourth
    index. Small meshes only -- the point is exhaustiveness, and N_k^3
    triples is what that costs.
    """
    grid = core._RegularKMesh(list(mesh), list(shift))
    modulus = [2 * n for n in mesh]
    n_k = len(grid)
    addresses = [[int(x) for x in grid.address(i)] for i in range(n_k)]

    for ki in range(n_k):
        for ka in range(n_k):
            for kj in range(n_k):
                ai, aa, aj = addresses[ki], addresses[ka], addresses[kj]
                kb, wrap = grid.conserved(ai, aa, aj)
                kb = [int(x) for x in kb]
                wrap = [int(x) for x in wrap]
                assert [ai[d] - aa[d] + aj[d] for d in range(3)] == [
                    kb[d] + modulus[d] * wrap[d] for d in range(3)
                ]
                # k_b is a point of the k-grid, not of the transfer group.
                assert [kb[d] % 2 for d in range(3)] == list(shift)
                index = grid.conserved_index(ki, ka, kj)
                assert index == grid.index(kb)
                # The same identity in fractional coordinates, exactly.
                fi = _exact_fraction(ai, modulus)
                fa = _exact_fraction(aa, modulus)
                fj = _exact_fraction(aj, modulus)
                fb = _exact_fraction(kb, modulus)
                assert [fi[d] - fa[d] + fj[d] for d in range(3)] == [
                    fb[d] + wrap[d] for d in range(3)
                ]


def test_wrap_vectors_are_the_expected_reciprocal_lattice_vectors():
    """Concrete wraps, not just self-consistent ones."""
    grid = core._RegularKMesh([2, 2, 2])
    zero = [0, 0, 0]
    edge = list(grid.address(grid.index([0, 0, 2])))  # k_frac = (0, 0, 1/2)

    def _ints(values):
        return [int(v) for v in values]

    # Gamma is its own negative with no wrap.
    reduced, wrap = grid.negate(zero)
    assert _ints(reduced) == zero and _ints(wrap) == zero

    # The zone-boundary point is its own negative, one G away.
    reduced, wrap = grid.negate(edge)
    assert _ints(reduced) == edge
    assert _ints(wrap) == [0, 0, -1]

    # k_i - k_a + k_j with k_a = Gamma and both others at the boundary
    # lands back on Gamma, carrying exactly one reciprocal-lattice vector.
    kb, wrap = grid.conserved(edge, zero, edge)
    assert _ints(kb) == zero
    assert _ints(wrap) == [0, 0, 1]

    # Gamma - boundary + Gamma wraps the other way.
    kb, wrap = grid.conserved(zero, edge, zero)
    assert _ints(kb) == edge
    assert _ints(wrap) == [0, 0, -1]


def test_conserved_index_agrees_with_the_address_route_on_a_shifted_mesh():
    """The index route drops the shift; the address route carries it.

    (2m_i + s) - (2m_a + s) + (2m_j + s) = 2(m_i - m_a + m_j) + s, so the
    index arithmetic is shift-independent while the address it names is
    not. Both have to reach the same point.
    """
    for shift in [(0, 0, 0), (1, 1, 1), (1, 0, 1)]:
        grid = core._RegularKMesh([3, 2, 2], list(shift))
        plain = core._RegularKMesh([3, 2, 2])
        n_k = len(grid)
        for ki, ka, kj in product(range(n_k), repeat=3):
            index = grid.conserved_index(ki, ka, kj)
            assert index == plain.conserved_index(ki, ka, kj)
            assert index == grid.index(
                list(grid.conserved(list(grid.address(ki)),
                                    list(grid.address(ka)),
                                    list(grid.address(kj)))[0])
            )


# --------------------------------------------------------------------------
# The project reference mesh
# --------------------------------------------------------------------------


def test_reference_mesh_metadata_and_lookups_allocate_no_table():
    """(8, 8, 8) works on demand, and no N_k^3 table exists to work from.

    512^3 is 134,217,728 entries -- over a gigabyte at 8 bytes each -- to
    store a number three integer divisions produce on the spot. The
    structural half of this assertion is the one that will still be true
    next year: there is no table-shaped entry point on the class at all,
    so a later chat cannot quietly add one without this test noticing.
    """
    exported = [name for name in dir(core._RegularKMesh)
                if not name.startswith("__")]
    assert not [name for name in exported if "table" in name.lower()], (
        f"a table-shaped entry point appeared on _RegularKMesh: {exported}"
    )

    before = _peak_rss_bytes()
    grid = core._RegularKMesh(list(REFERENCE_MESH), list(REFERENCE_SHIFT))
    assert tuple(grid.mesh) == REFERENCE_MESH
    assert tuple(grid.is_shift) == REFERENCE_SHIFT
    assert tuple(grid.doubled_modulus) == (16, 16, 16)
    assert len(grid) == 512

    n_k = len(grid)
    rng = np.random.default_rng(20260902)
    triples = rng.integers(0, n_k, size=(4096, 3))
    for ki, ka, kj in triples.tolist():
        index = grid.conserved_index(ki, ka, kj)
        assert 0 <= index < n_k
        assert index == grid.index(
            list(grid.conserved(list(grid.address(ki)),
                                list(grid.address(ka)),
                                list(grid.address(kj)))[0])
        )

    grew = _peak_rss_bytes() - before
    assert grew < 64 * 1024 * 1024, (
        f"the reference mesh grew peak RSS by {grew} bytes; an N_k^3 "
        f"table would be ~1.07e9"
    )


def test_reference_mesh_is_gamma_centred():
    """Gamma is in the set, at index 0, with no shift."""
    grid = core._RegularKMesh(list(REFERENCE_MESH), list(REFERENCE_SHIFT))
    assert list(grid.address(0)) == [0, 0, 0]
    assert np.asarray(grid.fractional_at(0), dtype=float).tolist() == [
        0.0, 0.0, 0.0
    ]
    assert grid.conserved_index(0, 0, 0) == 0
    assert grid.negate_index(0) == 0


# --------------------------------------------------------------------------
# is_shift is enforced on every route, not just the unreduced one (#691)
# --------------------------------------------------------------------------


def _crystal_cubic(a_bohr: float = 6.0):
    crystal = vq.Crystal()
    crystal.lattice = np.eye(3) * a_bohr
    crystal.fractional_coords = np.zeros((3, 1))
    crystal.species = [1]
    return crystal


@pytest.mark.parametrize("bad", [2, -1, 7, 100])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_irreducible_kpoints_refuses_a_shift_outside_zero_or_one(bad, axis):
    """The public export validated ``mesh`` but never ``is_shift`` (#691).

    It handed the value to spglib and then computed
    ``k_frac = (g + is_shift/2) / mesh``, so a shift of 2 silently named a
    different grid and a shift of 7 left the first Brillouin zone outright
    (``k_frac = 1.75`` on a two-division axis). ``vq.irreducible_kpoints``
    is in ``vibeqc.__all__``, so this is reachable without the screen in
    kpoints.py that ``KPoints.monkhorst_pack`` goes through.
    """
    shift = [0, 0, 0]
    shift[axis] = bad
    with pytest.raises(RuntimeError, match="must be 0 or 1"):
        vq.irreducible_kpoints(_crystal_cubic(), [2, 2, 2], shift)


@pytest.mark.parametrize("use_symmetry", [False, True])
@pytest.mark.parametrize("bad", [2, -1, 7])
def test_monkhorst_pack_refuses_the_same_shift_on_both_branches(
    use_symmetry, bad
):
    """The two branches of one function must have one contract (#691).

    Before the fix ``use_symmetry=False`` refused an out-of-range shift --
    it builds through the validated RegularKMesh -- while
    ``use_symmetry=True`` passed it to spglib unchecked. Same function,
    same documented contract, two behaviours selected by a boolean.
    """
    system = vq.PeriodicSystem(
        3, np.eye(3) * 6.0, [vq.Atom(1, [0.0, 0.0, 0.0])]
    )
    vq.attach_symmetry(system, symprec=1e-4)
    with pytest.raises(RuntimeError, match="must be 0 or 1"):
        core.monkhorst_pack(system, [2, 2, 2], [bad, 0, 0], use_symmetry)


@pytest.mark.parametrize("use_symmetry", [False, True])
@pytest.mark.parametrize("shift", [(0, 0, 0), (1, 0, 1), (1, 1, 1)])
def test_valid_shifts_still_build_on_both_branches(use_symmetry, shift):
    """The guard rejects only what the headers always said was invalid."""
    system = vq.PeriodicSystem(
        3, np.eye(3) * 6.0, [vq.Atom(1, [0.0, 0.0, 0.0])]
    )
    vq.attach_symmetry(system, symprec=1e-4)
    bm = core.monkhorst_pack(system, [2, 2, 2], list(shift), use_symmetry)
    assert len(bm.kpoints) >= 1
    assert tuple(bm.is_shift) == shift
    assert abs(sum(bm.weights) - 1.0) < 1e-12


def test_a_stray_shift_on_a_non_periodic_axis_is_still_tolerated():
    """Validation runs AFTER the non-periodic-axis normalisation.

    A 2-D system already forces mesh=1 and shift=0 on the vacuum axis, so
    a stray value there was never used. Rejecting it would be a new
    refusal class rather than the contract repair this is, so the guard
    deliberately sits downstream of the normalisation.
    """
    system = vq.PeriodicSystem(
        2, np.eye(3) * 6.0, [vq.Atom(1, [0.0, 0.0, 0.0])]
    )
    bm = core.monkhorst_pack(system, [2, 2, 1], [1, 1, 5], False)
    assert tuple(bm.is_shift) == (1, 1, 0)
    assert tuple(bm.mesh) == (2, 2, 1)
