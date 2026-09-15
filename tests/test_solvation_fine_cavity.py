"""The COSMO FINE Cavity of Klamt & Diedenhofen 2018 (doi:10.1002/jcc.25342).

Equation and step numbers refer to that paper. No reference segment set is
published, so these tests do not check segment-for-segment agreement with
COSMOtherm. What they check is everything that *can* be checked without one:
the limits where an exact answer exists, convergence in the grid spacing,
and the structural invariants a cavity must satisfy to be usable at all.

Almost all of it is pure geometry and needs no SCF.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from vibeqc.solvation.cavity import ANG_TO_BOHR
from vibeqc.solvation.fine_cavity import (
    GRID_SPACING_ANG,
    MIN_BASIS_AREA_BOHR2,
    MIN_SEGMENT_AREA_BOHR2,
    NSPA_FINE,
    NSPH_FINE,
    PD_A1,
    PD_A2,
    PD_C,
    PD_M,
    PD_N_NEAREST,
    FineCavity,
    PseudoDensityParams,
    build_fine_cavity,
    geodesic_sphere,
    log_pseudo_density,
    PROJECTION_SOFTNESS,
    marching_tetrahedra,
    quad_groups,
    molecular_frame,
    molecular_frame_jacobian,
    projection_radius,
    pseudo_density,
    sphere_points_for_count,
)

A = ANG_TO_BOHR

# Klamt 1998 section 5.1 fitted COSMO radii, angstrom.
KLAMT_RADII_ANG = {1: 1.30, 6: 2.00, 7: 1.83, 8: 1.72}


def _water():
    """Experimental water geometry, in bohr, with Klamt 1998 radii."""
    d = 0.9584 * A
    th = math.radians(104.45)
    pos = np.array(
        [[0.0, 0.0, 0.0], [d, 0.0, 0.0], [d * math.cos(th), d * math.sin(th), 0.0]]
    )
    z = [8, 1, 1]
    rad = np.array([KLAMT_RADII_ANG[e] * A for e in z])
    return pos, z, rad


# ---------------------------------------------------------------------
# Published parameters
# ---------------------------------------------------------------------


def test_published_parameters():
    """2018 after eq. 8 and workflow steps 2, 5, 6."""
    assert (PD_A1, PD_A2, PD_C, PD_M) == (-15.0, -9.0, 5.0, 4)
    assert PD_N_NEAREST == 5
    assert GRID_SPACING_ANG == 0.3
    assert (NSPH_FINE, NSPA_FINE) == (92, 162)
    assert MIN_BASIS_AREA_BOHR2 == 1e-3
    d = PseudoDensityParams()
    assert (d.a1, d.a2, d.c, d.m, d.n_nearest) == (-15.0, -9.0, 5.0, 4, 5)


# ---------------------------------------------------------------------
# Geodesic segment grids -- the COC "magic numbers"
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "frequency,count", [(1, 12), (2, 42), (3, 92), (4, 162), (5, 252)]
)
def test_geodesic_sphere_hits_the_magic_numbers(frequency, count):
    """2018: only ``10 i^2 3^j + 2`` point counts are allowed.

    An off-lattice grid would not tile the sphere evenly and the segment areas
    would inherit the unevenness, so the count is an invariant rather than a
    convenience.
    """
    pts = geodesic_sphere(frequency)
    assert pts.shape == (count, 3)
    np.testing.assert_allclose(np.linalg.norm(pts, axis=1), 1.0, atol=1e-12)
    # No duplicates: shared edges and corners must be merged, not counted twice.
    assert np.unique(np.round(pts, 9), axis=0).shape[0] == count


def test_geodesic_sphere_is_centrosymmetric_in_bulk():
    """The icosahedral lattice has no net direction; a lopsided grid would
    bias every segment area on the atom it paves."""
    pts = geodesic_sphere(4)
    assert np.allclose(pts.mean(axis=0), 0.0, atol=1e-10)


def test_sphere_points_for_count_rejects_a_non_magic_number():
    assert sphere_points_for_count(92).shape[0] == 92
    assert sphere_points_for_count(162).shape[0] == 162
    with pytest.raises(ValueError, match=r"not of the form"):
        sphere_points_for_count(100)


def test_geodesic_sphere_rejects_a_bad_frequency():
    with pytest.raises(ValueError, match="frequency must be"):
        geodesic_sphere(0)


# ---------------------------------------------------------------------
# The pseudo-density (eq. 8)
# ---------------------------------------------------------------------


def test_isolated_atom_iso_surface_is_exactly_the_vdw_sphere():
    """The single strongest check on eq. 8 as implemented.

    The atomic term is ``exp{a1 (tau - 1)}``, so for one atom PD is exactly 1
    where ``tau = 1``, i.e. on the sphere of the atom's own radius, for *any*
    ``a1``. If the relative distance were mis-defined -- absolute instead of
    radius-scaled, or squared -- this would land somewhere else.
    """
    pos = np.zeros((1, 3))
    for r_ang in (1.30, 1.72, 2.00):
        rad = np.array([r_ang * A])
        on = np.array([[r_ang * A, 0.0, 0.0], [0.0, -r_ang * A, 0.0]])
        np.testing.assert_allclose(
            pseudo_density(on, pos, rad), 1.0, rtol=1e-13
        )
        # Strictly inside and outside, monotone in radius.
        assert pseudo_density(np.array([[0.9 * r_ang * A, 0, 0]]), pos, rad)[0] > 1.0
        assert pseudo_density(np.array([[1.1 * r_ang * A, 0, 0]]), pos, rad)[0] < 1.0


def test_pair_term_is_largest_between_two_atoms():
    """The bond contribution is what fills the crevice.

    ``(1 - tau_a . tau_b)`` peaks where the two relative distance vectors are
    antiparallel, which is between the atoms. Compared at equal distance from
    both atoms, so only the angular factor differs.
    """
    d = 1.54 * A
    r = 2.00 * A
    pos = np.array([[0.0, 0.0, 0.0], [d, 0.0, 0.0]])
    rad = np.array([r, r])
    single = np.array([[0.0, 0.0, 0.0]])

    def pair_only(p):
        both = pseudo_density(p, pos, rad)
        a = pseudo_density(p, pos[:1], rad[:1])
        b = pseudo_density(p, pos[1:], rad[1:])
        return both - a - b

    between = np.array([[d / 2, 0.0, 0.0]])
    # Same distance from each atom, but off the internuclear axis: the
    # relative distance vectors are no longer antiparallel.
    half = d / 2
    off = np.array([[half, math.sqrt(max(half * half * 3.0, 0.0)), 0.0]])
    assert pair_only(between)[0] > pair_only(off)[0] > 0.0


def test_fine_cavity_extends_beyond_the_union_of_spheres():
    """The whole point of the construction, as a numerical statement.

    A union of atomic spheres has a sharp crease where two spheres meet, and
    the older cavity leaves the concave region there unpaved. The pair term
    pushes the iso-surface outside both spheres across a bond, so there exist
    points that are outside every van der Waals sphere yet inside the CFC.
    """
    d = 1.54 * A
    r = 2.00 * A
    pos = np.array([[0.0, 0.0, 0.0], [d, 0.0, 0.0]])
    rad = np.array([r, r])
    found = False
    for rho_ang in np.linspace(1.80, 1.95, 16):
        p = np.array([[d / 2, rho_ang * A, 0.0]])
        tau_min = min(
            np.linalg.norm(p[0] - pos[i]) / rad[i] for i in range(2)
        )
        if tau_min > 1.0 and pseudo_density(p, pos, rad)[0] > 1.0:
            found = True
            break
    assert found, "the pair term did not inflate the surface past the spheres"


def test_pseudo_density_decays_to_zero_far_away():
    pos, _, rad = _water()
    far = np.array([[40.0 * A, 0.0, 0.0]])
    assert pseudo_density(far, pos, rad)[0] < 1e-8


def test_log_pseudo_density_is_finite_and_signed_across_the_surface():
    """``ln PD`` is the field the triangulation cuts, so its sign must be the
    inside/outside test and it must never be NaN."""
    pos, _, rad = _water()
    inside = np.array([[0.0, 0.0, 0.0]])
    outside = np.array([[30.0 * A, 0.0, 0.0]])
    assert log_pseudo_density(inside, pos, rad)[0] > 0.0
    v = log_pseudo_density(outside, pos, rad)[0]
    assert np.isfinite(v) and v < 0.0


def test_nearest_atom_truncation_matches_the_full_sum_when_inactive():
    """With five atoms or fewer the truncation must be a no-op."""
    pos, _, rad = _water()
    grid = pos.mean(axis=0) + np.random.default_rng(0).normal(size=(50, 3))
    full = pseudo_density(grid, pos, rad, PseudoDensityParams(n_nearest=None))
    trunc = pseudo_density(grid, pos, rad, PseudoDensityParams(n_nearest=5))
    np.testing.assert_allclose(full, trunc, rtol=1e-13)


def test_pseudo_density_rejects_inconsistent_input():
    pos = np.zeros((2, 3))
    with pytest.raises(ValueError, match="radii for"):
        pseudo_density(np.zeros((1, 3)), pos, np.array([1.0]))
    with pytest.raises(ValueError, match="must be positive"):
        pseudo_density(np.zeros((1, 3)), pos, np.array([1.0, 0.0]))


# ---------------------------------------------------------------------
# Marching tetrahedra (step 3)
# ---------------------------------------------------------------------


def _sphere_field(radius):
    return lambda p: np.linalg.norm(p, axis=1) - radius


@pytest.mark.parametrize("spacing", [0.4, 0.2, 0.1])
def test_marching_tetrahedra_closes_the_surface(spacing):
    """Every edge shared by exactly two triangles.

    The paper requires the inscribed tetrahedron's two mirror images to
    alternate between neighbouring cubes so they share a face diagonal.
    Without that the triangulation cracks, and a cavity with a hole leaks
    field rather than screening it.
    """
    s = marching_tetrahedra(
        _sphere_field(2.0), np.full(3, -3.0), np.full(3, 3.0), spacing
    )
    assert s.n_triangles > 0
    assert s.is_closed()


def test_marching_tetrahedra_converges_on_a_sphere():
    """Area and volume both converge, which needs consistent winding.

    Area is a norm and survives mixed winding; volume is signed and cancels to
    near zero if the triangles are not consistently oriented. Checking both is
    what makes this a test of the orientation pass and not just of the cuts.
    """
    r = 2.0
    exact_a = 4.0 * math.pi * r * r
    exact_v = 4.0 / 3.0 * math.pi * r ** 3
    errs_a, errs_v = [], []
    for h in (0.4, 0.2, 0.1):
        s = marching_tetrahedra(
            _sphere_field(r), np.full(3, -3.0), np.full(3, 3.0), h
        )
        errs_a.append(abs(s.area / exact_a - 1.0))
        errs_v.append(abs(s.enclosed_volume() / exact_v - 1.0))
    # Monotone improvement, and better than 0.1 % at the finest spacing.
    assert errs_a[0] > errs_a[1] > errs_a[2]
    assert errs_v[0] > errs_v[1] > errs_v[2]
    assert errs_a[-1] < 1e-3
    assert errs_v[-1] < 1e-3


def test_marching_tetrahedra_volume_is_not_cancelled_by_mixed_winding():
    """Guard the specific failure this cost a debugging cycle: before the
    orientation pass the sphere volume came out as 0.0000 while the area was
    already correct to 0.5 %."""
    s = marching_tetrahedra(
        _sphere_field(2.0), np.full(3, -3.0), np.full(3, 3.0), 0.2
    )
    v = s.enclosed_volume()
    assert v > 0.5 * (4.0 / 3.0 * math.pi * 8.0)


def test_marching_tetrahedra_returns_empty_when_the_box_misses_the_surface():
    s = marching_tetrahedra(
        _sphere_field(2.0), np.full(3, 10.0), np.full(3, 12.0), 0.2
    )
    assert s.n_triangles == 0


def test_quadratic_and_linear_edge_roots_agree_closely():
    """Quadratic interpolation is the paper's choice; on a near-linear field
    it should refine the linear answer, not change it."""
    r = 2.0
    kw = dict(lower=np.full(3, -3.0), upper=np.full(3, 3.0), spacing=0.2)
    lin = marching_tetrahedra(_sphere_field(r), quadratic=False, **kw)
    quad = marching_tetrahedra(_sphere_field(r), quadratic=True, **kw)
    exact = 4.0 * math.pi * r * r
    assert abs(quad.area / exact - 1.0) < 5e-3
    assert abs(lin.area / exact - 1.0) < 5e-3


# ---------------------------------------------------------------------
# The assembled cavity (steps 4-6)
# ---------------------------------------------------------------------


def test_water_fine_cavity_is_well_formed():
    pos, z, rad = _water()
    cav = build_fine_cavity(pos, z, rad)
    assert isinstance(cav, FineCavity)
    assert cav.n_points > 100
    assert cav.surface.is_closed()
    assert np.all(cav.weights > 0.0)
    np.testing.assert_allclose(np.linalg.norm(cav.normals, axis=1), 1.0, atol=1e-10)
    assert set(np.unique(cav.point_atom)) == {0, 1, 2}


def test_coarsening_conserves_the_triangulated_area():
    """Every basis point carries real surface, so segment areas must sum to
    the triangulated area exactly. Sub-threshold points are *joined*, not
    dropped -- dropping them would shrink the cavity instead of coarsening it.
    """
    pos, z, rad = _water()
    cav = build_fine_cavity(pos, z, rad)
    # Not exact since #757: the smooth Becke partition gives *every* segment
    # direction some weight, so a floor on segment area is needed where the
    # hard assignment simply left unused directions empty. The loss is
    # bounded by that floor times the number of dropped directions, and
    # asserting the bound rather than a loosened tolerance keeps the test
    # about area conservation instead of about a magic number.
    lost = cav.surface.area - cav.total_area
    n_dropped = sum(int((~r.keep).sum()) for r in cav.coarsening)
    assert abs(lost) <= n_dropped * MIN_SEGMENT_AREA_BOHR2 + 1e-12
    # The floor drops what it drops, and #771's smooth step 5 gives every atom
    # a longer tail of small segments, so slightly more falls under it: measured
    # 6.3e-07 at 0.40 A, 7.1e-07 at 0.30 and 1.8e-06 at 0.25. The bound above is
    # the one that matters -- it is the floor's own arithmetic, not a tolerance
    # -- and this second one only keeps the total honest.
    assert abs(lost) / cav.surface.area < 5e-6


def test_segments_lie_on_or_outside_their_atom_sphere():
    """Step 6: "Segment centers with atom center distances smaller than the
    COSMO radius of the atom are moved outward to the sphere."

    An area-weighted mean of points on a curved surface always falls inside
    it, so without that rule every segment would sit just under the cavity --
    and a screening charge inside the surface it is meant to screen is a
    physical error, not a rounding one.
    """
    pos, z, rad = _water()
    cav = build_fine_cavity(pos, z, rad)
    d = np.linalg.norm(cav.points - cav.atom_positions[cav.point_atom], axis=1)
    assert np.min(d / cav.atom_radii[cav.point_atom]) >= 1.0 - 1e-12


def test_segment_counts_are_chosen_for_equal_paving_density():
    """NSPH = 92 and NSPA = 162 give hydrogen and heavy atoms the *same*
    segment density, not a coarser one for hydrogen.

    The two magic numbers sit in almost exactly the ratio of the sphere areas
    they pave: ``162/92 = 1.761`` against ``(1.72/1.30)^2 = 1.750``, a 0.6 %
    difference. That is the design intent of the CFC -- homogeneous paving --
    and it is worth pinning, because the naive reading (hydrogen gets fewer
    points, so hydrogen is paved more coarsely) is wrong and would licence
    "fixing" the constants in the wrong direction.
    """
    r_h, r_o = KLAMT_RADII_ANG[1], KLAMT_RADII_ANG[8]
    assert (NSPA_FINE / NSPH_FINE) == pytest.approx((r_o / r_h) ** 2, rel=0.01)

    # The geometry is mirror-symmetric and axis-aligned, which used to be
    # required: with lab-fixed segment directions (#769) the two equivalent
    # hydrogens paved differently in any other orientation -- 1.72 against
    # 1.60 -- and this test measured that defect instead of the NSPH/NSPA ratio
    # it is about. The molecule-fixed frame removed the need; the geometry is
    # kept so the numbers below stay comparable with the ones recorded then.
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 1.498, -1.159], [0.0, -1.498, -1.159]])
    z = [8, 1, 1]
    rad = np.array([KLAMT_RADII_ANG[8], KLAMT_RADII_ANG[1], KLAMT_RADII_ANG[1]]) * A
    cav = build_fine_cavity(pos, z, rad)
    per_atom_area = np.zeros(3)
    np.add.at(per_atom_area, cav.point_atom, cav.weights)
    counts = np.bincount(cav.point_atom, minlength=3)
    # Count only segments carrying real area. Since #757 every direction gets
    # some Becke weight, so the raw count measures the geodesic grid rather
    # than the paving: the question here is whether NSPH and NSPA pave at
    # equal density, which is about segments that hold surface.
    # The bound is looser than the 1.15 this started with, and #771 is why. A
    # basis point now feeds more than one atom, so a hydrogen picks up a thin
    # share of the surface near the oxygen, spread over all 92 of its
    # directions: its live count rises faster than its area does. The measured
    # ratio moves with the threshold that defines "live" -- 1.21 at 1e-03, 1.15
    # at 1e-02, 1.13 at 5e-02 -- which is the signature of that tail rather than
    # of unequal paving. What this test really guards is asserted above and is
    # unchanged: a wrong pair of constants gives 1.75, not 1.21.
    live = cav.weights > 1e-3
    counts = np.bincount(cav.point_atom[live], minlength=3)
    density = counts / per_atom_area
    assert density.max() / density.min() < 1.25


@pytest.mark.parametrize("spacing", [0.4, 0.3, 0.2])
def test_cavity_area_and_volume_converge_in_the_grid_spacing(spacing):
    """Areas must not depend strongly on delta, or the sigma profile the
    cavity feeds would be an artefact of the grid."""
    pos, z, rad = _water()
    cav = build_fine_cavity(pos, z, rad, grid_spacing_ang=spacing)
    area_ang2 = cav.total_area / A ** 2
    vol_ang3 = cav.surface.enclosed_volume() / A ** 3
    # Measured 42.80-43.05 A^2 and 25.55-25.85 A^3 over delta = 0.4 to 0.15 A.
    assert 42.0 < area_ang2 < 44.0
    assert 25.0 < vol_ang3 < 26.5
    assert cav.surface.is_closed()


def test_cavity_is_stable_under_rigid_rotation():
    """A cavity area that depends on molecular orientation is a grid artefact.

    It used to be one: the marching lattice was axis-aligned, so this held only
    to 2e-02 and was written with that tolerance. Since #769 the construction
    runs in the solute's own frame and it holds exactly, so the tolerance is
    machine precision -- and stating it that way is what would catch a
    regression of #769 in a test that is not about the frame.
    """
    pos, z, rad = _water()
    a = build_fine_cavity(pos, z, rad, grid_spacing_ang=0.25)
    ca, sa = math.cos(0.7), math.sin(0.7)
    rot = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    b = build_fine_cavity(pos @ rot.T, z, rad, grid_spacing_ang=0.25)
    assert b.total_area == pytest.approx(a.total_area, rel=1e-13)
    assert b.surface.enclosed_volume() == pytest.approx(
        a.surface.enclosed_volume(), rel=1e-13
    )


# ---------------------------------------------------------------------
# The molecule-fixed frame (workflow step 1, #769)
# ---------------------------------------------------------------------


def _rotation(axis, angle_deg):
    ax = np.asarray(axis, dtype=float)
    ax = ax / np.linalg.norm(ax)
    ang = math.radians(angle_deg)
    K = np.array([[0.0, -ax[2], ax[1]], [ax[2], 0.0, -ax[0]], [-ax[1], ax[0], 0.0]])
    return np.eye(3) + math.sin(ang) * K + (1.0 - math.cos(ang)) * (K @ K)


# A generic axis and a set of angles with nothing special about them: an
# orientation dependence that happened to vanish for a rotation about a
# molecular symmetry axis would not be the thing #769 is about.
_AXIS = (0.31, -0.87, 0.39)
_ANGLES = (0.0, 1.0, 13.0, 47.0, 90.0)


def test_molecular_frame_is_the_papers_step_1():
    """Each axis is the one step 1 names, on a geometry simple enough to read.

    Four atoms placed so that the farthest from the centroid, and the one with
    the largest component orthogonal to that direction, are different atoms and
    are both unambiguous -- otherwise the test would pass on a frame that got
    the selection rule wrong.
    """
    pos = np.array([
        [0.0, 0.0, 0.0],
        [6.0, 0.0, 0.0],       # farthest from the centroid -> x
        [0.0, 2.5, 0.0],       # largest orthogonal component -> y
        [1.0, 0.5, 0.25],
    ])
    R = molecular_frame(pos)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-14)

    d = pos - pos.mean(axis=0)
    far = int(np.argmax(np.linalg.norm(d, axis=1)))
    assert far == 1
    np.testing.assert_allclose(R[0], d[far] / np.linalg.norm(d[far]), atol=1e-14)
    orth = d - np.outer(d @ R[0], R[0])
    wide = int(np.argmax(np.linalg.norm(orth, axis=1)))
    assert wide == 2
    np.testing.assert_allclose(
        R[1], orth[wide] / np.linalg.norm(orth[wide]), atol=1e-14
    )
    np.testing.assert_allclose(R[2], np.cross(R[0], R[1]), atol=1e-14)


@pytest.mark.parametrize("angle", _ANGLES)
def test_molecular_frame_is_rotation_equivariant(angle):
    """``frame(Q x) == frame(x) Q^T`` -- the property the whole fix rests on.

    Nothing else about the frame matters for #769: the construction downstream
    is deterministic in the molecular coordinates, so if the axes turn exactly
    with the solute then so does the cavity.
    """
    pos, z, _ = _water()
    Q = _rotation(_AXIS, angle)
    np.testing.assert_allclose(
        molecular_frame(pos @ Q.T), molecular_frame(pos) @ Q.T, atol=1e-13
    )
    # And on a solute with no symmetry at all, so the equivalence is not an
    # artefact of water's.
    asym = np.array([[0.0, 0.0, 0.0], [2.61, 0.31, -0.12],
                     [-0.63, 1.55, 0.44], [3.19, -0.47, 1.71]])
    np.testing.assert_allclose(
        molecular_frame(asym @ Q.T), molecular_frame(asym) @ Q.T, atol=1e-13
    )


def test_an_exact_tie_is_broken_by_atom_index_not_by_rounding():
    """Water's hydrogens are exactly equidistant from the centroid.

    A bare ``argmax`` over the distances is then decided by the last bits of
    two numbers that are equal in exact arithmetic, and rotating the solute
    changes those bits: measured, the x-axis flipped between the two hydrogens
    from one orientation to the next, which cost the frame its equivariance
    (error 2.0 -- the axes were antiparallel) even though every distance was
    unchanged. The tie goes to the lower atom index, which no rotation can
    touch.
    """
    pos, z, _ = _water()
    d = pos - pos.mean(axis=0)
    r = np.linalg.norm(d, axis=1)
    assert r[1] == pytest.approx(r[2], rel=1e-15), "the tie this test is about"

    R = molecular_frame(pos)
    np.testing.assert_allclose(R[0], d[1] / r[1], atol=1e-14)   # the lower index
    for angle in _ANGLES:
        Q = _rotation(_AXIS, angle)
        rot_pos = pos @ Q.T
        rot_d = rot_pos - rot_pos.mean(axis=0)
        picked = molecular_frame(rot_pos)[0] @ Q       # back into the lab frame
        np.testing.assert_allclose(picked, d[1] / r[1], atol=1e-13)
        # The rounding a bare argmax would have read really is present.
        assert np.linalg.norm(rot_d[1]) != np.linalg.norm(rot_d[2]) or angle == 0.0


def test_molecular_frame_when_the_solute_determines_no_frame():
    """One atom, and a linear solute: both leave step 1 with axes to spare.

    A single centre is isotropic, so its frame cannot matter and the identity is
    returned. A linear solute fixes only the x-axis, and no rotation-equivariant
    choice of the other two exists -- the rotations about x fix x and would have
    to fix y as well -- so the azimuth is completed deterministically and stays
    orientation-dependent. What is guaranteed is that the frame is a proper
    orthonormal rotation, which is what the construction downstream needs.
    """
    np.testing.assert_array_equal(molecular_frame(np.zeros((1, 3))), np.eye(3))
    np.testing.assert_array_equal(
        molecular_frame(np.full((3, 3), 1.7)), np.eye(3)
    )
    linear = np.array([[0.0, 0.0, -2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 2.2]])
    R = molecular_frame(linear)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-14)
    np.testing.assert_allclose(np.abs(R[0]), [0.0, 0.0, 1.0], atol=1e-14)


def test_molecular_frame_rejects_bad_input():
    with pytest.raises(ValueError, match=r"positions must be"):
        molecular_frame(np.zeros(3))
    with pytest.raises(ValueError, match="no atoms"):
        molecular_frame(np.zeros((0, 3)))


@pytest.mark.parametrize("angle", _ANGLES[1:])
def test_the_molecular_frame_makes_the_cavity_equivariant(angle):
    """#769: in the molecule-fixed frame the cavity *rotates with* the solute.

    The strong statement, not an area agreeing to a tolerance: the same number
    of segments, each in the same place and carrying the same area once the
    rotation is undone. Measured at 0.40 A on water, over rotations of 1 to 90
    degrees about a generic axis: positions to 1.3e-14 bohr, areas to 2.2e-14
    bohr^2, and a total area that does not move in any digit.

    The lab-frame contrast is asserted too, so the test cannot pass because the
    cavity stopped depending on the geometry at all: there the segment count
    swings between 235 and 239 and the total area over 9.8e-02 bohr^2 (6.4e-04
    relative), which is the defect being fixed.
    """
    pos, z, rad = _water()
    Q = _rotation(_AXIS, angle)

    ref = build_fine_cavity(pos, z, rad, grid_spacing_ang=0.40, frame="molecular")
    rot = build_fine_cavity(pos @ Q.T, z, rad, grid_spacing_ang=0.40,
                            frame="molecular")
    assert rot.n_points == ref.n_points
    np.testing.assert_allclose(rot.points @ Q, ref.points, atol=1e-12)
    np.testing.assert_allclose(rot.weights, ref.weights, atol=1e-12)
    np.testing.assert_allclose(rot.normals @ Q, ref.normals, atol=1e-12)
    assert rot.total_area == pytest.approx(ref.total_area, rel=1e-14)
    assert rot.surface.enclosed_volume() == pytest.approx(
        ref.surface.enclosed_volume(), rel=1e-14
    )
    # The atoms come back where they were given, not in the working frame.
    np.testing.assert_allclose(ref.atom_positions, pos, atol=1e-13)
    # Segments still sit on or outside their sphere after the round trip.
    dist = np.linalg.norm(ref.points - ref.atom_positions[ref.point_atom], axis=1)
    assert np.min(dist / ref.atom_radii[ref.point_atom]) >= 1.0 - 1e-12

    lab_ref = build_fine_cavity(pos, z, rad, grid_spacing_ang=0.40, frame="lab")
    lab_rot = build_fine_cavity(pos @ Q.T, z, rad, grid_spacing_ang=0.40,
                                frame="lab")
    assert abs(lab_rot.total_area - lab_ref.total_area) > 1e-5 * lab_ref.total_area


def test_the_molecular_frame_is_the_default_and_lab_is_still_reachable():
    """Step 1 is part of the construction, not an option on it.

    The default was ``"lab"`` while the frame had no nuclear derivative. It has
    one, so the CFC is now built the way the paper describes, and the record
    carries the frame it used. ``"lab"`` stays reachable: it is the pre-#769
    behaviour, and the right choice for a linear solute, whose molecule-fixed
    frame has no derivative at all.
    """
    pos, z, rad = _water()
    implicit = build_fine_cavity(pos, z, rad, grid_spacing_ang=0.40)
    explicit = build_fine_cavity(pos, z, rad, grid_spacing_ang=0.40,
                                 frame="molecular")
    np.testing.assert_array_equal(implicit.points, explicit.points)
    np.testing.assert_array_equal(implicit.weights, explicit.weights)
    assert implicit.frame.shape == (3, 3)
    np.testing.assert_allclose(implicit.frame, molecular_frame(pos), atol=1e-14)
    assert implicit.surface.frame.shape == (3, 3)

    lab = build_fine_cavity(pos, z, rad, grid_spacing_ang=0.40, frame="lab")
    assert lab.frame.size == 0
    assert lab.surface.frame.size == 0
    # Not the same cavity: the two discretize the same surface differently,
    # which is the whole content of #769.
    assert lab.total_area != implicit.total_area


def test_build_fine_cavity_rejects_an_unknown_frame():
    pos, z, rad = _water()
    with pytest.raises(ValueError, match="frame must be 'lab' or 'molecular'"):
        build_fine_cavity(pos, z, rad, frame="principal")


def test_the_inner_chain_still_refuses_a_frame_it_cannot_differentiate():
    """The composition is the only route; the stages themselves stay honest.

    :class:`FineCavityDerivative` and :func:`iso_point_jacobians` differentiate
    a marching lattice and a set of segment directions that are *constants*, so
    they refuse a cavity whose frame turns with the solute rather than dropping
    that contribution -- which is not a small one: it is 2.3e+01 of a gradient
    whose largest component is 2.9e+01. Reaching the chain the right way, with
    the cavity reversed into the frame it was built in, is
    :class:`FrameCavityDerivative`'s job and the dispatcher's.
    """
    from vibeqc.solvation.cavity_derivative import (
        FineCavityDerivative,
        FrameCavityDerivative,
        cavity_derivative,
    )
    from vibeqc.solvation.fine_cavity import to_construction_frame
    from vibeqc.solvation.fine_cavity_gradient import iso_point_jacobians

    pos, z, rad = _water()
    cav = build_fine_cavity(pos, z, rad, grid_spacing_ang=0.45, frame="molecular")
    with pytest.raises(NotImplementedError, match=r"#769.*cpcm_gradient_fd"):
        FineCavityDerivative(cav)
    with pytest.raises(NotImplementedError, match="molecule-fixed frame"):
        iso_point_jacobians(cav.surface, cav.atom_positions, cav.atom_radii)

    # The dispatcher routes on the frame, not on a flag.
    assert isinstance(cavity_derivative(cav), FrameCavityDerivative)
    lab = build_fine_cavity(pos, z, rad, grid_spacing_ang=0.45, frame="lab")
    assert isinstance(cavity_derivative(lab), FineCavityDerivative)
    with pytest.raises(ValueError, match=r"expected a \(3, 3\) frame"):
        FrameCavityDerivative(lab)

    # And the reversal really does recover the construction-frame record: the
    # same cavity a direct build on molecular coordinates would have produced.
    # ``frame="lab"`` there because that is precisely what the construction did
    # -- step 1 once, then the ordinary workflow on the rotated coordinates.
    origin = pos.mean(axis=0)
    direct = build_fine_cavity(
        (pos - origin) @ cav.frame.T, z, rad, grid_spacing_ang=0.45, frame="lab"
    )
    back = to_construction_frame(cav)
    np.testing.assert_allclose(back.points, direct.points, atol=1e-12)
    np.testing.assert_array_equal(back.weights, direct.weights)
    assert back.frame.size == 0 and back.surface.frame.size == 0


def test_molecular_frame_jacobian_satisfies_its_exact_invariants():
    """Translation and rotation, neither of which needs a displaced geometry.

    The frame does not move when the solute is translated, so
    ``sum_A d(e_i)/d(R_A) = 0``. It turns rigidly when the solute is rotated, so
    contracting with the rigid-rotation field ``omega x R_A`` gives
    ``omega x e_i``. Both hold to machine precision; finite differences are
    carried as a third, weaker check because they are the only one that would
    catch a sign error common to every axis.
    """
    asym = np.array([[0.0, 0.0, 0.0], [2.61, 0.31, -0.12],
                     [-0.63, 1.55, 0.44], [3.19, -0.47, 1.71]])
    water, _, _ = _water()
    for pos in (asym, water):
        jac = molecular_frame_jacobian(pos)
        frame = molecular_frame(pos)
        assert jac.shape == (3, 3, pos.shape[0], 3)
        assert np.abs(jac.sum(axis=2)).max() < 1e-13

        for omega in (np.array([0.3, -0.7, 0.2]), np.array([1.0, 0.0, 0.0])):
            field = np.cross(omega[None, :], pos)
            got = np.einsum("ijAc,Ac->ij", jac, field)
            np.testing.assert_allclose(
                got, np.cross(omega[None, :], frame), atol=1e-13
            )

    # Finite differences only on the asymmetric solute. Water's equilibrium
    # geometry sits *on* a tie -- its two hydrogens are exactly equidistant from
    # the centroid, and so are the orthogonal components that pick the y-axis --
    # so displacing an atom flips a selection and the frame jumps: FD reads
    # 1e+06 for an entry the analytic value puts at 3.5e-02. The frame really is
    # discontinuous there. The energy is not, which is what
    # ``test_the_frames_tie_is_invisible_in_the_energy`` measures.
    h = 1e-6
    jac = molecular_frame_jacobian(asym)
    for a in range(asym.shape[0]):
        for c in range(3):
            up, dn = asym.copy(), asym.copy()
            up[a, c] += h
            dn[a, c] -= h
            fd = (molecular_frame(up) - molecular_frame(dn)) / (2.0 * h)
            np.testing.assert_allclose(jac[:, :, a, c], fd, atol=1e-8)


def test_molecular_frame_jacobian_refuses_a_frame_that_has_no_derivative():
    """A linear solute's azimuth is a discontinuous lab-derived choice.

    Displace any atom off the axis and step 1 leaves the completion branch for
    the geometric one, where the y-axis is the azimuth of that displacement --
    an arbitrary angle away. The within-branch derivative is easy to write and
    is contradicted by finite differences by 1e+06, with the rotation identity
    failing outright at 2.0e-01, so it is refused instead.
    """
    linear = np.array([[0.0, 0.0, -2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 2.2]])
    with pytest.raises(NotImplementedError, match="linear"):
        molecular_frame_jacobian(linear)
    with pytest.raises(NotImplementedError, match="every atom sits at the centre"):
        molecular_frame_jacobian(np.zeros((3, 3)))
    # A single atom is not the same case: its frame is the identity for every
    # geometry it can have, so the derivative is genuinely zero.
    np.testing.assert_array_equal(
        molecular_frame_jacobian(np.zeros((1, 3))), np.zeros((3, 3, 1, 3))
    )
    # Nearly linear is stiff, not refused, and still satisfies the invariants.
    near = np.array([[0.0, 0.0, -2.0], [1e-3, 0.0, 0.0], [0.0, 0.0, 2.2]])
    jac = molecular_frame_jacobian(near)
    assert np.abs(jac).max() > 1e2
    assert np.abs(jac.sum(axis=2)).max() < 1e-11


def test_the_frame_term_is_what_makes_the_torque_vanish():
    """The exact oracle for the composition, and the negative control with it.

    A cavity built in a molecule-fixed frame is a function of the internal
    geometry alone, so a rigid rotation moves each segment by ``omega x p_S``
    and moves no area at all. Contracting the reverse pass against that field
    must reproduce it exactly -- no displaced geometries, no tolerance.

    Zeroing only the frame Jacobian is the control that makes the test mean
    something. Translation is *unaffected* by it (1.8e-15 either way), so the
    invariant that guarded #729 would not have caught this at all; the rotation
    residual goes from 1.2e-16 to 1.7e+00.
    """
    from vibeqc.solvation.cavity_derivative import cavity_derivative

    pos = np.array([[0.0, 0.0, 0.0], [2.61, 0.31, -0.12],
                    [-0.63, 1.55, 0.44], [3.19, -0.47, 1.71]])
    z = [8, 6, 1, 1]
    from vibeqc.solvation.cavity import atom_radii_bohr
    rad = atom_radii_bohr(np.array(z), None, 1.2, 0.0)
    cav = build_fine_cavity(pos, z, rad, grid_spacing_ang=0.45, frame="molecular")

    deriv = cavity_derivative(cav)
    trans = deriv.translation_residuals()
    rot = deriv.rotation_residuals()
    assert max(trans) < 1e-11, trans
    assert max(rot) < 1e-11, rot

    rng = np.random.default_rng(3)
    adj = rng.normal(size=(cav.n_points, 3))
    full = deriv.contract(adj, None)
    deriv._frame_jac = np.zeros_like(deriv._frame_jac)
    assert max(deriv.translation_residuals()) < 1e-11   # blind to the frame
    assert max(deriv.rotation_residuals()) > 1e-2       # not blind to it
    dropped = deriv.contract(adj, None)
    assert np.abs(full - dropped).max() > 0.5 * np.abs(full).max()


def test_single_atom_cavity_reproduces_the_sphere():
    """The one case with an exact answer end to end."""
    r_ang = 1.72
    rad = np.array([r_ang * A])
    cav = build_fine_cavity(np.zeros((1, 3)), [8], rad, grid_spacing_ang=0.15)
    exact = 4.0 * math.pi * (r_ang * A) ** 2
    assert cav.total_area == pytest.approx(exact, rel=5e-3)
    d = np.linalg.norm(cav.points, axis=1)
    np.testing.assert_allclose(d, r_ang * A, rtol=2e-2)


def test_projection_radius_eases_the_hard_rule(monkeypatch):
    """Step 6's outward projection, as a function (#770).

    The paper pushes a segment centre onto the sphere when its area-weighted
    mean falls inside. A hard maximum is C0 but not C1, so the segment
    position's slope jumps where a centre crosses. This eases it over a band,
    and the three properties that makes acceptable are checked here: it never
    puts a segment *inside* the sphere, it leaves the far field alone, and it is
    the paper's rule in the limit of a vanishing band.
    """
    from vibeqc.solvation import fine_cavity as fc

    rho = 3.0
    dist = np.array([1.0, 2.9, 2.999, 3.0, 3.001, 3.1, 5.0, 50.0])
    r_eff, sigma = projection_radius(dist, rho)

    # Never inside: that is the physical error the projection exists to stop.
    assert np.all(r_eff >= rho)
    # On the sphere well inside, and untouched well outside.
    assert r_eff[0] == pytest.approx(rho, abs=1e-14)
    assert r_eff[-1] == pytest.approx(50.0, abs=1e-14)
    # Monotone ramp, half-way across at the sphere.
    assert sigma[0] == pytest.approx(0.0, abs=1e-12)
    assert sigma[3] == pytest.approx(0.5, abs=1e-12)
    assert sigma[-1] == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.diff(sigma) >= 0.0)          # saturates at both ends
    assert sigma[2] < sigma[3] < sigma[4]         # and is strict across the band

    h = 1e-8
    fd = (projection_radius(dist + h, rho)[0]
          - projection_radius(dist - h, rho)[0]) / (2.0 * h)
    np.testing.assert_allclose(sigma, fd, atol=1e-7)

    # The band is narrow against what it eases: a sixth of the median
    # projection depth (0.019 bohr against radii of 2.7 to 3.9), and it spans a
    # typical finite-difference step, which is the consumer that needs it.
    assert PROJECTION_SOFTNESS * rho == pytest.approx(3.0e-3, rel=1e-12)

    # And it is the paper's rule in the limit.
    monkeypatch.setattr(fc, "PROJECTION_SOFTNESS", 1e-12)
    np.testing.assert_allclose(
        projection_radius(dist, rho)[0], np.maximum(dist, rho), atol=1e-10
    )


def test_the_projection_no_longer_kinks_where_a_segment_crosses_its_sphere(
    monkeypatch,
):
    """#770 fixed, with the hard rule as the control.

    Bracketing a displacement at which a segment centre crosses its own sphere,
    and sampling an area-weighted functional either side of it at 1e-06 bohr:
    with the hard maximum the slope wanders by 1.8e-03 relative across the
    crossing's neighbourhood, and with the band it wanders by 2.6e-06. On the
    asymmetric solute the same measurement isolates a clean slope *jump*, 0.071
    percent with the hard rule and 0.000 percent with the band.

    The control matters here more than usual: the crossing is bracketed to
    1e-10 bohr, so a test that only sampled the eased construction would pass
    whether or not the easing did anything.
    """
    from vibeqc.solvation import fine_cavity as fc

    # The geometry the crossing below was bracketed on: _water()'s is a
    # different one and puts its crossings elsewhere.
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 1.498, -1.159],
                    [0.0, -1.498, -1.159]])
    z = [8, 1, 1]
    rad = np.array([KLAMT_RADII_ANG[e] * A for e in z])
    v = np.array([0.31, -0.87, 0.39])
    v = v / np.linalg.norm(v)

    def cavity(t):
        p = pos.copy()
        p[0, 2] += t
        return build_fine_cavity(p, z, rad, grid_spacing_ang=0.45)

    def functional(t):
        c = cavity(t)
        return float(np.dot(c.weights, c.points @ v))

    lo, hi = -0.0325, -0.0300
    n_lo = int(cavity(lo).projected.sum())
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        if int(cavity(mid).projected.sum()) == n_lo:
            lo = mid
        else:
            hi = mid
    t_c = 0.5 * (lo + hi)
    assert int(cavity(hi).projected.sum()) != n_lo, "no crossing was bracketed"

    offsets = np.array([-3e-6, -2e-6, -1e-6, 1e-6, 2e-6, 3e-6])

    def slope_spread():
        e = np.array([functional(t_c + d) for d in offsets])
        secants = np.diff(e) / np.diff(offsets)
        return float((secants.max() - secants.min()) / abs(secants.mean()))

    eased = slope_spread()
    monkeypatch.setattr(fc, "PROJECTION_SOFTNESS", 1e-12)
    hard = slope_spread()
    assert eased < 1e-4, eased
    assert hard > 20 * eased, (hard, eased)


def test_quad_groups_pairs_the_tetragon_halves():
    """The grouping step 4 needs (#779).

    A tetrahedron with two corners inside the surface and two outside cuts four
    edges and emits a tetragon as two triangles; every other case emits one
    triangle. The grouping has to survive welding, which renumbers vertices and
    drops triangles that collapsed onto a line.
    """
    pos, z, rad = _water()
    surface = build_fine_cavity(pos, z, rad, grid_spacing_ang=0.45).surface
    lone, pairs, corners = quad_groups(surface)

    # Every triangle is accounted for exactly once.
    assert int(lone.sum()) + 2 * pairs.shape[0] == surface.n_triangles
    assert pairs.shape[0] > 0, "this surface has no tetragons to test"
    assert len(set(pairs.ravel().tolist())) == pairs.size, "a triangle in two quads"
    assert not lone[pairs.ravel()].any(), "a quad half also counted as lone"

    # A tetragon is four distinct corners, and its two triangles share exactly
    # the two that form the diagonal.
    assert corners.shape == (pairs.shape[0], 4)
    for (i, j), quad in zip(pairs, corners):
        a, b = set(surface.triangles[i].tolist()), set(surface.triangles[j].tolist())
        assert len(a | b) == 4
        assert len(a & b) == 2                    # the shared diagonal
        assert set(quad.tolist()) == a | b

    # Roughly two fifths of a real triangulation is tetragon halves, which is
    # why the diagonal matters at all.
    assert 0.2 < 2 * pairs.shape[0] / surface.n_triangles < 0.6


def test_the_tetragon_diagonal_does_not_move_area_between_corners():
    """#779 fixed, with the per-triangle rule as the control.

    Step 3 splits a tetragon on its shorter diagonal, which is a hard
    comparison and flips where the two diagonals are equal. The quad's total
    area survives that flip, but "a third of each triangle to each of its three
    corners" does not: the same total lands on the four corners differently,
    and each corner's basis area jumps by about 0.023 bohr^2.

    Assigning a quarter of the tetragon's area to each of its four corners is
    the same principle -- the paper's "equally assigned to the corner points"
    -- applied to the figure step 3 actually produced, and no diagonal can
    change it. Across a flip bracketed to 1e-9 bohr the shipped rule moves a
    basis area by 1e-06 where the per-triangle rule moves it by 2.3e-02.
    """
    pos, z, rad = _water()
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 1.498, -1.159], [0.0, -1.498, -1.159]])
    rad = np.array([KLAMT_RADII_ANG[e] * A for e in z])

    def surface_at(t):
        p = pos.copy()
        p[0, 2] += t
        return build_fine_cavity(p, z, rad, grid_spacing_ang=0.45).surface

    def per_triangle(s):
        """The rule this replaced: a third of each triangle to each corner."""
        out = np.zeros(s.vertices.shape[0])
        np.add.at(out, s.triangles.ravel(), np.repeat(s.triangle_areas() / 3.0, 3))
        return out

    def per_tetragon(s):
        """The shipped rule."""
        out = np.zeros(s.vertices.shape[0])
        lone, pairs, corners = quad_groups(s)
        area = s.triangle_areas()
        np.add.at(out, s.triangles[lone].ravel(), np.repeat(area[lone] / 3.0, 3))
        np.add.at(
            out, corners.ravel(), np.repeat(area[pairs].sum(axis=1) / 4.0, 4)
        )
        return out

    # Bracket the flip by watching the triangle list reconnect.
    lo, hi = -0.015456, -0.015449
    ref = surface_at(lo).triangles
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        if np.array_equal(surface_at(mid).triangles, ref):
            lo = mid
        else:
            hi = mid
    s_lo, s_hi = surface_at(lo), surface_at(hi)
    changed = int((s_lo.triangles != s_hi.triangles).any(axis=1).sum())
    assert changed > 0, "no diagonal flip was bracketed"
    assert s_lo.vertices.shape == s_hi.vertices.shape
    assert np.abs(s_lo.vertices - s_hi.vertices).max() < 1e-8, "vertices moved"

    old_jump = np.abs(per_triangle(s_lo) - per_triangle(s_hi)).max()
    new_jump = np.abs(per_tetragon(s_lo) - per_tetragon(s_hi)).max()
    assert old_jump > 1e-3, old_jump
    assert new_jump < 1e-4, new_jump
    assert new_jump < old_jump / 100.0

    # The total is untouched either way: it is the distribution that was wrong.
    assert per_triangle(s_lo).sum() == pytest.approx(
        per_tetragon(s_lo).sum(), rel=1e-12
    )


def test_switching_is_unity_for_a_fine_cavity():
    """The CFC has no switching function; the attribute exists so consumers
    written against the Lebedev cavity keep working, and is exactly 1 so
    nothing silently reads zeros."""
    pos, z, rad = _water()
    cav = build_fine_cavity(pos, z, rad)
    np.testing.assert_array_equal(cav.switching, np.ones(cav.n_points))


def test_build_fine_cavity_requires_consistent_input():
    pos, z, rad = _water()
    with pytest.raises(ValueError, match=r"positions must be"):
        build_fine_cavity(np.zeros(3), z, rad)
    with pytest.raises(ValueError, match="one\nentry per atom|one entry per atom"):
        build_fine_cavity(pos, [8, 1], rad)


# ---------------------------------------------------------------------
# Wiring into run_cpcm_scf
# ---------------------------------------------------------------------


def test_shared_radius_derivation_is_used_by_both_cavity_and_gradient():
    """One formula for the cavity sphere radius.

    A gradient built on radii that differ from the ones the energy's cavity
    used would be wrong in exactly the silent way #546 and #548 were, so the
    gradient delegates rather than re-deriving.
    """
    from vibeqc.solvation.cavity import atom_radii_bohr
    from vibeqc.solvation.gradient import _atom_radii_bohr

    z = [8, 1, 1, 6]
    for scale, probe, override in ((1.20, 0.0, None), (1.0, 1.385, {8: 1.72})):
        np.testing.assert_array_equal(
            _atom_radii_bohr(np.array(z), override, scale, probe),
            atom_radii_bohr(z, override, scale, probe),
        )
    # And it is the documented formula, not an accident.
    assert atom_radii_bohr([8], {8: 2.0}, 1.5, 0.5)[0] == pytest.approx(
        (2.0 * 1.5 + 0.5) * A, rel=1e-14
    )


def test_solvent_model_validates_the_cavity_kind():
    vq = pytest.importorskip("vibeqc")
    assert vq.SolventModel(epsilon=78.39).cavity == "lebedev"
    assert vq.SolventModel(epsilon=78.39, cavity="FINE").cavity == "fine"
    with pytest.raises(ValueError, match="must be 'lebedev' or 'fine'"):
        vq.SolventModel(epsilon=78.39, cavity="bogus")


def test_solvent_model_validates_the_fine_frame():
    vq = pytest.importorskip("vibeqc")
    assert vq.SolventModel(epsilon=78.39).fine_frame == "molecular"
    assert vq.SolventModel(epsilon=78.39, fine_frame="LAB").fine_frame == "lab"
    with pytest.raises(ValueError, match="must be 'lab' or 'molecular'"):
        vq.SolventModel(epsilon=78.39, fine_frame="principal")


def _scf_water(vq):
    return vq.Molecule(
        [vq.Atom(8, (0.0, 0.0, 0.0)),
         vq.Atom(1, (0.0, 1.498, -1.159)),
         vq.Atom(1, (0.0, -1.498, -1.159))], 0, 1)


def _scf_water_untied(vq):
    """The same water with one OH lengthened, so the step-1 selection is not tied.

    Symmetric water's hydrogens are exactly equidistant from the centroid, so
    the molecule-fixed frame's ``argmax`` sits on a tie and the frame there is
    *discontinuous* rather than merely steep -- which is a legitimate place for
    an analytic derivative and a finite difference to disagree, and no place to
    measure whether the chain rule is right. Lengthening one bond by 0.1 bohr
    removes the tie and leaves everything else about the test unchanged. What
    the tie itself costs is measured by
    ``test_the_gradient_is_ill_conditioned_near_a_frame_tie``.
    """
    return vq.Molecule(
        [vq.Atom(8, (0.0, 0.0, 0.0)),
         vq.Atom(1, (0.0, 1.598, -1.159)),
         vq.Atom(1, (0.0, -1.498, -1.159))], 0, 1)


def test_run_cpcm_scf_accepts_the_fine_cavity():
    vq = pytest.importorskip("vibeqc")
    mol = _scf_water(vq)
    basis = vq.BasisSet(mol, "sto-3g")
    sm = vq.SolventModel(
        epsilon=78.39, name="water", cavity="fine",
        fine_grid_spacing_ang=0.40, max_macro_iter=40, tol_e_solv=1e-9,
    )
    sol = vq.run_cpcm_scf(mol, basis, method="rhf", solvent=sm)
    assert sol.converged
    assert isinstance(sol.cavity, FineCavity)
    assert sol.cavity.surface.is_closed()
    # A reaction field on a closed surface must stabilise the solute.
    assert sol.e_solv < 0.0
    # Positive definite, which the spacing sweep below gates properly. Read
    # through build_cavity_A_matrix so this exercises the kernel the driver
    # actually used rather than re-picking one here.
    from vibeqc.solvation.cpcm import build_cavity_A_matrix

    ev = np.linalg.eigvalsh(build_cavity_A_matrix(sol.cavity))
    assert np.min(ev) > 0.0


def test_cfc_a_matrix_is_positive_definite_at_every_spacing():
    """#744: definiteness must not depend on the grid spacing.

    ``A`` positive definite is not housekeeping -- Lange & Herbert
    (doi:10.1063/1.3511297) eq. 2.25 makes it the condition for the
    polarization energy to be a *minimum*. An indefinite ``A`` means the
    reaction field can raise the energy and the apparent charges solve
    nothing, while ``run_cpcm_scf`` reports convergence and a negative
    ``e_solv`` regardless.

    Pre-fix, with the point-charge kernel: 10 of 29 spacings were indefinite,
    worst -8.62, and which ones was pure luck. A single pinned spacing is what
    let that through, so this sweeps.

    The range stops at 0.44 A because coarser grids do not resolve the
    162-direction segment grid and are refused outright -- see
    :func:`test_a_coarse_grid_is_refused_rather_than_left_singular`. The
    paper's own value is 0.3 A.
    """
    vq = pytest.importorskip("vibeqc")
    from vibeqc.solvation.cavity import atom_radii_bohr
    from vibeqc.solvation.cpcm import build_cavity_A_matrix

    pos = np.array([[0.0, 0.0, 0.0], [0.0, 1.498, -1.159], [0.0, -1.498, -1.159]])
    z = [8, 1, 1]
    rho = atom_radii_bohr(np.array(z), None, 1.2, 0.0)
    checked = 0
    for spacing in np.round(np.arange(0.24, 0.53, 0.01), 3):
        try:
            cav = build_fine_cavity(pos, z, rho, grid_spacing_ang=float(spacing))
        except RuntimeError as exc:
            # Refused for under-resolution, which is the other half of the
            # contract and is asserted separately below. Never silently
            # singular, which is what this pair together rules out.
            assert "too coarse to resolve" in str(exc)
            continue
        ev = np.linalg.eigvalsh(build_cavity_A_matrix(cav))
        assert np.min(ev) > 0.0, f"indefinite A at {spacing} A: {np.min(ev)}"
        checked += 1
    # Most of the sweep must actually be usable, or the assertion above could
    # pass by refusing everything.
    assert checked >= 18, f"only {checked} of 29 spacings produced a cavity"


def test_a_coarse_grid_is_refused_rather_than_left_singular():
    """Duplicate segments are refused, not merged and not shipped.

    A basis grid that does not resolve the segment grid leaves some Becke
    cells supported by a single basis point, and two such cells put their
    segments on the *same* point. ``A`` is the Gram matrix of the segments'
    Gaussians, so two identical Gaussians make it rank-deficient: measured
    minimum eigenvalue -1.0e-14 against a norm of 6.6e+03, i.e. exactly zero,
    and the screening is undefined along that null direction.

    Merging the pair would be worse: a Gaussian of area ``a`` is wider than
    two of area ``a/2``, so its self-energy differs by sqrt(2) and merging as
    segments approach coincidence would reintroduce a discontinuity of exactly
    the kind #757 removed. Resolution is the real requirement, so the error
    says so.
    """
    vq = pytest.importorskip("vibeqc")
    from vibeqc.solvation.cavity import atom_radii_bohr

    pos = np.array([[0.0, 0.0, 0.0], [0.0, 1.498, -1.159], [0.0, -1.498, -1.159]])
    z = [8, 1, 1]
    rho = atom_radii_bohr(np.array(z), None, 1.2, 0.0)
    # The spacing at which this bites has moved twice, and both moves are the
    # same story from different sides: a cell is starved when it is supported by
    # a single basis point. #769 fixed the direction grid's orientation relative
    # to the surface, and #771 let a basis point feed more than one atom, so
    # cells are fed from more points. On this solute the guard first fired at
    # 0.44 A, then at 0.70 A after #769, and it still fires at 0.70 A now -- in
    # *both* frames, the earlier lab-versus-molecular contrast having been
    # absorbed by the partition.
    for frame in ("lab", "molecular"):
        with pytest.raises(RuntimeError, match="too coarse to resolve"):
            build_fine_cavity(pos, z, rho, grid_spacing_ang=0.70, frame=frame)
        assert build_fine_cavity(
            pos, z, rho, grid_spacing_ang=0.50, frame=frame
        ).n_points > 0


def test_point_charge_kernel_is_what_was_indefinite():
    """The fix is the kernel, not the cavity: pin the defect's cause.

    Rebuilding the same CFC segments with the point-charge kernel must still
    be indefinite. Without this, a later change that quietly restored the
    point-charge kernel for the CFC would leave the sweep above passing for
    the wrong reason -- or worse, someone would "simplify" the two
    representations back into one.
    """
    vq = pytest.importorskip("vibeqc")
    from vibeqc.solvation.cavity import atom_radii_bohr
    from vibeqc.solvation.cpcm import POINT_CHARGE, build_A_matrix
    from vibeqc.solvation.fine_cavity import build_fine_cavity

    pos = np.array([[0.0, 0.0, 0.0], [0.0, 1.498, -1.159], [0.0, -1.498, -1.159]])
    z = [8, 1, 1]
    rho = atom_radii_bohr(np.array(z), None, 1.2, 0.0)
    cav = build_fine_cavity(pos, z, rho, grid_spacing_ang=0.35)
    ev = np.linalg.eigvalsh(
        build_A_matrix(cav.points, cav.weights, representation=POINT_CHARGE)
    )
    # Strongly indefinite, not marginally: the point-charge kernel puts an
    # unbounded 1/r against a bounded diagonal. The magnitude is not pinned to
    # a number, because #757 changed the segment set that produces it -- what
    # must not change is the sign.
    assert np.min(ev) < -1.0


def test_default_cavity_is_unchanged_by_the_fine_cavity_wiring():
    """Selecting a cavity must be opt-in. The two constructions give
    different segment sets, so a changed default would move every solvated
    energy in the suite."""
    vq = pytest.importorskip("vibeqc")
    mol = _scf_water(vq)
    basis = vq.BasisSet(mol, "sto-3g")
    kw = dict(epsilon=78.39, name="water", n_points_per_sphere=110,
              max_macro_iter=40, tol_e_solv=1e-9)
    implicit = vq.run_cpcm_scf(
        mol, basis, method="rhf", solvent=vq.SolventModel(**kw))
    explicit = vq.run_cpcm_scf(
        mol, basis, method="rhf", solvent=vq.SolventModel(cavity="lebedev", **kw))
    assert not isinstance(implicit.cavity, FineCavity)
    assert implicit.energy == pytest.approx(explicit.energy, rel=1e-12)
    assert implicit.cavity.n_points == explicit.cavity.n_points


def _atom_assignment(vq, positions, Zs, rho):
    """Which atom each surface vertex is assigned to (the step-5 argmin)."""
    from vibeqc.solvation.fine_cavity import _relative_distance, build_fine_cavity

    cav = build_fine_cavity(positions, Zs, rho, grid_spacing_ang=0.40)
    tau = _relative_distance(cav.surface.vertices, positions, rho)
    return np.argmin(tau, axis=1)


def test_cfc_energy_is_smooth_across_a_former_assignment_boundary():
    """#757 fixed: no jump where basis points reassign between *segments*.

    Before, step 6 put each basis point in exactly one segment by nearest
    centre, so a point on a boundary flipped at an infinitesimal displacement
    and its *whole* area moved: three of 1230 points reassigning shifted one
    segment weight by 0.333 bohr^2 and the energy by ~6e-7 Ha. Scanning one
    coordinate, every secant slope read 0.0232856 except across that boundary,
    which read 0.0111408 -- a 52 percent error.

    With a Becke partition there is no segment boundary to cross. Intervals
    where a basis point changes **atom** are excluded, because that is the
    step-5 assignment and a separate defect (#771) -- including them would
    make this test fail for a reason it is not about, and the exclusion is
    made explicit rather than absorbed into a loose tolerance.

    Water cannot show any of this; its symmetry hides it, which is why the
    solute here has none.
    """
    vq = pytest.importorskip("vibeqc")
    from vibeqc.solvation.cavity import atom_radii_bohr

    pos = np.array([[0.0, 0.0, 0.0], [2.61, 0.31, -0.12],
                    [-0.63, 1.55, 0.44], [3.19, -0.47, 1.71]])
    Zs = [8, 6, 1, 1]
    rho = atom_radii_bohr(np.array(Zs), None, 1.2, 0.0)
    sm = vq.SolventModel(
        epsilon=78.39, cavity="fine", fine_grid_spacing_ang=0.40,
        max_macro_iter=80, tol_e_solv=1e-12,
    )

    def sample(dz):
        p = pos.copy()
        p[0, 2] += dz
        m = vq.Molecule([vq.Atom(Zs[i], tuple(p[i])) for i in range(4)], 0, 1)
        e = float(vq.run_cpcm_scf(m, vq.BasisSet(m, "sto-3g"),
                                  method="rhf", solvent=sm).energy)
        return e, _atom_assignment(vq, p, Zs, rho)

    offsets = np.linspace(-3.0e-4, 3.0e-4, 13)
    samples = [sample(d) for d in offsets]
    slopes = []
    for i in range(len(offsets) - 1):
        (e0, o0), (e1, o1) = samples[i], samples[i + 1]
        if o0.shape != o1.shape or np.any(o0 != o1):
            continue                      # #771, not what this tests
        slopes.append((e1 - e0) / (offsets[i + 1] - offsets[i]))
    slopes = np.array(slopes)
    # Seven of twelve intervals were clean when this was written; the other
    # five cross a #771 atom-assignment boundary. Enough to be meaningful, and
    # the floor is stated so the test cannot pass by excluding everything.
    assert slopes.size >= 6, f"only {slopes.size} of 12 intervals were clean"
    spread = (slopes.max() - slopes.min()) / abs(slopes.mean())
    # 52 percent before. Measured 1.6e-05 here, which is curvature plus the
    # C1 projection kink of #770.
    assert spread < 1e-4, f"slope spread {spread:.2e} across the scan"


def test_the_frames_tie_is_invisible_in_the_energy():
    """The frame jumps at a tie; the energy does not, and here is why.

    Step 1 picks its axes by two ``argmax`` selections, and both tie at a
    symmetric geometry -- water's equilibrium structure ties in *both*. That
    looked like a defect of the #757 class: a discontinuity on a codimension-1
    surface, crossed during an ordinary geometry optimization.

    It is not, when the tie is protected by symmetry. The two candidate frames
    are then related by a symmetry operation of the solute, so they lay the
    lattice and the segment grids over the same surface and produce the same
    cavity relabeled. Scanning the oxygen through the crossing, the energy is
    even in the displacement to every digit (measured 2e-12 between +/-1e-05
    and +/-3e-04 Ha at the extremes, which is the SCF threshold), and the
    analytic gradient at the tie itself agrees with a full-SCF finite difference
    to 1.9e-06 Ha/bohr.

    An *accidental* tie -- two atoms equidistant from the centroid with no
    symmetry relating them -- is a different matter and remains an open
    question; see the handover. This test pins the case that actually occurs.
    """
    vq = pytest.importorskip("vibeqc")
    pos, z, _ = _water()
    d = pos - pos.mean(axis=0)
    r = np.linalg.norm(d, axis=1)
    assert r[1] == pytest.approx(r[2], rel=1e-14), "the tie this test is about"

    def energy(dz):
        p = pos.copy()
        p[0, 2] += dz
        mol = vq.Molecule([vq.Atom(z[i], tuple(p[i])) for i in range(3)], 0, 1)
        sm = vq.SolventModel(
            epsilon=78.39, cavity="fine", fine_grid_spacing_ang=0.40,
            fine_frame="molecular", max_macro_iter=80, tol_e_solv=1e-12,
        )
        e = float(vq.run_cpcm_scf(
            mol, vq.BasisSet(mol, "sto-3g"), method="rhf", solvent=sm
        ).energy)
        return e, molecular_frame(p)

    for step in (1e-4, 3e-4):
        (e_up, frame_up), (e_dn, frame_dn) = energy(step), energy(-step)
        assert e_up == pytest.approx(e_dn, abs=1e-11), step
        # And the selection itself did not flicker across the crossing.
        np.testing.assert_allclose(frame_up, frame_dn, atol=1e-3)


def test_an_accidental_frame_tie_is_a_measured_discontinuity():
    """The frame's one real defect, sized rather than left as a worry.

    ``test_the_frames_tie_is_invisible_in_the_energy`` covers the tie that
    symmetry protects. This is the other kind: two atoms equidistant from the
    centroid with *no* symmetry relating them, which is a codimension-1 surface
    through generic geometries and so is crossed by ordinary optimization paths.
    The two candidate frames are then genuinely different, the cavity is
    re-paved, and the energy jumps.

    Measured on the asymmetric solute, with one hydrogen pushed out along its
    own radius until it ties with the carbon: the selected x-axis flips across
    the crossing, the energy jumps by 5.8e-06 Ha, and the secant slope over the
    crossing interval reads 0.0019 against 0.0595 on either side -- a 97 percent
    error, the same shape as the #757 assignment defect (52 percent) and about
    ten times its size in energy.

    That is the argument against flipping the default to ``"molecular"``
    unconditionally, and for a smooth frame if one is wanted; see the handover.
    Pinned here so it cannot grow unnoticed, and so the next reader gets a
    number instead of a caveat. The assertions are a ceiling on the jump and the
    statement that it is *localized* -- the slopes away from the crossing agree
    -- both of which survive a fix.
    """
    vq = pytest.importorskip("vibeqc")
    base = np.array([[0.0, 0.0, 0.0], [2.61, 0.31, -0.12],
                     [-0.63, 1.55, 0.44], [3.19, -0.47, 1.71]])
    zs = [8, 6, 1, 1]
    centre = base.mean(axis=0)
    d0 = np.linalg.norm(base - centre, axis=1)
    runner_up = int(np.argsort(d0)[-2])

    def geom(t):
        p = base.copy()
        p[runner_up] = centre + (p[runner_up] - centre) * t
        return p

    def gap(t):
        d = np.linalg.norm(geom(t) - geom(t).mean(axis=0), axis=1)
        return d[runner_up] - d.max()

    lo, hi = 1.0, 1.6                       # bisect onto the tie
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if gap(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    t_tie = 0.5 * (lo + hi)

    def energy(t):
        p = geom(t)
        mol = vq.Molecule([vq.Atom(zs[i], tuple(p[i])) for i in range(4)], 0, 1)
        sm = vq.SolventModel(
            epsilon=78.39, cavity="fine", fine_grid_spacing_ang=0.40,
            fine_frame="molecular", max_macro_iter=80, tol_e_solv=1e-12,
        )
        return float(vq.run_cpcm_scf(
            mol, vq.BasisSet(mol, "sto-3g"), method="rhf", solvent=sm
        ).energy)

    offsets = np.array([-4e-4, -2e-4, -5e-5, 5e-5, 2e-4, 4e-4])
    e = np.array([energy(t_tie + d) for d in offsets])
    secants = np.diff(e) / np.diff(offsets)
    below, across, above = secants[:2], secants[2], secants[3:]
    clean = np.concatenate([below, above])
    # Localized: the slope away from the crossing is one slope.
    assert (clean.max() - clean.min()) / abs(clean.mean()) < 0.05, secants
    # And the jump itself, as the slope the crossing interval fails to carry.
    jump = abs((across - clean.mean()) * (offsets[3] - offsets[2]))
    # The bound is a bound, not a target: this defect is open, and what the
    # number measures is how far apart two arbitrary-but-equivalent
    # discretizations of the same surface sit. It therefore moves whenever the
    # discretization does, in either direction and for no reason to do with
    # this defect -- #779's change to step 4 took it from 5.8e-06 to 1.1e-05
    # without touching the frame at all. Read a change here as "the cavity
    # changed", and re-measure rather than inferring a regression.
    assert jump < 2e-5, f"the accidental-tie jump grew to {jump:.2e} Ha"


def test_cfc_energy_is_rotationally_invariant_in_the_molecular_frame():
    """#769, at the level that decides whether it matters: the energy.

    A solvated energy cannot depend on how the solute is oriented in the
    laboratory -- the cavity is a property of the molecule. In the lab frame it
    does, because the marching lattice and the segment directions are laid out
    along the laboratory axes: measured here on water at 0.40 A, RHF/STO-3G,
    eps = 78.39, the energy moves by 1.0e-05 Ha over rotations of 0 to 90
    degrees about a generic axis, of which 1.8e-06 Ha appears in the first
    degree. That is larger than every discontinuity this workstream has removed
    from the CFC (#744 6e-7 Ha, #757 6e-7 Ha), and it also means the gradient
    carries a spurious torque, so the six rigid-body modes are not zero.

    In the molecule-fixed frame of workflow step 1 it does not: the spread is
    5.7e-14 Ha, i.e. the SCF convergence threshold. The asymmetric solute of
    ``test_cfc_energy_is_smooth_across_a_former_assignment_boundary`` behaves the
    same way -- 1.3e-05 Ha in the lab frame, 1.7e-13 Ha in the molecular one.

    Both frames are measured in one test on purpose. The molecular figure alone
    would also be satisfied by an energy that had stopped depending on the
    geometry, and the lab figure is the size of the defect being fixed.
    """
    vq = pytest.importorskip("vibeqc")
    pos, z, _ = _water()
    # Three orientations: the reference, one degree (the smallest rotation with
    # a measurable effect) and a large generic one.
    angles = (0.0, 1.0, 47.0)

    def energy(rotated, frame):
        mol = vq.Molecule(
            [vq.Atom(z[i], tuple(rotated[i])) for i in range(len(z))], 0, 1
        )
        sm = vq.SolventModel(
            epsilon=78.39, cavity="fine", fine_grid_spacing_ang=0.40,
            fine_frame=frame, max_macro_iter=80, tol_e_solv=1e-12,
        )
        return float(vq.run_cpcm_scf(
            mol, vq.BasisSet(mol, "sto-3g"), method="rhf", solvent=sm
        ).energy)

    spreads = {}
    for frame in ("lab", "molecular"):
        e = np.array([energy(pos @ _rotation(_AXIS, a).T, frame) for a in angles])
        spreads[frame] = float(e.max() - e.min())
    assert spreads["lab"] > 1e-6, spreads
    assert spreads["molecular"] < 1e-10, spreads


def test_the_step_5_atom_partition_makes_the_energy_smooth():
    """#771 fixed: no jump where a basis point changes *atom*.

    Step 5 assigned each basis point to the atom of smallest relative distance
    by ``argmin``, so a point on a boundary flipped at an infinitesimal
    displacement and its whole area moved to the other atom's segment grid.
    Measured before the fix, in this frame and on this solute: 0.19 bohr^2 of
    area moved at once, the secant across the crossing read 0.0227 against
    0.0212 on either side -- a 6.9 percent error -- and the energy stepped by
    2.9e-08 Ha. Crossings were frequent: twelve coordinates scanned over
    +/-6e-03 bohr found 30 of them.

    The cells are smooth now (:mod:`vibeqc.solvation.atom_partition`). Scanning
    straight through where that crossing was, the secant spread is 1.4e-03 --
    fifty times smaller -- and what is left is the scan's own curvature rather
    than a step, which is what the monotone secants assert.

    Water cannot show any of this; its symmetry hides it, which is why the
    solute here has none.
    """
    vq = pytest.importorskip("vibeqc")

    pos = np.array([[0.0, 0.0, 0.0], [2.61, 0.31, -0.12],
                    [-0.63, 1.55, 0.44], [3.19, -0.47, 1.71]])
    zs = [8, 6, 1, 1]

    def energy(t):
        p = pos.copy()
        p[1, 2] += t
        mol = vq.Molecule([vq.Atom(zs[i], tuple(p[i])) for i in range(4)], 0, 1)
        sm = vq.SolventModel(
            epsilon=78.39, cavity="fine", fine_grid_spacing_ang=0.40,
            max_macro_iter=80, tol_e_solv=1e-12,
        )
        return float(vq.run_cpcm_scf(
            mol, vq.BasisSet(mol, "sto-3g"), method="rhf", solvent=sm
        ).energy)

    # The old crossing sat at t = 4.42332e-04; scan straight through it.
    offsets = np.linspace(4.42332e-04 - 3e-5, 4.42332e-04 + 3e-5, 7)
    e = np.array([energy(t) for t in offsets])
    secants = np.diff(e) / np.diff(offsets)
    spread = (secants.max() - secants.min()) / abs(secants.mean())
    assert spread < 5e-3, f"slope spread {spread:.2e} across the old crossing"
    steps = np.diff(secants)
    assert np.all(steps > 0) or np.all(steps < 0), secants


def test_the_atom_partition_is_a_partition_of_the_step_5_rule():
    """The two properties step 5's replacement has to have (#771).

    It must conserve area exactly -- the CFC's area feeds COSMO-RS sigma
    profiles, so a partition that leaked would be worse than the discontinuity
    it replaces -- and it must still be *the same rule*, not a new one. The
    second is the criterion that disqualified Becke's distance coordinate,
    which moved 13 to 16 percent of the total area between atoms; this one
    moves under 2 percent, shrinking with the grid, and its dominant cell
    agrees with the hard ``argmin`` for 99.8 percent of points.
    """
    from vibeqc.solvation.atom_partition import atom_partition
    from vibeqc.solvation.cavity import atom_radii_bohr
    from vibeqc.solvation.fine_cavity import _relative_distance

    pos = np.array([[0.0, 0.0, 0.0], [2.61, 0.31, -0.12],
                    [-0.63, 1.55, 0.44], [3.19, -0.47, 1.71]])
    zs = [8, 6, 1, 1]
    rho = atom_radii_bohr(np.array(zs), None, 1.2, 0.0)

    def probe(spacing):
        cavity = build_fine_cavity(pos, zs, rho, grid_spacing_ang=spacing)
        pts = cavity.surface.vertices
        part = atom_partition(pts, pos, rho, cavity.grid_spacing)
        tri, ta = cavity.surface.triangles, cavity.surface.triangle_areas()
        area = np.zeros(pts.shape[0])
        np.add.at(area, tri.ravel(), np.repeat(ta / 3.0, 3))
        owner = np.argmin(_relative_distance(pts, pos, rho), axis=1)
        per_hard = np.zeros(len(zs))
        np.add.at(per_hard, owner, area)
        shift = np.abs(area @ part.weights - per_hard).max() / area.sum()
        return part, owner, float((part.weights > 0).sum(axis=1).mean()), shift

    part, hard, per_point, shift = probe(0.40)
    assert part.row_sum_residual() < 1e-14
    assert np.all(part.weights >= 0.0)
    assert (np.argmax(part.weights, axis=1) == hard).mean() > 0.99
    assert per_point < 2.0
    assert shift < 0.02

    # Both properties improve with the grid, which is what makes the marching
    # spacing the right width to smooth over: the ambiguity is a discretization
    # ambiguity, so the fuzzy rule tends to the hard one in the same limit.
    _, _, fine_per_point, fine_shift = probe(0.20)
    assert fine_per_point < per_point
    assert fine_shift < shift


def test_the_atom_partition_adjoints_are_exact():
    """Its two identities, then finite differences.

    A partition built from *relative* distances cannot notice the point and
    every atom moving together, so the adjoints sum to zero across that motion;
    and rows summing to one make a constant adjoint contract to nothing. Both
    are exact and need no displaced geometries. Finite differences are carried
    as a third check because they are the only one that would catch a sign
    error common to every atom.
    """
    from vibeqc.solvation.atom_partition import atom_partition
    from vibeqc.solvation.cavity import atom_radii_bohr

    pos = np.array([[0.0, 0.0, 0.0], [2.61, 0.31, -0.12],
                    [-0.63, 1.55, 0.44], [3.19, -0.47, 1.71]])
    zs = [8, 6, 1, 1]
    rho = atom_radii_bohr(np.array(zs), None, 1.2, 0.0)
    cav = build_fine_cavity(pos, zs, rho, grid_spacing_ang=0.45)
    pts = cav.surface.vertices
    h = cav.grid_spacing
    part = atom_partition(pts, pos, rho, h)

    rng = np.random.default_rng(0)
    adj = rng.normal(size=part.weights.shape)
    d_pts, d_atoms = part.vjp(adj)
    assert np.abs(d_pts.sum(axis=0) + d_atoms.sum(axis=0)).max() < 1e-11
    flat = part.vjp(np.ones_like(adj))
    assert max(np.abs(flat[0]).max(), np.abs(flat[1]).max()) < 1e-12

    def value(p, c):
        return float(np.einsum(
            "ba,ba->", adj, atom_partition(p, c, rho, h).weights
        ))

    eps = 1e-6
    for a in range(pos.shape[0]):
        for i in range(3):
            up, dn = pos.copy(), pos.copy()
            up[a, i] += eps
            dn[a, i] -= eps
            fd = (value(pts, up) - value(pts, dn)) / (2.0 * eps)
            assert abs(d_atoms[a, i] - fd) < 1e-6, (a, i, d_atoms[a, i], fd)


def test_the_atom_partition_rejects_bad_input():
    from vibeqc.solvation.atom_partition import atom_partition

    pts = np.zeros((4, 3))
    pos = np.eye(3)
    rho = np.ones(3)
    with pytest.raises(ValueError, match="points must be"):
        atom_partition(np.zeros(3), pos, rho, 1.0)
    with pytest.raises(ValueError, match="one radius per atom"):
        atom_partition(pts, pos, np.ones(2), 1.0)
    with pytest.raises(ValueError, match="radii must be positive"):
        atom_partition(pts, pos, np.zeros(3), 1.0)
    with pytest.raises(ValueError, match="marching spacing must be positive"):
        atom_partition(pts, pos, rho, 0.0)


def test_the_gradient_is_ill_conditioned_near_a_frame_tie():
    """What step 1's tie costs the gradient, which is not what it costs the energy.

    #769 established that a *symmetry-induced* tie is invisible in the energy:
    the two candidate frames are related by a symmetry operation of the solute,
    so they discretize the same surface and the energy is smooth across the
    crossing. That is still true and still tested.

    The gradient is a different matter, and this pins it. At the tie the frame
    does not vary steeply, it **jumps** between two finite directions, so the
    chain rule through it is not a route to the energy's derivative there.
    Measured on symmetric water: the analytic gradient reports a force on the
    oxygen along the tie-breaking direction of 1.9e-05 Ha/bohr, in a component
    the molecule's mirror symmetry requires to vanish and which a converged
    finite difference puts at 1e-11.

    It is a local defect, not a global one. Lengthening one bond to break the
    tie -- 0.1 bohr is ample -- brings analytic and finite difference back to
    4.7e-07 Ha/bohr, better than the lab frame manages on the same solute. And
    it is below the convergence thresholds an optimizer uses, so what it costs
    in practice is that a symmetric structure is not exactly a stationary point
    of the analytic gradient.

    Averaging the frame Jacobian over the tied atoms was tried, and is recorded
    in ``molecular_frame_jacobian`` as wrong: there are no two one-sided
    derivatives to average across a jump, and the averaged Jacobian breaks the
    exact rotation identity that the real one satisfies to 1e-16.
    """
    vq = pytest.importorskip("vibeqc")
    from vibeqc.solvation.fine_cavity import _leading_set
    from vibeqc.solvation.gradient import cpcm_gradient_fd

    def analytic_gradient(mol):
        basis = vq.BasisSet(mol, "sto-3g")
        sm = vq.SolventModel(
            epsilon=78.39, cavity="fine", fine_grid_spacing_ang=0.4,
            fine_frame="molecular", max_macro_iter=80, tol_e_solv=1e-12,
        )
        sol = vq.run_cpcm_scf(mol, basis, method="rhf", solvent=sm)
        return vq.cpcm_gradient(sol.scf, mol, basis, sol, method="rhf"), sm

    tied = _scf_water(vq)
    pos = np.array([a.xyz for a in tied.atoms])
    d = pos - pos.mean(axis=0)
    assert _leading_set(np.linalg.norm(d, axis=1), 1.6).size == 2, "the tie"

    # No finite difference needed on this half: the solute is mirror-symmetric
    # about the plane the oxygen would move out of, so that force is zero by
    # symmetry and any value the gradient reports is the defect itself.
    at_tie, _ = analytic_gradient(tied)
    assert abs(at_tie[0, 1]) > 1e-6, at_tie
    # The lab frame, which has no step-1 selection to tie, gets it right.
    lab_sm = vq.SolventModel(
        epsilon=78.39, cavity="fine", fine_grid_spacing_ang=0.4,
        fine_frame="lab", max_macro_iter=80, tol_e_solv=1e-12,
    )
    lab_basis = vq.BasisSet(tied, "sto-3g")
    lab_sol = vq.run_cpcm_scf(tied, lab_basis, method="rhf", solvent=lab_sm)
    lab = vq.cpcm_gradient(lab_sol.scf, tied, lab_basis, lab_sol, method="rhf")
    assert abs(lab[0, 1]) < 1e-12

    # Break the tie and the composed chain is back to agreeing with the
    # outermost oracle, better than the lab frame does on the same solute.
    untied = _scf_water_untied(vq)
    off_tie, sm = analytic_gradient(untied)
    fd = cpcm_gradient_fd(untied, "sto-3g", method="rhf", solvent=sm,
                          step_bohr=2e-3)
    worst = float(np.max(np.abs(off_tie - fd)))
    assert worst < 5e-6
    assert worst < 0.1 * abs(at_tie[0, 1])


def test_analytic_gradient_on_a_fine_cavity_matches_full_scf_fd():
    """The assembled analytic gradient, against the outermost oracle.

    ``cpcm_gradient_fd`` re-converges the whole calculation at displaced
    geometries, so it assumes nothing about the cavity. Agreement with it is
    what licenses the analytic path.

    The tolerance is set by the *reference*, not by the gradient. A full-SCF
    FD on this system carries about 3e-9 Ha of convergence noise, so its own
    accuracy bottoms out near 1.5e-6 Ha/bohr at h = 2e-3 and degrades on both
    sides -- roundoff as h shrinks, O(h^2) truncation as h grows. The step
    used here sits at that minimum.
    """
    vq = pytest.importorskip("vibeqc")
    from vibeqc.solvation.gradient import cpcm_gradient_fd

    mol = _scf_water_untied(vq)
    basis = vq.BasisSet(mol, "sto-3g")
    sm = vq.SolventModel(
        epsilon=78.39, name="water", cavity="fine",
        fine_grid_spacing_ang=0.4, max_macro_iter=80, tol_e_solv=1e-12,
    )
    sol = vq.run_cpcm_scf(mol, basis, method="rhf", solvent=sm)
    analytic = vq.cpcm_gradient(sol.scf, mol, basis, sol, method="rhf")
    fd = cpcm_gradient_fd(mol, "sto-3g", method="rhf", solvent=sm, step_bohr=2e-3)
    assert np.max(np.abs(analytic - fd)) < 5e-6


def test_analytic_gradient_on_a_molecule_fixed_cavity_matches_full_scf_fd():
    """The composed chain, against the outermost oracle, and the torque with it.

    ``cpcm_gradient_fd`` re-converges the whole calculation at displaced
    geometries, so it assumes nothing about the cavity *or* the frame -- its
    displacements move the frame exactly as the analytic chain claims they do.
    Agreement with it is what licenses :class:`FrameCavityDerivative`.

    The second assertion is the one only this frame can pass. A molecule in free
    space feels no torque, and the lab-frame CFC does: measured on the
    asymmetric solute, the analytic gradient's net torque is 2.9e-05 Ha, which
    is the rotational counterpart of the 1.0e-05 Ha the energy moves when the
    solute is turned. In the molecule-fixed frame it is 1.0e-13. Analytic
    agreement with FD also improves, 9.0e-05 to 1.4e-05 Ha/bohr, though that
    number is dominated by the FD reference's own noise.
    """
    vq = pytest.importorskip("vibeqc")
    from vibeqc.solvation.gradient import cpcm_gradient_fd

    mol = _scf_water_untied(vq)
    basis = vq.BasisSet(mol, "sto-3g")
    sm = vq.SolventModel(
        epsilon=78.39, cavity="fine", fine_grid_spacing_ang=0.4,
        fine_frame="molecular", max_macro_iter=80, tol_e_solv=1e-12,
    )
    sol = vq.run_cpcm_scf(mol, basis, method="rhf", solvent=sm)
    analytic = vq.cpcm_gradient(sol.scf, mol, basis, sol, method="rhf")
    fd = cpcm_gradient_fd(mol, "sto-3g", method="rhf", solvent=sm, step_bohr=2e-3)
    # Measured 4.7e-07 here, against 3.6e-06 for the same solute in the lab
    # frame: away from a frame tie the composed chain is the *better* gradient,
    # not merely an equal one.
    assert np.max(np.abs(analytic - fd)) < 5e-6

    # The torque, on a solute whose symmetry does not zero it for free.
    pos = np.array([[0.0, 0.0, 0.0], [2.61, 0.31, -0.12],
                    [-0.63, 1.55, 0.44], [3.19, -0.47, 1.71]])
    zs = [8, 6, 1, 1]
    asym = vq.Molecule([vq.Atom(zs[i], tuple(pos[i])) for i in range(4)], 0, 1)
    asym_basis = vq.BasisSet(asym, "sto-3g")
    torque = {}
    for frame in ("lab", "molecular"):
        model = vq.SolventModel(
            epsilon=78.39, cavity="fine", fine_grid_spacing_ang=0.40,
            fine_frame=frame, max_macro_iter=80, tol_e_solv=1e-12,
        )
        res = vq.run_cpcm_scf(asym, asym_basis, method="rhf", solvent=model)
        g = vq.cpcm_gradient(res.scf, asym, asym_basis, res, method="rhf")
        # Translational invariance holds in both frames; it is not the check.
        assert np.abs(g.sum(axis=0)).max() < 1e-10, frame
        torque[frame] = float(np.abs(np.cross(pos, g).sum(axis=0)).max())
    assert torque["molecular"] < 1e-10, torque
    assert torque["lab"] > 1e-6, torque


def test_cfc_gradient_matches_fd_to_the_noise_floor_on_an_asymmetric_solute():
    """The sharpest available check on the assembled CFC gradient.

    Water is a weak test: C2v forces several gradient components to zero, so a
    sign error in those directions cannot show. This solute has no symmetry
    element and no vanishing component.

    The step is chosen to stay on one smooth branch of the construction
    (#757), which is what makes the comparison sharp: on a single branch the
    analytic gradient reproduces the finite difference to ~1e-9 Ha/bohr, five
    orders better than the 1.5e-6 the water V-curve bottoms out at. Crossing
    an assignment boundary instead gives 1e-5 to 1e-3 -- a property of the
    energy surface, not of the derivative, which is precisely why that has to
    be excluded deliberately rather than absorbed into a loose tolerance.
    """
    vq = pytest.importorskip("vibeqc")
    pos = np.array([[0.0, 0.0, 0.0], [2.61, 0.31, -0.12],
                    [-0.63, 1.55, 0.44], [3.19, -0.47, 1.71]])
    Zs = [8, 6, 1, 1]
    sm = vq.SolventModel(
        epsilon=78.39, cavity="fine", fine_grid_spacing_ang=0.40,
        max_macro_iter=80, tol_e_solv=1e-12,
    )
    mol = vq.Molecule([vq.Atom(Zs[i], tuple(pos[i])) for i in range(4)], 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")
    sol = vq.run_cpcm_scf(mol, basis, method="rhf", solvent=sm)
    analytic = vq.cpcm_gradient(sol.scf, mol, basis, sol, method="rhf")

    # No component is zero, so nothing can hide.
    assert np.min(np.abs(analytic)) > 1e-3
    assert np.max(np.abs(analytic.sum(axis=0))) < 1e-10

    ref = sol.cavity
    h = 1.0e-4
    checked = 0
    for a in range(4):
        for c in range(3):
            energies, same_branch = [], True
            for sgn in (+1, -1):
                p = pos.copy()
                p[a, c] += sgn * h
                m = vq.Molecule(
                    [vq.Atom(Zs[i], tuple(p[i])) for i in range(4)], 0, 1
                )
                s = vq.run_cpcm_scf(m, vq.BasisSet(m, "sto-3g"),
                                    method="rhf", solvent=sm)
                energies.append(float(s.energy))
                cav = s.cavity
                # The discrete choices that survive #757: the step-5 atom
                # assignment (#771), step 6's projection flag (#770), and the
                # segment-area floor. ``basis_segment`` is no longer one of
                # them -- since #757 it is the argmax of a smooth partition, a
                # diagnostic that moves constantly without the construction
                # changing at all, so testing it here would reject every
                # displacement.
                same_branch = same_branch and (
                    cav.n_points == ref.n_points
                    and cav.projected.shape == ref.projected.shape
                    and not np.any(cav.projected != ref.projected)
                    and all(
                        np.array_equal(a.keep, b.keep)
                        for a, b in zip(cav.coarsening, ref.coarsening)
                    )
                )
            if not same_branch:
                continue
            fd = (energies[0] - energies[1]) / (2 * h)
            assert abs(analytic[a, c] - fd) < 1e-7, f"component {a},{c}"
            checked += 1
    # 9 of 12 stayed on one branch when this was written; the other 3 cross a
    # #757 assignment boundary even at h = 1e-4. The floor is set below that
    # so the test is not brittle to which components happen to be clean, but
    # high enough that it cannot pass by skipping everything.
    assert checked >= 8, f"only {checked} of 12 components stayed on branch"


def test_fine_cavity_gradient_carries_no_net_force():
    """Translational invariance, to machine precision and without any FD.

    A rigid translation cannot change the energy, so the gradient must sum to
    zero over atoms. This is the check that caught the missing box-translation
    term: the CFC's marching grid is anchored to the molecule, and a
    derivative that treats it as fixed leaves a spurious net force. Measured
    then at 4.8e-5 Ha/bohr, against components of order 2e-2 -- small enough
    to pass any eyeball test of the gradient itself, and it also made the
    analytic-vs-FD residual plateau at 1.7e-5 instead of tracking the FD's own
    noise floor down to 1.5e-6.
    """
    vq = pytest.importorskip("vibeqc")
    mol = _scf_water(vq)
    basis = vq.BasisSet(mol, "sto-3g")
    sm = vq.SolventModel(
        epsilon=78.39, name="water", cavity="fine",
        fine_grid_spacing_ang=0.4, max_macro_iter=80, tol_e_solv=1e-12,
    )
    sol = vq.run_cpcm_scf(mol, basis, method="rhf", solvent=sm)
    grad = vq.cpcm_gradient(sol.scf, mol, basis, sol, method="rhf")
    assert np.max(np.abs(grad.sum(axis=0))) < 1e-10


def test_finite_difference_gradient_works_on_a_fine_cavity():
    """The escape hatch the refusal message points at must actually exist.

    ``cpcm_gradient_fd`` re-converges the whole calculation at displaced
    geometries, so it makes no assumption about the cavity construction at
    all -- which is precisely why it stays valid where the analytic path does
    not.
    """
    vq = pytest.importorskip("vibeqc")
    mol = vq.Molecule(
        [vq.Atom(8, (0.0, 0.0, 0.0)),
         vq.Atom(1, (1.81, 0.0, 0.20)),
         vq.Atom(1, (-0.38, 1.75, 0.0))], 0, 1)
    sm = vq.SolventModel(
        epsilon=2.27, name="benzene", variant="cosmo", cavity="fine",
        fine_grid_spacing_ang=0.4, max_macro_iter=40, tol_e_solv=1e-9,
    )
    g = np.asarray(
        vq.cpcm_gradient_fd(mol, "sto-3g", method="rhf", solvent=sm,
                            step_bohr=2e-3),
        dtype=np.float64,
    )
    assert g.shape == (3, 3)
    assert np.all(np.isfinite(g))
    assert np.abs(g).max() > 1e-4      # displaced geometry: real forces


def test_conductor_surface_and_cosmors_consume_a_fine_cavity():
    """The layer boundary holds: COSMO-RS does not know which cavity it got.

    This is the reason the CFC was worth implementing -- a sigma profile built
    on a surface whose concave regions are paved, rather than one that leaves
    them open and piles charge at the edges.
    """
    vq = pytest.importorskip("vibeqc")
    from vibeqc.solvation import cosmors as rs
    from vibeqc.solvation.surface import build_conductor_surface

    mol = _scf_water(vq)
    basis = vq.BasisSet(mol, "sto-3g")
    sm = vq.SolventModel(
        epsilon=math.inf, name="conductor", variant="cosmo", cavity="fine",
        fine_grid_spacing_ang=0.35, max_macro_iter=40, tol_e_solv=1e-9,
    )
    sol = vq.run_cpcm_scf(mol, basis, method="rhf", solvent=sm)
    assert sol.screening.is_conductor and sol.screening.f == 1.0

    surf = build_conductor_surface(
        sol, method="rhf", basis="sto-3g", cavity_kind="fine-cfc",
        energy_conductor=float(sol.energy),
    )
    assert surf.n_segments == sol.cavity.n_points
    assert surf.provenance.cavity_kind == "fine-cfc"

    desc = rs.segment_descriptors(surf, rs.KLAMT_1998.r_av)
    # Physical screening charge density, as in Klamt 1998 Figure 3.
    assert np.max(np.abs(desc.sigma)) < 0.03
    prof = rs.sigma_profile(desc)
    assert prof.area() == pytest.approx(desc.total_area, rel=1e-9)
    pot = rs.sigma_potential_segments([desc], np.array([1.0]), rs.KLAMT_1998)
    assert np.all(np.isfinite(pot.mu_tilde))
