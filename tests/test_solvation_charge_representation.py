"""Surface-charge representations: point charges vs York-Karplus Gaussians.

The two kernels are not a style choice. The point-charge form ``1/r_ij`` is
singular at contact, and Lange & Herbert (J. Chem. Phys. 133, 244111 (2010),
doi:10.1063/1.3511297, p. 244111-8) state the dependency: point charges
"necessitate the use of an alternative switching function, as close approach
of these point charges must be avoided". A Lebedev cavity has that switching
function; a CFC has none and cannot have one, so it needs a kernel that
removes the singularity rather than avoiding it (#744).

These tests pin the parameter chain end to end, because the whole point of the
fix is that it introduces no fitted number of its own.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vibeqc.solvation.cavity import atom_radii_bohr, build_cavity
from vibeqc.solvation.cpcm import (
    CPCM_DIAG_ALPHA_BARE,
    GAUSSIAN_CHARGE,
    POINT_CHARGE,
    YORK_KARPLUS_ZETA,
    build_A_matrix,
    build_cavity_A_matrix,
    gaussian_exponents,
    york_karplus_zeta,
)
from vibeqc.solvation.fine_cavity import ANG_TO_BOHR, build_fine_cavity

WATER = np.array([[0.0, 0.0, 0.0], [0.0, 1.498, -1.159], [0.0, -1.498, -1.159]])
Z = [8, 1, 1]


def _fine(spacing=0.4):
    rho = atom_radii_bohr(np.array(Z), None, 1.2, 0.0)
    return build_fine_cavity(WATER, Z, rho, grid_spacing_ang=spacing)


# ---------------------------------------------------------------------
# The parameter chain, which must contain no number of our own
# ---------------------------------------------------------------------


def test_zeta_table_is_york_karplus_table_1_verbatim():
    """Transcription check on the one piece of imported data.

    York & Karplus, J. Phys. Chem. A 103, 11060 (1999),
    doi:10.1021/jp992097l, Table 1: optimized Gaussian exponents for a unit
    sphere, fitted to give the exact Born ion energy of a conductor together
    with a uniform surface charge distribution.
    """
    published = {
        14: 4.865, 26: 4.855, 50: 4.893, 110: 4.901, 194: 4.903,
        302: 4.905, 434: 4.906, 590: 4.905, 770: 4.899, 974: 4.907,
        1202: 4.907,
    }
    assert YORK_KARPLUS_ZETA == published
    # Flat above 110 points, which is what licenses a dense-limit value for a
    # cavity that is not a Lebedev grid at all.
    dense = [v for k, v in published.items() if k >= 110]
    assert max(dense) - min(dense) < 0.01


def test_the_two_papers_parameterizations_agree():
    """The cross-check that says the bridge is right, not merely consistent.

    Lange & Herbert give the point-charge diagonal as ``C_S sqrt(4 pi / a_i)``
    (Table III) and the Gaussian diagonal as ``zeta sqrt(2 / (pi a_i F_i))``
    (eq. 4.1). Equating them yields ``zeta = C_S pi sqrt(2)``. Feeding their
    Lebedev ``C_S = 1.104`` through that relation must reproduce York &
    Karplus' independently fitted zeta -- two papers, two parameterizations,
    two fitting procedures, arriving at the same number.
    """
    zeta_from_cs = 1.104 * math.pi * math.sqrt(2.0)
    assert abs(zeta_from_cs - 4.9049) < 1e-4
    # Within 0.1 percent of every York-Karplus entry at 110 points and above.
    for n in (110, 194, 302, 434, 590, 974, 1202):
        assert abs(zeta_from_cs - YORK_KARPLUS_ZETA[n]) / zeta_from_cs < 1.1e-3


def test_vibeqc_uses_the_gepol_constant_on_lebedev_grids():
    """Records a known, separate discrepancy rather than letting it hide.

    Lange & Herbert Table III gives ``C_S = 1.0694`` for GEPOL grids and
    ``C_S = 1.104`` for Lebedev grids. vibe-qc discretizes with Lebedev grids
    and uses 1.0694, i.e. an implied Gaussian width of 4.751 against York &
    Karplus' fitted 4.901 -- about 3 percent narrow.

    Not changed here: it would move every shipped solvation reference, which
    is a maintainer decision, and it is unrelated to the definiteness defect
    this module fixes. Pinned so the next reader sees a deliberate choice
    instead of assuming the literature value is in use.
    """
    assert CPCM_DIAG_ALPHA_BARE == 1.0694
    implied = CPCM_DIAG_ALPHA_BARE * math.pi * math.sqrt(2.0)
    assert abs(implied - 4.7512) < 1e-4
    assert abs(implied - YORK_KARPLUS_ZETA[110]) / implied > 0.03


def test_exponent_formula_matches_york_karplus_eq_61_with_radius_scaling():
    """``zeta_i = zeta sqrt(F_i/a_i)`` against the published form.

    York & Karplus eq. 61 gives ``zeta_k = zeta / sqrt(w_k)`` on the unit
    sphere and state the scaling ``zeta_k(R) = zeta_k(1)/R``. With
    ``a_i = w_i R^2 F_i`` the area form must reproduce it exactly. The area
    form is the one implemented, because a CFC has no ``w_i`` or ``R`` to
    speak of.
    """
    rng = np.random.default_rng(0)
    zeta = 4.901
    for _ in range(5):
        w_unit = float(rng.uniform(0.05, 0.4))     # unit-sphere weight
        R = float(rng.uniform(2.0, 6.0))           # sphere radius, bohr
        F = float(rng.uniform(0.2, 1.0))           # switching factor
        area = w_unit * R * R * F
        published = zeta / (R * math.sqrt(w_unit))
        ours = gaussian_exponents(
            np.array([area]), np.array([F]), zeta=zeta
        )[0]
        assert abs(ours - published) < 1e-12 * published


def test_zeta_selection_falls_back_to_the_dense_limit():
    assert york_karplus_zeta(110) == 4.901
    assert york_karplus_zeta(302) == 4.905
    assert york_karplus_zeta(None) == 4.907      # not a Lebedev grid
    assert york_karplus_zeta(137) == 4.907       # not a tabulated level


# ---------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------


def test_gaussian_diagonal_equals_the_point_charge_diagonal_at_matched_width():
    """The diagonals are the *same object* in two parameterizations.

    ``zeta_i sqrt(2/pi)`` and ``C_S sqrt(4 pi / a_i)`` coincide identically
    when ``zeta = C_S pi sqrt(2)``. This is why the fix needs no new fitted
    parameter: the existing diagonal already implies a Gaussian width, and the
    defect was that the off-diagonal was the point-charge limit of a different
    scheme.
    """
    areas = np.array([0.05, 0.4, 1.0, 2.5, 9.0])
    for c_s in (1.0694, 1.104):
        zeta = c_s * math.pi * math.sqrt(2.0)
        z = gaussian_exponents(areas, None, zeta=zeta)
        gaussian = z * math.sqrt(2.0 / math.pi)
        point = c_s * np.sqrt(4.0 * math.pi / areas)
        assert np.allclose(gaussian, point, rtol=1e-14, atol=0.0)


def test_gaussian_off_diagonal_reduces_to_one_over_r_when_far_apart():
    """``erf(x)/r -> 1/r``, so the correction is local by construction.

    A kernel change that perturbed well-separated pairs would move every
    energy. This one does not: ``erf`` saturates to 1 in double precision
    around ``x ~ 6``, so beyond a couple of bohr the two kernels agree to
    machine precision and the entire difference lives where segments are
    close -- which is the only place the point-charge form was wrong.
    """
    pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]])
    w = np.array([1.0, 1.0])
    g = build_A_matrix(pts, w, representation=GAUSSIAN_CHARGE, zeta=4.901)
    p = build_A_matrix(pts, w, representation=POINT_CHARGE)
    assert abs(g[0, 1] - p[0, 1]) < 1e-15
    assert abs(g[0, 1] - 0.25) < 1e-15


def test_gaussian_off_diagonal_is_bounded_at_contact():
    """The whole point: finite where ``1/r`` diverges."""
    w = np.array([1.0, 1.0])
    contact = None
    for sep in (1.0, 1e-2, 1e-4, 0.0):
        pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, sep]])
        g = build_A_matrix(pts, w, representation=GAUSSIAN_CHARGE, zeta=4.901)
        assert np.isfinite(g).all()
        contact = g[0, 1]
    # At coincidence the off-diagonal must equal the diagonal: two identical
    # Gaussians sitting on top of each other interact exactly as strongly as
    # one interacts with itself. With unit areas that is zeta sqrt(2/pi).
    assert abs(contact - 4.901 * math.sqrt(2.0 / math.pi)) < 1e-12
    diag = build_A_matrix(
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), w,
        representation=GAUSSIAN_CHARGE, zeta=4.901,
    )
    assert abs(diag[0, 1] - diag[0, 0]) < 1e-12


@pytest.mark.parametrize("spacing", [0.28, 0.30, 0.35, 0.40])
def test_gaussian_A_is_positive_definite_where_point_charge_A_is_not(spacing):
    """Definiteness is structural for a Gram matrix, not geometric.

    ``A`` under the Gaussian representation is the Gram matrix of normalized
    spherical Gaussians in the Coulomb inner product, and that form is
    positive definite on charge densities. So this cannot fail for *any*
    geometry -- which is a much stronger statement than "it passes on the
    spacings we tried", and is the reason a separation threshold was the wrong
    fix.
    """
    # Spacings coarser than ~0.44 A are refused outright since #757 (they
    # leave duplicate segments), so the sweep stays inside the resolved range;
    # the paper's own value is 0.3 A.
    cav = _fine(spacing)
    gaussian = np.linalg.eigvalsh(build_cavity_A_matrix(cav))
    point = np.linalg.eigvalsh(
        build_A_matrix(cav.points, cav.weights, representation=POINT_CHARGE)
    )
    assert np.min(gaussian) > 0.0
    # And it is better conditioned, not merely positive.
    assert np.min(gaussian) > np.min(point)


def test_gaussian_A_stays_definite_on_deliberately_coincident_segments():
    """The adversarial case: segments placed on top of each other.

    Two segments at the same point make ``A`` singular, because two coincident
    patches genuinely carry one degree of freedom. Singular is the honest
    answer; indefinite is not. The point-charge kernel gives neither -- it
    gives an infinity.
    """
    pts = np.array([[0.0, 0.0, 0.0], [1e-9, 0.0, 0.0], [0.0, 0.0, 3.0]])
    w = np.array([0.5, 0.5, 0.5])
    ev = np.linalg.eigvalsh(
        build_A_matrix(pts, w, representation=GAUSSIAN_CHARGE, zeta=4.901)
    )
    assert np.min(ev) > -1e-10          # never negative
    with np.errstate(divide="ignore"):
        point = build_A_matrix(pts, w, representation=POINT_CHARGE)
    assert not np.isfinite(point).all() or np.linalg.eigvalsh(point).min() < 0


# ---------------------------------------------------------------------
# Dispatch: the cavity decides
# ---------------------------------------------------------------------


def test_each_cavity_declares_the_representation_its_construction_requires():
    lebedev = build_cavity(
        atom_positions_bohr=WATER, atom_numbers=Z, n_points_per_sphere=110
    )
    assert lebedev.charge_representation == POINT_CHARGE
    assert _fine().charge_representation == GAUSSIAN_CHARGE


def test_lebedev_energies_are_untouched_by_the_fix():
    """The compatibility anchor.

    The Lebedev cavity keeps the point-charge kernel, so its ``A`` must be
    bit-identical to what ``build_A_matrix`` produced before the
    representation existed. Anything else would silently move every shipped
    solvation reference.
    """
    lebedev = build_cavity(
        atom_positions_bohr=WATER, atom_numbers=Z, n_points_per_sphere=110
    )
    via_cavity = build_cavity_A_matrix(lebedev)
    direct = build_A_matrix(lebedev.points, lebedev.weights)
    assert np.array_equal(via_cavity, direct)


def test_unknown_representation_is_refused():
    pts = np.zeros((2, 3))
    pts[1, 0] = 1.0
    with pytest.raises(ValueError, match="unknown representation"):
        build_A_matrix(pts, np.ones(2), representation="smeared-dipoles")
