"""COSMO-RS / COSMOSPACE thermodynamic layer (issue #558).

Equations and constants are Klamt 1995 (doi:10.1021/j100007a062) and Klamt,
Jonas, Buerger & Lohrenz 1998 (doi:10.1021/jp980017s). Equation numbers below
refer to the 1998 paper.

The layer is method-independent by construction: its only input is a
``ConductorSurface`` plus a named parameterization, so most of this file needs
no SCF and no compiled core.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from vibeqc.solvation import cosmors as rs
from vibeqc.solvation.cosmors.parameters import KLAMT_1998, R_KCAL, T_ROOM_K
from vibeqc.solvation.surface import (
    ConductorSurface,
    SurfaceProvenance,
    segment_normals,
)


# ---------------------------------------------------------------------
# Parameterization: the published constants, and their provenance
# ---------------------------------------------------------------------


def test_klamt1998_constants_match_the_paper():
    """Section 5.1 values, pinned so a typo cannot drift in silently."""
    p = KLAMT_1998
    assert p.a_eff == 7.1
    assert p.alpha_prime == 1288.0
    assert p.f_corr == 2.4
    assert p.r_av == 0.5
    assert p.c_hb == 7400.0
    assert p.sigma_hb == 0.0082
    assert p.lambda_comb == 0.14
    assert p.dispersion == {1: -0.041, 6: -0.037, 7: -0.027, 8: -0.042, 17: -0.052}
    assert p.cavity_radii == {1: 1.30, 6: 2.00, 7: 1.83, 8: 1.72, 17: 2.05}
    # beta = kT/a_eff; the paper quotes 0.0832 using its rounded kT = 0.592.
    assert p.beta(T_ROOM_K) == pytest.approx(0.0832, abs=3e-4)
    # r_eff = sqrt(a_eff/pi); the paper quotes 1.5 A.
    assert math.sqrt(p.a_eff / math.pi) == pytest.approx(1.5, abs=5e-3)


def test_parameterization_records_the_protocol_it_was_fitted_to():
    """A parameter set is inseparable from its protocol, so it carries it.

    Klamt 1998 fitted against DMol/BPW91/DNP with COSMO at f(eps)=1 and
    NSPA=92. Applying the constants to a surface built another way is a
    transfer across protocols, and the run has to be able to say so.
    """
    assert KLAMT_1998.fitted_protocol == "dmol/bpw91/dnp/cosmo-inf/nspa92"
    assert "10.1021/jp980017s" in KLAMT_1998.reference
    assert rs.get_parameterization("klamt1998") is KLAMT_1998
    assert rs.get_parameterization(None) is KLAMT_1998
    with pytest.raises(ValueError, match="unknown COSMO-RS parameterization"):
        rs.get_parameterization("nope")


def test_unfitted_elements_are_reported_not_assumed_zero():
    ok, missing = KLAMT_1998.supports_elements([1, 6, 8])
    assert ok and missing == []
    ok, missing = KLAMT_1998.supports_elements([1, 6, 16, 35])
    assert not ok and missing == [16, 35]


# ---------------------------------------------------------------------
# The surface record
# ---------------------------------------------------------------------


def _toy_surface(charges=None, areas=None):
    n = 4
    positions = np.array(
        [[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, -2.0, 0.0]]
    )
    owner = np.array([0, 0, 1, 1])
    atom_pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    a = np.full(n, 3.0) if areas is None else np.asarray(areas, float)
    q = np.array([0.01, -0.01, 0.02, -0.02]) if charges is None else np.asarray(
        charges, float
    )
    return ConductorSurface(
        positions=positions,
        areas=a,
        normals=segment_normals(positions, owner, atom_pos),
        segment_atom=owner,
        charges=q,
        atomic_numbers=np.array([8, 1]),
        atom_positions=atom_pos,
        energy_gas=-76.0,
        energy_conductor=-76.01,
        provenance=SurfaceProvenance(
            method="rhf", basis="sto-3g", cavity_kind="lebedev-csc",
            cavity_recipe={"n_points_per_sphere": 110},
            screening_variant="cosmo", epsilon=float("inf"),
            charge=0, multiplicity=1,
        ),
    )


def test_surface_round_trips_deterministically():
    surf = _toy_surface()
    again = ConductorSurface.from_json(surf.to_json())
    np.testing.assert_array_equal(again.positions, surf.positions)
    np.testing.assert_array_equal(again.charges, surf.charges)
    np.testing.assert_array_equal(again.areas, surf.areas)
    assert again.provenance == surf.provenance
    assert again.to_json() == surf.to_json()


def test_surface_rejects_an_unknown_schema_version():
    """Refuse to guess at a layout rather than silently misreading fields."""
    payload = _toy_surface().to_dict()
    payload["schema_version"] = 99
    with pytest.raises(ValueError, match="unsupported schema version"):
        ConductorSurface.from_dict(payload)


def test_surface_rejects_inconsistent_segment_counts():
    surf = _toy_surface()
    with pytest.raises(ValueError, match="expected"):
        ConductorSurface(
            positions=surf.positions, areas=np.ones(3), normals=surf.normals,
            segment_atom=surf.segment_atom, charges=surf.charges,
            atomic_numbers=surf.atomic_numbers, atom_positions=surf.atom_positions,
            energy_gas=None, energy_conductor=None, provenance=surf.provenance,
        )


def test_sigma_is_charge_over_area_and_finite_on_degenerate_segments():
    """A zero-area segment carries no charge to spread.

    Letting 0/0 through would put NaN into every downstream profile, and a
    switched cavity really does produce near-zero-area segments.
    """
    surf = _toy_surface(areas=[3.0, 3.0, 3.0, 0.0], charges=[0.03, 0.0, 0.0, 0.0])
    sigma = surf.sigma
    assert np.all(np.isfinite(sigma))
    assert sigma[0] == pytest.approx(0.01)
    assert sigma[3] == 0.0


def test_ideal_screening_energy_is_none_when_an_endpoint_is_missing():
    """None, never zero: zero would read as 'no screening gain'."""
    surf = _toy_surface()
    assert surf.ideal_screening_energy == pytest.approx(0.01, abs=1e-12)
    bare = ConductorSurface(
        positions=surf.positions, areas=surf.areas, normals=surf.normals,
        segment_atom=surf.segment_atom, charges=surf.charges,
        atomic_numbers=surf.atomic_numbers, atom_positions=surf.atom_positions,
        energy_gas=None, energy_conductor=-76.0, provenance=surf.provenance,
    )
    assert bare.ideal_screening_energy is None


def test_segment_normals_point_outward_and_are_unit():
    surf = _toy_surface()
    n = surf.normals
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-12)
    centres = surf.atom_positions[surf.segment_atom]
    outward = surf.positions - centres
    for i in range(surf.n_segments):
        assert float(n[i] @ outward[i]) > 0.0


# ---------------------------------------------------------------------
# Sigma averaging and profiles (eq. 11, 14, 16)
# ---------------------------------------------------------------------


def test_averaging_preserves_a_constant_field():
    """Eq. 11 is a normalised weighted mean, so a constant survives it."""
    rng = np.random.default_rng(3)
    pos = rng.normal(size=(12, 3)) * 2.0
    areas = rng.uniform(0.5, 3.0, size=12)
    const = np.full(12, 0.0137)
    np.testing.assert_allclose(
        rs.average_sigma(pos, areas, const, 0.5), const, rtol=1e-12
    )


def test_averaging_is_a_contraction_toward_the_mean():
    """Coarse-graining must reduce spread; that is its whole purpose."""
    rng = np.random.default_rng(5)
    pos = rng.normal(size=(30, 3))
    areas = np.full(30, 1.0)
    raw = rng.normal(scale=0.01, size=30)
    avg = rs.average_sigma(pos, areas, raw, 0.5)
    assert avg.std() < raw.std()


def test_sigma_profile_conserves_area():
    """A lost segment silently changes A^X, so binning must conserve it."""
    rng = np.random.default_rng(7)
    desc = rs.SegmentDescriptors(
        areas=rng.uniform(0.5, 3.0, size=40),
        sigma=rng.normal(scale=0.008, size=40),
        sigma_perp=np.zeros(40),
        sigma_raw=np.zeros(40),
    )
    prof = rs.sigma_profile(desc)
    assert prof.area() == pytest.approx(desc.total_area, rel=1e-12)
    assert prof.total_area == pytest.approx(desc.total_area, rel=1e-12)


def test_sigma_profile_clamps_rather_than_dropping_outliers():
    desc = rs.SegmentDescriptors(
        areas=np.array([2.0, 2.0]),
        sigma=np.array([-10.0, 10.0]),      # far outside any sane grid
        sigma_perp=np.zeros(2), sigma_raw=np.zeros(2),
    )
    prof = rs.sigma_profile(desc)
    assert prof.area() == pytest.approx(4.0, rel=1e-12)


def test_sigma_profile_is_continuous_in_the_segment_charge():
    """Linear binning keeps every derived property continuous.

    A nearest-bin histogram makes a chemical potential jump when a segment
    crosses a bin edge, which would show up as noise in any scan.
    """
    grid = rs.default_sigma_grid()
    edge = 0.5 * (grid[10] + grid[11])
    profiles = []
    for eps in (-1e-9, 1e-9):
        desc = rs.SegmentDescriptors(
            areas=np.array([2.0]), sigma=np.array([edge + eps]),
            sigma_perp=np.zeros(1), sigma_raw=np.zeros(1),
        )
        profiles.append(rs.sigma_profile(desc, grid).p)
    # Straddling a bin edge by 2e-9 in sigma may move at most that fraction
    # of the segment's area between the two bins. Nearest-bin assignment
    # would move the whole segment and fail this by five orders of magnitude.
    scale = float(np.max(profiles[0]))
    assert np.max(np.abs(profiles[0] - profiles[1])) < 1e-5 * scale


def test_mixture_profile_is_the_mole_fraction_weighted_sum():
    """Eq. 16."""
    grid = rs.default_sigma_grid()
    a = rs.sigma_profile(
        rs.SegmentDescriptors(np.array([4.0]), np.array([-0.005]),
                              np.zeros(1), np.zeros(1)), grid)
    b = rs.sigma_profile(
        rs.SegmentDescriptors(np.array([6.0]), np.array([+0.005]),
                              np.zeros(1), np.zeros(1)), grid)
    mix = rs.mixture_sigma_profile([a, b], np.array([0.25, 0.75]))
    np.testing.assert_allclose(mix.p, 0.25 * a.p + 0.75 * b.p, rtol=1e-12)
    assert mix.total_area == pytest.approx(0.25 * 4.0 + 0.75 * 6.0, rel=1e-12)
    # A pure component reproduces itself.
    pure = rs.mixture_sigma_profile([a, b], np.array([1.0, 0.0]))
    np.testing.assert_allclose(pure.p, a.p, rtol=1e-12)


def test_mixture_profile_rejects_mismatched_grids():
    a = rs.sigma_profile(
        rs.SegmentDescriptors(np.array([1.0]), np.array([0.0]),
                              np.zeros(1), np.zeros(1)),
        rs.default_sigma_grid(n_bins=21))
    b = rs.sigma_profile(
        rs.SegmentDescriptors(np.array([1.0]), np.array([0.0]),
                              np.zeros(1), np.zeros(1)),
        rs.default_sigma_grid(n_bins=31))
    with pytest.raises(ValueError, match="different sigma grids"):
        rs.mixture_sigma_profile([a, b], np.array([0.5, 0.5]))


# ---------------------------------------------------------------------
# Interaction energies (eq. 22, 26)
# ---------------------------------------------------------------------


def test_misfit_vanishes_for_the_ideally_paired_contact():
    """E_misfit is the penalty for failing to pair, not a binding energy."""
    for s in (0.0, 0.005, 0.02, -0.013):
        assert rs.misfit_energy(s, -s, KLAMT_1998) == pytest.approx(0.0, abs=1e-15)


def test_misfit_is_positive_symmetric_and_quadratic():
    p = KLAMT_1998
    assert rs.misfit_energy(0.01, 0.01, p) > 0.0
    assert rs.misfit_energy(0.01, 0.004, p) == pytest.approx(
        rs.misfit_energy(0.004, 0.01, p), rel=1e-14
    )
    # (alpha'/2)(s+s')^2 with no correlation term.
    assert rs.misfit_energy(0.01, 0.006, p) == pytest.approx(
        0.5 * p.alpha_prime * (0.016 ** 2), rel=1e-12
    )


def test_misfit_correlation_term_enters_only_with_sigma_perp():
    """Eq. 26 reduces to eq. 7 when the second descriptor is absent."""
    p = KLAMT_1998
    plain = rs.misfit_energy(0.01, 0.006, p)
    with_perp = rs.misfit_energy(0.01, 0.006, p, 0.002, 0.001)
    assert with_perp != plain
    assert with_perp == pytest.approx(
        0.5 * p.alpha_prime * 0.016 * (0.016 + p.f_corr * 0.003), rel=1e-12
    )
    assert rs.misfit_energy(0.01, 0.006, p, 0.0, 0.0) == pytest.approx(
        plain, rel=1e-12
    )


def test_hydrogen_bond_is_zero_for_nonpolar_and_stabilising_for_polar():
    """Eq. 22: needs opposite signs both beyond the threshold."""
    p = KLAMT_1998
    assert rs.hydrogen_bond_energy(0.002, -0.002, p) == 0.0   # below threshold
    assert rs.hydrogen_bond_energy(0.02, 0.02, p) == 0.0      # same sign
    strong = rs.hydrogen_bond_energy(0.015, -0.015, p)
    assert strong < 0.0
    assert rs.hydrogen_bond_energy(-0.015, 0.015, p) == pytest.approx(
        strong, rel=1e-14
    )
    assert strong == pytest.approx(
        p.c_hb * (0.015 - p.sigma_hb) * (-0.015 + p.sigma_hb), rel=1e-12
    )


def test_hydrogen_bond_magnitude_is_physical():
    """One strong contact should be a few kcal/mol, not tens.

    E_hb is per unit area; multiplying by a_eff gives the per-contact energy.
    Klamt 1998 section 5.1 puts the extra H-bond energy of water in water at
    about 1.9 kcal/mol, so a strongly polar contact of a couple of kcal/mol is
    the right order and a sign that the units are consistent.
    """
    p = KLAMT_1998
    per_contact = abs(rs.hydrogen_bond_energy(0.015, -0.015, p)) * p.a_eff
    assert 0.5 < per_contact < 10.0


# ---------------------------------------------------------------------
# The sigma potential (eq. 18, 23-25)
# ---------------------------------------------------------------------


def _bimodal_profile(grid=None):
    grid = rs.default_sigma_grid() if grid is None else grid
    desc = rs.SegmentDescriptors(
        areas=np.array([12.0, 12.0, 8.0]),
        sigma=np.array([-0.012, +0.012, 0.0]),
        sigma_perp=np.array([0.001, -0.001, 0.0]),
        sigma_raw=np.zeros(3),
    )
    return desc, rs.sigma_profile(desc, grid, label="toy")


def test_sigma_potential_satisfies_its_own_defining_equation():
    """The strongest available check: verify the fixed point, not the loop.

    Eq. 18 defines mu~ implicitly. Re-substituting the converged potential
    into the right-hand side must reproduce it, which tests the equation as
    implemented rather than merely that the iteration stopped moving.
    """
    _, prof = _bimodal_profile()
    p = KLAMT_1998
    pot = rs.sigma_potential_profile(prof, p)

    grid = prof.sigma_grid
    beta = p.beta(T_ROOM_K)
    e = (
        rs.misfit_energy(grid[:, None], grid[None, :], p)
        + rs.hydrogen_bond_energy(grid[:, None], grid[None, :], p)
    ) / beta
    w = prof.normalized * prof.dsigma
    rhs = -np.log(np.sum(w[None, :] * np.exp(-e + pot.mu_tilde[None, :]), axis=1))
    np.testing.assert_allclose(pot.mu_tilde, rhs, atol=1e-8, rtol=0.0)


def test_sigma_potential_of_a_neutral_ensemble_is_symmetric():
    """A profile symmetric in sigma must give a potential symmetric in sigma.

    The interaction energy depends on sigma+sigma' and the H-bond term is
    symmetric under swapping donor and acceptor, so nothing in the model can
    break that symmetry. If it breaks, an asymmetric term crept in.
    """
    grid = rs.default_sigma_grid(n_bins=61)
    desc = rs.SegmentDescriptors(
        areas=np.array([10.0, 10.0]), sigma=np.array([-0.011, +0.011]),
        sigma_perp=np.zeros(2), sigma_raw=np.zeros(2),
    )
    pot = rs.sigma_potential_profile(rs.sigma_profile(desc, grid), KLAMT_1998)
    np.testing.assert_allclose(
        pot.mu_tilde, pot.mu_tilde[::-1], atol=1e-8, rtol=0.0
    )


def test_sigma_potential_units_relate_mu_and_mu_tilde():
    _, prof = _bimodal_profile()
    pot = rs.sigma_potential_profile(prof, KLAMT_1998)
    np.testing.assert_allclose(pot.mu, pot.beta * pot.mu_tilde, rtol=1e-14)
    assert pot.beta == pytest.approx(KLAMT_1998.beta(T_ROOM_K), rel=1e-14)


def test_segment_form_matches_the_profile_form_without_correlation():
    """With f_corr = 0 and one descriptor the two formulations agree.

    They are different models when the correlation term is on (section 3.4),
    so this pins that the difference is *only* that term and not an
    inconsistency in how the two are assembled.
    """
    from dataclasses import replace

    p0 = replace(KLAMT_1998, f_corr=0.0)
    desc = rs.SegmentDescriptors(
        areas=np.array([9.0, 9.0, 6.0]),
        sigma=np.array([-0.010, +0.010, 0.001]),
        sigma_perp=np.zeros(3), sigma_raw=np.zeros(3),
    )
    grid = rs.default_sigma_grid(n_bins=201)
    prof = rs.sigma_profile(desc, grid)
    a = rs.sigma_potential_profile(prof, p0)
    b = rs.sigma_potential_segments([desc], np.array([1.0]), p0, sigma_grid=grid)
    probe = np.array([-0.010, 0.0, 0.010])
    np.testing.assert_allclose(a(probe), b(probe), atol=2e-3)


def test_pair_interaction_derivative_is_exact_away_from_its_corner():
    """``dE_int/dsigma`` against finite differences (Direct COSMO-RS, #558).

    Sinnecker, Rajendran, Klamt, Diedenhofen & Neese 2006
    (doi:10.1021/jp056016z) eq. 19 makes the feedback potential
    ``phi_t = a_t (d mu_S / dq)|_{q_t}``, which with ``sigma = q/a`` is
    ``mu_S'(sigma_t)``. Its chain bottoms out in this pair derivative, so an
    error here is an error in every Direct COSMO-RS number.
    """
    p = KLAMT_1998
    rng = np.random.default_rng(0)
    a = rng.uniform(-0.03, 0.03, 2000)
    b = rng.uniform(-0.03, 0.03, 2000)
    h = 1e-7
    analytic = rs.interaction_energy_dsigma(a, b, p)
    fd = (
        rs.interaction_energy(a + h, b, p) - rs.interaction_energy(a - h, b, p)
    ) / (2 * h)
    # A central difference straddling a corner measures the average of the two
    # one-sided slopes, so exclude the handful of samples that sit on one.
    on_corner = np.abs(np.abs(a) - p.sigma_hb) < 1e-5
    assert on_corner.sum() < 10, "the exclusion must stay a handful, not a sieve"
    err = np.abs(analytic - fd)[~on_corner]
    assert err.max() < 1e-6 * max(np.abs(fd[~on_corner]).max(), 1.0)


def test_the_hydrogen_bond_corner_is_real_and_survives_the_average():
    """What Direct COSMO-RS inherits from Klamt's H-bond term, measured.

    ``E_hb = c_hb max[0, acc - sigma_hb] min[0, don + sigma_hb]`` is piecewise
    linear in each density, so its derivative steps where a density crosses
    ``+-sigma_hb``. That corner is in the published functional form and no
    algebra removes it.

    The part worth pinning is that the sigma potential does **not** smooth it
    away. ``d mu~/d sigma`` is a Boltzmann average over solvent partners, and it
    is tempting to expect an average of many partners to round a corner off --
    but every partner's term steps at the *same* sigma, so the average steps
    with them. Measured on a bimodal profile: the potential's derivative jumps
    by 40.3 across ``sigma = sigma_hb``, against values of order 70 either side.

    This is the reason a Direct COSMO-RS feedback cannot be assumed to converge
    smoothly: a segment whose sigma sits near a threshold sees a discontinuous
    potential from one macro-iteration to the next.
    """
    _, prof = _bimodal_profile()
    p = KLAMT_1998
    pot = rs.sigma_potential_profile(prof, p)
    eps = 1e-9

    # At the pair level first, so the cause is unambiguous.
    partner = np.array([-0.02])
    lo = rs.interaction_energy_dsigma(np.array([p.sigma_hb - eps]), partner, p)[0]
    hi = rs.interaction_energy_dsigma(np.array([p.sigma_hb + eps]), partner, p)[0]
    assert abs(hi - lo) > 1.0, (lo, hi)

    # And then that the average carries it through rather than damping it.
    d_lo = pot.derivative(np.array([p.sigma_hb - eps]))[0]
    d_hi = pot.derivative(np.array([p.sigma_hb + eps]))[0]
    assert abs(d_hi - d_lo) > 1.0, (d_lo, d_hi)
    # A jump and a steep slope look alike at one step size, so separate them by
    # halving it: across a corner the change is the jump and does not shrink,
    # while away from one it is curvature and halves with the interval.
    def change_across(centre, half_width):
        lo = pot.derivative(np.array([centre - half_width]))[0]
        hi = pot.derivative(np.array([centre + half_width]))[0]
        return abs(hi - lo)

    wide = change_across(p.sigma_hb, 1e-6)
    narrow = change_across(p.sigma_hb, 5e-7)
    assert narrow > 0.9 * wide, (narrow, wide)          # a step, not a slope

    for centre in (0.0, 0.02, -0.02):
        wide = change_across(centre, 1e-6)
        narrow = change_across(centre, 5e-7)
        assert narrow < 0.6 * wide, (centre, narrow, wide)   # a slope, not a step


def test_sigma_potential_evaluates_and_differentiates_itself_exactly():
    """The potential carries its solvent ensemble, so it is a function.

    Both constructors end in ``mu~(sigma) = -logsumexp_j(log w_j -
    E~(sigma, sigma_j) + mu~_j)`` and then keep only a table of it. Direct
    COSMO-RS needs the derivative, and differentiating a linear interpolant
    gives a piecewise-constant answer that is discontinuous at every grid point
    -- so the ensemble is kept and the derivative taken from the expression.

    Two things are checked, and the first is what makes the second safe: the
    evaluated form must agree with the table it was tabulated into, or the
    energy and its feedback potential would be describing different functions.
    """
    _, prof = _bimodal_profile()
    pot = rs.sigma_potential_profile(prof, KLAMT_1998)
    assert pot.has_ensemble

    # At the grid points the two must agree to machine precision, because that
    # is the fixed point the table *is*. This is the check that makes the
    # evaluated form trustworthy as the same function.
    np.testing.assert_allclose(
        pot.evaluate(pot.sigma_grid), pot.mu_tilde, atol=1e-9, rtol=0.0
    )

    # Between them they must not: ``__call__`` interpolates linearly and
    # ``evaluate`` does not, so the gap is the tabulation error and measuring it
    # is the point. On the default 1e-03 grid it reaches 9.9e-03, which is why
    # Direct COSMO-RS uses the evaluated form -- a feedback potential built on a
    # curve that is 1e-02 away from the energy's own would be inconsistent at a
    # level far above anything else in the loop.
    midpoints = 0.5 * (pot.sigma_grid[:-1] + pot.sigma_grid[1:])
    gap = np.abs(pot.evaluate(midpoints) - pot(midpoints)).max()
    assert 1e-4 < gap < 1e-1, gap

    sigma = np.linspace(-0.025, 0.025, 41)

    # And exact against a finite difference of itself, away from the corner.
    h = 1e-8
    fd = (pot.evaluate(sigma + h) - pot.evaluate(sigma - h)) / (2 * h)
    analytic = pot.derivative(sigma)
    clean = np.abs(np.abs(sigma) - KLAMT_1998.sigma_hb) > 1e-4
    err = np.abs(analytic - fd)[clean]
    assert err.max() < 1e-6 * np.abs(fd[clean]).max()

    # Scalars in, scalars out -- the feedback loop calls this per segment.
    assert isinstance(pot.derivative(0.01), float)
    assert isinstance(pot.evaluate(0.01), float)


def test_a_sigma_potential_without_its_ensemble_refuses_to_evaluate():
    """Interpolating is always available; evaluating is not, and says so.

    A hand-built potential has a table and no solvent behind it. Returning an
    interpolated derivative there would be the quiet-wrong-answer failure this
    workstream keeps finding, so it fails closed instead.
    """
    grid = rs.default_sigma_grid()
    bare = rs.SigmaPotential(grid, np.zeros_like(grid), KLAMT_1998, T_ROOM_K, 0, 0.0)
    assert not bare.has_ensemble
    assert np.allclose(bare(np.array([0.0, 0.01])), 0.0)      # interpolation works
    with pytest.raises(ValueError, match="without its solvent ensemble"):
        bare.derivative(np.array([0.0]))
    with pytest.raises(ValueError, match="without its solvent ensemble"):
        bare.evaluate(np.array([0.0]))


def test_sigma_potential_reports_nonconvergence_rather_than_returning_garbage():
    _, prof = _bimodal_profile()
    with pytest.raises(RuntimeError, match="no self-consistent sigma potential"):
        rs.sigma_potential_profile(prof, KLAMT_1998, max_iter=2, tol=1e-14)


def test_sigma_potential_rejects_an_empty_ensemble():
    empty = rs.SigmaProfile(rs.default_sigma_grid(),
                            np.zeros(rs.default_sigma_grid().size), 0.0)
    with pytest.raises(ValueError, match="no area"):
        rs.sigma_potential_profile(empty, KLAMT_1998)


# ---------------------------------------------------------------------
# Chemical potentials and derived properties (eq. 19-21, 30)
# ---------------------------------------------------------------------


def test_chemical_potential_restoring_part_scales_with_solute_area():
    """Eq. 20 integrates the unnormalised solute profile, so it is extensive."""
    desc, prof = _bimodal_profile()
    pot = rs.sigma_potential_profile(prof, KLAMT_1998)
    big = rs.sigma_profile(
        rs.SegmentDescriptors(desc.areas * 3.0, desc.sigma,
                              desc.sigma_perp, desc.sigma_raw),
        prof.sigma_grid,
    )
    mu1 = rs.chemical_potential(prof, pot, KLAMT_1998, solvent_area=prof.total_area)
    mu3 = rs.chemical_potential(big, pot, KLAMT_1998, solvent_area=prof.total_area)
    assert mu3.restoring == pytest.approx(3.0 * mu1.restoring, rel=1e-12)
    # The combinatorial term depends on the solvent, not the solute.
    assert mu3.combinatorial == pytest.approx(mu1.combinatorial, rel=1e-14)


def test_chemical_potential_records_its_provenance():
    _, prof = _bimodal_profile()
    pot = rs.sigma_potential_profile(prof, KLAMT_1998)
    mu = rs.chemical_potential(prof, pot, KLAMT_1998, solvent_area=prof.total_area)
    assert mu.parameterization == "klamt1998"
    assert mu.temperature_k == pytest.approx(T_ROOM_K)
    assert mu.total == pytest.approx(mu.restoring + mu.combinatorial, rel=1e-14)


def test_activity_coefficient_of_a_solute_in_itself_is_one():
    """gamma -> 1 in the pure-solute limit: the cheapest real physical check."""
    _, prof = _bimodal_profile()
    pot = rs.sigma_potential_profile(prof, KLAMT_1998)
    mu = rs.chemical_potential(prof, pot, KLAMT_1998, solvent_area=prof.total_area)
    assert rs.ln_activity_coefficient(mu, mu) == pytest.approx(0.0, abs=1e-14)
    assert rs.activity_coefficient(mu, mu) == pytest.approx(1.0, rel=1e-14)


def test_partition_coefficient_is_antisymmetric():
    _, prof = _bimodal_profile()
    pot = rs.sigma_potential_profile(prof, KLAMT_1998)
    a = rs.chemical_potential(prof, pot, KLAMT_1998, solvent_area=40.0)
    b = rs.chemical_potential(prof, pot, KLAMT_1998, solvent_area=90.0)
    assert rs.log_partition_coefficient(a, b) == pytest.approx(
        -rs.log_partition_coefficient(b, a), rel=1e-12
    )
    with pytest.raises(ValueError, match="molar volume ratio"):
        rs.log_partition_coefficient(a, b, molar_volume_ratio=0.0)


def test_solvation_free_energy_is_the_difference_of_two_references():
    _, prof = _bimodal_profile()
    pot = rs.sigma_potential_profile(prof, KLAMT_1998)
    mu = rs.chemical_potential(prof, pot, KLAMT_1998, solvent_area=prof.total_area)
    assert rs.solvation_free_energy(mu, 2.5) == pytest.approx(mu.total - 2.5)


def test_gas_reference_refuses_an_element_the_fit_never_covered():
    """Silence would read as 'this element does not disperse'."""
    surf = _toy_surface()
    bad = ConductorSurface(
        positions=surf.positions, areas=surf.areas, normals=surf.normals,
        segment_atom=surf.segment_atom, charges=surf.charges,
        atomic_numbers=np.array([16, 1]),           # sulfur: not in the 1998 fit
        atom_positions=surf.atom_positions,
        energy_gas=-76.0, energy_conductor=-76.01, provenance=surf.provenance,
    )
    desc = rs.segment_descriptors(bad, KLAMT_1998.r_av)
    with pytest.raises(ValueError, match="no dispersion constant"):
        rs.gas_phase_chemical_potential(bad, desc, KLAMT_1998)


def test_gas_reference_refuses_a_missing_ideal_screening_energy():
    surf = _toy_surface()
    bare = ConductorSurface(
        positions=surf.positions, areas=surf.areas, normals=surf.normals,
        segment_atom=surf.segment_atom, charges=surf.charges,
        atomic_numbers=surf.atomic_numbers, atom_positions=surf.atom_positions,
        energy_gas=None, energy_conductor=None, provenance=surf.provenance,
    )
    desc = rs.segment_descriptors(bare, KLAMT_1998.r_av)
    with pytest.raises(ValueError, match="no ideal screening energy"):
        rs.gas_phase_chemical_potential(bare, desc, KLAMT_1998)


def test_per_element_area_partitions_the_total():
    surf = _toy_surface()
    desc = rs.segment_descriptors(surf, KLAMT_1998.r_av)
    areas = rs.per_element_area(surf, desc)
    assert set(areas) == {8, 1}
    assert sum(areas.values()) == pytest.approx(desc.total_area, rel=1e-12)


# ---------------------------------------------------------------------
# End to end, against a real conductor-limit COSMO run
# ---------------------------------------------------------------------


def test_end_to_end_from_a_conductor_limit_scf():
    """Water at eps = inf through to a chemical potential.

    Exercises the layer boundary: an SCF produces a surface, and everything
    after it is method-independent.
    """
    vq = pytest.importorskip("vibeqc")
    from vibeqc.solvation.surface import build_conductor_surface

    mol = vq.Molecule(
        [vq.Atom(8, [0.0, 0.0, 0.0]),
         vq.Atom(1, [0.0, 1.498, -1.159]),
         vq.Atom(1, [0.0, -1.498, -1.159])], 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")
    sm = vq.SolventModel(
        epsilon=math.inf, name="conductor", variant="cosmo",
        n_points_per_sphere=110, max_macro_iter=40, tol_e_solv=1e-9)
    sol = vq.run_cpcm_scf(mol, basis, method="rhf", solvent=sm)
    assert sol.converged
    # The conductor limit is exact, not merely large.
    assert sol.screening.f == 1.0
    assert sol.screening.is_conductor

    surf = build_conductor_surface(
        sol, method="rhf", basis="sto-3g", energy_conductor=float(sol.energy))
    assert surf.n_segments > 100
    assert surf.provenance.epsilon == math.inf
    assert surf.provenance.screening_variant == "cosmo"

    desc = rs.segment_descriptors(surf, KLAMT_1998.r_av)
    # Averaged sigma must land in the physical range Klamt 1998 Figure 3 plots
    # (about +-0.02 e/A^2). The raw per-segment q/s does not: a switched
    # cavity has near-zero-area segments whose raw ratio is meaningless, which
    # is exactly why eq. 11 averages before anything downstream uses it.
    assert np.max(np.abs(desc.sigma)) < 0.03

    prof = rs.sigma_profile(desc, label="water")
    assert prof.area() == pytest.approx(desc.total_area, rel=1e-9)

    pot = rs.sigma_potential_segments([desc], np.array([1.0]), KLAMT_1998)
    assert np.all(np.isfinite(pot.mu_tilde))

    mu = rs.chemical_potential(
        prof, pot, KLAMT_1998, solvent_area=desc.total_area)
    assert math.isfinite(mu.total)
    assert rs.ln_activity_coefficient(mu, mu) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------
# COSMOSPACE -- Klamt, Krooshof & Taylor, AIChE J. 48, 2332 (2002)
# ---------------------------------------------------------------------

from vibeqc.solvation.cosmors.cosmospace import (  # noqa: E402
    SG_COORDINATION_NUMBER,
    binary_segment_activity_coefficient,
    exchange_energy_matrix,
    residual_ln_activity_coefficient,
    segment_activity_coefficients,
    staverman_guggenheim_ln_gamma,
    tau_matrix,
)


def test_exchange_energy_is_a_difference_not_a_sum():
    """The self-energy subtraction of eq. 17 is the whole physical content.

    For the pure misfit energy ``(alpha'/2)(s+s')^2``, referencing pair
    energies to the pure-segment pairs collapses to

        u_ab - (u_aa + u_bb)/2 = -(a_eff alpha'/2) (s_a - s_b)^2

    i.e. it depends on how *unlike* two segments are, not on their sum, and is
    never positive. Getting the subtraction wrong turns the difference back
    into a sum and still converges, so this is pinned against the closed form.
    Verified where the hydrogen-bond term is inactive, so the misfit is the
    only contribution.
    """
    p = KLAMT_1998
    for pair in ([0.005, -0.005], [0.004, -0.001], [0.006, 0.002]):
        s = np.array(pair)
        assert rs.hydrogen_bond_energy(s[0], s[1], p) == 0.0, "HB must be off"
        x = exchange_energy_matrix(s, p)
        expected = -(p.a_eff * p.alpha_prime / 2.0) * (s[0] - s[1]) ** 2
        assert x[0, 1] == pytest.approx(expected, rel=1e-12)
        assert x[0, 1] <= 0.0


def test_exchange_energy_has_zero_diagonal_and_is_symmetric():
    s = np.array([0.012, -0.004, 0.0, -0.011])
    x = exchange_energy_matrix(s, KLAMT_1998)
    np.testing.assert_allclose(np.diag(x), 0.0, atol=1e-13)
    np.testing.assert_allclose(x, x.T, rtol=1e-13, atol=1e-15)


def test_tau_is_symmetric_and_unity_for_identical_segments():
    s = np.array([0.007, 0.007, -0.003])
    tau = tau_matrix(exchange_energy_matrix(s, KLAMT_1998))
    np.testing.assert_allclose(tau, tau.T, rtol=1e-14)
    np.testing.assert_allclose(np.diag(tau), 1.0, rtol=1e-13)
    # Segments 0 and 1 are identical, so exchanging them costs nothing.
    assert tau[0, 1] == pytest.approx(1.0, rel=1e-13)


def test_cosmospace_iteration_matches_the_closed_form_binary_solution():
    """Eq. 16 solved iteratively must reproduce the eq. 19 analytic result.

    The closed form exists independently of the iteration, so this is a real
    oracle rather than a self-consistency restatement, and it is the strongest
    check available on the solver.
    """
    p = KLAMT_1998
    s = np.array([0.010, -0.010])
    tau = tau_matrix(exchange_energy_matrix(s, p))
    for theta_a in (1e-12, 1e-6, 0.05, 0.2, 0.35, 0.5, 0.8, 0.95):
        theta = np.array([theta_a, 1.0 - theta_a])
        sol = segment_activity_coefficients(theta, tau)
        analytic = binary_segment_activity_coefficient(theta_a, tau[0, 1])
        assert sol.gamma[0] == pytest.approx(analytic, rel=1e-9)


def test_cosmospace_dilute_limit_is_the_reciprocal_tau():
    """Klamt 2002 after eq. 19: gamma^A -> 1/tau_AB as Theta^A -> 0."""
    p = KLAMT_1998
    tau = tau_matrix(exchange_energy_matrix(np.array([0.009, -0.009]), p))
    assert binary_segment_activity_coefficient(0.0, tau[0, 1]) == pytest.approx(
        1.0 / tau[0, 1], rel=1e-14
    )
    # Approached continuously from above, all the way down. Eq. 19 as
    # printed returns NaN here: it subtracts a square root from 1 and divides
    # by 2 omega Theta_A^2, and the subtraction cancels catastrophically long
    # before the ratio does. The implementation rearranges it so Theta_A
    # cancels analytically, which is what makes infinite dilution -- the
    # regime activity coefficients matter most in -- computable at all.
    for tiny in (1e-6, 1e-9, 1e-12, 1e-15):
        g = binary_segment_activity_coefficient(tiny, tau[0, 1])
        assert np.isfinite(g)
        assert g == pytest.approx(1.0 / tau[0, 1], rel=1e-6)
    # And the other end: pure A has unit activity by definition.
    assert binary_segment_activity_coefficient(1.0, tau[0, 1]) == pytest.approx(
        1.0, rel=1e-14
    )


def test_cosmospace_solution_satisfies_equation_16():
    """Thermodynamic consistency follows from eq. 16 holding (Appendix B).

    Assert on the equation's own residual, not on the iteration's step size:
    a damped iteration can stop moving while still not solving anything.
    """
    rng = np.random.default_rng(11)
    s = rng.normal(scale=0.008, size=6)
    tau = tau_matrix(exchange_energy_matrix(s, KLAMT_1998))
    theta = rng.uniform(0.05, 1.0, size=6)
    theta /= theta.sum()
    sol = segment_activity_coefficients(theta, tau)
    assert sol.gibbs_duhem_residual() < 1e-9


def test_ideal_segments_give_unit_activity_coefficients():
    """tau == 1 everywhere means no exchange interaction, so gamma == 1."""
    theta = np.array([0.2, 0.3, 0.5])
    sol = segment_activity_coefficients(theta, np.ones((3, 3)))
    np.testing.assert_allclose(sol.gamma, 1.0, rtol=1e-10)


def test_segment_fractions_must_be_normalised():
    tau = np.ones((2, 2))
    with pytest.raises(ValueError, match="must sum to 1"):
        segment_activity_coefficients(np.array([0.3, 0.3]), tau)
    with pytest.raises(ValueError, match="negative Theta"):
        segment_activity_coefficients(np.array([1.3, -0.3]), tau)


def test_cosmospace_reports_nonconvergence():
    p = KLAMT_1998
    tau = tau_matrix(exchange_energy_matrix(np.array([0.02, -0.02]), p))
    with pytest.raises(RuntimeError, match="no solution after"):
        segment_activity_coefficients(
            np.array([0.5, 0.5]), tau, max_iter=1, tol=1e-15
        )


def test_residual_activity_coefficient_vanishes_for_a_compound_in_itself():
    """Eq. 13 references each segment to its own pure ensemble.

    gamma_i^R -> 1 for pure i is the definition, so this must be exactly zero
    rather than merely small.
    """
    n_i = np.array([3.0, 5.0])
    ln_g = np.array([-0.4, 0.25])
    assert residual_ln_activity_coefficient(n_i, ln_g, ln_g) == 0.0


def test_residual_activity_coefficient_weights_by_segment_count():
    n_i = np.array([2.0, 4.0])
    mix = np.array([0.1, -0.3])
    pure = np.array([0.0, 0.0])
    assert residual_ln_activity_coefficient(n_i, mix, pure) == pytest.approx(
        2.0 * 0.1 + 4.0 * -0.3, rel=1e-14
    )


def test_staverman_guggenheim_vanishes_when_components_are_the_same_size():
    """No size or shape mismatch means no combinatorial contribution."""
    x = np.array([0.3, 0.7])
    r = np.array([2.5, 2.5])
    q = np.array([2.0, 2.0])
    np.testing.assert_allclose(
        staverman_guggenheim_ln_gamma(x, r, q), 0.0, atol=1e-12
    )


def test_staverman_guggenheim_is_nonzero_for_mismatched_sizes():
    x = np.array([0.5, 0.5])
    ln_g = staverman_guggenheim_ln_gamma(
        x, np.array([1.0, 5.0]), np.array([1.0, 4.0])
    )
    assert np.any(np.abs(ln_g) > 1e-3)
    assert SG_COORDINATION_NUMBER == 10.0


def test_staverman_guggenheim_is_normalisation_invariant():
    """Mole fractions are normalised internally, so an unnormalised input
    that describes the same composition must give the same answer."""
    r = np.array([1.0, 3.0])
    q = np.array([1.0, 2.5])
    a = staverman_guggenheim_ln_gamma(np.array([0.25, 0.75]), r, q)
    b = staverman_guggenheim_ln_gamma(np.array([1.0, 3.0]), r, q)
    np.testing.assert_allclose(a, b, rtol=1e-12)
