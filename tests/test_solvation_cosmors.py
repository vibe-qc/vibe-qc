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


def _toy_sigma_potential(params=KLAMT_1998):
    desc = rs.SegmentDescriptors(
        areas=np.array([12.0, 12.0, 8.0]),
        sigma=np.array([-0.012, +0.012, 0.0]),
        sigma_perp=np.array([0.001, -0.001, 0.0]),
        sigma_raw=np.zeros(3),
    )
    return rs.sigma_potential_profile(
        rs.sigma_profile(desc, rs.default_sigma_grid(), label="toy"), params
    )


def _null_sigma_potential():
    """A solvent with no interactions at all, so ``mu_S`` vanishes identically.

    ``E_int == 0`` makes the fixed point ``mu~ = -log(sum_j w_j) = 0``, which is
    what gives Direct COSMO-RS an exact limit to be checked against.
    """
    import dataclasses

    return _toy_sigma_potential(
        dataclasses.replace(KLAMT_1998, alpha_prime=0.0, c_hb=0.0, f_corr=0.0)
    )


def _conductor_water(vq, **overrides):
    """Water at eps=inf on the fine cavity, at KLAMT_1998's own radii."""
    mol = vq.Molecule(
        [vq.Atom(8, (0.0, 0.0, 0.0)),
         vq.Atom(1, (0.0, 1.498, -1.159)),
         vq.Atom(1, (0.0, -1.498, -1.159))], 0, 1,
    )
    kw = dict(
        epsilon=math.inf, variant="cosmo", cavity="fine",
        fine_grid_spacing_ang=0.35, radii=dict(KLAMT_1998.cavity_radii),
        radii_scale=1.0, solvent_probe_radius_ang=0.0,
        max_macro_iter=60, tol_e_solv=1e-10,
    )
    kw.update(overrides)
    return mol, vq.run_cpcm_scf(
        mol, vq.BasisSet(mol, "sto-3g"), method="rhf",
        solvent=vq.SolventModel(**kw),
    )


def test_a_computed_water_potential_has_the_shape_water_should_have():
    """A real solvent potential, and a physics check on it (#558).

    Everything before this used hand-built toy ensembles. This one comes from
    an actual conductor COSMO run on water, through the same chain a user takes:
    surface -> segment descriptors -> sigma potential.

    The check is the shape, because that is what says the chain produced water
    rather than an arbitrary curve. Water's sigma potential is **positive** near
    ``sigma = 0`` -- the cost of putting a nonpolar surface in water, the
    hydrophobic effect -- and **negative at both extremes**, where a strongly
    polarized surface can donate or accept a hydrogen bond. A potential that
    was monotonic, or negative in the middle, would not be water.
    """
    vq = pytest.importorskip("vibeqc")

    _, solvent = _conductor_water(vq)
    assert solvent.converged and solvent.screening.is_conductor
    pot = rs.solvent_sigma_potential(
        solvent, KLAMT_1998, method="rhf", basis="sto-3g"
    )

    mu = pot.beta * pot.evaluate(np.array([-0.020, -0.010, 0.0, 0.010, 0.020]))
    assert mu[2] > 0.0, mu                       # nonpolar surfaces pay
    assert mu[0] < 0.0 and mu[4] < 0.0, mu       # both H-bond wings gain
    assert mu[0] < mu[1] and mu[4] < mu[3], mu   # and deepen outward

    # The protocol transfer is recorded rather than hidden: these constants were
    # fitted on a different method, basis and cavity, and no code can repair
    # that -- it can only make it visible.
    assert pot.protocol == "rhf/sto-3g/fine-cfc"
    assert pot.protocol != KLAMT_1998.fitted_protocol


def test_a_computed_water_potential_drives_direct_cosmors():
    """The real solvent, in the loop it was built for.

    Positive is the right sign here and is worth stating, because it looks like
    a bug: the COSMO-RS term for a small solute in water comes out *unfavourable*
    by about half a kcal/mol on top of the conductor energy. That follows from
    the shape above -- most of a small solute's surface sits at moderate sigma,
    where water's potential is positive -- and it is the hydrophobic penalty,
    not a sign error.
    """
    vq = pytest.importorskip("vibeqc")

    mol, solvent = _conductor_water(vq)
    pot = rs.solvent_sigma_potential(
        solvent, KLAMT_1998, method="rhf", basis="sto-3g"
    )
    direct = vq.run_cpcm_scf(
        mol, vq.BasisSet(mol, "sto-3g"), method="rhf",
        solvent=vq.SolventModel(
            epsilon=math.inf, variant="dcosmo-rs", sigma_potential=pot,
            cavity="fine", fine_grid_spacing_ang=0.35,
            radii=dict(KLAMT_1998.cavity_radii), radii_scale=1.0,
            solvent_probe_radius_ang=0.0, max_macro_iter=80, tol_e_solv=1e-10,
        ),
    )
    assert direct.converged
    fb = direct.direct_feedback
    assert fb.energy > 0.0
    assert abs(fb.energy) < 0.01                       # a few kcal/mol, not wild
    assert float(direct.energy) > float(solvent.energy)
    # The feedback stays inside what the parameterization covers.
    assert fb.fraction_outside_grid < 0.02, fb.fraction_outside_grid


def test_the_solvent_potential_refuses_a_screened_run():
    """The trap this helper exists for.

    ``build_conductor_surface`` takes the stored charges, and those are the
    ideal ``q*`` only at eps=inf; at finite dielectric vibe-qc stores
    ``q = f q*``. A profile built from them is scaled by ``f`` and **nothing
    downstream can tell**: it looks like a perfectly reasonable sigma profile.
    """
    vq = pytest.importorskip("vibeqc")

    _, screened = _conductor_water(vq, epsilon=78.39)
    with pytest.raises(ValueError, match="must be in the conductor limit"):
        rs.solvent_sigma_potential(
            screened, KLAMT_1998, method="rhf", basis="sto-3g"
        )


def test_the_solvent_potential_refuses_foreign_cavity_radii():
    """A parameter set is fitted against a radius set, and sigma = q/a.

    Building the solvent surface with the default scaled-Bondi radii instead of
    the parameterization's own changes every sigma the constants were fitted to
    reproduce, and the resulting potential looks entirely ordinary.
    """
    vq = pytest.importorskip("vibeqc")

    _, bondi = _conductor_water(vq, radii=None, radii_scale=1.20)
    with pytest.raises(ValueError, match="radii the parameterization was not"):
        rs.solvent_sigma_potential(
            bondi, KLAMT_1998, method="rhf", basis="sto-3g"
        )


def test_direct_feedback_potential_is_the_energys_derivative():
    """The one check that validates the whole derivation at once (#558).

    Sinnecker et al. 2006 state the operator (eq. 18) but not the energy. Both
    follow from taking the free energy to be the conductor COSMO energy plus
    ``sum_t a_t mu_S(sigma_t)``: differentiating that third term with respect to
    the density gives the potential of a charge distribution ``q^dRS = -A^-1
    phi^dRS``, which is exactly eq. 18's extra charges. So ``phi^dRS`` must be
    ``dE_rs/dq`` -- and if it is not, the operator is not the derivative of the
    energy it is paired with, which no SCF can recover from.

    This also pins every unit conversion in one shot: the ``beta`` that turns
    the dimensionless ``mu~`` into kcal/(mol angstrom^2), the bohr-to-angstrom
    area conversion, and the Hartree conversion.
    """
    from vibeqc.solvation.cosmors.direct import direct_cosmors_feedback

    pot = _toy_sigma_potential()
    rng = np.random.default_rng(1)
    n = 20
    q = rng.normal(0.0, 0.002, n)
    areas = rng.uniform(0.5, 3.0, n)
    feedback = direct_cosmors_feedback(q, areas, pot)

    h = 1e-9
    for t in range(n):
        up, dn = q.copy(), q.copy()
        up[t] += h
        dn[t] -= h
        fd = (
            direct_cosmors_feedback(up, areas, pot).energy
            - direct_cosmors_feedback(dn, areas, pot).energy
        ) / (2.0 * h)
        assert abs(feedback.phi[t] - fd) < 1e-6 * max(abs(fd), 1.0), (t, fd)


def test_direct_cosmors_reduces_to_conductor_cosmo_for_a_null_solvent():
    """The exact limit, and the only end-to-end check that needs no tolerance.

    A solvent whose interaction energy vanishes has ``mu_S == 0`` and
    ``mu_S' == 0``, so the correction charges are zero, the energy term is zero,
    and Direct COSMO-RS must return the conductor COSMO number *exactly* -- not
    approximately. Anything else means the wiring adds or drops something:
    a stray screening factor, a double-counted operator trace, a unit slip.
    """
    vq = pytest.importorskip("vibeqc")

    mol = vq.Molecule(
        [vq.Atom(8, (0.0, 0.0, 0.0)),
         vq.Atom(1, (0.0, 1.498, -1.159)),
         vq.Atom(1, (0.0, -1.498, -1.159))], 0, 1,
    )
    basis = vq.BasisSet(mol, "sto-3g")
    common = dict(
        epsilon=math.inf, cavity="fine", fine_grid_spacing_ang=0.40,
        max_macro_iter=60, tol_e_solv=1e-10,
    )
    conductor = vq.run_cpcm_scf(
        mol, basis, method="rhf",
        solvent=vq.SolventModel(variant="cosmo", **common),
    )
    direct = vq.run_cpcm_scf(
        mol, basis, method="rhf",
        solvent=vq.SolventModel(
            variant="dcosmo-rs", sigma_potential=_null_sigma_potential(), **common
        ),
    )
    assert direct.converged
    assert direct.direct_feedback is not None
    assert direct.direct_feedback.energy == pytest.approx(0.0, abs=1e-14)
    assert np.abs(direct.direct_feedback.charges).max() < 1e-14
    assert float(direct.energy) == pytest.approx(float(conductor.energy), abs=1e-11)

    # A real solvent must move it, or the limit above proves only that zero
    # times anything is zero.
    real = vq.run_cpcm_scf(
        mol, basis, method="rhf",
        solvent=vq.SolventModel(
            variant="dcosmo-rs", sigma_potential=_toy_sigma_potential(), **common
        ),
    )
    assert real.converged
    assert abs(float(real.energy) - float(conductor.energy)) > 1e-4
    # And the shift must be the COSMO-RS term itself, not something else.
    assert float(real.energy) - float(conductor.energy) == pytest.approx(
        real.direct_feedback.energy, rel=0.35
    )


def test_direct_cosmors_refuses_the_things_it_cannot_do():
    """Four refusals, each for a measured reason rather than a taste."""
    vq = pytest.importorskip("vibeqc")

    pot = _toy_sigma_potential()
    ok = dict(epsilon=math.inf, cavity="fine", sigma_potential=pot)

    with pytest.raises(ValueError, match="needs a sigma_potential"):
        vq.SolventModel(epsilon=math.inf, cavity="fine", variant="dcosmo-rs")

    # A conductor, because the sigma potential *is* the solvent: scaling by
    # f(eps) as well would count it twice.
    with pytest.raises(ValueError, match="requires epsilon=inf"):
        vq.SolventModel(**{**ok, "variant": "dcosmo-rs", "epsilon": 78.39})

    # The fine cavity, because sigma = q/a on a switched Lebedev cavity is
    # ill-conditioned: its smallest segment is ~1e+07 below the median.
    with pytest.raises(ValueError, match="requires cavity='fine'"):
        vq.SolventModel(**{**ok, "variant": "dcosmo-rs", "cavity": "lebedev"})

    # And a potential that can actually be differentiated.
    bare = rs.SigmaPotential(
        rs.default_sigma_grid(), np.zeros(rs.default_sigma_grid().size),
        KLAMT_1998, T_ROOM_K, 0, 0.0,
    )
    with pytest.raises(ValueError, match="carry its solvent ensemble"):
        vq.SolventModel(**{**ok, "variant": "dcosmo-rs", "sigma_potential": bare})


def _smoothed_params(width=5e-4):
    import dataclasses

    return dataclasses.replace(KLAMT_1998, hb_smoothing=width)


def _net_torque(positions, gradient):
    """``sum_A R_A x F_A`` with ``F = -gradient``; zero for a free molecule."""
    return np.linalg.norm(np.sum(np.cross(positions, -gradient), axis=0))


def test_cpcm_gradient_refuses_an_unsmoothed_direct_cosmors_result():
    """With Klamt's published hydrogen-bond corner there is no gradient to return.

    ``cpcm_gradient`` once did not know the variant existed. Handed a Direct
    COSMO-RS result it returned the conductor-COSMO gradient -- no
    ``sum_t a_t mu_S(sigma_t)`` term, no ``q^dRS`` operator -- and nothing about
    that number looked wrong: its translation residual was 2.1e-15. Against a
    central difference of the energy the run reported it was 1.4e-03 Ha/bohr
    off on asymmetric water, one component with the wrong sign, where the
    conductor gradient at the same settings agrees to 1.0e-06.

    The analytic gradient now exists, but only for a smoothed hydrogen-bond
    term: with the published corner the energy is not smooth in the geometry
    (``test_the_hydrogen_bond_corner_is_real_and_survives_the_average``), so
    an unsmoothed result still fails closed, and the message says what to set.
    """
    vq = pytest.importorskip("vibeqc")
    from vibeqc.solvation.gradient import _direct_cosmors_feedback_for_gradient

    mol = vq.Molecule(
        [vq.Atom(8, (0.0, 0.0, 0.0)),
         vq.Atom(1, (0.0, 1.498, -1.159)),
         vq.Atom(1, (0.0, -1.498, -1.159))], 0, 1,
    )
    basis = vq.BasisSet(mol, "sto-3g")
    common = dict(
        epsilon=math.inf, cavity="fine", fine_grid_spacing_ang=0.40,
        max_macro_iter=60, tol_e_solv=1e-10,
    )
    direct = vq.run_cpcm_scf(
        mol, basis, method="rhf",
        solvent=vq.SolventModel(
            variant="dcosmo-rs", sigma_potential=_toy_sigma_potential(), **common
        ),
    )
    assert direct.converged
    assert direct.direct_feedback.hb_smoothing == 0.0

    with pytest.raises(
        NotImplementedError, match="no nuclear gradient for a Direct COSMO-RS"
    ) as refused:
        vq.cpcm_gradient(direct.scf, mol, basis, direct)
    message = str(refused.value)
    # It says why no gradient exists, not merely that none is implemented ...
    assert "sigma_hb" in message and "not differentiable" in message
    # ... and what makes one available, and what is available instead.
    assert "hb_smoothing=5e-4" in message
    assert "variant='cosmo'" in message and "cpcm_gradient_fd" in message

    # Keyed on the variant, not on what the variant requires: a conductor run on
    # the same eps=inf fine cavity passes the gate. Its full gradient costs
    # ~25 s and is pinned by the fine-cavity gradient tests, so only the gate
    # is exercised here.
    conductor = vq.run_cpcm_scf(
        mol, basis, method="rhf",
        solvent=vq.SolventModel(variant="cosmo", **common),
    )
    assert conductor.direct_feedback is None
    assert _direct_cosmors_feedback_for_gradient(conductor) is None


def test_direct_cosmors_gradient_reduces_to_the_conductor_gradient_for_a_null_solvent():
    """The exact limit of the gradient, needing no finite difference.

    A solvent with no interactions has ``mu_S == 0`` and ``mu_S' == 0``, so the
    correction charges, the cross term and the area term all vanish and the
    Direct COSMO-RS gradient must be the conductor COSMO gradient *exactly*.
    Smoothing the hydrogen-bond term of a null solvent leaves it null, so the
    gate passes and every extra term is exercised at zero: a stray factor, a
    unit slip or a wrong sign in the wiring would not survive it.
    """
    vq = pytest.importorskip("vibeqc")
    import dataclasses

    mol = _scf_water_untied(vq)
    basis = vq.BasisSet(mol, "sto-3g")
    common = dict(
        epsilon=math.inf, cavity="fine", fine_grid_spacing_ang=0.40,
        max_macro_iter=60, tol_e_solv=1e-10,
    )
    null = _toy_sigma_potential(dataclasses.replace(
        KLAMT_1998, alpha_prime=0.0, c_hb=0.0, f_corr=0.0, hb_smoothing=5e-4,
    ))
    conductor = vq.run_cpcm_scf(
        mol, basis, method="rhf", solvent=vq.SolventModel(variant="cosmo", **common),
    )
    direct = vq.run_cpcm_scf(
        mol, basis, method="rhf",
        solvent=vq.SolventModel(variant="dcosmo-rs", sigma_potential=null, **common),
    )
    assert direct.converged and direct.direct_feedback.hb_smoothing == 5e-4
    assert np.abs(direct.direct_feedback.charges).max() < 1e-14
    assert np.abs(direct.direct_feedback.mu).max() < 1e-14

    g_cond = vq.cpcm_gradient(conductor.scf, mol, basis, conductor, method="rhf")
    g_direct = vq.cpcm_gradient(direct.scf, mol, basis, direct, method="rhf")
    assert np.max(np.abs(g_direct - g_cond)) < 1e-10


def _scf_water_untied(vq):
    """Water with one OH lengthened by 0.1 bohr, so the molecule-fixed frame's
    step-1 selection is not tied (the fine-cavity gradient tests measure what
    the tie itself costs)."""
    return vq.Molecule(
        [vq.Atom(8, (0.0, 0.0, 0.0)),
         vq.Atom(1, (0.0, 1.598, -1.159)),
         vq.Atom(1, (0.0, -1.498, -1.159))], 0, 1,
    )


def test_direct_cosmors_gradient_matches_full_scf_fd():
    """The assembled Direct COSMO-RS gradient, against the outermost oracle.

    ``cpcm_gradient_fd`` re-converges the whole Direct COSMO-RS calculation at
    displaced geometries, so it assumes nothing about the cavity, the frame or
    the feedback. In a computed water potential with the hydrogen-bond term
    smoothed to 5e-4 e/A^2 the energy is smooth, and the analytic gradient,
    which the conductor gradient alone misses by 1.4e-03 Ha/bohr on this
    solute, agrees with the central difference at the fine-cavity gradient's
    own floor. The two exact invariants come free: a free molecule feels no
    net force and, in its molecule-fixed frame, no net torque.
    """
    vq = pytest.importorskip("vibeqc")
    from vibeqc.solvation.gradient import cpcm_gradient_fd

    pot = _computed_potential(vq, _WATER_ANG, params=_smoothed_params())
    mol = _scf_water_untied(vq)
    basis = vq.BasisSet(mol, "sto-3g")
    sm = vq.SolventModel(
        variant="dcosmo-rs", sigma_potential=pot,
        **_klamt_fine(fine_grid_spacing_ang=0.40, tol_e_solv=1e-12),
    )
    sol = vq.run_cpcm_scf(mol, basis, method="rhf", solvent=sm)
    assert sol.converged
    assert abs(sol.direct_feedback.energy) > 1e-4, "the solvent must do something"

    analytic = vq.cpcm_gradient(sol.scf, mol, basis, sol, method="rhf")
    positions = np.array([a.xyz for a in mol.atoms])
    assert np.abs(analytic.sum(axis=0)).max() < 1e-9
    assert _net_torque(positions, analytic) < 1e-7

    fd = cpcm_gradient_fd(mol, "sto-3g", method="rhf", solvent=sm, step_bohr=2e-3)
    worst = float(np.max(np.abs(analytic - fd)))
    assert worst < 5e-6, worst
    # The conductor gradient is not an acceptable stand-in.
    conductor_only = vq.cpcm_gradient(
        sol.scf, mol, basis, dataclasses_replace(sol, direct_feedback=None,
                                                 solvent_variant="cosmo"),
        method="rhf",
    )
    assert np.max(np.abs(conductor_only - fd)) > 50 * worst


def dataclasses_replace(obj, **changes):
    import dataclasses

    return dataclasses.replace(obj, **changes)


_HOCL_ANG = [(8, (0.0, 0.0, 0.0)), (17, (1.75, 0.0, 0.0)), (1, (-0.1736, 0.9848, 0.0))]


def test_run_job_optimizes_on_the_direct_cosmors_surface(tmp_path):
    """A geometry optimization in a real solvent walks the Direct COSMO-RS surface.

    At the reported geometry the gradient *of that surface* is below the
    threshold the run converged to, which needs no reference geometry.

    The solute is hypochlorous acid, whose minimum has no symmetry: water's
    is symmetric, which ties the molecule-fixed frame's step-1 selection, and
    there the analytic gradient carries a spurious force that Direct COSMO-RS
    amplifies past the optimizer's threshold
    (``test_the_frame_tie_cusp_is_amplified_by_direct_cosmors_and_projected_out``,
    #535):
    measured, 200 BFGS steps at a converged energy and geometry with the
    reported maximum force stuck at 0.072 eV/A.
    """
    vq = pytest.importorskip("vibeqc")
    fmax_default_ha_bohr = 0.05 * 0.0194469064593167  # run_job's fmax, in Ha/bohr

    pot = _computed_potential(vq, _WATER_ANG, params=_smoothed_params())
    sm = vq.SolventModel(
        variant="dcosmo-rs", sigma_potential=pot, **_klamt_fine(fine_grid_spacing_ang=0.40),
    )
    stem = tmp_path / "hocl_drs_opt"
    res = vq.run_job(
        _mol_ang(vq, _HOCL_ANG), basis="sto-3g", method="rhf", optimize=True,
        solvent=sm, output=str(stem), progress=False, write_molden_file=False,
    )
    assert res.solvent_result.solvent_variant == "dcosmo-rs"
    lines = stem.with_suffix(".xyz").read_text().splitlines()
    xyz = np.array([[float(x) for x in ln.split()[1:4]] for ln in lines[2:5]])
    mol = vq.Molecule(
        [vq.Atom(z, tuple(r * _ANG_TO_BOHR)) for z, r in zip((8, 17, 1), xyz)], 0, 1
    )
    basis = vq.BasisSet(mol, "sto-3g")
    sol = vq.run_cpcm_scf(mol, basis, method="rhf", solvent=sm)
    gmax = float(np.abs(vq.cpcm_gradient(sol.scf, mol, basis, sol, method="rhf")).max())
    assert gmax < fmax_default_ha_bohr, gmax
    # It moved: the start is not the minimum (RHF/STO-3G puts O-Cl at 1.743 A
    # against the 1.75 A start, and the angle relaxes as well).
    start = np.array([xyz0 for _, xyz0 in _HOCL_ANG])
    start -= start[0]
    assert np.abs(np.linalg.norm(xyz[1:] - xyz[0], axis=1)
                  - np.linalg.norm(start[1:], axis=1)).max() > 3e-3


def test_the_frame_tie_cusp_is_amplified_by_direct_cosmors_and_projected_out():
    """The same frame tie as the conductor's, 70 times larger, and removed (#535).

    ``test_the_frame_tie_is_projected_out_of_the_gradient`` pins the mechanism:
    at a symmetric geometry the molecule-fixed frame picks one of several
    symmetry-equivalent atoms, the energy has a cusp there, and the raw gradient
    is one branch's slope. For the conductor energy that is 1.9e-05 Ha/bohr,
    below any optimizer threshold. For Direct COSMO-RS it is 1.4e-03 on
    symmetric water (0.07 eV/A against run_job's 0.05 eV/A target): the
    COSMO-RS term is far more sensitive to how the lattice cuts the surface.
    A central finite difference gives the average of the two branches; the
    gradient projected onto the point group's symmetric subspace (the model's
    default) gives the same.

    The lab frame is not a remedy, and the test records why: its Direct
    COSMO-RS energy has steps (1e-3 Ha at this geometry), so the finite
    difference there grows as the step shrinks (0.34, 1.0, 1.3 Ha/bohr at
    h = 2e-3, 1e-3, 5e-4) and the user guide's table keeps the numbers.
    """
    vq = pytest.importorskip("vibeqc")
    from vibeqc.solvation.gradient import cpcm_gradient_fd

    pot = _computed_potential(vq, _WATER_ANG, params=_smoothed_params())
    # The symmetric minimum a Direct COSMO-RS walk of water reaches at
    # RHF/STO-3G in a computed water potential: O-H 0.9857 A, H-H 1.5131 A.
    symmetric = _mol_ang(vq, [
        (8, (0.0, 0.0, 0.0)), (1, (0.0, 0.75655, 0.63172)), (1, (0.0, -0.75655, 0.63172)),
    ])
    basis = vq.BasisSet(symmetric, "sto-3g")

    def run(tol):
        sm = vq.SolventModel(
            variant="dcosmo-rs", sigma_potential=pot,
            gradient_symmetry_tol_bohr=tol, **_klamt_fine(fine_grid_spacing_ang=0.40),
        )
        sol = vq.run_cpcm_scf(symmetric, basis, method="rhf", solvent=sm)
        return vq.cpcm_gradient(sol.scf, symmetric, basis, sol, method="rhf"), sm

    raw, _ = run(0.0)
    # The oxygen's force along y must vanish by the mirror symmetry; the tie
    # puts more than 1e-3 Ha/bohr there.
    assert abs(raw[0, 1]) > 1e-3, raw

    projected, sm = run(1e-4)
    assert abs(projected[0, 1]) < 1e-12, projected
    assert np.abs(projected.sum(axis=0)).max() < 1e-12
    fd = cpcm_gradient_fd(symmetric, "sto-3g", method="rhf", solvent=sm, step_bohr=1e-3)
    assert abs(fd[0, 1]) < 1e-4, fd        # the finite difference has no offset
    worst = float(np.max(np.abs(projected - fd)))
    assert worst < DRS_TIED_PROJECTED_BOUND, worst
    assert float(np.max(np.abs(raw - fd))) > 50.0 * worst


#: Measured projected-versus-finite-difference disagreement at the symmetric
#: geometry above: 1.0e-05 (the raw gradient's is 1.4e-03); the bound keeps about
#: five times that.
DRS_TIED_PROJECTED_BOUND = 5.0e-5


def test_run_job_converges_a_symmetric_solute_on_the_direct_cosmors_surface(tmp_path):
    """The walk that stalled now converges (#535).

    Water's minimum is symmetric, which ties the molecule-fixed frame. Before the
    gradient was projected onto the point group, a Direct COSMO-RS walk of
    water reached its minimum at BFGS step 6 and then spent 194 more steps with
    displacements of 6e-10 A and the reported maximum force fixed at
    0.072 eV/A, against a 0.05 eV/A target, and raised
    ``_ASEGeometryOptimizationError``. The energy was constant to 1e-12 eV from
    step 59 on; only the cusp's force kept it from stopping.
    """
    vq = pytest.importorskip("vibeqc")

    pot = _computed_potential(vq, _WATER_ANG, params=_smoothed_params())
    sm = vq.SolventModel(
        variant="dcosmo-rs", sigma_potential=pot, **_klamt_fine(fine_grid_spacing_ang=0.40),
    )
    stem = tmp_path / "water_drs_opt"
    res = vq.run_job(
        _scf_water_untied(vq), basis="sto-3g", method="rhf", optimize=True,
        solvent=sm, output=str(stem), progress=False, write_molden_file=False,
    )
    assert res.solvent_result.solvent_variant == "dcosmo-rs"
    lines = stem.with_suffix(".xyz").read_text().splitlines()
    xyz = np.array([[float(x) for x in ln.split()[1:4]] for ln in lines[2:5]])
    oh = [float(np.linalg.norm(xyz[k] - xyz[0])) for k in (1, 2)]
    # It reached the symmetric minimum the stalled walk sat at.
    assert abs(oh[0] - oh[1]) < 2e-3 and abs(sum(oh) / 2 - 0.9857) < 3e-3, oh
    assert abs(float(res.energy) - (-74.971945)) < 2e-5, float(res.energy)


def test_direct_cosmors_route_cites_the_whole_calculation():
    """One solvation row fires per variant, so the Direct COSMO-RS row is all of it.

    The registry picks ``routes.solvation[variant]`` and nothing else, so this
    row cannot lean on the conductor rows: it carries Sinnecker, Rajendran,
    Klamt, Diedenhofen & Neese 2006 (doi:10.1021/jp056016z), the COSMO-RS
    sigma potential and its parameterization, conductor COSMO, and the COSMO
    FINE Cavity the variant requires with the Gaussian-charge diagonal that
    cavity uses, and Anderson 1965 for the step that accelerates its
    macro-iteration. Not Scalmani-Frisch, whose point-charge diagonal belongs to
    the Lebedev cavity that Direct COSMO-RS refuses.
    """
    from vibeqc.output.citations import load_default_database

    db = load_default_database()
    direct = db.assemble(uses_cpcm=True, solvent_variant="dcosmo-rs")
    keys = {c.key for c in direct.citations}
    assert not any("solvent variant" in w for w in direct.warnings), direct.warnings
    assert {
        "sinnecker_direct_cosmors_2006",
        "klamt_cosmors_1995",
        "klamt_cosmors_parametrization_1998",
        "klamt_schuurmann_cosmo_1993",
        "klamt_fine_cavity_2018",
        "lange_herbert_swig_2010",
        "anderson_mixing_1965",
    } <= keys
    assert "scalmani_frisch_csc_2010" not in keys
    (paper,) = [c for c in direct.citations if c.key == "sinnecker_direct_cosmors_2006"]
    assert paper.doi == "10.1021/jp056016z"

    # The conductor variants must not claim a method they did not run.
    for variant in ("cpcm", "cosmo"):
        conductor = db.assemble(uses_cpcm=True, solvent_variant=variant)
        assert "sinnecker_direct_cosmors_2006" not in {c.key for c in conductor.citations}


def test_a_direct_cosmors_run_cites_its_own_variant(tmp_path):
    """``run_cpcm_scf(output=...)`` used to cite ``"cpcm"`` whatever it ran.

    The route above is dead unless the driver passes the variant through, and
    the emitter is wrapped in a best-effort ``except``, so a regression here is
    silent: the only witness is the written bibliography.
    """
    vq = pytest.importorskip("vibeqc")

    mol = vq.Molecule(
        [vq.Atom(8, (0.0, 0.0, 0.0)),
         vq.Atom(1, (0.0, 1.498, -1.159)),
         vq.Atom(1, (0.0, -1.498, -1.159))], 0, 1,
    )
    sol = vq.run_cpcm_scf(
        mol, vq.BasisSet(mol, "sto-3g"), method="rhf",
        solvent=vq.SolventModel(
            variant="dcosmo-rs", sigma_potential=_toy_sigma_potential(),
            epsilon=math.inf, cavity="fine", fine_grid_spacing_ang=0.40,
            max_macro_iter=60, tol_e_solv=1e-10,
        ),
        output=tmp_path / "dcosmors",
    )
    assert sol.converged
    written = {
        p.name: p.read_text(errors="replace")
        for p in tmp_path.iterdir() if p.is_file()
    }
    assert written, "run_cpcm_scf(output=...) wrote no citation files"
    cites = [name for name, text in written.items() if "sinnecker_direct_cosmors_2006" in text]
    assert cites, sorted(written)


def test_direct_cosmors_reports_where_its_feedback_is_fragile():
    """The two diagnostics, and why a converged run still needs them.

    Klamt's hydrogen-bond term makes ``mu_S'`` discontinuous at
    ``sigma = +-sigma_hb``, and a quarter of a real solute's segments sit within
    a bin of it -- so the first question when the macro-iteration will not
    settle is how many segments are parked on that corner. The second is how
    much *area* is being extrapolated beyond the sigma potential's fitted range,
    which on the fine cavity is a fraction of a percent and on a switched
    Lebedev cavity would be most of it.
    """
    vq = pytest.importorskip("vibeqc")

    mol = vq.Molecule(
        [vq.Atom(8, (0.0, 0.0, 0.0)),
         vq.Atom(1, (0.0, 1.498, -1.159)),
         vq.Atom(1, (0.0, -1.498, -1.159))], 0, 1,
    )
    sol = vq.run_cpcm_scf(
        mol, vq.BasisSet(mol, "sto-3g"), method="rhf",
        solvent=vq.SolventModel(
            variant="dcosmo-rs", sigma_potential=_toy_sigma_potential(),
            epsilon=math.inf, cavity="fine", fine_grid_spacing_ang=0.40,
            max_macro_iter=60, tol_e_solv=1e-10,
        ),
    )
    fb = sol.direct_feedback
    assert 0.0 <= fb.fraction_near_hb_corner <= 1.0
    assert 0.0 <= fb.fraction_outside_grid < 0.05, fb.fraction_outside_grid
    # The fine cavity is what keeps sigma inside the parameterization; if this
    # ever fails, the cavity changed and the feedback is being extrapolated.
    assert np.percentile(np.abs(fb.sigma), 99) < 0.05


# ---------------------------------------------------------------------
# Direct COSMO-RS convergence beyond water and methanol (next step 4).
#
# Water and methanol were the only solutes ever run through the feedback loop.
# Measured on RHF/STO-3G in a computed water potential (fine cavity 0.35 A,
# KLAMT_1998 radii): methanol 11, acetic acid 11, acetate 9, ammonium 5,
# benzene 7 macro-iterations; water in a computed methanol potential 10. All
# monotone in the tail, none damped, none oscillating -- including water in
# methanol with 15 percent of its surface on the H-bond corner.
# ---------------------------------------------------------------------

_ANG_TO_BOHR = 1.8897261254578281
_WATER_ANG = [(8, (0.0, 0.0, 0.0)), (1, (0.0, 0.757, 0.587)), (1, (0.0, -0.757, 0.587))]
_METHANOL_ANG = [
    (6, (-0.047, 0.664, 0.0)), (8, (-0.047, -0.758, 0.0)), (1, (-1.086, 0.975, 0.0)),
    (1, (0.444, 1.086, 0.890)), (1, (0.444, 1.086, -0.890)), (1, (0.861, -1.050, 0.0)),
]
_ACETATE_ANG = [
    (6, (-0.943, -0.081, 0.0)), (6, (0.536, 0.162, 0.0)), (8, (1.212, -0.880, 0.0)),
    (8, (0.980, 1.330, 0.0)), (1, (-1.448, 0.881, 0.0)),
    (1, (-1.252, -0.652, 0.878)), (1, (-1.252, -0.652, -0.878)),
]
_AMMONIUM_ANG = [
    (7, (0.0, 0.0, 0.0)), (1, (0.595, 0.595, 0.595)), (1, (-0.595, -0.595, 0.595)),
    (1, (-0.595, 0.595, -0.595)), (1, (0.595, -0.595, -0.595)),
]


def _mol_ang(vq, atoms, charge=0):
    return vq.Molecule(
        [vq.Atom(z, [c * _ANG_TO_BOHR for c in xyz]) for z, xyz in atoms], charge, 1
    )


def _klamt_fine(**overrides):
    kw = dict(
        epsilon=math.inf, cavity="fine", fine_grid_spacing_ang=0.35,
        radii=dict(KLAMT_1998.cavity_radii), radii_scale=1.0,
        solvent_probe_radius_ang=0.0, max_macro_iter=80, tol_e_solv=1e-10,
    )
    kw.update(overrides)
    return kw


def _computed_potential(vq, solvent_atoms, params=KLAMT_1998):
    mol = _mol_ang(vq, solvent_atoms)
    conductor = vq.run_cpcm_scf(
        mol, vq.BasisSet(mol, "sto-3g"), method="rhf",
        solvent=vq.SolventModel(variant="cosmo", **_klamt_fine()),
    )
    return rs.solvent_sigma_potential(conductor, params, method="rhf", basis="sto-3g")


def _direct(vq, mol, pot, **overrides):
    return vq.run_cpcm_scf(
        mol, vq.BasisSet(mol, "sto-3g"), method="rhf",
        solvent=vq.SolventModel(
            variant="dcosmo-rs", sigma_potential=pot, **_klamt_fine(**overrides)
        ),
    )


def _monotone_tail(sol, n=4):
    tail = [h["delta_e_solv"] for h in sol.macro_history][-n:]
    return len({float(np.sign(d)) for d in tail}) == 1, tail


@pytest.mark.parametrize(
    "atoms, charge", [(_ACETATE_ANG, -1), (_AMMONIUM_ANG, +1)], ids=["acetate", "ammonium"]
)
def test_direct_cosmors_converges_for_both_ion_charges(atoms, charge):
    """Ions converge, and say that their surface is past the fitted range.

    An ion drives sigma far beyond what a sigma potential is fitted over, so
    this is where a feedback loop built on a piecewise-linear H-bond term was
    most likely to break. It does not: 9 and 5 macro-iterations, monotone.

    The second assertion pins a caveat rather than a success. Measured, 1.6
    and 8.2 percent of the ions' area sits beyond the grid (sigma to 0.38
    e/A^2), where the potential is extrapolated through the quadratic misfit
    term; with a solvent potential computed at RHF/STO-3G that extrapolation
    gives acetate +11.7 and ammonium -11.0 kcal/mol of COSMO-RS energy. The
    diagnostic has to report it, so a user can see the number is not in range.
    """
    vq = pytest.importorskip("vibeqc")
    pot = _computed_potential(vq, _WATER_ANG)
    sol = _direct(vq, _mol_ang(vq, atoms, charge), pot)

    assert sol.converged
    assert sol.n_macro_iter <= 20, sol.n_macro_iter
    monotone, tail = _monotone_tail(sol)
    assert monotone, tail
    assert sol.direct_feedback.fraction_outside_grid > 0.005


def _water_sigma_percentiles(vq, method, basis, options=None):
    """Area-weighted 1st and 99th percentiles of water's averaged sigma, e/A^2."""
    from vibeqc.solvation.surface import build_conductor_surface

    mol = _mol_ang(vq, _WATER_ANG)
    res = vq.run_cpcm_scf(
        mol, vq.BasisSet(mol, basis), method=method, options=options,
        solvent=vq.SolventModel(variant="cosmo", **_klamt_fine()),
    )
    surf = build_conductor_surface(
        res, method=method, basis=basis, cavity_kind="fine-cfc",
        energy_conductor=float(res.energy),
    )
    desc = rs.segment_descriptors(surf, KLAMT_1998.r_av)
    s, a = np.asarray(desc.sigma), np.asarray(desc.areas)
    order = np.argsort(s)
    cw = np.cumsum(a[order]) / a.sum()
    return float(s[order][np.searchsorted(cw, 0.01)]), float(s[order][np.searchsorted(cw, 0.99)])


def test_the_recommended_solvent_protocol_widens_the_water_profile():
    """BPW91/def2-TZVP water polarizes toward the fitted range; RHF/STO-3G does not.

    ``KLAMT_1998`` was fitted on DMol/BPW91/DNP surfaces, and the user guide
    recommends BPW91/def2-TZVP as the nearest recipe vibe-qc can run. Measured,
    it widens water's profile (1st to 99th percentile) from -0.012..+0.014 at
    RHF/STO-3G to -0.017..+0.018 e/A^2, on both the donor and the acceptor
    side. That width is what gives the ions of the step 4 panel the sign a
    hydrogen-bonding solvent should give them (acetate +11.7 -> -15.0,
    ammonium -11.0 -> -19.5 kcal/mol); the Direct COSMO-RS runs themselves are
    too slow for this lane and are recorded in the guide.
    """
    vq = pytest.importorskip("vibeqc")
    lo_min, hi_min = _water_sigma_percentiles(vq, "rhf", "sto-3g")
    opts = vq.RKSOptions()
    opts.functional = "bpw91"
    lo, hi = _water_sigma_percentiles(vq, "rks", "def2-tzvp", opts)

    assert -0.013 < lo_min and hi_min < 0.0145, (lo_min, hi_min)
    assert lo < -0.0155 and hi > 0.0165, (lo, hi)
    assert -0.025 < lo and hi < 0.025, "the profile must stay on the fitted grid"


def test_direct_cosmors_converges_in_a_second_computed_solvent(monkeypatch):
    """A second solvent, and the corner parking that did not break it.

    Water in a methanol potential computed the same way converges in 10
    macro-iterations of the plain loop, monotonically, with 14.8 percent of its
    surface within a bin of +-sigma_hb, the most of any case measured. The
    corner is a discontinuity in mu_S', not a trap for the loop. Monotone is a
    property of the plain fixed-point map, so it is checked there (the Anderson
    step patched to the plain update); the shipped, accelerated loop takes 9
    and may alternate in sign once the change is below 1e-7, which is how an
    extrapolation converges rather than an oscillation.
    """
    vq = pytest.importorskip("vibeqc")
    from vibeqc.solvation.cosmors import potential as rs_potential

    pot = _computed_potential(vq, _METHANOL_ANG)
    mol = _mol_ang(vq, _WATER_ANG)
    sol = _direct(vq, mol, pot)
    monkeypatch.setattr(
        rs_potential, "_anderson_step", lambda xs, rs_, mixing: xs[-1] + mixing * rs_[-1]
    )
    plain = _direct(vq, mol, pot)

    assert sol.converged and plain.converged
    monotone, tail = _monotone_tail(plain)
    assert monotone, tail
    assert sol.n_macro_iter <= plain.n_macro_iter
    assert float(sol.energy) == pytest.approx(float(plain.energy), abs=1e-9)
    assert sol.direct_feedback.fraction_near_hb_corner > 0.05


def test_direct_cosmors_total_is_converged_to_the_tolerance_it_claims():
    """The loop watches the conductor term only; the total still converges.

    The macro-iteration stops on ``|delta e_solv| < tol_e_solv``, where
    ``e_solv`` is ``1/2 q.V``. The COSMO-RS energy is part of the reported
    total but not of the criterion, so it could in principle still be moving.
    Measured, it is not: at tolerances 1e-4 to 1e-8 the reported total of both
    ions sits within 6e-6 *of the tolerance* of a tightly converged run. Pinned
    on ammonium, whose last COSMO-RS step was the largest relative to the
    conductor step.
    """
    vq = pytest.importorskip("vibeqc")
    pot = _computed_potential(vq, _WATER_ANG)
    mol = _mol_ang(vq, _AMMONIUM_ANG, +1)
    tight = _direct(vq, mol, pot, tol_e_solv=1e-12)
    loose = _direct(vq, mol, pot, tol_e_solv=1e-6)

    assert tight.converged and loose.converged
    assert abs(float(loose.energy) - float(tight.energy)) < 1e-6


def test_direct_cosmors_acceleration_damps_an_oscillation_and_keeps_the_answer(monkeypatch):
    """Anderson on the feedback potential shortens an oscillation, not the answer.

    The macro-loop extrapolates ``phi`` over its last four iterates (Anderson,
    J. ACM 12, 547 (1965)). Replacing that step with the plain update recovers
    the unaccelerated loop. Acetate at RHF/3-21G in the RHF/STO-3G water
    potential is the cheapest case measured that oscillates: the plain loop
    flips the sign of ``delta e_solv`` on every one of its last ten
    macro-iterations and needs 15, the accelerated one 13 with half the flips,
    and both land on the same energy (to 1e-13 measured). At BPW91 the same
    solute goes from 26 to 18 at def2-SVP and from 60 to 15 at def2-TZVP. The
    feedback the run reports is still
    eq. 19 of Sinnecker et al. 2006 at the final conductor charges, not the
    extrapolated potential the operator last used.
    """
    vq = pytest.importorskip("vibeqc")
    from vibeqc.solvation.cosmors import potential as rs_potential
    from vibeqc.solvation.cosmors.direct import direct_cosmors_feedback

    pot = _computed_potential(vq, _WATER_ANG)
    mol = _mol_ang(vq, _ACETATE_ANG, -1)
    basis = vq.BasisSet(mol, "3-21g")

    def run():
        sm = vq.SolventModel(variant="dcosmo-rs", sigma_potential=pot, **_klamt_fine())
        return vq.run_cpcm_scf(mol, basis, method="rhf", solvent=sm)

    def flips(sol):
        d = np.sign([h["delta_e_solv"] for h in sol.macro_history][-11:])
        return int(np.count_nonzero(d[1:] != d[:-1]))

    accelerated = run()
    monkeypatch.setattr(
        rs_potential, "_anderson_step", lambda xs, rs_, mixing: xs[-1] + mixing * rs_[-1]
    )
    plain = run()

    assert accelerated.converged and plain.converged
    assert float(accelerated.energy) == pytest.approx(float(plain.energy), abs=1e-9)
    assert flips(plain) >= 8, "the unaccelerated loop should oscillate here"
    assert accelerated.n_macro_iter < plain.n_macro_iter
    assert flips(accelerated) < flips(plain)
    fb = accelerated.direct_feedback
    again = direct_cosmors_feedback(accelerated.cpcm.q, accelerated.cavity.weights, pot)
    np.testing.assert_allclose(fb.phi, again.phi, rtol=0, atol=1e-12)


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


# ---------------------------------------------------------------------
# The smoothed hydrogen-bond term (next step 5, decided 2026-09-29).
#
# Klamt 1998 eq. 22 has corners at +-sigma_hb, so mu_S' jumps there and a
# Direct COSMO-RS energy is not differentiable in the geometry. A positive
# ``Parameterization.hb_smoothing`` replaces the corners with softplus ramps.
# The default is 0.0, the published form exactly.
# ---------------------------------------------------------------------

_HB_GRID = np.linspace(-0.03, 0.03, 241)


def _published_hb(a, b, params):
    """Klamt 1998 eq. 22 written out as published, the fixed reference."""
    acc, don = np.maximum(a, b), np.minimum(a, b)
    return params.c_hb * np.maximum(0.0, acc - params.sigma_hb) * np.minimum(
        0.0, don + params.sigma_hb
    )


def test_the_default_parameterization_is_unsmoothed():
    assert KLAMT_1998.hb_smoothing == 0.0
    a, b = np.meshgrid(_HB_GRID, _HB_GRID)
    np.testing.assert_array_equal(
        rs.hydrogen_bond_energy(a, b, KLAMT_1998), _published_hb(a, b, KLAMT_1998)
    )


def test_the_symmetric_rewrite_is_the_published_form_exactly():
    """``c_hb [f(a) g(b) + f(b) g(a)]`` *is* eq. 22, before any smoothing.

    Everything below rests on this identity: it removes the acceptor/donor
    choice, which is what lets smoothed ramps give a function with no new
    corner at ``a == b``.
    """
    from vibeqc.solvation.cosmors.potential import _hb_ramps

    a, b = np.meshgrid(_HB_GRID, _HB_GRID)
    fa, _, ga, _ = _hb_ramps(a, KLAMT_1998)
    fb, _, gb, _ = _hb_ramps(b, KLAMT_1998)
    np.testing.assert_allclose(
        KLAMT_1998.c_hb * (fa * gb + fb * ga), _published_hb(a, b, KLAMT_1998),
        rtol=0, atol=1e-15,
    )


def test_the_smoothed_term_tends_to_the_published_form_linearly():
    import dataclasses

    a, b = np.meshgrid(_HB_GRID, _HB_GRID)
    ref = _published_hb(a, b, KLAMT_1998)
    err = {}
    for w in (1e-3, 1e-4, 1e-5):
        p = dataclasses.replace(KLAMT_1998, hb_smoothing=w)
        err[w] = float(np.max(np.abs(rs.hydrogen_bond_energy(a, b, p) - ref)))
        # Each ramp moves by at most w ln 2 and its partner is bounded by
        # 0.03 + sigma_hb on this grid, so the product moves by less than
        # c_hb * w ln 2 * 2 * 0.04 (plus a w^2 term that is negligible here).
        assert err[w] <= KLAMT_1998.c_hb * w * np.log(2) * 0.08 * 1.01, (w, err[w])
    # Linear in the width, so the published form is the w -> 0 limit. Measured
    # where the error peaks -- exactly on the corner -- not as a maximum over a
    # grid, which under-samples it once w drops below the grid spacing. There
    # f_w(sigma_hb) = w ln 2 against f = 0, and against a strong donor g_w is g
    # to exponential accuracy, so the error is c_hb w ln 2 |g(partner)| exactly.
    s_hb = KLAMT_1998.sigma_hb
    donor = -0.02
    g_donor = donor + s_hb
    for w in (1e-3, 1e-4, 1e-5):
        p = dataclasses.replace(KLAMT_1998, hb_smoothing=w)
        corner = float(rs.hydrogen_bond_energy(np.array([s_hb]), np.array([donor]), p)[0])
        assert corner == pytest.approx(KLAMT_1998.c_hb * w * np.log(2) * g_donor, rel=1e-6), w


def test_the_smoothed_derivative_is_exact_and_continuous_through_the_corner():
    """Where the published derivative jumps, the smoothed one is continuous.

    Measured against central differences at points on and around
    ``+-sigma_hb``, then across the corner itself: the published form steps by
    ``c_hb`` times the partner's factor there, the smoothed one by nothing as
    the probe closes in.
    """
    import dataclasses

    w = 1e-3
    p = dataclasses.replace(KLAMT_1998, hb_smoothing=w)
    s_hb = KLAMT_1998.sigma_hb
    partner = np.array([-0.02])                     # a strong donor
    xs = np.array([-0.02, -s_hb - w, -s_hb, s_hb - w, s_hb, s_hb + w, 0.02])
    h = 1e-8
    def d_int(x, params):
        return float(rs.interaction_energy_dsigma(np.array([x]), partner, params)[0])

    def e_int(x, params):
        return float(rs.interaction_energy(np.array([x]), partner, params)[0])

    for x in xs:
        fd = (e_int(x + h, p) - e_int(x - h, p)) / (2 * h)
        assert d_int(x, p) == pytest.approx(fd, rel=1e-6, abs=1e-6), x

    # Continuity: across the corner the difference shrinks in proportion to the
    # probe width, because the second derivative there is finite (about
    # c_hb |g| / (4 w), 2.2e4 per unit sigma here). A jump would not shrink.
    step = {eps: abs(d_int(s_hb + eps, p) - d_int(s_hb - eps, p)) for eps in (1e-6, 1e-8)}
    assert step[1e-8] == pytest.approx(step[1e-6] / 100, rel=0.05), step
    hard_jump = abs(d_int(s_hb + 1e-8, KLAMT_1998) - d_int(s_hb - 1e-8, KLAMT_1998))
    assert step[1e-8] < 1e-3 * hard_jump, (step, hard_jump)
    assert hard_jump > 10.0, hard_jump                # the corner the smoothing removes


def test_the_smoothed_term_is_symmetric_and_leaves_a_null_solvent_null():
    import dataclasses

    a, b = np.meshgrid(_HB_GRID, _HB_GRID)
    p = dataclasses.replace(KLAMT_1998, hb_smoothing=1e-3)
    e = rs.hydrogen_bond_energy(a, b, p)
    np.testing.assert_array_equal(e, e.T)
    # c_hb = 0 keeps the null-solvent limit of Direct COSMO-RS exact at any width.
    null = dataclasses.replace(p, c_hb=0.0)
    assert np.all(rs.hydrogen_bond_energy(a, b, null) == 0.0)


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


def _polar_ensemble(n=300, centre=0.017, width=0.004, seed=12345):
    """A water-like solvent surface: two broad lobes at +-``centre`` e/A^2.

    Seeded, so the ensemble -- and every number below -- is the same on every
    run. Its lobes sit where a solvent computed near the parameterization's own
    protocol puts them (BP86/def2-TZVP water spans about +-0.017), which is
    exactly the regime the plain fixed point could not reach.
    """
    rng = np.random.default_rng(seed)
    sigma = np.concatenate(
        [rng.normal(-centre, width, n // 2), rng.normal(centre, width, n // 2)]
    )
    return rs.SegmentDescriptors(
        areas=np.full(n, 0.35), sigma=sigma,
        sigma_perp=np.zeros(n), sigma_raw=np.zeros(n),
    )


def test_sigma_potential_converges_on_a_realistic_polar_ensemble():
    """The solvent solver reaches a realistic surface within its own cap.

    The plain damped iteration contracts linearly, and on polar surfaces too
    slowly: this ensemble needed 6029 steps to 1e-10 against a cap of 500, and
    a real BP86/def2-TZVP water surface 504. Both were reported as failures
    though the iteration was still converging. Anderson acceleration reaches
    the same fixed point in a few dozen steps.

    The oracle is the fixed-point equation itself, not a reference number:
    evaluated at the ensemble's own segments, the potential must reproduce the
    segment potentials it was converged to.
    """
    desc = _polar_ensemble()
    pot = rs.sigma_potential_segments([desc], np.array([1.0]), KLAMT_1998)

    assert pot.n_iter < 100, pot.n_iter
    assert pot.residual < 1e-10
    _, sigma, mu_seg, sigma_perp = pot._ensemble
    np.testing.assert_allclose(pot.evaluate(sigma, sigma_perp), mu_seg, rtol=0, atol=1e-9)


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



def test_gas_reference_terms_carry_klamts_signs():
    """Each term of ``mu_gas`` moves the way the physics does (#435).

    The gas phase sits above the ideally screened state by the screening gain
    ``E_gas - E_COSMO`` (Klamt 2000, doi:10.1016/S0378-3812(00)00357-5,
    eq. 19; Klamt 1995 quotes water's as -9.3 kcal/mol in the opposite
    convention), by the dispersion it loses (``-sum gamma_k A_k``,
    ``gamma_k < 0``) and by the ring term (``-omega n``, ``omega < 0``); its
    standard-state entropy lowers it by ``eta R T``, -5.4 kcal/mol (Klamt 1998,
    doi:10.1021/jp980017s, section 5.1). Eq. 21 as printed has the first and
    the last sign the other way, and read literally it put every nonpolar
    liquid about 10 kcal/mol too volatile.
    """
    from vibeqc.solvation.cosmors.thermo import HARTREE_TO_KCAL

    surf = _toy_surface()

    def with_energies(e_gas, e_cond):
        return ConductorSurface(
            positions=surf.positions, areas=surf.areas, normals=surf.normals,
            segment_atom=surf.segment_atom, charges=surf.charges,
            atomic_numbers=surf.atomic_numbers, atom_positions=surf.atom_positions,
            energy_gas=e_gas, energy_conductor=e_cond, provenance=surf.provenance,
        )

    unscreened = with_energies(-76.0, -76.0)
    screened = with_energies(-76.0, -76.01)
    desc = rs.segment_descriptors(unscreened, KLAMT_1998.r_av)
    mu0 = rs.gas_phase_chemical_potential(unscreened, desc, KLAMT_1998)

    gain = rs.gas_phase_chemical_potential(screened, desc, KLAMT_1998) - mu0
    assert gain == pytest.approx(0.01 * HARTREE_TO_KCAL, rel=1e-12)
    override = rs.gas_phase_chemical_potential(
        unscreened, desc, KLAMT_1998, ideal_screening_energy_hartree=0.01
    )
    assert override - mu0 == pytest.approx(gain, rel=1e-12)
    ring = rs.gas_phase_chemical_potential(unscreened, desc, KLAMT_1998, n_ring_atoms=6)
    assert ring - mu0 == pytest.approx(6 * 0.21, rel=1e-12)

    areas = rs.per_element_area(unscreened, desc)
    dispersion = sum(KLAMT_1998.dispersion[z] * a for z, a in areas.items())
    eta_rt = KLAMT_1998.eta_entropy * R_KCAL * T_ROOM_K
    assert dispersion < 0.0
    assert eta_rt == pytest.approx(-5.4, abs=0.03)
    assert mu0 == pytest.approx(-dispersion + eta_rt, rel=1e-12)


# Experimental log10(p_vap / bar) at 298 K from Klamt 1995
# (doi:10.1021/j100007a062) Table 2, the ring-atom count, and a G2-set geometry
# (angstrom, as distributed with ASE).
_VAPOUR_PRESSURE_PANEL = {
    "water": (-1.7, 0, [
        (8, (0.0, 0.0, 0.1193)), (1, (0.0, 0.7632, -0.477)), (1, (0.0, -0.7632, -0.477)),
    ]),
    "methanol": (-0.8, 0, [
        (6, (-0.0471, 0.6644, 0.0)), (8, (-0.0471, -0.7586, 0.0)),
        (1, (-1.093, 0.9698, 0.0)), (1, (0.8785, -1.0485, 0.0)),
        (1, (0.4371, 1.0804, 0.8918)), (1, (0.4371, 1.0804, -0.8918)),
    ]),
    "ethanol": (-1.3, 0, [
        (6, (1.1682, -0.4004, 0.0)), (6, (0.0, 0.5595, 0.0)),
        (8, (-1.1901, -0.2277, 0.0)), (1, (-1.9466, 0.3815, 0.0)),
        (1, (0.0426, 1.2075, 0.8869)), (1, (0.0426, 1.2075, -0.8869)),
        (1, (2.1159, 0.1448, 0.0)), (1, (1.1286, -1.0372, 0.8859)),
        (1, (1.1286, -1.0372, -0.8859)),
    ]),
    "propanone": (-0.6, 0, [
        (8, (0.0, 0.0, 1.4056)), (6, (0.0, 0.0, 0.1791)), (6, (0.0, 1.2855, -0.6163)),
        (6, (0.0, -1.2855, -0.6163)), (1, (0.0, 2.1349, 0.0665)),
        (1, (0.0, -2.1349, 0.0665)), (1, (-0.8811, 1.3315, -1.264)),
        (1, (0.8811, 1.3315, -1.264)), (1, (0.8811, -1.3315, -1.264)),
        (1, (-0.8811, -1.3315, -1.264)),
    ]),
    "benzene": (-1.0, 6, [
        (6, (0.0, 1.3952, 0.0)), (6, (1.2083, 0.6976, 0.0)), (6, (1.2083, -0.6976, 0.0)),
        (6, (0.0, -1.3952, 0.0)), (6, (-1.2083, -0.6976, 0.0)),
        (6, (-1.2083, 0.6976, 0.0)), (1, (0.0, 2.4824, 0.0)), (1, (2.1498, 1.2412, 0.0)),
        (1, (2.1498, -1.2412, 0.0)), (1, (0.0, -2.4824, 0.0)),
        (1, (-2.1498, -1.2412, 0.0)), (1, (-2.1498, 1.2412, 0.0)),
    ]),
    "ethane": (1.6, 0, [
        (6, (0.0, 0.0, 0.7622)), (6, (0.0, 0.0, -0.7622)), (1, (0.0, 1.019, 1.1572)),
        (1, (-0.8824, -0.5095, 1.1572)), (1, (0.8824, -0.5095, 1.1572)),
        (1, (0.0, -1.019, -1.1572)), (1, (-0.8824, 0.5095, -1.1572)),
        (1, (0.8824, 0.5095, -1.1572)),
    ]),
    "cyclobutane": (0.1, 4, [
        (6, (0.0, 1.0711, 0.1476)), (6, (0.0, -1.0711, 0.1476)),
        (6, (-1.0711, 0.0, -0.1476)), (6, (1.0711, 0.0, -0.1476)),
        (1, (0.0, 1.9869, -0.4501)), (1, (0.0, 1.3429, 1.2075)),
        (1, (0.0, -1.9869, -0.4501)), (1, (0.0, -1.3429, 1.2075)),
        (1, (-1.9869, 0.0, 0.4501)), (1, (-1.3429, 0.0, -1.2075)),
        (1, (1.9869, 0.0, 0.4501)), (1, (1.3429, 0.0, -1.2075)),
    ]),
    "tetrachloromethane": (-0.3, 0, [
        (6, (0.0, 0.0, 0.0)), (17, (1.0213, 1.0213, 1.0213)),
        (17, (-1.0213, -1.0213, 1.0213)), (17, (-1.0213, 1.0213, -1.0213)),
        (17, (1.0213, -1.0213, -1.0213)),
    ]),
}


def test_cosmors_reproduces_pure_liquid_vapour_pressures():
    """``RT ln p_vap = mu_S(X in X) - mu_gas(X)`` against experiment (#435).

    A pure liquid's free energy of vaporization in COSMO-RS's reference states
    (1 bar gas, 1 mol/mol liquid) is the whole chain: its own conductor
    surface, its own sigma potential, its chemical potential in itself, and the
    gas reference of eq. 21. The panel spans polar and nonpolar liquids and two
    ring compounds, so each term of the gas reference is exercised. Measured
    at RHF/STO-3G on the fine cavity (0.35 A, KLAMT_1998 radii): rms 1.23 and
    largest error 2.13 kcal/mol. With eq. 21's two signs as printed it was rms
    7.69 and 11.07, with every nonpolar liquid about 10 kcal/mol too volatile;
    water alone looked right, because the two errors cancel there.
    """
    vq = pytest.importorskip("vibeqc")
    from vibeqc.solvation.surface import build_conductor_surface

    rt = R_KCAL * T_ROOM_K
    errors = {}
    for name, (log_p, n_ring, atoms) in _VAPOUR_PRESSURE_PANEL.items():
        mol = _mol_ang(vq, atoms)
        res = vq.run_cpcm_scf(
            mol, vq.BasisSet(mol, "sto-3g"), method="rhf",
            solvent=vq.SolventModel(variant="cosmo", **_klamt_fine()),
        )
        surf = build_conductor_surface(
            res, method="rhf", basis="sto-3g", cavity_kind="fine-cfc",
            energy_conductor=float(res.energy),
        )
        desc = rs.segment_descriptors(surf, KLAMT_1998.r_av)
        pot = rs.solvent_sigma_potential(res, KLAMT_1998, method="rhf", basis="sto-3g")
        mu = rs.chemical_potential(
            rs.sigma_profile(desc), pot, KLAMT_1998, solvent_area=desc.total_area
        )
        mu_gas = rs.gas_phase_chemical_potential(
            surf, desc, KLAMT_1998, n_ring_atoms=n_ring
        )
        errors[name] = rs.solvation_free_energy(mu, mu_gas) - rt * math.log(10.0) * log_p

    rms = math.sqrt(sum(e * e for e in errors.values()) / len(errors))
    assert rms < 1.6, errors
    assert max(abs(e) for e in errors.values()) < 3.0, errors


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


# ---------------------------------------------------------------------
# run_job(cosmors=...): COSMO-RS thermodynamics as a job (M0 of
# docs/design_cosmors_run_job.md; maintainer decisions Q1/Q3/Q4, 2026-09-30).
# ---------------------------------------------------------------------


def _water_potential(vq):
    """Water's own sigma potential at RHF/STO-3G, and the solute molecule."""
    water = _mol_ang(vq, _VAPOUR_PRESSURE_PANEL["water"][2])
    solvent = vq.run_cpcm_scf(
        water, vq.BasisSet(water, "sto-3g"), method="rhf",
        solvent=vq.SolventModel(variant="cosmo", **_klamt_fine()),
    )
    pot = rs.solvent_sigma_potential(solvent, KLAMT_1998, method="rhf", basis="sto-3g")
    return water, pot


def test_run_job_cosmors_is_the_library_chain(tmp_path):
    """The job's dG_solv is the library chain on the job's own surface.

    Design note Q6: the orchestration adds no physics, so the number the job
    reports must equal the chain run by hand on the conductor result it
    produced, to machine precision. Water in its own potential is also a pure
    liquid, so dG_solv is RT ln(p_vap/bar), the quantity the vapour-pressure
    panel checks against experiment. The ring term goes through the job as
    Klamt 1998 section 5.1 states it: -0.21 kcal/mol per ring atom on dG_solv.
    """
    vq = pytest.importorskip("vibeqc")
    from vibeqc.solvation.surface import build_conductor_surface

    water, pot = _water_potential(vq)
    opts = vq.CosmoRSOptions(pot)
    assert opts.solvent_area == pytest.approx(pot.solvent_area)
    stem = tmp_path / "water"
    res = vq.run_job(
        water, basis="sto-3g", method="rhf", cosmors=opts,
        output=str(stem), progress=False, write_molden_file=False,
    )
    cr = res.cosmors

    surf = build_conductor_surface(
        res.solvent_result, method="rhf", basis="sto-3g", cavity_kind="fine-cfc",
        energy_conductor=float(res.solvent_result.energy),
    )
    desc = rs.segment_descriptors(surf, KLAMT_1998.r_av)
    mu = rs.chemical_potential(
        rs.sigma_profile(desc), pot, KLAMT_1998, solvent_area=pot.solvent_area
    )
    mu_gas = rs.gas_phase_chemical_potential(surf, desc, KLAMT_1998)
    assert cr.dg_solv == pytest.approx(rs.solvation_free_energy(mu, mu_gas), abs=1e-10)
    assert cr.mu_solvent.total == pytest.approx(mu.total, abs=1e-10)
    assert cr.mu_gas == pytest.approx(mu_gas, abs=1e-10)
    assert cr.solute_protocol == pot.protocol == "rhf/sto-3g/fine-cfc"
    assert cr.parameterization == KLAMT_1998.name
    assert not cr.protocol_matches_fit

    out = stem.with_suffix(".out").read_text()
    assert "## COSMO-RS (klamt1998, 298.15 K)" in out
    assert f"{cr.dg_solv:12.4f} kcal/mol" in out
    assert "ring atoms given: 0" in out
    assert "dmol/bpw91/dnp/cosmo-inf/nspa92" in out
    bib = stem.with_suffix(".bibtex").read_text()
    for key in ("klamt_cosmors_parametrization_1998", "klamt_cosmors_1995",
                "klamt_fine_cavity_2018"):
        assert key in bib, key

    ringed = vq.run_job(
        water, basis="sto-3g", method="rhf",
        cosmors=vq.CosmoRSOptions(pot, n_ring_atoms=2),
        output=str(tmp_path / "ringed"), progress=False, write_molden_file=False,
    )
    assert ringed.cosmors.dg_solv - cr.dg_solv == pytest.approx(2 * -0.21, abs=1e-9)


def test_run_job_cosmors_refuses_a_solute_at_another_protocol(tmp_path):
    """Q3: a solute and a solvent at different levels are refused, not mixed."""
    vq = pytest.importorskip("vibeqc")
    water, pot = _water_potential(vq)
    opts = vq.CosmoRSOptions(pot)
    kw = dict(cosmors=opts, output=str(tmp_path / "x"), progress=False,
              write_molden_file=False)
    with pytest.raises(ValueError, match="different protocol.*basis '3-21g'"):
        vq.run_job(water, basis="3-21g", method="rhf", **kw)
    with pytest.raises(ValueError, match="different protocol.*method 'bpw91'"):
        vq.run_job(water, basis="sto-3g", method="rks", functional="bpw91", **kw)


def test_run_job_cosmors_refuses_a_surface_it_cannot_use(tmp_path):
    """The solvent model must give the surface the parameters were fitted on."""
    vq = pytest.importorskip("vibeqc")
    water, pot = _water_potential(vq)
    kw = dict(cosmors=vq.CosmoRSOptions(pot), output=str(tmp_path / "x"),
              progress=False, write_molden_file=False)
    with pytest.raises(ValueError, match="epsilon=inf"):
        vq.run_job(water, basis="sto-3g", method="rhf", solvent="water", **kw)
    with pytest.raises(ValueError, match="not klamt1998's own"):
        vq.run_job(
            water, basis="sto-3g", method="rhf",
            solvent=vq.SolventModel(epsilon=math.inf, variant="cosmo", cavity="fine"),
            **kw,
        )
    with pytest.raises(ValueError, match="dcosmo-rs"):
        vq.run_job(
            water, basis="sto-3g", method="rhf",
            solvent=vq.SolventModel(
                variant="dcosmo-rs", sigma_potential=pot, **_klamt_fine()
            ),
            **kw,
        )


def test_run_job_cosmors_refuses_what_m0_does_not_cover(tmp_path):
    """Correlated methods and geometry optimization are outside M0."""
    vq = pytest.importorskip("vibeqc")
    water, pot = _water_potential(vq)
    kw = dict(cosmors=vq.CosmoRSOptions(pot), output=str(tmp_path / "x"),
              progress=False, write_molden_file=False)
    with pytest.raises(NotImplementedError, match="mean-field"):
        vq.run_job(water, basis="sto-3g", method="mp2", **kw)
    with pytest.raises(NotImplementedError, match="geometry optimization"):
        vq.run_job(water, basis="sto-3g", method="rhf", optimize=True, **kw)


def test_cosmors_options_say_what_is_missing():
    """Q4 and the combinatorial term: the options refuse what they cannot use."""
    pot = _toy_sigma_potential()
    with pytest.raises(ValueError, match="no solvent surface area"):
        rs.CosmoRSOptions(pot)
    opts = rs.CosmoRSOptions(pot, solvent_area=40.0)
    assert opts.n_ring_atoms == 0 and opts.solvent_area == 40.0
    with pytest.raises(ValueError, match=">= 0"):
        rs.CosmoRSOptions(pot, n_ring_atoms=-1, solvent_area=40.0)
    with pytest.raises(TypeError, match="SigmaPotential"):
        rs.CosmoRSOptions("water", solvent_area=40.0)


def test_cosmors_route_adds_to_the_reaction_field_row():
    """``extra_solvation`` fires a route beside the one reaction-field row.

    The registry picks exactly one ``routes.solvation`` row for the model that
    screened the solute. COSMO-RS thermodynamics computed afterwards on that
    surface add their own row rather than replacing it, and a key with no row
    is warned about rather than dropped silently.
    """
    from vibeqc.output.citations import load_default_database

    db = load_default_database()
    both = db.assemble(
        uses_cpcm=True, solvent_variant="cosmo_fine", extra_solvation=("cosmors",)
    )
    keys = {c.key for c in both.citations}
    assert {
        "klamt_cosmors_1995", "klamt_cosmors_parametrization_1998",
        "klamt_fine_cavity_2018",
    } <= keys
    assert not any("solvation" in w for w in both.warnings), both.warnings
    alone = db.assemble(uses_cpcm=True, solvent_variant="cosmo_fine")
    assert "klamt_cosmors_parametrization_1998" not in {c.key for c in alone.citations}
    odd = db.assemble(
        uses_cpcm=True, solvent_variant="cosmo_fine", extra_solvation=("nonsense",)
    )
    assert any("nonsense" in w for w in odd.warnings)


# ---------------------------------------------------------------------
# M1 of docs/design_cosmors_run_job.md: solvents by name, cached per user;
# and Q4's ring count perceived on request.
# ---------------------------------------------------------------------


def _library_water_entry():
    """A library entry for water, independent of the shipped table's content."""
    from vibeqc.solvation.cosmors.solvent_geometries import SolventGeometry

    atoms = tuple(
        (z, tuple(c * _ANG_TO_BOHR for c in xyz))
        for z, xyz in _VAPOUR_PRESSURE_PANEL["water"][2]
    )
    return SolventGeometry(
        name="water", atoms=atoms, protocol="test fixture", energy_hartree=0.0,
    )


@pytest.fixture
def _water_library(monkeypatch):
    from vibeqc.solvation.cosmors import solvent_geometries

    monkeypatch.setattr(
        solvent_geometries, "SOLVENT_GEOMETRIES", {"water": _library_water_entry()}
    )


def test_ring_atoms_are_the_atoms_on_cycles():
    """Klamt counts atoms that belong to a ring: a fused system counts each once,
    a chain between two rings counts none, and a molecule without a cycle has
    no ring atoms at all."""
    vq = pytest.importorskip("vibeqc")
    from vibeqc.solvation.cosmors.rings import ring_atom_count, ring_atoms

    benzene = _mol_ang(vq, _VAPOUR_PRESSURE_PANEL["benzene"][2])
    assert ring_atom_count(benzene) == 6
    assert ring_atom_count(_mol_ang(vq, _VAPOUR_PRESSURE_PANEL["water"][2])) == 0
    assert ring_atom_count(_mol_ang(vq, _VAPOUR_PRESSURE_PANEL["ethane"][2])) == 0
    assert ring_atom_count(_mol_ang(vq, _VAPOUR_PRESSURE_PANEL["cyclobutane"][2])) == 4

    # Naphthalene's carbon skeleton: two fused hexagons, ten ring atoms.
    skeleton = []
    for shift in (0.0, 2.4249):
        for k in range(6):
            a = math.radians(60 * k + 30)
            p = (1.40 * math.cos(a) + shift, 1.40 * math.sin(a), 0.0)
            if all(np.linalg.norm(np.subtract(p, q)) > 0.5 for _, q in skeleton):
                skeleton.append((6, p))
    assert len(skeleton) == 10
    naph = _mol_ang(vq, skeleton)
    assert ring_atom_count(naph) == 10
    assert ring_atoms([a.Z for a in naph.atoms], np.array([a.xyz for a in naph.atoms])) == list(range(10))

    # A two-carbon bridge between two rings: the ring atoms are counted, the
    # bridge is not, since it lies on no cycle.
    ring_a = [(6, p) for _, p in _VAPOUR_PRESSURE_PANEL["benzene"][2][:6]]
    ring_b = [(6, (x + 7.0, y, z)) for _, (x, y, z) in ring_a]
    bridge = [(6, (1.2083 + 1.3, 0.6976 + 0.7, 0.0)), (6, (7.0 - 1.2083 - 1.3, -0.6976 - 0.7, 0.0))]
    # bring the bridge carbons within bonding distance of each other
    bridge = [(6, (2.6, 1.2, 0.0)), (6, (3.9, 0.3, 0.0)), (6, (4.6, -0.9, 0.0))]
    bridged = _mol_ang(vq, ring_a + ring_b + bridge)
    assert ring_atom_count(bridged) == 12


def test_cosmors_options_take_exactly_one_solvent_spec():
    """A potential or a library name, never both and never neither, and the
    settings that only one of them can carry are refused on the other."""
    pot = _toy_sigma_potential()
    with pytest.raises(ValueError, match="exactly one"):
        rs.CosmoRSOptions()
    with pytest.raises(ValueError, match="exactly one"):
        rs.CosmoRSOptions(pot, solvent="water", solvent_area=40.0)
    with pytest.raises(TypeError, match="solvent name"):
        rs.CosmoRSOptions(solvent=pot)
    # A given potential fixes its temperature and parameter set.
    with pytest.raises(ValueError, match="fixes the temperature"):
        rs.CosmoRSOptions(pot, solvent_area=40.0, temperature_k=310.0)
    same_t = rs.CosmoRSOptions(pot, solvent_area=40.0, temperature_k=pot.temperature_k)
    assert same_t.temperature_k == pot.temperature_k
    import dataclasses

    other = dataclasses.replace(KLAMT_1998, name="other", c_hb=0.0)
    with pytest.raises(ValueError, match="parameter_set= differs"):
        rs.CosmoRSOptions(pot, solvent_area=40.0, parameter_set=other)
    assert rs.CosmoRSOptions(pot, solvent_area=40.0, parameter_set="klamt1998").parameterization is KLAMT_1998
    # Ring atoms: a count or "auto".
    assert rs.CosmoRSOptions(pot, solvent_area=40.0, n_ring_atoms="AUTO").n_ring_atoms == "auto"
    with pytest.raises(ValueError, match="integer or 'auto'"):
        rs.CosmoRSOptions(pot, solvent_area=40.0, n_ring_atoms="six")


def test_cosmors_options_for_a_library_solvent(_water_library, tmp_path):
    """A library solvent resolves its name, defaults its temperature and
    parameter set, and refuses an area it would otherwise silently override."""
    opts = rs.CosmoRSOptions(solvent="H2O", cache_dir=tmp_path)
    assert opts.solvent == "water"
    assert opts.temperature_k == pytest.approx(298.15)
    assert opts.parameterization is KLAMT_1998
    assert opts.sigma_potential is None
    with pytest.raises(ValueError, match="no solvent geometry for 'mercury'"):
        rs.CosmoRSOptions(solvent="mercury")
    with pytest.raises(ValueError, match="cannot be given with a library solvent"):
        rs.CosmoRSOptions(solvent="water", solvent_area=40.0)
    with pytest.raises(ValueError, match="positive temperature"):
        rs.CosmoRSOptions(solvent="water", temperature_k=-1.0)
    assert rs.CosmoRSOptions(solvent="water", temperature_k=350.0).temperature_k == 350.0


def test_cosmors_cache_dir_follows_xdg(monkeypatch, tmp_path):
    from vibeqc.solvation.cosmors.solvents import cosmors_cache_dir

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert cosmors_cache_dir() == tmp_path / "xdg" / "vibeqc" / "cosmors"
    assert cosmors_cache_dir(tmp_path / "mine") == tmp_path / "mine"
    monkeypatch.delenv("XDG_CACHE_HOME")
    assert cosmors_cache_dir().parts[-3:] == (".cache", "vibeqc", "cosmors")


def test_a_library_solvent_surface_is_computed_once_and_read_back(_water_library, tmp_path):
    """The cached record is the computed one, bit for bit, and the key covers
    what the surface depends on: a different basis, parameter radii or
    cavity spacing is a different file, never a stale read."""
    vq = pytest.importorskip("vibeqc")
    import dataclasses

    from vibeqc.solvation.cosmors.solvents import (
        solvent_sigma_potential_from_surface,
        solvent_surface,
    )

    kw = dict(method="rhf", basis="sto-3g", fine_grid_spacing_ang=0.35, cache_dir=tmp_path)
    surface, source = solvent_surface("water", **kw)
    assert source == "computed"
    files = sorted(tmp_path.glob("water__rhf__sto-3g__*.json"))
    assert len(files) == 1
    again, source = solvent_surface("water", **kw)
    assert source == "cache"
    np.testing.assert_array_equal(again.charges, surface.charges)
    np.testing.assert_array_equal(again.areas, surface.areas)
    assert again.provenance == surface.provenance
    assert again.provenance.descriptor() == "rhf/sto-3g/fine-cfc"
    assert "test fixture" in (again.provenance.notes or "")
    assert surface.energy_gas is not None and surface.energy_conductor is not None

    # The potential from the cached record is the one solvent_sigma_potential
    # builds from the run itself, so the two routes agree to machine precision.
    mol = vq.Molecule([vq.Atom(z, xyz) for z, xyz in _library_water_entry().atoms], 0, 1)
    direct_run = vq.run_cpcm_scf(
        mol, vq.BasisSet(mol, "sto-3g"), method="rhf",
        solvent=vq.SolventModel(variant="cosmo", **_klamt_fine()),
    )
    from_run = rs.solvent_sigma_potential(direct_run, KLAMT_1998, method="rhf", basis="sto-3g")
    from_cache = solvent_sigma_potential_from_surface(again, KLAMT_1998)
    np.testing.assert_allclose(from_cache.mu_tilde, from_run.mu_tilde, rtol=0, atol=1e-9)
    assert from_cache.solvent_area == pytest.approx(from_run.solvent_area, abs=1e-9)
    assert from_cache.protocol == from_run.protocol

    # Different protocol or settings: different file.
    other = dataclasses.replace(KLAMT_1998, cavity_radii={**KLAMT_1998.cavity_radii, 8: 1.80})
    solvent_surface("water", **{**kw, "params": other})
    solvent_surface("water", **{**kw, "fine_grid_spacing_ang": 0.40})
    assert len(sorted(tmp_path.glob("water__rhf__sto-3g__*.json"))) == 3
    with pytest.raises(ValueError, match="names no functional"):
        solvent_surface("water", **{**kw, "method": "rks"})
    with pytest.raises(ValueError, match="no cavity radius"):
        solvent_surface(
            "water", **{**kw, "params": dataclasses.replace(KLAMT_1998, cavity_radii={1: 1.3})}
        )


def test_run_job_cosmors_by_solvent_name_is_the_chain_on_the_library_surface(
    _water_library, tmp_path
):
    """``cosmors=CosmoRSOptions(solvent="water")`` is M1: the solvent surface at
    the solute's protocol from the cache, the same chain as a given potential,
    and the output says where the solvent came from. A second job reads the
    surface back. Ring atoms perceived on request enter as Klamt's term."""
    vq = pytest.importorskip("vibeqc")
    from vibeqc.solvation.cosmors.solvents import (
        solvent_sigma_potential_from_surface,
        solvent_surface,
    )

    cache = tmp_path / "cache"
    benzene = _mol_ang(vq, _VAPOUR_PRESSURE_PANEL["benzene"][2])
    stem = tmp_path / "benzene"
    res = vq.run_job(
        benzene, basis="sto-3g", method="rhf",
        cosmors=vq.CosmoRSOptions(solvent="water", n_ring_atoms="auto", cache_dir=cache),
        output=str(stem), progress=False, write_molden_file=False,
    )
    cr = res.cosmors
    assert cr.solvent_name == "water" and cr.solvent_source == "computed"
    assert cr.n_ring_atoms == 6 and cr.ring_atoms_perceived
    assert cr.solute_protocol == cr.solvent_protocol == "rhf/sto-3g/fine-cfc"
    out = stem.with_suffix(".out").read_text()
    assert "Solvent                   = water, surface computed now and cached" in out
    assert "ring atoms: 6, perceived from covalent-radius bonds" in out

    # The same number by hand, from the cached surface.
    surface, source = solvent_surface(
        "water", method="rhf", basis="sto-3g", cache_dir=cache,
    )
    assert source == "cache"
    pot = solvent_sigma_potential_from_surface(surface, KLAMT_1998)
    given = vq.run_job(
        benzene, basis="sto-3g", method="rhf",
        cosmors=vq.CosmoRSOptions(pot, n_ring_atoms=6),
        output=str(tmp_path / "given"), progress=False, write_molden_file=False,
    ).cosmors
    assert cr.dg_solv == pytest.approx(given.dg_solv, abs=1e-9)
    assert cr.mu_solvent.total == pytest.approx(given.mu_solvent.total, abs=1e-9)
    assert given.solvent_source == "given" and not given.ring_atoms_perceived

    # The second job by name reads the cache, and a temperature moves only the
    # potential: the surface is reused.
    warm = vq.run_job(
        benzene, basis="sto-3g", method="rhf",
        cosmors=vq.CosmoRSOptions(solvent="water", temperature_k=320.0, cache_dir=cache),
        output=str(tmp_path / "warm"), progress=False, write_molden_file=False,
    ).cosmors
    assert warm.solvent_source == "cache" and warm.temperature_k == 320.0
    assert warm.n_ring_atoms == 0 and not warm.ring_atoms_perceived
    assert "surface from the per-user cache" in (tmp_path / "warm.out").read_text()
    assert len(list(cache.glob("water__rhf__sto-3g__*.json"))) == 1
    assert warm.dg_solv != pytest.approx(cr.dg_solv, abs=1e-3)


def test_a_library_solvent_runs_kohn_sham_at_the_solutes_functional(_water_library, tmp_path):
    """A KS solute gets a KS solvent: the protocol token is the functional's
    name, the solvent runs RKS with it on run_job's default grid, and the
    recorded protocol is what the solute is checked against."""
    vq = pytest.importorskip("vibeqc")

    water = _mol_ang(vq, _VAPOUR_PRESSURE_PANEL["water"][2])
    res = vq.run_job(
        water, basis="sto-3g", method="rks", functional="bpw91",
        cosmors=vq.CosmoRSOptions(solvent="water", cache_dir=tmp_path),
        output=str(tmp_path / "ks"), progress=False, write_molden_file=False,
    )
    cr = res.cosmors
    assert cr.solute_protocol == cr.solvent_protocol == "bpw91/sto-3g/fine-cfc"
    assert cr.solvent_source == "computed"
    assert len(list(tmp_path.glob("water__bpw91__sto-3g__*.json"))) == 1
    # Water in its own potential is a pure liquid: dG_solv is RT ln(p_vap/bar),
    # which must be of the order the vapour-pressure panel measures, not a
    # number from a solvent at another level.
    assert -6.0 < cr.dg_solv < 4.0, cr.dg_solv


def test_the_shipped_solvent_library_is_sound():
    """Every library solvent is one connected molecule at a recorded protocol,
    within the elements the default parameterization covers, and resolves
    through the CPCM preset aliases (``"H2O"`` is water here too)."""
    vq = pytest.importorskip("vibeqc")
    from vibeqc.solvation.cosmors import available_solvents, solvent_molecule
    from vibeqc.solvation.cosmors.rings import perceive_bonds
    from vibeqc.solvation.cosmors.solvent_geometries import SOLVENT_GEOMETRIES

    names = available_solvents()
    assert {"water", "methanol", "ethanol", "benzene"} <= set(names)
    for name in names:
        entry = SOLVENT_GEOMETRIES[name]
        assert entry.name == name
        assert "optimized" in entry.protocol and "vibe-qc" in entry.protocol
        mol = solvent_molecule(name)
        zs = [int(a.Z) for a in mol.atoms]
        assert set(zs) <= set(KLAMT_1998.cavity_radii), (name, zs)
        pos = np.array([a.xyz for a in mol.atoms])
        bonds = perceive_bonds(zs, pos)
        # Connected: a walk over the bonds reaches every atom.
        seen, frontier = {0}, [0]
        while frontier:
            i = frontier.pop()
            for a, b in bonds:
                for u, v in ((a, b), (b, a)):
                    if u == i and v not in seen:
                        seen.add(v)
                        frontier.append(v)
        assert seen == set(range(len(zs))), name
        # No two atoms on top of each other, no bond absurdly long.
        d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
        np.fill_diagonal(d, np.inf)
        assert d.min() > 0.9 * _ANG_TO_BOHR, name
    assert solvent_molecule("H2O").atoms[0].Z == 8
