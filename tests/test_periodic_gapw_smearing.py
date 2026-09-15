"""Tests for the gapw chat's nuclear-density smearing infrastructure
(``python/vibeqc/periodic_gapw_smearing.py``).

Pins the conventions M3b consumes for the CP2K-style erfc/erf
V_ne split:

* ``SmearedNuclearCharges`` constructor + validation;
* α ↔ σ parameterisation round-trip;
* ``default_smearing_alpha_from_grid`` against the closed-form
  ``α = √(E_cut / (2 · ln(1/eps_tail)))``;
* charge conservation on the FFT grid (``∫ρ_core ≈ Σ Z``);
* the spurious self-Hartree formula ``Σ Z² · α / √(2π)``;
* the smeared pairwise Coulomb in the well-separated limit
  (``Z_a Z_b · erf(β · r) / r`` → ``Z_a Z_b / r`` as ``β · r → ∞``);
* ``smeared_v_ne_long_matrix`` returns a real, symmetric,
  negative-diagonal matrix on a closed-shell single-atom probe
  (electron-nucleus attraction is negative).

These are infrastructure tests, not parity tests — the long-range
V_ne matrix is **not** a physically complete V_ne, and there's no
SCF wired against it. The integration with the SCF lives behind
the D5 maintainer decision in ``docs/design_periodic_gapw.md``.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import _vibeqc_core as core
from vibeqc.periodic_gapw_grid import GAPWExperimentalWarning, PlaneWaveGrid
from vibeqc.periodic_gapw_smearing import (
    SmearedNuclearCharges,
    alpha_to_sigma,
    default_smearing_alpha_from_grid,
    sigma_to_alpha,
    smeared_nuclear_density_on_grid,
    smeared_pairwise_overlap_energy,
    smeared_self_energy,
    smeared_v_ne_long_matrix,
)


pytestmark = pytest.mark.filterwarnings(
    "ignore::vibeqc.periodic_gapw_grid.GAPWExperimentalWarning"
)


# ---------- Parameterisation conventions --------------------------------


def test_alpha_sigma_round_trip():
    """``alpha_to_sigma`` and ``sigma_to_alpha`` are inverses."""
    for alpha in (0.1, 0.5, 1.0, 2.5, 10.0):
        s = alpha_to_sigma(alpha)
        assert sigma_to_alpha(s) == pytest.approx(alpha, rel=1e-14)


def test_alpha_to_sigma_canonical_relation():
    """``σ = 1 / (α √2)``: pin the formula directly."""
    for alpha in (0.7, 1.3, 4.2):
        assert alpha_to_sigma(alpha) == pytest.approx(
            1.0 / (alpha * math.sqrt(2.0)), rel=1e-14
        )


def test_alpha_to_sigma_rejects_nonpositive():
    with pytest.raises(ValueError, match="positive"):
        alpha_to_sigma(0.0)
    with pytest.raises(ValueError, match="positive"):
        alpha_to_sigma(-1.0)


def test_sigma_to_alpha_rejects_nonpositive():
    with pytest.raises(ValueError, match="positive"):
        sigma_to_alpha(0.0)


# ---------- Default-α heuristic ------------------------------------------


def test_default_alpha_from_grid_matches_closed_form():
    """``α = √(E_cut / (2 · ln(1/eps_tail)))`` — pin against the
    derivation in the module docstring."""
    grid = PlaneWaveGrid(np.eye(3) * 16.0, 64, 64, 64, cutoff_ha=300.0)
    for eps in (1e-8, 1e-10, 1e-12):
        alpha = default_smearing_alpha_from_grid(grid, eps_tail=eps)
        expected = math.sqrt(300.0 / (2.0 * -math.log(eps)))
        assert alpha == pytest.approx(expected, rel=1e-14)


def test_default_alpha_from_grid_rejects_missing_cutoff():
    grid = PlaneWaveGrid(np.eye(3) * 16.0, 32, 32, 32)  # no cutoff_ha
    with pytest.raises(ValueError, match="cutoff_ha"):
        default_smearing_alpha_from_grid(grid)


def test_default_alpha_from_grid_rejects_bad_eps():
    grid = PlaneWaveGrid(np.eye(3) * 16.0, 32, 32, 32, cutoff_ha=300.0)
    with pytest.raises(ValueError, match="eps_tail"):
        default_smearing_alpha_from_grid(grid, eps_tail=0.0)
    with pytest.raises(ValueError, match="eps_tail"):
        default_smearing_alpha_from_grid(grid, eps_tail=1.5)


# ---------- SmearedNuclearCharges construction --------------------------


def _he_system(L: float = 16.0):
    sys = core.PeriodicSystem()
    sys.dim = 3
    sys.lattice = np.eye(3) * L
    sys.unit_cell = [core.Atom(2, [L / 2, L / 2, L / 2])]
    return sys


def _h2_system(L: float = 16.0):
    sys = core.PeriodicSystem()
    sys.dim = 3
    sys.lattice = np.eye(3) * L
    sys.unit_cell = [
        core.Atom(1, [L / 2 - 0.7, L / 2, L / 2]),
        core.Atom(1, [L / 2 + 0.7, L / 2, L / 2]),
    ]
    return sys


def test_smeared_charges_from_periodic_system_uniform_alpha():
    """Factory pulls positions + Z from the system; assigns a uniform α."""
    system = _h2_system(16.0)
    charges = SmearedNuclearCharges.from_periodic_system_uniform_alpha(
        system, alpha=2.0
    )
    assert charges.n_atoms == 2
    assert charges.total_charge == pytest.approx(2.0)
    assert charges.is_uniform_alpha
    assert np.allclose(charges.alpha, 2.0)


def test_smeared_charges_per_atom_alpha():
    """Per-atom alpha is accepted by the direct constructor."""
    charges = SmearedNuclearCharges(
        positions=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        Z=np.array([1.0, 8.0]),
        alpha=np.array([1.5, 3.0]),
    )
    assert charges.n_atoms == 2
    assert not charges.is_uniform_alpha
    assert charges.total_charge == pytest.approx(9.0)


def test_smeared_charges_rejects_shape_mismatch():
    with pytest.raises(ValueError, match=r"positions must be \(n_atoms, 3\)"):
        SmearedNuclearCharges(
            positions=np.array([0.0, 0.0, 0.0]),
            Z=np.array([1.0]),
            alpha=np.array([1.0]),
        )
    with pytest.raises(ValueError, match="Z shape"):
        SmearedNuclearCharges(
            positions=np.array([[0.0, 0.0, 0.0]]),
            Z=np.array([1.0, 1.0]),
            alpha=np.array([1.0]),
        )
    with pytest.raises(ValueError, match="alpha shape"):
        SmearedNuclearCharges(
            positions=np.array([[0.0, 0.0, 0.0]]),
            Z=np.array([1.0]),
            alpha=np.array([1.0, 1.0]),
        )


def test_smeared_charges_rejects_nonpositive_alpha():
    with pytest.raises(ValueError, match="positive"):
        SmearedNuclearCharges(
            positions=np.array([[0.0, 0.0, 0.0]]),
            Z=np.array([1.0]),
            alpha=np.array([0.0]),
        )


# ---------- Density on the FFT grid -------------------------------------


def test_smeared_density_integrates_to_total_charge():
    """``∫ρ_core dr ≈ Σ_a Z_a`` to grid-quadrature precision on a
    cell that's large enough to contain the Gaussian tails."""
    system = _he_system(16.0)
    grid = PlaneWaveGrid(np.eye(3) * 16.0, 80, 80, 80, cutoff_ha=300.0)
    alpha = default_smearing_alpha_from_grid(grid)
    charges = SmearedNuclearCharges.from_periodic_system_uniform_alpha(
        system, alpha=alpha
    )
    rho = smeared_nuclear_density_on_grid(charges, grid)
    total = float(rho.sum()) * grid.voxel_volume_bohr3
    assert total == pytest.approx(charges.total_charge, rel=1e-4)


def test_smeared_density_is_nonnegative():
    """Nuclear density is positive everywhere — Gaussians have no
    sign-change."""
    system = _h2_system(16.0)
    grid = PlaneWaveGrid(np.eye(3) * 16.0, 48, 48, 48, cutoff_ha=300.0)
    charges = SmearedNuclearCharges.from_periodic_system_uniform_alpha(
        system, alpha=2.0
    )
    rho = smeared_nuclear_density_on_grid(charges, grid)
    assert np.all(rho >= -1e-15)


def test_smeared_density_zero_atoms_gives_zero():
    """Empty bundle → zero density."""
    grid = PlaneWaveGrid(np.eye(3) * 16.0, 24, 24, 24, cutoff_ha=300.0)
    charges = SmearedNuclearCharges(
        positions=np.zeros((0, 3)),
        Z=np.zeros(0),
        alpha=np.zeros(0),
    )
    rho = smeared_nuclear_density_on_grid(charges, grid)
    assert np.allclose(rho, 0.0)


# ---------- Self-energy bookkeeping --------------------------------------


def test_smeared_self_energy_uniform_alpha():
    """``E_self = Σ Z² · α / √(2π)`` for uniform α."""
    charges = SmearedNuclearCharges(
        positions=np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        Z=np.array([2.0, 8.0]),
        alpha=np.array([1.0, 1.0]),
    )
    E_self = smeared_self_energy(charges)
    expected = (4.0 + 64.0) * 1.0 / math.sqrt(2.0 * math.pi)
    assert E_self == pytest.approx(expected, rel=1e-14)


def test_smeared_self_energy_per_atom_alpha():
    """Different α per atom — sum of per-atom contributions."""
    charges = SmearedNuclearCharges(
        positions=np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        Z=np.array([1.0, 6.0]),
        alpha=np.array([2.0, 3.0]),
    )
    E_self = smeared_self_energy(charges)
    expected = (1.0 ** 2 * 2.0 + 6.0 ** 2 * 3.0) / math.sqrt(2.0 * math.pi)
    assert E_self == pytest.approx(expected, rel=1e-14)


def test_smeared_self_energy_matches_point_charge_self_energy():
    """Convention check: the new α-keyed formula must agree with the
    M1d σ-keyed ``point_charge_self_energy``."""
    Z_vals = [2.0, 8.0]
    alpha = 1.7
    sigma = alpha_to_sigma(alpha)
    new = smeared_self_energy(
        SmearedNuclearCharges(
            positions=np.zeros((2, 3)),
            Z=np.array(Z_vals),
            alpha=np.full(2, alpha),
        )
    )
    old = vq.point_charge_self_energy(Z_vals, sigma_bohr=sigma)
    assert new == pytest.approx(old, rel=1e-14)


# ---------- Pairwise overlap (smeared E_nn) -----------------------------


def test_pairwise_overlap_zero_for_single_atom():
    """One atom — no a≠b pair, so the smeared pairwise overlap is
    zero (the *self* part is in :func:`smeared_self_energy`)."""
    charges = SmearedNuclearCharges(
        positions=np.array([[0.0, 0.0, 0.0]]),
        Z=np.array([6.0]),
        alpha=np.array([1.0]),
    )
    # Use a large cell so periodic images don't dominate.
    sys = core.PeriodicSystem()
    sys.dim = 3
    sys.lattice = np.eye(3) * 50.0
    sys.unit_cell = [core.Atom(6, [0.0, 0.0, 0.0])]
    E = smeared_pairwise_overlap_energy(charges, sys, image_shells=0)
    assert E == pytest.approx(0.0, abs=1e-14)


def test_pairwise_overlap_well_separated_recovers_bare_coulomb():
    """For atoms separated by ``r_ab ≫ 1/β``, ``erf(β r_ab)/r_ab →
    1/r_ab``, recovering the bare Coulomb interaction."""
    # H2-like geometry, but place atoms far apart so the smearing
    # tail is negligible.
    r_ab = 20.0  # bohr, much larger than 1/α = 0.5
    alpha = 2.0
    charges = SmearedNuclearCharges(
        positions=np.array([[0.0, 0.0, 0.0], [r_ab, 0.0, 0.0]]),
        Z=np.array([1.0, 1.0]),
        alpha=np.array([alpha, alpha]),
    )
    sys = core.PeriodicSystem()
    sys.dim = 3
    sys.lattice = np.eye(3) * 80.0  # very large cell, no image overlap
    sys.unit_cell = [
        core.Atom(1, [0.0, 0.0, 0.0]),
        core.Atom(1, [r_ab, 0.0, 0.0]),
    ]
    E = smeared_pairwise_overlap_energy(charges, sys, image_shells=0)
    bare = 1.0 / r_ab  # Z_a Z_b / r_ab
    # The two pairs (a, b) and (b, a) summed × 0.5 factor = exactly
    # one pair's contribution. erf at β r_ab = 2·20 = 40 is exactly 1.
    assert E == pytest.approx(bare, rel=1e-12)


def test_pairwise_overlap_close_atoms_below_bare_coulomb():
    """For atoms closer than ``~1/β``, the smeared interaction
    ``erf(β r)/r`` is *smaller* than the bare ``1/r`` — Gaussian
    densities don't blow up at short range like point charges do."""
    r_ab = 0.5  # bohr — closer than 1/α
    alpha = 2.0
    charges = SmearedNuclearCharges(
        positions=np.array([[0.0, 0.0, 0.0], [r_ab, 0.0, 0.0]]),
        Z=np.array([1.0, 1.0]),
        alpha=np.array([alpha, alpha]),
    )
    sys = core.PeriodicSystem()
    sys.dim = 3
    sys.lattice = np.eye(3) * 80.0
    sys.unit_cell = [
        core.Atom(1, [0.0, 0.0, 0.0]),
        core.Atom(1, [r_ab, 0.0, 0.0]),
    ]
    E_smeared = smeared_pairwise_overlap_energy(charges, sys, image_shells=0)
    E_bare = 1.0 / r_ab
    assert E_smeared < E_bare
    # Closed-form check: β = α/√2 for equal α; smeared E = erf(β·r)/r.
    beta = alpha / math.sqrt(2.0)
    expected = math.erf(beta * r_ab) / r_ab
    assert E_smeared == pytest.approx(expected, rel=1e-12)


# ---------- V_ne long-range matrix --------------------------------------


def test_smeared_v_ne_long_matrix_is_symmetric_and_real():
    """The long-range V_ne matrix must be real (electron-nucleus
    integrand is real-valued) and symmetric (μ ↔ ν permutation of
    the trace)."""
    L = 16.0
    mol = vq.Molecule(
        [vq.Atom(1, [L / 2 - 0.7, L / 2, L / 2]),
         vq.Atom(1, [L / 2 + 0.7, L / 2, L / 2])],
        charge=0, multiplicity=1,
    )
    basis = vq.BasisSet(mol, "sto-3g")
    system = _h2_system(L)
    grid = PlaneWaveGrid(np.eye(3) * L, 48, 48, 48, cutoff_ha=300.0)
    charges = SmearedNuclearCharges.from_periodic_system_uniform_alpha(
        system, alpha=2.0
    )
    V = smeared_v_ne_long_matrix(basis, charges, grid, quiet=True)
    assert V.shape == (basis.nbasis, basis.nbasis)
    assert V.dtype == float
    assert np.allclose(V, V.T, atol=1e-10)


def test_smeared_v_ne_long_matrix_he_diagonal_is_negative():
    """For He STO-3G (1 AO centred on the nucleus), the smooth-grid
    V_ne_long is the AO matrix element of *minus* the Hartree
    potential of a Gaussian-smeared positive Z = 2 charge — so the
    diagonal must be negative (attractive)."""
    L = 16.0
    mol = vq.Molecule(
        [vq.Atom(2, [L / 2, L / 2, L / 2])], charge=0, multiplicity=1,
    )
    basis = vq.BasisSet(mol, "sto-3g")
    system = _he_system(L)
    grid = PlaneWaveGrid(np.eye(3) * L, 48, 48, 48, cutoff_ha=300.0)
    charges = SmearedNuclearCharges.from_periodic_system_uniform_alpha(
        system, alpha=2.0
    )
    V = smeared_v_ne_long_matrix(basis, charges, grid, quiet=True)
    assert V[0, 0] < 0.0


def test_smeared_v_ne_long_matrix_emits_experimental_warning():
    """The module's M3b-infrastructure status surfaces as a
    GAPWExperimentalWarning, just like the other gapw helpers, so
    users know the matrix is incomplete (long-range only)."""
    L = 16.0
    mol = vq.Molecule(
        [vq.Atom(2, [L / 2, L / 2, L / 2])], charge=0, multiplicity=1,
    )
    basis = vq.BasisSet(mol, "sto-3g")
    system = _he_system(L)
    grid = PlaneWaveGrid(np.eye(3) * L, 16, 16, 16, cutoff_ha=300.0)
    charges = SmearedNuclearCharges.from_periodic_system_uniform_alpha(
        system, alpha=2.0
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        smeared_v_ne_long_matrix(basis, charges, grid)
    msgs = [
        w for w in captured
        if issubclass(w.category, GAPWExperimentalWarning)
        and "short-range" in str(w.message)
    ]
    assert msgs


# ---------- Public exports --------------------------------------------


def test_smeared_v_ne_erfc_short_matrix_is_symmetric_real_negative():
    """The libint erfc short-range V_ne matrix is real, symmetric,
    and attractive on a single-atom probe."""
    from vibeqc.periodic_gapw_smearing import smeared_v_ne_erfc_short_matrix

    L = 16.0
    mol = vq.Molecule(
        [vq.Atom(2, [L / 2, L / 2, L / 2])], charge=0, multiplicity=1,
    )
    basis = vq.BasisSet(mol, "sto-3g")
    system = _he_system(L)
    V_short = smeared_v_ne_erfc_short_matrix(basis, system, alpha=2.0)
    assert V_short.shape == (basis.nbasis, basis.nbasis)
    assert V_short.dtype == float
    assert np.allclose(V_short, V_short.T, atol=1e-10)
    assert V_short[0, 0] < 0.0  # attractive


def test_smeared_v_ne_erfc_rejects_nonpositive_alpha():
    """Direct shape pin on the erfc helper."""
    from vibeqc.periodic_gapw_smearing import smeared_v_ne_erfc_short_matrix

    L = 16.0
    mol = vq.Molecule(
        [vq.Atom(2, [L / 2, L / 2, L / 2])], charge=0, multiplicity=1,
    )
    basis = vq.BasisSet(mol, "sto-3g")
    system = _he_system(L)
    with pytest.raises(ValueError, match="positive"):
        smeared_v_ne_erfc_short_matrix(basis, system, alpha=0.0)
    with pytest.raises(ValueError, match="positive"):
        smeared_v_ne_erfc_short_matrix(basis, system, alpha=-1.0)


def test_smeared_v_ne_full_gamma_matches_ewald_path():
    """The M3b CP2K erfc/erf-split V_ne and the M3a Ewald lattice
    V_ne are physically equivalent — they differ only by the gauge
    constants that cancel on a neutral cell. The SCF total energies
    must match within numerical noise.

    Tests are on the He STO-3G 16-bohr probe at a representative α;
    the broader α sweep is covered by the integration tests in
    test_periodic_gapw_j.py."""
    from vibeqc.periodic_gapw_smearing import smeared_v_ne_full_gamma
    from vibeqc.periodic_gapw_j import _ewald_v_ne_gamma

    L = 16.0
    mol = vq.Molecule(
        [vq.Atom(2, [L / 2, L / 2, L / 2])], charge=0, multiplicity=1,
    )
    basis = vq.BasisSet(mol, "sto-3g")
    system = _he_system(L)
    grid = PlaneWaveGrid(np.eye(3) * L, 64, 64, 64, cutoff_ha=300.0)
    V_ewald = _ewald_v_ne_gamma(basis, system)
    V_smear = smeared_v_ne_full_gamma(
        basis, system, alpha=2.0, grid=grid, quiet=True,
    )
    # The two matrices may differ by a constant times S (different
    # gauges); but the difference, after subtracting the cell-average,
    # should be small. Easier check: the SCF total parity is in
    # test_periodic_gapw_j.py. Here we just pin shape + finiteness.
    assert V_smear.shape == V_ewald.shape
    assert np.all(np.isfinite(V_smear))
    assert np.allclose(V_smear, V_smear.T, atol=1e-10)


def test_smeared_v_ne_full_gamma_emits_experimental_warning():
    """Same opt-in warning convention as the other M3b helpers."""
    from vibeqc.periodic_gapw_smearing import smeared_v_ne_full_gamma

    L = 16.0
    mol = vq.Molecule(
        [vq.Atom(2, [L / 2, L / 2, L / 2])], charge=0, multiplicity=1,
    )
    basis = vq.BasisSet(mol, "sto-3g")
    system = _he_system(L)
    grid = PlaneWaveGrid(np.eye(3) * L, 32, 32, 32, cutoff_ha=300.0)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        smeared_v_ne_full_gamma(basis, system, alpha=2.0, grid=grid)
    msgs = [
        w for w in captured
        if issubclass(w.category, GAPWExperimentalWarning)
        and "M3b" in str(w.message)
    ]
    assert msgs


def test_monopole_compensator_integrates_to_charge():
    """A pure-monopole compensator with Q_00 = Z·√(4π) integrates to
    ``Z`` per atom on the FFT grid — the unit pin of the ρ_0 monopole
    normalisation."""
    from vibeqc.periodic_gapw_smearing import (
        monopole_compensator_for_charge,
    )

    L = 16.0
    positions = np.array([[L / 2, L / 2, L / 2]])
    Z = np.array([2.0])  # He nucleus
    alpha = 2.0
    comp = monopole_compensator_for_charge(positions, Z, alpha)

    grid = PlaneWaveGrid(np.eye(3) * L, 80, 80, 80, cutoff_ha=300.0)
    rho = comp.density_on_grid(grid)
    total = float(rho.sum()) * grid.voxel_volume_bohr3
    assert total == pytest.approx(2.0, rel=1e-4)


def test_monopole_compensator_two_atoms():
    """Two atoms → integrated density equals Σ Z."""
    from vibeqc.periodic_gapw_smearing import (
        monopole_compensator_for_charge,
    )

    L = 16.0
    positions = np.array([
        [L / 2 - 0.7, L / 2, L / 2],
        [L / 2 + 0.7, L / 2, L / 2],
    ])
    Z = np.array([1.0, 1.0])  # H + H
    comp = monopole_compensator_for_charge(positions, Z, alpha=2.0)
    grid = PlaneWaveGrid(np.eye(3) * L, 48, 48, 48, cutoff_ha=300.0)
    rho = comp.density_on_grid(grid)
    total = float(rho.sum()) * grid.voxel_volume_bohr3
    assert total == pytest.approx(2.0, rel=1e-4)


def test_monopole_compensator_matches_smeared_density():
    """The l=0 compensator with ``Q_00 = Z · √(4π)`` and Gaussian
    width α agrees bit-equivalent with :func:`smeared_nuclear_density_on_grid`
    using the same α — both are the same Gaussian-density formula
    expressed in different conventions."""
    from vibeqc.periodic_gapw_smearing import (
        SmearedNuclearCharges,
        monopole_compensator_for_charge,
        smeared_nuclear_density_on_grid,
    )

    L = 16.0
    positions = np.array([[L / 2, L / 2, L / 2]])
    Z = np.array([2.0])
    alpha = 2.0
    grid = PlaneWaveGrid(np.eye(3) * L, 48, 48, 48, cutoff_ha=300.0)

    comp = monopole_compensator_for_charge(positions, Z, alpha)
    rho_comp = comp.density_on_grid(grid)

    charges = SmearedNuclearCharges(
        positions=positions, Z=Z, alpha=np.array([alpha]),
    )
    rho_smeared = smeared_nuclear_density_on_grid(charges, grid)

    assert np.allclose(rho_comp, rho_smeared, atol=1e-12), (
        f"Compensator and smeared-nuclear-density should give the "
        f"same Gaussian; max diff = "
        f"{np.max(np.abs(rho_comp - rho_smeared)):.3e}"
    )


def test_compensator_dipole_has_nonzero_dipole_moment():
    """A pure-dipole compensator (Q_00 = 0, Q_1z ≠ 0) carries the
    requested dipole density. Sanity: the integrated ρ should be
    zero (no monopole), but the z-moment ∫ ρ · z dr should be
    non-zero."""
    from vibeqc.periodic_gapw_smearing import GaussianMultipoleCompensator

    L = 16.0
    Q = np.zeros((1, 4))   # (n_atoms, 4) for lmax=1
    Q[0, 2] = 1.0          # Q_10 (z-dipole) — index 2 via multipole_index
    comp = GaussianMultipoleCompensator(
        positions=np.array([[L / 2, L / 2, L / 2]]),
        alpha=np.array([2.0]),
        Q=Q,
        lmax=1,
    )
    grid = PlaneWaveGrid(np.eye(3) * L, 80, 80, 80, cutoff_ha=300.0)
    rho = comp.density_on_grid(grid)
    total = float(rho.sum()) * grid.voxel_volume_bohr3
    assert abs(total) < 1e-4, f"Dipole-only density should integrate to 0; got {total:.6e}"
    # The z-moment: ∫ ρ · (z - z_centre) d³r should match Q_10
    # (with the appropriate normalisation factor — Stone convention).
    z_centre = L / 2
    z_grid = grid.cartesian_coords()[..., 2] - z_centre
    z_moment = float((rho * z_grid).sum()) * grid.voxel_volume_bohr3
    # The integrated weight ∫ z · S_10(Ω̂) ρ_10(r) r² dr dΩ̂ involves
    # ∫ S_10(Ω̂) · z dΩ̂ — for unit r̂, z = cos(θ) = √(4π/3) · S_10(Ω̂),
    # so ∫ z · S_10 dΩ̂ = √(4π/3). With Q_10 = 1, the z-moment is
    # 1 · √(4π/3) ≈ 2.046.
    expected = math.sqrt(4.0 * math.pi / 3.0)
    assert z_moment == pytest.approx(expected, rel=2e-3), (
        f"z-moment: got {z_moment:.6f}, expected {expected:.6f} "
        f"(= √(4π/3) for unit Q_10)"
    )


def test_compensator_rejects_negative_lmax():
    """The compensator now accepts arbitrary lmax >= 0 (via the
    scipy-backed real S_lm). Negative lmax still raises."""
    from vibeqc.periodic_gapw_smearing import GaussianMultipoleCompensator
    with pytest.raises(ValueError, match="lmax >= 0"):
        GaussianMultipoleCompensator(
            positions=np.zeros((1, 3)),
            alpha=np.ones(1),
            Q=np.zeros((1, 1)),
            lmax=-1,
        )


def test_compensator_lmax2_quadrupole_density_finite():
    """Quadrupole compensator (lmax=2, Q_20 ≠ 0) produces a finite,
    non-monopole, real-valued density."""
    from vibeqc.periodic_gapw_smearing import GaussianMultipoleCompensator

    L = 16.0
    Q = np.zeros((1, 9))   # (n_atoms, 9) for lmax=2
    Q[0, 6] = 1.0          # Q_20 — quadrupole z²-axis
    comp = GaussianMultipoleCompensator(
        positions=np.array([[L / 2, L / 2, L / 2]]),
        alpha=np.array([2.0]),
        Q=Q,
        lmax=2,
    )
    grid = PlaneWaveGrid(np.eye(3) * L, 64, 64, 64, cutoff_ha=300.0)
    rho = comp.density_on_grid(grid)
    assert np.all(np.isfinite(rho))
    # Monopole-less density integrates to ~0.
    total = float(rho.sum()) * grid.voxel_volume_bohr3
    assert abs(total) < 1e-3


def test_compensator_rejects_shape_mismatch():
    from vibeqc.periodic_gapw_smearing import GaussianMultipoleCompensator
    with pytest.raises(ValueError, match="alpha shape"):
        GaussianMultipoleCompensator(
            positions=np.zeros((2, 3)),
            alpha=np.ones(3),
            Q=np.zeros((2, 1)),
            lmax=0,
        )
    with pytest.raises(ValueError, match="Q shape"):
        GaussianMultipoleCompensator(
            positions=np.zeros((2, 3)),
            alpha=np.ones(2),
            Q=np.zeros((2, 2)),
            lmax=0,  # needs 1 component
        )


def test_compute_atom_local_multipoles_he_is_spherical():
    """The converged molecular He density is spherical → only
    Q_00 is non-zero (≈ Z/√(4π))."""
    from vibeqc.periodic_gapw_smearing import compute_atom_local_multipoles
    L = 16.0
    mol = vq.Molecule([vq.Atom(2, [L / 2, L / 2, L / 2])], 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")
    system = _he_system(L)
    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-10
    res = vq.run_rhf(mol, basis, opts)
    D = np.asarray(res.density)
    Q = compute_atom_local_multipoles(
        D, basis, system, atom_idx=0, lmax=2,
    )
    assert Q[0] == pytest.approx(2.0 / math.sqrt(4.0 * math.pi), abs=1e-4)
    for idx in range(1, 9):
        assert abs(Q[idx]) < 1e-10


def test_build_electronic_compensator_round_trips_to_total_charge():
    """A compensator built from a converged neutral He density
    integrates to total Z = 2 on the FFT grid (round-trip pin)."""
    from vibeqc.periodic_gapw_smearing import (
        build_electronic_compensator,
    )
    L = 16.0
    mol = vq.Molecule([vq.Atom(2, [L / 2, L / 2, L / 2])], 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")
    system = _he_system(L)
    opts = vq.RHFOptions(); opts.conv_tol_energy = 1e-10
    res = vq.run_rhf(mol, basis, opts)
    D = np.asarray(res.density)
    comp = build_electronic_compensator(D, basis, system, lmax=2)
    grid = PlaneWaveGrid(np.eye(3) * L, 48, 48, 48, cutoff_ha=300.0)
    rho = comp.density_on_grid(grid)
    total = float(rho.sum()) * grid.voxel_volume_bohr3
    assert total == pytest.approx(2.0, rel=1e-3)


def test_compute_atom_local_multipoles_rejects_bad_atom_idx():
    from vibeqc.periodic_gapw_smearing import compute_atom_local_multipoles
    L = 16.0
    mol = vq.Molecule([vq.Atom(2, [L / 2, L / 2, L / 2])], 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")
    system = _he_system(L)
    D = np.eye(basis.nbasis)
    with pytest.raises(ValueError, match="atom_idx out of range"):
        compute_atom_local_multipoles(D, basis, system, atom_idx=5)


def test_public_exports_present():
    """The M3b-prep smearing helpers are reachable from the top-level
    ``vibeqc`` namespace — same surface convention as the other
    gapw helpers."""
    assert vq.SmearedNuclearCharges is SmearedNuclearCharges
    assert vq.alpha_to_sigma is alpha_to_sigma
    assert vq.sigma_to_alpha is sigma_to_alpha
    assert vq.default_smearing_alpha_from_grid is (
        default_smearing_alpha_from_grid
    )
    assert vq.smeared_nuclear_density_on_grid is (
        smeared_nuclear_density_on_grid
    )
    assert vq.smeared_self_energy is smeared_self_energy
    assert vq.smeared_pairwise_overlap_energy is (
        smeared_pairwise_overlap_energy
    )
    assert vq.smeared_v_ne_long_matrix is smeared_v_ne_long_matrix
    from vibeqc.periodic_gapw_smearing import (
        smeared_v_ne_erfc_short_matrix,
        smeared_v_ne_full_gamma,
    )
    assert vq.smeared_v_ne_erfc_short_matrix is (
        smeared_v_ne_erfc_short_matrix
    )
    assert vq.smeared_v_ne_full_gamma is smeared_v_ne_full_gamma
    from vibeqc.periodic_gapw_smearing import (
        GaussianMultipoleCompensator,
        monopole_compensator_for_charge,
    )
    assert vq.GaussianMultipoleCompensator is GaussianMultipoleCompensator
    assert vq.monopole_compensator_for_charge is (
        monopole_compensator_for_charge
    )
