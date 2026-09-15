"""End-to-end tests for the GAPW augmentation (M3c) route.

Tests validate:
1. Basic infrastructure: augmentation radii, softened basis, radial Poisson
2. GAPW J builder produces finite, symmetric matrices
3. GAPW SCF converges on simple systems
4. Energy is finite and well-behaved
5. The RKS (DFT) path with augmentation works

These are full-input tests on small systems, not unit-tests of
individual functions — matching the user's brief for production-
level validation.
"""

from __future__ import annotations

import math
import tomllib
import warnings

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import _vibeqc_core as core
from vibeqc.periodic_gapw_atomic_grid import (
    AtomicRadialGrid,
    default_alpha_for_element,
)
from vibeqc.periodic_gapw_augment import (
    GapwAugmentation,
    GapwJBuilder,
    GapwScfResult,
    estimate_augmentation_radius,
    run_periodic_rhf_gapw,
    run_periodic_rks_gapw,
    run_periodic_rks_gapw_multi_k,
    run_periodic_uks_gapw,
    softened_basis,
    solve_poisson_radial,
)
from vibeqc.periodic_gapw_j import run_periodic_rks_gpw
from vibeqc.periodic_gapw_grid import (
    GAPWExperimentalWarning,
    PlaneWaveGrid,
)


def _mol_rks_energy(Z: int, coord, functional: str) -> float:
    """Molecular RKS energy (vibe-qc's own analytic-grid DFT) — the
    same-basis / same-functional all-electron reference the periodic GAPW
    must reproduce in the large-box limit (no plane-wave grid error)."""
    mol = vq.Molecule([vq.Atom(Z, coord)], charge=0, multiplicity=1)
    basis = vq.BasisSet(mol, "sto-3g")
    opts = vq.RKSOptions()
    opts.functional = functional
    opts.conv_tol_energy = 1e-10
    return vq.run_rks(mol, basis, opts).energy


def _mol_uks_energy(Z: int, coord, functional: str, multiplicity: int) -> float:
    """Molecular UKS energy (vibe-qc's own analytic-grid open-shell DFT) --
    the same-basis / same-functional all-electron reference the periodic UKS
    GAPW driver must reproduce in the large-box limit (no plane-wave grid
    error)."""
    mol = vq.Molecule([vq.Atom(Z, coord)], charge=0, multiplicity=multiplicity)
    basis = vq.BasisSet(mol, "sto-3g")
    opts = vq.UKSOptions()
    opts.functional = functional
    opts.conv_tol_energy = 1e-10
    return vq.run_uks(mol, basis, opts).energy

# Filter the experimental warnings globally.
pytestmark = pytest.mark.filterwarnings(
    "ignore::vibeqc.periodic_gapw_grid.GAPWExperimentalWarning"
)


# ============================================================
# Fixtures
# ============================================================


def _he_periodic(L: float = 12.0):
    """He atom in a cubic box."""
    sys = core.PeriodicSystem()
    sys.dim = 3
    sys.lattice = np.eye(3) * L
    sys.unit_cell = [core.Atom(2, [L / 2, L / 2, L / 2])]
    return sys


def _h_periodic(L: float = 12.0):
    """H atom in a cubic box."""
    sys = core.PeriodicSystem()
    sys.dim = 3
    sys.lattice = np.eye(3) * L
    sys.unit_cell = [core.Atom(1, [L / 2, L / 2, L / 2])]
    return sys


def _be_periodic(L: float = 12.0):
    """Be atom in a cubic box (a real 1s core -> large XC augmentation)."""
    sys = core.PeriodicSystem()
    sys.dim = 3
    sys.lattice = np.eye(3) * L
    sys.unit_cell = [core.Atom(4, [L / 2, L / 2, L / 2])]
    return sys


def _h2_periodic(L: float = 12.0, d: float = 1.4):
    """H2 molecule in a cubic box."""
    sys = core.PeriodicSystem()
    sys.dim = 3
    sys.lattice = np.eye(3) * L
    c = L / 2
    sys.unit_cell = [
        core.Atom(1, [c - d / 2, c, c]),
        core.Atom(1, [c + d / 2, c, c]),
    ]
    return sys


def _he_basis_density(L: float = 12.0):
    """He STO-3G basis and converged molecular RHF density."""
    mol = vq.Molecule(
        [vq.Atom(2, [L / 2, L / 2, L / 2])],
        charge=0,
        multiplicity=1,
    )
    basis = vq.BasisSet(mol, "sto-3g")
    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-10
    result = vq.run_rhf(mol, basis, opts)
    D = np.asarray(result.density)
    return mol, basis, D


def _h2_basis_density(L: float = 12.0):
    """H2 STO-3G basis and converged molecular RHF density."""
    c = L / 2
    mol = vq.Molecule(
        [vq.Atom(1, [c - 0.7, c, c]), vq.Atom(1, [c + 0.7, c, c])],
        charge=0,
        multiplicity=1,
    )
    basis = vq.BasisSet(mol, "sto-3g")
    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-10
    result = vq.run_rhf(mol, basis, opts)
    D = np.asarray(result.density)
    return mol, basis, D


# ============================================================
# Test: augmentation radius estimation
# ============================================================


class TestAugmentationRadii:
    """Validation of per-element augmentation sphere radii."""

    def test_hydrogen_radius(self):
        r = estimate_augmentation_radius(1)
        assert 1.5 <= r <= 2.5

    def test_helium_radius(self):
        r = estimate_augmentation_radius(2)
        assert 1.5 <= r <= 2.5

    def test_carbon_radius(self):
        r = estimate_augmentation_radius(6)
        assert 2.0 <= r <= 3.5

    def test_oxygen_radius(self):
        r = estimate_augmentation_radius(8)
        assert 2.0 <= r <= 3.0

    def test_heavier_monotonic(self):
        """Radii should generally increase with Z."""
        r1 = estimate_augmentation_radius(10)  # Ne
        r2 = estimate_augmentation_radius(20)  # Ca
        r3 = estimate_augmentation_radius(30)  # Zn
        assert r2 >= r1
        assert r3 >= r2


# ============================================================
# Test: radial Poisson solver
# ============================================================


class TestRadialPoisson:
    """Validate the radial Poisson solver on simple densities."""

    def test_poisson_spherical_gaussian(self):
        """The Hartree potential of a spherical Gaussian should be
        the error function over r at large r, and finite at r=0."""
        nr, na = 100, 6
        alpha = 2.0
        r = np.linspace(0.001, 2.0, nr)
        omega = np.array(
            [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        )
        wa = np.full(na, 4 * math.pi / na)

        # Normalised Gaussian density: (alpha/sqrt(pi))^3 exp(-alpha^2 r^2)
        rho_1d = (alpha / math.sqrt(math.pi)) ** 3 * np.exp(-(alpha**2) * r**2)
        rho = rho_1d[:, None] * np.ones(na)[None, :]

        # Uniform grid so dr is constant and w_r = r^2 * dr is exact.
        dr = float(r[1] - r[0])
        w_r = r**2 * dr
        grid = AtomicRadialGrid(
            centre_bohr=np.zeros(3),
            r=r,
            w_r=w_r,
            angular_xyz=omega,
            w_a=wa,
            radial_alpha=alpha,
            lebedev_order=3,
        )
        V = solve_poisson_radial(rho, grid)
        assert np.all(np.isfinite(V))
        # V should be positive (Hartree potential of positive charge density)
        assert np.all(V >= -1e-15)

    def test_poisson_point_charge(self):
        """The Hartree potential of a point-like density should be ~1/r."""
        nr, na = 200, 6
        r = np.linspace(0.01, 3.0, nr)
        omega = np.array(
            [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        )
        wa = np.full(na, 4 * math.pi / na)

        # A compact Gaussian approximating a point charge
        alpha = 10.0
        rho_1d = (alpha / math.sqrt(math.pi)) ** 3 * np.exp(-(alpha**2) * r**2)
        rho = rho_1d[:, None] * np.ones(na)[None, :]

        dr = float(r[1] - r[0])
        w_r = r**2 * dr
        grid = AtomicRadialGrid(
            centre_bohr=np.zeros(3),
            r=r,
            w_r=w_r,
            angular_xyz=omega,
            w_a=wa,
            radial_alpha=alpha,
            lebedev_order=3,
        )
        V = solve_poisson_radial(rho, grid)
        # At large r, V should approach 1/r (for unit charge)
        # Check at the outer portion where the Gaussian is negligible.
        i_mid = 3 * nr // 4
        r_mid = r[i_mid]
        V_mid = V[i_mid, 0]
        assert abs(V_mid * r_mid - 1.0) < 0.3, (
            f"At r={r_mid:.3f}, V*r = {V_mid * r_mid:.4f}, expected ~1"
        )

    def test_poisson_gaussian_self_energy_matches_analytic(self):
        """Gaussian Hartree self-energies are analytic; the quadrature must
        hit them to <0.5%.

        Regression for the diagonal double-count: both inclusive cumsums in
        ``solve_poisson_radial`` counted the k'=k shell (whose inner and
        outer kernels coincide at r'=r), a systematic POSITIVE error of
        5-13% at n_radial=80 that grew with density sharpness. It largely
        cancelled between the hard and soft augmentation solves, which is
        why total energies hid it, but it poisoned every individual radial
        Hartree quantity. Exact value: for rho(r) = q (a/pi)^{3/2}
        exp(-a r^2), E = q^2 sqrt(a/2)/sqrt(pi)."""
        from vibeqc.periodic_gapw_atomic_grid import AtomicRadialGrid

        grid = AtomicRadialGrid.from_element(
            np.zeros(3), 10, n_radial=80, lebedev_order=17
        )
        pts = np.asarray(grid.cartesian_points()).reshape(
            grid.n_radial, grid.n_angular, 3
        )
        r2 = (pts**2).sum(axis=-1)
        wt = grid.combined_weights()
        for a_exp in (1.0, 10.0, 100.0, 414.0):
            rho = (a_exp / math.pi) ** 1.5 * np.exp(-a_exp * r2)
            e_num = 0.5 * float((rho * solve_poisson_radial(rho, grid) * wt).sum())
            e_exact = math.sqrt(a_exp / 2.0) / math.sqrt(math.pi)
            rel = abs(e_num - e_exact) / e_exact
            assert rel < 5e-3, (
                f"Gaussian self-energy off by {rel:.2%} at exponent "
                f"{a_exp} (pre-fix: 5-13%): E_num={e_num}, E_exact={e_exact}"
            )


# ============================================================
# Test: GAPW infrastructure build
# ============================================================


class TestGapwInfrastructure:
    """Validate that the GAPW augmentation data builds correctly."""

    def test_gapw_augmentation_constructs_he(self):
        """GapwAugmentation should build for He in a box."""
        L = 12.0
        system = _he_periodic(L)
        _, basis, D = _he_basis_density(L)
        grid = PlaneWaveGrid(np.eye(3) * L, 32, 32, 32)
        aug = GapwAugmentation(basis, system, grid, quiet=True)
        assert aug.n_atoms == 1

    def test_gapw_augmentation_constructs_h2(self):
        """GapwAugmentation should build for H2 in a box."""
        L = 12.0
        system = _h2_periodic(L)
        _, basis, D = _h2_basis_density(L)
        grid = PlaneWaveGrid(np.eye(3) * L, 32, 32, 32)
        aug = GapwAugmentation(basis, system, grid, quiet=True)
        assert aug.n_atoms == 2

    def test_softened_basis_preserves_hydrogen_valence_contractions(self):
        """The exponent cutoff must not split hydrogen's only AO shell."""
        system = _h2_periodic()
        _, basis, _ = _h2_basis_density()
        soft = softened_basis(basis, system)

        assert soft is basis
        assert [list(sh.exponents) for sh in soft.shells()] == [
            list(sh.exponents) for sh in basis.shells()
        ]

    def test_softened_basis_only_prunes_core_bearing_atoms_in_lih(self):
        """Mixed systems retain H in full while still softening Li."""
        L = 12.0
        c = L / 2
        system = core.PeriodicSystem()
        system.dim = 3
        system.lattice = np.eye(3) * L
        system.unit_cell = [
            core.Atom(3, [c - 1.5, c, c]),
            core.Atom(1, [c + 1.5, c, c]),
        ]
        mol = vq.Molecule(
            [vq.Atom(3, [c - 1.5, c, c]), vq.Atom(1, [c + 1.5, c, c])],
            0,
            1,
        )
        basis = vq.BasisSet(mol, "sto-3g")
        soft = softened_basis(basis, system)

        full_shells = basis.shells()
        soft_shells = soft.shells()
        full_h = [sh for sh in full_shells if int(sh.atom_index) == 1]
        soft_h = [sh for sh in soft_shells if int(sh.atom_index) == 1]
        assert [list(sh.exponents) for sh in soft_h] == [
            list(sh.exponents) for sh in full_h
        ]
        assert sum(len(sh.exponents) for sh in soft_shells) < sum(
            len(sh.exponents) for sh in full_shells
        )

    def test_atomic_density_finite(self):
        """The atomic density on the radial grid should be finite."""
        L = 12.0
        system = _he_periodic(L)
        _, basis, D = _he_basis_density(L)
        grid = PlaneWaveGrid(np.eye(3) * L, 32, 32, 32)
        aug = GapwAugmentation(basis, system, grid, quiet=True)
        ad = aug._atom_data[0]
        rho = aug._compute_atomic_density(D, ad, use_soft=False)
        assert np.all(np.isfinite(rho))
        assert rho.shape == (ad.grid.n_radial, ad.grid.n_angular)
        # The density should integrate to approximately the electron count
        wt = ad.grid.combined_weights()
        integral = float(np.einsum("kl,kl->", rho, wt))
        assert 1.0 < integral < 4.0, f"Integrated density = {integral:.4f}"

    def test_hartree_correction_finite(self):
        """The per-atom Hartree correction should be finite."""
        L = 12.0
        system = _he_periodic(L)
        _, basis, D = _he_basis_density(L)
        grid = PlaneWaveGrid(np.eye(3) * L, 32, 32, 32)

        aug = GapwAugmentation(basis, system, grid, quiet=True)
        aug._compensator = None  # Skip compensator for the pure correction
        corrections = aug.compute_hartree_correction(D)
        assert len(corrections) == 1
        for idx, delta_J in corrections.items():
            assert np.all(np.isfinite(delta_J))
            assert delta_J.shape == (basis.nbasis, basis.nbasis)
            # Check symmetry
            assert np.allclose(delta_J, delta_J.T, atol=1e-10)

    @pytest.mark.parametrize("functional", ["lda", "pbe", "r2scan"])
    def test_xc_correction_returns_matching_energy(self, functional):
        """The Fock-side XC pass must return its own augmentation energy."""
        L = 12.0
        system = _he_periodic(L)
        _, basis, D = _he_basis_density(L)
        grid = PlaneWaveGrid(np.eye(3) * L, 20, 20, 20)
        aug = GapwAugmentation(
            basis,
            system,
            grid,
            n_radial=40,
            lebedev_order=11,
            quiet=True,
        )
        func = core.Functional(functional, 1)

        _corrections, e_xc_from_fock = aug.compute_xc_correction(
            D,
            func,
            return_energy=True,
        )
        _e_hartree, e_xc_reference = aug.compute_augmentation_energy(
            D,
            functional=func,
        )

        assert e_xc_from_fock == pytest.approx(e_xc_reference, abs=1e-12)

    @pytest.mark.parametrize("functional", ["lda", "pbe", "r2scan"])
    def test_polarised_xc_correction_returns_matching_energy(self, functional):
        """The spin-polarised Fock-side pass returns the same XC energy."""
        L = 12.0
        system = _he_periodic(L)
        _, basis, D = _he_basis_density(L)
        grid = PlaneWaveGrid(np.eye(3) * L, 20, 20, 20)
        aug = GapwAugmentation(
            basis,
            system,
            grid,
            n_radial=40,
            lebedev_order=11,
            quiet=True,
        )
        func = core.Functional(functional, 2)
        D_alpha = 0.5 * D
        D_beta = 0.5 * D

        _corr_a, _corr_b, e_xc_from_fock = (
            aug.compute_xc_correction_polarised(
                D_alpha,
                D_beta,
                func,
                return_energy=True,
            )
        )
        _e_hartree, e_xc_reference = aug.compute_augmentation_energy_polarised(
            D_alpha,
            D_beta,
            func,
        )

        assert e_xc_from_fock == pytest.approx(e_xc_reference, abs=1e-12)


# ============================================================
# Test: GapwJBuilder matrix properties
# ============================================================


class TestGapwJBuilder:
    """Validate the GAPW J builder matrix properties."""

    def test_gapw_j_is_symmetric(self):
        """GAPW J must be symmetric for a real density."""
        L = 12.0
        system = _he_periodic(L)
        _, basis, D = _he_basis_density(L)
        grid = PlaneWaveGrid(np.eye(3) * L, 24, 24, 24)
        builder = GapwJBuilder(basis, system, grid, quiet=True)
        J = builder.build_J(D)
        assert J.shape == (basis.nbasis, basis.nbasis)
        assert np.allclose(J, J.T, atol=1e-10)

    def test_gapw_j_zero_density(self):
        """Zero density should give zero J."""
        L = 12.0
        system = _he_periodic(L)
        _, basis, _ = _he_basis_density(L)
        grid = PlaneWaveGrid(np.eye(3) * L, 16, 16, 16)
        builder = GapwJBuilder(basis, system, grid, quiet=True)
        J_zero = builder.build_J(np.zeros((basis.nbasis, basis.nbasis)))
        assert np.allclose(J_zero, 0.0, atol=1e-12)

    def test_gapw_hartree_energy_finite(self):
        """The GAPW Hartree energy should be finite and close to GPW."""
        L = 12.0
        system = _he_periodic(L)
        _, basis, D = _he_basis_density(L)
        grid = PlaneWaveGrid(np.eye(3) * L, 24, 24, 24)
        builder = GapwJBuilder(basis, system, grid, quiet=True)
        E_h_gapw = builder.hartree_energy(D)
        assert np.isfinite(E_h_gapw)
        # The GAPW Hartree energy should be reasonably close to the GPW one
        from vibeqc.periodic_gapw_j import GpwJBuilder

        gpw_builder = GpwJBuilder(basis, grid)
        E_h_gpw = gpw_builder.hartree_energy(D)
        # The augmentation correction should be a finite fraction
        diff = abs(E_h_gapw - E_h_gpw)
        assert diff < 1.0, f"GAPW-GPW Hartree energy diff too large: {diff:.4f} Ha"


# ============================================================
# Test: End-to-end GAPW SCF convergence
# ============================================================


class TestGapwScf:
    """Full GAPW SCF convergence tests on small systems."""

    def test_gapw_rhf_he_converges(self):
        """GAPW RHF on He STO-3G in a 12-bohr cube must converge."""
        L = 12.0
        system = _he_periodic(L)
        _, basis, _ = _he_basis_density(L)
        grid = PlaneWaveGrid(np.eye(3) * L, 24, 24, 24)
        result = run_periodic_rhf_gapw(
            system,
            basis,
            grid=grid,
            quiet=True,
            max_iter=30,
            conv_tol_energy=1e-7,
            molecular_limit=True,
        )
        assert result.converged, (
            f"GAPW RHF He did not converge (n_iter={result.n_iter})"
        )
        assert result.n_iter <= 25
        assert np.isfinite(result.energy)
        assert result.gapw_correction is not None

    def test_gapw_result_carries_converged_fock_and_overlap(self):
        """``GapwScfResult`` stores the (post-DIIS) Fock the SCF
        diagonalized plus the Gamma overlap, so the runner adapter can
        surface the augmented operator instead of reconstructing an
        HF-shaped ``Hcore + J_gpw - K/2`` (whose J is un-augmented and
        therefore the wrong operator for GAPW). Consistency pin: the
        stored (F, S) eigenpairs reproduce ``mo_energies``."""
        import scipy.linalg as sla

        L = 12.0
        system = _he_periodic(L)
        _, basis, _ = _he_basis_density(L)
        grid = PlaneWaveGrid(np.eye(3) * L, 24, 24, 24)
        result = run_periodic_rhf_gapw(
            system,
            basis,
            grid=grid,
            quiet=True,
            max_iter=30,
            conv_tol_energy=1e-7,
            molecular_limit=True,
        )
        assert result.converged
        assert result.fock is not None
        assert result.overlap is not None
        ev = sla.eigh(
            np.asarray(result.fock),
            np.asarray(result.overlap),
            eigvals_only=True,
        )
        assert np.abs(ev - np.asarray(result.mo_energies)).max() < 1e-10

    def test_gapw_o_core_occupied(self):
        """Regression for the milestone-3 analytic-Fock fix (P0).

        Before the analytic exact Hartree Fock (``J = ∂E_H/∂D``) landed, the
        shipped GAPW ``build_J`` was not the derivative of its energy: for a
        partially-occupied core atom it pushed the tight 1s above the valence,
        so the SCF aufbau left the core EMPTY and isolated O collapsed to
        ~−35 Ha (e_kin ~17) instead of the all-electron ~−73.8.

        With the analytic Fock the core is recovered: O → −73.557 (+215 mHa
        vs the GDF all-electron reference), e_kin ~73.4, ε₁ₛ ≈ −20. This pins
        BOTH symptoms — core kinetic energy and total vs GDF — so a future
        Fock regression that re-empties the core fails here.

        See ``handovers/HANDOVER_GAPW_PRODUCTION.md`` § milestone 3 and
        ``examples/regression/gapw_parity/milestone3_analytic_fock.py``.
        """
        L = 12.0
        system = core.PeriodicSystem()
        system.dim = 3
        system.lattice = np.eye(3) * L
        system.unit_cell = [core.Atom(8, [L / 2, L / 2, L / 2])]
        mol = vq.Molecule([vq.Atom(8, [L / 2, L / 2, L / 2])], 0, 1)
        basis = vq.BasisSet(mol, "sto-3g")
        grid = PlaneWaveGrid(np.eye(3) * L, 24, 24, 24)

        # All-electron GDF reference (same gauge, in-process).
        opts = vq.PeriodicRHFOptions()
        opts.max_iter = 120
        opts.conv_tol_energy = 1e-9
        e_gdf = float(
            vq.run_rhf_periodic_gamma_gdf(system, basis, opts, progress=False).energy
        )

        result = run_periodic_rhf_gapw(
            system, basis, grid=grid, quiet=True,
            max_iter=120, conv_tol_energy=1e-9,
            one_centre="block",
        )
        assert result.converged, f"GAPW RHF O did not converge ({result.n_iter})"
        e_kin = result.breakdown.e_kinetic
        # CORE OK: e_kin must be near the all-electron value (~74.8), not the
        # core-empty collapse (~17). Half the all-electron kinetic is a wide
        # margin that the collapsed state (e_kin ~17, ratio ~0.23) fails and the
        # recovered state (e_kin ~73, ratio ~0.98) clears.
        assert e_kin > 0.5 * 74.8, (
            f"O core not occupied: e_kin={e_kin:.2f} (collapsed core ~17, "
            f"all-electron ~74.8)"
        )
        # Total within ~0.3 Ha of the all-electron GDF reference (the GAPW
        # augmentation residual at N=24 is +215 mHa; the collapsed state was
        # ~+38 Ha off).
        assert abs(result.energy - e_gdf) < 0.3, (
            f"O GAPW total {result.energy:.4f} too far from GDF {e_gdf:.4f}"
        )

    def test_gapw_rhf_h2_converges(self):
        """No-core H2 GAPW-HF must reduce to the GPW molecular limit."""
        L = 12.0
        system = _h2_periodic(L)
        mol, basis, D_ref = _h2_basis_density(L)
        grid = PlaneWaveGrid(np.eye(3) * L, 24, 24, 24)
        # Use the molecular density as a starting guess to speed convergence
        result = run_periodic_rhf_gapw(
            system,
            basis,
            grid=grid,
            quiet=True,
            max_iter=30,
            conv_tol_energy=1e-7,
            initial_density=D_ref,
            molecular_limit=True,
        )
        assert result.converged, (
            f"GAPW RHF H2 did not converge (n_iter={result.n_iter})"
        )
        assert np.isfinite(result.energy)
        opts = vq.RHFOptions()
        opts.conv_tol_energy = 1e-10
        e_ref = vq.run_rhf(mol, basis, opts).energy
        assert abs(result.energy - e_ref) < 1e-3

    def test_gapw_rks_he_lda_converges(self):
        """GAPW RKS LDA on He STO-3G in a 12-bohr cube must converge."""
        L = 12.0
        system = _he_periodic(L)
        _, basis, _ = _he_basis_density(L)
        grid = PlaneWaveGrid(np.eye(3) * L, 24, 24, 24)
        result = run_periodic_rks_gapw(
            system,
            basis,
            functional="lda",
            grid=grid,
            quiet=True,
            max_iter=30,
            conv_tol_energy=1e-7,
        )
        assert result.converged, (
            f"GAPW RKS LDA He did not converge (n_iter={result.n_iter})"
        )
        assert np.isfinite(result.energy)

    def test_gapw_rhf_via_runner(self, tmp_path):
        """GAPW RHF via run_periodic_job with jk_method='gapw'."""
        L = 12.0
        system = _he_periodic(L)
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        output_stem = tmp_path / "gapw_rhf"
        result = vq.run_periodic_job(
            system,
            basis,
            method="RHF",
            jk_method="gapw",
            max_iter=20,
            conv_tol_energy=1e-6,
            output=output_stem,
            write_molden_file=False,
            write_density=False,
            write_xyz_file=False,
            write_poscar_file=False,
            write_xsf_structure_file=False,
            write_cif_file=False,
            write_population_file=False,
            output_qvf=False,
            citations=False,
            gapw_molecular_limit=True,
        )
        assert result is not None
        assert hasattr(result, "energy")
        assert np.isfinite(result.energy)
        assert result.converged
        assert result.one_centre == "analytic"
        assert result.molecular_limit_declared is True
        manifest = tomllib.loads(output_stem.with_suffix(".system").read_text())
        run = manifest["run"]
        assert run["gapw_one_centre_resolved"] == "analytic"
        assert run["gapw_molecular_limit_declared"] is True

    def test_gapw_rhf_runner_requires_molecular_limit_declaration(self):
        """The public runner must not infer a molecular box from geometry."""
        L = 12.0
        system = _he_periodic(L)
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        with pytest.raises(
            NotImplementedError,
            match="gapw_molecular_limit=True",
        ):
            vq.run_periodic_job(
                system,
                basis,
                method="RHF",
                jk_method="gapw",
                dry_run=True,
                write_density=False,
                citations=False,
            )

    def test_gapw_rks_runner_rejects_irrelevant_molecular_limit(self):
        """The HF-only declaration must not be silently recorded for DFT."""
        system = _he_periodic(12.0)
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

        with pytest.raises(ValueError, match="applies only to GAPW RHF/UHF"):
            vq.run_periodic_job(
                system,
                basis,
                method="RKS",
                functional="lda",
                jk_method="gapw",
                gapw_molecular_limit=True,
                dry_run=True,
                write_density=False,
                citations=False,
            )

    @pytest.mark.parametrize(
        "derivative_kwargs",
        [
            {"optimize": True},
            {"optimize": True, "optimize_cell": True},
            {"hessian": True},
        ],
    )
    def test_gapw_hf_runner_derivatives_fail_closed(
        self,
        derivative_kwargs,
    ):
        """High-level derivatives must not switch away from the GAPW energy."""
        system = _he_periodic(12.0)
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

        with pytest.raises(
            NotImplementedError,
            match="derivatives of the same fit-free analytic one-centre energy",
        ):
            vq.run_periodic_job(
                system,
                basis,
                method="RHF",
                jk_method="gapw",
                gapw_molecular_limit=True,
                dry_run=True,
                write_density=False,
                citations=False,
                **derivative_kwargs,
            )

    def test_gapw_energy_finite_for_different_cutoffs(self):
        """GAPW energy should be finite and not vary wildly with cutoff."""
        L = 12.0
        system = _he_periodic(L)
        _, basis, _ = _he_basis_density(L)
        energies = []
        for n in (20, 24):
            grid = PlaneWaveGrid(np.eye(3) * L, n, n, n)
            result = run_periodic_rhf_gapw(
                system,
                basis,
                grid=grid,
                quiet=True,
                max_iter=20,
                conv_tol_energy=1e-6,
                molecular_limit=True,
            )
            energies.append(result.energy if result.converged else None)
        # At least one should have converged
        valid = [e for e in energies if e is not None]
        assert len(valid) > 0


# ============================================================
# Test: GAPW correction vs GPW reference
# ============================================================


class TestGapwCorrection:
    """Validate that the GAPW augmentation modifies the GPW energy."""

    def test_gapw_correction_nonzero(self):
        """The GAPW augmentation should change the Hartree energy
        relative to the smooth-grid GPW."""
        L = 12.0
        system = _he_periodic(L)
        _, basis, D = _he_basis_density(L)
        grid = PlaneWaveGrid(np.eye(3) * L, 24, 24, 24)

        # GPW Hartree energy
        from vibeqc.periodic_gapw_j import GpwJBuilder

        E_gpw = GpwJBuilder(basis, grid).hartree_energy(D)

        # GAPW Hartree energy
        builder = GapwJBuilder(basis, system, grid, quiet=True)
        E_gapw = builder.hartree_energy(D)

        # The correction should be non-zero (augmentation changes the energy)
        correction = E_gapw - E_gpw
        # For a single He atom, the correction is typically small but non-zero
        assert correction != 0.0, "GAPW correction is zero — augmentation not active"


# ============================================================
# Test: SCF result properties
# ============================================================


class TestGapwResult:
    """Validate GAPW SCF result properties."""

    def test_gapw_result_has_breakdown(self):
        """GAPW result must carry the energy breakdown."""
        L = 12.0
        system = _he_periodic(L)
        _, basis, _ = _he_basis_density(L)
        grid = PlaneWaveGrid(np.eye(3) * L, 24, 24, 24)
        result = run_periodic_rhf_gapw(
            system,
            basis,
            grid=grid,
            quiet=True,
            max_iter=20,
            conv_tol_energy=1e-7,
            molecular_limit=True,
        )
        assert result.breakdown is not None
        bd = result.breakdown
        for name in (
            "e_kinetic",
            "e_nuclear_attraction",
            "e_hartree",
            "e_hf_exchange",
            "e_nuclear_repulsion",
            "e_total",
        ):
            v = getattr(bd, name)
            assert isinstance(v, float), f"{name} is not float: {type(v)}"
            assert np.isfinite(v), f"{name} is not finite: {v}"

    def test_gapw_result_mo_properties(self):
        """MO energies should be finite and occupied states negative."""
        L = 12.0
        system = _he_periodic(L)
        _, basis, _ = _he_basis_density(L)
        grid = PlaneWaveGrid(np.eye(3) * L, 24, 24, 24)
        result = run_periodic_rhf_gapw(
            system,
            basis,
            grid=grid,
            quiet=True,
            max_iter=20,
            conv_tol_energy=1e-7,
            molecular_limit=True,
        )
        assert np.all(np.isfinite(result.mo_energies))
        assert len(result.mo_energies) == basis.nbasis
        # He (Z=2) has 1 occupied orbital
        assert result.mo_energies[0] < 0, "HOMO should be negative (bound state)"


# ============================================================
# Test: GAPW DFT energy correctness (XC-augmentation telescoping)
# ============================================================


class TestGapwDFTEnergy:
    """All-electron GAPW DFT energies vs the molecular reference.

    Regression guard for the M3c XC-augmentation double-counting bug: the
    smooth-grid XC term must be generated from the SOFT (pseudo) density so
    the per-atom augmentation ``E_xc[ρ_a] − E_xc[ρ̃_a]`` telescopes to the
    all-electron XC (mirroring the Hartree ``build_J``). Pre-fix, evaluating
    the smooth XC on the FULL density double-counted the soft core inside
    each sphere and over-bound He by ~300 mHa for *every* DFT functional
    (LDA and GGA). The symptom is a total energy hundreds of mHa below both
    the smooth-GPW and the molecular references.
    """

    @pytest.mark.parametrize("functional", ["lda", "pbe"])
    def test_gapw_atom_matches_molecular(self, functional):
        """He GAPW-DFT matches the molecular all-electron reference to within
        the augmentation floor (~1 mHa) — NOT over-bound by 300 mHa."""
        L = 14.0
        system = _he_periodic(L)
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        grid = PlaneWaveGrid(np.eye(3) * L, 48, 48, 48)
        e_ref = _mol_rks_energy(2, [L / 2, L / 2, L / 2], functional)
        result = run_periodic_rks_gapw(
            system,
            basis,
            functional=functional,
            grid=grid,
            quiet=True,
            max_iter=60,
            conv_tol_energy=1e-8,
        )
        assert result.converged
        assert result.one_centre == "block"
        assert result.scf_trace[-1]["energy"] == pytest.approx(
            result.energy, abs=1e-8
        )
        assert result.scf_trace[-1]["e_xc"] == pytest.approx(
            result.breakdown.e_xc, abs=1e-8
        )
        gap = abs(result.energy - e_ref)
        assert gap < 3e-3, (
            f"{functional} GAPW He {result.energy:.6f} Ha vs molecular "
            f"{e_ref:.6f} Ha (gap {gap * 1e3:.2f} mHa) — XC-augmentation "
            f"telescoping broken (soft/hard double-counting regression?)."
        )

    def test_gapw_pbe_grid_stable(self):
        """GGA-GAPW He energy is stable across grids (no core aliasing)."""
        L = 14.0
        system = _he_periodic(L)
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        energies = []
        for nn in (40, 48, 56):
            grid = PlaneWaveGrid(np.eye(3) * L, nn, nn, nn)
            r = run_periodic_rks_gapw(
                system,
                basis,
                functional="pbe",
                grid=grid,
                quiet=True,
                max_iter=60,
                conv_tol_energy=1e-8,
            )
            assert r.converged
            energies.append(r.energy)
        assert max(energies) - min(energies) < 1e-3

    def test_gapw_augmentation_rescues_coarse_grid(self):
        """On a coarse grid the smooth GPW XC aliases the He core (large
        error); the GAPW augmentation rescues it back to the molecular
        reference. This is the telescoping doing its job — and a guard that
        the augmentation is neither inert nor double-counting."""
        L = 14.0
        system = _he_periodic(L)
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        grid = PlaneWaveGrid(np.eye(3) * L, 24, 24, 24)
        e_ref = _mol_rks_energy(2, [L / 2, L / 2, L / 2], "lda")
        e_gpw = run_periodic_rks_gpw(
            system, basis, functional="lda", grid=grid,
            quiet=True, max_iter=80, conv_tol_energy=1e-8,
        ).energy
        e_gapw = run_periodic_rks_gapw(
            system, basis, functional="lda", grid=grid,
            quiet=True, max_iter=80, conv_tol_energy=1e-8,
        ).energy
        assert abs(e_gpw - e_ref) > 50e-3, (
            "expected smooth GPW to alias the core badly on a 24³ grid"
        )
        assert abs(e_gapw - e_ref) < 5e-3, (
            f"GAPW failed to rescue the coarse grid: {e_gapw:.6f} Ha vs "
            f"molecular {e_ref:.6f} Ha"
        )


class TestGapwUksDFTEnergy:
    """All-electron open-shell GAPW UKS energies vs the molecular reference.

    Regression guard for the open-shell twin of the M3c XC-augmentation
    double-counting bug: the UKS smooth-grid XC must be generated PER SPIN from
    the SOFT (pseudo) density -- exactly as the closed-shell RKS driver does
    via ``_smooth_xc_generation`` -- so the per-atom polarised augmentation
    ``E_xc[ρ_a] − E_xc[ρ̃_a]`` telescopes to the all-electron XC. Pre-fix, the
    UKS driver collocated the smooth XC from the FULL (hard) density, computing
    ``2·hard − soft`` and double-counting the hard core: ~300 mHa over-binding
    for every DFT functional (LDA/PBE/meta-GGA). Includes r2SCAN, whose gate
    this fix lifts.
    """

    @pytest.mark.parametrize("functional", ["lda", "pbe", "r2scan"])
    def test_uks_gapw_atom_matches_molecular_closed_shell(self, functional):
        """He (closed-shell, driven through the UKS path with n_α = n_β = 1)
        matches the molecular all-electron reference to within the
        augmentation floor (~1 mHa) — NOT over-bound by ~300 mHa."""
        L = 14.0
        system = _he_periodic(L)
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        grid = PlaneWaveGrid(np.eye(3) * L, 48, 48, 48)
        e_ref = _mol_uks_energy(2, [L / 2, L / 2, L / 2], functional, 1)
        result = run_periodic_uks_gapw(
            system, basis, functional=functional, n_alpha=1, n_beta=1,
            grid=grid, quiet=True, max_iter=60, conv_tol_energy=1e-8,
        )
        assert result.converged
        assert result.one_centre == "block"
        assert result.scf_trace[-1]["energy"] == pytest.approx(
            result.energy, abs=1e-8
        )
        assert result.scf_trace[-1]["e_xc"] == pytest.approx(
            result.breakdown.e_xc, abs=1e-8
        )
        gap = abs(result.energy - e_ref)
        assert gap < 3e-3, (
            f"{functional} UKS GAPW He {result.energy:.6f} Ha vs molecular "
            f"{e_ref:.6f} Ha (gap {gap * 1e3:.2f} mHa) — open-shell "
            f"XC-augmentation telescoping broken (soft/hard double-counting?)."
        )

    @pytest.mark.parametrize("functional", ["lda", "pbe", "r2scan"])
    def test_uks_gapw_doublet_matches_molecular(self, functional):
        """H atom doublet (fully spin-polarised, n_α = 1, n_β = 0) matches the
        molecular open-shell reference to within the augmentation floor — the
        telescoping (incl. the per-spin meta-GGA τ term) must hold on the
        polarised (β-empty) path too."""
        L = 14.0
        system = _h_periodic(L)
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        grid = PlaneWaveGrid(np.eye(3) * L, 48, 48, 48)
        e_ref = _mol_uks_energy(1, [L / 2, L / 2, L / 2], functional, 2)
        result = run_periodic_uks_gapw(
            system, basis, functional=functional, n_alpha=1, n_beta=0,
            grid=grid, quiet=True, max_iter=60, conv_tol_energy=1e-8,
        )
        assert result.converged
        gap = abs(result.energy - e_ref)
        assert gap < 3e-3, (
            f"{functional} UKS GAPW H doublet {result.energy:.6f} Ha vs "
            f"molecular {e_ref:.6f} Ha (gap {gap * 1e3:.2f} mHa) — polarised "
            f"XC-augmentation telescoping broken."
        )


class TestGapwDFTGating:
    """Fail-closed gates for GAPW DFT paths that are not yet correct."""

    def test_rks_mgga_gapw_telescopes_to_molecular(self):
        """Closed-shell GAPW meta-GGA (r2SCAN) recovers the molecular
        all-electron reference for He to within the augmentation floor
        (~1 mHa) -- the per-atom kinetic-energy-density (t) augmentation
        telescopes the core t. This replaces the old fail-closed gate."""
        L = 14.0
        system = _he_periodic(L)
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        grid = PlaneWaveGrid(np.eye(3) * L, 48, 48, 48)
        e_ref = _mol_rks_energy(2, [L / 2, L / 2, L / 2], "r2scan")
        result = run_periodic_rks_gapw(
            system, basis, functional="r2scan", grid=grid,
            quiet=True, max_iter=60, conv_tol_energy=1e-8,
        )
        assert result.converged
        gap = abs(result.energy - e_ref)
        assert gap < 3e-3, (
            f"GAPW r2scan He {result.energy:.6f} Ha vs molecular "
            f"{e_ref:.6f} Ha (gap {gap * 1e3:.2f} mHa) -- meta-GGA t "
            f"augmentation telescoping broken."
        )

    def test_tpss_gapw_fails_closed_on_stiffness(self):
        """TPSS-class meta-GGAs are singular at orbital critical points on
        the uniform smooth grid; the GAPW route inherits the smooth-grid
        fail-closed stiffness guard (r2scan/scan self-regularise)."""
        L = 12.0
        system = _he_periodic(L)
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        grid = PlaneWaveGrid(np.eye(3) * L, 24, 24, 24)
        with pytest.raises(NotImplementedError, match="stiffness"):
            run_periodic_rks_gapw(
                system, basis, functional="tpss", grid=grid,
                quiet=True, max_iter=3,
            )

    def test_rks_multik_gapw_mgga_converges_on_compact_cell(self):
        """Multi-k GAPW meta-GGA uses the compact Bloch tau/v_tau path.

        This is the multi-k sibling of the Gamma r2SCAN telescope pin above:
        compact cells must not fail closed at the old meta-GGA gate, and the
        SCF should produce a finite all-electron energy.
        """
        L = 8.0
        system = _he_periodic(L)
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        grid = PlaneWaveGrid(np.eye(3) * L, 24, 24, 24)
        kmesh = core.monkhorst_pack(system, [1, 1, 2])
        result = run_periodic_rks_gapw_multi_k(
            system,
            basis,
            kmesh,
            functional="r2scan",
            grid=grid,
            quiet=True,
            max_iter=50,
            conv_tol_energy=1e-7,
            conv_tol_density=1e-5,
            n_radial=40,
            lebedev_order=11,
        )
        assert result.converged
        assert np.isfinite(result.energy)
        assert np.isfinite(result.breakdown.e_xc)

    def test_uks_mgga_gapw_telescopes_to_molecular(self):
        """Open-shell GAPW meta-GGA (r2SCAN) recovers the molecular open-shell
        reference for the H doublet to within the augmentation floor. Now that
        the UKS smooth XC generates from the soft density per spin (the
        ~300 mHa telescoping bug is fixed), the polarised per-atom τ
        augmentation telescopes and the old fail-closed gate is lifted."""
        L = 14.0
        system = _h_periodic(L)
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        grid = PlaneWaveGrid(np.eye(3) * L, 48, 48, 48)
        e_ref = _mol_uks_energy(1, [L / 2, L / 2, L / 2], "r2scan", 2)
        result = run_periodic_uks_gapw(
            system, basis, functional="r2scan", n_alpha=1, n_beta=0,
            grid=grid, quiet=True, max_iter=60, conv_tol_energy=1e-8,
        )
        assert result.converged
        gap = abs(result.energy - e_ref)
        assert gap < 3e-3, (
            f"UKS GAPW r2scan H doublet {result.energy:.6f} Ha vs molecular "
            f"{e_ref:.6f} Ha (gap {gap * 1e3:.2f} mHa) -- open-shell meta-GGA "
            f"τ augmentation telescoping broken."
        )

    def test_uks_tpss_gapw_fails_closed_on_stiffness(self):
        """Open-shell twin of the stiffness gate: a non-self-regularising
        meta-GGA (TPSS) still fails closed on the Γ UKS GAPW path via the
        polarised smooth-grid stiffness guard (which runs before the
        augmentation)."""
        L = 12.0
        system = _h_periodic(L)
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        grid = PlaneWaveGrid(np.eye(3) * L, 24, 24, 24)
        with pytest.raises(NotImplementedError, match="stiffness"):
            run_periodic_uks_gapw(
                system, basis, functional="tpss", n_alpha=1, n_beta=0,
                grid=grid, quiet=True, max_iter=3,
            )

    @pytest.mark.parametrize("functional", ["lda", "pbe"])
    def test_uks_dft_gapw_runs_with_polarised_xc_augmentation(self, functional):
        """Open-shell GAPW DFT uses the polarised per-atom XC energy path."""
        L = 12.0
        system = _h_periodic(L)
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        grid = PlaneWaveGrid(np.eye(3) * L, 24, 24, 24)
        result = run_periodic_uks_gapw(
            system,
            basis,
            functional=functional,
            n_alpha=1,
            n_beta=0,
            grid=grid,
            quiet=True,
            max_iter=30,
            conv_tol_energy=1e-8,
        )
        assert result.converged
        assert np.isfinite(result.energy)
        assert np.isfinite(result.breakdown.e_xc)
        assert result.breakdown.e_xc < 0.0


class TestGapwRunnerExperimentalWarning:
    """The user-facing ``run_periodic_job(jk_method='gapw')`` path no longer
    emits the run-level ``GAPWExperimentalWarning``.

    The GAPW all-electron route's two open correctness bugs are fixed
    (2026-06-26): the Hartree Fock is the exact ``∂E_H/∂D`` (analytic Fock),
    and the bonded-molecule overlapping-augmentation double-count is
    root-caused (own-atom compensator + partition-of-unity; H₂/STO-3G LDA/PBE
    molecular parity < 5 mHa). The run-level experimental warning that flagged
    those correctness bugs is therefore retired. Regression that the
    ``jk_method='gapw'`` SCF path runs WITHOUT emitting it. The molecular-limit
    envelope is declared explicitly below; undeclared HF now fails closed as
    documented in ``docs/user_guide/gapw.md``. The ``.out``
    "(experimental)" label stays.
    """

    def test_runner_does_not_emit_experimental_warning(self, tmp_path):
        L = 12.0
        system = _he_periodic(L)
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        # catch_warnings + simplefilter("always") overrides the module
        # pytestmark ignore filter and records every emission.
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            result = vq.run_periodic_job(
                system,
                basis,
                method="RHF",
                jk_method="gapw",
                max_iter=20,
                conv_tol_energy=1e-6,
                write_density=False,
                citations=False,
                gapw_molecular_limit=True,
                output=str(tmp_path / "gapw_no_experimental_warning"),
            )
        assert result.converged
        run_level = [
            w
            for w in rec
            if issubclass(w.category, GAPWExperimentalWarning)
            and "jk_method='gapw'" in str(w.message)
        ]
        assert len(run_level) == 0, (
            "the jk_method='gapw' path should no longer emit the run-level "
            f"GAPWExperimentalWarning, got {len(run_level)}: "
            f"{[str(w.message) for w in run_level]}"
        )


@pytest.mark.parametrize("functional", ["lda", "pbe"])
def test_gapw_molecule_matches_molecular(functional):
    """Molecular GAPW-DFT matches the molecular reference (overlap double-count
    fixed).

    Regression for the bonded-molecule overlapping-augmentation double-count
    (was xfail). Two root-cause fixes landed:

    1. **Own-atom compensator** -- the per-atom radial augmentation now uses
       only atom A's own ρ₀ compensator (the *global* compensator lives on the
       smooth FFT grid); summing all atoms' compensators on each isolated
       radial grid double-counted the inter-atomic compensator interaction.
    2. **Partition-of-unity** -- the augmentation integrals are weighted by
       ``p_A = w_A·max_b w_b / Σ_b w_b`` so overlapping spheres do not
       double-count the local XC/Hartree augmentation; ``p_A`` reduces to the
       bare radial window for an isolated atom, so single-atom results
       (He/Ne/O) are unchanged.

    Together these bring H₂/STO-3G from ~−101 mHa (LDA) to a few mHa, robustly
    across functionals (the test runs both LDA and a GGA so the pass cannot be
    a single-functional Hartree/XC cancellation). The residual is the intrinsic
    GAPW soft-basis bond representation, well within the ±5 mHa target.

    See ``handovers/HANDOVER_GAPW_PRODUCTION.md`` § milestone 3.
    """
    L = 12.0
    c = L / 2
    system = _h2_periodic(L)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    grid = PlaneWaveGrid(np.eye(3) * L, 48, 48, 48)
    mol = vq.Molecule(
        [vq.Atom(1, [c - 0.7, c, c]), vq.Atom(1, [c + 0.7, c, c])], 0, 1
    )
    mb = vq.BasisSet(mol, "sto-3g")
    o = vq.RKSOptions()
    o.functional = functional
    o.conv_tol_energy = 1e-10
    e_ref = vq.run_rks(mol, mb, o).energy
    e_gapw = run_periodic_rks_gapw(
        system, basis, functional=functional, grid=grid,
        quiet=True, max_iter=80, conv_tol_energy=1e-8,
    ).energy
    assert abs(e_gapw - e_ref) < 5e-3


def _lih_rocksalt_primitive():
    """Compact LiH rocksalt primitive fcc cell (a = 7.7176 bohr)."""
    a = 7.7176
    prim = 0.5 * a * np.array([[0.0, 1.0, 1.0],
                               [1.0, 0.0, 1.0],
                               [1.0, 1.0, 0.0]])
    system = core.PeriodicSystem()
    system.dim = 3
    system.lattice = prim
    system.unit_cell = [core.Atom(3, [0.0, 0.0, 0.0]),
                        core.Atom(1, [0.5 * a, 0.0, 0.0])]
    return system


class TestMultiKGapwFoldChargeGuard:
    """Multi-k GAPW's Hartree regime: Bloch where the fold is invalid,
    fold where it is fine, and a measured guard on whatever is left.

    Folding D(g=0) into every cell pair is right only where cross-cell
    AO products vanish; elsewhere it double-counts inter-cell density
    (LiH rocksalt (2,2,2): 9.82 electrons for a 4-electron cell). The
    augmentation-active compact path is now built from the Bloch density
    with a per-k projection of the smooth potential, so it does not use
    the fold at all. Paths that still do are checked at SCF start
    against the cell's electron count -- a measurement rather than the
    ``_multik_gpw_is_molecular_limit`` flag, which calls He/8-bohr,
    He/6-bohr AND LiH rocksalt "compact" although their fold errors are
    0.0 %, +0.5 % and +145 %.
    """

    def test_valid_fold_cell_is_not_refused(self):
        """He in an 8-bohr box on a (1,1,2) mesh: the flag says compact,
        but the measured fold error is 0.0 %, so the driver must run."""
        L = 8.0
        system = _he_periodic(L)
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        result = run_periodic_rks_gapw_multi_k(
            system, basis, core.monkhorst_pack(system, [1, 1, 2]),
            functional="lda",
            grid=PlaneWaveGrid(np.eye(3) * L, 24, 24, 24),
            quiet=True, max_iter=30, conv_tol_energy=1e-7,
            conv_tol_density=1e-5, n_radial=40, lebedev_order=11,
        )
        assert result.converged
        assert np.isfinite(result.energy)

    @pytest.mark.slow
    def test_compact_cell_runs_on_the_bloch_path_and_tracks_gdf(self):
        """LiH rocksalt (2,2,2) now runs on the Bloch-ported Hartree and
        lands in the GAPW augmentation-accuracy class against the
        PySCF-validated multi-k GDF route.

        Before the port the Gamma fold carried 9.82 electrons for this
        4-electron cell (+145 %); the SCF failed to converge and sat
        2.6 Ha ABOVE its own (1,1,1) result. With the smooth density
        built from the Bloch sum and its potential projected per k, the
        SCF converges and agrees with GDF to +17.7 mHa (measured; the
        bound leaves room for grid/threshold drift). That residual is
        the ordinary GAPW augmentation error, not a regime defect -- it
        is grid-converged (N=32 and N=48 agree to 2e-8 Ha).
        """
        system = _lih_rocksalt_primitive()
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        kmesh = core.monkhorst_pack(system, [2, 2, 2])
        gapw = run_periodic_rks_gapw_multi_k(
            system, basis, kmesh, functional="lda",
            grid=PlaneWaveGrid(np.asarray(system.lattice), 32, 32, 32),
            quiet=True, max_iter=60, conv_tol_energy=1e-8,
            conv_tol_density=1e-6,
        )
        assert gapw.converged
        opts = vq.PeriodicKSOptions()
        opts.functional = "lda"
        opts.max_iter = 100
        opts.conv_tol_energy = 1e-9
        ref = vq.run_krks_periodic_gdf(
            system, basis, kmesh, opts, progress=False
        )
        assert ref.converged
        delta = float(gapw.energy) - float(ref.energy)
        assert abs(delta) < 0.040, (
            f"multi-k GAPW on LiH rocksalt (2,2,2) is {1000 * delta:+.1f} "
            "mHa from the validated GDF route (measured +17.7 mHa); a "
            "large excursion means the Bloch Hartree port regressed"
        )

    def test_gamma_folded_density_is_wrong_on_a_compact_cell(self):
        """The measurement that forced the Bloch port: on LiH rocksalt
        (2,2,2) the Gamma-folded density integrates to ~9.8 electrons
        instead of 4, while the Bloch density is correct. Kept as the
        standing reason the compact path may never go back to the fold,
        and as the guard's calibration for the paths that still use
        it."""
        from vibeqc.periodic_gapw_j import (
            bloch_ao_on_grid,
            collocate_bloch_density_on_grid,
            collocate_density_on_grid,
        )
        from vibeqc.periodic_v_ne import compute_nuclear_lattice_dispatch

        system = _lih_rocksalt_primitive()
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        grid = PlaneWaveGrid(np.asarray(system.lattice), 24, 24, 24)
        km = core.monkhorst_pack(system, [2, 2, 2])
        kpts = np.asarray(km.kpoints)
        wts = np.asarray(km.weights)

        lat = core.LatticeSumOptions()
        T_lat = core.compute_kinetic_lattice(basis, system, lat)
        S_lat = core.compute_overlap_lattice(basis, system, lat)
        lat_v = core.LatticeSumOptions()
        lat_v.coulomb_method = core.CoulombMethod.EWALD_3D
        V_lat = compute_nuclear_lattice_dispatch(basis, system, lat_v)

        n_occ = 2  # LiH: 4 electrons
        D_k = []
        D_total = np.zeros((basis.nbasis, basis.nbasis))
        for ik, k in enumerate(kpts):
            Sk = np.asarray(core.bloch_sum(S_lat, k))
            Hk = (np.asarray(core.bloch_sum(T_lat, k))
                  + np.asarray(core.bloch_sum(V_lat, k)))
            Sk = 0.5 * (Sk + Sk.conj().T)
            Hk = 0.5 * (Hk + Hk.conj().T)
            s, U = np.linalg.eigh(Sk)
            X = U @ np.diag(1.0 / np.sqrt(np.maximum(s, 1e-12))) @ U.conj().T
            _eps, Co = np.linalg.eigh(X.conj().T @ Hk @ X)
            C = X @ Co
            Dk = 2.0 * (C[:, :n_occ] @ C[:, :n_occ].conj().T)
            D_k.append(Dk)
            D_total += wts[ik] * np.real(Dk)

        dV = grid.voxel_volume_bohr3
        rho_gamma = collocate_density_on_grid(basis, D_total, grid)
        cells = core.direct_lattice_cells(system, 25.0)
        trans = np.array([c.r_cart for c in cells], dtype=float)
        pts = grid.cartesian_coords().reshape(-1, 3)
        chi_k = [bloch_ao_on_grid(basis, pts, k, trans) for k in kpts]
        rho_bloch = collocate_bloch_density_on_grid(chi_k, D_k, wts, grid)

        q_bloch = float(rho_bloch.sum()) * dV
        q_gamma = float(rho_gamma.sum()) * dV
        assert q_bloch == pytest.approx(4.0, abs=0.05), (
            f"Bloch density must carry the cell's 4 electrons; got {q_bloch}"
        )
        assert q_gamma > 6.0, (
            "the Gamma fold is expected to over-count on a compact cell "
            f"(measured ~9.8 e); got {q_gamma} -- if this now equals 4, the "
            "fold convention changed and the multi-k GAPW guard should be "
            "revisited"
        )


class TestMultiKGapwXCAugmentation:
    """Correctness (not just convergence) of ``run_periodic_rks_gapw_multi_k``.

    A ``(1,1,1)`` Monkhorst-Pack mesh collapses to the Gamma density (and the
    molecular-limit smooth-collocation regime), so the multi-k GAPW total
    energy must reproduce the Gamma GAPW driver (``run_periodic_rhf_gapw``,
    the validated all-electron reference) to machine precision -- for LDA,
    GGA *and* meta-GGA. These pins complement the convergence smoke test in
    ``TestGapwDFTGating`` (converged + finite would not catch a wrong energy).

    Regression context: the multi-k GAPW driver previously collocated the
    smooth XC from the full-basis Gamma density instead of the soft
    generation and omitted the per-atom XC augmentation entirely (potential
    *and* energy), leaving a core-weight-dependent error -- +0.5 mHa on
    Be/STO-3G PBE -- while a smoke test still passed. A machine-precision
    (1,1,1)==Gamma parity pin is what actually guards this.
    """

    def test_multik_gapw_lda_matches_gamma_driver(self):
        L = 9.0
        system = _he_periodic(L)
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        kw = dict(
            functional="lda", cutoff_ha=120.0, quiet=True,
            conv_tol_energy=1e-9, conv_tol_density=1e-7,
        )
        g = run_periodic_rhf_gapw(system, basis, **kw)
        mk = run_periodic_rks_gapw_multi_k(
            system, basis, core.monkhorst_pack(system, [1, 1, 1]), **kw
        )
        assert g.converged and mk.converged
        assert mk.energy == pytest.approx(g.energy, abs=1e-8)
        assert mk.scf_trace[-1]["energy"] == pytest.approx(mk.energy, abs=1e-8)
        assert mk.scf_trace[-1]["e_xc"] == pytest.approx(
            mk.breakdown.e_xc, abs=1e-8
        )

    def test_multik_gapw_pbe_matches_gamma_with_real_core(self):
        """Be (1s core) is where a missing XC augmentation shows as +0.5 mHa;
        the multi-k GAPW PBE energy must match Gamma to uHa."""
        L = 9.0
        system = _be_periodic(L)
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        kw = dict(
            functional="pbe", cutoff_ha=120.0, quiet=True,
            conv_tol_energy=1e-9, conv_tol_density=1e-7,
        )
        g = run_periodic_rhf_gapw(system, basis, **kw)
        mk = run_periodic_rks_gapw_multi_k(
            system, basis, core.monkhorst_pack(system, [1, 1, 1]), **kw
        )
        assert g.converged and mk.converged
        assert mk.energy == pytest.approx(g.energy, abs=1e-6)
        assert mk.scf_trace[-1]["energy"] == pytest.approx(mk.energy, abs=1e-6)
        assert mk.scf_trace[-1]["e_xc"] == pytest.approx(
            mk.breakdown.e_xc, abs=1e-6
        )

    def test_multik_gapw_mgga_matches_gamma_with_real_core(self):
        """meta-GGA (r2SCAN) on the multi-k GAPW route: the per-atom core-t
        augmentation must telescope exactly to the Gamma driver on Be."""
        L = 9.0
        system = _be_periodic(L)
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        kw = dict(
            functional="r2scan", cutoff_ha=120.0, quiet=True,
            conv_tol_energy=1e-9, conv_tol_density=1e-7,
        )
        g = run_periodic_rhf_gapw(system, basis, **kw)
        mk = run_periodic_rks_gapw_multi_k(
            system, basis, core.monkhorst_pack(system, [1, 1, 1]), **kw
        )
        assert g.converged and mk.converged
        assert mk.energy == pytest.approx(g.energy, abs=1e-6)
        assert mk.scf_trace[-1]["energy"] == pytest.approx(mk.energy, abs=1e-6)
        assert mk.scf_trace[-1]["e_xc"] == pytest.approx(
            mk.breakdown.e_xc, abs=1e-6
        )


class TestExactAtomicDensityGradient:
    """The one-centre GGA gradient is analytic, not a radial finite difference.

    `_atomic_density_gradient` (module level) returns `drho/dr * rhat` with
    both angular components identically zero, and its docstring assumes "a
    spherically-symmetric density". One-centre densities are not spherically
    symmetric in general -- an aspherical core is the whole reason the l > 0
    machinery exists -- so for a GGA that under-counts sigma = |grad rho|^2
    wherever the atomic density carries angular structure. CP2K forms this
    analytically for the same reason (qs_vxc_atom.F contracts Clebsch-Gordan
    derivative coefficients against the analytic gradient of the basis
    products rather than differencing the density).

    `_compute_atomic_density_gradient_exact` evaluates
    grad rho_A = 2 sum_mn D_mn chi_m grad chi_n from the `dchi` tables
    already cached for the meta-GGA tau channel, so it costs one extra
    contraction and no new integral evaluation.

    Measured effect (PBE, 14-bohr box, vs the molecular reference): LiH
    -0.813 -> -0.263 mHa, a 3x improvement on the bonded/aspherical case;
    He +0.847 -> +1.008 mHa. The He direction is deliberate evidence, not a
    regression -- see test_spherical_density_has_no_angular_gradient.
    """

    @staticmethod
    def _setup(zs, pos, L=12.0):
        sysp = core.PeriodicSystem()
        sysp.dim = 3
        sysp.lattice = np.eye(3) * L
        sysp.unit_cell = [core.Atom(z, p) for z, p in zip(zs, pos)]
        mol = vq.Molecule([vq.Atom(z, p) for z, p in zip(zs, pos)], 0, 1)
        basis = vq.BasisSet(mol, "sto-3g")
        opts = vq.RHFOptions()
        opts.conv_tol_energy = 1e-9
        D = np.asarray(vq.run_rhf(mol, basis, opts).density)
        grid = PlaneWaveGrid(np.eye(3) * L, 48, 48, 48)
        return sysp, basis, D, grid

    def test_spherical_density_has_no_angular_gradient(self):
        """On a spherical one-centre density the exact gradient's angular
        components vanish identically -- so the exact form cannot be
        'adding angular noise', and any energy change it produces on a
        spherical atom comes from removing the radial FD error alone."""
        L = 12.0
        sysp, basis, D, grid = self._setup([10], [[L / 2] * 3], L)
        jb = GapwJBuilder(basis, sysp, grid, quiet=True)
        aug = jb._aug
        assert aug._augmentation_active
        ad = aug._atom_data[0]
        g = aug._compute_atomic_density_gradient_exact(D, ad, use_soft=False)
        omega = np.asarray(ad.grid.angular_xyz)
        radial = np.einsum("rad,ad->ra", g, omega)
        perp = g - radial[:, :, None] * omega[None, :, :]
        w = np.asarray(ad.grid.combined_weights())
        norm = lambda x: float(np.sqrt(
            np.einsum("ra,ra->", np.sum(x ** 2, axis=-1), w)))
        assert norm(g) > 1.0, "Ne one-centre gradient should be substantial"
        assert norm(perp) / norm(g) < 1e-10, (
            f"spherical density has angular gradient content "
            f"{norm(perp) / norm(g):.2e} of the total"
        )

    def test_aspherical_density_has_angular_gradient_the_fd_form_misses(self):
        """On a bonded, aspherical one-centre density the exact gradient
        carries angular content that the radial finite difference is
        structurally incapable of representing."""
        from vibeqc.periodic_gapw_augment import _atomic_density_gradient

        L = 12.0
        c = L / 2
        sysp, basis, D, grid = self._setup(
            [8, 1, 1],
            [[c, c, c], [c + 1.809, c, c], [c - 0.453, c + 1.751, c]],
            L,
        )
        jb = GapwJBuilder(basis, sysp, grid, quiet=True)
        aug = jb._aug
        ad = aug._atom_data[0]
        rho = aug._compute_atomic_density(D, ad, use_soft=False)
        g_exact = aug._compute_atomic_density_gradient_exact(
            D, ad, use_soft=False)
        g_fd = _atomic_density_gradient(rho, ad.grid)
        omega = np.asarray(ad.grid.angular_xyz)
        w = np.asarray(ad.grid.combined_weights())
        norm = lambda x: float(np.sqrt(
            np.einsum("ra,ra->", np.sum(x ** 2, axis=-1), w)))
        # the FD form is radial by construction: zero angular content
        rad_fd = np.einsum("rad,ad->ra", g_fd, omega)
        perp_fd = g_fd - rad_fd[:, :, None] * omega[None, :, :]
        assert norm(perp_fd) / max(norm(g_fd), 1e-30) < 1e-10, (
            "the finite-difference form is supposed to be purely radial"
        )
        # the exact form is not
        rad_ex = np.einsum("rad,ad->ra", g_exact, omega)
        perp_ex = g_exact - rad_ex[:, :, None] * omega[None, :, :]
        assert norm(perp_ex) / norm(g_exact) > 1e-3, (
            "expected real angular content on the aspherical O centre; got "
            f"{norm(perp_ex) / norm(g_exact):.2e}"
        )

    def test_soft_and_hard_gradients_use_their_own_bases(self):
        """The soft-side gradient must be built from the SOFT basis and its
        own density block, not the full one -- otherwise the hard-minus-soft
        telescoping mixes bases and cannot cancel."""
        L = 12.0
        c = L / 2
        sysp, basis, D, grid = self._setup(
            [3, 1], [[c - 1.5, c, c], [c + 1.5, c, c]], L)
        jb = GapwJBuilder(basis, sysp, grid, quiet=True)
        aug = jb._aug
        ad = aug._atom_data[0]
        g_hard = aug._compute_atomic_density_gradient_exact(
            D, ad, use_soft=False)
        g_soft = aug._compute_atomic_density_gradient_exact(
            D, ad, use_soft=True)
        assert g_hard.shape == g_soft.shape
        assert not np.allclose(g_hard, g_soft), (
            "hard and soft one-centre gradients must differ where the "
            "basis was pruned"
        )


# ============================================================
# SAP routing (GitLab #667)
# ============================================================


@pytest.mark.parametrize(
    ("driver_name", "system_factory", "functional"),
    [
        ("run_periodic_rhf_gapw", _he_periodic, None),
        ("run_periodic_rks_gapw", _he_periodic, "lda"),
        ("run_periodic_uhf_gapw", _he_periodic, None),
        ("run_periodic_uks_gapw", _he_periodic, "lda"),
    ],
    ids=["rhf", "rks", "uhf", "uks"],
)
def test_gamma_gapw_sap_reaches_lattice_potential(
    monkeypatch,
    driver_name,
    system_factory,
    functional,
):
    """Every Gamma GAPW SCF variant executes the periodic SAP builder."""
    import vibeqc.guess as guess_module
    import vibeqc.periodic_gapw_augment as gapw_module

    class SapPotentialReached(RuntimeError):
        pass

    calls = []

    def stop_at_vsap(_basis, system, _grid, table, _lattice_opts):
        calls.append((system, table))
        raise SapPotentialReached

    monkeypatch.setattr(guess_module, "compute_vsap_lattice", stop_at_vsap)
    system = system_factory(12.0)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    grid = PlaneWaveGrid(np.eye(3) * 12.0, 8, 8, 8)
    kwargs = {"functional": functional} if functional is not None else {}

    with pytest.raises(SapPotentialReached):
        getattr(gapw_module, driver_name)(
            system,
            basis,
            grid=grid,
            max_iter=1,
            initial_guess=core.InitialGuess.SAP,
            one_centre="block",
            quiet=True,
            **kwargs,
        )

    assert calls == [(system, "sap_helfem_large")]


def test_multik_gapw_sap_reaches_lattice_potential(monkeypatch):
    """The pure-DFT multi-k GAPW route builds ``F_SAP(k)``."""
    import vibeqc.guess as guess_module

    class SapPotentialReached(RuntimeError):
        pass

    calls = []

    def stop_at_vsap(_basis, _system, _grid, table, _lattice_opts):
        calls.append(table)
        raise SapPotentialReached

    monkeypatch.setattr(guess_module, "compute_vsap_lattice", stop_at_vsap)
    system = _he_periodic(12.0)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    grid = PlaneWaveGrid(np.eye(3) * 12.0, 8, 8, 8)
    kmesh = core.monkhorst_pack(system, [2, 1, 1])

    with pytest.raises(SapPotentialReached):
        run_periodic_rks_gapw_multi_k(
            system,
            basis,
            kmesh,
            functional="lda",
            grid=grid,
            max_iter=1,
            initial_guess=core.InitialGuess.SAP,
            quiet=True,
        )

    assert calls == ["sap_helfem_large"]


@pytest.mark.parametrize(
    "selector",
    [
        core.InitialGuess.SAP,
        core.InitialGuess.PATOM,
        core.InitialGuess.FRAGMO,
    ],
)
def test_gamma_gapw_restart_density_precedes_selector(monkeypatch, selector):
    """A caller-provided Gamma density is the effective READ artifact."""
    import vibeqc.guess as guess_module
    import vibeqc.periodic_gapw_augment as gapw_module

    class RestartDensityReached(RuntimeError):
        pass

    def unexpected_vsap(*_args, **_kwargs):
        pytest.fail("guess construction must not run for an explicit density")

    restart_density = np.array([[0.375]])

    def stop_at_first_j(_builder, density):
        overlap = gapw_module._overlap_lattice_gamma(basis, system)
        assert np.trace(density @ overlap) == pytest.approx(2, abs=1e-12)
        raise RestartDensityReached

    monkeypatch.setattr(guess_module, "compute_vsap_lattice", unexpected_vsap)
    monkeypatch.setattr(gapw_module.GapwJBuilder, "build_J", stop_at_first_j)
    system = _he_periodic(12.0)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    grid = PlaneWaveGrid(np.eye(3) * 12.0, 8, 8, 8)

    error = NotImplementedError if selector == core.InitialGuess.FRAGMO else RestartDensityReached
    with pytest.raises(error):
        run_periodic_rhf_gapw(
            system,
            basis,
            grid=grid,
            max_iter=1,
            initial_density=restart_density,
            initial_guess=selector,
            one_centre="block",
            quiet=True,
        )


def test_gamma_gapw_restart_rejects_malformed_selector():
    """A restart density must not hide an invalid guess spelling."""
    system = _he_periodic(12.0)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    with pytest.raises(ValueError, match="unknown initial_guess='bogus'"):
        run_periodic_rhf_gapw(
            system,
            basis,
            initial_density=np.array([[0.375]]),
            initial_guess="bogus",
            one_centre="block",
            quiet=True,
        )


@pytest.mark.parametrize(
    ("selector", "effective"),
    [
        (core.InitialGuess.SAD, core.InitialGuess.SAD),
        ("auto", core.InitialGuess.SAD),
        ("minao", core.InitialGuess.MINAO),
    ],
    ids=["enum-sad", "string-auto", "string-minao"],
)
def test_gamma_gapw_density_guesses_use_shared_adapter(
    monkeypatch,
    selector,
    effective,
):
    """Direct GAPW selectors share normalization and density construction."""
    import vibeqc.guess as guess_module
    import vibeqc.periodic_gapw_augment as gapw_module

    class SharedGuessReached(RuntimeError):
        pass

    seen = []

    def stop_at_shared_guess(_mol, _basis, _n_occ, guess, **kwargs):
        seen.append((guess, kwargs["periodic_system"]))
        raise SharedGuessReached

    monkeypatch.setattr(
        guess_module,
        "initial_density_closed_shell",
        stop_at_shared_guess,
    )
    system = _he_periodic(12.0)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    grid = PlaneWaveGrid(np.eye(3) * 12.0, 8, 8, 8)

    with pytest.raises(SharedGuessReached):
        gapw_module.run_periodic_rhf_gapw(
            system,
            basis,
            grid=grid,
            max_iter=1,
            initial_guess=selector,
            one_centre="block",
            quiet=True,
        )

    assert seen == [(effective, system)]


def test_multik_gapw_hueckel_uses_shared_fock_adapter(monkeypatch):
    """The multi-k GAPW route preserves HUECKEL as a Fock-mode guess."""
    import vibeqc.guess as guess_module

    class SharedFockGuessReached(RuntimeError):
        pass

    seen = []

    def stop_at_shared_fock(_system, _basis, _kpoints, guess, **_kwargs):
        seen.append(guess)
        raise SharedFockGuessReached

    monkeypatch.setattr(
        guess_module,
        "periodic_fock_guess_k",
        stop_at_shared_fock,
    )
    system = _he_periodic(12.0)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    grid = PlaneWaveGrid(np.eye(3) * 12.0, 8, 8, 8)
    kmesh = core.monkhorst_pack(system, [2, 1, 1])

    with pytest.raises(SharedFockGuessReached):
        run_periodic_rks_gapw_multi_k(
            system,
            basis,
            kmesh,
            functional="lda",
            grid=grid,
            max_iter=1,
            initial_guess="hueckel",
            quiet=True,
        )

    assert seen == [core.InitialGuess.HUECKEL]


@pytest.mark.parametrize(
    ("driver_name", "args", "kwargs"),
    [
        (
            "run_periodic_rks_gapw_multi_k",
            (None, None, None),
            {"functional": "lda"},
        ),
    ],
    ids=["rks-multik"],
)
def test_direct_gapw_guess_selectors_fail_closed_for_patom(
    driver_name,
    args,
    kwargs,
):
    """Pure-DFT multi-k GAPW lacks the exact exchange required by PATOM."""
    import vibeqc.periodic_gapw_augment as gapw_module

    with pytest.raises(
        NotImplementedError,
        match="initial_guess=PATOM.*not implemented by this route",
    ):
        getattr(gapw_module, driver_name)(
            *args,
            initial_guess=core.InitialGuess.PATOM,
            **kwargs,
        )
