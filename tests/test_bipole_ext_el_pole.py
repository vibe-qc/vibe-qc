"""Tests for BIPOLE density Fourier and spheropole infrastructure.

The complete-AO-density reciprocal Ewald diagnostic is not CRYSTAL's
``EXT EL-POLE`` multipole/penetration decomposition.  These tests pin
that distinction as well as the underlying AO-density invariants.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import CoulombMethod, InitialGuess, LatticeSumOptions
from vibeqc._aopair_ft import ao_pair_fourier_transform
from vibeqc._vibeqc_core import compute_overlap_lattice
from vibeqc.bipole_ext_el_pole import (
    ElectronicReciprocalEwaldResult,
    _libint_ylm_correction_per_ao,
    compute_cell_density_fourier,
    compute_electronic_reciprocal_ewald_energy,
    compute_ext_el_spheropole,
    compute_reciprocal_lattice_vectors,
    crystal_default_ewald_alpha,
)
from vibeqc.bipole_lattice_self_energy import cell_volume_bohr3
from vibeqc.guess import initial_density_closed_shell

ANG2BOHR = 1.0 / 0.529177210903


@pytest.fixture
def mgo_primitive():
    a = 4.21 * ANG2BOHR
    lattice = (a / 2.0) * np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    atoms = [
        vq.Atom(12, [0.0, 0.0, 0.0]),
        vq.Atom(8, [a / 2.0, a / 2.0, a / 2.0]),
    ]
    system = vq.PeriodicSystem(3, lattice, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


# ---------------------------------------------------------------------
# crystal_default_ewald_alpha
# ---------------------------------------------------------------------
def test_crystal_default_alpha_formula():
    """α = 2.8 / V^{1/3}."""
    V = 1000.0  # bohr³
    alpha = crystal_default_ewald_alpha(V)
    expected = 2.8 / V ** (1.0 / 3.0)
    assert math.isclose(alpha, expected, rel_tol=1e-12)


def test_crystal_default_alpha_mgo_value():
    """For MgO FCC primitive (V ≈ 126 bohr³): α ≈ 0.558 bohr⁻¹."""
    alpha = crystal_default_ewald_alpha(125.8876)  # MgO V
    assert math.isclose(alpha, 0.5587, rel_tol=1e-3)


def test_crystal_default_alpha_lih_value():
    """For LiH FCC primitive (V ≈ 28.7 bohr³): α ≈ 0.915 bohr⁻¹."""
    alpha = crystal_default_ewald_alpha(28.7)
    assert math.isclose(alpha, 0.915, abs_tol=0.02)


def test_crystal_default_alpha_invalid():
    with pytest.raises(ValueError):
        crystal_default_ewald_alpha(0.0)
    with pytest.raises(ValueError):
        crystal_default_ewald_alpha(-1.0)


# ---------------------------------------------------------------------
# compute_reciprocal_lattice_vectors
# ---------------------------------------------------------------------
def test_reciprocal_vectors_dim_check():
    lat = np.eye(3)
    sys2d = vq.PeriodicSystem(2, lat, [vq.Atom(1, [0, 0, 0])])
    with pytest.raises(ValueError):
        compute_reciprocal_lattice_vectors(sys2d, 1.0)


def test_reciprocal_vectors_K_max_positive():
    system = vq.PeriodicSystem(
        3,
        5.0 * np.eye(3),
        [vq.Atom(1, [0, 0, 0])],
    )
    with pytest.raises(ValueError):
        compute_reciprocal_lattice_vectors(system, 0.0)


def test_reciprocal_vectors_excludes_K_zero():
    """K=0 is excluded from the list (singular for 1/K² kernel)."""
    system = vq.PeriodicSystem(
        3,
        5.0 * np.eye(3),
        [vq.Atom(1, [0, 0, 0])],
    )
    K = compute_reciprocal_lattice_vectors(system, 5.0)
    K2 = (K**2).sum(axis=1)
    assert (K2 > 0.0).all()


def test_reciprocal_vectors_cubic_count():
    """For a cubic cell with a = 5 bohr, |b| = 2π/5 ≈ 1.257 bohr⁻¹.
    K_max = 3 → at most (2·3+1)³ - 1 = 342 vectors in the K-box,
    but only those inside the K_max sphere are kept."""
    system = vq.PeriodicSystem(
        3,
        5.0 * np.eye(3),
        [vq.Atom(1, [0, 0, 0])],
    )
    K = compute_reciprocal_lattice_vectors(system, 3.0)
    K2 = (K**2).sum(axis=1)
    # All retained vectors have |K| ≤ 3.0.
    assert (K2 <= 9.0 + 1e-10).all()
    # K=0 excluded.
    assert (K2 > 1e-12).all()
    # Reasonable count (not zero, not absurd).
    assert 50 < len(K) < 500


def test_reciprocal_vectors_follow_column_lattice_convention_on_skew_cell():
    """The core stores direct and reciprocal lattice vectors as columns."""
    lattice = np.array(
        [
            [7.0, 0.4, 0.2],
            [0.3, 8.0, 0.5],
            [0.1, 0.6, 9.0],
        ]
    )
    system = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(2, [0.0, 0.0, 0.0])],
    )
    K_max = 2.1
    actual = compute_reciprocal_lattice_vectors(system, K_max)

    reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T
    grid = np.arange(-6, 7)
    n1, n2, n3 = np.meshgrid(grid, grid, grid, indexing="ij")
    idx = np.stack([n1.ravel(), n2.ravel(), n3.ravel()], axis=-1)
    oracle = idx @ reciprocal.T
    oracle_norm_sq = np.einsum("ij,ij->i", oracle, oracle)
    oracle = oracle[
        (oracle_norm_sq > 1e-12) & (oracle_norm_sq <= K_max**2)
    ]

    def _sorted_set(vectors):
        return np.array(sorted(map(tuple, np.round(vectors, 12))))

    assert np.array_equal(_sorted_set(actual), _sorted_set(oracle))


# ---------------------------------------------------------------------
# compute_cell_density_fourier — basic shape + properties
# ---------------------------------------------------------------------
def test_cell_density_fourier_shape(mgo_primitive):
    system, basis = mgo_primitive
    opts = LatticeSumOptions()
    opts.cutoff_bohr = 8.0
    n_occ = system.n_electrons() // 2
    S_lat = compute_overlap_lattice(basis, system, opts)
    D_sad = initial_density_closed_shell(
        system.unit_cell_molecule(),
        basis,
        n_occ,
        InitialGuess.SAD,
        is_periodic=True,
    )
    for g_idx in range(len(S_lat.cells)):
        is_g0 = (np.asarray(S_lat.cells[g_idx].index) == np.array([0, 0, 0])).all()
        S_lat.set_block(
            g_idx, np.asarray(D_sad) if is_g0 else np.zeros_like(np.asarray(D_sad))
        )
    K = compute_reciprocal_lattice_vectors(system, 5.0)
    rho_hat = compute_cell_density_fourier(S_lat, basis, K)
    assert rho_hat.shape == (len(K),)
    assert rho_hat.dtype == np.complex128


def test_cell_density_fourier_zero_mode_is_electron_count(mgo_primitive):
    """The corrected mixed-s/p transform obeys rho_hat(0) = sum_mn P_mn S_mn."""
    system, basis = mgo_primitive
    opts = LatticeSumOptions()
    opts.cutoff_bohr = 8.0
    n_occ = system.n_electrons() // 2
    P_real = compute_overlap_lattice(basis, system, opts)
    D_sad = initial_density_closed_shell(
        system.unit_cell_molecule(),
        basis,
        n_occ,
        InitialGuess.SAD,
        is_periodic=True,
    )
    for g_idx in range(len(P_real.cells)):
        is_g0 = (
            np.asarray(P_real.cells[g_idx].index) == np.array([0, 0, 0])
        ).all()
        P_real.set_block(
            g_idx,
            np.asarray(D_sad) if is_g0 else np.zeros_like(np.asarray(D_sad)),
        )

    rho_zero = compute_cell_density_fourier(
        P_real,
        basis,
        np.zeros((1, 3)),
    )[0]
    overlap = np.asarray(vq.compute_overlap(basis))
    raw_zero = ao_pair_fourier_transform(basis, np.zeros((1, 3)))[:, :, 0]
    correction = _libint_ylm_correction_per_ao(basis)
    corrected_zero = raw_zero * np.outer(correction, correction)
    electron_count = float(np.einsum("mn,mn->", np.asarray(D_sad), overlap))

    np.testing.assert_allclose(corrected_zero.real, overlap, atol=2e-13, rtol=0.0)
    np.testing.assert_allclose(corrected_zero.imag, 0.0, atol=2e-13, rtol=0.0)
    assert electron_count == pytest.approx(20.0, abs=2e-13)
    assert rho_zero.real == pytest.approx(electron_count, abs=2e-13)
    assert rho_zero.imag == pytest.approx(0.0, abs=2e-13)


# ---------------------------------------------------------------------
# Reciprocal electronic Ewald diagnostic -- runs cleanly, returns finite
# ---------------------------------------------------------------------
def test_electronic_reciprocal_ewald_energy_runs_cleanly(mgo_primitive):
    system, basis = mgo_primitive
    opts = LatticeSumOptions()
    opts.cutoff_bohr = 8.0
    n_occ = system.n_electrons() // 2
    S_lat = compute_overlap_lattice(basis, system, opts)
    D_sad = initial_density_closed_shell(
        system.unit_cell_molecule(),
        basis,
        n_occ,
        InitialGuess.SAD,
        is_periodic=True,
    )
    for g_idx in range(len(S_lat.cells)):
        is_g0 = (np.asarray(S_lat.cells[g_idx].index) == np.array([0, 0, 0])).all()
        S_lat.set_block(
            g_idx, np.asarray(D_sad) if is_g0 else np.zeros_like(np.asarray(D_sad))
        )

    result = compute_electronic_reciprocal_ewald_energy(
        S_lat,
        basis,
        system,
        precision=1e-6,
    )
    assert isinstance(result, ElectronicReciprocalEwaldResult)
    assert math.isfinite(result.energy)
    assert result.energy > 0.0
    assert result.energy == pytest.approx(3.76425138661156, abs=2e-12)
    assert result.n_k_vectors == 530
    # The α used should match the requested default.
    V = cell_volume_bohr3(system)
    assert math.isclose(
        result.alpha_bohr_inv,
        crystal_default_ewald_alpha(V),
        rel_tol=1e-12,
    )

    # Independent operator identity: the renamed scalar must equal the
    # density contraction of the production J^LR real-space blocks.
    from vibeqc.bipole_fock_ewald import compute_J_long_range_real_space_blocks

    j_lr_blocks = compute_J_long_range_real_space_blocks(
        S_lat,
        basis,
        system,
        result.alpha_bohr_inv,
        precision=1e-6,
    )
    operator_energy = 0.5 * sum(
        float(np.einsum("mn,mn->", np.asarray(p), np.asarray(j)))
        for p, j in zip(S_lat.blocks, j_lr_blocks)
    )
    assert result.energy == pytest.approx(operator_energy, abs=2e-12)


def test_electronic_reciprocal_ewald_energy_dim_check():
    """Requires dim=3."""
    lat = np.eye(3) * 5.0
    sys2d = vq.PeriodicSystem(2, lat, [vq.Atom(2, [0, 0, 0])])  # He (closed shell)
    basis = vq.BasisSet(sys2d.unit_cell_molecule(), "sto-3g")
    opts = LatticeSumOptions()
    opts.cutoff_bohr = 5.0
    S_lat = compute_overlap_lattice(basis, sys2d, opts)
    with pytest.raises(ValueError):
        compute_electronic_reciprocal_ewald_energy(S_lat, basis, sys2d)


def test_electronic_reciprocal_ewald_energy_parameter_validation(mgo_primitive):
    system, basis = mgo_primitive
    opts = LatticeSumOptions()
    opts.cutoff_bohr = 1.0
    density = compute_overlap_lattice(basis, system, opts)

    invalid_cases = [
        ({"alpha_bohr_inv": 0.0}, "alpha"),
        ({"alpha_bohr_inv": -1.0}, "alpha"),
        ({"alpha_bohr_inv": math.inf}, "alpha"),
        ({"alpha_bohr_inv": math.nan}, "alpha"),
        ({"precision": 0.0}, "precision"),
        ({"precision": -1.0}, "precision"),
        ({"precision": 1.0}, "precision"),
        ({"precision": math.inf}, "precision"),
        ({"precision": math.nan}, "precision"),
        ({"pad_factor": 0.0}, "pad_factor"),
        ({"pad_factor": 0.5}, "pad_factor"),
        ({"pad_factor": -1.0}, "pad_factor"),
        ({"pad_factor": math.inf}, "pad_factor"),
        ({"pad_factor": math.nan}, "pad_factor"),
    ]
    for kwargs, parameter in invalid_cases:
        with pytest.raises(ValueError, match=parameter):
            compute_electronic_reciprocal_ewald_energy(
                density,
                basis,
                system,
                **kwargs,
            )


# ---------------------------------------------------------------------
# compute_ext_el_spheropole — partial implementation (2026-05-18)
# ---------------------------------------------------------------------
def test_ext_el_spheropole_matches_crystal_mgo_sad(mgo_primitive):
    """``compute_ext_el_spheropole`` reproduces CRYSTAL's sealed CYC 0
    value on the canonical MgO/STO-3G SAD density.

    Sealed CRYSTAL reference (mgo_sto3g_enecycle.out): ``EXT
    EL-SPHEROPOLE = +4.1191890135 Ha`` at CYC 0.

    The kernel is the exact bond-symmetrised second moment
    ``⟨φ_μ|(r−A)²+(r−B−g)²|φ_ν⟩`` computed via libint ``emultipole2``
    (``compute_multipole_moments_lattice``). This reproduces CRYSTAL to
    < 0.02 mHa — far tighter than the prior per-(l_a,l_b) spheropole C++
    kernel, whose empirically-fudged p-p term was ~7 mHa off (hence the
    old ±20 % bound). The tight bound here would FAIL on that fudge,
    locking in the exact integral.
    """
    system, basis = mgo_primitive
    opts = LatticeSumOptions()
    opts.cutoff_bohr = 10.0
    n_occ = system.n_electrons() // 2
    S_lat = compute_overlap_lattice(basis, system, opts)
    D_sad = initial_density_closed_shell(
        system.unit_cell_molecule(),
        basis,
        n_occ,
        InitialGuess.SAD,
        is_periodic=True,
    )
    for g_idx in range(len(S_lat.cells)):
        is_g0 = (np.asarray(S_lat.cells[g_idx].index) == np.array([0, 0, 0])).all()
        S_lat.set_block(
            g_idx, np.asarray(D_sad) if is_g0 else np.zeros_like(np.asarray(D_sad))
        )
    E_sphero = compute_ext_el_spheropole(S_lat, basis, system, opts)
    assert E_sphero > 0.0
    # Tight CRYSTAL parity (actual residual ~7 µHa; the 7-mHa fudge fails this).
    assert math.isclose(E_sphero, 4.1191890135, abs_tol=2e-3), (
        f"MgO spheropole {E_sphero:.6f} Ha vs CRYSTAL 4.1191890 Ha"
    )


def test_ext_el_spheropole_higher_l_d_shell():
    """Higher-l (d-shell) path: NiO/STO-3G exercises the Ni 3d shell.

    The rewritten kernel has **no per-l branching** — every AO pair goes
    through the same l-agnostic bond-symmetrised-second-moment assembly
    from exact ``emultipole2`` moments. So d/f correctness follows from
    the s/p CRYSTAL parity in
    ``test_ext_el_spheropole_matches_crystal_mgo_sad`` (0.0073 mHa). This
    test pins the exact value on a d-containing system as a regression
    guard against any reintroduced d fallback/fudge.

    NB — this value is NOT a CRYSTAL parity check: CRYSTAL's NiO reference
    uses an open-shell UHF + SPINLOCK density, whereas this regression uses
    vibe-qc's closed-shell SAD density (current Mulliken Ni 3d population:
    8.0). The spheropole *kernel* is validated exact on MgO; this guards
    the d code path itself.
    """
    ANG2BOHR = 1.0 / 0.529177210903
    a = 4.164 * ANG2BOHR  # NiO rocksalt, matching nio_sto3g.d12
    lattice = (a / 2.0) * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )
    system = vq.PeriodicSystem(
        3, lattice, [vq.Atom(28, [0.0, 0.0, 0.0]), vq.Atom(8, [a / 2.0] * 3)]
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    # Confirm the basis genuinely carries d shells (Ni 3d) — the point of the test.
    assert any(int(sh.l) == 2 for sh in basis.shells())

    opts = LatticeSumOptions()
    opts.cutoff_bohr = 10.0
    n_occ = system.n_electrons() // 2
    S_lat = compute_overlap_lattice(basis, system, opts)
    D_sad = np.asarray(
        initial_density_closed_shell(
            system.unit_cell_molecule(), basis, n_occ, InitialGuess.SAD,
            is_periodic=True,
        )
    )
    home_idx = next(
        g_idx for g_idx, cell in enumerate(S_lat.cells)
        if (np.asarray(cell.index) == np.array([0, 0, 0])).all()
    )
    S_home = np.asarray(S_lat.blocks[home_idx], dtype=float).copy()
    d_pop = float(np.trace((D_sad @ S_home)[13:18, 13:18]))
    assert math.isclose(d_pop, 8.0, abs_tol=1e-10)

    for g_idx in range(len(S_lat.cells)):
        is_g0 = (np.asarray(S_lat.cells[g_idx].index) == np.array([0, 0, 0])).all()
        S_lat.set_block(g_idx, D_sad if is_g0 else np.zeros_like(D_sad))

    E_sphero = compute_ext_el_spheropole(S_lat, basis, system, opts)
    assert math.isfinite(E_sphero) and E_sphero > 0.0
    # Exact bond-symmetrised second moment at vibe-qc's NiO SAD density.
    # (Regresses to a distinct value if d ever falls back to the s-s kernel.)
    assert math.isclose(E_sphero, 12.1710097208, abs_tol=5e-3), (
        f"NiO d-shell spheropole {E_sphero:.6f} Ha; expected exact 12.1710 Ha"
    )


def test_ext_el_spheropole_dim_check():
    """raises ``ValueError`` for 1D / 2D systems (3D-only formula)."""
    lat = np.eye(3)
    sys2d = vq.PeriodicSystem(2, lat, [vq.Atom(2, [0.0, 0.0, 0.0])])
    basis = vq.BasisSet(sys2d.unit_cell_molecule(), "sto-3g")
    opts = LatticeSumOptions()
    opts.cutoff_bohr = 10.0
    S_lat = compute_overlap_lattice(basis, sys2d, opts)
    with pytest.raises(ValueError, match="dim=3"):
        compute_ext_el_spheropole(S_lat, basis, sys2d, opts)


@pytest.mark.slow
def test_ext_el_spheropole_matches_crystal_mgo(mgo_primitive):
    """Eventually: spheropole should match CRYSTAL's
    ``::: EXT EL-SPHEROPOLE = +4.119 Ha`` on MgO STO-3G CYC 0
    (sealed CRYSTAL reference).

    Currently skipped — see test description."""
    system, basis = mgo_primitive
    opts = LatticeSumOptions()
    opts.cutoff_bohr = 12.0
    n_occ = system.n_electrons() // 2
    S_lat = compute_overlap_lattice(basis, system, opts)
    D_sad = initial_density_closed_shell(
        system.unit_cell_molecule(),
        basis,
        n_occ,
        InitialGuess.SAD,
        is_periodic=True,
    )
    for g_idx in range(len(S_lat.cells)):
        is_g0 = (np.asarray(S_lat.cells[g_idx].index) == np.array([0, 0, 0])).all()
        S_lat.set_block(
            g_idx, np.asarray(D_sad) if is_g0 else np.zeros_like(np.asarray(D_sad))
        )
    E_sphero = compute_ext_el_spheropole(S_lat, basis, system, opts)
    # Within 10 mHa of CRYSTAL's sealed reference (0.24% tolerance).
    assert math.isclose(E_sphero, +4.1191890135, abs_tol=0.010)
