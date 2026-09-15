"""Open-shell (UHF/UKS) GAPW analytic one-centre mode.

Pins the 2026-08-01 open-shell wiring of ``one_centre="analytic"``:
the fit-free analytic-ERI Hartree augmentation on the spin-summed
density plus the spin-polarised projector-density XC augmentation
(mirroring CP2K's per-spin one-centre densities in
``calculate_vxc_atom``, qs_vxc_atom.F). Measured at 48^3 / 12-bohr
box / STO-3G vs molecular references:

    O triplet UHF   analytic +12.36 mHa   (block +15.03)
    OH doublet UHF  analytic +12.61 mHa   (multi-atom open-shell)
    O triplet LDA   analytic +12.37 mHa   (block +15.03)

The DFT envelope matches the closed-shell driver: single-atom cells
only (fitted off-centre XC densities are variationally soft, and the
softness grows with basis richness). HF has no such restriction.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import _vibeqc_core as core
from vibeqc.periodic_gapw_augment import (
    GapwJBuilder,
    run_periodic_uhf_gapw,
    run_periodic_uks_gapw,
)
from vibeqc.periodic_gapw_grid import PlaneWaveGrid

pytestmark = pytest.mark.filterwarnings(
    "ignore::vibeqc.periodic_gapw_grid.GAPWExperimentalWarning"
)

_L = 12.0
_C = _L / 2


def _setup(zs, pos, multiplicity):
    system = core.PeriodicSystem()
    system.dim = 3
    system.lattice = np.eye(3) * _L
    system.unit_cell = [core.Atom(z, p) for z, p in zip(zs, pos)]
    system.multiplicity = multiplicity
    mol = vq.Molecule(
        [vq.Atom(z, p) for z, p in zip(zs, pos)], 0, multiplicity)
    basis = vq.BasisSet(mol, "sto-3g")
    return system, mol, basis


@pytest.mark.slow
def test_uhf_analytic_o_triplet_matches_molecular():
    """O atom triplet UHF in analytic mode lands at the closed-shell
    identity accuracy (+12.4 mHa measured; block +15.0) -- the Hartree
    augmentation only sees the spin-summed density, so the closed-shell
    validation carries over."""
    system, mol, basis = _setup([8], [[_C] * 3], 3)
    opts = vq.UHFOptions()
    opts.conv_tol_energy = 1e-10
    e_mol = float(vq.run_uhf(mol, basis, opts).energy)
    grid = PlaneWaveGrid(np.eye(3) * _L, 48, 48, 48)
    r = run_periodic_uhf_gapw(
        system, basis, n_alpha=5, n_beta=3, grid=grid, quiet=True,
        max_iter=150, conv_tol_energy=1e-9, one_centre="analytic")
    assert r.converged
    err = float(r.energy) - e_mol
    assert abs(err) < 0.020, (
        f"O triplet UHF analytic error {1000 * err:+.1f} mHa "
        f"exceeds 20 mHa"
    )

    # The result must retain the augmented post-DIIS operators that produced
    # its reported orbitals. Reconstructing a smooth-GPW Fock downstream is a
    # different Hamiltonian for the analytic one-centre route.
    import scipy.linalg as sla

    assert r.fock_alpha is not None
    assert r.fock_beta is not None
    assert r.overlap is not None
    eps_alpha = sla.eigh(
        np.asarray(r.fock_alpha),
        np.asarray(r.overlap),
        eigvals_only=True,
    )
    eps_beta = sla.eigh(
        np.asarray(r.fock_beta),
        np.asarray(r.overlap),
        eigvals_only=True,
    )
    np.testing.assert_allclose(eps_alpha, r.mo_energies_alpha, atol=1e-10)
    np.testing.assert_allclose(eps_beta, r.mo_energies_beta, atol=1e-10)

    # The GAPW breakdown must close using the augmented Hartree term, not the
    # smooth-grid helper's pre-augmentation decomposition.
    bd = r.breakdown
    closed_total = (
        bd.e_kinetic
        + bd.e_nuclear_attraction
        + bd.e_hartree
        + bd.e_hf_exchange
        + bd.e_xc
        + bd.e_nuclear_repulsion
        + bd.e_dft_plus_u
    )
    assert closed_total == pytest.approx(r.energy, abs=1e-10)


def test_uhf_runner_adapter_bypasses_gpw_reconstruction_with_stored_focks(
    monkeypatch,
):
    """Stored GAPW operators make smooth-GPW J/K reconstruction unnecessary."""
    from vibeqc import periodic_gapw_j as gpw_j
    from vibeqc.periodic_gapw_runner_adapter import (
        gpw_uhf_result_to_runner_shape,
    )

    def _unexpected_reconstruction(*_args, **_kwargs):
        raise AssertionError("adapter attempted smooth-GPW Fock reconstruction")

    monkeypatch.setattr(
        gpw_j,
        "_kinetic_lattice_gamma",
        _unexpected_reconstruction,
    )

    fock_alpha = np.diag([-0.7, 0.2])
    fock_beta = np.diag([-0.6, 0.3])
    overlap = np.eye(2)
    result = SimpleNamespace(
        energy=-1.0,
        breakdown=SimpleNamespace(
            e_kinetic=1.0,
            e_nuclear_attraction=-2.0,
            e_hartree=0.5,
            e_hf_exchange=-0.5,
            e_nuclear_repulsion=0.0,
            e_xc=0.0,
            functional=None,
        ),
        density_alpha=np.diag([1.0, 0.0]),
        density_beta=np.zeros((2, 2)),
        mo_coeffs_alpha=np.eye(2),
        mo_coeffs_beta=np.eye(2),
        mo_energies_alpha=np.diag(fock_alpha),
        mo_energies_beta=np.diag(fock_beta),
        n_alpha=1,
        n_beta=0,
        n_iter=2,
        converged=True,
        grid=object(),
        fock_alpha=fock_alpha,
        fock_beta=fock_beta,
        overlap=overlap,
        scf_trace=(),
    )

    shaped = gpw_uhf_result_to_runner_shape(
        result, SimpleNamespace(lattice=np.eye(3) * 12), object())

    np.testing.assert_array_equal(shaped.fock_alpha, fock_alpha)
    np.testing.assert_array_equal(shaped.fock_beta, fock_beta)
    np.testing.assert_array_equal(shaped.overlap, overlap)


@pytest.mark.slow
def test_uhf_default_oh_radical_matches_molecular():
    """OH radical exercises the multi-atom UHF default.

    The default resolves to the fit-free analytic Hartree construction and
    gives +12.6 mHa. Multi-atom cells are allowed for HF -- only the DFT XC
    telescoping is restricted to single atoms.
    """
    system, mol, basis = _setup(
        [8, 1], [[_C] * 3, [_C + 1.832, _C, _C]], 2)
    opts = vq.UHFOptions()
    opts.conv_tol_energy = 1e-10
    e_mol = float(vq.run_uhf(mol, basis, opts).energy)
    grid = PlaneWaveGrid(np.eye(3) * _L, 48, 48, 48)
    r = run_periodic_uhf_gapw(
        system, basis, n_alpha=5, n_beta=4, grid=grid, quiet=True,
        max_iter=150, conv_tol_energy=1e-9, molecular_limit=True)
    assert r.converged
    assert r.one_centre == "analytic"
    assert r.molecular_limit_declared is True
    err = float(r.energy) - e_mol
    assert abs(err) < 0.020, (
        f"OH doublet UHF analytic error {1000 * err:+.1f} mHa "
        f"exceeds 20 mHa"
    )


@pytest.mark.slow
@pytest.mark.parametrize("functional", ["lda", "pbe"])
def test_uks_analytic_o_triplet_matches_molecular(functional):
    """O atom triplet UKS in analytic mode: the polarised
    projector-density XC augmentation plus the soft-block smooth-XC
    projection land at the single-atom identity accuracy (LDA +12.37,
    PBE +12.37 mHa measured; block +15.0/+13.3).

    PBE gets a looser density criterion: on this fixture the density
    rotates inside the degenerate 2p manifold while the energy is flat
    to ~1e-12 Ha (BLOCK mode stalls identically, dD ~3e-6 at 150
    iterations, so this is a pre-existing conditioning trait of the
    open-shell GAPW GGA path, not an analytic-mode regression). The
    energy convergence + accuracy assertions are the meaningful pins.
    """
    system, mol, basis = _setup([8], [[_C] * 3], 3)
    uks = vq.UKSOptions()
    uks.functional = functional
    uks.conv_tol_energy = 1e-10
    e_mol = float(vq.run_uks(mol, basis, uks).energy)
    grid = PlaneWaveGrid(np.eye(3) * _L, 48, 48, 48)
    conv_tol_density = 1e-7 if functional == "lda" else 1e-4
    r = run_periodic_uks_gapw(
        system, basis, functional=functional, n_alpha=5, n_beta=3,
        grid=grid, quiet=True, max_iter=150, conv_tol_energy=1e-9,
        conv_tol_density=conv_tol_density, one_centre="analytic")
    assert r.converged
    assert abs(r.scf_trace[-1]["delta_e"]) < 1e-9
    err = float(r.energy) - e_mol
    assert abs(err) < 0.030, (
        f"O triplet UKS/{functional} analytic error "
        f"{1000 * err:+.1f} mHa exceeds 30 mHa"
    )


def test_polarised_projector_xc_fock_is_exact():
    """The polarised projector XC augmentation's per-spin Fock
    matrices are the exact derivative of its energy (directional FD
    parity), for LDA and GGA alike -- including the v_sigma_ab cross
    terms, probed with asymmetric spin densities."""
    pos = [[_C, _C, _C], [_C + 1.809, _C, _C],
           [_C - 0.453, _C + 1.751, _C]]
    system, mol, basis = _setup([8, 1, 1], pos, 1)
    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-10
    D = np.asarray(vq.run_rhf(mol, basis, opts).density)
    D_a = 0.62 * D
    D_b = 0.38 * D
    grid = PlaneWaveGrid(np.eye(3) * _L, 24, 24, 24)
    jb = GapwJBuilder(basis, system, grid, one_centre="analytic",
                      quiet=True)
    rng = np.random.default_rng(11)
    dir_a = rng.standard_normal(D.shape)
    dir_a = 0.5 * (dir_a + dir_a.T)
    dir_b = rng.standard_normal(D.shape)
    dir_b = 0.5 * (dir_b + dir_b.T)
    t = 1e-6
    for func_name in ("lda", "pbe"):
        func = core.Functional(func_name, 2)
        corr_a, corr_b, _e = jb._analytic_xc_correction_polarised(
            D_a, D_b, func)
        V_a = sum(corr_a.values())
        V_b = sum(corr_b.values())
        _1, _2, e_p = jb._analytic_xc_correction_polarised(
            D_a + t * dir_a, D_b + t * dir_b, func)
        _3, _4, e_m = jb._analytic_xc_correction_polarised(
            D_a - t * dir_a, D_b - t * dir_b, func)
        fd = (e_p - e_m) / (2 * t)
        an = float(np.einsum("ij,ij->", V_a, dir_a)
                   + np.einsum("ij,ij->", V_b, dir_b))
        assert abs(fd - an) < 1e-6 * max(1.0, abs(fd)), (
            f"{func_name}: polarised XC-aug FD {fd} vs analytic {an}"
        )


def test_polarised_matches_unpolarised_on_closed_shell():
    """Consistency: an even spin split reproduces the closed-shell
    projector XC augmentation -- energy identical and each spin
    channel's correction equal to the unpolarised one (the libxc
    spin-scaling identity carried through the projector chain)."""
    pos = [[_C, _C, _C], [_C + 1.809, _C, _C],
           [_C - 0.453, _C + 1.751, _C]]
    system, mol, basis = _setup([8, 1, 1], pos, 1)
    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-10
    D = np.asarray(vq.run_rhf(mol, basis, opts).density)
    grid = PlaneWaveGrid(np.eye(3) * _L, 24, 24, 24)
    jb = GapwJBuilder(basis, system, grid, one_centre="analytic",
                      quiet=True)
    for func_name in ("lda", "pbe"):
        func1 = core.Functional(func_name, 1)
        func2 = core.Functional(func_name, 2)
        corr, e_unpol = jb._analytic_xc_correction(D, func1)
        corr_a, corr_b, e_pol = jb._analytic_xc_correction_polarised(
            0.5 * D, 0.5 * D, func2)
        assert e_pol == pytest.approx(e_unpol, abs=1e-10), func_name
        for ia in corr:
            np.testing.assert_allclose(
                corr_a[ia], corr[ia], atol=1e-10,
                err_msg=f"{func_name} alpha atom {ia}")
            np.testing.assert_allclose(
                corr_b[ia], corr[ia], atol=1e-10,
                err_msg=f"{func_name} beta atom {ia}")


def test_uks_analytic_multi_atom_fails_closed():
    """UKS analytic DFT keeps the closed-shell envelope: any
    multi-atom cell fails closed (fitted off-centre XC densities are
    variationally soft, softness grows with basis richness)."""
    system, mol, basis = _setup(
        [8, 1], [[_C] * 3, [_C + 1.832, _C, _C]], 2)
    grid = PlaneWaveGrid(np.eye(3) * _L, 16, 16, 16)
    with pytest.raises(NotImplementedError, match="single-atom cells"):
        run_periodic_uks_gapw(
            system, basis, functional="lda", n_alpha=5, n_beta=4,
            grid=grid, quiet=True, one_centre="analytic")


def test_uks_projector_mode_rejected():
    """one_centre='projector' is HF-only; the UKS driver refuses it
    outright."""
    system, mol, basis = _setup([8], [[_C] * 3], 3)
    grid = PlaneWaveGrid(np.eye(3) * _L, 16, 16, 16)
    with pytest.raises(NotImplementedError, match="HF-only"):
        run_periodic_uks_gapw(
            system, basis, functional="lda", n_alpha=5, n_beta=3,
            grid=grid, quiet=True, one_centre="projector")


def test_uhf_projector_multi_atom_fails_closed():
    """one_centre='projector' on the UHF driver keeps the single-atom
    envelope of the RHF driver."""
    system, mol, basis = _setup(
        [8, 1], [[_C] * 3, [_C + 1.832, _C, _C]], 2)
    grid = PlaneWaveGrid(np.eye(3) * _L, 16, 16, 16)
    with pytest.raises(NotImplementedError, match="single-atom cells"):
        run_periodic_uhf_gapw(
            system, basis, n_alpha=5, n_beta=4, grid=grid, quiet=True,
            one_centre="projector")
