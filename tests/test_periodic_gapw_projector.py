"""Tests for the projector-based GAPW one-centre expansion module.

Pins the 2026-07-29 oracle results (see
``examples/regression/gapw_parity/oracle_projector_augmentation_wip2.py``
and ``handovers/HANDOVER_GAPW_PRODUCTION.md``): the H2O fixed-density
Hartree identity closes to tens of mHa where the production
block-restricted augmentation is -4.2 Ha off, and the same
construction covers single atoms. Energy-evaluator only - the
production SCF is unchanged until the analytic Fock lands.
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import _vibeqc_core as core
from vibeqc.periodic_gapw_augment import (
    GapwJBuilder,
    collocate_density_on_grid,
    softened_basis,
)
from vibeqc.periodic_gapw_grid import PlaneWaveGrid
from vibeqc.periodic_gapw_j import _ewald_v_ne_gamma
from vibeqc.periodic_gapw_projector import (
    build_one_centre_expansions,
    projector_hartree_energy,
    pruned_atom_indices,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore::vibeqc.periodic_gapw_grid.GAPWExperimentalWarning"
)

_L = 12.0


def _setup(zs, pos):
    system = core.PeriodicSystem()
    system.dim = 3
    system.lattice = np.eye(3) * _L
    system.unit_cell = [core.Atom(z, p) for z, p in zip(zs, pos)]
    mol = vq.Molecule([vq.Atom(z, p) for z, p in zip(zs, pos)], 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")
    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-10
    D = np.asarray(vq.run_rhf(mol, basis, opts).density)
    return system, mol, basis, D


def _gauge_target(basis, mol, system, D):
    """Gauge-consistent periodic Hartree target: E_free - (dVne + Enn).

    Both relations were verified analytically against the image-gauge
    expansion (handover 2026-07-29): dVne = -sum_A Z_A [xi N +
    (2pi/3V) Q_r] + h.o.t. and Enn = xi/2 (sum Z)^2 + shape terms, so
    their sum pins the Hartree gauge shift to ~mHa.
    """
    eri = np.asarray(core.compute_eri(basis))
    e_free = 0.5 * float(np.einsum("ij,ijkl,kl->", D, eri, D))
    v_per = _ewald_v_ne_gamma(basis, system)
    v_mol = np.asarray(core.compute_nuclear(basis, mol))
    d_vne = float(np.einsum("ij,ij->", D, v_per - v_mol))
    e_nn_per = float(
        core.ewald_nuclear_repulsion(system, core.EwaldOptions()))
    # dEnn is periodic MINUS molecular nuclear repulsion (zero for a
    # single atom).
    atoms = list(system.unit_cell)
    e_nn_mol = 0.0
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            rij = float(np.linalg.norm(
                np.asarray(list(atoms[i].xyz))
                - np.asarray(list(atoms[j].xyz))))
            e_nn_mol += int(atoms[i].Z) * int(atoms[j].Z) / rij
    return e_free - (d_vne + e_nn_per - e_nn_mol)


def _projector_e_h(system, basis, D, n_fft=64):
    grid = PlaneWaveGrid(np.eye(3) * _L, n_fft, n_fft, n_fft)
    jb = GapwJBuilder(basis, system, grid, quiet=True)
    aug = jb._aug
    soft = aug._soft_basis
    idx = aug._soft_indices
    D_soft = D[np.ix_(idx, idx)]
    atom_grids = {ia: ad.grid for ia, ad in enumerate(aug._atom_data)}
    exps = build_one_centre_expansions(
        basis, soft, system, atom_grids)
    rho_t = collocate_density_on_grid(
        soft, D_soft, grid, cache=jb._soft_collocation_cache)
    e_h = projector_hartree_energy(
        exps, atom_grids, D, D_soft, rho_t, grid)
    return e_h, exps


def test_pruned_atom_detection_h2o():
    """Only O is pruned in H2O/STO-3G (H shells are preserved), so only
    O gets an augmentation sphere."""
    c = _L / 2
    pos = [[c, c, c], [c + 1.809, c, c], [c - 0.453, c + 1.751, c]]
    system, mol, basis, D = _setup([8, 1, 1], pos)
    soft = softened_basis(basis, system)
    assert pruned_atom_indices(basis, soft) == [0]


@pytest.mark.slow
def test_h2o_hartree_identity_closes():
    """The H2O fixed-density Hartree identity closes to <60 mHa.

    Production block restriction sits at -4213 mHa on the same
    measurement; the oracle's best point is +9 mHa. The bound leaves
    room for FFT/radial-grid defaults while still being two orders of
    magnitude tighter than production.
    """
    c = _L / 2
    pos = [[c, c, c], [c + 1.809, c, c], [c - 0.453, c + 1.751, c]]
    system, mol, basis, D = _setup([8, 1, 1], pos)
    e_h, exps = _projector_e_h(system, basis, D)
    target = _gauge_target(basis, mol, system, D)
    err = e_h - target
    assert abs(err) < 0.060, (
        f"H2O projector-augmentation identity error {1000 * err:+.1f} mHa "
        f"(production: -4213 mHa)"
    )
    assert len(exps) == 1 and exps[0].atom_index == 0


@pytest.mark.slow
def test_single_atom_identity_ne_and_he():
    """Single atoms under the same construction: Ne within 40 mHa
    (production windowed form: +53), He within 5 mHa."""
    for z, bound in ((10, 0.040), (2, 0.005)):
        system, mol, basis, D = _setup([z], [[_L / 2] * 3])
        e_h, _ = _projector_e_h(system, basis, D)
        target = _gauge_target(basis, mol, system, D)
        err = e_h - target
        assert abs(err) < bound, (
            f"Z={z}: projector-augmentation identity error "
            f"{1000 * err:+.1f} mHa exceeds {1000 * bound:.0f} mHa"
        )


def test_fock_is_exact_derivative_of_the_energy():
    """The analytic Fock is dE/dD exactly: the quadratic identity
    1/2 tr(D J) == E holds to machine precision and a directional
    finite difference of the energy matches tr(J . Dir)."""
    from vibeqc.periodic_gapw_projector import (
        projector_hartree_energy_and_fock,
    )

    system, mol, basis, D = _setup([10], [[_L / 2] * 3])
    grid = PlaneWaveGrid(np.eye(3) * _L, 32, 32, 32)
    jb = GapwJBuilder(basis, system, grid, quiet=True)
    aug = jb._aug
    soft = aug._soft_basis
    idx = np.asarray(aug._soft_indices)
    atom_grids = {ia: ad.grid for ia, ad in enumerate(aug._atom_data)}
    exps = build_one_centre_expansions(basis, soft, system, atom_grids)
    cache = jb._soft_collocation_cache
    soft_chi = np.asarray(cache.chi)

    def rho_t(Dm):
        return collocate_density_on_grid(
            soft, Dm[np.ix_(idx, idx)], grid, cache=cache)

    E, J = projector_hartree_energy_and_fock(
        exps, atom_grids, D, idx, rho_t(D), soft_chi, grid)
    assert abs(0.5 * float(np.einsum("ij,ij->", D, J)) - E) < 1e-9

    rng = np.random.default_rng(3)
    direction = rng.standard_normal(D.shape)
    direction = 0.5 * (direction + direction.T)
    t = 1e-5
    e_p = projector_hartree_energy(
        exps, atom_grids, D + t * direction,
        (D + t * direction)[np.ix_(idx, idx)],
        rho_t(D + t * direction), grid)
    e_m = projector_hartree_energy(
        exps, atom_grids, D - t * direction,
        (D - t * direction)[np.ix_(idx, idx)],
        rho_t(D - t * direction), grid)
    fd = (e_p - e_m) / (2 * t)
    an = float(np.einsum("ij,ij->", J, direction))
    assert abs(fd - an) < 1e-6 * max(1.0, abs(fd))


@pytest.mark.slow
def test_projector_scf_single_atoms():
    """Opt-in projector-mode SCF on single atoms lands at the
    fixed-density identity accuracy (no off-centre fits, no
    variational hole): Ne within 30 mHa of the molecular limit
    (block mode: +53), He within 2 mHa."""
    from vibeqc.periodic_gapw_augment import run_periodic_rhf_gapw

    for z, bound in ((10, 0.030), (2, 0.002)):
        system, mol, basis, _D = _setup([z], [[_L / 2] * 3])
        opts = vq.RHFOptions()
        opts.conv_tol_energy = 1e-10
        e_mol = float(vq.run_rhf(mol, basis, opts).energy)
        grid = PlaneWaveGrid(np.eye(3) * _L, 48, 48, 48)
        r = run_periodic_rhf_gapw(
            system, basis, grid=grid, quiet=True, max_iter=120,
            conv_tol_energy=1e-9, one_centre="projector")
        assert r.converged
        err = float(r.energy) - e_mol
        assert abs(err) < bound, (
            f"Z={z}: projector SCF error {1000 * err:+.1f} mHa "
            f"exceeds {1000 * bound:.0f} mHa"
        )


def test_projector_scf_multi_atom_fails_closed():
    """Multi-atom cells fail closed in projector mode: off-centre
    fit-error directions are variationally exploitable (the H2O SCF
    converges Ha-scale below the reference despite a +33 mHa
    fixed-density functional). Guard until that is solved."""
    from vibeqc.periodic_gapw_augment import run_periodic_rhf_gapw

    c = _L / 2
    pos = [[c, c, c], [c + 1.809, c, c], [c - 0.453, c + 1.751, c]]
    system, mol, basis, _D = _setup([8, 1, 1], pos)
    grid = PlaneWaveGrid(np.eye(3) * _L, 24, 24, 24)
    with pytest.raises(NotImplementedError, match="single-atom cells"):
        run_periodic_rhf_gapw(
            system, basis, grid=grid, quiet=True,
            one_centre="projector")


def test_analytic_fock_is_exact_derivative():
    """The analytic-ERI mode's Fock is dE/dD exactly (quadratic
    identity to machine precision + directional FD parity)."""
    from vibeqc.periodic_gapw_projector import (
        analytic_hartree_energy_and_fock,
        build_analytic_augmentation,
    )

    system, mol, basis, D = _setup([10], [[_L / 2] * 3])
    grid = PlaneWaveGrid(np.eye(3) * _L, 32, 32, 32)
    jb = GapwJBuilder(basis, system, grid, quiet=True)
    soft = jb._aug._soft_basis
    idx = np.asarray(jb._aug._soft_indices)
    cache = jb._soft_collocation_cache
    soft_chi = np.asarray(cache.chi)
    aug = build_analytic_augmentation(basis, soft, system, grid)

    def eval_at(Dm):
        rho_t = collocate_density_on_grid(
            soft, Dm[np.ix_(idx, idx)], grid, cache=cache)
        return analytic_hartree_energy_and_fock(
            aug, Dm, idx, rho_t, soft_chi, grid)

    E, J = eval_at(D)
    assert abs(0.5 * float(np.einsum("ij,ij->", D, J)) - E) < 1e-9
    rng = np.random.default_rng(5)
    direction = rng.standard_normal(D.shape)
    direction = 0.5 * (direction + direction.T)
    t = 1e-5
    e_p, _ = eval_at(D + t * direction)
    e_m, _ = eval_at(D - t * direction)
    fd = (e_p - e_m) / (2 * t)
    an = float(np.einsum("ij,ij->", J, direction))
    assert abs(fd - an) < 1e-6 * max(1.0, abs(fd))


def test_analytic_partition_conserves_total_pruned_charge():
    """The compensator charges must sum to the exact global
    tr(S_hard D) - tr(S_soft D_soft) for EVERY density, physical or
    not. Regression for the -3.0 Ha H2O variational hole: a per-atom
    Mulliken split did not sum to the exact total (unpruned atoms'
    Mulliken populations also shift between the bases), and the SCF
    mined the resulting Madelung-scale net-charge mismatch."""
    from vibeqc.periodic_gapw_augment import softened_basis
    from vibeqc.periodic_gapw_projector import (
        build_analytic_augmentation,
    )

    c = _L / 2
    pos = [[c, c, c], [c + 1.809, c, c], [c - 0.453, c + 1.751, c]]
    system, mol, basis, _D = _setup([8, 1, 1], pos)
    soft = softened_basis(basis, system)
    grid = PlaneWaveGrid(np.eye(3) * _L, 16, 16, 16)
    aug = build_analytic_augmentation(basis, soft, system, grid)

    S_h = np.asarray(core.compute_overlap(basis))
    S_s = np.asarray(core.compute_overlap(soft))
    jb = GapwJBuilder(basis, system, grid, quiet=True)
    idx = np.asarray(jb._aug._soft_indices)
    rng = np.random.default_rng(9)
    for _ in range(3):
        Dr = rng.standard_normal(S_h.shape)
        Dr = Dr + Dr.T  # arbitrary symmetric, deliberately unphysical
        D_soft = Dr[np.ix_(idx, idx)]
        exact = (float(np.einsum("ij,ij->", S_h, Dr))
                 - float(np.einsum("ij,ij->", S_s, D_soft)))
        total = sum(
            float(np.einsum("ij,ij->", aug.w_hard[i], Dr))
            - float(np.einsum("ij,ij->", aug.w_soft[i], D_soft))
            for i in range(len(aug.pruned))
        )
        assert abs(total - exact) < 1e-10


@pytest.mark.slow
def test_analytic_mode_scf_h2o_and_molecules():
    """THE headline gate: the fit-free analytic mode's H2O SCF lands
    within 30 mHa of the molecular reference - where the legacy block
    augmentation is -8.86 Ha wrong (the filed public-route bug) and
    the fitted projector mode had a -3.7 Ha variational hole. LiH
    within 20 mHa; H2 (no pruned atoms) within 2 mHa."""
    from vibeqc.periodic_gapw_augment import run_periodic_rhf_gapw

    c = _L / 2
    cases = [
        ([8, 1, 1],
         [[c, c, c], [c + 1.809, c, c], [c - 0.453, c + 1.751, c]],
         0.030),
        ([3, 1], [[c - 1.5075, c, c], [c + 1.5075, c, c]], 0.020),
        ([1, 1], [[c - 0.7, c, c], [c + 0.7, c, c]], 0.002),
    ]
    for zs, pos, bound in cases:
        system, mol, basis, _D = _setup(zs, pos)
        opts = vq.RHFOptions()
        opts.conv_tol_energy = 1e-10
        e_mol = float(vq.run_rhf(mol, basis, opts).energy)
        grid = PlaneWaveGrid(np.eye(3) * _L, 48, 48, 48)
        r = run_periodic_rhf_gapw(
            system, basis, grid=grid, quiet=True, max_iter=150,
            conv_tol_energy=1e-9, one_centre="analytic")
        assert r.converged
        err = float(r.energy) - e_mol
        assert abs(err) < bound, (
            f"{zs}: analytic-mode SCF error {1000 * err:+.1f} mHa "
            f"exceeds {1000 * bound:.0f} mHa"
        )


def test_analytic_mode_dft_multi_atom_fails_closed():
    """DFT in analytic mode supports single-atom cells only: the
    multi-atom XC telescoping runs on fitted off-centre one-centre
    densities whose variational softness GROWS with basis richness
    (LiH LDA -10.6 -> -105 mHa from STO-3G to def2-SVP; H2O up to
    -21 Ha at def2-SVP). Both a two-atom and a three-atom cell must
    refuse."""
    from vibeqc.periodic_gapw_augment import run_periodic_rhf_gapw

    c = _L / 2
    grid = PlaneWaveGrid(np.eye(3) * _L, 16, 16, 16)
    for zs, pos in (
        ([8, 1, 1],
         [[c, c, c], [c + 1.809, c, c], [c - 0.453, c + 1.751, c]]),
        ([3, 1], [[c - 1.5075, c, c], [c + 1.5075, c, c]]),
    ):
        system, mol, basis, _D = _setup(zs, pos)
        with pytest.raises(NotImplementedError,
                           match="single-atom cells"):
            run_periodic_rhf_gapw(
                system, basis, grid=grid, quiet=True,
                functional="lda", one_centre="analytic")


@pytest.mark.slow
def test_analytic_mode_rks_atoms_across_bases():
    """DFT on the analytic mode's validated envelope (single atoms,
    exact on-centre XC densities): Ne lands at HF-quality accuracy
    across bases. Also the regression pin for the soft-block V_xc
    projection fix - before it, the full-basis smooth-XC projection
    broke Fock/energy consistency and the Ne/6-31G SCF walked +373
    mHa uphill from the exact molecular density (+393 vs +15 now)."""
    from vibeqc.periodic_gapw_augment import run_periodic_rhf_gapw

    c = _L / 2
    cases = [
        ("sto-3g", "lda", 0.030),
        ("6-31g", "lda", 0.030),
        ("sto-3g", "pbe", 0.030),
    ]
    for bname, func, bound in cases:
        system = core.PeriodicSystem()
        system.dim = 3
        system.lattice = np.eye(3) * _L
        system.unit_cell = [core.Atom(10, [_L / 2] * 3)]
        mol = vq.Molecule([vq.Atom(10, [_L / 2] * 3)], 0, 1)
        basis = vq.BasisSet(mol, bname)
        opts = vq.RKSOptions()
        opts.functional = func
        opts.conv_tol_energy = 1e-10
        e_mol = float(vq.run_rks(mol, basis, opts).energy)
        grid = PlaneWaveGrid(np.eye(3) * _L, 48, 48, 48)
        r = run_periodic_rhf_gapw(
            system, basis, grid=grid, quiet=True, max_iter=150,
            conv_tol_energy=1e-9, functional=func,
            one_centre="analytic")
        assert r.converged
        err = float(r.energy) - e_mol
        assert abs(err) < bound, (
            f"Ne/{bname}/{func}: analytic-mode RKS error "
            f"{1000 * err:+.1f} mHa exceeds {1000 * bound:.0f} mHa"
        )


def test_projector_xc_correction_fock_is_exact():
    """The projector-density XC augmentation's Fock matrices are the
    exact derivative of its energy, for LDA and GGA alike
    (directional FD parity ~1e-9). This is the consistency the SCF
    needs; the variational softness of fitted XC densities at minimal
    bases is a separate, guarded matter."""
    c = _L / 2
    pos = [[c, c, c], [c + 1.809, c, c], [c - 0.453, c + 1.751, c]]
    system, mol, basis, D = _setup([8, 1, 1], pos)
    grid = PlaneWaveGrid(np.eye(3) * _L, 32, 32, 32)
    jb = GapwJBuilder(basis, system, grid, one_centre="analytic",
                      quiet=True)
    rng = np.random.default_rng(4)
    direction = rng.standard_normal(D.shape)
    direction = 0.5 * (direction + direction.T)
    t = 1e-6
    for func_name in ("lda", "pbe"):
        func = core.Functional(func_name, 1)
        corr, _e = jb._analytic_xc_correction(D, func)
        V = sum(corr.values())
        _c1, e_p = jb._analytic_xc_correction(D + t * direction, func)
        _c2, e_m = jb._analytic_xc_correction(D - t * direction, func)
        fd = (e_p - e_m) / (2 * t)
        an = float(np.einsum("ij,ij->", V, direction))
        assert abs(fd - an) < 1e-6 * max(1.0, abs(fd)), (
            f"{func_name}: XC-aug FD {fd} vs analytic {an}"
        )


@pytest.mark.slow
def test_analytic_mode_hf_atomization_sub_mha():
    """The atomization-workflow gate: molecular HF atomization energies
    on the analytic route match exact molecular references to <= 2.5
    mHa (measured: H2 +0.01, H2O +0.85 at L=12; LiH +0.98 at L=16 -
    the per-system absolute residuals of ~+7..+12 mHa cancel between
    the Hund-ground-state atoms and the molecule). LiH needs the
    larger box: at L=12 the diffuse Li 2s is compressed and the
    cancellation degrades to -11.4 mHa, which the L-ladder pins as a
    finite-size effect, not a formulation error."""
    from vibeqc.periodic_gapw_augment import (
        run_periodic_rhf_gapw,
        run_periodic_uhf_gapw,
    )

    def mol_uhf_atom(Z, mult):
        mol = vq.Molecule([vq.Atom(Z, [0, 0, 0])], 0, mult)
        b = vq.BasisSet(mol, "sto-3g")
        o = vq.UHFOptions()
        o.conv_tol_energy = 1e-10
        return float(vq.run_uhf(mol, b, o).energy)

    def mol_rhf(zs, pos):
        mol = vq.Molecule(
            [vq.Atom(z, p) for z, p in zip(zs, pos)], 0, 1)
        b = vq.BasisSet(mol, "sto-3g")
        o = vq.RHFOptions()
        o.conv_tol_energy = 1e-10
        return float(vq.run_rhf(mol, b, o).energy)

    def gapw_atom(Z, mult, L, N):
        cc = L / 2
        system = core.PeriodicSystem()
        system.dim = 3
        system.lattice = np.eye(3) * L
        system.unit_cell = [core.Atom(Z, [cc] * 3)]
        mol = vq.Molecule([vq.Atom(Z, [cc] * 3)], 0, mult)
        b = vq.BasisSet(mol, "sto-3g")
        n_a = (Z + (mult - 1)) // 2
        r = run_periodic_uhf_gapw(
            system, b, grid=PlaneWaveGrid(np.eye(3) * L, N, N, N),
            quiet=True, max_iter=200, conv_tol_energy=1e-9,
            n_alpha=n_a, n_beta=Z - n_a, one_centre="analytic")
        assert r.converged
        return float(r.energy)

    def gapw_mol(zs, pos, L, N):
        system = core.PeriodicSystem()
        system.dim = 3
        system.lattice = np.eye(3) * L
        system.unit_cell = [core.Atom(z, p) for z, p in zip(zs, pos)]
        mol = vq.Molecule(
            [vq.Atom(z, p) for z, p in zip(zs, pos)], 0, 1)
        b = vq.BasisSet(mol, "sto-3g")
        r = run_periodic_rhf_gapw(
            system, b, grid=PlaneWaveGrid(np.eye(3) * L, N, N, N),
            quiet=True, max_iter=200, conv_tol_energy=1e-9,
            one_centre="analytic")
        assert r.converged
        return float(r.energy)

    cases = [
        ("H2", [1, 1], [(1, 2), (1, 2)], 12.0, 48,
         lambda cc: [[cc - 0.7, cc, cc], [cc + 0.7, cc, cc]]),
        ("H2O", [8, 1, 1], [(8, 3), (1, 2), (1, 2)], 12.0, 48,
         lambda cc: [[cc, cc, cc], [cc + 1.809, cc, cc],
                     [cc - 0.453, cc + 1.751, cc]]),
        ("LiH", [3, 1], [(3, 2), (1, 2)], 16.0, 64,
         lambda cc: [[cc - 1.5075, cc, cc], [cc + 1.5075, cc, cc]]),
    ]
    for name, zs, atom_specs, L, N, posf in cases:
        pos = posf(L / 2)
        e_ref = (sum(mol_uhf_atom(Z, m) for Z, m in atom_specs)
                 - mol_rhf(zs, pos))
        e_gapw = (sum(gapw_atom(Z, m, L, N) for Z, m in atom_specs)
                  - gapw_mol(zs, pos, L, N))
        err = e_gapw - e_ref
        assert abs(err) < 0.0025, (
            f"{name}: atomization delta {1000 * err:+.2f} mHa "
            f"exceeds 2.5 mHa"
        )


def test_overlapping_pruned_spheres_fail_closed():
    """Two pruned atoms closer than their combined sphere radii must
    refuse (the flat-region experiment broke the identity at Ha level
    when one atom's fit reached another's core)."""
    c = _L / 2
    # Two O atoms 1.9 bohr apart: both pruned, spheres ~1.6 bohr each.
    pos = [[c, c, c], [c + 1.9, c, c]]
    system = core.PeriodicSystem()
    system.dim = 3
    system.lattice = np.eye(3) * _L
    system.unit_cell = [core.Atom(8, pos[0]), core.Atom(8, pos[1])]
    mol = vq.Molecule([vq.Atom(8, pos[0]), vq.Atom(8, pos[1])], 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")
    soft = softened_basis(basis, system)
    from vibeqc.periodic_gapw_atomic_grid import AtomicRadialGrid

    grids = {
        0: AtomicRadialGrid.from_element(np.asarray(pos[0]), 8,
                                         n_radial=40, lebedev_order=17),
        1: AtomicRadialGrid.from_element(np.asarray(pos[1]), 8,
                                         n_radial=40, lebedev_order=17),
    }
    with pytest.raises(NotImplementedError, match="overlapping pruned"):
        build_one_centre_expansions(basis, soft, system, grids)
