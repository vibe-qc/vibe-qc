"""Multi-k GPW UKS/ROKS pure-DFT coverage.

``run_periodic_uks_gpw_multi_k`` mirrors the closed-shell multi-k GPW
driver with two spin channels: molecular-limit cells fold each spin density
to a Gamma-summed density matrix; compact crystals build per-spin Bloch
densities and project ``V_H + v_eff_s`` per k onto Bloch AOs. LDA/GGA and
meta-GGA are supported (per-spin Bloch kinetic-energy density with the
generalised-KS ``½∫v_tau ∇χ·∇χ`` Fock block); hybrids raise up-front.

Pins:

* **Γ-mesh parity** -- a (1,1,1) Monkhorst-Pack mesh reproduces the Γ-only
  open-shell GPW driver's converged energy on the same grid (both regimes
  collapse to the same Gamma physics);
* **singlet == RKS parity** -- for a closed-shell cell (n_alpha == n_beta)
  the UKS multi-k energy matches the RKS multi-k energy on both the
  molecular-limit and the compact Bloch regimes, for GGA *and* meta-GGA;
* **doublet molecular-limit multi-k** converges to a finite, sane energy;
* hybrids fail closed; stiffness-dominated meta-GGAs (TPSS) fail closed.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import _vibeqc_core as core
from vibeqc.periodic_gapw_grid import GAPWExperimentalWarning, PlaneWaveGrid
from vibeqc.periodic_gapw_j import run_periodic_rks_gpw_multi_k
from vibeqc.periodic_gapw_open_shell import (
    run_periodic_roks_gpw_multi_k,
    run_periodic_uks_gpw,
    run_periodic_uks_gpw_multi_k,
)

warnings.simplefilter("ignore", GAPWExperimentalWarning)


@pytest.mark.parametrize("route", ["gpw-rks", "gpw-uks", "gpw-roks", "gapw-rks"])
@pytest.mark.parametrize("length", [6., 12.])
def test_multik_grid_shared_solve_preserves_frames_and_occupations(monkeypatch, route, length):
    """Compare complete physical iterations in compact and vacuum regimes."""
    from types import SimpleNamespace
    import scipy.linalg as sla
    from vibeqc.periodic_gapw_augment import run_periodic_rks_gapw_multi_k
    from vibeqc.periodic_gapw_j import multik_one_electron_lattice_options
    from vibeqc.solvers import _scf

    system, basis, _ = _h2_singlet_system(length)
    grid = PlaneWaveGrid(system.lattice, 18, 18, 18)
    kmesh = core.monkhorst_pack(system, [3, 1, 1])
    drivers = {"gpw-rks": run_periodic_rks_gpw_multi_k,
               "gpw-uks": run_periodic_uks_gpw_multi_k,
               "gpw-roks": run_periodic_roks_gpw_multi_k,
               "gapw-rks": run_periodic_rks_gapw_multi_k}
    kwargs = dict(functional="lda", grid=grid, quiet=True, max_iter=80)
    if route == "gapw-rks":
        kwargs.update(n_radial=24, lebedev_order=17)
    original = _scf.solve_dense_fock
    frames = set()
    channels = set()

    def checked(fock, **kw):
        result = original(fock, **kw)
        metadata = result.operator_metadata
        assert metadata.kind == "ks"
        assert metadata.generation >= 1
        assert np.iscomplexobj(fock)
        frames.add(metadata.basis_id)
        channels.add((metadata.spin, metadata.kpoint))
        if kw.get("X") is None:
            expected = sla.eigh(fock, eigvals_only=True)
            assert result.residual_frame == "orthogonal"
        else:
            expected = sla.eigh(fock, kw["overlap"], eigvals_only=True)
            assert result.residual_frame == "original"
        np.testing.assert_allclose(result.eigenvalues, expected, atol=1e-10, rtol=0)
        assert max(result.residuals) < 1e-7
        return result

    monkeypatch.setattr(_scf, "solve_dense_fock", checked)
    if route == "gapw-rks" and length == 6.:
        # H2 has no active core augmentation. Its compact Gamma fold is
        # invalid; sharing a solver must not bypass the physical guard.
        with pytest.raises(NotImplementedError, match="Gamma-folded density"):
            drivers[route](system, basis, kmesh, **kwargs)
        assert not frames
        return
    result = drivers[route](system, basis, kmesh, **kwargs)
    assert result.converged
    projected = route in ("gpw-uks", "gpw-roks") or (route == "gpw-rks" and length == 6.)
    assert frames == ({"orthogonal-scf"} if projected else {"ao-scf"})
    spins = ("alpha", "beta") if route == "gpw-uks" else (
        ("restricted-open",) if route == "gpw-roks" else ("restricted",))
    assert channels == {(spin, tuple(k)) for spin in spins for k in kmesh.kpoints}

    # Re-run with precisely the old coordinate transformation and NumPy solve.
    def legacy(fock, **kw):
        x = kw.get("X")
        matrix = fock if x is None else x.conj().T @ fock @ x
        e, y = np.linalg.eigh(matrix)
        return SimpleNamespace(eigenvalues=e, eigenvectors=y if x is None else x @ y)

    monkeypatch.setattr(_scf, "solve_dense_fock", legacy)
    reference = drivers[route](system, basis, kmesh, **kwargs)
    assert reference.converged
    assert result.energy == pytest.approx(reference.energy, abs=1e-8)
    spinful = route in ("gpw-uks", "gpw-roks")
    density_fields = ("density_alpha", "density_beta") if spinful else ("density",)
    for field in density_fields:
        np.testing.assert_allclose(getattr(result, field), getattr(reference, field), atol=1e-6, rtol=0)

    overlap_lattice = core.compute_overlap_lattice(
        basis, system, multik_one_electron_lattice_options(basis, system))
    if spinful:
        pairs = [(result.mo_coeffs_alpha_k, result.occupations_alpha_k, 1.),
                 (result.mo_coeffs_beta_k, result.occupations_beta_k, 1.)]
    else:
        pairs = [(result.mo_coeffs_k, result.occupations_k, 2.)]
    for coefficients, occupations, electrons in pairs:
        charge = 0.
        for c, occ, k, weight in zip(coefficients, occupations, kmesh.kpoints, kmesh.weights):
            s = np.asarray(core.bloch_sum(overlap_lattice, k))
            np.testing.assert_allclose(c.conj().T @ s @ c, np.eye(c.shape[1]), atol=1e-9, rtol=0)
            density = (c * np.asarray(occ)) @ c.conj().T
            charge += weight * np.trace(density @ s).real
        assert charge == pytest.approx(electrons, abs=1e-8)


def _h_doublet_system(L: float = 12.0):
    sysp = core.PeriodicSystem()
    sysp.dim = 3
    sysp.lattice = np.eye(3) * L
    sysp.unit_cell = [core.Atom(1, [L / 2, L / 2, L / 2])]
    sysp.multiplicity = 2
    mol = vq.Molecule(list(sysp.unit_cell), 0, 2)
    basis = vq.BasisSet(mol, "sto-3g")
    grid = PlaneWaveGrid(np.eye(3) * L, 24, 24, 24)
    return sysp, basis, grid


def _h2_singlet_system(L: float = 12.0):
    sysp = core.PeriodicSystem()
    sysp.dim = 3
    sysp.lattice = np.eye(3) * L
    sysp.unit_cell = [
        core.Atom(1, [L / 2 - 0.7, L / 2, L / 2]),
        core.Atom(1, [L / 2 + 0.7, L / 2, L / 2]),
    ]
    mol = vq.Molecule(list(sysp.unit_cell), 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")
    grid = PlaneWaveGrid(np.eye(3) * L, 24, 24, 24)
    return sysp, basis, grid


def _si_primitive_system(a: float = 5.13155129):
    sysp = core.PeriodicSystem()
    sysp.dim = 3
    sysp.lattice = np.array(
        [[0.0, a, a], [a, 0.0, a], [a, a, 0.0]], dtype=float
    )
    sysp.unit_cell = [
        core.Atom(14, [0.0, 0.0, 0.0]),
        core.Atom(14, [0.5 * a, 0.5 * a, 0.5 * a]),
    ]
    mol = vq.Molecule(list(sysp.unit_cell), 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")
    return sysp, basis


def _li_doublet_compact_system(L: float = 6.0):
    sysp = core.PeriodicSystem()
    sysp.dim = 3
    sysp.lattice = np.eye(3) * L
    sysp.unit_cell = [core.Atom(3, [L / 2, L / 2, L / 2])]
    sysp.multiplicity = 2
    mol = vq.Molecule(list(sysp.unit_cell), 0, 2)
    basis = vq.BasisSet(mol, "sto-3g")
    grid = PlaneWaveGrid(np.eye(3) * L, 16, 16, 16)
    return sysp, basis, grid


def test_roks_multik_one_electron_matches_uks_and_shares_orbitals():
    """The one-electron limit reduces to UKS while retaining one spatial
    orbital set at every k point."""
    sysp, basis, grid = _h_doublet_system()
    kmesh = core.monkhorst_pack(sysp, [1, 1, 2])
    uks = run_periodic_uks_gpw_multi_k(
        sysp, basis, kmesh, functional="lda", grid=grid, quiet=True,
        conv_tol_energy=1e-9, conv_tol_density=1e-7,
    )
    roks = run_periodic_roks_gpw_multi_k(
        sysp, basis, kmesh, functional="lda", grid=grid, quiet=True,
        conv_tol_energy=1e-9, conv_tol_density=1e-7,
    )
    assert uks.converged and roks.converged
    assert roks.method == "roks"
    assert roks.energy == pytest.approx(uks.energy, abs=1e-10)
    assert all(
        np.allclose(Ca, Cb, atol=1e-12)
        for Ca, Cb in zip(roks.mo_coeffs_alpha_k, roks.mo_coeffs_beta_k)
    )


def test_roks_multik_singlet_matches_rks():
    """The closed-shell limit reduces to multi-k RKS on the same grid."""
    sysp, basis, grid = _h2_singlet_system()
    kmesh = core.monkhorst_pack(sysp, [1, 1, 2])
    rks = run_periodic_rks_gpw_multi_k(
        sysp, basis, kmesh, functional="pbe", grid=grid, quiet=True,
        conv_tol_energy=1e-9, conv_tol_density=1e-7,
    )
    roks = run_periodic_roks_gpw_multi_k(
        sysp, basis, kmesh, functional="pbe", grid=grid, quiet=True,
        conv_tol_energy=1e-9, conv_tol_density=1e-7,
    )
    assert rks.converged and roks.converged
    assert roks.energy == pytest.approx(rks.energy, abs=5e-6)


@pytest.mark.parametrize("length", [6., 12.])
@pytest.mark.parametrize("functional", ["lda", "pbe", "r2scan"])
@pytest.mark.parametrize("max_iter", [0, 1])
def test_uks_multik_terminal_energy_uses_returned_density(length, functional, max_iter, record_property):
    """A capped spin-polarized result must evaluate one density, also at NOITER.

    Independently contract the FFT Hartree energy and evaluate XC on the
    returned density. A diffuse, fractional READ seed makes the first SCF
    update large enough to expose stale nonlinear terms in either regime.
    """
    import vibeqc.periodic_gapw_j as gj
    import vibeqc.periodic_gapw_open_shell as go

    system, basis, grid = _li_doublet_compact_system(length)
    mesh = core.monkhorst_pack(system, [1, 1, 2])
    weights = np.asarray(mesh.weights)
    overlap = core.compute_overlap_lattice(
        basis, system, gj.multik_one_electron_lattice_options(basis, system))
    seed = ([], [])
    for k in mesh.kpoints:
        s = np.asarray(core.bloch_sum(overlap, k))
        for blocks, count in zip(seed, (2., 1.)):
            blocks.append(count / basis.nbasis * np.linalg.inv(s))

    result = run_periodic_uks_gpw_multi_k(
        system, basis, mesh, functional=functional, grid=grid, quiet=True,
        max_iter=max_iter, initial_guess="READ", initial_density_k=seed,
    )
    assert not result.converged and result.n_iter == max_iter
    if max_iter == 0:
        density_k = seed
        assert result.scf_trace == ()
    else:
        density_k = tuple([
            (c * np.asarray(o)) @ c.conj().T for c, o in zip(coefficients, occupations)
        ] for coefficients, occupations in (
            (result.mo_coeffs_alpha_k, result.occupations_alpha_k),
            (result.mo_coeffs_beta_k, result.occupations_beta_k),
        ))
    for blocks, returned in zip(density_k, (result.density_alpha, result.density_beta)):
        folded = sum(w * d.real for w, d in zip(weights, blocks))
        np.testing.assert_allclose(returned, folded, atol=1e-12, rtol=0)

    cache = gj.build_gpw_collocation_cache(basis, grid)
    func = core.Functional(functional, 2)
    if gj._multik_gpw_is_molecular_limit(system):
        rho_a, rho_b = [gj.collocate_density_on_grid(basis, d, grid, cache=cache)
                        for d in (result.density_alpha, result.density_beta)]
        e_xc, *_ = go._evaluate_xc_polarised_on_grid(
            rho_a, rho_b, grid, func, basis=basis,
            density_matrix_alpha=result.density_alpha,
            density_matrix_beta=result.density_beta, cache=cache,
        )
    else:
        points = grid.cartesian_coords().reshape(-1, 3)
        translations = np.asarray([c.r_cart for c in core.direct_lattice_cells(system, 25.)])
        chi = [gj.bloch_ao_on_grid(basis, points, k, translations) for k in mesh.kpoints]
        rho_a, rho_b = [gj.collocate_bloch_density_on_grid(chi, ds, weights, grid)
                        for ds in density_k]
        tau = [None, None]
        if func.kind == core.XCKind.MGGA:
            tau = [gj.compute_bloch_kinetic_energy_density(
                chi, ds, weights, mesh.kpoints, points, grid, recip=cache.recip,
            ) for ds in density_k]
        e_xc, *_ = go._xc_effective_potentials_polarised_grid(
            rho_a, rho_b, grid, func, tau_alpha=tau[0], tau_beta=tau[1], cache=cache,
        )
    rho = rho_a + rho_b
    potential = core.solve_poisson_coulomb(rho, grid.lattice_bohr)
    e_hartree = .5 * np.sum(rho * potential) * grid.voxel_volume_bohr3
    expected = (result.breakdown.e_kinetic + result.breakdown.e_nuclear_attraction
                + result.breakdown.e_nuclear_repulsion + e_hartree + e_xc)
    record_property("energy", result.energy)
    record_property("density_energy", expected)
    record_property("energy_error", result.energy - expected)
    assert result.breakdown.e_hartree == pytest.approx(e_hartree, abs=1e-10, rel=0)
    assert result.breakdown.e_xc == pytest.approx(e_xc, abs=1e-10, rel=0)
    assert result.energy == pytest.approx(expected, abs=1e-10, rel=0)
    if result.scf_trace:
        assert result.scf_trace[-1]["energy"] == pytest.approx(result.energy, abs=1e-12, rel=0)
        assert result.scf_trace[-1]["e_xc"] == pytest.approx(e_xc, abs=1e-12, rel=0)
        previous = result.scf_trace[-2]["energy"] if len(result.scf_trace) > 1 else 0.
        assert result.scf_trace[-1]["delta_e"] == pytest.approx(result.energy - previous, abs=1e-12, rel=0)


@pytest.mark.parametrize("length", [6., 12.])
@pytest.mark.parametrize("functional", ["lda", "pbe", "r2scan"])
@pytest.mark.parametrize("reference", ["uks", "roks"])
def test_multik_iteration_energy_uses_evaluated_density(
    length, functional, reference, record_property,
):
    """Every live iteration energy is E[D_in], including its convergence dE.

    NOITER and a one-step capped run provide E[D_0] and E[D_1]. Their
    returned-density reconstruction is checked independently above. A
    two-step run must report those same energies before its final rebuild.
    Fractional spin densities make D_in and D_out appreciably different.
    """
    from vibeqc.periodic_gapw_j import multik_one_electron_lattice_options
    from vibeqc.progress import ProgressLogger

    class CaptureIterations(ProgressLogger):
        def __init__(self):
            super().__init__(verbose=0)
            self.rows = []

        def iteration(self, it, **values):
            self.rows.append(values)

    system, basis, grid = _li_doublet_compact_system(length)
    mesh = core.monkhorst_pack(system, [1, 1, 2])
    overlap = core.compute_overlap_lattice(
        basis, system, multik_one_electron_lattice_options(basis, system))
    seed = ([], [])
    for k in mesh.kpoints:
        s = np.asarray(core.bloch_sum(overlap, k))
        for blocks, count in zip(seed, (2., 1.)):
            blocks.append(count / basis.nbasis * np.linalg.inv(s))
    driver = (run_periodic_uks_gpw_multi_k if reference == "uks"
              else run_periodic_roks_gpw_multi_k)
    kwargs = dict(functional=functional, grid=grid, quiet=True,
                  initial_guess="READ", initial_density_k=seed)
    initial = driver(system, basis, mesh, max_iter=0, **kwargs)
    one_step = driver(system, basis, mesh, max_iter=1, **kwargs)
    progress = CaptureIterations()
    result = driver(system, basis, mesh, max_iter=2, progress=progress, **kwargs)
    assert len(progress.rows) == result.n_iter == 2
    assert np.linalg.norm(one_step.density_alpha - initial.density_alpha) > 1e-3
    expected = [initial.energy, one_step.energy]
    errors = [row["energy"] - energy for row, energy in zip(progress.rows, expected)]
    record_property("iteration_energy_errors", errors)
    np.testing.assert_allclose(
        [row["energy"] for row in progress.rows], expected, atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        [row["dE"] for row in progress.rows],
        [initial.energy, one_step.energy - initial.energy], atol=1e-10, rtol=0)
    assert result.scf_trace[0]["energy"] == pytest.approx(initial.energy, abs=1e-10, rel=0)
    assert result.scf_trace[0]["e_xc"] == pytest.approx(
        initial.breakdown.e_xc, abs=1e-10, rel=0)
    # The terminal row is rebuilt at D_out, preserving the returned-state
    # contract rather than relabelling a previous iteration's energy.
    assert result.scf_trace[-1]["energy"] == pytest.approx(result.energy, abs=1e-12, rel=0)
    assert result.scf_trace[-1]["delta_e"] == pytest.approx(
        result.energy - initial.energy, abs=1e-10, rel=0)


def test_roks_multik_compact_li_doublet_converges_spin_pure():
    """A compact multi-electron open shell exercises Bloch collocation and
    per-k Roothaan coupling rather than the molecular-limit shortcut."""
    sysp, basis, grid = _li_doublet_compact_system()
    kmesh = core.monkhorst_pack(sysp, [1, 1, 2])
    roks = run_periodic_roks_gpw_multi_k(
        sysp, basis, kmesh, functional="lda", grid=grid, quiet=True,
        max_iter=80,
    )
    assert roks.converged
    assert roks.n_alpha == 2 and roks.n_beta == 1
    assert -8.0 < roks.energy < -6.0
    assert roks.scf_trace[-1]["energy"] == pytest.approx(roks.energy, abs=1e-12)
    assert roks.scf_trace[-1]["e_xc"] == pytest.approx(
        roks.breakdown.e_xc, abs=1e-12
    )
    assert np.linalg.norm(roks.density_alpha - roks.density_beta) > 1e-3
    assert all(
        np.allclose(Ca, Cb, atol=1e-12)
        for Ca, Cb in zip(roks.mo_coeffs_alpha_k, roks.mo_coeffs_beta_k)
    )


def test_roks_multik_hybrid_fails_closed():
    sysp, basis, grid = _h_doublet_system()
    kmesh = core.monkhorst_pack(sysp, [1, 1, 2])
    with pytest.raises(NotImplementedError, match="hybrid"):
        run_periodic_roks_gpw_multi_k(
            sysp, basis, kmesh, functional="b3lyp", grid=grid, quiet=True,
        )


def test_uks_multik_gamma_mesh_matches_gamma_driver():
    """(1,1,1) multi-k UKS reproduces the Γ-only UKS GPW energy on the same
    grid + functional (H doublet, LDA)."""
    sysp, basis, grid = _h_doublet_system()
    gamma = run_periodic_uks_gpw(
        sysp, basis, functional="lda", grid=grid, quiet=True,
        conv_tol_energy=1e-9, conv_tol_density=1e-7,
    )
    kmesh = core.monkhorst_pack(sysp, [1, 1, 1])
    multik = run_periodic_uks_gpw_multi_k(
        sysp, basis, kmesh, functional="lda", grid=grid, quiet=True,
        conv_tol_energy=1e-9, conv_tol_density=1e-7,
    )
    assert gamma.converged and multik.converged
    assert multik.n_alpha == 1 and multik.n_beta == 0
    assert multik.energy == pytest.approx(gamma.energy, abs=5e-7)


def test_uks_multik_singlet_matches_rks_molecular_limit():
    """Molecular-limit multi-k: closed-shell H2 through the UKS driver
    (n_alpha == n_beta) matches the RKS multi-k energy."""
    sysp, basis, grid = _h2_singlet_system()
    kmesh = core.monkhorst_pack(sysp, [1, 1, 2])
    rks = run_periodic_rks_gpw_multi_k(
        sysp, basis, kmesh, functional="pbe", grid=grid, quiet=True,
        conv_tol_energy=1e-9, conv_tol_density=1e-7,
    )
    uks = run_periodic_uks_gpw_multi_k(
        sysp, basis, kmesh, functional="pbe", grid=grid, quiet=True,
        conv_tol_energy=1e-9, conv_tol_density=1e-7,
    )
    assert rks.converged and uks.converged
    assert uks.energy == pytest.approx(rks.energy, abs=5e-6)


def test_uks_multik_singlet_matches_rks_compact():
    """Compact Bloch regime: closed-shell Si primitive through the UKS
    driver matches the RKS multi-k compact energy (same Bloch density,
    per-k projection, lindep handling, and DIIS class)."""
    sysp, basis = _si_primitive_system()
    kmesh = core.monkhorst_pack(sysp, [2, 2, 2])
    rks = run_periodic_rks_gpw_multi_k(
        sysp, basis, kmesh, functional="pbe", cutoff_ha=110.0, quiet=True,
        max_iter=80,
    )
    uks = run_periodic_uks_gpw_multi_k(
        sysp, basis, kmesh, functional="pbe", cutoff_ha=110.0, quiet=True,
        max_iter=80,
    )
    assert rks.converged and uks.converged
    assert uks.energy == pytest.approx(rks.energy, abs=1e-5)


def test_uks_multik_doublet_molecular_limit_converges():
    """H-atom doublet on a (1,1,2) molecular-limit mesh converges to a
    finite, sane spin-polarised energy."""
    sysp, basis, grid = _h_doublet_system()
    kmesh = core.monkhorst_pack(sysp, [1, 1, 2])
    res = run_periodic_uks_gpw_multi_k(
        sysp, basis, kmesh, functional="lda", grid=grid, quiet=True,
    )
    assert res.converged
    assert np.isfinite(res.energy)
    # Boxed H atom LDA lands near -0.45 Ha; generous physical window.
    assert -0.7 < res.energy < -0.2
    # Spin-polarised: alpha is occupied, beta empty.
    assert float(np.trace(res.density_alpha)) > 0.5
    assert float(np.abs(res.density_beta).max()) < 1e-8


def test_uks_multik_mgga_singlet_matches_rks_molecular_limit():
    """Molecular-limit multi-k meta-GGA: closed-shell H2 through the UKS
    driver (n_alpha == n_beta) matches the RKS multi-k r2SCAN energy. The
    per-spin Gamma-summed tau + generalised-KS V_tau reduce to the
    closed-shell path."""
    sysp, basis, grid = _h2_singlet_system()
    kmesh = core.monkhorst_pack(sysp, [1, 1, 2])
    rks = run_periodic_rks_gpw_multi_k(
        sysp, basis, kmesh, functional="r2scan", grid=grid, quiet=True,
        conv_tol_energy=1e-9, conv_tol_density=1e-7,
    )
    uks = run_periodic_uks_gpw_multi_k(
        sysp, basis, kmesh, functional="r2scan", grid=grid, quiet=True,
        conv_tol_energy=1e-9, conv_tol_density=1e-7,
    )
    assert rks.converged and uks.converged
    assert uks.energy == pytest.approx(rks.energy, abs=5e-6)


def test_uks_multik_mgga_singlet_matches_rks_compact():
    """Compact Bloch regime meta-GGA: closed-shell Si primitive through the
    UKS driver matches the RKS multi-k compact r2SCAN energy (per-spin Bloch
    tau, per-k generalised-KS V_tau projection)."""
    sysp, basis = _si_primitive_system()
    kmesh = core.monkhorst_pack(sysp, [2, 2, 2])
    rks = run_periodic_rks_gpw_multi_k(
        sysp, basis, kmesh, functional="r2scan", cutoff_ha=110.0, quiet=True,
        max_iter=80,
    )
    uks = run_periodic_uks_gpw_multi_k(
        sysp, basis, kmesh, functional="r2scan", cutoff_ha=110.0, quiet=True,
        max_iter=80,
    )
    assert rks.converged and uks.converged
    assert uks.energy == pytest.approx(rks.energy, abs=1e-5)


def test_uks_multik_hybrid_fails_closed():
    sysp, basis, grid = _h2_singlet_system()
    kmesh = core.monkhorst_pack(sysp, [1, 1, 2])
    with pytest.raises(NotImplementedError, match="hybrid"):
        run_periodic_uks_gpw_multi_k(
            sysp, basis, kmesh, functional="b3lyp", grid=grid, quiet=True,
        )


def test_uks_multik_tpss_fails_closed_on_stiffness():
    """Non-self-regularising meta-GGAs (TPSS-class) blow the uniform-grid
    stiffness guard where tau dips below the von Weizsacker floor; the
    driver must fail closed rather than return a contaminated energy."""
    sysp, basis, grid = _h2_singlet_system()
    kmesh = core.monkhorst_pack(sysp, [1, 1, 2])
    with pytest.raises(NotImplementedError, match="stiffness"):
        run_periodic_uks_gpw_multi_k(
            sysp, basis, kmesh, functional="tpss", grid=grid, quiet=True,
            conv_tol_energy=1e-9, conv_tol_density=1e-7,
        )
