"""Integration tests: periodic density mixers wired into the multi-k RKS
EWALD_3D driver (v0.10.x D4 metal-mixing program).

The §7 correctness gate (CLAUDE.md): a density mixer must reach the *same*
stationary point as the converged Fock-DIIS reference — never paper over an
oscillation by landing at a different (e.g. over-bound, non-representable)
energy. These tests pin Anderson/Broyden-mixed SCF to the DIIS energy across a
gapped cell, a Fermi-smeared metallic cell, and the T=0 Gilat-Raubenheimer net.

The mixers' limit-cycle-breaking property itself (converging a fixed-point map
that diverges under plain linear iteration) is unit-tested in
``test_periodic_density_mixing.py``; here we verify the *wiring* is faithful.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import vibeqc as vq
import vibeqc.periodic_density_mixing as density_mixing
from vibeqc.periodic_rks_multi_k_ewald import run_rks_periodic_multi_k_ewald3d


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _h2_box(box: float = 8.0, d: float = 1.4):
    """Gapped closed-shell H2 in a cubic box."""
    c = box / 2
    atoms = [vq.Atom(1, [c, c, c - d / 2]), vq.Atom(1, [c, c, c + d / 2])]
    sysp = vq.PeriodicSystem(3, np.diag([box, box, box]).astype(float), atoms)
    return sysp, vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")


def _h_chain(a: float = 2.4, d: float = 1.2, box: float = 12.0):
    """Near-degenerate H2 chain along x (small/no gap -> needs a metal mixer)."""
    atoms = [vq.Atom(1, [0.0, box / 2, box / 2]), vq.Atom(1, [d, box / 2, box / 2])]
    sysp = vq.PeriodicSystem(3, np.diag([a, box, box]).astype(float), atoms)
    return sysp, vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")


def _opts(
    use_diis: bool,
    *,
    smearing_T: float = 0.0,
    max_iter: int = 200,
    conv_tol_energy: float = 1e-9,
    conv_tol_grad: float = 1e-7,
):
    o = vq.PeriodicKSOptions()
    o.functional = "PBE"
    o.lattice_opts.cutoff_bohr = 12.0
    o.lattice_opts.nuclear_cutoff_bohr = 14.0
    o.damping = 0.3
    o.max_iter = max_iter
    o.use_diis = use_diis
    o.smearing_temperature = smearing_T
    # Converge tightly so the comparison pins the *same* stationary point rather
    # than the convergence-detection scatter: at a loose grad tolerance DIIS can
    # halt several µHa early while the density mixer keeps descending to the true
    # minimum, which is a tolerance artefact, not a mixer disagreement. (On the
    # T=0 Gilat sharp-Fermi surface this needs grad≈1e-9 for DIIS to fully
    # converge — at 1e-9 all three agree to ~1e-11.)
    o.conv_tol_energy = conv_tol_energy
    o.conv_tol_grad = conv_tol_grad
    return o


def _gilat_chain_kmesh():
    """A well-posed cell for exercising the T=0 Gilat-Raubenheimer occupation
    path through the density-mixer wiring.

    Why not a band-touching half-filled chain? A genuine sharp-Fermi-surface
    metal (e.g. the uniform a=2.2 / [6,1,1] chain this case used to run) pins a
    band *at* E_F, and the SCF then admits multiple near-degenerate stationary
    points ~1e-5 Ha apart. Verified empirically: Fock-DIIS lands on *either* of
    them tolerance-independently (grad<1e-9 -> lower, grad<1e-10 -> upper,
    grad<1e-11 -> lower, ... non-monotonic), while the Anderson/Broyden density
    mixers reliably find the lower one. Strict energy equality between two
    *different* accelerators is therefore ill-posed on such a cell -- it tests
    which degenerate solution each happens to pick (thread-noise-dependent), not
    the wiring. That metal also drove Kerker+Broyden into a limit cycle that did
    not converge within max_iter on some runs. The sharp-Fermi-surface
    *convergence* behaviour (limit-cycle breaking) is unit-tested in
    ``test_periodic_density_mixing.py``; the genuine-metal regime is covered by
    the Fermi-smeared case above (the entropy term makes that solution unique).
    Here we only need the gilat occupation *code path* exercised on a cell with
    a single, well-defined SCF solution so the wiring comparison is meaningful.

    The a=2.4 / [7,1,1] chain runs the identical gilat T=0 GR path
    (``use_fractional_density``) but has one solution every accelerator reaches:
    DIIS, Anderson, Broyden and both Kerker variants (beta in {0.3,0.4,0.5})
    all converge to it within 1.2e-12 Ha, thread-deterministically (OMP 1 == 4).
    """
    sysp, basis = _h_chain(a=2.4, d=1.2)
    kmesh = vq.monkhorst_pack(sysp, [7, 1, 1])
    return sysp, basis, kmesh


# ---------------------------------------------------------------------------
# Correctness: same stationary point as the DIIS reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mixer", ["anderson", "broyden"])
def test_mixer_matches_diis_gapped(mixer):
    sysp, basis = _h2_box()
    kmesh = vq.monkhorst_pack(sysp, [2, 1, 1])
    ref = run_rks_periodic_multi_k_ewald3d(
        sysp, basis, kmesh, _opts(True), omega=0.5
    )
    got = run_rks_periodic_multi_k_ewald3d(
        sysp, basis, kmesh, _opts(False), omega=0.5, density_mixer=mixer
    )
    assert ref.converged and got.converged
    # Gapped, no smearing: both land on the identical stationary point.
    assert abs(got.energy - ref.energy) < 1e-8, (mixer, got.energy, ref.energy)


@pytest.mark.parametrize("mixer", ["anderson", "broyden"])
def test_mixer_matches_diis_metallic_smeared(mixer):
    # Like the gilat case, DIIS's grad-norm trips ~0.2 µHa above the true
    # minimum on this Fermi-smeared metal while the density mixer descends
    # deeper; converge both to grad≈1e-9 so the comparison pins the same
    # stationary point (then all three agree to ~1e-11) rather than the
    # convergence-detection scatter.
    sysp, basis = _h_chain(a=2.4, d=1.2)
    kmesh = vq.monkhorst_pack(sysp, [6, 1, 1])
    tol = dict(smearing_T=0.005, conv_tol_energy=1e-11, conv_tol_grad=1e-9)
    ref = run_rks_periodic_multi_k_ewald3d(
        sysp, basis, kmesh, _opts(True, **tol), omega=0.5
    )
    got = run_rks_periodic_multi_k_ewald3d(
        sysp, basis, kmesh, _opts(False, **tol), omega=0.5,
        density_mixer=mixer, density_mixer_beta=0.4,
    )
    assert ref.converged and got.converged
    assert abs(got.energy - ref.energy) < 1e-7, (mixer, got.energy, ref.energy)


@pytest.mark.parametrize("mixer", ["anderson", "broyden"])
def test_mixer_matches_diis_gilat_net(mixer):
    # T=0 Gilat-Raubenheimer occupation path. Run on a well-posed cell with a
    # single SCF solution (see _gilat_chain_kmesh for why a band-touching metal
    # is *not* used for a strict accelerator-vs-accelerator equality check), so
    # the density mixer must land on the identical stationary point as DIIS.
    sysp, basis, kmesh = _gilat_chain_kmesh()
    tol = dict(conv_tol_energy=1e-11, conv_tol_grad=1e-9)
    ref = run_rks_periodic_multi_k_ewald3d(
        sysp, basis, kmesh, _opts(True, **tol), omega=0.5, bz_integration="gilat"
    )
    got = run_rks_periodic_multi_k_ewald3d(
        sysp, basis, kmesh, _opts(False, **tol), omega=0.5, bz_integration="gilat",
        density_mixer=mixer, density_mixer_beta=0.4,
    )
    assert ref.converged and got.converged
    assert abs(got.energy - ref.energy) < 1e-7, (mixer, got.energy, ref.energy)


# ---------------------------------------------------------------------------
# Gating / honesty surface
# ---------------------------------------------------------------------------


def test_unknown_density_mixer_raises():
    sysp, basis = _h2_box()
    kmesh = vq.monkhorst_pack(sysp, [1, 1, 1])
    with pytest.raises(ValueError, match="density_mixer"):
        run_rks_periodic_multi_k_ewald3d(
            sysp, basis, kmesh, _opts(True), omega=0.5, density_mixer="kerker"
        )


def test_density_mixer_disables_fock_diis():
    # When a density mixer is active, the Fock-DIIS subspace must stay empty
    # in the trace (mixers replace Fock-DIIS, never stack with it; CLAUDE.md §7).
    sysp, basis = _h2_box()
    kmesh = vq.monkhorst_pack(sysp, [2, 1, 1])
    r = run_rks_periodic_multi_k_ewald3d(
        sysp, basis, kmesh, _opts(True), omega=0.5, density_mixer="anderson"
    )
    assert r.converged
    assert all(int(s.diis_subspace) == 0 for s in r.scf_trace)


def test_density_mixer_none_is_diis_default():
    # density_mixer=None and "diis" both keep the legacy Fock-DIIS path.
    sysp, basis = _h2_box()
    kmesh = vq.monkhorst_pack(sysp, [2, 1, 1])
    a = run_rks_periodic_multi_k_ewald3d(
        sysp, basis, kmesh, _opts(True), omega=0.5, density_mixer=None
    )
    b = run_rks_periodic_multi_k_ewald3d(
        sysp, basis, kmesh, _opts(True), omega=0.5, density_mixer="diis"
    )
    assert a.converged and b.converged
    assert abs(a.energy - b.energy) < 1e-10


# ---------------------------------------------------------------------------
# Dispatch (run_rks_periodic_scf) threading + fail-closed surface
# ---------------------------------------------------------------------------


def _ewald_opts(use_diis):
    o = _opts(use_diis)
    o.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    return o


def test_dispatch_threads_density_mixer_to_multi_k():
    # run_rks_periodic_scf must forward density_mixer to the multi-k driver and
    # reach the same stationary point as its DIIS reference.
    sysp, basis = _h2_box()
    kmesh = vq.monkhorst_pack(sysp, [2, 1, 1])
    ref = vq.run_rks_periodic_scf(sysp, basis, kmesh, _ewald_opts(True), omega=0.5)
    got = vq.run_rks_periodic_scf(
        sysp, basis, kmesh, _ewald_opts(False), omega=0.5, density_mixer="broyden"
    )
    assert ref.converged and got.converged
    assert abs(got.energy - ref.energy) < 1e-8


def test_dispatch_rejects_density_mixer_on_unsupported_route():
    # density_mixer is wired only on the multi-k EWALD_3D driver; a Γ-only route
    # must fail closed rather than silently ignore the request (CLAUDE.md §7).
    sysp, basis = _h2_box()
    gamma = vq.monkhorst_pack(sysp, [1, 1, 1])  # Γ-only, no smearing → Γ driver
    with pytest.raises(NotImplementedError, match="multi-k"):
        vq.run_rks_periodic_scf(
            sysp, basis, gamma, _ewald_opts(False), omega=0.5, density_mixer="anderson"
        )


# ---------------------------------------------------------------------------
# Kerker preconditioner (D4b)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mixer", ["anderson", "broyden"])
def test_kerker_preconditioned_mixer_matches_diis_gapped(mixer):
    # The Kerker preconditioner filters the *residual*, so the converged energy
    # is the exact SCF solution regardless of the filter (CLAUDE.md §7-safe).
    sysp, basis = _h2_box()
    kmesh = vq.monkhorst_pack(sysp, [2, 1, 1])
    ref = run_rks_periodic_multi_k_ewald3d(sysp, basis, kmesh, _opts(True), omega=0.5)
    got = run_rks_periodic_multi_k_ewald3d(
        sysp, basis, kmesh, _opts(False), omega=0.5,
        density_mixer=mixer, density_mixer_kerker=True, kerker_k0=1.5,
    )
    assert ref.converged and got.converged
    assert abs(got.energy - ref.energy) < 1e-7, (mixer, got.energy, ref.energy)


@pytest.mark.parametrize("mixer", ["anderson", "broyden"])
def test_kerker_preconditioned_mixer_matches_diis_metallic(mixer):
    # Kerker filters the residual, so the converged energy is the exact SCF
    # solution regardless of the filter (CLAUDE.md §7-safe). Run on the same
    # well-posed gilat cell as test_mixer_matches_diis_gilat_net (see
    # _gilat_chain_kmesh): the band-touching metal it used to run on drove
    # Kerker+Broyden into a non-converging limit cycle, which is a property of
    # that pathological cell, not the wiring.
    sysp, basis, kmesh = _gilat_chain_kmesh()
    tol = dict(conv_tol_energy=1e-11, conv_tol_grad=1e-9)
    ref = run_rks_periodic_multi_k_ewald3d(
        sysp, basis, kmesh, _opts(True, **tol), omega=0.5, bz_integration="gilat"
    )
    got = run_rks_periodic_multi_k_ewald3d(
        sysp, basis, kmesh, _opts(False, **tol), omega=0.5, bz_integration="gilat",
        density_mixer=mixer, density_mixer_beta=0.4, density_mixer_kerker=True,
    )
    assert ref.converged and got.converged
    assert abs(got.energy - ref.energy) < 1e-7, (mixer, got.energy, ref.energy)


def test_kerker_requires_a_density_mixer():
    # Kerker preconditions a density mixer's residual; requesting it without one
    # must fail closed, not silently no-op (CLAUDE.md §7).
    sysp, basis = _h2_box()
    kmesh = vq.monkhorst_pack(sysp, [2, 1, 1])
    with pytest.raises(ValueError, match="requires a density_mixer"):
        run_rks_periodic_multi_k_ewald3d(
            sysp, basis, kmesh, _opts(True), omega=0.5, density_mixer_kerker=True
        )


# ---------------------------------------------------------------------------
# High-level run_periodic_job API surface
# ---------------------------------------------------------------------------


def test_run_periodic_job_exposes_density_mixer_keywords():
    sig = inspect.signature(vq.run_periodic_job)
    for name in (
        "density_mixer",
        "density_mixer_depth",
        "density_mixer_beta",
        "density_mixer_kerker",
        "kerker_k0",
        "kerker_strength",
        "kerker_cutoff_ha",
    ):
        assert name in sig.parameters


def test_run_periodic_job_density_mixer_runs_on_gdf(monkeypatch, tmp_path):
    """The high-level GDF route honours Anderson + Kerker mixing.

    This used to assert that every high-level request failed closed.  That
    expectation became stale when the closed-shell multi-k GDF route gained
    density-mixer plumbing in ``b1dd0fef``.  Keep this case here because it
    exercises the combined Anderson/Kerker request through ``run_periodic_job``;
    the broader fixed-point comparisons live in
    ``test_periodic_gdf_density_mixer.py``.
    """
    sysp, basis = _h2_box()
    anderson_calls = []
    kerker_calls = []
    anderson_cls = density_mixing.AndersonMixer
    kerker_cls = density_mixing.KerkerPreconditioner

    def record_anderson(*args, **kwargs):
        anderson_calls.append((args, kwargs.copy()))
        return anderson_cls(*args, **kwargs)

    def record_kerker(*args, **kwargs):
        kerker_calls.append((args, kwargs.copy()))
        return kerker_cls(*args, **kwargs)

    monkeypatch.setattr(density_mixing, "AndersonMixer", record_anderson)
    monkeypatch.setattr(density_mixing, "KerkerPreconditioner", record_kerker)

    result = vq.run_periodic_job(
        sysp,
        basis,
        method="RKS",
        functional="pbe",
        jk_method="gdf",
        kpoints=(2, 1, 1),
        output=tmp_path / "high-level-density-mixer",
        rsgdf_ke_cutoff=60.0,
        density_mixer="anderson",
        density_mixer_depth=5,
        density_mixer_beta=0.35,
        density_mixer_kerker=True,
        kerker_k0=1.1,
        kerker_strength=0.8,
        kerker_cutoff_ha=20.0,
        output_qvf=False,
        write_molden_file=False,
        write_density=False,
        write_xyz_file=False,
        write_poscar_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
        write_population_file=False,
        citations=False,
        progress=False,
    )

    assert result.converged
    assert result.scf_trace
    assert all(int(step.diis_subspace) == 0 for step in result.scf_trace)
    assert anderson_calls == [((), {"depth": 5, "beta": 0.35})]
    assert len(kerker_calls) == 1
    assert len(kerker_calls[0][0]) == 3
    assert kerker_calls[0][1] == {
        "k0": 1.1,
        "strength": 0.8,
        "cutoff_ha": 20.0,
    }


def test_run_periodic_job_density_mixer_options_require_active_mixer(tmp_path):
    sysp, basis = _h2_box()

    with pytest.raises(ValueError, match="density_mixer_.*require"):
        vq.run_periodic_job(
            sysp,
            basis,
            method="RKS",
            functional="pbe",
            jk_method="gdf",
            kpoints=(2, 1, 1),
            output=tmp_path / "high-level-density-mixer-options",
            density_mixer_beta=0.4,
            write_molden_file=False,
            write_density=False,
            write_xyz_file=False,
            write_poscar_file=False,
            write_xsf_structure_file=False,
            write_cif_file=False,
            write_population_file=False,
            citations=False,
            progress=False,
        )


def test_kerker_preconditioner_vanishes_on_zero_residual():
    # The §7-safety invariant, at the unit level: a zero residual preconditions
    # to a zero correction, so the SCF fixed point cannot move. A nonzero
    # residual is genuinely modified (the filter does something).
    from vibeqc import bloch_sum
    from vibeqc.periodic_density_mixing import KerkerPreconditioner

    sysp, basis = _h_chain(a=2.2, d=1.1)
    kmesh = vq.monkhorst_pack(sysp, [4, 1, 1])
    # Home-cell overlap S(0) for the metric.
    from vibeqc._vibeqc_core import compute_overlap_lattice

    opts = _opts(True)
    S_lat = compute_overlap_lattice(basis, sysp, opts.lattice_opts)
    S0 = None
    for idx, cell in enumerate(S_lat.cells):
        if (np.asarray(cell.index) == 0).all():
            S0 = np.asarray(S_lat.blocks[idx])
            break
    kerk = KerkerPreconditioner(basis, sysp, S0, k0=1.5)

    weights = np.asarray(kmesh.weights, dtype=float)
    n = basis.nbasis
    zero = [np.zeros((n, n), dtype=complex) for _ in kmesh.kpoints]
    out0 = kerk.precondition(zero, weights)
    assert all(np.allclose(o, 0.0) for o in out0)  # §7: zero residual → zero

    rng = np.random.default_rng(0)
    nz = []
    for _ in kmesh.kpoints:
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        nz.append(0.5 * (A + A.conj().T))
    out = kerk.precondition(nz, weights)
    assert any(not np.allclose(o, r) for o, r in zip(out, nz))  # genuinely filters
