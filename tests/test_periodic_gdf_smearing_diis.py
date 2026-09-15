"""Smeared-SCF + DIIS zero-commutator trap regression tests.

On symmetry-locked fixtures whose FDS-SDF commutator vanishes
identically (H2 in a cubic box / sto-3g: the sigma_g/sigma_u
eigenvectors are fixed by symmetry, so [F, D] == 0 from iteration 1 for
ANY occupations), the finite-temperature (Fermi-Dirac) SCF with DIIS
used to terminate after ~3 iterations claiming convergence (dE = 0
exactly) at a state whose occupations were NOT the Fermi filling of the
converged Fock's own eigenvalues: DIIS received zero error vectors, its
Pulay B matrix was exactly singular, the unconstrained extrapolation
weights froze the mixed Fock's eigenvalues, and the energy/commutator
convergence tests could not see the occupation inconsistency (measured
2026-07-30: stored occ_alpha [0.998955, 0.001045] vs the converged
Fock's own filling [0.999126, 0.000874] on the 2.4-bohr H2 scout;
free energy off ~1e-4 Ha, analytic-vs-FD forces off 3.3e-3).

Two-part fix pinned here:

* ``periodic_scf_accelerators._below_extrapolation_floor`` -- every
  accelerator's ``extrapolate_*`` returns the input Fock unchanged and
  records nothing when the commutator error is numerical noise on the
  Fock scale, so the plain Fock is diagonalised and the occupation
  fixed point converges exactly as with ``use_diis=False``.
* An occupation self-consistency guard in the smeared convergence
  tests of the Γ (``run_pbc_gdf_uhf`` / ``run_pbc_gdf_uks``) and
  multi-k (``run_krhf_periodic_gdf`` / ``run_kuhf_periodic_gdf``) GDF
  drivers: convergence additionally requires the stored occupations to
  be the Fermi filling of the current plain Fock's own eigenvalues
  (``vibeqc.smearing.smeared_occupation_selfconsistency_tolerance``).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla
import vibeqc as vq
from vibeqc._vibeqc_core import LatticeSumOptions
from vibeqc.periodic_scf_accelerators import (
    MultiKPeriodicSCFAccelerator,
    PeriodicSCFAccelerator,
)
from vibeqc.smearing import (
    SmearingOptions,
    apply_smearing_open_shell,
    closed_shell_periodic_occupations,
    smeared_occupation_selfconsistency_tolerance,
)

KBT = 0.05  # Ha -- the 2026-07-30 scout temperature


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lat_opts():
    opts = LatticeSumOptions()
    opts.cutoff_bohr = 12.0
    opts.nuclear_cutoff_bohr = 12.0
    return opts


def _h2_box(bond=2.4):
    """H2 at ``bond`` bohr in the 12-bohr box: the symmetry-locked
    zero-commutator fixture of the 2026-07-30 measurement."""
    box = 12.0
    c = box / 2.0
    system = vq.PeriodicSystem(3, np.eye(3) * box, [
        vq.Atom(1, [c, c, c - bond / 2.0]),
        vq.Atom(1, [c, c, c + bond / 2.0]),
    ])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _opts(use_diis):
    o = vq.PeriodicRHFOptions()
    o.lattice_opts = _lat_opts()
    o.max_iter = 200
    o.conv_tol_energy = 1e-11
    o.smearing_temperature = KBT
    o.use_diis = use_diis
    return o


def _run_gamma_uhf(use_diis):
    system, basis = _h2_box()
    with pytest.warns(UserWarning, match="apply_aft_correction=False"):
        return vq.run_pbc_gdf_uhf(
            system, basis, options=_opts(use_diis),
            aux_basis="def2-svp-jk", gdf_linear_dep_threshold=1e-9,
            exxdiv="ewald", gdf_method="compcell", compcell_eta=1.0,
            apply_aft_correction=False, progress=False,
        )


def _run_multik_krhf(use_diis):
    from vibeqc.periodic_k_gdf import run_krhf_periodic_gdf

    system, basis = _h2_box()
    return run_krhf_periodic_gdf(
        system, basis, (2, 1, 1), _opts(use_diis),
        aux_basis="def2-svp-jk", use_compcell=True,
        gdf_method="rsgdf", rsgdf_ke_cutoff=60.0,
        bz_integration="smearing", progress=False,
    )


# ---------------------------------------------------------------------------
# Accelerator zero-commutator floor (unit level)
# ---------------------------------------------------------------------------


def _accel_opts():
    o = vq.PeriodicRHFOptions()
    o.use_diis = True
    return o


def test_gamma_accelerator_skips_extrapolation_at_zero_error():
    """Zero error vectors: extrapolate_* returns the Fock unchanged and
    records nothing (the singular-B trap never forms)."""
    rng = np.random.default_rng(7)
    accel = PeriodicSCFAccelerator(_accel_opts())
    zero = np.zeros((4, 4))
    focks = []
    for _ in range(4):
        F = rng.normal(size=(4, 4))
        F = 0.5 * (F + F.T)
        focks.append(F)
        F_ex = accel.extrapolate_rhf(
            F, error=zero, density=np.eye(4), energy=-1.0,
            mo_coeffs=np.eye(4), mo_energies=np.zeros(4), n_occ=1,
        )
        assert np.array_equal(np.asarray(F_ex), F)
    assert accel.subspace_size == 0

    # A genuine error re-engages the history.
    err = 1e-3 * rng.normal(size=(4, 4))
    accel.extrapolate_rhf(
        focks[-1], error=err, density=np.eye(4), energy=-1.0,
        mo_coeffs=np.eye(4), mo_energies=np.zeros(4), n_occ=1,
    )
    assert accel.subspace_size == 1


def test_gamma_uhf_accelerator_skips_extrapolation_at_zero_error():
    rng = np.random.default_rng(11)
    accel = PeriodicSCFAccelerator(_accel_opts())
    zero = np.zeros((3, 3))
    Fa = rng.normal(size=(3, 3))
    Fa = 0.5 * (Fa + Fa.T)
    Fb = Fa + 0.01 * np.eye(3)
    Fa_ex, Fb_ex = accel.extrapolate_uhf(
        Fa, Fb, error_alpha=zero, error_beta=zero,
        density_alpha=np.eye(3), density_beta=np.eye(3), energy=-1.0,
        mo_coeffs_alpha=np.eye(3), mo_coeffs_beta=np.eye(3),
        mo_energies_alpha=np.zeros(3), mo_energies_beta=np.zeros(3),
        n_alpha=1, n_beta=1,
    )
    assert np.array_equal(np.asarray(Fa_ex), Fa)
    assert np.array_equal(np.asarray(Fb_ex), Fb)
    assert accel.subspace_size == 0


def test_multik_accelerator_skips_extrapolation_at_zero_error():
    rng = np.random.default_rng(13)
    accel = MultiKPeriodicSCFAccelerator(_accel_opts())
    zero = np.zeros((3, 3), dtype=complex)
    F_k = []
    for _ in range(2):
        F = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        F_k.append(0.5 * (F + F.conj().T))
    F_ex = accel.extrapolate_rhf(
        F_k, error_k_list=[zero, zero],
        density_k_list=[np.eye(3, dtype=complex)] * 2, energy=-1.0,
        mo_coeffs_k_list=[np.eye(3, dtype=complex)] * 2, n_occ=1,
        weights=[0.5, 0.5], cells=None, kpoints=[np.zeros(3)] * 2,
    )
    for F_in, F_out in zip(F_k, F_ex):
        assert np.array_equal(np.asarray(F_out), F_in)
    assert accel.subspace_size == 0


def test_occupation_selfconsistency_tolerance_scaling():
    """sqrt(tol_E) with a 1e-9 floor: commensurate with the Mermin
    free energy's O(df^2) stationarity in the occupations."""
    assert smeared_occupation_selfconsistency_tolerance(1e-11) == (
        pytest.approx(np.sqrt(1e-11))
    )
    assert smeared_occupation_selfconsistency_tolerance(1e-20) == 1e-9


# ---------------------------------------------------------------------------
# Driver-level regressions on the zero-commutator fixture
# ---------------------------------------------------------------------------


def test_gamma_uhf_smeared_diis_occupations_are_fock_selfconsistent():
    """Γ UHF, DIIS ON, zero-commutator fixture: the SCF must converge to
    the same occupation fixed point as plain iteration, with stored
    occupations equal to the Fermi filling of the returned Fock's own
    eigenvalues. Pre-fix this terminated at n_iter=3 with stored
    occ_alpha [0.998955, 0.001045] vs recomputed [0.999126, 0.000874]."""
    r = _run_gamma_uhf(use_diis=True)
    assert r.converged
    assert r.n_iter > 3, (
        f"n_iter={r.n_iter}: the ~3-iteration DIIS termination trap"
    )

    # Re-diagonalise the returned Fock and re-apply the Fermi filling.
    def refill(F):
        eps = sla.eigh(np.asarray(F), np.asarray(r.overlap),
                       eigvals_only=True)
        return eps

    smear = SmearingOptions.from_legacy_kwarg(KBT)
    a_res, b_res = apply_smearing_open_shell(
        [refill(r.fock_alpha)], [refill(r.fock_beta)], weights=[1.0],
        n_alpha=1, n_beta=1, smearing=smear,
    )
    for stored, own in (
        (r.occupations_alpha, a_res.occupations_per_k[0]),
        (r.occupations_beta, b_res.occupations_per_k[0]),
    ):
        assert np.max(np.abs(
            np.sort(np.asarray(stored, dtype=float))[::-1]
            - np.sort(np.asarray(own, dtype=float))[::-1]
        )) < 1e-8

    # Same fixed point as plain iteration (the use_diis=False reference).
    r0 = _run_gamma_uhf(use_diis=False)
    assert r0.converged
    assert abs(r.free_energy - r0.free_energy) < 1e-9
    assert np.max(np.abs(
        np.asarray(r.occupations_alpha) - np.asarray(r0.occupations_alpha)
    )) < 1e-8


def test_multik_krhf_smeared_diis_occupations_are_fock_selfconsistent():
    """Multi-k (2,1,1) KRHF, DIIS ON, zero-commutator fixture: same
    regression as the Γ test (pre-fix: n_iter=3, FD mismatch 1.6e-2)."""
    r = _run_multik_krhf(use_diis=True)
    assert r.converged
    assert r.n_iter > 3, (
        f"n_iter={r.n_iter}: the ~3-iteration DIIS termination trap"
    )

    eps_chk = [
        sla.eigh(np.asarray(r.fock[i]), np.asarray(r.overlap[i]),
                 eigvals_only=True)
        for i in range(len(r.fock))
    ]
    occ_chk, _, _ = closed_shell_periodic_occupations(
        eps_chk, r.kpoint_weights, 2.0, 1, KBT,
    )
    for stored, own in zip(r.occupations, occ_chk):
        assert np.max(np.abs(
            np.asarray(stored, dtype=float) - np.asarray(own, dtype=float)
        )) < 1e-8

    r0 = _run_multik_krhf(use_diis=False)
    assert r0.converged
    assert abs(r.free_energy - r0.free_energy) < 1e-9


def test_entropy_of_a_mixed_density_uses_metric_natural_occupations():
    from vibeqc.periodic_k_density import _fermi_density_entropy
    # A convex mixture of two noncommuting pure one-electron projectors.
    a = np.array([[1., 0.], [0., 0.]])
    b = np.full((2, 2), .5)
    mixed = .5 * (a + b)
    expected_values = np.array([.5 - np.sqrt(.125), .5 + np.sqrt(.125)])
    expected = -2 * np.sum(expected_values * np.log(expected_values)
                           + (1-expected_values) * np.log1p(-expected_values))
    overlap = np.diag([.3, 1.4, 4.])
    x = np.diag(1 / np.sqrt(np.diag(overlap)))[:, :2]
    density = 2 * x @ mixed @ x.T
    actual = _fermi_density_entropy([density], [overlap], [x], [1.], 2.)
    assert actual == pytest.approx(expected, rel=0, abs=2e-14)
    # The result is invariant under nonunitary AO coordinate changes too.
    change = np.array([[1., .2j, .1], [0., 1.3, -.2j], [.1, 0., .8]])
    inverse = np.linalg.inv(change)
    transformed = _fermi_density_entropy(
        [inverse @ density @ inverse.conj().T],
        [change.conj().T @ overlap @ change], [inverse @ x], [1.], 2.,
    )
    assert transformed == pytest.approx(expected, rel=0, abs=2e-14)


def test_density_entropy_refuses_nonrepresentable_density():
    from vibeqc.periodic_k_density import _fermi_density_entropy
    with pytest.raises(ValueError, match='fermionic interval'):
        _fermi_density_entropy([np.diag([1.1, -.1])], [np.eye(2)], [np.eye(2)], [1.], 1.)


def test_smeared_terminal_energy_entropy_and_returned_density_are_one_state():
    from vibeqc.periodic_k_density import _fermi_density_entropy
    from vibeqc.pbc_gdf import _canonical_orthogonalizer
    system, basis = _h2_box()
    options = _opts(True)
    options.max_iter = 2
    options.initial_guess = vq.InitialGuess.HCORE
    options.damping = .35
    result = vq.run_krhf_periodic_gdf(
        system, basis, (2, 1, 1), options, aux_basis='def2-svp-jk',
        use_compcell=True, rsgdf_ke_cutoff=20., progress=False,
    )
    x = [_canonical_orthogonalizer(s, 1e-7)[0] for s in result.overlap]
    expected = _fermi_density_entropy(result.density, result.overlap, x, result.kpoint_weights, 2.)
    assert result.entropy == pytest.approx(expected, rel=0, abs=2e-13)
    assert result.free_energy == pytest.approx(result.energy - KBT * expected, rel=0, abs=2e-14)
    assert result.scf_trace[-1].energy == pytest.approx(result.free_energy, rel=0, abs=2e-14)


@pytest.mark.parametrize('open_shell', [False, True])
@pytest.mark.parametrize('max_iter', [0, 1])
def test_smeared_nonrepresentable_restart_uses_seed_fock_before_first_result(
    open_shell, max_iter, monkeypatch,
):
    """A normalized READ guess can overfill one k point's retained space.

    Such a guess selects a Fock but cannot itself be the returned thermal
    state, even at max_iter=1. No artificial entropy or clipped density is
    accepted, and the global particle count survives the Fermi refill.
    """
    from vibeqc.periodic_k_density import _fermi_density_entropy
    from vibeqc.pbc_gdf import _canonical_orthogonalizer
    system, basis = _h2_box()
    options = _opts(True)
    options.max_iter = max_iter
    if max_iter == 0:
        import vibeqc.periodic_k_density as density_module
        def unexpected_thermal_step(*args, **kwargs):
            pytest.fail("max_iter=0 must retain the selected initial guess")
        monkeypatch.setattr(density_module, '_needs_fermi_seed_step', unexpected_thermal_step)
    options.initial_guess = vq.InitialGuess.READ
    seed = [np.diag([1., 0.]).astype(complex), np.zeros((2, 2), complex)]
    if open_shell:
        system.charge = 1
        system.multiplicity = 2
        result = vq.run_kuhf_periodic_gdf(
            system, basis, (2, 1, 1), options, aux_basis='def2-svp-jk',
            initial_density_k=(seed, [np.zeros((2, 2), complex) for _ in seed]),
            rsgdf_ke_cutoff=20., progress=False,
        )
        channels = [result.density_alpha, result.density_beta]
        populations, capacity = [1., 0.], 1.
    else:
        result = vq.run_krhf_periodic_gdf(
            system, basis, (2, 1, 1), options, aux_basis='def2-svp-jk',
            initial_density_k=seed, use_compcell=True,
            rsgdf_ke_cutoff=20., progress=False,
        )
        channels = [result.density]
        populations, capacity = [2.], 2.
    if max_iter == 0:
        assert result.n_iter == len(result.scf_trace) == 0
        assert not result.converged
        np.testing.assert_array_equal(channels[0][1], 0.)
        return
    x = [_canonical_orthogonalizer(s, 1e-7)[0] for s in result.overlap]
    expected = sum(_fermi_density_entropy(
        channel, result.overlap, x, result.kpoint_weights, capacity,
    ) for channel in channels)
    for channel, population in zip(channels, populations):
        count = sum(w * np.trace(d @ s).real for w, d, s in zip(
            result.kpoint_weights, channel, result.overlap,
        ))
        assert count == pytest.approx(population, abs=1e-10)
    assert result.n_iter == len(result.scf_trace) == 1
    assert result.entropy == pytest.approx(expected, rel=0, abs=2e-13)
    assert result.free_energy == pytest.approx(result.energy - KBT * expected, rel=0, abs=2e-14)
    assert result.scf_trace[0].energy == pytest.approx(result.free_energy, rel=0, abs=2e-14)


@pytest.mark.parametrize('ks', [False, True])
def test_gamma_smeared_entropy_uses_both_accepted_spin_densities(ks):
    from contextlib import nullcontext
    from vibeqc.pbc_gdf import _canonical_orthogonalizer
    from vibeqc.periodic_k_density import _fermi_density_entropy

    system, basis = _h2_box()
    options = vq.PeriodicKSOptions() if ks else _opts(False)
    options.lattice_opts = _lat_opts()
    options.smearing_temperature = KBT
    options.use_diis = False
    options.max_iter = 2
    options.initial_guess = vq.InitialGuess.HCORE
    options.damping = .35
    driver = vq.run_pbc_gdf_uks if ks else vq.run_pbc_gdf_uhf
    keywords = {'functional': 'pbe'} if ks else {}
    warning = nullcontext() if ks else pytest.warns(
        UserWarning, match='apply_aft_correction=False',
    )
    with warning:
        result = driver(
            system, basis, options, aux_basis='def2-svp-jk',
            gdf_method='compcell', compcell_eta=1.0,
            apply_aft_correction=False, progress=False, **keywords,
        )
    x = _canonical_orthogonalizer(result.overlap, 1e-7)[0]
    expected = sum(
        _fermi_density_entropy([density], [result.overlap], [x], [1.0], 1.0)
        for density in (result.density_alpha, result.density_beta)
    )
    assert result.entropy == pytest.approx(expected, rel=0, abs=2e-13)
    assert result.free_energy == pytest.approx(result.energy - KBT * expected, rel=0, abs=2e-14)


@pytest.mark.parametrize("weights", [[0.5, 0.5], [0.25, 0.75]])
def test_gdf_zero_temperature_spin_fill_is_global_and_matches_restricted(weights):
    from vibeqc.periodic_k_gdf import _gdf_open_shell_occupations
    from vibeqc.smearing import SmearingOptions
    from vibeqc.smearing.apply import _global_aufbau_with_mu

    # A band crossing invalidates a fixed number of occupied bands at each k.
    eps = [np.array([-3., -2., 2.]), np.array([-1., 1., 3.])]
    a, b = _gdf_open_shell_occupations(
        eps, eps, weights=weights, n_alpha=1., n_beta=1.,
        smearing=SmearingOptions(),
    )
    closed, mu = _global_aufbau_with_mu(eps, weights, 2., occ_value=2.)
    assert a.mu == b.mu == mu
    for oa, ob, oc in zip(a.occupations_per_k, b.occupations_per_k, closed):
        np.testing.assert_array_equal(oa + ob, oc)
    for channel in (a, b):
        assert sum(w * np.sum(o) for w, o in zip(
            weights, channel.occupations_per_k,
        )) == pytest.approx(1., abs=2e-15)
        assert channel.entropy == channel.free_energy_correction == 0.
    assert a.occupations_per_k[0][1] > 0.
    assert a.occupations_per_k[1][0] < 1.


def test_gdf_gamma_spin_frontier_ensemble_preserves_explicit_pattern_hold():
    from vibeqc.pbc_gdf import _open_shell_gamma_occupy
    from vibeqc.smearing import SmearingOptions
    from vibeqc.smearing.apply import _global_aufbau_with_mu

    C = np.eye(4)
    eps = np.array([-2., 0., 0., 2.])
    options = SmearingOptions()
    Da, Db, oa, ob, *_ = _open_shell_gamma_occupy(
        C, eps, C, eps, 2, 2, options,
    )
    closed, _ = _global_aufbau_with_mu([eps], [1.], 4.)
    np.testing.assert_array_equal(oa, [1., .5, .5, 0.])
    np.testing.assert_array_equal(oa, ob)
    np.testing.assert_array_equal(Da + Db, np.diag(closed[0]))
    # The hold deliberately selects an excited orbital by column order.
    order = [0, 3, 1, 2]
    Da, Db, oa, ob, *_ = _open_shell_gamma_occupy(
        C[:, order], eps[order], C[:, order], eps[order], 2, 2, options,
        preserve_column_order=True,
    )
    np.testing.assert_array_equal(Da, np.diag([1., 0., 0., 1.]))
    np.testing.assert_array_equal(Da, Db)
    np.testing.assert_array_equal(oa, [1., 1., 0., 0.])
