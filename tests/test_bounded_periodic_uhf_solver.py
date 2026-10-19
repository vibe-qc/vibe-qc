"""Independent fixed-spin, whole-k algebra for the bounded UHF leaf."""
import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_bounded_periodic_rhf_solver import _case as rhf_case, _options, _orthogonalizers


def case(k=3, n=3, alpha=2, beta=1, interaction=.006):
    data = rhf_case(k, n)
    a = np.array([data['operator'], .7 * data['operator'][:, ::-1, ::-1]])
    # Variational spin/k coupling E2=lambda/(2K^2)*(sum Tr(A D))^2.
    # The production solver knows neither A nor this factorization.
    data.update(alpha=alpha, beta=beta, operator=a,
                kernel=np.ascontiguousarray(interaction / k * np.outer(a.ravel(), a.swapaxes(-1, -2).ravel())),
                bias=np.zeros((2, k, n, n), complex))
    return data


def plan(c, o=None, rank=None, other=0):
    k, n, _ = c['s'].shape
    return core._plan_bounded_periodic_uhf_linear_model(k, n, c['alpha'], c['beta'],
        n if rank is None else rank, _options() if o is None else o, other)


def caps(c, o=None, **changes):
    p = plan(c, o)
    value = core._BoundedPeriodicRHFCaps()
    value.maximum_owned_numerical_bytes = p.peak_owned_numerical_bytes
    value.maximum_total_numerical_bytes = p.total_numerical_bytes
    value.maximum_work_units = p.maximum_work_units
    value.maximum_fock_calls = 128
    for key, v in changes.items():
        setattr(value, key, v)
    return value


def run(c, o=None, limits=None, **kwargs):
    o = _options() if o is None else o
    return core._solve_bounded_periodic_uhf_linear_model(c['s'], c['h'], c['kernel'], c['bias'],
        c['alpha'], c['beta'], c['nuclear'], o, caps(c, o) if limits is None else limits, **kwargs)


def diagonalize(c, f):
    x = _orthogonalizers(c, _options())
    e, u = np.linalg.eigh(x.conj().swapaxes(-1, -2) @ f @ x)
    orbitals = x @ u
    density = np.array([orbitals[s, :, :, :occ] @ orbitals[s, :, :, :occ].conj().swapaxes(-1, -2)
                        for s, occ in enumerate((c['alpha'], c['beta']))])
    return density, e, orbitals


def fock(c, d):
    return c['h'][None] + c['bias'] + (c['kernel'] @ d.ravel()).reshape(d.shape)


def energy(c, d, f):
    return c['nuclear'] + np.einsum('skij,skji->', d, c['h'][None] + f).real / (2 * len(c['s']))


@pytest.mark.parametrize('k,n,alpha,beta', [(1, 3, 2, 1), (3, 3, 2, 1), (8, 3, 1, 2), (2, 4, 3, 1), (3, 3, 1, 0), (2, 3, 0, 1), (3, 3, 1, 1)])
def test_coupled_spin_and_k_fixed_point_matches_independent_numpy(k, n, alpha, beta):
    c = case(k, n, alpha, beta)
    result = run(c)
    assert result.converged
    d, f = result.density_copy(), result.fock_copy()
    expected, _, _ = diagonalize(c, np.stack([c['h']] * 2))
    for _ in range(100):
        updated, _, _ = diagonalize(c, fock(c, expected))
        if np.linalg.norm(updated - expected) < 1e-14:
            expected = updated
            break
        expected = updated
    else:
        pytest.fail('independent fixed point did not converge')
    np.testing.assert_allclose(d, expected, atol=4e-9, rtol=0)
    np.testing.assert_allclose(f, fock(c, d), atol=3e-12, rtol=0)
    assert result.final_snapshot.energy_per_cell == pytest.approx(energy(c, d, f), abs=3e-12)
    np.testing.assert_allclose(d @ c['s'] @ d, d, atol=3e-12, rtol=0)
    np.testing.assert_allclose(np.einsum('skij,kji->sk', d, c['s']), np.repeat([[alpha], [beta]], k, axis=1), atol=3e-12)
    coeff, eps = result.coefficients_copy(), result.orbital_energies_copy()
    np.testing.assert_allclose(f @ coeff, c['s'] @ (coeff * eps[:, :, None, :]), atol=3e-12)
    x = _orthogonalizers(c, _options())
    residual = x.conj().swapaxes(-1, -2) @ (f @ d @ c['s'] - c['s'] @ d @ f) @ x
    assert result.final_snapshot.commutator_frobenius_rms == pytest.approx(np.linalg.norm(residual) / np.sqrt(k), abs=3e-12)
    gaps = [eps[s, :, occ].min() - eps[s, :, occ - 1].max() if occ else np.inf
            for s, occ in enumerate((alpha, beta))]
    np.testing.assert_allclose(result.final_snapshot.spin_band_gap, gaps, atol=3e-12)
    assert result.final_snapshot.global_band_gap == pytest.approx(min(gaps), abs=3e-12)


def test_every_capped_epoch_preserves_evaluated_spin_densities_and_history():
    c = case()
    events = []
    final = run(c, callback=lambda row: events.append(row))
    assert final.converged
    d, _, _ = diagonalize(c, np.stack([c['h']] * 2))
    for iteration, row in enumerate(events, 1):
        r = run(c, _options(maximum_iterations=iteration))
        np.testing.assert_allclose(r.density_copy(), d, atol=3e-12, rtol=0)
        assert row.energy_per_cell == pytest.approx(energy(c, d, fock(c, d)), abs=3e-12)
        assert row.energy_per_cell == r.final_snapshot.energy_per_cell
        assert row.evaluated_iteration == iteration
        d, _, _ = diagonalize(c, fock(c, d))
    assert run(c, _options(maximum_iterations=1)).final_snapshot.maximum_density_closure_frobenius > 1e-4


@pytest.mark.parametrize('kind', ['work', 'calls', 'cancel'])
def test_stops_retain_last_evaluated_both_spin_state(kind):
    c = case()
    one = run(c, _options(maximum_iterations=1))
    changes, kwargs = {}, {}
    status = core._BoundedPeriodicRHFStatus
    if kind == 'work':
        changes['maximum_work_units'] = one.final_snapshot.charged_work_units
        expected = status.WORK_LIMIT
    elif kind == 'calls':
        changes['maximum_fock_calls'] = 1
        expected = status.FOCK_CALL_LIMIT
    else:
        kwargs['callback'] = lambda row: False
        expected = status.CANCELLED
    result = run(c, limits=caps(c, **changes), **kwargs)
    assert result.final_snapshot.status == expected
    for name in ('density_copy', 'fock_copy', 'coefficients_copy', 'orbital_energies_copy'):
        np.testing.assert_array_equal(getattr(result, name)(), getattr(one, name)())


@pytest.mark.parametrize('field', ['maximum_owned_numerical_bytes', 'maximum_total_numerical_bytes'])
def test_exact_byte_cap_and_one_byte_short_admission(field):
    c = case()
    limits = caps(c)
    assert run(c, limits=limits).converged
    setattr(limits, field, getattr(limits, field) - 1)
    with pytest.raises(ValueError, match='byte cap'):
        run(c, limits=limits)


def test_memory_counts_share_overlap_and_count_both_spin_owners():
    c = case()
    p = plan(c)
    k, n, _ = c['s'].shape
    assert p.n_kpoints == k and p.n_alpha == 2 and p.n_beta == 1
    assert p.borrowed_input_bytes == 32 * k * n * n
    assert p.orthogonalizer_bytes == 16 * k * n * n
    assert p.density_bytes == p.fock_bytes == 32 * k * n * n
    assert p.coefficient_bytes == 32 * k * n * n
    assert p.orbital_energy_bytes == 16 * k * n
    assert plan(c, other=31).total_numerical_bytes == p.total_numerical_bytes + 31


@pytest.mark.parametrize('alpha,beta', [(0, 0), (3, 1), (1, 3), (4, 0)])
def test_invalid_occupation_counts_refuse(alpha, beta):
    with pytest.raises(ValueError, match='fixed-spin'):
        plan(case(alpha=alpha, beta=beta))


def test_fixed_spin_gap_does_not_mix_spin_chemical_potentials():
    c = case(k=1, interaction=0)
    # Each fixed-spin sector is insulating even though beta LUMO is below
    # alpha HOMO. Spin transfer is excluded by the specified populations.
    # Variational spin-specific response: G_alpha=5*S*Tr(S D_alpha).
    a = np.zeros_like(c['bias'])
    a[0] = c['s']
    c['kernel'] = np.ascontiguousarray(5 * np.outer(a.ravel(), a.swapaxes(-1, -2).ravel()))
    r = run(c)
    assert r.converged
    assert r.final_snapshot.global_lumo < r.final_snapshot.global_homo
    assert r.final_snapshot.global_band_gap > .05
    # A gap inside an occupied spin sector still fails the insulating gate.
    c['h'] = np.zeros_like(c['h'])
    c['kernel'][:] = 0
    assert run(c).final_snapshot.status == core._BoundedPeriodicRHFStatus.NON_INSULATING


@pytest.mark.parametrize('bad', ['nan', 'hermitian', 'rank', 'diis', 'missing', 'provider'])
def test_invalid_operators_and_callbacks_fail_without_result(bad):
    c, o, kwargs = case(), _options(), {}
    if bad == 'nan':
        c['h'][0, 0, 0] = np.nan
    elif bad == 'hermitian':
        c['bias'][1, 0, 0, 1] = 1j
    elif bad == 'rank':
        c['s'][:, -1, :] = 0
        c['s'][:, :, -1] = 0
    elif bad == 'diis':
        o.maximum_diis_history = 1
    elif bad == 'missing':
        kwargs['omit_last_output'] = True
    else:
        kwargs['fail_on_call'] = 1
    with pytest.raises((ValueError, RuntimeError, OverflowError)):
        run(c, o, **kwargs)


def test_spin_exchange_and_complex_ao_phase_covariance():
    c = case()
    baseline = run(c)
    swapped = dict(c, alpha=c['beta'], beta=c['alpha'], bias=c['bias'][::-1].copy())
    indices = np.arange(c['bias'].size).reshape(c['bias'].shape)[::-1].ravel()
    swapped['kernel'] = np.ascontiguousarray(c['kernel'][np.ix_(indices, indices)])
    r = run(swapped)
    np.testing.assert_allclose(r.density_copy()[::-1], baseline.density_copy(), atol=4e-12)
    assert r.final_snapshot.energy_per_cell == pytest.approx(baseline.final_snapshot.energy_per_cell, abs=4e-12)
    rng = np.random.default_rng(3141)
    phase = np.exp(1j * rng.normal(size=c['s'].shape[:2]))
    transport = phase[:, :, None] * phase[:, None, :].conj()
    spin_transport = np.broadcast_to(transport, c['bias'].shape).ravel()
    transformed = dict(c, s=np.ascontiguousarray(c['s'] * transport), h=np.ascontiguousarray(c['h'] * transport),
                       kernel=np.ascontiguousarray(spin_transport[:, None] * c['kernel'] * spin_transport[None, :].conj()))
    r = run(transformed)
    np.testing.assert_allclose(r.density_copy(), baseline.density_copy() * transport, atol=4e-12)
    assert r.final_snapshot.energy_per_cell == pytest.approx(baseline.final_snapshot.energy_per_cell, abs=4e-12)


def test_restricted_limit_matches_rhf_without_changing_physical_k_labels():
    from tests.test_bounded_periodic_rhf_solver import _run as rhf_run
    rc = rhf_case(k=3, n=3, interaction=.006)
    c = dict(rc, alpha=1, beta=1, bias=np.zeros((2, 3, 3, 3), complex),
             kernel=np.ascontiguousarray(np.tile(rc['kernel'], (2, 2))))
    unrestricted, restricted = run(c), rhf_run(rc)
    assert unrestricted.converged and restricted.converged
    assert unrestricted.memory.n_kpoints == restricted.memory.n_kpoints == 3
    np.testing.assert_allclose(unrestricted.density_copy().sum(axis=0), restricted.density_copy(), atol=4e-9)
    for spin in range(2):
        np.testing.assert_allclose(unrestricted.fock_copy()[spin], restricted.fock_copy(), atol=4e-11)
    assert unrestricted.final_snapshot.energy_per_cell == pytest.approx(restricted.final_snapshot.energy_per_cell, abs=4e-12)


def test_uniform_truncated_overlap_space_preserves_spin_projectors():
    c = case(k=2, n=4, alpha=2, beta=1)
    c['s'][:, -1, :] = 0
    c['s'][:, :, -1] = 0
    c['s'][:, -1, -1] = 1e-14  # positive, deliberately discarded direction
    r = run(c)
    assert r.converged and r.memory.retained_rank == 3
    d = r.density_copy()
    np.testing.assert_allclose(d @ c['s'] @ d, d, atol=3e-12)
    np.testing.assert_allclose(d[:, :, -1, :], 0, atol=0)
    # A different retained rank at one physical k is never silently padded.
    c['s'][0, -1, -1] = 1
    with pytest.raises(ValueError, match='differs across k'):
        run(c)


def test_second_spin_and_other_k_density_affect_the_shared_response():
    c = case(k=3)
    one = run(c, _options(maximum_iterations=1))
    changed = run(dict(c, beta=0), _options(maximum_iterations=1))
    assert np.max(abs(one.fock_copy()[0] - changed.fock_copy()[0])) > 1e-5
    # The explicit model couples every physical k and both spins.
    a = c['operator']
    perturbation = np.zeros_like(a)
    perturbation[1, 2] = a[1, 2]
    response = (c['kernel'] @ perturbation.ravel()).reshape(a.shape)
    assert all(np.linalg.norm(response[s, k]) > 1e-6 for s in range(2) for k in range(3))


@pytest.mark.parametrize('bad', ['dtype', 'layout', 'spin_extent', 'kernel_extent'])
def test_tiny_binding_rejects_ambiguous_spin_or_array_layout(bad):
    c = case()
    if bad == 'dtype':
        c['s'] = c['s'].real.copy()
    elif bad == 'layout':
        c['h'] = np.asfortranarray(c['h'])
    elif bad == 'spin_extent':
        c['bias'] = c['bias'][0]
    else:
        c['kernel'] = c['kernel'][:-1].copy()
    with pytest.raises(ValueError, match='complex128|shapes disagree'):
        run(c)


def test_progress_exception_propagates_and_scalar_snapshots_remain_owned():
    c = case()
    rows = []
    r = run(c, callback=lambda row: rows.append(row))
    energies = [row.energy_per_cell for row in rows]
    run(dict(c, beta=0))
    assert [row.energy_per_cell for row in rows] == energies
    assert rows[-1].energy_per_cell == r.final_snapshot.energy_per_cell

    def fail(row):
        raise RuntimeError('observer stopped')
    with pytest.raises(RuntimeError, match='observer stopped'):
        run(c, callback=fail)
