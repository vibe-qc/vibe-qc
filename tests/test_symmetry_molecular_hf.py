"""Method bridge against ordered finite-source contractions and native SCF."""
from dataclasses import asdict
import json

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.manta import ReferenceJKBuilder
from vibeqc.solvers.operators import OperatorFrame
from vibeqc.symmetry_shared import Budget, SpaceIdentity
from vibeqc.symmetry_ao import build_molecular_space_action
from vibeqc.symmetry_quartets import qualify_manta_quartets
from vibeqc.symmetry_quartets_screening import QuartetScreening
from vibeqc.symmetry_molecular_hf import ScreenedMolecularHF, ScreenedMolecularJK
from tests.test_symmetry_quartets import molecular_case, masked_source, dense_jk

BUDGET = Budget(512 << 20, 10**12)


def case(kind='water', *, executor='native', cap=.3):
    mol, basis, rotations, translations = molecular_case(kind)
    provider = ReferenceJKBuilder(basis)
    plan = qualify_manta_quartets(mol, basis, rotations, translations,
        provider=provider, source_tolerance=1e-10, budget=BUDGET)
    screen = QuartetScreening(plan, maximum_omitted_norm=cap, budget=BUDGET)
    jk = ScreenedMolecularJK(screen, budget=BUDGET, executor=executor)
    return jk, plan, provider, mol, basis, rotations, translations


def method(jk, mol, basis, *, retained=False):
    s = np.asarray(vq.compute_overlap(basis))
    eig, vec = np.linalg.eigh(s)
    x = vec/np.sqrt(eig)
    if retained:
        x = x[:, 1:]
    frame = OperatorFrame(basis_id='test-native-molecular', n=basis.nbasis,
        ordering='native AO', image_gauge='molecular', overlap=s, X=x)
    h = np.asarray(vq.compute_kinetic(basis)+vq.compute_nuclear(basis, mol))
    return ScreenedMolecularHF(jk, frame=frame, hcore=h, nuclear_energy=0.), frame, h


@pytest.mark.parametrize('kind', ['water', 'ammonia', 'cartesian'])
@pytest.mark.parametrize('executor', ['numpy', 'native'])
def test_nonsymmetric_native_bridge_and_covariance(kind, executor):
    jk, plan, provider, mol, basis, rotations, translations = case(kind, executor=executor)
    _, masked = masked_source(plan, provider, jk.screening)
    d = np.random.default_rng(514).normal(size=(basis.nbasis,)*2)
    j, k, receipt = jk.build_channels((d,), targets=('jk',))
    expected = dense_jk(masked, d)
    np.testing.assert_allclose(j[0], expected[0], atol=2e-11, rtol=0)
    np.testing.assert_allclose(k[0], expected[1], atol=2e-11, rtol=0)
    np.testing.assert_allclose(jk.build_g_rhf(d), j[0]-.5*k[0], atol=2e-11, rtol=0)
    np.testing.assert_allclose(jk.build_g_rhf(d, 0.), j[0], atol=2e-11, rtol=0)
    assert jk.last_build.targets == ('j',)
    identity = SpaceIdentity('molecule', 'basis', 'AO', 'metric')
    for r, t in zip(rotations, translations):
        action = build_molecular_space_action(mol, basis, np.ascontiguousarray(r), t,
            source=identity, target=identity, budget=BUDGET)
        u = action.apply(np.eye(basis.nbasis, dtype=complex))
        assert np.max(np.abs(u.imag)) == 0.
        moved = u.real@d@u.real.T
        for actual, original in ((jk.build_J(moved), j[0]), (jk.build_K(moved), k[0])):
            np.testing.assert_allclose(u.conj().T@actual@u, original, atol=3e-11, rtol=0)
    assert receipt.work.shell_calls == len(plan.orbits)-jk.screening.omitted_representatives
    assert receipt.scalar_products == 2*(basis.nbasis**4-jk.screening.omitted_integral_values)
    assert jk.totals.builds == 3+2*len(rotations)
    json.dumps(asdict(jk.totals), allow_nan=False)


@pytest.mark.parametrize('spin_method', ['RHF', 'UHF'])
@pytest.mark.parametrize('executor', ['numpy', 'native'])
def test_hf_state_energy_response_ownership_and_source(spin_method, executor):
    jk, plan, provider, mol, basis, *_ = case('ammonia', executor=executor)
    hf, frame, h = method(jk, mol, basis, retained=True)
    full, eri = masked_source(plan, provider, jk.screening)
    rng = np.random.default_rng(997)
    count = 1 if spin_method == 'RHF' else 2
    raw = rng.normal(size=(count, basis.nbasis, basis.nbasis))
    d = tuple(.01*a@a.T for a in raw)
    result = hf.evaluate(d, method=spin_method, generation=0)
    total = sum(d)
    j = dense_jk(eri, total)[0]
    exchange = .5 if count == 1 else 1.
    k = [dense_jk(eri, a)[1] for a in d]
    f = [h+j-exchange*b for b in k]
    expected = sum(np.einsum('pq,qp', a, h+.5*j-.5*exchange*b) for a, b in zip(d, k))
    assert abs(result.build.energy.electronic-expected) < 2e-11
    full_j = dense_jk(full, total)[0]
    full_k = [dense_jk(full, a)[1] for a in d]
    full_energy = sum(np.einsum('pq,qp', a, h+.5*full_j-.5*exchange*b) for a, b in zip(d, full_k))
    assert abs(result.build.energy.electronic-full_energy) <= result.electronic_error_bound+2e-11
    for channel, reference in zip(result.build.channels, f):
        np.testing.assert_allclose(channel.operator.matrix, reference, atol=2e-11, rtol=0)
        assert channel.operator.frame is frame
    direction = tuple(a+a.T for a in rng.normal(size=raw.shape))
    step = 2e-5
    energies = [hf.evaluate(tuple(a+sign*step*b for a, b in zip(d, direction)),
        method=spin_method, generation=1).build.energy.electronic for sign in (1, -1)]
    derivative = sum(np.einsum('pq,qp', a, b) for a, b in zip(direction, f))
    assert abs((energies[0]-energies[1])/(2*step)-derivative) < 2e-9
    source_id = result.build.source.identity
    saved = tuple(a.copy() for a in d)
    h[:] = 0.
    for a in d:
        a[:] = 0.
    later = hf.evaluate(d, method=spin_method, generation=2)
    assert later.build.source.identity == source_id
    assert later.build.identity != result.build.identity
    for channel, reference in zip(result.build.channels, saved):
        np.testing.assert_array_equal(channel.density, reference)
        with pytest.raises(ValueError):
            channel.density.setflags(write=True)
    assert result.receipt.targets == (('jk',) if count == 1 else ('j', 'k', 'k'))
    assert len(result.fock_error_bounds) == count
    # The fixed receipt includes neither current density nor its error bound.
    assert 'density_error_bounds' not in json.loads(hf.source.details_json)


def test_bridge_refusals_budget_and_native_controls():
    jk, plan, provider, mol, basis, *_ = case()
    d = np.eye(basis.nbasis)
    required = jk.requirements(('jk',))
    with pytest.raises(MemoryError):
        ScreenedMolecularJK(jk.screening,
            budget=Budget(required.maximum_bytes-1, required.maximum_work))
    exact = ScreenedMolecularJK(jk.screening, budget=required)
    exact.build_g_rhf(d)
    for invalid in (d.astype(complex), d.astype(np.float32), d.tolist()):
        with pytest.raises(TypeError, match='real float64'):
            jk.build_J(invalid)
    for invalid in (d[:-1], np.full_like(d, np.nan)):
        with pytest.raises(ValueError, match='density'):
            jk.build_J(invalid)
    with pytest.raises(TypeError, match='certificate'):
        ScreenedMolecularJK(plan, budget=BUDGET)
    with pytest.raises((RuntimeError, ValueError)):
        jk.build_K_erf(d, .3)
    hf, _, _ = method(jk, mol, basis)
    bad = d.copy(); bad[0, 1] = 1.
    with pytest.raises(ValueError, match='Hermitian'):
        hf.evaluate((bad,), method='RHF', generation=0)
    assert jk.totals.builds == 0  # every refusal preceded source work
    jk.set_incremental(True)
    jk.set_schwarz_threshold(1.)
    jk.reset_state()
    assert not jk.incremental_active() and not jk.uses_runtime_schwarz_threshold()
    source_id = jk.screening.source_identity
    before = jk.build_K_pair(d, 2*d)
    assert jk.totals.builds == 1 and jk.last_build.targets == ('k', 'k')
    np.testing.assert_allclose(before[1], 2*before[0], atol=1e-13, rtol=0)
    jk.reset_state()
    assert jk.totals.builds == 1 and jk.screening.source_identity == source_id


def test_energy_bound_underflow_is_not_reported_as_zero():
    jk, _, _, mol, basis, *_ = case()
    hf, _, _ = method(jk, mol, basis)
    with pytest.raises(ValueError, match='nonrepresentable HF error bound'):
        hf.evaluate((1e-200*np.eye(basis.nbasis),), method='RHF', generation=0)


@pytest.mark.parametrize('spin_method', ['RHF', 'UHF'])
def test_native_scf_uses_screened_builder(spin_method):
    jk, _, provider, mol, basis, *_ = case(cap=1e-12)
    _, frame, h = method(jk, mol, basis)
    if spin_method == 'RHF':
        opts = vq.RHFOptions()
        driver = vq.run_rhf_scf_with_jk
        args = (basis, 10, frame.overlap, h, 0.)
    else:
        opts = vq.UHFOptions()
        driver = vq.run_uhf_scf_with_jk
        args = (basis, 5, 5, frame.overlap, h, 0.)
    opts.conv_tol_energy = 1e-11
    opts.conv_tol_grad = 1e-9
    reduced = driver(*args, jk, opts)
    exact = driver(*args, provider, opts)
    assert reduced.converged and exact.converged
    assert abs(reduced.energy-exact.energy) < 1e-10
    assert jk.totals.builds > 2
    assert jk.totals.work.shell_calls > 0
