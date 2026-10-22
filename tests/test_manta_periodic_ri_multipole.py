"""Ordered finite GDF multipoles, paired RI and public method compatibility."""
import dataclasses
import itertools

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core
from tests.test_manta_periodic_fitting import CAPS
from tests.test_manta_periodic_gdf import case, independent, legacy
from tests.test_manta_periodic_gdf_tiles import assemble
from tests.test_manta_periodic_gdf_actions import inputs, check_actions
from tests.test_manta_periodic_ri_paired import paired_case, paired_job
from tests.test_manta_periodic_ri_q import builder, densities, reference


def compact_basis(basis, system, scale=5.):
    shells = basis.shells()
    for shell in shells:
        shell.exponents = [e*scale for e in shell.exponents]
    return vq.BasisSet(system.unit_cell_molecule(), shells, 'multipole-test', True)


def controls(o, error=1., order=0):
    o.coupling_multipole_error = error; o.coupling_multipole_order = order
    o.coupling_multipole_extents.minimum_radius = .1


def multipole_case(gamma=True, shift=False, *, fitted=False, order=0):
    a = paired_case(shift) if fitted else case(gamma, shift)
    a['options'].omega = .2; a['options'].auxiliary_cutoff_bohr = 8.
    for name in ('orbital', 'auxiliary'):
        a[name] = compact_basis(a[name], a['system'])
    controls(a['options'], 1. if order == 0 else 100., order)
    return a


@pytest.mark.parametrize('gamma', [False, True])
@pytest.mark.parametrize('order', [0, 2, 4])
def test_ordered_source_and_complex_adjoints_with_resolvable_far_terms(gamma, order):
    a = multipole_case(gamma, order=order); source = manta.PeriodicGDFTiles(**a)
    dense = manta.compute_periodic_gdf(**a); t = assemble(source)
    np.testing.assert_allclose(t, dense.three_center, atol=4e-13, rtol=3e-13)
    exact_m, exact_t = independent(a)
    np.testing.assert_allclose(dense.metric, exact_m, atol=5e-12, rtol=3e-13)
    np.testing.assert_allclose(exact_t, legacy(a)[1], atol=5e-12, rtol=3e-13)
    error = np.max(abs(t-exact_t)); bound = dense.receipt.evaluated_coupling_error_bound
    assert error <= bound+5e-12
    if order < 4: assert error > 1e-12
    assert 0 < bound <= a['options'].coupling_multipole_error
    assert dense.receipt.short_range.multipole_images > 0
    x, c = inputs(source); f, adj = check_actions(source, t, x, c, a['budget'])
    assert f.receipt.multipole_images == adj.receipt.multipole_images > 0
    assert f.receipt.coupling_error_bound == adj.receipt.coupling_error_bound == bound
    for result, expected in [(f, np.einsum('kpij,kij->p', exact_t.conj(), x)),
                             (adj, np.einsum('kpij,p->kij', exact_t, c))]:
        assert np.max(abs(result.values-expected)) <= result.receipt.approximation_bound+5e-12
    a['options'].coupling_multipole_error = 0.; exact = manta.PeriodicGDFTiles(**a)
    np.testing.assert_array_equal(source.metric().values, exact.metric().values)
    assert source.metric_source_identity == exact.metric_source_identity
    assert source.three_center_source_identity != exact.three_center_source_identity
    assert source.receipt.reference_three_center_source_identity == exact.three_center_source_identity
    # Caller mutation cannot move the source's density-independent partition.
    np.testing.assert_array_equal(source.forward(x, a['budget']).values, f.values)
    assert source.forward(2*x, a['budget']).receipt.multipole_images == f.receipt.multipole_images


def test_image_gauge_and_reciprocal_relabelling_preserve_the_approximate_map():
    a, moved = multipole_case(False), multipole_case(False, True)
    first, second = manta.PeriodicGDFTiles(**a), manta.PeriodicGDFTiles(**moved)
    t, u = assemble(first), assemble(second); lattice = a['system'].lattice[:, 0]
    for ik, k in enumerate(a['ket_kpoints']):
        auxiliary = np.exp(1j*np.array([0., a['q']@lattice]))
        bra = np.exp(1j*np.array([0., (k-a['q'])@lattice]))
        ket = np.exp(-1j*np.array([0., k@lattice]))
        np.testing.assert_allclose(u[ik], t[ik]*auxiliary[:, None, None]*bra[None, :, None]*ket[None, None, :], atol=5e-12, rtol=0)
    reciprocal = a['system'].reciprocal_lattice() @ np.array([1, -1, 2])
    shifted = dict(a, q=a['q']+reciprocal, ket_kpoints=a['ket_kpoints']+reciprocal)
    # Keep the physical G+q vectors fixed; reciprocal labels change by an integer.
    np.testing.assert_allclose(assemble(manta.PeriodicGDFTiles(**shifted)), t, atol=5e-12, rtol=0)


@pytest.mark.parametrize('rank', [0, 1, 6])
def test_fixed_metric_rank_energy_derivative_and_evaluated_error(rank):
    a = multipole_case(fitted=True); full = builder(a); e = np.linalg.eigvalsh(full.metric)
    cutoff = 1e-12 if rank == 6 else (2*e[-1] if rank == 0 else .5*(e[-2]+e[-1]))
    b = builder(a, block=6, absolute_cutoff=cutoff); d = densities(); w = np.array([.27, .73])
    build = lambda density: b.build(density, w, make_k=False)
    result = build(d); t = assemble(manta.PeriodicGDFTiles(**a)); m, exact_t = legacy(a)[:2]
    j, _, energy, _, _ = reference(m, t, d, w, absolute=cutoff)
    exact_j, _, exact_energy, _, _ = reference(m, exact_t, d, w, absolute=cutoff)
    np.testing.assert_allclose(result.J, j, atol=5e-12, rtol=2e-12)
    assert result.receipt.coulomb_energy == pytest.approx(energy, abs=5e-12)
    assert np.max(abs(result.J-exact_j)) <= result.receipt.approximation_j_bound+5e-12
    assert abs(energy-exact_energy) <= result.receipt.approximation_energy_bound+5e-12
    assert result.receipt.coulomb_contraction == 'paired-multipole'
    assert result.receipt.coupling_passes == 2 and result.receipt.coupling_multipole_images > 0
    a['options'].coupling_multipole_error = 0.; exact = builder(a, block=6, absolute_cutoff=cutoff)
    for name in ('metric', 'factor', 'projector'):
        np.testing.assert_array_equal(getattr(b, name), getattr(exact, name))
    delta = densities(seed=231); step = 2e-5
    derivative = (build(d+step*delta).receipt.coulomb_energy-build(d-step*delta).receipt.coulomb_energy)/(2*step)
    assert derivative == pytest.approx(np.einsum('k,skij,kji->', w, delta, result.J).real, abs=2e-10)
    zero = build(np.zeros_like(d))
    assert zero.receipt.approximation_j_bound == zero.receipt.approximation_energy_bound == 0.
    assert zero.receipt.coupling_multipole_images == result.receipt.coupling_multipole_images
    with pytest.raises(ValueError, match='independent exchange'):
        b.build(d, w)


@pytest.mark.parametrize('field,quantity', CAPS)
def test_all_joint_caps_and_no_values_before_admission(field, quantity):
    a = multipole_case(); source = manta.PeriodicGDFTiles(**a); b = a['budget']
    p = source.plan_coupling(b); x, _ = inputs(source)
    for cap, value in CAPS: setattr(b, cap, getattr(p, value))
    source.forward(x, b)
    setattr(b, field, getattr(p, quantity)-1); x[:] = np.nan
    with pytest.raises((ValueError, RuntimeError), match='budget'):
        source.forward(x, b)


def test_direct_fallback_zero_control_and_invalid_options():
    a = case(True); controls(a['options'], 1e-20, 6)
    source = manta.PeriodicGDFTiles(**a); t = assemble(source)
    assert source.forward(*[inputs(source)[0], a['budget']]).receipt.multipole_images == 0
    a['options'].coupling_multipole_error = 0.
    np.testing.assert_array_equal(t, assemble(manta.PeriodicGDFTiles(**a)))
    for value in [-1., float('nan'), float('inf')]:
        a['options'].coupling_multipole_error = value
        with pytest.raises(ValueError): manta.PeriodicGDFTiles(**a)
    a['options'].coupling_multipole_error = 1.; a['options'].coupling_screen_error = .1
    with pytest.raises(ValueError, match='cannot combine'): manta.PeriodicGDFTiles(**a)
    with pytest.raises(ValueError, match='cannot combine'):
        manta.PeriodicFittingOptions(.2, 8., coupling_multipole_error=1., coupling_screen_error=.1)
    for order in [-1, 13, True, 2.5]:
        with pytest.raises(ValueError): manta.PeriodicFittingOptions(.2, 8., coupling_multipole_order=order)


def multipole_job(directory, method='uhf', species='Li', limit=100):
    job, a, k = paired_job(directory, method, species, limit, block=6)
    for name in ('basis', 'auxiliary'):
        a[name] = compact_basis(a[name], a['system'])
    job['basis'] = a['basis']; job['coulomb'] = dataclasses.replace(job['coulomb'], auxiliary_basis=a['auxiliary'])
    controls(a['options'].fitting, .1, 0)
    job['periodic_fitting'] = dataclasses.replace(job['periodic_fitting'], coupling_multipole_error=.1,
        coupling_multipole_order=0, coupling_multipole_extents=a['options'].fitting.coupling_multipole_extents)
    return job, a, k


def multipole_hf_reference(a, k, method):
    from tests.test_manta_periodic_mixed_hf import reference as hf_reference
    s, h, exact = hf_reference(a, k, method)
    common = dict(orbital=a['basis'], auxiliary=a['auxiliary'], system=a['system'],
        options=a['options'].fitting, budget=a['budget'], q=np.zeros(3), ket_kpoints=k,
        vectors=a['reciprocal_vectors'])
    m, exact_t, _ = legacy(common); t = assemble(manta.PeriodicGDFTiles(**common)); w = np.full(len(k), 1./len(k))
    def build(d):
        spin = np.stack([.5*d, .5*d]) if method == 'rhf' else d
        f, energy = exact(d); j, _, ej, _, _ = reference(m, t, spin, w)
        original_j = reference(m, exact_t, spin, w)[0]
        return f+j-original_j, (energy[0], ej, *energy[2:])
    return s, h, build, exact, (m, t, exact_t)


@pytest.mark.parametrize('method,species,limit', [('rhf', 'Be', 100), ('uhf', 'Li', 100), ('uhf', 'Li', 3)])
def test_public_hf_uses_far_terms_and_keeps_unified_fock_and_saved_density(tmp_path, method, species, limit):
    from tests.test_manta_periodic_ri_domains import explicit
    from tests.test_manta_periodic_rhf import _nuclear_reference
    from tests.test_manta_periodic_ri_job import _as_fock_build
    from vibeqc.guess_read import _spin_density_k_from_qvf_bloch_payload
    job, args, k = multipole_job(tmp_path, method, species, limit)
    domains = getattr(manta, 'build_periodic_ri_'+method+'_domains')(**args); a = explicit(args, domains)
    _, _, build, exact, _ = multipole_hf_reference(a, k, method)
    result = vq.run_periodic_job(**job); n = result.native; f, energies = build(n.density); ef, ee = exact(n.density)
    assert result.converged == (limit == 100)
    np.testing.assert_allclose(n.fock, f, atol=8e-11, rtol=0)
    assert result.energy == pytest.approx(sum(_nuclear_reference(a)[0])+sum(energies), abs=8e-11)
    r = n.receipt.fitted_fock
    assert r.coupling_multipole_images > 0 and r.coupling_passes == 2
    assert np.max(abs(n.fock-ef)) <= r.approximation_j_bound+8e-11
    assert abs(sum(energies)-sum(ee)) <= r.approximation_energy_bound+8e-11
    state = _as_fock_build(result)
    assert state.energy.total == pytest.approx(result.energy, abs=1e-13)
    assert all(ch.operator.metadata.symmetry_block is None for ch in state.channels)
    saved = _spin_density_k_from_qvf_bloch_payload(job['output'].with_suffix('.qvf'))
    np.testing.assert_array_equal(saved, np.stack([.5*n.density]*2) if method == 'rhf' else n.density)
    assert result.integral_provenance['coupling_approximation']['coupling_multipole_images'] == r.coupling_multipole_images
    assert 'neese_bubblepole_2025' in job['output'].with_suffix('.references').read_text()


@pytest.mark.parametrize('pure', [False, True])
def test_signed_angular_leaf_contraction_against_libint(pure):
    from tests.test_manta_periodic_gdf import quadrature_long_range
    from tests.test_manta_periodic_fitting import libint_reference
    a = case(False); system = vq.PeriodicSystem(3, np.diag([6., 6.2, 6.4]), [vq.Atom(2, [.2, .3, .4])])
    mol = system.unit_cell_molecule()
    a.update(system=system,
        orbital=vq.BasisSet(mol, [core.ShellInfo(0, 1, pure, [4., 7.], [1., -.3], [.2, .3, .4])], 'angular-pair', True),
        auxiliary=vq.BasisSet(mol, [core.ShellInfo(0, 2, pure, [6., 9.], [1., -.25], [.2, .3, .4])], 'angular-aux', True),
        vectors=np.array(list(itertools.product(range(-1, 2), repeat=3))) @ system.reciprocal_lattice().T+a['q'])
    a['options'].pair_cutoff_bohr = 1.; a['options'].auxiliary_cutoff_bohr = 14.; a['options'].omega = .2
    controls(a['options'], 1., 2)
    source = manta.PeriodicGDFTiles(**a); tile = source.evaluate(0, 0, 0)
    assert tile.receipt.short_range.multipole_images > 0
    t = assemble(source); sm, st = libint_reference(a); lm, lt = quadrature_long_range(a)
    m, exact_t = sm+lm, st+lt
    np.testing.assert_allclose(source.metric().values, m, atol=5e-12, rtol=2e-12)
    assert np.max(abs(t-exact_t)) <= tile.receipt.evaluated_coupling_error_bound+5e-12
    check_actions(source, t, *inputs(source), a['budget'])


def multipole_ks_job(directory, spins=1, limit=100):
    from tests.test_manta_periodic_ri_paired import paired_ks_job
    job, args, a, k, old = paired_ks_job(directory, spins, limit)
    for name in ('basis', 'auxiliary'):
        args[name] = compact_basis(args[name], args['system'])
        a[name] = args[name]
    old['basis'] = args['basis']
    job['basis'] = args['basis']; job['coulomb'] = dataclasses.replace(job['coulomb'], auxiliary_basis=args['auxiliary'])
    controls(args['options'].fitting, .1, 0)
    job['periodic_fitting'] = dataclasses.replace(job['periodic_fitting'], coupling_multipole_error=.1,
        coupling_multipole_order=0, coupling_multipole_extents=args['options'].fitting.coupling_multipole_extents)
    return job, args, a, k, old


def multipole_ks_reference(a, k, old):
    from tests.test_manta_periodic_ri_ks import reference as ks_reference
    s, h, exact = ks_reference(a, k, old)
    common = dict(orbital=a['basis'], auxiliary=a['auxiliary'], system=a['system'],
        options=a['options'].fitting, budget=a['budget'], q=np.zeros(3), ket_kpoints=k, vectors=a['reciprocal_vectors'])
    m, exact_t, _ = legacy(common); t = assemble(manta.PeriodicGDFTiles(**common)); w = np.full(len(k), 1./len(k))
    def build(d):
        spin = np.repeat(.5*d, 2, axis=0) if a['functional'].spin == 1 else d
        f, energies, populations = exact(d); j, _, ej, _, _ = reference(m, t, spin, w)
        original_j = reference(m, exact_t, spin, w)[0]
        return f+j-original_j, (energies[0], ej, *energies[2:]), populations
    return s, h, build, exact, (m, t, exact_t)


@pytest.mark.parametrize('spins', [1, 2])
def test_public_ks_retains_multipole_receipt_and_same_xc(tmp_path, spins):
    from tests.test_manta_periodic_ri_ks_domains import explicit
    from tests.test_manta_periodic_ri_ks import components, as_fock_build
    from tests.test_manta_periodic_rhf import _nuclear_reference
    job, args, a, k, old = multipole_ks_job(tmp_path, spins, limit=3); explicit(args, a, old)
    _, _, build, exact, _ = multipole_ks_reference(a, k, old)
    result = vq.run_periodic_job(**job); n = result.native
    f, energies, _ = build(n.density); ef, ee, _ = exact(n.density)
    np.testing.assert_allclose(n.fock, f, atol=8e-11, rtol=0)
    np.testing.assert_allclose(components(n), energies, atol=8e-11, rtol=0)
    assert result.energy == pytest.approx(sum(_nuclear_reference(old)[0])+sum(energies), abs=8e-11)
    r = result.integral_provenance['fitted_fock']
    assert r['coupling_multipole_images'] > 0
    assert 'neese_bubblepole_2025' in job['output'].with_suffix('.references').read_text()
    assert np.max(abs(n.fock-ef)) <= r['approximation_j_bound']+8e-11
    assert abs(sum(energies)-sum(ee)) <= r['approximation_energy_bound']+8e-11
    assert as_fock_build(n, k).energy.total == pytest.approx(result.energy, abs=1e-13)


@pytest.mark.parametrize('field', ['coupling_multipole_error', 'coupling_screen_error'])
def test_ccm_fit_export_cannot_promote_j_only_approximation_to_exchange(field):
    from vibeqc.periodic.ccm import CCMSystem
    from vibeqc.periodic.ccm.fit_source import MANTAFitSource
    from vibeqc.periodic.ccm.neutral import ccm_neutral_cderi_fold
    from tests.test_ccm_manta_fit_source import _cell, _options, _budget, CUTOFF_HA
    ccm = CCMSystem(_cell(), (2, 1, 1), 'sto-3g'); options = _options()
    setattr(options.fitting, field, .1)
    # Exact exchange selection would otherwise bypass the native fitted-K gate.
    options.exchange_algorithm = 'exact'
    with pytest.raises(NotImplementedError, match='J-only'):
        ccm_neutral_cderi_fold(ccm, fit_source=MANTAFitSource(options, CUTOFF_HA, budget=_budget()))
