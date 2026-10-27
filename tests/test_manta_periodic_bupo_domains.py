"""Certified reciprocal cutoff selection over immutable BUPO HF sources."""
import json
import numpy as np
import pytest
import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core
from tests.test_manta_periodic_bupo_hf import independent, as_fock_build
from tests.test_manta_periodic_exchange_bounds import hf_case, ordered_fourier
from tests.test_manta_periodic_ewald import cube
from tests.test_manta_periodic_pairs import options as pair_options
from tests.test_manta_periodic_bubbles import bubble_options
from tests.test_manta_periodic_bupo import options as bupo_options


def fixture(*, spin='restricted', z=2, shifted=False, q0='strict-zero-mode', iterations=40,
            mesh_shape=None, skew=False, left=False):
    a, c = hf_case(spin=spin, z=z, shifted=shifted, q0=q0, radius=0, iterations=iterations,
                   image_radius=2 if z == 3 else 1, cutoff=6.1 if z == 3 else 1., eta=.5 if z == 3 else .6)
    if mesh_shape is not None or skew or left:
        cell = np.array([[6., .6, -.2], [0., 6.3, .3], [0., 0., 6.5]]) if skew else c['cell'].copy()
        if left:
            cell[:, 0] *= -1
        system = vq.PeriodicSystem(3, cell, a['system'].unit_cell)
        system.multiplicity = a['system'].multiplicity
        mesh = manta.RegularKMesh(mesh_shape or ((2, 1, 1) if shifted else (1, 1, 1)),
                                 (1, 0, 0) if shifted else (0, 0, 0))
        k = np.array([system.reciprocal_lattice() @ mesh.fractional(mesh.address(i)) for i in range(len(mesh))])
        weights = np.full(len(mesh), 1./len(mesh))
        pairs = manta.PeriodicPairSource(c['basis'], cell, c['images'], k, weights, pair_options(0, c['cutoff'], .2))
        tree = manta.PeriodicBubbleTree(pairs, bubble_options())
        ewald = manta.PeriodicEwaldSource(pairs, c['eta'], cube(c['real_radius']), cube(c['reciprocal_radius'], reciprocal=True), 64)
        options = bupo_options(0.)
        options.bound_infinite_domain = options.bound_pair_images = True
        a.update(electronic=manta.PeriodicBupoSource(tree, ewald, cube(0), options), system=system, mesh=mesh)
        c.update(cell=cell, k=k, weights=weights, pairs=pairs)
    a.pop('exchange_vectors')
    a.pop('transfer_offsets')
    return a, c


def brute_spheres(a, c, cutoff, radius):
    """Independent integer-box enumeration, ordinary NumPy geometry.

    Away from an exact radial boundary, membership is the mathematical
    energy sphere. Boundary predicate parity has its own exact cubic test.
    """
    base = cube(radius) @ (2*np.pi*np.linalg.inv(c['cell']))
    values = []
    for target in c['k']:
        for source in c['k']:
            q = base + target - source
            norm = np.sum(q*q, axis=1)
            values.append(q[(norm > 1e-28) & (norm <= 2*cutoff)])
    return values


@pytest.mark.parametrize('shape,shifted,skew,left', [((1, 1, 1), False, False, False),
    ((2, 1, 1), True, False, False), ((3, 1, 1), True, True, False), ((2, 2, 1), True, True, True)])
def test_independent_spheres_and_opposite_support(shape, shifted, skew, left):
    a, c = fixture(mesh_shape=shape, shifted=shifted, skew=skew, left=left, iterations=1)
    cutoff = 3.17
    d = manta.build_periodic_bupo_hf_exchange_domains(**a, reciprocal_energy_cutoff=cutoff)
    assert d.receipt.exchange_vectors == d.receipt.conjugacy_vectors == len(d.exchange_vectors)
    assert d.plan.exchange_candidates >= len(d.exchange_vectors)
    refs = brute_spheres(a, c, cutoff, d.plan.exchange_bound_cube_radius+1)
    inverse = np.linalg.inv(a['system'].reciprocal_lattice())
    nk = len(c['k'])
    for pair, expected in enumerate(refs):
        actual = d.exchange_vectors[d.transfer_offsets[pair]:d.transfer_offsets[pair+1]]
        np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=3e-15)
        target, source = divmod(pair, nk)
        labels = (actual-c['k'][target]+c['k'][source]) @ inverse.T
        np.testing.assert_allclose(labels, np.rint(labels), atol=3e-15)
        assert np.max(abs(np.rint(labels))) <= d.plan.exchange_bound_cube_radius
        opposite = source*nk+target
        other = d.exchange_vectors[d.transfer_offsets[opposite]:d.transfer_offsets[opposite+1]]
        np.testing.assert_array_equal(actual, -other[::-1])
    assert a['options'].exchange_bound_cube_radius == 0
    q = d.exchange_vectors
    q[:] = np.nan
    offsets = d.transfer_offsets
    offsets[:] = 0
    assert np.isfinite(d.exchange_vectors).all() and d.transfer_offsets[-1] > 0
    with pytest.raises(AttributeError):
        d.plan.exchange_bound_cube_radius = 99


def test_exact_gamma_boundary_and_empty_sphere():
    a, c = fixture(iterations=1)
    cutoff = float(np.linalg.norm(a['system'].reciprocal_lattice()[:, 0])**2/2)
    d = manta.build_periodic_bupo_hf_exchange_domains(**a, reciprocal_energy_cutoff=cutoff)
    assert len(d.exchange_vectors) == 6
    empty = manta.build_periodic_bupo_hf_exchange_domains(**a, reciprocal_energy_cutoff=1e-5)
    assert empty.exchange_vectors.shape == (0, 3)
    assert empty.receipt.conjugacy_vectors == 0
    # Empty support is representable; its missing-K bound must fail HF.
    with pytest.raises(ValueError, match='exchange accuracy target'):
        core._manta.run_periodic_bupo_hf_from_cutoff(**a, reciprocal_energy_cutoff=1e-5)


def test_joint_admission_before_generation():
    a, _ = fixture(shifted=True, iterations=1)
    p = manta.plan_periodic_bupo_hf_from_cutoff(**a, reciprocal_energy_cutoff=3.17)
    limits = dict(memory_bytes=p.modeled_peak_bytes, shell_calls=p.shell_calls,
                  primitive_quartets=p.primitive_pairs, scalar_terms=p.scalar_terms, kernel_terms=p.kernel_terms)
    b = manta.ResourceBudget()
    for name, value in limits.items():
        setattr(b, name, value)
    a['budget'] = b
    exact = manta.build_periodic_bupo_hf_exchange_domains(**a, reciprocal_energy_cutoff=3.17)
    assert exact.plan.modeled_peak_bytes == p.modeled_peak_bytes
    for name, value in limits.items():
        setattr(b, name, value-1)
        with pytest.raises((ValueError, OverflowError), match='budget'):
            manta.build_periodic_bupo_hf_exchange_domains(**a, reciprocal_energy_cutoff=3.17)
        setattr(b, name, value)
    assert p.scalar_terms == p.hf.scalar_terms+p.domain_scalar_terms
    assert p.kernel_terms == p.hf.kernel_terms+p.domain_kernel_terms


@pytest.mark.parametrize('value', [0., -1., float('nan'), float('inf')])
def test_invalid_cutoff(value):
    a, _ = fixture(iterations=1)
    with pytest.raises(ValueError, match='finite positive'):
        manta.plan_periodic_bupo_hf_from_cutoff(**a, reciprocal_energy_cutoff=value)


def test_owned_completion_policy_and_original_source_gate():
    a, _ = fixture(iterations=1)
    for bound, radius in [(False, 0), (True, 1)]:
        a['options'].bound_exchange = bound
        a['options'].exchange_bound_cube_radius = radius
        with pytest.raises(ValueError, match='factory-owned'):
            manta.plan_periodic_bupo_hf_from_cutoff(**a, reciprocal_energy_cutoff=3.)
    a['options'].bound_exchange = True
    a['options'].exchange_bound_cube_radius = 0
    a['mesh'] = manta.RegularKMesh((2, 1, 1), (0, 0, 0))
    with pytest.raises(ValueError, match='original source lattice and uniform full mesh'):
        manta.build_periodic_bupo_hf_exchange_domains(**a, reciprocal_energy_cutoff=3.)


def calculation(**kwargs):
    a, c = fixture(**kwargs)
    cutoff = 900. if kwargs.get('z') == 3 else 500.
    domains = manta.build_periodic_bupo_hf_exchange_domains(**a, reciprocal_energy_cutoff=cutoff)
    result = core._manta.run_periodic_bupo_hf_from_cutoff(**a, reciprocal_energy_cutoff=cutoff)
    a['options'].exchange_bound_cube_radius = domains.plan.exchange_bound_cube_radius
    explicit = dict(a, exchange_vectors=domains.exchange_vectors, transfer_offsets=domains.transfer_offsets)
    control = core._manta.run_periodic_bupo_hf(**explicit)
    reference = independent(explicit, c, fourier_reference=ordered_fourier,
                            cache_exchange=True, density_metric='orthonormal')
    same = reference['build'](result.density)
    return a, c, domains, result, control, reference, same


@pytest.mark.parametrize('spin,z,shifted,q0,iterations', [('restricted', 2, False, 'strict-zero-mode', 40),
    ('unrestricted', 1, True, 'BvK-ewald', 40), ('unrestricted', 3, True, 'BvK-ewald', 1)])
def test_complete_cutoff_hf_independent_reference(spin, z, shifted, q0, iterations):
    a, c, domains, result, control, reference, same = calculation(spin=spin, z=z, shifted=shifted, q0=q0, iterations=iterations)
    assert result.converged == (iterations > 1)
    for name in ('density', 'fock', 'hcore', 'overlap'):
        np.testing.assert_array_equal(getattr(result, name), getattr(control, name))
    assert result.receipt.energy == control.receipt.energy
    np.testing.assert_allclose(result.fock, same['fock'], atol=3e-11, rtol=1e-12)
    assert result.receipt.energy == pytest.approx(same['energy'], abs=3e-11)
    if result.converged:
        np.testing.assert_allclose(result.density, reference['density'], atol=3e-9, rtol=2e-10)
        assert result.receipt.energy == pytest.approx(reference['converged']['energy'], abs=3e-11)
    else:
        np.testing.assert_allclose(result.density, reference['initial'], atol=3e-11, rtol=1e-12)
    assert result.receipt.exchange_bound_cube_radius == domains.plan.exchange_bound_cube_radius
    assert result.receipt.domain_identity == domains.receipt.identity
    assert result.receipt.electrostatic_accuracy_passed and result.receipt.one_electron_accuracy_passed and result.receipt.exchange_accuracy_passed
    assert result.receipt.maximum_observed_exchange_matrix_error <= 1e-8
    assert result.receipt.maximum_observed_exchange_energy_error <= 1e-8
    state = as_fock_build(result, c['k'])
    assert state.energy.total == pytest.approx(result.receipt.energy, abs=3e-12)
    for channel, d, f in zip(state.channels, result.density.reshape(-1, result.plan.n_basis, result.plan.n_basis),
                              result.fock.reshape(-1, result.plan.n_basis, result.plan.n_basis)):
        np.testing.assert_array_equal(channel.density, d)
        np.testing.assert_allclose(channel.operator.apply(np.eye(result.plan.n_basis, dtype=complex)), f, atol=1e-13)


def test_selected_provenance_preserves_executed_domain_and_budget(monkeypatch):
    from vibeqc import output
    a, _ = fixture(iterations=1)
    budget = a.pop('budget')
    events = []
    monkeypatch.setattr(output, 'record', lambda *args, **kw: events.append(kw['integral_provenance']))
    result = manta.run_periodic_bupo_hf_from_cutoff(**a, reciprocal_energy_cutoff=500.,
        integral_engine=manta.IntegralEngineOptions(backend='manta', budget=budget))
    evidence = events[-1]
    json.dumps(evidence, allow_nan=False)
    assert budget.other_live_bytes == 0 and a['options'].exchange_bound_cube_radius == 0
    assert evidence['bound_exchange'] and evidence['exchange_bound_cube_radius'] > 0
    assert evidence['domain_identity'] == result.receipt.domain_identity
    assert evidence['domain_exchange_vectors'] == evidence['domain_conjugacy_vectors'] > 0
    assert evidence['domain_selection'] == 'original-bupo-source;certified-reciprocal-energy-sphere'
    assert evidence['domain_live_bytes'] > 0 and not evidence['symmetry_reduced']
