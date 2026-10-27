"""Fitted pure/global KS: independent raw fits, quadrature and functional SCF."""
import itertools
import json

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core
from vibeqc.bipole_fock_ewald import probe_charge_madelung_supercell
from tests.test_manta_periodic_mixed_hf import case as hf_case
from tests.test_manta_periodic_ks import oracle as semilocal_oracle, independent_scf, occupy
from tests.test_manta_periodic_gdf import legacy
from tests.test_manta_periodic_exchange import _quadrature_bloch
from tests.test_manta_periodic_rhf import _nuclear_reference
from tests.test_manta_periodic_global_hybrid_ks import components, EMPTY_HYBRID
from tests.test_manta_periodic_semilocal_xc import bloch_values_gradients


CAPS = [('memory_bytes', 'modeled_peak_bytes'), ('shell_calls', 'shell_calls'),
        ('primitive_quartets', 'primitive_work'), ('scalar_terms', 'scalar_terms'), ('kernel_terms', 'kernel_terms')]


def case(profile='pure', spins=1, name=None, mesh=(2, 1, 1), shift=(1, 0, 0),
         species=None, correction='BvK-ewald', order=7):
    method = 'rhf' if spins == 1 else 'uhf'
    a, k = hf_case(method, 'J' if profile in ('pure', 'J') else 'K', mesh, shift, species, correction)
    old = dict(a, exchange_vectors=a['pair_vectors'] if profile in ('pure', 'J') else -a['pair_vectors'])
    o = manta.PeriodicRIKSOptions() if profile == 'pure' else manta.PeriodicRIGlobalHybridKSOptions()
    hfo = a['options']; o.fitting = hfo.fitting; o.ri = hfo.ri
    for field in ('alpha', 'nuclear_cutoff_bohr', 'pair_cutoff_bohr', 'reciprocal_batch'):
        setattr(o.ks, field, getattr(hfo, field))
    o.ks.scf.maximum_iterations = 100
    if profile != 'pure':
        o.coulomb_algorithm = 'exact' if profile == 'K' else 'ri'
        o.exchange_algorithm = 'exact' if profile == 'J' else 'ri'
        o.exchange_q0 = correction
    x, w = np.polynomial.legendre.leggauss(order)
    indices = np.array(list(itertools.product(range(order), repeat=3)))
    lattice = a['system'].lattice
    points = np.ascontiguousarray(((x[indices]+1)/2)@lattice.T)
    qw = np.prod(w[indices]/2, axis=1)*abs(np.linalg.det(lattice))
    images = np.array(list(itertools.product(range(-1, 2), repeat=3)), float)@lattice.T
    a.update(options=o, functional=vq.Functional(name or ('pbe' if profile == 'pure' else 'pbe0'), spin=spins),
             xc_images=images, points=points, quadrature_weights=qw)
    a['exchange_vectors'] = a.pop('pair_vectors')
    if profile == 'pure':
        a.pop('exchange_vectors'); a.pop('transfer_offsets')
    old['options'] = o.ks
    return a, k, old


def run(a):
    return getattr(core._manta, 'run_periodic_ri_'+('global_hybrid_ks' if 'exchange_vectors' in a else 'ks'))(**a)


def plan(a):
    args = [a[n] for n in ('basis', 'auxiliary', 'system', 'mesh', 'functional')]+[len(a['reciprocal_vectors'])]
    hybrid = 'exchange_vectors' in a
    if hybrid: args.append(int(max(np.diff(a['transfer_offsets']))))
    args += [len(a[n]) for n in ('translations', 'xc_images', 'points')]+[a['options'], a['budget']]
    return getattr(manta, 'plan_periodic_ri_'+('global_hybrid_ks' if hybrid else 'ks'))(*args)


def reference(a, k, old, fraction=None):
    """No MANTA contractions: libint raw fits, Gaussian Fourier and libxc values."""
    s, h, semilocal = semilocal_oracle(a, k, old)
    o = a['options']; ns = a['functional'].spin; nk = len(k); w = np.full(nk, 1./nk)
    hybrid = 'exchange_vectors' in a
    fit_j = not hybrid or o.coulomb_algorithm == 'ri'; fit_k = hybrid and o.exchange_algorithm == 'ri'
    if fraction is None:
        fraction = 0. if not hybrid else {'pbe0': .25, 'b3lyp': .2, EMPTY_HYBRID: .2}[a['functional'].name.lower()]
    volume = abs(np.linalg.det(a['system'].lattice)); g = a['reciprocal_vectors']
    common = dict(orbital=a['basis'], auxiliary=a['auxiliary'], system=a['system'], options=o.fitting, budget=a['budget'])
    def factor(q, ket, vectors):
        m, t, _ = legacy(dict(common, q=q, ket_kpoints=ket, vectors=vectors))
        m = np.tril(m)+np.tril(m, -1).conj().T; m[np.diag_indices(len(m))] = m.diagonal().real
        e, u = np.linalg.eigh(m)
        keep = e > max(o.ri.absolute_cutoff, o.ri.relative_cutoff*max(0., e[-1]))
        return np.einsum('pl,kpmn->klmn', (u[:, keep]/np.sqrt(e[keep])).conj(), t)
    gamma = factor(np.zeros(3), k, g); pairs = {}; direct = {}
    if hybrid:
        for i in range(nk):
            for j in range(nk):
                lo, hi = a['transfer_offsets'][i*nk+j:i*nk+j+2]; q = a['exchange_vectors'][lo:hi]
                if fit_k and i != j:
                    pairs[i, j] = factor(k[j]-k[i], k[j:j+1], q)[0]
                elif not fit_k:
                    direct[i, j] = (_quadrature_bloch(a['basis'], q, a['translations'], k[i], o.ks.pair_cutoff_bohr),
                                    w[j]*4*np.pi/volume/np.sum(q*q, axis=1))
    fj = np.stack([_quadrature_bloch(a['basis'], g, a['translations'], -point, o.ks.pair_cutoff_bohr) for point in k])
    kernel = 4*np.pi/volume/np.sum(g*g, axis=1)
    xi = 0. if not hybrid or o.exchange_q0 == 'strict-zero-mode' else probe_charge_madelung_supercell(
        a['system'], tuple(a['mesh'].mesh), alpha=o.madelung_alpha, precision=1e-14)
    def build(d):
        f, energies, populations = semilocal(d); e1, ej, ex = energies
        spin = np.repeat(.5*d, 2, axis=0) if ns == 1 else d
        if fit_j:
            rho = np.einsum('k,klmn,skmn->l', w, gamma.conj(), spin)
            j = np.einsum('klmn,l->kmn', gamma, rho); ej = .5*np.vdot(rho, rho).real
            rho_direct = np.einsum('k,skmn,kgmn->g', w, d, fj)
            j_direct = np.einsum('g,g,kgmn->kmn', kernel, rho_direct, fj.conj())
            f += j-j_direct
        exchange = np.zeros_like(spin)
        if fit_k:
            exchange = np.einsum('k,klmp,skpq,klnq->skmn', w, gamma, spin, gamma.conj())
            for (i, j), b in pairs.items():
                exchange[:, i] += w[j]*np.einsum('lmp,spq,lnq->smn', b, spin[:, j], b.conj())
        elif hybrid:
            for (i, j), (b, coeff) in direct.items():
                exchange[:, i] += np.einsum('g,gai,sab,gbj->sij', coeff, b.conj(), spin[:, j], b)
        correction = -fraction/(2 if ns == 1 else 1)*xi*(s@d@s)
        ek = -.5*fraction*np.einsum('k,skij,skji->', w, spin, exchange).real
        ec = .5*np.einsum('k,skij,skji->', w, d, correction).real
        return f-fraction*exchange[:ns]+correction, (e1, ej, ex, ek, ec), populations
    return s, h, build


@pytest.mark.parametrize('species,spins', [('He', 1), ('Be', 1), ('H', 2), ('Li', 2)])
@pytest.mark.parametrize('name', ['lda', 'pbe', 'blyp'])
@pytest.mark.parametrize('mesh,shift', [((1, 1, 1), (0, 0, 0)), ((2, 1, 1), (1, 0, 0))])
def test_pure_fitted_scf_matches_independent_functional(species, spins, name, mesh, shift):
    a, k, old = case(spins=spins, species=species, name=name, mesh=mesh, shift=shift)
    check_scf(a, k, old)


@pytest.mark.parametrize('spins', [1, 2])
@pytest.mark.parametrize('profile', ['J', 'K', 'JK'])
@pytest.mark.parametrize('name', ['pbe0', 'b3lyp'])
@pytest.mark.parametrize('mesh,shift', [((1, 1, 1), (0, 0, 0)), ((2, 1, 1), (1, 0, 0))])
@pytest.mark.parametrize('correction', ['strict-zero-mode', 'BvK-ewald'])
def test_global_fitted_scf_matches_independent_functional(spins, profile, name, mesh, shift, correction):
    a, k, old = case(profile, spins, name, mesh, shift, correction=correction)
    check_scf(a, k, old)


def check_scf(a, k, old):
    r = run(a); assert r.converged and r.receipt.confirmed
    s, h, build = reference(a, k, old)
    populations = (r.plan.n_alpha,) if r.plan.spins == 1 else (r.plan.n_alpha, r.plan.n_beta)
    d = independent_scf(s, h, build, populations, r.plan.spins)
    np.testing.assert_allclose(r.density, d, atol=5e-9, rtol=0)
    np.testing.assert_allclose(r.overlap, s, atol=3e-12, rtol=0)
    np.testing.assert_allclose(r.hcore, h, atol=8e-12, rtol=0)
    f, energies, populations = build(r.density)
    if r.plan.spins == 2 and not r.plan.n_beta and a['functional'].name.lower() in ('pbe', 'pbe0', 'b3lyp'):
        # Same libxc endpoint conditioning as direct KS. Keep energies and the
        # occupied spin strict; measure the minority sensitivity independently.
        np.testing.assert_allclose(r.fock[0], f[0], atol=8e-11, rtol=0)
        base, sensitivity = minority_conditioning(a, k, r.density)
        assert np.max(abs(r.fock[1]-f[1])) <= max(1e-9, 2*sensitivity)
        if a['functional'].name.lower() == 'pbe':
            xc = manta.compute_periodic_semilocal_xc(a['basis'], a['functional'], a['system'].lattice,
                a['xc_images'], k, np.full(len(k), 1./len(k)), a['points'], a['quadrature_weights'], r.density)
            # The discrepancy is already present in the XC-only source. The
            # fitted contribution must still meet the independent matrix gate.
            np.testing.assert_allclose(r.fock[1]-f[1], xc.values[1]-base, atol=8e-11, rtol=0)
    else:
        np.testing.assert_allclose(r.fock, f, atol=1e-9, rtol=0)
    np.testing.assert_allclose(components(r), energies, atol=8e-11, rtol=0)
    np.testing.assert_allclose(r.receipt.xc.electrons_grid, populations, atol=8e-11, rtol=0)
    assert r.receipt.energy == pytest.approx(sum(_nuclear_reference(old)[0])+sum(build(d)[1]), abs=8e-11)
    wrong = sum(_nuclear_reference(old)[0])+.5*np.einsum('skij,skji->', r.density, r.hcore+r.fock).real/len(k)
    assert abs(wrong-r.receipt.energy) > .01
    assert r.plan.fitting.has_j == (r.receipt.coulomb_algorithm == 'ri')
    assert r.plan.fitting.has_k == (r.receipt.exchange_algorithm == 'ri')
    assert r.solver_memory.provider_retained_bytes >= r.plan.fitting.retained_source_bytes
    assert r.receipt.shell_calls <= r.plan.shell_calls and r.receipt.primitive_pairs <= r.plan.primitive_pairs
    assert r.receipt.primitive_points <= r.plan.primitive_points
    assert r.receipt.fock_calls == r.solver_snapshot.fock_calls+1


def minority_conditioning(a, k, d):
    """Observed one-epsilon libxc sensitivity, not an absolute accuracy bound."""
    b, g = bloch_values_gradients(dict(basis=a['basis'], lattice=a['system'].lattice,
        images=a['xc_images'], momenta=k, points=a['points']))
    w = np.full(len(k), 1./len(k)); qw = a['quadrature_weights']
    rho = np.einsum('k,kgi,skij,kgj->sg', w, b, d, b.conj()).real
    grad = (np.einsum('k,akgi,skij,kgj->sag', w, g, d, b.conj())
            +np.einsum('k,kgi,skij,akgj->sag', w, b, d, g.conj())).real
    aa = np.sum(grad[0]**2, axis=0); zero = np.zeros_like(aa)
    def minority(alpha):
        _, _, vb, _, ab, _ = a['functional'].eval_polarised(alpha, zero, aa, zero, zero)
        flow = ab*grad[0]
        return (np.einsum('g,kgi,g,kgj->kij', qw, b.conj(), vb, b)
                +np.einsum('g,akgi,ag,kgj->kij', qw, g.conj(), flow, b)
                +np.einsum('g,kgi,ag,akgj->kij', qw, b.conj(), flow, g))
    base = minority(rho[0])
    sensitivity = max(np.max(abs(minority(rho[0]*(1+sign*np.finfo(float).eps))-base)) for sign in (-1, 1))
    return base, sensitivity


@pytest.mark.parametrize('profile', ['pure', 'J', 'K', 'JK'])
@pytest.mark.parametrize('spins', [1, 2])
@pytest.mark.parametrize('field,quantity', CAPS)
def test_joint_caps_admit_all_sources_before_payload(profile, spins, field, quantity):
    a, _, _ = case(profile, spins); a['options'].ks.scf.maximum_iterations = 3
    a['budget'].other_live_bytes = 12345; p = plan(a)
    for name, bound in CAPS: setattr(a['budget'], name, getattr(p, bound))
    r = run(a); assert r.receipt.fock_calls == 3
    setattr(a['budget'], field, getattr(p, quantity)-1)
    a['points'][:] = np.nan; a['reciprocal_vectors'][:] = np.nan
    with pytest.raises((ValueError, RuntimeError), match='budget'): plan(a)
    with pytest.raises((ValueError, RuntimeError), match='budget'): run(a)


@pytest.mark.parametrize('profile', ['pure', 'J', 'K', 'JK'])
@pytest.mark.parametrize('spins', [1, 2])
def test_capped_functional_epochs_keep_fixed_sources_and_actual_density(profile, spins):
    a, k, old = case(profile, spins); s, h, build = reference(a, k, old)
    populations = (2,) if spins == 1 else (2, 1)
    d = occupy(np.repeat(h[None], spins, axis=0), s, populations, spins)
    sources, metrics = [], []
    for limit in (1, 2, 3):
        a['options'].ks.scf.maximum_iterations = limit; r = run(a)
        assert not r.converged and r.receipt.fock_calls == limit
        np.testing.assert_allclose(r.density, d, atol=8e-11, rtol=0)
        np.testing.assert_allclose(r.fock, build(d)[0], atol=8e-11, rtol=0)
        np.testing.assert_allclose(components(r), build(d)[1], atol=8e-11, rtol=0)
        sources.append(r.receipt.source_identity)
        metrics.append([m.projector_identity for m in r.receipt.fitting_info.metrics])
        d = occupy(build(d)[0], s, populations, spins)
    assert len(set(sources)) == 1 and metrics[0] == metrics[1] == metrics[2]
    assert np.max(abs(r.density-d)) > 1e-5


@pytest.mark.parametrize('profile', ['pure', 'J', 'K', 'JK'])
@pytest.mark.parametrize('spins', [1, 2])
def test_native_fock_is_independent_functional_energy_derivative(profile, spins):
    a, k, old = case(profile, spins); a['options'].ks.scf.maximum_iterations = 2
    r = run(a); _, _, build = reference(a, k, old)
    rng = np.random.default_rng(9281); v = rng.normal(size=r.density.shape)+1j*rng.normal(size=r.density.shape)
    d = r.density; adjoint = v.conj().swapaxes(-1, -2); delta = v@d+d@adjoint
    derivative = np.einsum('skij,skji->', delta, r.fock).real/len(k)
    for step in (2e-5, 1e-5):
        energies = []
        for t in (-2*step, -step, step, 2*step):
            density = (np.eye(d.shape[-1])+t*v)@d@(np.eye(d.shape[-1])+t*adjoint)
            energies.append(sum(build(density)[1]))
        finite_difference = (energies[0]-8*energies[1]+8*energies[2]-energies[3])/(12*step)
        assert finite_difference == pytest.approx(derivative, abs=5e-9)


@pytest.mark.parametrize('profile', ['pure', 'J', 'K', 'JK'])
@pytest.mark.parametrize('spins', [1, 2])
@pytest.mark.parametrize('cutoff', [10., 1e9])
def test_truncated_and_empty_projectors_keep_complete_functional(profile, spins, cutoff):
    a, k, old = case(profile, spins); a['options'].ri.absolute_cutoff = cutoff
    a['options'].ks.scf.maximum_iterations = 3; r = run(a)
    _, _, build = reference(a, k, old)
    np.testing.assert_allclose(r.fock, build(r.density)[0], atol=8e-11, rtol=0)
    np.testing.assert_allclose(components(r), build(r.density)[1], atol=8e-11, rtol=0)
    assert all(m.rank < a['auxiliary'].nbasis for m in r.receipt.fitting_info.metrics)
    assert abs(r.receipt.xc_energy) > .01
    if cutoff == 1e9:
        assert all(m.rank == 0 for m in r.receipt.fitting_info.metrics)
        assert not r.plan.fitting.has_j or r.receipt.coulomb_energy == 0.
        assert not r.plan.fitting.has_k or r.receipt.uncorrected_exchange_energy == 0.


@pytest.mark.parametrize('spins', [1, 2])
def test_source_selection_does_not_relabel_the_same_evaluated_density(spins):
    results = []
    for profile in ('pure', 'J', 'K', 'JK'):
        a, _, _ = case(profile, spins); a['options'].ks.scf.maximum_iterations = 1
        results.append(run(a))
    for r in results[1:]:
        np.testing.assert_array_equal(r.density, results[0].density)
        assert r.receipt.overlap_identity == results[0].receipt.overlap_identity
        assert r.receipt.final_density_identity == results[0].receipt.final_density_identity
    assert len({r.receipt.source_identity for r in results}) == 4
    assert len({r.receipt.final_fock_identity for r in results}) == 4


def as_fock_build(r, k):
    from vibeqc.fock_state import FockBuild, EnergyComponents, SourceProvenance
    from vibeqc.solvers.operators import OperatorFrame
    ns, nk, n = r.plan.spins, r.plan.momenta, r.plan.n_basis; receipt = r.receipt
    frames = []
    for i, s in enumerate(r.overlap):
        e, u = np.linalg.eigh(s); rank = r.solver_memory.retained_rank
        frames.append(OperatorFrame(basis_id=receipt.overlap_identity, n=n, ordering='libint-shell-axial-v1',
            image_gauge=f'native-exp(+ikR):k-index={i}', overlap=s, X=u[:, -rank:]/np.sqrt(e[-rank:])))
    energy = EnergyComponents(electronic=sum(components(r)[:-1]), nuclear=receipt.nuclear_energy,
        components=tuple(zip(('one_electron', 'coulomb', 'xc', 'uncorrected_exchange'), components(r)[:-1])),
        corrections=(('exchange_zero_mode', receipt.exchange_correction_energy),))
    source = SourceProvenance.capture('manta', native_source_identity=receipt.source_identity,
        coulomb_algorithm=receipt.coulomb_algorithm, exchange_algorithm=receipt.exchange_algorithm,
        coulomb_identity=receipt.coulomb_identity, exchange_identity=receipt.exchange_identity,
        xc_identity=receipt.xc.source_identity, fitting_identity=receipt.fitting_info.source_identity,
        projectors=[m.projector_identity for m in receipt.fitting_info.metrics])
    spins = ['restricted-total'] if ns == 1 else ['alpha', 'beta']
    return FockBuild.from_dense(method='RKS' if ns == 1 else 'UKS', generation=r.solver_snapshot.evaluated_iteration,
        source=source, energy=energy, densities=r.density.reshape(ns*nk, n, n), focks=r.fock.reshape(ns*nk, n, n),
        frames=frames*ns, spins=[s for s in spins for _ in range(nk)], kpoints=[tuple(p) for p in k]*ns,
        weights=[1./nk]*(ns*nk), kind='ks')


@pytest.mark.parametrize('profile', ['pure', 'J', 'K', 'JK'])
@pytest.mark.parametrize('spins', [1, 2])
def test_selected_provenance_and_shared_fock_keep_method_energy(monkeypatch, profile, spins):
    from vibeqc import output
    a, k, _ = case(profile, spins); a['options'].ks.scf.maximum_iterations = 3
    events = []; monkeypatch.setattr(output, 'record', lambda *args, **kw: events.append(kw['integral_provenance']))
    budget = a.pop('budget'); a['integral_engine'] = manta.IntegralEngineOptions(backend='manta', budget=budget)
    r = getattr(manta, 'run_periodic_ri_'+('ks' if profile == 'pure' else 'global_hybrid_ks'))(**a)
    e = events[-1]; json.dumps(e, allow_nan=False)
    assert e['fitted_fock']['component_only'] and not e['symmetry_reduced']
    assert e['coulomb_algorithm'] == r.receipt.coulomb_algorithm
    assert e['exchange_algorithm'] == r.receipt.exchange_algorithm
    assert (e['hartree'] is None) == r.plan.fitting.has_j
    assert e['energy'] == pytest.approx(e['nuclear_energy']+e['one_electron_energy']+e['coulomb_energy']+e['xc_energy']+e['exchange_energy'])
    assert e['citation_methods'][0] == 'manta_periodic_ri_'+('ks' if profile == 'pure' else 'global_hybrid_ks')
    assert budget.other_live_bytes == 0
    state = as_fock_build(r, k)
    assert state.energy.total == pytest.approx(r.receipt.energy, abs=1e-13)
    for channel, d, f in zip(state.channels, r.density.reshape(-1, r.plan.n_basis, r.plan.n_basis),
                             r.fock.reshape(-1, r.plan.n_basis, r.plan.n_basis)):
        np.testing.assert_array_equal(channel.density, d)
        np.testing.assert_allclose(channel.operator.apply(np.eye(r.plan.n_basis, dtype=complex)), f, atol=1e-13, rtol=0)
        assert channel.operator.metadata.symmetry_block is None
        assert channel.operator.metadata.kind == 'ks'


@pytest.mark.parametrize('profile', ['pure', 'J', 'K', 'JK'])
@pytest.mark.parametrize('spins', [1, 2])
@pytest.mark.parametrize('field,quantity', CAPS)
def test_selected_complete_native_and_output_caps(profile, spins, field, quantity, monkeypatch):
    from vibeqc import output
    from vibeqc.manta_periodic import _periodic_ri_ks_configuration
    a, _, _ = case(profile, spins); a['options'].ks.scf.maximum_iterations = 3
    p = plan(a); budget = a.pop('budget'); engine = manta.IntegralEngineOptions(backend='manta', budget=budget)
    _, _, python_bytes = _periodic_ri_ks_configuration(a['options'], engine, a['mesh'], profile != 'pure')
    limit = getattr(p, quantity)+(python_bytes if field == 'memory_bytes' else 0)
    setattr(budget, field, limit); a['integral_engine'] = engine
    calls = []; monkeypatch.setattr(output, 'record', lambda *args, **kw: calls.append(kw))
    selected = getattr(manta, 'run_periodic_ri_'+('ks' if profile == 'pure' else 'global_hybrid_ks'))
    selected(**a); assert len(calls) == 1
    setattr(budget, field, limit-1); a['points'][:] = np.nan
    with pytest.raises((ValueError, RuntimeError), match='budget'): selected(**a)
    assert len(calls) == 1


@pytest.mark.parametrize('profile', ['J', 'K', 'JK'])
@pytest.mark.parametrize('name', [EMPTY_HYBRID, 'pbe0', 'b3lyp'])
@pytest.mark.parametrize('mesh,shift', [((1, 1, 1), (0, 0, 0)), ((2, 1, 1), (1, 0, 0))])
def test_empty_spin_global_hybrids_retain_full_method_and_endpoint_evidence(profile, name, mesh, shift):
    a, k, old = case(profile, 2, name, mesh, shift, species='H')
    check_scf(a, k, old)


@pytest.mark.parametrize('profile', ['pure', 'J', 'K', 'JK'])
@pytest.mark.parametrize('spins', [1, 2])
def test_multicenter_ao_and_auxiliary_images_preserve_shared_fock_frame(profile, spins):
    from vibeqc.symmetry_shared import BlockSpaceAction, Budget, SpaceIdentity
    a, k, _ = case(profile, spins); a['options'].ks.scf.maximum_iterations = 3
    charges = (2, 1, 1) if spins == 1 else (1, 1, 1)
    atoms = [vq.Atom(z, xyz) for z, xyz in zip(charges, ([0., 0., 0.], [.9, .2, 0.], [-.2, 1.1, .1]))]
    system = vq.PeriodicSystem(3, a['system'].lattice, atoms); system.multiplicity = spins
    shells, auxiliary = a['basis'].shells(), a['auxiliary'].shells()
    for i, shell in enumerate(shells):
        shell.atom_index = i; shell.origin = atoms[i].xyz
    for shell in auxiliary:
        shell.origin = atoms[0].xyz
    a.update(system=system, translations=a['xc_images'].copy(),
        basis=vq.BasisSet(system.unit_cell_molecule(), shells, 'ri-ks-multicenter', True),
        auxiliary=vq.BasisSet(system.unit_cell_molecule(), auxiliary, 'ri-ks-multicenter-aux', True))
    original = run(a)
    shifts = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])@system.lattice.T
    for shell, delta in zip(shells, shifts):
        shell.origin = np.array(shell.origin)+delta
    for i, (shell, delta) in enumerate(zip(auxiliary, shifts[::-1])):
        shell.atom_index = i; shell.origin = np.array(shell.origin)+delta
    for key, values in (('basis', shells), ('auxiliary', auxiliary)):
        images = vq.PeriodicSystem(3, system.lattice, [vq.Atom(z, shell.origin) for z, shell in zip(charges, values)])
        images.multiplicity = spins
        a[key] = vq.BasisSet(images.unit_cell_molecule(), values, 'ri-ks-images', True)
    moved = run(a); phase = np.exp(1j*k@shifts.T); transform = phase[:, :, None]*phase[:, None, :].conj()
    for name in ('density', 'fock', 'overlap', 'hcore'):
        np.testing.assert_allclose(getattr(moved, name), getattr(original, name)*transform, atol=2e-10, rtol=0)
    assert moved.receipt.energy == pytest.approx(original.receipt.energy, abs=1e-10)
    assert np.max(abs(moved.density.imag)) > .01
    for ik in range(len(k)):
        source = SpaceIdentity('finite-ri-ks-test', original.receipt.overlap_identity, 'AO', f'original-k{ik}')
        destination = SpaceIdentity('finite-ri-ks-test', moved.receipt.overlap_identity, 'AO', f'shifted-k{ik}')
        action = BlockSpaceAction(source, destination, np.arange(4, dtype=np.int64), np.arange(3, dtype=np.int64),
            np.ascontiguousarray(phase[ik]), antiunitary=False, budget=Budget(2**20, 10**6))
        for spin in range(spins):
            np.testing.assert_allclose(action.push_density(original.density[spin, ik]), moved.density[spin, ik], atol=2e-10, rtol=0)
            np.testing.assert_allclose(action.pull_operator(moved.fock[spin, ik]), original.fock[spin, ik], atol=2e-10, rtol=0)
    state = as_fock_build(moved, k)
    assert state.energy.total == pytest.approx(original.receipt.energy, abs=1e-10)
    for channel, matrix in zip(state.channels, moved.fock.reshape(-1, 3, 3)):
        np.testing.assert_allclose(channel.operator.apply(np.eye(3, dtype=complex)), matrix, atol=1e-13, rtol=0)
        assert channel.operator.metadata.symmetry_block is None


@pytest.mark.parametrize('profile', ['pure', 'J', 'K', 'JK'])
@pytest.mark.parametrize('spins', [1, 2])
def test_rectangular_overlap_frame_preserves_full_physical_fock(profile, spins):
    from scipy.linalg import eigh
    a, k, old = case(profile, spins); a['options'].ks.scf.maximum_iterations = 3
    shells = a['basis'].shells(); shells.append(shells[0])
    a['basis'] = old['basis'] = vq.BasisSet(a['system'].unit_cell_molecule(), shells, 'ri-ks-duplicate-shell', True)
    r = run(a); s, h, build = reference(a, k, old); rank = a['basis'].nbasis-1
    x = []
    for metric in s:
        values, vectors = eigh(metric); x.append(vectors[:, -rank:]/np.sqrt(values[-rank:]))
    x = np.array(x); populations = (r.plan.n_alpha,) if spins == 1 else (r.plan.n_alpha, r.plan.n_beta)
    def projector(f):
        reduced = x.conj().swapaxes(-1, -2)@f@x
        return x@occupy(reduced, np.repeat(np.eye(rank)[None], len(k), axis=0), populations, spins)@x.conj().swapaxes(-1, -2)
    d = projector(np.repeat(h[None], spins, axis=0))
    for _ in range(2): d = projector(build(d)[0])
    np.testing.assert_allclose(r.density, d, atol=2e-10, rtol=0)
    np.testing.assert_allclose(r.fock, build(r.density)[0], atol=8e-11, rtol=0)
    np.testing.assert_allclose(components(r), build(r.density)[1], atol=8e-11, rtol=0)
    assert r.coefficients.shape == (spins, len(k), rank+1, rank)
    state = as_fock_build(r, k)
    expected = ([2*r.plan.n_alpha]*len(k) if spins == 1 else
                [r.plan.n_alpha]*len(k)+[r.plan.n_beta]*len(k))
    np.testing.assert_allclose(state.metric_electron_counts, expected, atol=1e-12, rtol=0)
    for index, channel in enumerate(state.channels):
        op = channel.operator; assert op.frame.X.shape == (rank+1, rank)
        reduced = op.X.conj().T@op.apply(op.X)
        eps, _ = eigh(reduced)
        np.testing.assert_allclose(eps, r.orbital_energies.reshape(-1, rank)[index], atol=8e-11, rtol=0)


@pytest.mark.parametrize('profile', ['pure', 'J', 'K', 'JK'])
@pytest.mark.parametrize('name', ['scan', 'b2plyp', 'hse06', 'wb97x-v'])
def test_incomplete_or_physical_range_functionals_refuse_before_payload(profile, name):
    a, _, _ = case(profile, 2, name); a['points'][:] = np.nan
    with pytest.raises(ValueError, match='complete native'): plan(a)
    with pytest.raises(ValueError, match='complete native'): run(a)


@pytest.mark.parametrize('algorithms', [('exact', 'exact'), ('bup', 'ri'), ('ri', 'cosx'), ('ri-bupo', 'exact')])
def test_unqualified_global_source_pairs_refuse_before_payload(algorithms):
    a, _, _ = case('JK'); a['options'].coulomb_algorithm, a['options'].exchange_algorithm = algorithms
    a['points'][:] = np.nan
    with pytest.raises(ValueError, match='at least one RI'): plan(a)
    with pytest.raises(ValueError, match='at least one RI'): run(a)


@pytest.mark.parametrize('profile', ['pure', 'J', 'K', 'JK'])
@pytest.mark.parametrize('cutoff', [None, 2.])
def test_physical_and_fitting_pair_supports_must_match(profile, cutoff):
    a, _, _ = case(profile); a['options'].ks.pair_cutoff_bohr = cutoff; a['points'][:] = np.nan
    with pytest.raises(ValueError, match='identical explicit'): plan(a)
    with pytest.raises(ValueError, match='identical explicit'): run(a)


@pytest.mark.parametrize('profile', ['pure', 'J', 'K', 'JK'])
def test_layout_batch_and_mode_blocks_do_not_relabel_sources(profile):
    a, _, _ = case(profile, 2); a['options'].ks.scf.maximum_iterations = 3; original = run(a)
    for key, value in list(a.items()):
        if isinstance(value, np.ndarray): a[key] = np.asfortranarray(value)
    a['options'].ri.mode_block = 1; a['options'].ks.grid_batch = 7; changed = run(a)
    np.testing.assert_allclose(changed.fock, original.fock, atol=8e-11, rtol=0)
    assert changed.receipt.source_identity == original.receipt.source_identity
    # Exports own their storage; mutating one export cannot rewrite the result.
    fock = changed.fock; fock[:] = np.nan
    np.testing.assert_allclose(changed.fock, original.fock, atol=8e-11, rtol=0)


@pytest.mark.parametrize('profile', ['pure', 'J', 'K', 'JK'])
def test_restricted_and_equal_spin_limits_keep_distinct_method_leaves(profile):
    a, _, _ = case(profile, 1); r = run(a)
    a['functional'] = vq.Functional(a['functional'].name, spin=2); u = run(a)
    assert r.converged and u.converged
    np.testing.assert_allclose(u.density, np.repeat(.5*r.density, 2, axis=0), atol=5e-9, rtol=0)
    np.testing.assert_allclose(u.fock, np.repeat(r.fock, 2, axis=0), atol=2e-9, rtol=0)
    np.testing.assert_allclose(components(u), components(r), atol=8e-11, rtol=0)
    assert u.receipt.energy == pytest.approx(r.receipt.energy, abs=8e-11)
    assert r.plan.spins == 1 and u.plan.spins == 2
    assert r.receipt.fitting_info.source_identity == u.receipt.fitting_info.source_identity
    assert r.receipt.source_identity != u.receipt.source_identity


@pytest.mark.parametrize('profile', ['pure', 'J', 'K', 'JK'])
def test_selected_scientific_output_serializes_actual_components_and_citations(tmp_path, profile):
    from vibeqc.output import OutputChannel, structured_log
    from vibeqc.output.citations import load_default_database
    a, _, _ = case(profile, 2); a['options'].ks.scf.maximum_iterations = 3
    budget = a.pop('budget'); a['integral_engine'] = manta.IntegralEngineOptions(backend='manta', budget=budget)
    selected = getattr(manta, 'run_periodic_ri_'+('ks' if profile == 'pure' else 'global_hybrid_ks'))
    with OutputChannel.to_file(tmp_path/'science.out'), structured_log(tmp_path/'science.jsonl'):
        r = selected(**a)
    event = json.loads((tmp_path/'science.jsonl').read_text()); e = event['integral_provenance']
    assert e['energy'] == r.receipt.energy and e['operator_kind'] == 'ks'
    assert e['final_density_identity'] == r.receipt.final_density_identity
    assert e['fitting']['metrics'][0]['projector_identity'] == r.receipt.fitting_info.metrics[0].projector_identity
    assert e['fitted_fock']['component_only'] and not e['converged']
    text = (tmp_path/'science.out').read_text()
    assert f'J={r.receipt.coulomb_algorithm}; K={r.receipt.exchange_algorithm}' in text
    db = load_default_database()
    keys = {citation.key for method in e['citation_methods']
            for citation in db.assemble(method=method, numerics=e['citation_numerics']).citations}
    assert {'sun_berkelbach_gdf_2017', 'ye_berkelbach_rsgdf_2021', 'libxc_2018', 'neumann_semilocal_1996'} <= keys


@pytest.mark.parametrize('profile', ['pure', 'J'])
def test_direct_entry_points_cannot_discard_fitted_options(profile):
    a, _, _ = case(profile); a.pop('auxiliary'); a['points'][:] = np.nan
    method = 'ks' if profile == 'pure' else 'global_hybrid_ks'
    with pytest.raises(TypeError, match='incompatible function arguments'):
        getattr(core._manta, 'run_periodic_'+method)(**a)
    counts = [a[n] for n in ('basis', 'system', 'mesh', 'functional')]+[len(a['reciprocal_vectors'])]
    if profile != 'pure': counts.append(int(max(np.diff(a['transfer_offsets']))))
    counts += [len(a[n]) for n in ('translations', 'xc_images', 'points')]+[a['options'], a['budget']]
    with pytest.raises(TypeError, match='incompatible function arguments'):
        getattr(manta, 'plan_periodic_'+method)(*counts)
