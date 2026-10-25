"""Mixed exact/RI HF: independent finite sources, real method leaves and output."""
import dataclasses
import json
import tomllib

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core
from vibeqc.bipole_fock_ewald import probe_charge_madelung_supercell
from vibeqc.guess_read import _spin_density_k_from_qvf_bloch_payload
from tests.test_manta_periodic_exchange import _quadrature_bloch
from tests.test_manta_periodic_gdf import legacy
from tests.test_manta_periodic_rhf import _components, _nuclear_reference
from tests.test_manta_periodic_uhf import _occupy, _scf_reference
from tests.test_manta_periodic_ri_uhf import case as spin_case
from tests.test_manta_periodic_ri_rhf import case as restricted_case, scf_reference, occupy, CAPS
from tests.test_manta_periodic_ri_rhf import job_case as restricted_job
from tests.test_manta_periodic_ri_job import job_case as spin_job, _as_fock_build
from tests.test_manta_periodic_ri_domains import explicit
from tests.test_manta_periodic_job import _no_computed_artifacts


def case(method, target, mesh=(2, 1, 1), shift=(1, 0, 0), species=None, correction='BvK-ewald'):
    species = species or ('Be' if method == 'rhf' else 'Li')
    a, k = (restricted_case if method == 'rhf' else spin_case)(mesh, shift, species, correction)
    a['options'].coulomb_algorithm = 'ri' if target == 'J' else 'exact'
    a['options'].exchange_algorithm = 'ri' if target == 'K' else 'exact'
    if target == 'J':
        pieces = []
        for i in range(len(k)):
            for j in range(len(k)):
                lo, hi = a['transfer_offsets'][i*len(k)+j:i*len(k)+j+2]
                pieces.append(a['reciprocal_vectors'].copy() if i == j else -a['pair_vectors'][lo:hi])
        a['pair_vectors'] = np.concatenate(pieces)
        a['transfer_offsets'] = np.r_[0, np.cumsum([len(p) for p in pieces])].astype(np.int64)
    return a, k


def cutoff_case(*args, **kwargs):
    a, k = case(*args, **kwargs)
    return dict({n: a[n] for n in ('basis', 'auxiliary', 'system', 'mesh', 'options', 'budget')},
                reciprocal_energy_cutoff=6.), k


def reference(a, k, method):
    """Libint raw fits, Gaussian quadrature exact terms and SciPy occupations."""
    o = a['options']; nk = len(k); w = np.full(nk, 1./nk)
    fit_j, fit_k = o.coulomb_algorithm == 'ri', o.exchange_algorithm == 'ri'
    direct_q = -a['pair_vectors'] if fit_k else a['pair_vectors']
    s, h, _ = _components(dict(a, exchange_vectors=direct_q), k)
    g = a['reciprocal_vectors']; volume = abs(np.linalg.det(a['system'].lattice))
    common = dict(orbital=a['basis'], auxiliary=a['auxiliary'], system=a['system'], options=o.fitting, budget=a['budget'])
    def factor(q, ket, vectors):
        m, t, _ = legacy(dict(common, q=q, ket_kpoints=ket, vectors=vectors))
        m = np.tril(m)+np.tril(m, -1).conj().T
        m[np.diag_indices(len(m))] = m.diagonal().real
        e, u = np.linalg.eigh(m)
        keep = e > max(o.ri.absolute_cutoff, o.ri.relative_cutoff*max(0., e[-1]))
        return np.einsum('pl,kpmn->klmn', (u[:, keep]/np.sqrt(e[keep])).conj(), t)
    gamma = factor(np.zeros(3), k, g); pairs = {}; exact = {}
    for i in range(nk):
        for j in range(nk):
            lo, hi = a['transfer_offsets'][i*nk+j:i*nk+j+2]
            vectors = a['pair_vectors'][lo:hi]
            if fit_k and i != j:
                pairs[i, j] = factor(k[j]-k[i], k[j:j+1], vectors)[0]
            elif not fit_k:
                exact[i, j] = (_quadrature_bloch(a['basis'], vectors, a['translations'], k[i], o.pair_cutoff_bohr),
                               w[j]*4*np.pi/volume/np.sum(vectors*vectors, axis=1))
    fj = np.stack([_quadrature_bloch(a['basis'], g, a['translations'], -point, o.pair_cutoff_bohr) for point in k])
    coulomb_kernel = 4*np.pi/volume/np.sum(g*g, axis=1)
    xi = 0. if o.exchange_q0 == 'strict-zero-mode' else probe_charge_madelung_supercell(
        a['system'], tuple(a['mesh'].mesh), alpha=o.madelung_alpha, precision=1e-14)
    def build(d):
        spin = np.stack([.5*d, .5*d]) if method == 'rhf' else d
        if fit_j:
            rho = np.einsum('k,klmn,skmn->l', w, gamma.conj(), spin)
            j = np.einsum('klmn,l->kmn', gamma, rho); ej = .5*np.vdot(rho, rho).real
        else:
            rho = np.einsum('k,kmn,kgmn->g', w, spin.sum(axis=0), fj)
            j = np.einsum('g,g,kgmn->kmn', coulomb_kernel, rho, fj.conj())
            ej = .5*np.dot(coulomb_kernel, abs(rho)**2)
        if fit_k:
            exchange = np.einsum('k,klmp,skpq,klnq->skmn', w, gamma, spin, gamma.conj())
            for (i, q), b in pairs.items():
                exchange[:, i] += w[q]*np.einsum('lmp,spq,lnq->smn', b, spin[:, q], b.conj())
        else:
            exchange = np.zeros_like(spin)
            for (i, q), (b, coefficient) in exact.items():
                exchange[:, i] += np.einsum('g,gai,sab,gbj->sij', coefficient, b.conj(), spin[:, q], b)
        correction = -xi*(s@spin@s)
        energy = (np.einsum('k,skij,kji->', w, spin, h).real, ej,
            -.5*np.einsum('k,skij,skji->', w, spin, exchange).real,
            .5*np.einsum('k,skij,skji->', w, spin, correction).real)
        f = h+j-exchange+correction
        if method == 'rhf':
            np.testing.assert_array_equal(f[0], f[1]); f = f[0]
        return f, energy
    return s, h, build


def solve_reference(a, k, method):
    s, h, build = reference(a, k, method)
    electrons = sum(atom.Z for atom in a['system'].unit_cell)
    if method == 'rhf':
        d = scf_reference(s, h, build, electrons//2)
    else:
        alpha = (electrons+a['system'].multiplicity-1)//2
        d = _scf_reference(s, h, build, (alpha, electrons-alpha))
    return d, s, h, build


def plan(a, method):
    return getattr(manta, 'plan_periodic_ri_'+method)(a['basis'], a['auxiliary'], a['system'], a['mesh'],
        len(a['reciprocal_vectors']), int(max(np.diff(a['transfer_offsets']))), len(a['translations']), a['options'], a['budget'])


@pytest.mark.parametrize('method,species', [('rhf', 'He'), ('rhf', 'Be'), ('uhf', 'H'), ('uhf', 'Li')])
@pytest.mark.parametrize('target', ['J', 'K'])
@pytest.mark.parametrize('mesh,shift', [((1, 1, 1), (0, 0, 0)), ((2, 1, 1), (0, 0, 0)), ((2, 1, 1), (1, 0, 0))])
@pytest.mark.parametrize('correction', ['strict-zero-mode', 'BvK-ewald'])
def test_physical_scf_and_each_component_match_independent_mixed_sources(method, species, target, mesh, shift, correction):
    a, k = case(method, target, mesh, shift, species, correction)
    r = getattr(core._manta, 'run_periodic_ri_'+method)(**a)
    assert r.converged and r.receipt.confirmed
    d, s, h, build = solve_reference(a, k, method)
    np.testing.assert_allclose(r.density, d, atol=5e-9, rtol=0)
    np.testing.assert_allclose(r.overlap, s, atol=3e-12, rtol=0)
    np.testing.assert_allclose(r.hcore, h, atol=8e-12, rtol=0)
    np.testing.assert_allclose(r.fock, build(r.density)[0], atol=8e-11, rtol=0)
    np.testing.assert_allclose([getattr(r.receipt, n) for n in ('one_electron_energy', 'coulomb_energy',
        'uncorrected_exchange_energy', 'exchange_correction_energy')], build(r.density)[1], atol=8e-11, rtol=0)
    assert r.receipt.energy == pytest.approx(sum(_nuclear_reference(a)[0])+sum(build(d)[1]), abs=8e-11)
    assert r.receipt.algorithm == 'finite-3d-mixed-ri-'+method
    assert r.receipt.coulomb_algorithm == a['options'].coulomb_algorithm
    assert r.receipt.exchange_algorithm == a['options'].exchange_algorithm
    assert r.plan.fock.has_j == (target == 'J') and r.plan.fock.has_k == (target == 'K')
    assert r.plan.fock.channels == (1 if target == 'J' else 1+len(k)*(len(k)-1))
    assert (r.receipt.exact_hartree is None) == (target == 'J')
    assert (r.receipt.exact_exchange is None) == (target == 'K')
    assert r.receipt.shell_calls <= r.plan.shell_calls and r.receipt.primitive_pairs <= r.plan.primitive_pairs
    assert r.plan.two_electron_peak_bytes > r.plan.fock.modeled_peak_bytes


@pytest.mark.parametrize('method', ['rhf', 'uhf'])
@pytest.mark.parametrize('target', ['J', 'K'])
def test_every_capped_epoch_matches_independent_recurrence(method, target):
    a, k = case(method, target); s, h, build = reference(a, k, method)
    fill = (lambda f: occupy(f, s, 2)) if method == 'rhf' else (lambda f: _occupy(f, s, (2, 1)))
    d = fill(h if method == 'rhf' else np.stack([h, h]))
    identities = []; projectors = []
    for limit in (1, 2, 3):
        a['options'].scf.maximum_iterations = limit
        r = getattr(core._manta, 'run_periodic_ri_'+method)(**a)
        assert not r.converged and r.receipt.fock_calls == limit
        np.testing.assert_allclose(r.density, d, atol=8e-11, rtol=0)
        np.testing.assert_allclose(r.fock, build(d)[0], atol=8e-11, rtol=0)
        identities.append(r.receipt.source_identity)
        projectors.append([m.projector_identity for m in r.receipt.fitting_info.metrics])
        d = fill(build(d)[0])
    assert len(set(identities)) == 1 and projectors[0] == projectors[1] == projectors[2]


@pytest.mark.parametrize('method', ['rhf', 'uhf'])
@pytest.mark.parametrize('target', ['J', 'K'])
@pytest.mark.parametrize('generated', [False, True])
@pytest.mark.parametrize('field,quantity', CAPS)
def test_complete_native_admission_counts_both_components(method, target, generated, field, quantity):
    if generated:
        a, _ = cutoff_case(method, target)
        planner = getattr(manta, 'plan_periodic_ri_'+method+'_from_cutoffs')
    else:
        a, _ = case(method, target); planner = lambda **kw: plan(kw, method)
    run = getattr(core._manta, 'run_periodic_ri_'+method+('_from_cutoffs' if generated else ''))
    a['budget'].other_live_bytes = 12345
    p = planner(**a)
    for name, planned in CAPS:
        setattr(a['budget'], name, getattr(p, planned))
    r = run(**a); assert r.converged
    setattr(a['budget'], field, getattr(p, quantity)-1)
    if not generated:
        a['reciprocal_vectors'][:] = np.nan
    with pytest.raises((ValueError, RuntimeError), match='budget'):
        planner(**a)
    with pytest.raises((ValueError, RuntimeError), match='budget'):
        run(**a)


def job_case(directory, method, target, correction='BvK-ewald'):
    make = restricted_job if method == 'rhf' else spin_job
    job, a, k = make(directory, (2, 1, 1), (1, 0, 0), 'Be' if method == 'rhf' else 'Li', correction)
    job['coulomb'] = dataclasses.replace(job['coulomb'], algorithm='ri' if target == 'J' else 'exact')
    job['exchange'] = manta.ExchangeBuildOptions(algorithm='ri' if target == 'K' else 'exact')
    a['options'].coulomb_algorithm = job['coulomb'].algorithm
    a['options'].exchange_algorithm = job['exchange'].algorithm
    return job, a, k


@pytest.mark.parametrize('method', ['rhf', 'uhf'])
@pytest.mark.parametrize('target', ['J', 'K'])
@pytest.mark.parametrize('correction', ['strict-zero-mode', 'BvK-ewald'])
@pytest.mark.parametrize('limit', [3, 100])
def test_public_output_and_shared_fock_retain_the_exact_evaluated_mixed_state(tmp_path, method, target, correction, limit):
    job, a, k = job_case(tmp_path, method, target, correction); job['max_iter'] = limit
    r = vq.run_periodic_job(**job); evidence = r.integral_provenance
    assert r.converged == (limit == 100)
    json.dumps(evidence, allow_nan=False)
    domains = getattr(manta, 'build_periodic_ri_'+method+'_domains')(**a)
    c = explicit(a, domains); _, _, build = reference(c, k, method)
    np.testing.assert_allclose(r.native.fock, build(r.native.density)[0], atol=8e-11, rtol=0)
    assert r.energy == pytest.approx(sum(_nuclear_reference(c)[0])+sum(build(r.native.density)[1]), abs=8e-11)
    state = _as_fock_build(r)
    assert state.energy.total == pytest.approx(r.energy, abs=1e-13)
    source = state.source.as_dict()['details']
    for name in ('coulomb_algorithm', 'exchange_algorithm', 'coulomb_identity', 'exchange_identity'):
        assert source[name] == evidence[name]
    assert all(ch.operator.metadata.symmetry_block is None for ch in state.channels)
    d = r.native.density
    saved = _spin_density_k_from_qvf_bloch_payload(job['output'].with_suffix('.qvf'))
    np.testing.assert_array_equal(saved, np.stack([.5*d, .5*d]) if method == 'rhf' else d)
    assert evidence['coulomb_identity'] == r.native.receipt.coulomb_identity
    assert evidence['exchange_identity'] == r.native.receipt.exchange_identity
    assert evidence['fitting']['has_j'] == (target == 'J') and evidence['fitting']['has_k'] == (target == 'K')
    direct = 'exact_exchange' if target == 'J' else 'exact_hartree'
    assert evidence[direct]['source_identity'] == getattr(r.native.receipt, direct).source_identity
    manifest = tomllib.loads(job['output'].with_suffix('.system').read_text())['run']
    assert manifest['jk_method_resolved'] == 'mixed-exact-ri'
    assert manifest['jk_method_executed'] == 'finite-3d-mixed-ri-'+method
    assert manifest['coulomb_source_identity'] == evidence['coulomb_identity']
    assert manifest['exchange_source_identity'] == evidence['exchange_identity']
    output = job['output'].with_suffix('.out').read_text()
    assert ('J=RI; K=exact reciprocal' if target == 'J' else 'J=exact reciprocal; K=RI') in output


@pytest.mark.parametrize('method', ['rhf', 'uhf'])
@pytest.mark.parametrize('target', ['J', 'K'])
@pytest.mark.parametrize('field,quantity', CAPS)
def test_public_admission_includes_source_and_output_owners(tmp_path, method, target, field, quantity):
    job, a, _ = job_case(tmp_path/'probe', method, target)
    first = vq.run_periodic_job(**job)
    p = getattr(manta, 'plan_periodic_ri_'+method+'_from_cutoffs')(**a)
    limit = getattr(p, quantity)
    if field == 'memory_bytes':
        limit += first.integral_provenance['python_reserved_bytes']+first.integral_provenance['job_reserved_bytes']
    setattr(job['integral_engine'].budget, field, limit); job['output'] = tmp_path/'exact'/'job'
    assert vq.run_periodic_job(**job).converged
    setattr(job['integral_engine'].budget, field, limit-1); job['output'] = tmp_path/'refused'/'job'
    with pytest.raises(ValueError, match='budget'):
        vq.run_periodic_job(**job)
    _no_computed_artifacts(job)


@pytest.mark.parametrize('method', ['rhf', 'uhf'])
def test_unspecified_exchange_preserves_the_exact_default(tmp_path, method):
    job, _, _ = job_case(tmp_path, method, 'J'); job['exchange'] = None
    r = vq.run_periodic_job(**job)
    assert r.native.receipt.exchange_algorithm == 'exact'
    assert r.native.plan.fock.has_j and not r.native.plan.fock.has_k


@pytest.mark.parametrize('method', ['rhf', 'uhf'])
@pytest.mark.parametrize('algorithms', [('exact', 'exact'), ('ri', 'none'), ('bupo', 'ri')])
def test_invalid_native_source_selections_refuse(method, algorithms):
    a, _ = case(method, 'J')
    a['options'].coulomb_algorithm, a['options'].exchange_algorithm = algorithms
    with pytest.raises(ValueError, match='at least one RI target'):
        plan(a, method)


@pytest.mark.parametrize('method', ['rhf', 'uhf'])
@pytest.mark.parametrize('target', ['J', 'K'])
@pytest.mark.parametrize('cutoff', [10., 1e9])
def test_truncated_and_empty_fits_preserve_complementary_exact_response(method, target, cutoff):
    a, k = case(method, target, species='He' if method == 'rhf' else 'H')
    a['options'].ri.absolute_cutoff = cutoff
    a['options'].scf.maximum_iterations = 3
    r = getattr(core._manta, 'run_periodic_ri_'+method)(**a)
    _, _, build = reference(a, k, method)
    np.testing.assert_allclose(r.fock, build(r.density)[0], atol=8e-11, rtol=0)
    assert r.receipt.energy == pytest.approx(sum(_nuclear_reference(a)[0])+sum(build(r.density)[1]), abs=8e-11)
    assert all(m.rank < a['auxiliary'].nbasis for m in r.receipt.fitting_info.metrics)
    if cutoff == 1e9:
        assert all(m.rank == 0 for m in r.receipt.fitting_info.metrics)
        assert (r.receipt.coulomb_energy if target == 'J' else r.receipt.uncorrected_exchange_energy) == 0.
    direct = r.receipt.exact_exchange if target == 'J' else r.receipt.exact_hartree
    assert direct.shell_calls > 0


@pytest.mark.parametrize('method', ['rhf', 'uhf'])
@pytest.mark.parametrize('target', ['J', 'K'])
def test_multicenter_image_gauges_preserve_mixed_fock_and_nuclear_frame_qvf(tmp_path, method, target):
    job, a, k = job_case(tmp_path/'original', method, target)
    charges = (2, 1, 1) if method == 'rhf' else (1, 1, 1)
    atoms = [vq.Atom(z, xyz) for z, xyz in zip(charges, ([0., 0., 0.], [.9, .2, 0.], [-.2, 1.1, .1]))]
    system = vq.PeriodicSystem(3, a['system'].lattice, atoms)
    system.multiplicity = 1 if method == 'rhf' else 2
    shells, aux = a['basis'].shells(), a['auxiliary'].shells()
    for i, shell in enumerate(shells):
        shell.atom_index = i; shell.origin = atoms[i].xyz
    for shell in aux:
        shell.origin = atoms[0].xyz
    job.update(system=system, basis=vq.BasisSet(system.unit_cell_molecule(), shells, 'mixed-multicenter', True), max_iter=3)
    job['coulomb'] = dataclasses.replace(job['coulomb'], auxiliary_basis=vq.BasisSet(
        system.unit_cell_molecule(), aux, 'mixed-multicenter-aux', True))
    original = vq.run_periodic_job(**job)
    shifts = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])@system.lattice.T
    for shell, delta in zip(shells, shifts):
        shell.origin = np.array(shell.origin)+delta
    images = vq.PeriodicSystem(3, system.lattice, [vq.Atom(z, shell.origin) for z, shell in zip(charges, shells)])
    images.multiplicity = system.multiplicity
    job['basis'] = vq.BasisSet(images.unit_cell_molecule(), shells, 'mixed-ao-images', True)
    for i, (shell, delta) in enumerate(zip(aux, shifts[::-1])):
        shell.atom_index = i; shell.origin = np.array(shell.origin)+delta
    images = vq.PeriodicSystem(3, system.lattice, [vq.Atom(z, shell.origin) for z, shell in zip(charges, aux)])
    images.multiplicity = system.multiplicity
    job['coulomb'] = dataclasses.replace(job['coulomb'], auxiliary_basis=vq.BasisSet(images.unit_cell_molecule(), aux, 'mixed-aux-images', True))
    job['output'] = tmp_path/'images'/'job'; moved = vq.run_periodic_job(**job)
    phase = np.exp(1j*k@shifts.T); transform = phase[:, :, None]*phase[:, None, :].conj()
    for name in ('density', 'fock', 'overlap', 'hcore'):
        np.testing.assert_allclose(getattr(moved.native, name), getattr(original.native, name)*transform, atol=2e-10, rtol=0)
    assert moved.energy == pytest.approx(original.energy, abs=1e-10)
    assert np.max(abs(moved.native.density.imag)) > .01
    state = _as_fock_build(moved)
    for channel, matrix in zip(state.channels, moved.native.fock.reshape(-1, 3, 3)):
        np.testing.assert_allclose(channel.operator.apply(np.eye(3, dtype=complex)), matrix, atol=1e-13, rtol=0)
    saved = _spin_density_k_from_qvf_bloch_payload(job['output'].with_suffix('.qvf'))
    expected = np.stack([.5*original.native.density]*2) if method == 'rhf' else original.native.density
    np.testing.assert_allclose(saved, expected, atol=1e-10, rtol=0)


@pytest.mark.parametrize('method', ['rhf', 'uhf'])
@pytest.mark.parametrize('target', ['J', 'K'])
def test_selected_cutoff_domains_keep_the_requested_exchange_orientation(method, target):
    a, k = cutoff_case(method, target)
    domains = getattr(manta, 'build_periodic_ri_'+method+'_domains')(**a)
    nk = len(k); inverse = np.linalg.inv(a['system'].reciprocal_lattice())
    for i in range(nk):
        for j in range(nk):
            lo, hi = domains.transfer_offsets[i*nk+j:i*nk+j+2]
            q = domains.pair_vectors[lo:hi]
            if i == j:
                if target == 'K':
                    assert len(q) == 0
                else:
                    np.testing.assert_array_equal(q, domains.reciprocal_vectors)
            else:
                transfer = k[j]-k[i] if target == 'K' else k[i]-k[j]
                labels = (q-transfer)@inverse.T
                np.testing.assert_allclose(labels, np.round(labels), atol=3e-14, rtol=0)
    result = getattr(core._manta, 'run_periodic_ri_'+method+'_from_cutoffs')(**a)
    direct = getattr(core._manta, 'run_periodic_ri_'+method)(**explicit(a, domains))
    np.testing.assert_array_equal(result.density, direct.density)
    np.testing.assert_array_equal(result.fock, direct.fock)
    assert result.receipt.source_identity == direct.receipt.source_identity
    assert ('ri-ket-minus-bra' if target == 'K' else 'exact-target-minus-source') in result.receipt.domain_selection
