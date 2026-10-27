"""Physical fitted spin SCF against libint sources and independent NumPy RI."""
import itertools

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core
from vibeqc.bipole_fock_ewald import probe_charge_madelung_supercell
from tests.test_manta_periodic_gdf import legacy
from tests.test_manta_periodic_rhf import _components, _nuclear_reference
from tests.test_manta_periodic_uhf import _case as uhf_case, _occupy, _scf_reference


def case(mesh=(1, 1, 1), shift=(0, 0, 0), species='Li', correction='strict-zero-mode'):
    c, k = uhf_case(mesh, shift, species, correction)
    o = manta.PeriodicRIUHFOptions()
    for name in ('alpha', 'nuclear_cutoff_bohr', 'pair_cutoff_bohr', 'reciprocal_batch', 'exchange_q0'):
        setattr(o, name, getattr(c['options'], name))
    o.scf.maximum_iterations = 100
    o.fitting.pair_cutoff_bohr = o.pair_cutoff_bohr
    o.fitting.auxiliary_cutoff_bohr = 8.
    o.fitting.omega = .65
    o.ri.mode_block = 2
    # Wider reciprocal/image domains are necessary for a positive finite
    # metric. No negative eigenvalue is hidden by changing the refusal limit.
    base = np.array(list(itertools.product(range(-2, 3), repeat=3))) @ c['system'].reciprocal_lattice().T
    g = base[np.any(base != 0., axis=1)].copy()
    pieces = [base + source - target if i != j else np.empty((0, 3))
              for i, target in enumerate(k) for j, source in enumerate(k)]
    c.pop('exchange_vectors')
    c.update(reciprocal_vectors=g, pair_vectors=np.ascontiguousarray(np.concatenate(pieces)),
             transfer_offsets=np.r_[0, np.cumsum([len(a) for a in pieces])].astype(np.int64), options=o)
    exponents = (.3, .5, 1.05, 1.6) if species == 'H' else (.4, 1.4, 4.8)
    shells = [core.ShellInfo(0, 0, True, [e], [1.], c['system'].unit_cell[0].xyz) for e in exponents]
    c['auxiliary'] = vq.BasisSet(c['system'].unit_cell_molecule(), shells, 'periodic-ri-uhf-test', True)
    for name in ('shell_calls', 'primitive_quartets', 'scalar_terms', 'kernel_terms'):
        setattr(c['budget'], name, 2**64 - 1)
    c['budget'].memory_bytes = 2**30
    return c, k


def reference(c, k):
    """Fixed libint metric/factors; the iteration uses no MANTA contractions."""
    s, h, _ = _components(dict(c, exchange_vectors=-c['pair_vectors']), k)
    o = c['options']; nk = len(k); w = np.ones(nk) / nk
    args = dict(orbital=c['basis'], auxiliary=c['auxiliary'], system=c['system'],
                options=o.fitting, budget=c['budget'])

    def factor(q, ket, vectors):
        m, t, _ = legacy(dict(args, q=q, ket_kpoints=ket, vectors=vectors))
        m = np.tril(m) + np.tril(m, -1).conj().T
        m[np.diag_indices(len(m))] = m.diagonal().real
        e, u = np.linalg.eigh(m)
        keep = e > max(o.ri.absolute_cutoff, o.ri.relative_cutoff * max(0., e[-1]))
        return np.einsum('pl,kpmn->klmn', (u[:, keep] / np.sqrt(e[keep])).conj(), t)

    gamma = factor(np.zeros(3), k, c['reciprocal_vectors'])
    pairs = {}
    for i in range(nk):
        for j in range(nk):
            if i == j:
                continue
            lo, hi = c['transfer_offsets'][i*nk+j:i*nk+j+2]
            pairs[i, j] = factor(k[j]-k[i], k[j:j+1], c['pair_vectors'][lo:hi])[0]
    xi = 0. if o.exchange_q0 == 'strict-zero-mode' else probe_charge_madelung_supercell(
        c['system'], tuple(c['mesh'].mesh), alpha=o.madelung_alpha, precision=1e-14)

    def build(d):
        rho = np.einsum('k,klmn,skmn->l', w, gamma.conj(), d)
        j = np.einsum('klmn,l->kmn', gamma, rho)
        exchange = np.einsum('k,klmp,skpq,klnq->skmn', w, gamma, d, gamma.conj())
        for (i, q), b in pairs.items():
            exchange[:, i] += w[q] * np.einsum('lmp,spq,lnq->smn', b, d[:, q], b.conj())
        correction = -xi * (s @ d @ s)
        e1 = np.einsum('k,skij,kji->', w, d, h).real
        ej = .5 * np.vdot(rho, rho).real
        ek = -.5 * np.einsum('k,skij,skji->', w, d, exchange).real
        ec = .5 * np.einsum('k,skij,skji->', w, d, correction).real
        return h + j - exchange + correction, (e1, ej, ek, ec)

    return s, h, build, xi


def plan(c):
    return manta.plan_periodic_ri_uhf(c['basis'], c['auxiliary'], c['system'], c['mesh'],
        len(c['reciprocal_vectors']), int(max(np.diff(c['transfer_offsets']))),
        len(c['translations']), c['options'], c['budget'])


@pytest.mark.parametrize('species', ['H', 'Li'])
@pytest.mark.parametrize('mesh,shift', [((1, 1, 1), (0, 0, 0)), ((2, 1, 1), (0, 0, 0)), ((2, 1, 1), (1, 0, 0))])
@pytest.mark.parametrize('correction', ['strict-zero-mode', 'BvK-ewald'])
def test_independent_scf_density_fock_and_every_energy_component(species, mesh, shift, correction):
    c, k = case(mesh, shift, species, correction)
    r = core._manta.run_periodic_ri_uhf(**c)
    assert r.converged and r.receipt.confirmed
    populations = (r.plan.n_alpha, r.plan.n_beta)
    assert populations == ((1, 0) if species == 'H' else (2, 1))
    s, h, build, xi = reference(c, k)
    np.testing.assert_allclose(r.momenta, k, atol=1e-15, rtol=0)
    d = _scf_reference(s, h, build, populations)
    np.testing.assert_allclose(r.overlap, s, atol=3e-12, rtol=0)
    np.testing.assert_allclose(r.hcore, h, atol=3e-12, rtol=0)
    np.testing.assert_allclose(r.density, d, atol=5e-9, rtol=0)
    f, energies = build(r.density)
    np.testing.assert_allclose(r.fock, f, atol=5e-11, rtol=0)
    np.testing.assert_allclose([r.receipt.one_electron_energy, r.receipt.coulomb_energy,
        r.receipt.uncorrected_exchange_energy, r.receipt.exchange_correction_energy], energies, atol=5e-11, rtol=0)
    nuclear, count = _nuclear_reference(c)
    assert r.receipt.nuclear_pair_images == count
    assert r.receipt.nuclear_energy == pytest.approx(sum(nuclear), abs=3e-12)
    assert r.receipt.energy == pytest.approx(sum(nuclear) + sum(build(d)[1]), abs=8e-11)
    assert r.receipt.madelung == pytest.approx(xi, abs=3e-12)
    np.testing.assert_allclose(r.density @ s @ r.density, r.density, atol=4e-12, rtol=0)
    np.testing.assert_allclose(np.einsum('skij,kji->sk', r.density, s),
        np.repeat(np.array(populations)[:, None], len(k), axis=1), atol=3e-12)
    assert r.receipt.fock_calls == r.solver_snapshot.fock_calls + 1
    assert r.receipt.shell_calls <= r.plan.shell_calls
    assert r.receipt.primitive_pairs <= r.plan.primitive_pairs
    assert len(r.receipt.fitting_info.metrics) == 1 + len(k)*(len(k)-1)
    assert all(m.rank == c['auxiliary'].nbasis for m in r.receipt.fitting_info.metrics)
    assert r.receipt.confirmation_fock_error < 1e-10 and r.receipt.confirmation_energy_error < 1e-11


@pytest.mark.parametrize('correction', ['strict-zero-mode', 'BvK-ewald'])
def test_every_capped_epoch_preserves_evaluated_density_and_fixed_projectors(correction):
    c, k = case((2, 1, 1), (1, 0, 0), correction=correction)
    s, h, build, _ = reference(c, k)
    d = _occupy(np.stack([h, h]), s, (2, 1))
    c['options'].scf.maximum_iterations = 4
    result = core._manta.run_periodic_ri_uhf(**c)
    assert len(result.scf_history) == 4
    for iteration in range(1, 5):
        c['options'].scf.maximum_iterations = iteration
        r = core._manta.run_periodic_ri_uhf(**c)
        assert not r.converged and not r.receipt.confirmed
        assert r.receipt.fock_calls == iteration
        np.testing.assert_allclose(r.density, d, atol=8e-11, rtol=0)
        np.testing.assert_allclose(r.fock, build(d)[0], atol=8e-11, rtol=0)
        assert result.scf_history[iteration-1].energy_per_cell == r.solver_snapshot.energy_per_cell
        assert r.receipt.fitting_info.source_identity == result.receipt.fitting_info.source_identity
        assert [m.projector_identity for m in r.receipt.fitting_info.metrics] == [
            m.projector_identity for m in result.receipt.fitting_info.metrics]
        d = _occupy(build(d)[0], s, (2, 1))
    copied = result.scf_history; copied.clear()
    assert len(result.scf_history) == 4


@pytest.mark.parametrize('field,planned', [('memory_bytes', 'modeled_peak_bytes'), ('shell_calls', 'shell_calls'),
    ('primitive_quartets', 'primitive_pairs'), ('scalar_terms', 'scalar_terms'), ('kernel_terms', 'kernel_terms')])
@pytest.mark.parametrize('correction', ['strict-zero-mode', 'BvK-ewald'])
@pytest.mark.parametrize('mesh', [(1, 1, 1), (2, 1, 1)])
def test_joint_resource_admission_before_payload_copy(field, planned, correction, mesh):
    c, _ = case(mesh=mesh, species='H', correction=correction)
    p = plan(c)
    setattr(c['budget'], field, getattr(p, planned))
    assert core._manta.run_periodic_ri_uhf(**c).converged
    setattr(c['budget'], field, getattr(p, planned)-1)
    c['reciprocal_vectors'][:] = np.nan
    with pytest.raises((ValueError, RuntimeError), match='budget'):
        core._manta.run_periodic_ri_uhf(**c)


@pytest.mark.parametrize('correction', ['strict-zero-mode', 'BvK-ewald'])
def test_rank_truncation_and_empty_fit_are_explicit_sources(correction):
    c, k = case((2, 1, 1), (1, 0, 0), species='H', correction=correction)
    sources = []
    for cutoff in (1e-12, 10., 1e9):
        c['options'].ri.absolute_cutoff = cutoff
        r = core._manta.run_periodic_ri_uhf(**c)
        assert r.converged
        s, h, build, _ = reference(c, k)
        d = _scf_reference(s, h, build, (1, 0))
        np.testing.assert_allclose(r.density, d, atol=5e-9, rtol=0)
        np.testing.assert_allclose(r.fock, build(r.density)[0], atol=8e-11, rtol=0)
        ranks = [m.rank for m in r.receipt.fitting_info.metrics]
        assert all(rank == (4 if cutoff == 1e-12 else 1 if cutoff == 10. else 0) for rank in ranks)
        sources.append(r.receipt.source_identity)
    assert len(set(sources)) == 3


@pytest.mark.parametrize('field,value,match', [('dim', 2, 'neutral'), ('charge', 1, 'neutral'),
    ('multiplicity', 0, 'multiplicity'), ('multiplicity', 1, 'populations'), ('multiplicity', 6, 'populations')])
def test_unsupported_physical_profile_precedes_payload_scan(field, value, match):
    c, _ = case(); setattr(c['system'], field, value)
    c['reciprocal_vectors'][:] = np.nan
    with pytest.raises(ValueError, match=match):
        core._manta.run_periodic_ri_uhf(**c)


def test_pair_source_consistency_and_negative_metric_refusal():
    c, _ = case(species='H')
    c['options'].fitting.pair_cutoff_bohr = 2.
    with pytest.raises(ValueError, match='identical explicit'):
        core._manta.run_periodic_ri_uhf(**c)
    c['options'].fitting.pair_cutoff_bohr = 3.
    c['options'].fitting.auxiliary_cutoff_bohr = 5.
    base = np.array(list(itertools.product(range(-1, 2), repeat=3))) @ c['system'].reciprocal_lattice().T
    c['reciprocal_vectors'] = base[np.any(base != 0., axis=1)].copy()
    with pytest.raises(RuntimeError, match='significant negative eigenvalue'):
        core._manta.run_periodic_ri_uhf(**c)


def test_correction_spin_weights_and_owned_exports():
    c, _ = case((2, 1, 1), (1, 0, 0))
    strict = core._manta.run_periodic_ri_uhf(**c)
    c['options'].exchange_q0 = 'BvK-ewald'
    corrected = core._manta.run_periodic_ri_uhf(**c)
    assert strict.converged and corrected.converged
    xi = corrected.receipt.madelung
    np.testing.assert_allclose(corrected.density, strict.density, atol=5e-9, rtol=0)
    assert corrected.receipt.energy-strict.receipt.energy == pytest.approx(-.5*xi*strict.plan.electrons, abs=8e-11)
    for spin, occupied in enumerate((strict.plan.n_alpha, strict.plan.n_beta)):
        expected = strict.orbital_energies[spin].copy(); expected[:, :occupied] -= xi
        np.testing.assert_allclose(corrected.orbital_energies[spin], expected, atol=5e-9, rtol=0)
    assert strict.receipt.fitting_info.source_identity == corrected.receipt.fitting_info.source_identity
    assert strict.receipt.source_identity != corrected.receipt.source_identity
    for name in ('density', 'fock', 'coefficients', 'orbital_energies', 'overlap', 'hcore', 'momenta'):
        saved = getattr(corrected, name).copy(); getattr(corrected, name)[:] = np.nan
        np.testing.assert_array_equal(getattr(corrected, name), saved)
    with pytest.raises(AttributeError):
        corrected.receipt.confirmed = False


@pytest.mark.parametrize('correction', ['strict-zero-mode', 'BvK-ewald'])
def test_density_identity_is_independent_of_the_method_correction(correction):
    c, k = case((2, 1, 1), (1, 0, 0), correction=correction)
    c['options'].scf.maximum_iterations = 3
    r = core._manta.run_periodic_ri_uhf(**c)
    # Use actual native momenta. Recomputing the reciprocal lattice via a
    # different inverse changes its last bits and therefore its frame digest.
    builder = manta.PeriodicRIFockBuilder(c['basis'], c['auxiliary'], c['system'], r.momenta,
        np.ones(len(k))/len(k), c['reciprocal_vectors'], c['pair_vectors'], c['transfer_offsets'],
        c['options'].fitting, c['options'].ri, c['budget'])
    evaluated = builder.build(r.hcore, r.density)
    assert r.receipt.final_density_identity == evaluated.receipt.density_identity
    assert r.receipt.fitting_info.source_identity == builder.info.source_identity


@pytest.mark.parametrize('correction', ['strict-zero-mode', 'BvK-ewald'])
def test_independent_ao_and_auxiliary_images_transport_the_physical_epoch(correction):
    c, k = case((2, 1, 1), (1, 0, 0), correction=correction)
    atoms = [vq.Atom(1, p) for p in ([0., 0., 0.], [.9, .2, 0.], [-.2, 1.1, .1])]
    c['system'] = vq.PeriodicSystem(3, c['system'].lattice, atoms)
    c['system'].multiplicity = 2
    shells = c['basis'].shells()
    for i, shell in enumerate(shells):
        shell.atom_index = i; shell.origin = atoms[i].xyz
    c['basis'] = vq.BasisSet(c['system'].unit_cell_molecule(), shells, 'ri-uhf-gauge', True)
    aux = c['auxiliary'].shells()
    for shell in aux:
        shell.origin = atoms[0].xyz
    c['auxiliary'] = vq.BasisSet(c['system'].unit_cell_molecule(), aux, 'ri-uhf-aux', True)
    c['options'].scf.maximum_iterations = 3
    original = core._manta.run_periodic_ri_uhf(**c)
    shifts = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]) @ c['system'].lattice.T
    for shell, delta in zip(shells, shifts):
        shell.origin = np.array(shell.origin) + delta
    image_system = vq.PeriodicSystem(3, c['system'].lattice, [vq.Atom(1, s.origin) for s in shells])
    image_system.multiplicity = 2
    c['basis'] = vq.BasisSet(image_system.unit_cell_molecule(), shells, 'ri-uhf-images', True)
    for i, (shell, delta) in enumerate(zip(aux, shifts[::-1])):
        shell.atom_index = i; shell.origin = np.array(shell.origin) + delta
    aux_system = vq.PeriodicSystem(3, c['system'].lattice, [vq.Atom(1, s.origin) for s in aux])
    aux_system.multiplicity = 2
    c['auxiliary'] = vq.BasisSet(aux_system.unit_cell_molecule(), aux, 'ri-uhf-aux-images', True)
    labels = np.array(list(itertools.product(range(-1, 2), repeat=3)))
    c['translations'] = np.ascontiguousarray(labels @ c['system'].lattice.T)
    moved = core._manta.run_periodic_ri_uhf(**c)
    phase = np.exp(1j*k@shifts.T)
    transport = phase[:, :, None] * phase[:, None, :].conj()
    assert np.max(abs(moved.density.imag)) > .01
    assert np.max(abs(moved.density-original.density)) > .01
    for name in ('density', 'fock', 'overlap', 'hcore'):
        np.testing.assert_allclose(getattr(moved, name), getattr(original, name)*transport, atol=8e-11, rtol=0)
    assert moved.receipt.energy == pytest.approx(original.receipt.energy, abs=8e-11)
    assert moved.receipt.fitting_info.frame_identity != original.receipt.fitting_info.frame_identity
    assert [m.rank for m in moved.receipt.fitting_info.metrics] == [m.rank for m in original.receipt.fitting_info.metrics]


@pytest.mark.parametrize('correction', ['strict-zero-mode', 'BvK-ewald'])
def test_restricted_limit_against_independent_spin_scf(correction):
    c, k = case((2, 1, 1), (1, 0, 0), species='H', correction=correction)
    atom = vq.Atom(2, c['system'].unit_cell[0].xyz)
    c['system'] = vq.PeriodicSystem(3, c['system'].lattice, [atom])
    for key in ('basis', 'auxiliary'):
        c[key] = vq.BasisSet(c['system'].unit_cell_molecule(), c[key].shells(), 'ri-restricted-limit', True)
    r = core._manta.run_periodic_ri_uhf(**c)
    assert r.converged and (r.plan.n_alpha, r.plan.n_beta) == (1, 1)
    s, h, build, _ = reference(c, k)
    d = _scf_reference(s, h, build, (1, 1))
    np.testing.assert_allclose(r.density, d, atol=5e-9, rtol=0)
    np.testing.assert_allclose(r.fock, build(r.density)[0], atol=8e-11, rtol=0)
    np.testing.assert_allclose(r.density[0], r.density[1], atol=0, rtol=0)
    np.testing.assert_allclose(r.fock[0], r.fock[1], atol=0, rtol=0)


def test_selected_output_citations_and_provenance_admission(monkeypatch):
    import json
    from vibeqc import output
    from vibeqc.output.citations import load_default_database
    c, _ = case((2, 1, 1), (1, 0, 0), species='H', correction='BvK-ewald')
    raw_plan = plan(c)
    budget = c.pop('budget'); engine = manta.IntegralEngineOptions(backend='manta', budget=budget)
    records = []; monkeypatch.setattr(output, 'record', lambda *a, **kw: records.append(kw))
    r = manta.run_periodic_ri_uhf(**c, integral_engine=engine)
    e = records[-1]['integral_provenance']
    json.dumps(e, allow_nan=False)
    assert e['fitting']['frame_identity'] == r.receipt.fitting_info.frame_identity
    assert e['fitting']['metrics'][0]['projector_identity'] == r.receipt.fitting_info.metrics[0].projector_identity
    assert e['spin_electrons_per_cell'] == [1, 0] and e['spin_band_gaps'][1] is None
    assert e['accuracy_qualification'] == 'finite-fitted-source-only' and not e['symmetry_reduced']
    assert e['operator_generation'] == r.solver_snapshot.evaluated_iteration
    assert e['citation_methods'] == ['manta_periodic_ri_uhf', 'manta_periodic_madelung']
    db = load_default_database()
    keys = {c.key for method in e['citation_methods']
            for c in db.assemble(method=method, numerics=e['citation_numerics']).citations}
    assert {'sun_berkelbach_gdf_2017', 'ye_berkelbach_rsgdf_2021', 'neese_shark_2023', 'ewald_lattice_sum_1921'} <= keys
    assert budget.other_live_bytes == 0
    assert r.plan.modeled_peak_bytes == raw_plan.modeled_peak_bytes + e['python_reserved_bytes']
    budget.memory_bytes = r.plan.modeled_peak_bytes
    assert manta.run_periodic_ri_uhf(**c, integral_engine=engine).converged
    budget.memory_bytes -= 1; c['reciprocal_vectors'][:] = np.nan
    with pytest.raises(ValueError, match='memory budget'):
        manta.run_periodic_ri_uhf(**c, integral_engine=engine)
