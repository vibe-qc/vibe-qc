"""Physical finite-source UHF: independently integrated spin SCF and correction."""
import numpy as np
import pytest
from scipy.linalg import eigh

import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core
from vibeqc.bipole_fock_ewald import probe_charge_madelung_supercell
from tests.test_manta_periodic_rhf import _case as rhf_case, _components, _nuclear_reference
from tests.test_manta_periodic_exchange import _quadrature_bloch


def _case(mesh=(1, 1, 1), shift=(0, 0, 0), species='H', correction='strict-zero-mode'):
    c, k = rhf_case(mesh, shift)
    charge, exponents = (1, [.8, .25]) if species == 'H' else (3, [2.4, .7, .2])
    system = vq.PeriodicSystem(3, c['system'].lattice, [vq.Atom(charge, [.1, -.15, .2])])
    system.multiplicity = 2
    shells = []
    for exponent in exponents:
        shell = core.ShellInfo(); shell.atom_index = 0; shell.origin = system.unit_cell[0].xyz
        shell.l = 0; shell.pure = True; shell.exponents = [exponent]; shell.coefficients = [1.]
        shells.append(shell)
    o = manta.PeriodicUHFOptions()
    o.alpha = .6; o.nuclear_cutoff_bohr = 4.; o.pair_cutoff_bohr = 3.; o.reciprocal_batch = 16
    o.scf.maximum_iterations = 100; o.exchange_q0 = correction
    c.update(system=system, basis=vq.BasisSet(system.unit_cell_molecule(), shells, 'manta-uhf-test', False), options=o)
    return c, k


def _reference(case, k):
    # Shared libint one-electron and Gaussian-quadrature witnesses; no
    # production MANTA density contraction is used to construct this oracle.
    s, h, _ = _components(case, k)
    b, o = case['basis'], case['options']
    g, q, cells = case['reciprocal_vectors'], case['exchange_vectors'], case['translations']
    weights = np.ones(len(k)) / len(k)
    factor = 4*np.pi / abs(np.linalg.det(case['system'].lattice))
    fj = np.stack([_quadrature_bloch(b, g, cells, -point, None) for point in k])
    transfers = []
    for target, kt in enumerate(k):
        for source in range(len(k)):
            pair = target*len(k) + source
            lo, hi = case['transfer_offsets'][pair:pair+2]
            vectors = q[lo:hi]
            transfers.append((_quadrature_bloch(b, vectors, cells, kt, None),
                              weights[source]*factor / np.sum(vectors*vectors, axis=1)))
    xi = 0. if o.exchange_q0 == 'strict-zero-mode' else probe_charge_madelung_supercell(
        case['system'], tuple(case['mesh'].mesh), alpha=o.madelung_alpha, precision=1e-14)

    def build(d):
        rho = np.einsum('k,skij,kgij->g', weights, d, fj)
        j = np.einsum('g,g,kgij->kij', factor / np.sum(g*g, axis=1), rho, fj.conj())
        exchange = np.zeros_like(d)
        for spin in range(2):
            for target in range(len(k)):
                for source in range(len(k)):
                    panel, kernel = transfers[target*len(k)+source]
                    exchange[spin, target] += np.einsum('g,gai,ab,gbj->ij', kernel, panel.conj(), d[spin, source], panel)
        correction = -xi * (s @ d @ s)
        f = h + j - exchange + correction
        e1 = np.einsum('k,skij,kji->', weights, d, h).real
        ej = .5*np.einsum('k,skij,kji->', weights, d, j).real
        ek = -.5*np.einsum('k,skij,skji->', weights, d, exchange).real
        ec = .5*np.einsum('k,skij,skji->', weights, d, correction).real
        return f, (e1, ej, ek, ec)
    return s, h, build, xi


def _occupy(f, s, populations):
    result = []
    for spin, occupied in enumerate(populations):
        coeff = np.array([eigh(fp, sp)[1][:, :occupied] for fp, sp in zip(f[spin], s)])
        result.append(coeff @ coeff.conj().swapaxes(-1, -2))
    return np.array(result)


def _scf_reference(s, h, build, populations):
    d = _occupy(np.stack([h, h]), s, populations)
    for _ in range(250):
        f, _ = build(d)
        updated = _occupy(f, s, populations)
        if np.max(abs(updated - d)) < 1e-12:
            return updated
        d = updated
    raise AssertionError('Independent finite-source UHF did not converge')


@pytest.mark.parametrize('species', ['H', 'Li'])
@pytest.mark.parametrize('mesh,shift', [((1, 1, 1), (0, 0, 0)), ((2, 1, 1), (0, 0, 0)), ((2, 1, 1), (1, 0, 0))])
@pytest.mark.parametrize('correction', ['strict-zero-mode', 'BvK-ewald'])
def test_physical_uhf_matches_independent_scf_and_energy_components(species, mesh, shift, correction):
    c, k = _case(mesh, shift, species, correction)
    r = core._manta.run_periodic_uhf(**c)
    assert r.converged and r.receipt.confirmed
    populations = (r.receipt.n_alpha, r.receipt.n_beta)
    assert populations == ((1, 0) if species == 'H' else (2, 1))
    assert r.plan.momenta == len(k) and r.density.shape == (2, len(k), c['basis'].nbasis, c['basis'].nbasis)
    s, h, build, xi = _reference(c, k)
    d = _scf_reference(s, h, build, populations)
    np.testing.assert_allclose(r.overlap, s, atol=3e-12, rtol=0)
    np.testing.assert_allclose(r.hcore, h, atol=3e-12, rtol=0)
    np.testing.assert_allclose(r.density, d, atol=4e-9, rtol=0)
    f, energies = build(r.density)
    np.testing.assert_allclose(r.fock, f, atol=4e-12, rtol=0)
    np.testing.assert_allclose([r.receipt.one_electron_energy, r.receipt.coulomb_energy,
        r.receipt.uncorrected_exchange_energy, r.receipt.exchange_correction_energy], energies, atol=4e-12, rtol=0)
    nuclear, count = _nuclear_reference(c)
    assert r.receipt.nuclear_pair_images == count
    assert r.receipt.nuclear_energy == pytest.approx(sum(nuclear), abs=3e-12)
    assert r.receipt.energy == pytest.approx(sum(nuclear) + sum(build(d)[1]), abs=4e-11)
    assert r.receipt.madelung == pytest.approx(xi, abs=3e-12)
    np.testing.assert_allclose(r.density @ s @ r.density, r.density, atol=4e-12, rtol=0)
    np.testing.assert_allclose(np.einsum('skij,kji->sk', r.density, s), np.repeat(np.array(populations)[:, None], len(k), axis=1), atol=3e-12)
    assert r.receipt.fock_calls == r.solver_snapshot.fock_calls + 1
    assert r.receipt.shell_calls <= r.plan.shell_calls and r.receipt.primitive_pairs <= r.plan.primitive_pairs
    assert r.receipt.confirmation_fock_error < 1e-10 and r.receipt.confirmation_energy_error < 1e-11


@pytest.mark.parametrize('species', ['H', 'Li'])
def test_correction_has_separate_spin_factors_and_no_extra_k_weight(species):
    c, k = _case((2, 1, 1), (1, 0, 0), species)
    strict = core._manta.run_periodic_uhf(**c)
    c['options'].exchange_q0 = 'BvK-ewald'
    corrected = core._manta.run_periodic_uhf(**c)
    assert strict.converged and corrected.converged
    xi = corrected.receipt.madelung
    np.testing.assert_allclose(corrected.density, strict.density, atol=4e-9, rtol=0)
    assert corrected.receipt.energy - strict.receipt.energy == pytest.approx(-.5*xi*strict.plan.electrons, abs=4e-11)
    for spin, occupied in enumerate((strict.plan.n_alpha, strict.plan.n_beta)):
        expected = strict.orbital_energies[spin].copy(); expected[:, :occupied] -= xi
        np.testing.assert_allclose(corrected.orbital_energies[spin], expected, atol=3e-9)
    assert corrected.receipt.source_identity != strict.receipt.source_identity
    assert corrected.receipt.final_density_identity != strict.receipt.final_density_identity
    assert corrected.receipt.exchange_correction_identity


@pytest.mark.parametrize('correction', ['strict-zero-mode', 'BvK-ewald'])
def test_every_capped_epoch_matches_independent_spin_recurrence(correction):
    c, k = _case((2, 1, 1), (1, 0, 0), 'Li', correction)
    s, h, build, _ = _reference(c, k)
    d = _occupy(np.stack([h, h]), s, (2, 1))
    c['options'].scf.maximum_iterations = 4
    result = core._manta.run_periodic_uhf(**c)
    assert len(result.scf_history) == 4
    for iteration in range(1, 5):
        c['options'].scf.maximum_iterations = iteration
        r = core._manta.run_periodic_uhf(**c)
        assert not r.converged and not r.receipt.confirmed
        assert r.receipt.fock_calls == iteration
        np.testing.assert_allclose(r.density, d, atol=4e-11, rtol=0)
        np.testing.assert_allclose(r.fock, build(d)[0], atol=4e-11, rtol=0)
        assert result.scf_history[iteration-1].energy_per_cell == r.solver_snapshot.energy_per_cell
        d = _occupy(build(d)[0], s, (2, 1))
    copied = result.scf_history; copied.clear()
    assert len(result.scf_history) == 4


@pytest.mark.parametrize('field,planned', [('memory_bytes', 'modeled_peak_bytes'), ('shell_calls', 'shell_calls'),
    ('primitive_quartets', 'primitive_pairs'), ('scalar_terms', 'scalar_terms'), ('kernel_terms', 'kernel_terms')])
@pytest.mark.parametrize('correction', ['strict-zero-mode', 'BvK-ewald'])
def test_aggregate_resource_admission_precedes_vector_copy(field, planned, correction):
    c, _ = _case(species='Li', correction=correction)
    p = manta.plan_periodic_uhf(c['basis'], c['system'], c['mesh'], len(c['reciprocal_vectors']),
        int(max(np.diff(c['transfer_offsets']))), len(c['translations']), c['options'], c['budget'])
    setattr(c['budget'], field, getattr(p, planned))
    assert core._manta.run_periodic_uhf(**c).converged
    setattr(c['budget'], field, getattr(p, planned)-1)
    c['exchange_vectors'][:] = np.nan
    with pytest.raises((ValueError, RuntimeError), match='budget'):
        core._manta.run_periodic_uhf(**c)


@pytest.mark.parametrize('field,value,match', [('dim', 2, 'neutral'), ('charge', 1, 'neutral'),
    ('multiplicity', 0, 'multiplicity'), ('multiplicity', 1, 'populations'), ('multiplicity', 6, 'populations')])
def test_physical_occupation_profile_refuses_before_payloads(field, value, match):
    c, _ = _case(species='Li'); setattr(c['system'], field, value)
    c['exchange_vectors'][:] = np.nan
    with pytest.raises(ValueError, match=match): core._manta.run_periodic_uhf(**c)


@pytest.mark.parametrize('correction', ['strict-zero-mode', 'BvK-ewald'])
def test_restricted_limit_and_shared_physical_sources(correction):
    c, _ = rhf_case((2, 1, 1), (1, 0, 0)); c['options'].exchange_q0 = correction
    rhf = core._manta.run_periodic_rhf(**c)
    o = manta.PeriodicUHFOptions()
    for field in ('alpha', 'nuclear_cutoff_bohr', 'pair_cutoff_bohr', 'reciprocal_batch', 'exchange_q0'):
        setattr(o, field, getattr(c['options'], field))
    c['options'] = o
    uhf = core._manta.run_periodic_uhf(**c)
    assert rhf.converged and uhf.converged
    np.testing.assert_allclose(uhf.density.sum(axis=0), rhf.density, atol=3e-10)
    np.testing.assert_allclose(uhf.fock, np.stack([rhf.fock, rhf.fock]), atol=3e-11)
    assert uhf.receipt.energy == pytest.approx(rhf.receipt.energy, abs=3e-12)
    for field in ('overlap_identity', 'kinetic_identity', 'nuclear_identity', 'nuclear_energy_identity', 'exchange_correction_identity'):
        assert getattr(uhf.receipt, field) == getattr(rhf.receipt, field)
    assert uhf.receipt.source_identity != rhf.receipt.source_identity


def test_selected_output_provenance_and_native_owned_exports(monkeypatch):
    from vibeqc import output
    c, _ = _case(correction='BvK-ewald'); budget = c.pop('budget')
    engine = manta.IntegralEngineOptions(backend='manta', budget=budget)
    records = []; monkeypatch.setattr(output, 'record', lambda *a, **kw: records.append(kw))
    r = manta.run_periodic_uhf(**c, integral_engine=engine)
    assert r.converged
    evidence = records[-1]['integral_provenance']
    assert evidence['spin_electrons_per_cell'] == [1, 0]
    assert evidence['spin_band_gaps'][1] is None
    assert evidence['citation_methods'] == ['manta_periodic_uhf', 'manta_periodic_madelung']
    assert budget.other_live_bytes == 0
    for field in ('density', 'fock', 'coefficients', 'orbital_energies', 'overlap', 'hcore'):
        saved = getattr(r, field).copy(); getattr(r, field)[:] = 0
        np.testing.assert_array_equal(getattr(r, field), saved)
    with pytest.raises(AttributeError): r.receipt.confirmed = False
    c['options'].scf.maximum_iterations = 1
    stopped = manta.run_periodic_uhf(**c, integral_engine=engine)
    assert not stopped.converged and records[-1]['integral_provenance']['status'] == 'ITERATION_LIMIT'


def test_history_storage_is_reserved_before_payload_scan():
    c, _ = _case(species='Li')
    r = core._manta.run_periodic_uhf(**c)
    assert len(r.scf_history) == r.solver_snapshot.evaluated_iteration == r.receipt.fock_calls-1
    assert r.plan.history_capacity == 100
    c['options'].scf.maximum_iterations = 200
    p = manta.plan_periodic_uhf(c['basis'], c['system'], c['mesh'], len(c['reciprocal_vectors']),
        int(max(np.diff(c['transfer_offsets']))), 1, c['options'], c['budget'])
    assert p.history_live_bytes == 2*r.plan.history_live_bytes
    assert p.modeled_peak_bytes-r.plan.modeled_peak_bytes == r.plan.history_live_bytes
    c['budget'].memory_bytes = p.modeled_peak_bytes-1; c['reciprocal_vectors'][:] = np.nan
    with pytest.raises(ValueError, match='memory budget'): core._manta.run_periodic_uhf(**c)


@pytest.mark.parametrize('correction', ['strict-zero-mode', 'BvK-ewald'])
def test_independent_shell_images_transport_both_spin_states(correction):
    import itertools
    c, k = _case((2, 1, 1), (1, 0, 0), 'Li', correction)
    atoms = [vq.Atom(1, p) for p in ([0., 0., 0.], [.9, .2, 0.], [-.2, 1.1, .1])]
    c['system'] = vq.PeriodicSystem(3, c['system'].lattice, atoms)
    c['system'].multiplicity = 2
    shells = c['basis'].shells()
    for index, shell in enumerate(shells):
        shell.atom_index = index; shell.origin = atoms[index].xyz
    c['basis'] = vq.BasisSet(c['system'].unit_cell_molecule(), shells, 'manta-uhf-gauge', False)
    c['options'].scf.maximum_iterations = 3
    original = core._manta.run_periodic_uhf(**c)
    shifts = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]) @ c['system'].lattice.T
    for shell, delta in zip(shells, shifts): shell.origin = np.array(shell.origin)+delta
    basis_cell = vq.PeriodicSystem(3, c['system'].lattice, [vq.Atom(1, shell.origin) for shell in shells])
    basis_cell.multiplicity = 2
    c['basis'] = vq.BasisSet(basis_cell.unit_cell_molecule(), shells, 'manta-uhf-image', False)
    labels = np.array(list(itertools.product(range(-1, 2), repeat=3)))
    c['translations'] = np.ascontiguousarray(labels @ c['system'].lattice.T)
    moved = core._manta.run_periodic_uhf(**c)
    phase = np.exp(1j*k@shifts.T)
    transport = phase[:, :, None]*phase[:, None, :].conj()
    assert moved.solver_snapshot.evaluated_iteration == original.solver_snapshot.evaluated_iteration
    assert np.max(abs(moved.density-original.density)) > .01
    for field in ('density', 'fock', 'overlap', 'hcore'):
        np.testing.assert_allclose(getattr(moved, field), getattr(original, field)*transport, atol=4e-10, rtol=0)
    assert moved.receipt.energy == pytest.approx(original.receipt.energy, abs=4e-11)


def test_gap_and_failed_terminal_confirmation_remain_distinct_verdicts():
    c, _ = _case(species='Li'); c['options'].scf.minimum_band_gap_hartree = 100.
    noninsulating = core._manta.run_periodic_uhf(**c)
    assert not noninsulating.converged and not noninsulating.receipt.confirmed
    assert noninsulating.solver_snapshot.status.name == 'NON_INSULATING'
    c, _ = _case(species='Li'); c['options'].confirmation_fock_tolerance = 1e-30
    unconfirmed = core._manta.run_periodic_uhf(**c)
    assert unconfirmed.solver_snapshot.converged
    assert unconfirmed.receipt.confirmation_fock_error > 1e-30
    assert not unconfirmed.converged and not unconfirmed.receipt.confirmed


@pytest.mark.parametrize('field,value,match', [('exchange_q0', 'unknown', 'exchange_q0'),
    ('madelung_alpha', 0., 'Madelung'), ('madelung_tolerance', 1., 'Madelung'),
    ('confirmation_energy_tolerance', float('nan'), 'confirmation')])
def test_invalid_physical_controls_refuse(field, value, match):
    c, _ = _case(); setattr(c['options'], field, value)
    c['exchange_vectors'][:] = np.nan
    with pytest.raises(ValueError, match=match): core._manta.run_periodic_uhf(**c)


@pytest.mark.parametrize('mesh,shift', [((1, 1, 1), (0, 0, 0)), ((2, 1, 1), (1, 0, 0))])
def test_refined_periodic_uhf_against_optional_pyscf_aft(mesh, shift):
    _check_refined_pyscf(mesh, shift)


def _check_refined_pyscf(mesh, shift, *, from_cutoffs=False):
    import itertools
    pytest.importorskip('pyscf.pbc')
    from pyscf.pbc import gto, df, scf, tools
    c, k = _case(mesh, shift, 'H', 'BvK-ewald')
    o = c['options']; o.pair_cutoff_bohr = 18.; o.nuclear_cutoff_bohr = 16.; o.reciprocal_batch = 128
    labels = np.array(list(itertools.product(range(-12, 13), repeat=3)), dtype=float)
    base = labels @ (2*np.pi*np.linalg.inv(c['system'].lattice))
    c['reciprocal_vectors'] = np.ascontiguousarray(base[np.any(labels, axis=1)])
    chunks = [base+t-s for t in k for s in k]; chunks = [q[np.any(q != 0., axis=1)] for q in chunks]
    c['exchange_vectors'] = np.ascontiguousarray(np.concatenate(chunks))
    c['transfer_offsets'] = np.r_[0, np.cumsum([len(q) for q in chunks])].astype(np.int64)
    images = np.array(list(itertools.product(range(-3, 4), repeat=3)), dtype=float)
    c['translations'] = np.ascontiguousarray(images @ c['system'].lattice.T)
    c['budget'].memory_bytes = 2**30; c['budget'].shell_calls = 10**9
    c['budget'].scalar_terms = 10**18; c['budget'].kernel_terms = 10**18
    if from_cutoffs:
        arguments = {name: c[name] for name in ('basis', 'system', 'mesh', 'options', 'budget')}
        result = core._manta.run_periodic_uhf_from_cutoffs(**arguments, reciprocal_energy_cutoff=60.)
    else:
        result = core._manta.run_periodic_uhf(**c)
    cell = gto.Cell(); cell.atom = [('H', c['system'].unit_cell[0].xyz)]
    cell.a = c['system'].lattice.T; cell.unit = 'Bohr'
    cell.basis = {'H': [[0, [.8, 1.]], [0, [.25, 1.]]]}
    cell.spin = 1; cell.precision = 1e-12; cell.verbose = 0; cell.build()
    mf = scf.KUHF(cell, k, exxdiv='ewald'); mf.with_df = df.AFTDF(cell, k); mf.with_df.mesh = [25]*3
    # This reference fixes one alpha and zero beta electrons at every k.
    # PySCF KUHF's empty-beta diagnostic currently reads an undefined
    # mo_energy_b; supply the admitted occupations without changing its Fock.
    def fixed_spin_occ(energies, coefficients=None):
        occupations = np.zeros_like(energies)
        for point, levels in enumerate(energies[0]):
            occupations[0, point, np.argmin(levels)] = 1.
        return occupations
    mf.nelec = (len(k), 0)  # PySCF counts populations over the sampled supercell.
    mf.get_occ = fixed_spin_occ
    mf.conv_tol = 1e-12; mf.max_cycle = 100; energy = mf.kernel()
    assert mf.converged and result.converged
    assert result.receipt.energy == pytest.approx(energy, abs=2e-9)
    assert result.receipt.nuclear_energy == pytest.approx(cell.energy_nuc(), abs=3e-11)
    assert result.receipt.madelung == pytest.approx(tools.pbc.madelung(cell, k), abs=3e-12)
    np.testing.assert_allclose(result.overlap, mf.get_ovlp(), atol=3e-10, rtol=0)
    np.testing.assert_allclose(result.hcore, mf.get_hcore(), atol=3e-10, rtol=0)
    np.testing.assert_allclose(result.density, mf.make_rdm1(), atol=3e-8, rtol=0)
    expected_fock = mf.get_hcore()[None] + mf.get_veff(dm_kpts=result.density)
    np.testing.assert_allclose(result.fock, expected_fock, atol=3e-9, rtol=0)


def test_physical_uhf_truncated_overlap_exports_rectangular_spin_orbitals():
    c, _ = _case(species='Li', correction='BvK-ewald')
    original = core._manta.run_periodic_uhf(**c)
    shells = c['basis'].shells(); shells.append(shells[0])
    c['basis'] = vq.BasisSet(c['system'].unit_cell_molecule(), shells, 'manta-uhf-rank', False)
    redundant = core._manta.run_periodic_uhf(**c)
    assert redundant.converged and original.converged
    assert redundant.solver_memory.retained_rank == 3
    assert redundant.coefficients.shape == (2, 1, 4, 3)
    assert redundant.receipt.energy == pytest.approx(original.receipt.energy, abs=4e-11)
    assert redundant.solver_snapshot.maximum_electron_count_error < 1e-10
    np.testing.assert_allclose(redundant.density @ redundant.overlap @ redundant.density,
                               redundant.density, atol=3e-11, rtol=0)
