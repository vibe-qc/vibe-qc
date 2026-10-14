"""Finite-Hamiltonian restart proofs; these are not native QC acceptance."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from vibeqc.periodic.chi.canonical_cc import ChiCCOptions, solve, evaluate_same_source_cc
from vibeqc.periodic.chi._cc_equations import triples
from tests.test_chi_cc_equations import model, determinant_reference


class Interrupted(RuntimeError):
    pass


class Dense:
    def __init__(self, g, no):
        self.g, self.no, self.shape = g, no, g.shape

    def __getitem__(self, key):
        return self.g[key]


def test_cc_iteration_limit_resume_preserves_diis_and_physical_solution():
    f, g, _, _ = model(n=6, no=2)
    integrals = Dense(g, 2)
    options = ChiCCOptions(energy_tolerance=1e-12, residual_tolerance=1e-10)
    fresh = solve(integrals, np.diag(f), options)
    states = []
    with pytest.raises(RuntimeError, match='did not converge'):
        solve(integrals, np.diag(f), replace(options, max_iter=3), progress_callback=states.append)
    assert states[-1]['iteration'] == 3
    assert states[-1]['diis_vectors'].shape[0] == 3
    assert states[-1]['trace'][0, 3] == -1.
    resumed = solve(integrals, np.diag(f), options, restart_state=states[-1])
    np.testing.assert_allclose(resumed[0], fresh[0], atol=1e-13)
    np.testing.assert_allclose(resumed[1], fresh[1], atol=1e-13)
    np.testing.assert_allclose(resumed[2], fresh[2], atol=1e-13)
    np.testing.assert_allclose(resumed[3], fresh[3], atol=1e-13)
    expected, r1, r2, _ = determinant_reference(f, g, resumed[1], resumed[2])
    assert resumed[0] == pytest.approx(expected, abs=1e-12)
    assert max(abs(r1).max(), abs(r2).max()) < 1e-10


def test_cc_forced_interruption_and_completed_revalidation():
    f, g, _, _ = model(n=6, no=2)
    integrals = Dense(g, 2)
    options = ChiCCOptions(energy_tolerance=1e-12, residual_tolerance=1e-10)
    states = []

    def interrupt(state):
        states.append(state)
        if state['iteration'] == 2:
            raise Interrupted('after complete iteration')

    with pytest.raises(Interrupted):
        solve(integrals, np.diag(f), options, progress_callback=interrupt)
    resumed = solve(integrals, np.diag(f), options, restart_state=states[-1],
                    progress_callback=states.append)
    completed = states[-1]
    assert completed['converged']
    restored = solve(integrals, np.diag(f), options, restart_state=completed)
    np.testing.assert_allclose(restored[2], resumed[2], atol=1e-13)
    broken = deepcopy(completed)
    broken['t1'][0, 0] += .01
    with pytest.raises(ValueError, match='convergence revalidation'):
        solve(integrals, np.diag(f), options, restart_state=broken)
    corrupted_diis = deepcopy(completed)
    corrupted_diis['diis_errors'][0, -1] += .01
    with pytest.raises(ValueError, match='DIIS history violates excitation spaces'):
        solve(integrals, np.diag(f), options, restart_state=corrupted_diis)
    for key in ('diis_vectors', 'trace', 't2'):
        broken = deepcopy(completed)
        broken[key] = broken[key][:-1]
        with pytest.raises(ValueError, match='checkpoint'):
            solve(integrals, np.diag(f), options, restart_state=broken)


def determinant_triples(f, g, t1, t2):
    _, _, _, (v, s, d, basis, ref) = determinant_reference(f, g, t1, t2)
    right, single = (v @ d)[:, ref], (v @ s)[:, ref]
    no = t1.shape[1]
    expected = 0j
    for idx, bits in enumerate(basis):
        if (bits & ((1 << no)-1)).bit_count() == no-3:
            den = sum(f[i, i] for i in range(no) if not bits & (1 << i))
            den -= sum(f[a, a] for a in range(no, len(f)) if bits & (1 << a))
            expected += right[idx].conjugate()*(right[idx]+single[idx])/den
    return expected


def test_complex_triples_interruption_exact_prefix_and_independent_sum():
    f, g, t1, t2 = (x.real for x in model())
    expected = determinant_triples(f, g, t1, t2)
    phase = np.exp(1j*np.arange(8)*.71)
    g = np.einsum('p,q,r,s,pqrs->pqrs', phase.conj(), phase.conj(), phase, phase, g)
    t1 = np.einsum('a,i,ai->ai', phase[4:].conj(), phase[:4], t1)
    t2 = np.einsum('a,b,i,j,abij->abij', phase[4:].conj(), phase[4:].conj(), phase[:4], phase[:4], t2)
    states = []

    def interrupt(state):
        states.append(state)
        if state['completed_work'] == 5:
            raise Interrupted('after complete triples unit')

    with pytest.raises(Interrupted):
        triples(t1, t2, np.diag(f), g, progress_callback=interrupt)
    saved = states[-1]
    assert saved['total_work'] == 16
    resumed = triples(t1, t2, np.diag(f), g, restart_state=saved, progress_callback=states.append)
    assert resumed == pytest.approx(expected, abs=2e-13)
    assert resumed == pytest.approx(triples(t1, t2, np.diag(f), g), abs=2e-13)
    assert [state['completed_work'] for state in states] == list(range(1, 17))
    for change in ({'completed_work': 6}, {'coverage_sha256': 'wrong'}, {'evaluated_work': 0},
                   {'total_work': 17}):
        with pytest.raises(ValueError, match='coverage'):
            triples(t1, t2, np.diag(f), g, restart_state=dict(saved, **change))


def test_complex_momentum_triples_preserves_skipped_coverage(monkeypatch):
    from tests.test_chi_canonical_mp2 import fixture
    from vibeqc.periodic.chi.canonical_mp2 import CanonicalFactorAdapter
    from vibeqc.periodic.chi.canonical_cc import FactorIntegrals
    ref, channels = fixture(monkeypatch, nk=3, source='classic', nao=2)
    c, e, no = channels[0]
    integrals = FactorIntegrals(CanonicalFactorAdapter(ref.hf_result.fitted_operator), c, no, 0, 1)
    eps = np.array([e[k, p] for k, p, spin in integrals.labels])
    n = integrals.no
    dense = integrals[(slice(None),)*4]
    den = eps[:n][None, None, :, None]+eps[:n][None, None, None, :]-eps[n:][:, None, None, None]-eps[n:][None, :, None, None]
    t2 = dense[n:, n:, :n, :n]/den
    t1 = np.zeros((len(eps)-n, n), complex)
    # A dense tensor sum includes zero forbidden excitations independently of
    # the production momentum predicate and traversal's coverage counters.
    expected = triples(t1, t2, eps, dense)
    states = []

    def interrupt(state):
        states.append(state)
        if state['completed_work'] == 73:
            raise Interrupted()

    with pytest.raises(Interrupted):
        triples(t1, t2, eps, integrals, progress_callback=interrupt)
    assert states[-1]['evaluated_work'] < 73
    resumed = triples(t1, t2, eps, integrals, restart_state=states[-1], progress_callback=states.append)
    assert resumed == pytest.approx(expected, abs=2e-13)
    assert states[-1]['completed_work'] == states[-1]['total_work'] == 400
    assert len(states) == 400


@pytest.mark.parametrize('unrestricted', [False, True])
def test_persistent_cc_iteration_limit_and_completed_result(monkeypatch, tmp_path, unrestricted):
    from tests.test_chi_canonical_mp2 import fixture
    from vibeqc.output.checkpoint import read_restart_state
    ref, _ = fixture(monkeypatch, nk=1, unrestricted=unrestricted, source='classic', nao=4)
    path = tmp_path/'cc.restart'
    opts = ChiCCOptions(compute_triples=False, energy_tolerance=1e-12, residual_tolerance=1e-10)
    frozen = (1, 0) if unrestricted else 0
    fresh = evaluate_same_source_cc(ref, frozen_core=frozen, options=opts)
    with pytest.raises(RuntimeError, match='did not converge'):
        evaluate_same_source_cc(ref, frozen_core=frozen, options=replace(opts, max_iter=2),
                                checkpoint_path=path, checkpoint_interval=7)
    metadata, arrays = read_restart_state(path)
    assert metadata['stage'] == 'cc_iteration'
    assert np.isfinite(arrays['trace']).all()
    resumed = evaluate_same_source_cc(ref, frozen_core=frozen, options=opts, resume_from=path)
    assert resumed.restart_mode == 'resumed-correlation'
    assert resumed.resumed_cc_iterations == 2
    assert resumed.diis_restart_policy == 'reused-compatible-history'
    assert resumed.iterations == fresh.iterations
    np.testing.assert_allclose(resumed.t1, fresh.t1, atol=1e-13)
    np.testing.assert_allclose(resumed.t2, fresh.t2, atol=1e-13)
    assert resumed.e_ccsd_per_cell == pytest.approx(fresh.e_ccsd_per_cell, abs=1e-13)
    metadata, _ = read_restart_state(path)
    assert metadata['stage'] == 'completed' and metadata['kind'] == 'completed_result'
    restored = evaluate_same_source_cc(ref, frozen_core=frozen, options=opts, resume_from=path)
    assert restored.restart_mode == 'restored-completed-result'
    assert restored.residual_norm == pytest.approx(fresh.residual_norm, abs=1e-14)
    assert not restored.t1.flags.writeable and not restored.t2.flags.writeable


def real_hamiltonian_uhf(monkeypatch):
    """Independent spin frames with arbitrary phases of a real Hamiltonian."""
    from tests.test_chi_canonical_mp2 import fixture, reseal
    ref, channels = fixture(monkeypatch, nk=1, unrestricted=True, source='classic', nao=4)
    hf = ref.hf_result
    for suffix, (c, e, n) in zip(('_alpha', '_beta'), channels):
        frame = np.array([np.linalg.qr(c[0].real)[0]])
        frame = frame*np.exp(1j*np.arange(4)*.47)[None, None, :]
        setattr(hf, 'mo_coeffs'+suffix, frame)
        setattr(hf, 'density'+suffix, frame[:, :, :n]@frame[:, :, :n].conj().transpose(0, 2, 1))
        setattr(hf, 'fock'+suffix, (frame*e[:, None, :])@frame.conj().transpose(0, 2, 1))
    return reseal(ref)


def test_persistent_streamed_triples_retains_cc_and_exact_completion(monkeypatch, tmp_path):
    from vibeqc.output import checkpoint
    ref = real_hamiltonian_uhf(monkeypatch)
    path = tmp_path/'triples.restart'
    fresh = evaluate_same_source_cc(ref)
    assert abs(fresh.e_triples_per_cell) > 1e-10
    assert fresh.triples_work_completed == fresh.triples_work_total == 10
    original = checkpoint.write_restart_state

    def interrupt(path, metadata, arrays):
        result = original(path, metadata, arrays)
        if metadata['stage'] == 'triples' and metadata['triples']['completed_work'] == 3:
            raise Interrupted('published complete prefix')
        return result

    with monkeypatch.context() as patch:
        patch.setattr(checkpoint, 'write_restart_state', interrupt)
        with pytest.raises(Interrupted):
            evaluate_same_source_cc(ref, checkpoint_path=path)
    metadata, arrays = checkpoint.read_restart_state(path)
    assert metadata['kind'] == 'correlation_state' and metadata['stage'] == 'triples'
    np.testing.assert_allclose(arrays['t1'], fresh.t1, atol=1e-13)
    np.testing.assert_allclose(arrays['t2'], fresh.t2, atol=1e-13)
    resumed = evaluate_same_source_cc(ref, resume_from=path)
    assert resumed.resumed_triples_work == 3
    assert resumed.triples_work_completed == resumed.triples_work_total == 10
    assert resumed.e_triples_per_cell == pytest.approx(fresh.e_triples_per_cell, abs=1e-13)
    assert resumed.convergence_history == fresh.convergence_history
    metadata, _ = checkpoint.read_restart_state(path)
    assert metadata['stage'] == 'completed'
    restored = evaluate_same_source_cc(ref, resume_from=path)
    assert restored.e_triples_per_cell == pytest.approx(fresh.e_triples_per_cell, abs=1e-13)
    assert restored.resumed_triples_work == 10


def test_cc_archive_rejects_schema_space_options_and_mutation(monkeypatch, tmp_path):
    from vibeqc.output.checkpoint import read_restart_state, write_restart_state
    ref = real_hamiltonian_uhf(monkeypatch)
    path = tmp_path/'original.restart'
    opts = ChiCCOptions(compute_triples=False)
    evaluate_same_source_cc(ref, options=opts, checkpoint_path=path)
    metadata, arrays = read_restart_state(path)
    altered = tmp_path/'altered.restart'
    for field, value in [('schema', 'vibeqc.chi.canonical-cc/v999'), ('stage', 'unknown')]:
        write_restart_state(altered, dict(metadata, **{field: value}), arrays)
        with pytest.raises(ValueError, match='schema|incompatible'):
            evaluate_same_source_cc(ref, options=opts, resume_from=altered)
    for key in ('eps', 'spin_labels', 't1'):
        changed = {k:np.array(v, copy=True) for k,v in arrays.items()}
        changed[key].flat[0] += .1 if key != 'spin_labels' else 1
        write_restart_state(altered, metadata, changed)
        with pytest.raises(ValueError, match='incompatible|convergence revalidation|spin/momentum'):
            evaluate_same_source_cc(ref, options=opts, resume_from=altered)
    with pytest.raises(ValueError, match='controls are incompatible'):
        evaluate_same_source_cc(ref, options=replace(opts, residual_tolerance=1e-7), resume_from=path)
    with pytest.raises(ValueError, match='controls are incompatible'):
        evaluate_same_source_cc(ref, options=opts, frozen_core=(1, 0), resume_from=path)
    with pytest.raises(ValueError, match='max_iter may only increase'):
        evaluate_same_source_cc(ref, options=replace(opts, max_iter=90), resume_from=path)
    # A checksum-valid archive cannot make a partial triples correction final.
    triples_path = tmp_path/'triples.restart'
    evaluate_same_source_cc(ref, checkpoint_path=triples_path)
    complete, values = read_restart_state(triples_path)
    complete['triples']['completed_work'] -= 1
    write_restart_state(altered, complete, values)
    with pytest.raises(ValueError, match='incomplete coverage'):
        evaluate_same_source_cc(ref, resume_from=altered)


def test_cc_solver_archive_continues_in_a_fresh_native_free_process(tmp_path):
    """Persist solver arrays, then use a separate interpreter and fresh DIIS objects."""
    from pathlib import Path
    import subprocess
    import sys
    from vibeqc.output.checkpoint import read_restart_state, write_restart_state
    f, g, _, _ = model(n=6, no=2)
    opts = ChiCCOptions()
    states = []
    with pytest.raises(RuntimeError, match='did not converge'):
        solve(Dense(g, 2), np.diag(f), replace(opts, max_iter=3), progress_callback=states.append)
    saved = states[-1]
    source, output = tmp_path/'input.restart', tmp_path/'output.restart'
    metadata = {k:saved[k] for k in ('iteration', 'converged')}
    arrays = {k:v for k,v in saved.items() if k not in metadata}
    arrays.update(g=g, eps=np.diag(f))
    write_restart_state(source, metadata, arrays)
    script = '''
import sys, types, pathlib, importlib.abc
root = pathlib.Path(sys.argv[1])
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if '_vibeqc_core' in fullname:
            raise ImportError('native code prohibited in restart source proof')
sys.meta_path.insert(0, NoNative())
for name, relative in [('vibeqc','python/vibeqc'), ('vibeqc.periodic','python/vibeqc/periodic'), ('vibeqc.periodic.chi','python/vibeqc/periodic/chi'), ('vibeqc.output','python/vibeqc/output')]:
    module = types.ModuleType(name)
    module.__path__ = [str(root/relative)]
    sys.modules[name] = module
from vibeqc.output.checkpoint import read_restart_state, write_restart_state
from vibeqc.periodic.chi.canonical_cc import solve, ChiCCOptions
class Dense:
    def __init__(self, g): self.g, self.no, self.shape = g, 2, g.shape
    def __getitem__(self, key): return self.g[key]
metadata, arrays = read_restart_state(sys.argv[2])
g, eps = arrays.pop('g'), arrays.pop('eps')
energy, t1, t2, trace, _ = solve(Dense(g), eps, ChiCCOptions(), restart_state=dict(metadata, **arrays))
write_restart_state(sys.argv[3], {'energy':energy, 'iterations':len(trace)}, {'t1':t1, 't2':t2})
'''
    proc = subprocess.run([sys.executable, '-c', script, str(Path(__file__).resolve().parents[1]),
                           str(source), str(output)], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stdout+proc.stderr
    restored, values = read_restart_state(output)
    fresh = solve(Dense(g, 2), np.diag(f), opts)
    assert restored['iterations'] == len(fresh[3])
    assert restored['energy'] == pytest.approx(fresh[0], abs=1e-13)
    np.testing.assert_allclose(values['t1'], fresh[1], atol=1e-13)
    np.testing.assert_allclose(values['t2'], fresh[2], atol=1e-13)


def test_momentum_triples_restart_matches_independent_determinants():
    """Complex phases, forbidden work units and a complete determinant oracle."""
    f, g, _, _ = (x.real for x in model(n=7, no=3))
    momentum = np.array([0, 1, 2, 0, 1, 2, 0])
    conserved = ((momentum[:, None, None, None]+momentum[None, :, None, None]
                  -momentum[None, None, :, None]-momentum[None, None, None, :]) % 3 == 0)
    g = g*conserved
    phases = np.exp(1j*np.arange(7)*.37)
    g = np.einsum('p,q,r,s,pqrs->pqrs', phases.conj(), phases.conj(), phases, phases, g)
    eps = np.diag(f)
    no, nv = 3, 4
    den = eps[:no][None, None, :, None]+eps[:no][None, None, None, :]-eps[no:][:, None, None, None]-eps[no:][None, :, None, None]
    t1 = np.zeros((nv, no), complex)
    t2 = g[no:, no:, :no, :no]/den
    expected = determinant_triples(f, g, t1, t2)
    assert abs(expected) > 1e-12

    class MomentumDense(Dense):
        def allowed_excitation(self, occupied, virtual):
            return (sum(momentum[i] for i in occupied)-sum(momentum[a] for a in virtual)) % 3 == 0

    states = []

    def interrupt(state):
        states.append(state)
        if state['completed_work'] == 2:
            raise Interrupted()

    integrals = MomentumDense(g, no)
    with pytest.raises(Interrupted):
        triples(t1, t2, eps, integrals, progress_callback=interrupt)
    assert states[-1]['evaluated_work'] == 1
    actual = triples(t1, t2, eps, integrals, restart_state=states[-1], progress_callback=states.append)
    assert states[-1]['total_work'] == states[-1]['completed_work'] == 4
    assert states[-1]['evaluated_work'] == 2
    assert actual == pytest.approx(expected, abs=2e-13)
