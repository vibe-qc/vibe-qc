"""Finite-source restart proofs; no native chemistry acceptance is inferred."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from vibeqc.output import checkpoint as storage
from vibeqc.periodic.chi import canonical_mp2 as mp
from vibeqc.periodic.chi import canonical_rohf_mp2 as rmp
from tests.test_chi_canonical_mp2 import fixture, spin_orbital_oracle
from tests.test_chi_rohf_mp2 import fixture as rohf_fixture, reseal


class ForcedInterruption(RuntimeError):
    pass


def interrupt_after(monkeypatch, completed):
    writer = storage.write_restart_state

    def publish(path, metadata, arrays):
        result = writer(path, metadata, arrays)
        if metadata['completed_work_units'] == completed:
            raise ForcedInterruption('forced after atomic publication')
        return result

    monkeypatch.setattr(storage, 'write_restart_state', publish)
    return writer


def components(result):
    return (result.e_aa_per_cell, result.e_bb_per_cell, result.e_ab_per_cell,
            getattr(result, 'e_singles_alpha_per_cell', 0.),
            getattr(result, 'e_singles_beta_per_cell', 0.))


@pytest.mark.parametrize('method,nk,core,stop', [
    ('RHF', 1, 0, 1), ('RHF', 3, 0, 28),
    ('UHF', 1, (1, 0), 2), ('UHF', 3, (1, 0), 55),
    ('ROHF', 1, 0, 1), ('ROHF', 1, 0, 2), ('ROHF', 3, 1, 30),
])
def test_interrupted_components_resume_without_replay(monkeypatch, tmp_path, method, nk, core, stop):
    if method == 'ROHF':
        ref, _ = rohf_fixture(monkeypatch, nk=nk, nc=2)
        evaluate = rmp.evaluate_same_source_rohf_mp2
    else:
        ref, _ = fixture(monkeypatch, nk=nk, unrestricted=method == 'UHF', source='classic')
        evaluate = mp.evaluate_same_source_mp2
    counts = dict(integrals=0, singles=0)
    integral, single = mp.CanonicalFactorAdapter.integral, rmp._single_term

    def counted_integral(*args, **kwargs):
        counts['integrals'] += 1
        return integral(*args, **kwargs)

    def counted_single(*args, **kwargs):
        counts['singles'] += 1
        return single(*args, **kwargs)

    monkeypatch.setattr(mp.CanonicalFactorAdapter, 'integral', counted_integral)
    monkeypatch.setattr(rmp, '_single_term', counted_single)
    options = dict(frozen_core=core, auxiliary_batch_size=1)
    expected = evaluate(ref, **options)
    expected_counts = counts.copy()
    counts.update(integrals=0, singles=0)
    path = tmp_path / 'correlation.vqr'
    writer = interrupt_after(monkeypatch, stop)
    with pytest.raises(ForcedInterruption):
        evaluate(ref, checkpoint_path=path, **options)
    metadata, arrays = storage.read_restart_state(path)
    assert metadata['status'] == 'unfinished-correlation'
    assert metadata['completed_work_units'] == stop
    assert sum(arrays['component_completed']) == stop
    monkeypatch.setattr(storage, 'write_restart_state', writer)
    actual = evaluate(ref, resume_from=path, **options)
    assert components(actual) == pytest.approx(components(expected), abs=2e-13)
    assert actual.energy == pytest.approx(expected.energy, abs=2e-13)
    assert actual.active_spaces == expected.active_spaces
    assert actual.restart_mode == 'resumed-correlation'
    assert actual.resumed_work_units == stop
    assert actual.completed_work_units == actual.total_work_units
    assert counts == expected_counts
    restored = evaluate(ref, resume_from=path, **options)
    assert components(restored) == components(actual)
    assert restored.restart_mode == 'restored-completed'
    assert restored.resumed_work_units == restored.total_work_units
    assert counts == expected_counts  # Neither singles nor doubles were replayed.
    assert str(tmp_path) not in json.dumps(storage.read_restart_state(path)[0])


@pytest.mark.parametrize('core', [0, (1, 0)])
def test_resumed_ump2_matches_independent_spin_orbital_sum(monkeypatch, tmp_path, core):
    ref, channels = fixture(monkeypatch, nk=3, unrestricted=True, source='classic')
    frozen = (core, core) if isinstance(core, int) else core
    expected = spin_orbital_oracle(ref.hf_result.fitted_operator, channels, frozen)
    path = tmp_path / 'ump2.vqr'
    writer = interrupt_after(monkeypatch, 3)
    with pytest.raises(ForcedInterruption):
        mp.evaluate_same_source_mp2(ref, frozen_core=core, checkpoint_path=path)
    monkeypatch.setattr(storage, 'write_restart_state', writer)
    actual = mp.evaluate_same_source_mp2(ref, frozen_core=core, resume_from=path)
    assert actual.e_corr_per_cell == pytest.approx(expected, abs=2e-13)


@pytest.mark.parametrize('mutation', ['coverage', 'cursor', 'status', 'schema', 'positive', 'unused', 'array-shape'])
def test_checksummed_but_invalid_work_state_refused(monkeypatch, tmp_path, mutation):
    ref, _ = fixture(monkeypatch, source='classic')
    path = tmp_path / 'state.vqr'
    writer = interrupt_after(monkeypatch, 1)
    with pytest.raises(ForcedInterruption):
        mp.evaluate_same_source_mp2(ref, checkpoint_path=path)
    monkeypatch.setattr(storage, 'write_restart_state', writer)
    metadata, arrays = storage.read_restart_state(path)
    arrays = {key: value.copy() for key, value in arrays.items()}
    if mutation == 'coverage':
        arrays['component_completed'][1] += 1
    elif mutation == 'cursor':
        metadata['completed_work_units'] += 1
    elif mutation == 'status':
        metadata['status'] = 'completed-result'
    elif mutation == 'schema':
        metadata['schema'] = 'vibeqc.chi.canonical-mp2-restart/v999'
    elif mutation == 'positive':
        arrays['accumulators'][0] = 1.
    elif mutation == 'unused':
        arrays['accumulators'][1] = -.1
    else:
        arrays['accumulators'] = arrays['accumulators'][:-1]
    writer(path, metadata, arrays)
    with pytest.raises(ValueError, match='restart'):
        mp.evaluate_same_source_mp2(ref, resume_from=path)


@pytest.mark.parametrize('options', [dict(frozen_core=1), dict(auxiliary_batch_size=1),
                                   dict(denominator_tolerance=1e-10)])
def test_changed_numerical_controls_refused(monkeypatch, tmp_path, options):
    ref, _ = fixture(monkeypatch, source='classic')
    path = tmp_path / 'state.vqr'
    mp.evaluate_same_source_mp2(ref, checkpoint_path=path)
    with pytest.raises(ValueError, match='incompatible'):
        mp.evaluate_same_source_mp2(ref, resume_from=path, **options)


@pytest.mark.parametrize('change', ['spin', 'frame', 'energy', 'source', 'rank'])
def test_different_independently_sealed_reference_refused(monkeypatch, tmp_path, change):
    ref, _ = fixture(monkeypatch, source='classic')
    path = tmp_path / 'state.vqr'
    mp.evaluate_same_source_mp2(ref, checkpoint_path=path)
    if change == 'spin':
        other, _ = fixture(monkeypatch, source='classic', unrestricted=True)
    else:
        other, _ = fixture(monkeypatch, source='classic')
        hf = other.hf_result
        if change == 'frame':
            hf.mo_coeffs = hf.mo_coeffs.copy()
            hf.mo_coeffs[:, :, 0] *= np.exp(.3j)
        elif change == 'energy':
            hf.energy -= .1
        else:
            # Same numeric factors, different physical source/support identity.
            from vibeqc.periodic_fitted_support import ChiFittedOperator
            from vibeqc.periodic_fit_source import factor_digest
            old = hf.fitted_operator
            factors = dict(old)
            if change == 'rank':
                records = json.loads(old.descriptor.executed_domains_json)
                for record in records['q_records']:
                    record.update(metric_eigenvalues=[1., 0.], metric_keep=[True, False], rank=1)
                factors = {pair: value.copy() for pair, value in old.items()}
                for value in factors.values():
                    value[1] = 0.
                changed = replace(old.descriptor, executed_domains_json=json.dumps(records),
                                  factor_sha256=factor_digest(factors))
            else:
                changed = replace(old.descriptor, source_signature='another-system')
            hf.fitted_operator = ChiFittedOperator(factors, changed, old._weights)
            hf.executed_fit_source = replace(changed, executed_domains_json=json.dumps(
                dict(fitted_operator_identity=hf.fitted_operator.identity)))
        other = reseal(other)
    with pytest.raises(ValueError, match='incompatible'):
        mp.evaluate_same_source_mp2(other, resume_from=path)


def test_no_work_frozen_result_round_trip(monkeypatch, tmp_path):
    ref, _ = fixture(monkeypatch, source='classic')
    path = tmp_path / 'empty.vqr'
    fresh = mp.evaluate_same_source_mp2(ref, frozen_core=1, checkpoint_path=path)
    assert fresh.restart_mode == 'fresh'
    assert fresh.total_work_units == fresh.completed_work_units == 0
    result = mp.evaluate_same_source_mp2(ref, frozen_core=1, resume_from=path)
    assert result.restart_mode == 'restored-completed'
    assert result.e_corr_per_cell == 0.


@pytest.mark.parametrize('interval', [0, -1, True, 1.5])
def test_invalid_checkpoint_interval_refused(monkeypatch, tmp_path, interval):
    ref, _ = fixture(monkeypatch, source='classic')
    with pytest.raises(ValueError, match='checkpoint_interval'):
        mp.evaluate_same_source_mp2(ref, checkpoint_path=tmp_path / 'bad.vqr', checkpoint_interval=interval)


def test_persistent_sr_remains_guarded(monkeypatch, tmp_path):
    ref, _ = fixture(monkeypatch, source='sr')
    with pytest.raises(NotImplementedError, match='classic only'):
        mp.evaluate_same_source_mp2(ref, checkpoint_path=tmp_path / 'sr.vqr')
    assert np.isfinite(mp.evaluate_same_source_mp2(ref).energy)


def test_failed_publication_reuses_last_complete_prefix(monkeypatch, tmp_path):
    ref, _ = rohf_fixture(monkeypatch, nk=3)
    expected = rmp.evaluate_same_source_rohf_mp2(ref)
    path = tmp_path / 'last-complete.vqr'
    writer = storage.write_restart_state

    def fail_before_publish(path, metadata, arrays):
        if metadata['completed_work_units'] == 3:
            raise OSError('forced publication failure')
        return writer(path, metadata, arrays)

    monkeypatch.setattr(storage, 'write_restart_state', fail_before_publish)
    with pytest.raises(OSError, match='forced'):
        rmp.evaluate_same_source_rohf_mp2(ref, checkpoint_path=path)
    assert storage.read_restart_state(path)[0]['completed_work_units'] == 2
    monkeypatch.setattr(storage, 'write_restart_state', writer)
    result = rmp.evaluate_same_source_rohf_mp2(ref, resume_from=path)
    assert result.resumed_work_units == 2
    assert components(result) == pytest.approx(components(expected), abs=2e-13)


def test_rohf_checkpoint_binds_shared_frozen_projector(monkeypatch, tmp_path):
    ref, _ = rohf_fixture(monkeypatch, nk=3, nc=2)
    path = tmp_path / 'shared-core.vqr'
    rmp.evaluate_same_source_rohf_mp2(ref, frozen_core=1, checkpoint_path=path)
    with pytest.raises(ValueError, match='incompatible'):
        rmp.evaluate_same_source_rohf_mp2(ref, frozen_core=0, resume_from=path)


def test_checkpoint_cadence_and_separate_resume_target(monkeypatch, tmp_path):
    ref, _ = fixture(monkeypatch, nk=3, source='classic')
    first, continued = tmp_path / 'first.vqr', tmp_path / 'continued.vqr'
    writer = interrupt_after(monkeypatch, 14)
    with pytest.raises(ForcedInterruption):
        mp.evaluate_same_source_mp2(ref, checkpoint_path=first, checkpoint_interval=7)
    first_bytes = first.read_bytes()
    monkeypatch.setattr(storage, 'write_restart_state', writer)
    result = mp.evaluate_same_source_mp2(ref, resume_from=first, checkpoint_path=continued,
                                         checkpoint_interval=11)
    assert result.resumed_work_units == 14
    assert first.read_bytes() == first_bytes
    assert storage.read_restart_state(continued)[0]['status'] == 'completed-result'


def test_portable_fresh_process_restores_completed_without_contraction(monkeypatch, tmp_path):
    """A new PID with no native-core import consumes the persisted work state."""
    ref, _ = fixture(monkeypatch, nk=3, unrestricted=True, source='classic')
    path = tmp_path / 'portable.vqr'
    expected = mp.evaluate_same_source_mp2(ref, checkpoint_path=path, frozen_core=(1, 0))
    script = r'''
import importlib.abc, json, pathlib, sys, types
root = pathlib.Path.cwd()
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if '_vibeqc_core' in fullname:
            raise ImportError('native imports forbidden in restart proof')
sys.meta_path.insert(0, NoNative())
for name, relative in [('vibeqc', 'python/vibeqc'), ('vibeqc.periodic', 'python/vibeqc/periodic'),
                       ('vibeqc.periodic.chi', 'python/vibeqc/periodic/chi'),
                       ('vibeqc.output', 'python/vibeqc/output'), ('tests', 'tests')]:
    module = types.ModuleType(name)
    module.__path__ = [str(root / relative)]
    sys.modules[name] = module
import pytest
from tests.test_chi_canonical_mp2 import fixture
from vibeqc.periodic.chi import canonical_mp2 as mp
def forbidden(*args, **kwargs):
    raise AssertionError('completed contributions were replayed')
with pytest.MonkeyPatch.context() as patch:
    reference, _ = fixture(patch, nk=3, unrestricted=True, source='classic')
    patch.setattr(mp.CanonicalFactorAdapter, 'integral', forbidden)
    result = mp.evaluate_same_source_mp2(reference, resume_from=sys.argv[1], frozen_core=(1, 0))
    print(json.dumps(dict(energy=result.energy, mode=result.restart_mode)))
'''
    process = subprocess.run([sys.executable, '-c', script, str(path)],
                             cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    actual = json.loads(process.stdout)
    assert actual['energy'] == pytest.approx(expected.energy, abs=2e-13)
    assert actual['mode'] == 'restored-completed'
