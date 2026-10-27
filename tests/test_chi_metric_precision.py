"""Independent small spectral controls; no native runtime required."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

mpmath = pytest.importorskip('mpmath')
spec = importlib.util.spec_from_file_location('classic_spectral_test',
    Path(__file__).parents[1]/'python/vibeqc/_classic_spectral.py')
spectral = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spectral)


def solve(m, threshold=1e-9, digits=35):
    return spectral.retained_spectral_construction(m, threshold, digits=digits,
        memory_byte_cap=spectral.spectral_workspace_bytes(len(m), digits))


@pytest.mark.parametrize('digits', [30, 45])
def test_complete_inverse_root_and_rank_boundary(digits):
    m = np.diag([.985e-9, 1.035e-9, 2., 2.])
    before = m.copy()
    e, u, w, receipt = solve(m, digits=digits)
    expected = np.diag([0., 1/np.sqrt(m[1, 1]), 1/np.sqrt(2.), 1/np.sqrt(2.)])
    np.testing.assert_allclose(w, expected, rtol=2e-16, atol=0)
    np.testing.assert_allclose(w @ m @ w, np.diag([0., 1., 1., 1.]), atol=5e-16)
    np.testing.assert_array_equal(m, before)
    assert receipt['rank'] == 3 and receipt['original_double_rank_preserved']
    assert receipt['inverse_root_precision_difference'] < 1e-20
    assert receipt['verification_digits'] == digits+20


def test_conjugate_metric_and_reordered_auxiliary_degeneracy():
    m = np.array([[2., .2j, 0., 0.], [-.2j, 1., 0., 0.],
                  [0., 0., 3., 0.], [0., 0., 0., 3.]], complex)
    _, _, w, _ = solve(m)
    _, _, opposite, _ = solve(m.conj())
    np.testing.assert_array_equal(w, opposite.conj())
    order = [3, 1, 0, 2]
    _, _, permuted, _ = solve(m[np.ix_(order, order)])
    np.testing.assert_allclose(permuted, w[np.ix_(order, order)], atol=1e-15)
    np.testing.assert_allclose(w @ m @ w, np.eye(4), atol=5e-16)


def test_threshold_equality_refuses_without_clipping():
    with pytest.raises(ValueError, match='boundary'):
        solve(np.diag([1e-9, 1.]))


def test_changed_rank_refuses(monkeypatch):
    original = np.linalg.eigh
    def wrong_rank(m):
        e, u = original(m)
        e[0] = 2e-9
        return e, u
    monkeypatch.setattr(np.linalg, 'eigh', wrong_rank)
    with pytest.raises(ValueError, match='rank'):
        solve(np.diag([.5e-9, 1.]))


def test_no_global_precision_mutation():
    precision = mpmath.mp.dps
    solve(np.eye(2))
    assert mpmath.mp.dps == precision


def test_budget_refuses_before_diagonalization(monkeypatch):
    def forbidden(*args):
        pytest.fail('diagonalization before workspace admission')
    monkeypatch.setattr(np.linalg, 'eigh', forbidden)
    with pytest.raises(MemoryError):
        spectral.retained_spectral_construction(np.eye(2), 1e-9, digits=35,
            memory_byte_cap=spectral.spectral_workspace_bytes(2, 35)-1)


@pytest.mark.parametrize('digits', [0, 29, 101, 35.5, True])
def test_invalid_precision(digits):
    with pytest.raises(ValueError):
        solve(np.eye(2), digits=digits)


def test_nonhermitian_metric_not_repaired():
    with pytest.raises(ValueError, match='Hermitian'):
        solve(np.array([[1., 1e-16], [0., 1.]]))


def test_bad_eigensystem_refuses(monkeypatch):
    clone = mpmath.mp.clone
    def broken_context():
        ctx = clone()
        eighe = ctx.eighe
        def broken(a):
            e, u = eighe(a)
            u[0, 0] += ctx.mpf('1e-15')
            return e, u
        ctx.eighe = broken
        return ctx
    monkeypatch.setattr(mpmath.mp, 'clone', broken_context)
    with pytest.raises(ValueError, match='did not converge'):
        solve(np.diag([1., 2.]))


def test_missing_optional_dependency_refuses(monkeypatch):
    import builtins
    original = builtins.__import__
    def missing(name, *args, **kwargs):
        if name == 'mpmath':
            raise ImportError('deliberate missing dependency')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', missing)
    with pytest.raises(ImportError, match='chi-precision'):
        solve(np.eye(2))
