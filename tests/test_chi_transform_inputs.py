"""Small algebra/admission tests; optional native checks need a matching core."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

# Keep the pure admission algebra executable without importing a native runtime.
_spec = spec_from_file_location('chi_transform_inputs', Path(__file__).resolve().parents[1] / 'python/vibeqc/periodic/chi/_transform_inputs.py')
inputs = module_from_spec(_spec)
_spec.loader.exec_module(inputs)


@pytest.mark.parametrize('bad', [.8, np.nan, np.inf, -np.inf, 2**63, -(2**63)-1, np.uint64(2**63), float(2**63), True, 1+0j, '1'])
def test_translation_rejects_before_cast(bad):
    with pytest.raises(ValueError):
        inputs.translations_array([[bad, 0, 0]])


@pytest.mark.parametrize('bad', [[], [0, 0, 0], [[0, 0]], np.zeros((1, 1, 3))])
def test_translation_shapes(bad):
    with pytest.raises(ValueError):
        inputs.translations_array(bad)


def test_integral_float_and_int64_limits_preserved():
    np.testing.assert_array_equal(inputs.translations_array([[1., -2., 0.]]), [[1, -2, 0]])
    assert inputs.translations_array([[-2**63, 2**63-1, 0]]).tolist() == [[-2**63, 2**63-1, 0]]


@pytest.mark.parametrize('weights', [[1.5,-.5], [np.nan,1], [np.inf,0], [1,1], [[.5,.5]], [.5], [.5+.1j,.5]])
def test_bad_weights(weights):
    with pytest.raises(ValueError):
        inputs.inverse_inputs(np.ones((2,1,1)), [[0,0,0],[.5,0,0]], weights)


@pytest.mark.parametrize('weights', [[.5,.5], [1.,0.], [.2,.8]])
def test_supported_weight_conventions(weights):
    assert inputs.inverse_inputs(np.ones((2,1,1)), [[0,0,0],[.5,0,0]], weights)[2].tolist() == weights


@pytest.mark.parametrize('matrices,k', [
    (np.ones((0,1,1)), np.empty((0,3))), (np.ones((1,0,0)), [[0,0,0]]),
    (np.ones((1,1,2)), [[0,0,0]]), (np.ones((1,1,1)), [0,0,0]),
    (np.ones((1,1,1)), [[np.nan,0,0]]), (np.ones((1,1,1)), [[1j,0,0]]),
    (np.array([[[np.inf]]]), [[0,0,0]]), (np.array([[[1+np.nan*1j]]]), [[0,0,0]]),
])
def test_finite_nonempty_shapes(matrices,k):
    with pytest.raises(ValueError):
        inputs.inverse_inputs(matrices,k)


@pytest.mark.parametrize('extension,k,translations', [
    (np.diag([3,1,1]), [[0,0,0],[1/3,0,0],[2/3,0,0]], [[0,0,0],[1,0,0],[2,0,0]]),
    ([[2,1,0],[0,2,0],[0,0,1]], [[0,0,0],[0,.5,0],[.5,.25,0],[.5,.75,0]], [[0,0,0],[0,1,0],[1,0,0],[1,1,0]]),
    ([[-2,1,0],[0,1,0],[0,0,1]], [[0,0,0],[.5,.5,0]], [[0,0,0],[1,0,0]]),
])
def test_complete_character_complex_round_trip(extension,k,translations):
    k = np.asarray(k); n=len(k); weights=np.full(n,1/n)
    inputs.complete_characters(k, weights, extension)
    rng=np.random.default_rng(419); matrices=rng.normal(size=(n,2,2))+1j*rng.normal(size=(n,2,2))
    phases=np.exp(-2j*np.pi*inputs.phase_arguments(k,np.array(translations)))
    real=np.einsum('rk,kij->rij',phases,matrices)/n
    # Independent forward formula at these small exact representatives.
    forward=np.exp(2j*np.pi*(k@np.asarray(translations).T))
    np.testing.assert_allclose(np.einsum('kr,rij->kij',forward,real),matrices,atol=3e-14,rtol=0)


@pytest.mark.parametrize('s', [np.zeros((3,3)), np.eye(2), np.diag([1.1,1,1]), np.diag([np.inf,1,1]), np.diag([2**62,4,1]), [[2**63,0,0],[0,1,0],[0,0,1]]])
def test_invalid_extension(s):
    with pytest.raises(ValueError): inputs.complete_characters([[0,0,0]],[1],s)


@pytest.mark.parametrize('k,w', [([[0,0,0],[0,0,0]],[.5,.5]), ([[0,0,0],[.25,0,0]],[.5,.5]), ([[0,0,0],[.5,0,0]],[1,0]), ([[0,0,0],[.5,0,0]],[.2,.8])])
def test_inconsistent_complete_characters(k,w):
    with pytest.raises(ValueError): inputs.complete_characters(k,w,np.diag([2,1,1]))


def test_large_extension_membership_is_exact():
    s=[[2,2**54+1,0],[0,1,0],[0,0,1]]
    with pytest.raises(ValueError): inputs.complete_characters([[0,0,0],[.5,0,0]],[.5,.5],s)
    inputs.complete_characters([[0,0,0],[.5,.5,0]],[.5,.5],s)


def test_large_translation_phase_and_reciprocal_representatives():
    r=np.array([[2**53+1,0,0],[-2**63,0,0],[2**63-1,0,0]],dtype=np.int64)
    np.testing.assert_array_equal(inputs.phase_arguments(np.array([[.5,0,0],[1e308,0,0]]),r), [[.5,0],[0,0],[.5,0]])


@pytest.mark.parametrize('mesh', [(2**63,1,1),(2**32,2**32,1),(0,1,1)])
def test_mesh_overflow(mesh):
    with pytest.raises(ValueError): inputs.checked_mesh_size(mesh)


@pytest.fixture
def kernel():
    return pytest.importorskip('vibeqc._vibeqc_core').aiccm2026dev_b_inverse_bloch_transform


@pytest.mark.parametrize('which,bad', [
    ('r',[[.8,0,0]]), ('r',[[2**63,0,0]]), ('r',[[True,0,0]]),
    ('r',[[1j,0,0]]), ('r',np.empty((0,3))), ('k',[[1j,0,0]]),
    ('k',[[np.inf,0,0]]), ('w',[-1.]), ('w',[np.nan]), ('w',[1j]),
    ('m',[[[np.nan]]]), ('m',np.empty((1,0,0))),
])
def test_direct_native_refuses_invalid_inputs(kernel,which,bad):
    args=dict(m=np.ones((1,1,1)), k=[[0,0,0]], r=[[0,0,0]], w=[1.])
    args[which]=bad
    with pytest.raises((ValueError,RuntimeError,TypeError)):
        kernel(args['m'],args['k'],args['r'],args['w'])


@pytest.mark.parametrize('weights', [[.5,.5],[1.,0.],[.2,.8]])
def test_native_weighted_sum_and_large_phase(kernel,weights):
    matrices=np.array([[[1+2j]],[[3-4j]]]); k=np.array([[0,0,0],[.5,0,0]])
    r=np.array([[0,0,0],[1,0,0],[2**53+1,0,0],[-2**63,0,0]])
    phases=np.array([[1,1],[1,-1],[1,-1],[1,1]])
    expected=np.einsum('rk,k,kij->rij',phases,weights,matrices)
    np.testing.assert_allclose(kernel(matrices,k,r,weights),expected,atol=2e-14,rtol=0)


def test_public_and_private_dispatch_reject_before_native(monkeypatch):
    scf=pytest.importorskip('vibeqc.periodic.chi.scf')
    calls=[]
    def forbidden(*args):
        calls.append(args)
        raise AssertionError('invalid data reached native dispatch')
    monkeypatch.setattr(scf,'_inverse_bloch_transform_native',forbidden)
    for r,w in [([[.8,0,0]],[.5,.5]), ([[0,0,0]],[1.5,-.5])]:
        with pytest.raises(ValueError): scf.inverse_bloch_transform(np.ones((2,1,1)),[[0,0,0],[.5,0,0]],r,w)
    assert not calls


def test_private_dispatch_cannot_narrow_invalid_data():
    scf=pytest.importorskip('vibeqc.periodic.chi.scf')
    for r,w in [([[.8,0,0]],[1.]), ([[0,0,0]],[-1.])]:
        with pytest.raises(ValueError):
            scf._inverse_bloch_transform_native(np.ones((1,1,1)), [[0,0,0]], r, w)


@pytest.mark.parametrize('name', ['aiccm2026dev_b_matrix_to_real_supercell','aiccm2026dev_b_canonical_wannier_coefficients','aiccm2026dev_b_lpq_to_real_supercell','aiccm2026dev_b_lpq_to_real_home_auxiliary'])
@pytest.mark.parametrize('bad', [.8, np.nan, 2**63, 1j])
def test_sibling_native_transform_translation_guards(name,bad):
    core=pytest.importorskip('vibeqc._vibeqc_core')
    data=np.ones((1,1,1),dtype=complex)
    if 'lpq' in name: data={(0,0):data}
    args=[data,[[0,0,0]],[[bad,0,0]]]
    if 'wannier' in name: args.append(1)
    if 'home_auxiliary' in name: args.append([1,1,1])
    with pytest.raises((ValueError,RuntimeError,TypeError)):
        getattr(core,name)(*args)


def test_native_factor_transform_rejects_nonfinite_cache():
    core=pytest.importorskip('vibeqc._vibeqc_core')
    with pytest.raises(ValueError):
        core.aiccm2026dev_b_lpq_to_real_supercell({(0,0):np.array([[[np.nan]]])},[[0,0,0]],[[0,0,0]])


def test_public_complete_character_round_trip(monkeypatch):
    scf=pytest.importorskip('vibeqc.periodic.chi.scf')
    k=np.array([[0,0,0],[0,.5,0],[.5,.25,0],[.5,.75,0]])
    r=np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,0]])
    s=[[2,1,0],[0,2,0],[0,0,1]]
    values=np.arange(16).reshape(4,2,2)+.2j
    native=scf.inverse_bloch_transform(values,k,r,extension_matrix=s)
    monkeypatch.setattr(scf,'_inverse_bloch_transform_native',lambda *a: None)
    fallback=scf.inverse_bloch_transform(values,k,r,extension_matrix=s)
    np.testing.assert_allclose(native,fallback,atol=2e-14,rtol=0)
    forward=np.exp(2j*np.pi*(k@r.T))
    np.testing.assert_allclose(np.einsum('kr,rij->kij',forward,fallback),values,atol=3e-14,rtol=0)
    with pytest.raises(ValueError): scf.inverse_bloch_transform(values,k,r,[1.,0,0,0],extension_matrix=s)
