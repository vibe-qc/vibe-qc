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


@pytest.mark.parametrize('suffix', ['', '_complex'])
@pytest.mark.parametrize('bad', [1.8, True, np.nan, 2**63, -1, 0])
def test_home_mo_mesh_rejects_before_narrowing(suffix, bad):
    core = pytest.importorskip('vibeqc._vibeqc_core')
    fn = getattr(core, 'aiccm2026dev_b_home_auxiliary_3index_mo_transform' + suffix)
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        fn(np.ones((1, 1, 1)), np.ones((1, 1)), np.ones((1, 1)), [bad, 1, 1])


@pytest.mark.parametrize('home', [False, True])
@pytest.mark.parametrize('suffix', ['', '_complex'])
@pytest.mark.parametrize('which', [0, 1, 2])
def test_mo_transform_rejects_nonfinite_operands(home, suffix, which):
    core = pytest.importorskip('vibeqc._vibeqc_core')
    name = 'aiccm2026dev_b_' + ('home_auxiliary_' if home else '') + '3index_mo_transform' + suffix
    args = [np.ones((1, 1, 1)), np.ones((1, 1)), np.ones((1, 1))]
    args[which].flat[0] = np.nan
    if home:
        args.append([1, 1, 1])
    with pytest.raises(ValueError, match='finite'):
        getattr(core, name)(*args)


@pytest.mark.parametrize('home', [False, True])
@pytest.mark.parametrize('which', [0, 1, 2])
def test_real_mo_transform_does_not_discard_imaginary_input(home, which):
    core = pytest.importorskip('vibeqc._vibeqc_core')
    args = [np.ones((1, 1, 1)), np.ones((1, 1)), np.ones((1, 1))]
    args[which] = args[which].astype(complex) * (1 + 2j)
    if home:
        args.append([1, 1, 1])
    name = 'aiccm2026dev_b_' + ('home_auxiliary_' if home else '') + '3index_mo_transform'
    with pytest.raises(ValueError, match='real'):
        getattr(core, name)(*args)


@pytest.mark.parametrize('suffix', ['', '_complex'])
def test_home_mo_preserves_empty_orbital_space(suffix):
    core = pytest.importorskip('vibeqc._vibeqc_core')
    fn = getattr(core, 'aiccm2026dev_b_home_auxiliary_3index_mo_transform' + suffix)
    result = fn(np.ones((1, 2, 2)), np.empty((2, 0)), np.ones((2, 1)), [2, 1, 1])
    assert result.shape == (2, 0, 1)


@pytest.mark.parametrize('bad', [1.8, True, np.nan, 2**63, -1, 0])
def test_permutation_boundaries_reject_invalid_counts(bad):
    core = pytest.importorskip('vibeqc._vibeqc_core')
    pno = pytest.importorskip('vibeqc.periodic.chi.pno')
    for fn in (core.aiccm2026dev_b_translation_permutations,
               pno.translation_permutations, pno._translation_permutations_native,
               pno._translation_permutations_python):
        for bands, mesh in [(1, [bad, 1, 1]), (bad, [1, 1, 1])]:
            with pytest.raises((ValueError, RuntimeError, TypeError)):
                fn(bands, mesh)
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        core.aiccm2026dev_b_pair_orbits([[0]], bad)
    for fn in (pno.enumerate_pair_orbits, pno._pair_orbits_from_native,
               pno._enumerate_pair_orbits_python):
        with pytest.raises(ValueError):
            fn([[0]], n_cells=bad)


@pytest.mark.parametrize('bad', [0.8, True, np.nan, 2**63])
def test_pair_permutations_reject_before_cast(bad):
    core = pytest.importorskip('vibeqc._vibeqc_core')
    pno = pytest.importorskip('vibeqc.periodic.chi.pno')
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        core.aiccm2026dev_b_pair_orbits([[bad]], 1)
    for fn in (pno.enumerate_pair_orbits, pno._pair_orbits_from_native,
               pno._enumerate_pair_orbits_python):
        with pytest.raises(ValueError):
            fn([[bad]], n_cells=1)


def test_translation_permutation_and_pair_closure_oracles():
    pno = pytest.importorskip('vibeqc.periodic.chi.pno')
    # Two bands on a three-cell cyclic group. Direct integer label action is
    # independent of the native and fallback group enumeration.
    expected = np.array([[((cell + shift) % 3) * 2 + band
                          for cell in range(3) for band in range(2)]
                         for shift in range(3)])
    for fn in (pno.translation_permutations, pno._translation_permutations_python):
        np.testing.assert_array_equal(fn(2, [3, 1, 1]), expected)
    native = pno.enumerate_pair_orbits(expected, n_cells=3)
    fallback = pno._enumerate_pair_orbits_python(expected, n_cells=3)
    assert native == fallback
    assert sorted(pair for orbit in native for pair in orbit.members) == [
        (i, j) for i in range(6) for j in range(i, 6)]
    assert sum(orbit.per_cell_weight for orbit in native) == 7.0


@pytest.mark.parametrize('mesh,bands', [([2**62, 4, 1], 1), ([2**32, 1, 1], 1), ([2, 1, 1], 2**62)])
def test_translation_permutation_output_overflow(mesh, bands):
    core = pytest.importorskip('vibeqc._vibeqc_core')
    pno = pytest.importorskip('vibeqc.periodic.chi.pno')
    for fn in (core.aiccm2026dev_b_translation_permutations, pno.translation_permutations):
        with pytest.raises((ValueError, RuntimeError)):
            fn(bands, mesh)


@pytest.mark.parametrize('name', ['tensor', 'matrix'])
@pytest.mark.parametrize('which,bad', [('mesh', [1.8, 1, 1]), ('mesh', [True, 1, 1]),
    ('mesh', [2**62, 4, 1]), ('atoms', [0.8]), ('atoms', [True]),
    ('atoms', [2**63-1]), ('density', [[[np.nan]]]), ('overlap', [[[np.inf]]]),
    ('density', [[[1+2j]]]), ('beta', [[[complex(1, np.nan)]]])])
def test_mayer_transform_rejects_invalid_raw_inputs(name, which, bad):
    core = pytest.importorskip('vibeqc._vibeqc_core')
    args = dict(density=[[[1.]]], overlap=[[[1.]]], atoms=[0], mesh=[1, 1, 1], beta=None)
    args[which] = bad
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        getattr(core, 'aiccm2026dev_b_mayer_pair_' + name)(
            args['density'], args['overlap'], args['atoms'], args['mesh'], args['beta'])


@pytest.mark.parametrize('mesh', [[1.8, 1, 1], [True, 1, 1], [2**62, 4, 1], [1, 1], [0, 1, 1]])
def test_adapter_and_property_mesh_admission(mesh):
    from types import SimpleNamespace
    posthf = pytest.importorskip('vibeqc.periodic.chi.posthf')
    properties = pytest.importorskip('vibeqc.periodic.chi.properties')
    with pytest.raises(ValueError):
        posthf._BRealDFAdapter(np.ones((1, 1, 1)), None, mesh)
    with pytest.raises(ValueError):
        properties._mesh_from_result(SimpleNamespace(aiccm2026dev_b=SimpleNamespace(mesh=mesh)))


def test_adapter_rejects_nonfinite_before_native(monkeypatch):
    posthf = pytest.importorskip('vibeqc.periodic.chi.posthf')
    with pytest.raises(ValueError):
        posthf._BRealDFAdapter(np.full((1, 1, 1), np.nan), None, (1, 1, 1))
    adapter = posthf._BRealDFAdapter(np.ones((1, 1, 1)), None, (1, 1, 1))
    def forbidden(*args):
        raise AssertionError('invalid coefficients reached native dispatch')
    monkeypatch.setattr(posthf, 'aiccm2026dev_b_home_auxiliary_3index_mo_transform', forbidden)
    with pytest.raises(ValueError):
        adapter.mo_transform([[np.inf]])


@pytest.mark.parametrize('bad', [True, 1.8, np.nan, 2**63])
def test_native_orbital_counts_reject_before_cast(bad):
    core = pytest.importorskip('vibeqc._vibeqc_core')
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        core.aiccm2026dev_b_canonical_wannier_coefficients(
            np.ones((1, 1, 1)), [[0, 0, 0]], [[0, 0, 0]], bad)
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        core.aiccm2026dev_b_localization_projector_audit(
            np.ones((1, 1)), np.ones((1, 1)), None, [[0, 0, 0]], [1, 1, 1], bad)


def test_localization_mesh_overflow_refuses_before_products():
    core = pytest.importorskip('vibeqc._vibeqc_core')
    with pytest.raises(RuntimeError, match='mesh product'):
        core.aiccm2026dev_b_localization_projector_audit(
            np.ones((1, 1)), np.ones((1, 1)), None, [[0, 0, 0]], [2**62, 4, 1], 1)
