"""Executable native acceptance for experimental chi physical fitted kernels.

This module must run with the ordinary suite provenance guard and a rebuilt
core. It never relaxes stale-core checks. Missing native dependencies are
skips, not acceptance. JUnit properties retain source hashes, kernel controls,
and numerical residuals; put the JUnit report outside the product checkout.

The primitive-s oracle follows the Gaussian product/Boys formula in Boys,
Proc. R. Soc. A 200, 542 (1950), doi:10.1098/rspa.1950.0036, and the periodic
SR/LR decomposition in Ye and Berkelbach, JCP 154, 131104 (2021),
doi:10.1063/5.0046617, Eqs. (12)-(14). Its plain integer enumeration, Boys
functions and analytic Gaussian Fourier transforms do not call native
integrals, fitting helpers or another quantum-chemistry program.
The authors' accepted manuscript is https://arxiv.org/abs/2102.02989.

Thresholds below are declared before execution. Raw finite-source equality
does not assert energy convergence. Classic and SR fits are independently
compared with a denser reciprocal oracle before their fixed-density checks.
These tests do not replace independent self-consistent RKS/UKS acceptance.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import erfc, hyp1f1


RAW_ATOL = 3e-11
RAW_RTOL = 3e-12
GRAM_ATOL = 3e-9
ORACLE_CONVERGENCE_ATOL = 1e-11
MATRIX_ATOL = 3e-9
FD_ATOL = 3e-8
FD_STEP = 1e-5
COLLAPSE_ATOL = 2e-11
OMEGA = .55
SPLIT = .7
METRIC_THRESHOLD = 1e-10
NATIVE_INPUTS = (
    'cpp/include/vibeqc/periodic_gdf_short_range.hpp',
    'cpp/src/periodic_gdf_short_range.cpp',
    'cpp/src/periodic_gdf_short_range_bindings.cpp',
)


@dataclass(frozen=True)
class Gaussian:
    exponent: float
    coefficient: float
    center: np.ndarray


@pytest.fixture(scope='module')
def native(request):
    assert not request.config.getoption('noconftest'), 'native acceptance requires the ordinary suite provenance guard'
    assert not getattr(request.config, '_vibeqc_stale_core', None), (
        'native acceptance refuses the stale-core override; rebuild before measuring')
    vq = pytest.importorskip('vibeqc', reason='native chi acceptance unmeasured: package unavailable')
    core = pytest.importorskip('vibeqc._vibeqc_core', reason='native chi acceptance unmeasured: core unavailable')
    root = Path(__file__).resolve().parents[1]
    loaded = Path(core.__file__).resolve()
    assert loaded.is_file(), 'native acceptance requires a real compiled core artifact'
    stale = [name for name in NATIVE_INPUTS
             if (root/name).stat().st_mtime_ns > loaded.stat().st_mtime_ns]
    assert not stale, 'native chi acceptance refuses stale kernel inputs; rebuild the core: '+', '.join(stale)
    function = getattr(core, 'compute_gdf_range_separated_integrals', None)
    assert callable(function), 'native chi acceptance requires the combined SR/LR integral binding'
    signature = function.__doc__ or ''
    assert 'omega_screen' in signature and 'kernel' in signature, (
        'native chi acceptance requires the rebuilt physical-kernel API; old bindings are not a skip')
    hashes = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in NATIVE_INPUTS}
    hashes['python/vibeqc/aux_basis.py'] = hashlib.sha256((root/'python/vibeqc/aux_basis.py').read_bytes()).hexdigest()
    hashes['python/vibeqc/periodic_fitted_support.py'] = hashlib.sha256(
        (root/'python/vibeqc/periodic_fitted_support.py').read_bytes()).hexdigest()
    hashes['tests/test_chi_hybrid_native.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    from vibeqc import aux_basis, periodic_fitted_support, periodic_fitted_exchange
    from vibeqc.periodic.chi import xc
    for module, relative in ((aux_basis, 'aux_basis.py'),
                             (periodic_fitted_support, 'periodic_fitted_support.py'),
                             (periodic_fitted_exchange, 'periodic_fitted_exchange.py'),
                             (xc, 'periodic/chi/xc.py')):
        expected = root/'python/vibeqc'/relative
        assert Path(module.__file__).resolve() == expected.resolve(), (
            'native acceptance requires Python sources from this tested checkout')
        hashes['python/vibeqc/'+relative] = hashlib.sha256(expected.read_bytes()).hexdigest()
    with loaded.open('rb') as stream:
        loaded_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
    # No private path is copied into product source; runtime receipts may
    # identify the actually loaded artifact in external test evidence.
    receipt = dict(guard='ordinary suite provenance plus required-input freshness and API capability',
                   loaded_core=str(loaded), loaded_core_mtime_ns=loaded.stat().st_mtime_ns,
                   loaded_core_sha256=loaded_sha256,
                   native_threads=int(core.get_num_threads()),
                   source_sha256=hashes)
    return vq, core, receipt


@pytest.fixture(autouse=True)
def source_receipt(native, record_property):
    record_property('chi_native_guard', json.dumps(native[2], sort_keys=True))


@pytest.fixture(scope='module')
def primitive_case(native):
    vq, core, receipt = native
    lattice = np.array([[3.8, .4, .1], [0., 4.1, .3], [0., 0., 4.3]])
    centers = (np.array([.2, .3, .4]), np.array([1.5, .7, .8]))
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(1, center) for center in centers])
    orbital = tuple(Gaussian(a, (2*a/np.pi)**.75, r) for a, r in zip((.7, 1.1), centers))
    auxiliary = tuple(Gaussian(a, (2*a/np.pi)**.75, r) for a, r in zip((.6, .9), centers))
    def make_basis(gaussians, name):
        return core.BasisSet(system.unit_cell_molecule(), [
            core.ShellInfo(i, 0, True, [g.exponent], [g.coefficient], g.center)
            for i, g in enumerate(gaussians)
        ], name, coefficients_pre_normalized=True)
    basis, aux = make_basis(orbital, 'chi-native-orbital'), make_basis(auxiliary, 'chi-native-auxiliary')
    # Pin the primitive interpretation independently of BasisSet defaults.
    for shells, gaussians in ((basis.shells(), orbital), (aux.shells(), auxiliary)):
        for shell, gaussian in zip(shells, gaussians):
            np.testing.assert_allclose(shell.coefficients, [gaussian.coefficient], rtol=0, atol=1e-15)
    return dict(system=system, basis=basis, aux=aux, lattice=lattice,
                orbital=orbital, auxiliary=auxiliary, receipt=receipt)


def _record(record_property, case, **data):
    record_property('chi_native_receipt', json.dumps(dict(case['receipt'], **data), sort_keys=True))


def _vectors(lattice, q, cutoff):
    """Independent complete reciprocal sphere, with its zero point retained."""
    reciprocal = 2*np.pi*np.linalg.inv(lattice).T
    radius = np.sqrt(2*cutoff)
    bounds = np.ceil((radius+np.linalg.norm(q))*np.linalg.norm(np.linalg.inv(reciprocal), axis=1)).astype(int)+1
    integers = np.asarray(list(product(*(range(-n, n+1) for n in bounds))))
    vectors = integers @ reciprocal.T + q
    return np.ascontiguousarray(vectors[np.sum(vectors**2, axis=1) <= radius**2+1e-12])


def _translations(lattice, radius, gaussians):
    centers = np.asarray([g.center for g in gaussians])
    reach = radius+np.linalg.norm(np.ptp(centers, axis=0))
    bounds = np.ceil(reach*np.linalg.norm(np.linalg.inv(lattice), axis=1)).astype(int)+1
    integers = np.asarray(list(product(*(range(-n, n+1) for n in bounds))))
    return integers @ lattice.T


def _charge_fourier(gaussians, vectors):
    p2 = np.sum(vectors**2, axis=1)
    return np.asarray([g.coefficient*(np.pi/g.exponent)**1.5
        *np.exp(-p2/(4*g.exponent)-1j*(vectors @ g.center)) for g in gaussians])


def _pair_fourier(case, vectors, ket, pair_cutoff):
    orbital = case['orbital']
    translations = _translations(case['lattice'], pair_cutoff, orbital)
    out = np.zeros((len(orbital), len(orbital), len(vectors)), dtype=complex)
    p2 = np.sum(vectors**2, axis=1)
    for mu, a in enumerate(orbital):
        for nu, b in enumerate(orbital):
            p = a.exponent+b.exponent
            for r in translations:
                end = b.center+r
                d2 = np.sum((a.center-end)**2)
                if d2 > pair_cutoff**2:
                    continue
                center = (a.exponent*a.center+b.exponent*end)/p
                out[mu, nu] += a.coefficient*b.coefficient*(np.pi/p)**1.5*np.exp(
                    -a.exponent*b.exponent/p*d2-p2/(4*p)-1j*(vectors @ center)+1j*(ket @ r))
    return out


def _two_charge_sr(a, b, distances_squared, omega):
    rho = a*b/(a+b)
    argument = rho*distances_squared
    t = omega/np.sqrt(omega**2+rho)
    return 2*np.pi**2.5/(a*b*np.sqrt(a+b)) * (
        hyp1f1(.5, 1.5, -argument)-t*hyp1f1(.5, 1.5, -t*t*argument))


def _real_sr(case, q, kets, split, pair_cutoff, auxiliary_cutoff):
    """Literal finite metric ball and three-center capsule sums."""
    orbital, auxiliary = case['orbital'], case['auxiliary']
    translations = _translations(case['lattice'], pair_cutoff+auxiliary_cutoff, orbital+auxiliary)
    metric = np.zeros((len(auxiliary), len(auxiliary)), complex)
    tensor = np.zeros((len(kets), len(auxiliary), len(orbital), len(orbital)), complex)
    for ip, p in enumerate(auxiliary):
        for iq, s in enumerate(auxiliary):
            d2 = np.sum((p.center-s.center-translations)**2, axis=1)
            keep = d2 <= auxiliary_cutoff**2
            metric[ip, iq] = p.coefficient*s.coefficient*np.sum(
                np.exp(1j*(translations[keep] @ q))
                *_two_charge_sr(p.exponent, s.exponent, d2[keep], split))
        auxiliary_centers = p.center+translations
        for mu, a in enumerate(orbital):
            for nu, b in enumerate(orbital):
                exponent = a.exponent+b.exponent
                for r in translations:
                    end = b.center+r
                    segment = end-a.center
                    d2 = segment @ segment
                    if d2 > pair_cutoff**2:
                        continue
                    t = np.clip((auxiliary_centers-a.center) @ segment/d2, 0, 1) if d2 else np.zeros(len(translations))
                    keep = np.sum((auxiliary_centers-a.center-t[:, None]*segment)**2, axis=1) <= auxiliary_cutoff**2
                    center = (a.exponent*a.center+b.exponent*end)/exponent
                    distance2 = np.sum((auxiliary_centers[keep]-center)**2, axis=1)
                    radial = (p.coefficient*a.coefficient*b.coefficient
                        *np.exp(-a.exponent*b.exponent/exponent*d2)
                        *_two_charge_sr(p.exponent, exponent, distance2, split))
                    value = np.sum(np.exp(-1j*(translations[keep] @ q))*radial)
                    tensor[:, ip, mu, nu] += np.exp(1j*(kets @ r))*value
    return metric, tensor


def _reciprocal_raw(case, vectors, kets, pair_cutoff, weights):
    auxiliary_ft = _charge_fourier(case['auxiliary'], vectors)
    metric = (auxiliary_ft.conj()*weights) @ auxiliary_ft.T
    tensors = [np.einsum('pg,mng,g->pmn', auxiliary_ft.conj(),
                        _pair_fourier(case, vectors, ket, pair_cutoff), weights) for ket in kets]
    return metric, np.asarray(tensors)


def _finite_source_oracle(case, q, kets, vectors, kernel, omega, split, pair_cutoff, auxiliary_cutoff):
    volume = abs(np.linalg.det(case['lattice']))
    p2 = np.sum(vectors**2, axis=1)
    weights = np.zeros(len(vectors))
    nonzero = p2 > 1e-24
    base = 4*np.pi/(volume*p2[nonzero])
    if kernel == 'erf':
        weights[nonzero] = base*np.exp(-p2[nonzero]/(4*omega**2))
        return _reciprocal_raw(case, vectors, kets, pair_cutoff, weights)
    weights[nonzero] = base*np.exp(-p2[nonzero]/(4*split**2))
    if kernel == 'erfc':
        weights[nonzero] -= base*np.exp(-p2[nonzero]/(4*omega**2))
    metric, tensor = _real_sr(case, q, kets, split, pair_cutoff, auxiliary_cutoff)
    rm, rt = _reciprocal_raw(case, vectors, kets, pair_cutoff, weights)
    metric += rm
    tensor += rt
    fractional = case['lattice'].T @ q/(2*np.pi)
    if np.linalg.norm(fractional-np.rint(fractional)) < 1e-12:
        charges = _charge_fourier(case['auxiliary'], np.zeros((1, 3)))[:, 0]
        g0 = np.pi/(volume*split**2)
        metric -= g0*np.outer(charges.conj(), charges)
        for i, ket in enumerate(kets):
            overlap = _pair_fourier(case, np.zeros((1, 3)), ket, pair_cutoff)[:, :, 0]
            tensor[i] -= g0*charges.conj()[:, None, None]*overlap
    return metric, tensor


def _native_raw(native, case, q, kets, vectors, kernel, omega, *, split=SPLIT, pair=5., auxiliary=6.):
    return native[1].compute_gdf_range_separated_integrals(
        case['basis'], case['aux'], case['system'], q, kets, vectors,
        split, pair, auxiliary, 2**20, 64*2**20, 1000000,
        integral_screen_error=0., kernel=kernel, omega_screen=omega)


@pytest.mark.parametrize('qfrac', [(0., 0., 0.), (.23, -.17, .11)], ids=['gamma', 'complex-q'])
@pytest.mark.parametrize('kernel', ['coulomb', 'erfc', 'erf'])
def test_native_physical_raw_integrals_match_independent_finite_oracle(native, primitive_case, record_property, qfrac, kernel):
    case = primitive_case
    q = 2*np.pi*np.linalg.inv(case['lattice']).T @ qfrac
    kets = np.array([[.23, .13, -.07], [-.11, .19, .17]])
    vectors = _vectors(case['lattice'], q, 12.)
    omega = 0. if kernel == 'coulomb' else OMEGA
    # Exercise both signs of the erfc reciprocal difference, eta < omega
    # and eta > omega. Finite raw sources need not themselves be converged.
    split = .35 if qfrac == (0., 0., 0.) else SPLIT
    metric, tensor, count = _native_raw(native, case, q, kets, vectors, kernel, omega, split=split)
    expected_m, expected_t = _finite_source_oracle(case, q, kets, vectors, kernel, omega, split, 5., 6.)
    residuals = dict(metric_max=float(np.max(abs(metric-expected_m))), tensor_max=float(np.max(abs(tensor-expected_t))))
    _record(record_property, case, kernel=kernel, omega=omega, split=split, qfrac=qfrac,
            cutoffs=dict(pair=5., auxiliary=6., reciprocal=12.), residuals=residuals,
            raw_atol=RAW_ATOL, raw_rtol=RAW_RTOL)
    assert count == np.count_nonzero(np.sum(vectors**2, axis=1) > 1e-24)
    np.testing.assert_allclose(metric, expected_m, atol=RAW_ATOL, rtol=RAW_RTOL)
    np.testing.assert_allclose(tensor, expected_t, atol=RAW_ATOL, rtol=RAW_RTOL)
    if qfrac != (0., 0., 0.):
        assert np.max(abs(expected_m.imag))+np.max(abs(expected_t.imag)) > 1e-3


@pytest.mark.parametrize('qfrac', [(0., 0., 0.), (.23, -.17, .11)], ids=['gamma', 'complex-q'])
def test_native_full_equals_primed_sr_plus_lr_before_fitting(native, primitive_case, record_property, qfrac):
    case = primitive_case
    q = 2*np.pi*np.linalg.inv(case['lattice']).T @ qfrac
    kets = np.array([[.23, .13, -.07]])
    vectors = _vectors(case['lattice'], q, 12.)
    raw = {kernel: _native_raw(native, case, q, kets, vectors, kernel, 0. if kernel == 'coulomb' else OMEGA)
           for kernel in ('coulomb', 'erfc', 'erf')}
    residuals = []
    for index in (0, 1):
        error = raw['coulomb'][index]-raw['erfc'][index]-raw['erf'][index]
        residuals.append(float(np.max(abs(error))))
        np.testing.assert_allclose(error, 0., atol=RAW_ATOL, rtol=0)
    _record(record_property, case, qfrac=qfrac, omega=OMEGA, split=SPLIT, raw_sum_residuals=residuals)


def _physical_oracle_gram(case, q, ket, kernel, omega, pair_cutoff=10., cutoff=64.):
    vectors = _vectors(case['lattice'], q, cutoff)
    p2 = np.sum(vectors**2, axis=1)
    weights = np.zeros(len(vectors))
    nonzero = p2 > 1e-24
    weights[nonzero] = 4*np.pi/(abs(np.linalg.det(case['lattice']))*p2[nonzero])
    if kernel != 'coulomb':
        x = p2[nonzero]/(4*omega**2)
        weights[nonzero] *= 1-np.exp(-x) if kernel == 'erfc' else np.exp(-x)
    metric, tensor = _reciprocal_raw(case, vectors, np.asarray([ket]), pair_cutoff, weights)
    assert np.linalg.eigvalsh(metric)[0] > METRIC_THRESHOLD
    t = tensor[0].reshape(len(case['auxiliary']), -1)
    # Return the AO-pair Gram orientation used by K, without a whitener.
    return (t.conj().T @ np.linalg.solve(metric, t)).conj()


def _fitted_leaf(case, source, q, ket, kernel, omega, cutoff, auxiliary_cutoff):
    from vibeqc.aux_basis import _build_lpq_range_separated_shared_q, build_lpq_bloch_native_fft_shared_q
    from vibeqc import _vibeqc_core as core
    records = []
    common = dict(ke_cutoff=cutoff, linear_dep_thr=METRIC_THRESHOLD,
                  kernel=kernel, omega_screen=omega, canonical_auxiliary_basis=True,
                  _support_recorder=records.append)
    if source == 'sr':
        result = _build_lpq_range_separated_shared_q(
            case['system'], case['basis'], case['aux'], [ket-q], q,
            omega=SPLIT, pair_cutoff=10., auxiliary_cutoff=auxiliary_cutoff,
            memory_byte_cap=512*2**20, native_workspace_byte_cap=64*2**20,
            image_candidate_cap=1000000, reciprocal_candidate_cap=1000000, **common)
        factor = result.factors[0]
    else:
        options = core.LatticeSumOptions()
        options.cutoff_bohr = 10.
        factor = build_lpq_bloch_native_fft_shared_q(
            case['system'], case['basis'], case['aux'], [ket-q], q,
            lat_opts=options, support_policy='physical_pairs', tail_chunk_g=128,
            memory_byte_cap=512*2**20, image_candidate_cap=100000,
            reciprocal_candidate_cap=1000000, **common)[0]
    return factor, json.loads(records[0])


@pytest.mark.parametrize('qfrac', [(0., 0., 0.), (.23, -.17, .11)], ids=['gamma', 'complex-q'])
@pytest.mark.parametrize('kernel', ['coulomb', 'erfc', 'erf'])
def test_classic_and_sr_sources_converge_independently_to_same_physical_fit(native, primitive_case, record_property, qfrac, kernel):
    case = primitive_case
    q = 2*np.pi*np.linalg.inv(case['lattice']).T @ qfrac
    ket = np.array([.23, .13, -.07])
    omega = 0. if kernel == 'coulomb' else OMEGA
    expected = _physical_oracle_gram(case, q, ket, kernel, omega)
    tighter = _physical_oracle_gram(case, q, ket, kernel, omega, cutoff=80.)
    oracle_delta = float(np.max(abs(tighter-expected)))
    np.testing.assert_allclose(expected, tighter, atol=ORACLE_CONVERGENCE_ATOL, rtol=0)
    errors, high_grams, ranks = {}, {}, {}
    for source in ('classic', 'sr'):
        errors[source], ranks[source] = [], []
        for cutoff, aux_cutoff in ((12., 10.), (40., 18.)):
            factor, record = _fitted_leaf(case, source, q, ket, kernel, omega, cutoff, aux_cutoff)
            flat = factor.reshape(factor.shape[0], -1)
            gram = flat.T @ flat.conj()
            errors[source].append(float(np.max(abs(gram-expected))))
            ranks[source].append(int(np.count_nonzero(record['metric_keep'])))
        high_grams[source] = gram
        assert ranks[source] == [2, 2]
        assert errors[source][1] <= max(1.2*errors[source][0], GRAM_ATOL)
        np.testing.assert_allclose(gram, expected, atol=GRAM_ATOL, rtol=0)
    _record(record_property, case, kernel=kernel, omega=omega, qfrac=qfrac,
            source_errors=errors, ranks=ranks, oracle_cutoffs=[64., 80.],
            oracle_refinement_delta=oracle_delta, gram_atol=GRAM_ATOL)
    np.testing.assert_allclose(high_grams['classic'], high_grams['sr'], atol=2*GRAM_ATOL, rtol=0)


def _ewald_scalar(case, eta):
    """Independent point-charge/background Ewald scalar, McClain et al. 2017.

    JCTC 13, 1209, doi:10.1021/acs.jctc.6b01184. Fixed generous spheres,
    a second eta, and the standard self/background terms qualify this scalar.
    """
    lattice = case['lattice']
    volume = abs(np.linalg.det(lattice))
    translations = _translations(lattice, 14., case['orbital'])
    radii = np.linalg.norm(translations, axis=1)
    r = radii[(radii > 0) & (radii <= 14.)]
    vectors = _vectors(lattice, np.zeros(3), 50.)
    p2 = np.sum(vectors**2, axis=1)
    p2 = p2[p2 > 0]
    return (2*eta/np.sqrt(np.pi)+np.pi/(volume*eta**2)
            -np.sum(erfc(eta*r)/r)-np.sum(4*np.pi*np.exp(-p2/(4*eta**2))/(volume*p2)))


@pytest.mark.parametrize('source', ['classic', 'sr'])
@pytest.mark.parametrize('kernel', ['coulomb', 'erfc', 'cam'])
def test_native_factors_fixed_density_matrices_zero_mode_fd_and_spin_collapse(native, primitive_case, record_property, source, kernel):
    from vibeqc.periodic_fitted_support import ChiCoulombControls, build_chi_fitted_operator
    from vibeqc.periodic_fitted_exchange import ChiFittedExchange
    from vibeqc.periodic.chi.xc import ChiExchangeAssembly
    from vibeqc.periodic_k_gdf import _madelung_for_kmesh
    case = primitive_case
    q = np.zeros(3)
    omega = 0. if kernel == 'coulomb' else OMEGA
    physical_kernel = 'erfc' if kernel == 'cam' else kernel
    c_full, c_sr = {'coulomb':(1., 0.), 'erfc':(0., 1.), 'cam':(1., -.65)}[kernel]
    extra = {} if source == 'classic' else dict(auxiliary_cutoff_bohr=18., omega_bohr_inv=SPLIT)
    controls = ChiCoulombControls(10., 10., 40., 12., 5., .6,
        metric_threshold=METRIC_THRESHOLD, image_candidate_cap=100000,
        reciprocal_candidate_cap=1000000, **extra)
    operator = build_chi_fitted_operator(case['system'], case['basis'], case['aux'],
        [q], [1.], source=source, controls=controls, kernel=physical_kernel, omega_screen=omega)
    full_operator = operator if kernel == 'coulomb' else build_chi_fitted_operator(
        case['system'], case['basis'], case['aux'], [q], [1.], source=source, controls=controls)
    overlap = _pair_fourier(case, np.zeros((1, 3)), q, 10.)[:, :, 0]
    gram = _physical_oracle_gram(case, q, q, physical_kernel, omega).reshape(2, 2, 2, 2)
    full_gram = gram if kernel == 'coulomb' else _physical_oracle_gram(
        case, q, q, 'coulomb', 0.).reshape(2, 2, 2, 2)
    xi = _madelung_for_kmesh(case['system'], (1, 1, 1))
    xi_oracle = _ewald_scalar(case, .65)
    np.testing.assert_allclose(xi_oracle, _ewald_scalar(case, .8), atol=2e-12, rtol=0)
    np.testing.assert_allclose(xi, xi_oracle, atol=2e-11, rtol=0)
    finite = np.pi/(abs(np.linalg.det(case['lattice']))*OMEGA**2)
    constant = c_full*xi_oracle+c_sr*finite
    gram = c_full*full_gram + (c_sr*gram if c_sr else 0.)
    assembly = ChiExchangeAssembly(c_full, c_sr, omega)
    exchange = ChiFittedExchange(full_operator, None if kernel == 'coulomb' else operator,
        assembly, [overlap], [1.], abs(np.linalg.det(case['lattice'])), xi)
    density = np.array([[.7, .08+.04j], [.08-.04j, .35]])
    beta = np.array([[.21, -.02+.01j], [-.02-.01j, .17]])
    hcore = np.array([[-1.2, .14+.03j], [.14-.03j, -.6]])
    exchange_coefficient = .37
    direction = np.array([[.13, -.07+.03j], [-.07-.03j, -.09]])
    def actual_k(d):
        return exchange.exchange([d])[0]
    expected_k = np.einsum('mrns,rs->mn', gram, density)+constant*overlap @ density @ overlap
    k = actual_k(density)
    np.testing.assert_allclose(k, expected_k, atol=MATRIX_ATOL, rtol=0)
    np.testing.assert_allclose(k, k.conj().T, atol=2e-12, rtol=0)
    # Finite restoration is exact overlap algebra, not the projected p=0 fit.
    nonzero_k = c_full*full_operator.exchange([density])[0]
    if c_sr:
        nonzero_k += c_sr*operator.exchange([density])[0]
    np.testing.assert_allclose(k-nonzero_k,
        (c_full*xi+c_sr*finite)*overlap @ density @ overlap, atol=2e-13, rtol=0)
    def energy(da, db):
        total = da+db
        j = full_operator.coulomb([total])[0]
        return float(np.trace(total @ hcore).real + .5*np.trace(total @ j).real
            -.5*exchange_coefficient*(np.trace(da @ actual_k(da)).real
                                      +np.trace(db @ actual_k(db)).real))
    total_open = density+beta
    j = full_operator.coulomb([total_open])[0]
    expected_j = np.einsum('mnrs,rs->mn', full_gram, total_open)
    expected_beta_k = np.einsum('mrns,rs->mn', gram, beta)+constant*overlap @ beta @ overlap
    f_alpha = hcore+j-exchange_coefficient*k
    f_beta = hcore+j-exchange_coefficient*actual_k(beta)
    np.testing.assert_allclose(j, expected_j, atol=MATRIX_ATOL, rtol=0)
    np.testing.assert_allclose(f_alpha, hcore+expected_j-exchange_coefficient*expected_k,
                               atol=2*MATRIX_ATOL, rtol=0)
    np.testing.assert_allclose(f_beta, hcore+expected_j-exchange_coefficient*expected_beta_k,
                               atol=2*MATRIX_ATOL, rtol=0)
    derivative = (energy(density+FD_STEP*direction, beta)-energy(density-FD_STEP*direction, beta))/(2*FD_STEP)
    target = float(np.trace(direction @ f_alpha).real)
    np.testing.assert_allclose(derivative, target, atol=FD_ATOL, rtol=0)
    total = 2*density
    closed_j = full_operator.coulomb([total])[0]
    rhf = float(np.trace(total @ hcore).real + .5*np.trace(total @ closed_j).real
                -.25*exchange_coefficient*np.trace(total @ actual_k(total)).real)
    uhf = energy(density, density)
    np.testing.assert_allclose(rhf, uhf, atol=COLLAPSE_ATOL, rtol=0)
    np.testing.assert_allclose(hcore+closed_j-.5*exchange_coefficient*actual_k(total),
        hcore+closed_j-exchange_coefficient*actual_k(density), atol=COLLAPSE_ATOL, rtol=0)
    _record(record_property, case, source=source, kernel=kernel, omega=omega,
        fitted_identity=operator.identity, zero_constant=constant,
        exchange_assembly=exchange.as_dict(), madelung_oracle=xi_oracle,
        matrix_max=float(max(np.max(abs(k-expected_k)), np.max(abs(j-expected_j)))),
        fd_residual=abs(derivative-target),
        collapse_residual=abs(rhf-uhf), matrix_atol=MATRIX_ATOL, fd_atol=FD_ATOL,
        self_consistent_acceptance=False)
