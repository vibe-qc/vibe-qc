"""Screened periodic exchange, its finite zero mode and positive small tail."""
from decimal import Decimal, localcontext
import itertools
import math

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta
from tests.test_manta_periodic_exchange import (
    _basis, _compute, _density, _transfers, _quadrature_bloch,
    LATTICE, CELLS, MOMENTA, WEIGHTS, LABELS,
)


def _build(kernel, omega, *, basis=None, density=None, zero=False, batch=4, budget=None):
    basis = _basis((0, 1)) if basis is None else basis
    density = _density(basis) if density is None else density
    q, offsets = _transfers()
    if zero:
        base = LABELS@(2*np.pi*np.linalg.inv(LATTICE))
        q = np.concatenate([base+t-s for t in MOMENTA for s in MOMENTA])
        offsets = np.arange(10, dtype=np.int64)*len(LABELS)
    return manta.compute_periodic_exchange(basis, LATTICE, q, offsets, CELLS, MOMENTA, WEIGHTS,
        density, batch, None, manta.ResourceBudget() if budget is None else budget, kernel, omega)


@pytest.mark.parametrize('kernel,zero', [('erf', False), ('erfc', False), ('erfc', True)])
@pytest.mark.parametrize('omega', [.15, .8, 4.])
def test_screened_response_against_independent_quadrature(kernel, zero, omega):
    basis = _basis((0, 1)); density = _density(basis)
    result = _build(kernel, omega, basis=basis, density=density, zero=zero)
    base = LABELS@(2*np.pi*np.linalg.inv(LATTICE))
    expected = np.zeros_like(density)
    for t, kt in enumerate(MOMENTA):
        for s, ks in enumerate(MOMENTA):
            q = base+kt-ks
            if not zero: q = q[np.any(q != 0., axis=1)]
            q2 = np.sum(q*q, axis=1)
            coefficient = np.empty(len(q))
            for i, norm in enumerate(q2):
                if norm == 0: coefficient[i] = np.pi/omega**2
                elif kernel == 'erf': coefficient[i] = 4*np.pi*np.exp(-norm/(4*omega**2))/norm
                else: coefficient[i] = -4*np.pi*np.expm1(-norm/(4*omega**2))/norm
            b = _quadrature_bloch(basis, q, CELLS, kt, None)
            expected[:, t] += WEIGHTS[s]/abs(np.linalg.det(LATTICE))*np.einsum('g,gai,dab,gbj->dij', coefficient, b.conj(), density[:, s], b)
    np.testing.assert_allclose(result.values, expected, atol=3e-11, rtol=2e-11)
    np.testing.assert_allclose(result.receipt.energies, -.5*np.einsum('k,dkij,dkji->d', WEIGHTS, density, expected).real, atol=3e-11, rtol=2e-11)
    assert result.receipt.kernel == kernel and result.receipt.omega == omega
    assert result.receipt.zero_vectors == (len(MOMENTA) if zero else 0)


def test_full_decomposition_excludes_finite_erfc_zero_mode_and_preserves_default_identity():
    basis = _basis((0, 1)); density = _density(basis); omega = .8
    full = _compute(basis, density)
    assert full.receipt.source_identity == '716f111e6a3aa29380e85d1a6f00975c0a00cca45c7ba4ca7f60051ecf06af22'
    lr = _build('erf', omega, basis=basis, density=density)
    sr = _build('erfc', omega, basis=basis, density=density)
    sr_zero = _build('erfc', omega, basis=basis, density=density, zero=True)
    np.testing.assert_allclose(lr.values+sr.values, full.values, atol=5e-12)
    overlap = manta.compute_bloch_one_electron(basis, 'overlap', CELLS, MOMENTA).values
    correction = np.stack([np.pi/omega**2/abs(np.linalg.det(LATTICE))*WEIGHTS[k]*
        (overlap[k].conj().T@density[:, k]@overlap[k]) for k in range(len(MOMENTA))], axis=1)
    np.testing.assert_allclose(lr.values+sr_zero.values, full.values+correction, atol=5e-12)
    assert len({r.receipt.source_identity for r in (full, lr, sr, sr_zero)}) == 4
    assert 'finite-q0-on-explicit-support' in sr_zero.receipt.boundary


@pytest.mark.parametrize('kernel,zero', [('erf', False), ('erfc', True)])
def test_screened_energy_derivative_and_weighted_self_adjoint_pairing(kernel, zero):
    basis = _basis((0, 1)); density = _density(basis); h = 1e-5
    result = _build(kernel, .7, basis=basis, density=density, zero=zero)
    direction = density[1, 0].copy(); plus = density.copy(); minus = density.copy()
    plus[0, 1] += h*direction; minus[0, 1] -= h*direction
    a = _build(kernel, .7, basis=basis, density=plus, zero=zero)
    b = _build(kernel, .7, basis=basis, density=minus, zero=zero)
    assert (a.receipt.energies[0]-b.receipt.energies[0])/(2*h) == pytest.approx(
        -WEIGHTS[1]*np.trace(direction@result.values[0, 1]).real, abs=1e-8, rel=1e-8)
    left = np.einsum('k,kij,kji->', WEIGHTS, density[0], result.values[1])
    right = np.einsum('k,kij,kji->', WEIGHTS, density[1], result.values[0])
    assert abs(left-right) < 2e-11


def test_zero_range_limits_and_batch_identity():
    basis = _basis((0, 1)); full = _compute(basis)
    sr = _build('erfc', 0., basis=basis)
    lr = _build('erf', 0., basis=basis)
    np.testing.assert_allclose(sr.values, full.values, atol=5e-12)
    np.testing.assert_array_equal(lr.values, np.zeros_like(lr.values))
    assert lr.receipt.matrix_products == 0
    a = _build('erfc', .6, zero=True, batch=1)
    b = _build('erfc', .6, zero=True, batch=32)
    np.testing.assert_allclose(a.values, b.values, atol=3e-12)
    assert a.receipt.source_identity == b.receipt.source_identity
    assert a.receipt.density_identity == b.receipt.density_identity
    assert _build('erfc', .7, zero=True).receipt.source_identity != b.receipt.source_identity


@pytest.mark.parametrize('omega', [1e8, 1e16, 1e100, 1e150])
def test_positive_erfc_tail_against_high_precision_scalar_gaussian(omega):
    from tests.test_manta_one_electron import _basis as scalar_basis
    basis = scalar_basis((0,), contracted=False); exponent = basis.shells()[0].exponents[0]
    q = LABELS@(2*np.pi*np.linalg.inv(LATTICE)); zero = np.zeros((1, 3))
    result = manta.compute_periodic_exchange(basis, LATTICE, q, np.array([0, len(q)], dtype=np.int64),
        zero, zero, np.ones(1), np.ones((1, 1, 1, 1), complex), kernel='erfc', omega=omega)
    with localcontext() as ctx:
        ctx.prec = 360
        pi = Decimal(str(np.pi)); w = Decimal(str(omega)); a = Decimal(str(exponent)); total = Decimal(0)
        for norm in np.sum(q*q, axis=1):
            x = Decimal(str(norm))
            coefficient = pi/w**2 if norm == 0 else 4*pi*(1-(-x/(4*w*w)).exp())/x
            total += coefficient*(-x/(4*a)).exp()
        expected = float(total/Decimal(str(abs(np.linalg.det(LATTICE)))))
    actual = result.values[0, 0, 0, 0]
    assert actual.real > 0 and actual.imag == 0
    assert actual.real == pytest.approx(expected, rel=3e-13, abs=0.)


def test_erfc_periodic_fourier_sum_matches_independent_real_space_gaussian_sum():
    # Poisson identity for spherical normalized rho(r)=(2a/pi)^1.5 exp(-2a*r^2).
    # Its pair interaction is [erf(sqrt(a)R)-erf(beta R)]/R;
    # beta=(1/a+1/omega^2)^-1/2. No MANTA/libint integral kernel enters this reference.
    from tests.test_manta_one_electron import _basis as scalar_basis
    basis = scalar_basis((0,), contracted=False); a = basis.shells()[0].exponents[0]; omega = .6
    length = 3.; lattice = np.eye(3)*length; zero = np.zeros((1, 3))
    labels = np.array(list(itertools.product(range(-7, 8), repeat=3)), dtype=float)
    q = np.ascontiguousarray(labels*(2*np.pi/length))
    result = manta.compute_periodic_exchange(basis, lattice, q, np.array([0, len(q)], dtype=np.int64),
        zero, zero, np.ones(1), np.ones((1, 1, 1, 1), complex), kernel='erfc', omega=omega)
    beta = 1/np.sqrt(1/a+1/omega**2)
    expected = 2/np.sqrt(np.pi)*(np.sqrt(a)-beta)
    for label in itertools.product(range(-5, 6), repeat=3):
        r = length*np.linalg.norm(label)
        if r: expected += (math.erfc(beta*r)-math.erfc(np.sqrt(a)*r))/r
    assert result.values[0, 0, 0, 0] == pytest.approx(expected, abs=2e-13, rel=2e-12)


@pytest.mark.parametrize('kernel,omega', [('coulomb', 1.), ('bad', 0.), ('erf', -1.), ('erfc', np.inf), ('erf', np.nan)])
def test_invalid_ranges_fail_count_plan_and_before_numerical_density_reads(kernel, omega):
    basis = _basis((0, 1)); density = _density(basis); density[:] = np.nan
    with pytest.raises(ValueError, match='kernel'):
        manta.plan_periodic_exchange(basis, 7, len(CELLS), len(MOMENTA), 2, kernel=kernel, omega=omega)
    with pytest.raises(ValueError, match='kernel'):
        _build(kernel, omega, basis=basis, density=density)


@pytest.mark.parametrize('kernel,omega', [('coulomb', 0.), ('erf', .7), ('erf', 0.), ('erfc', 0.)])
def test_singular_or_omitted_zero_modes_are_never_implicitly_changed(kernel, omega):
    with pytest.raises(ValueError, match='Q=0'):
        _build(kernel, omega, zero=True)


def test_selected_screened_adapter_reports_actual_kernel_and_zero_mode(monkeypatch):
    from vibeqc import output
    basis = _basis((0, 1)); density = _density(basis); q, offsets = _transfers()
    records = []; monkeypatch.setattr(output, 'record', lambda *args, **kwargs: records.append(kwargs))
    result = manta.build_periodic_exchange(basis, LATTICE, q, offsets, CELLS, MOMENTA, WEIGHTS,
        density, kernel='erfc', omega=.6, integral_engine=vq.IntegralEngineOptions(backend='manta'))
    r = records[0]['integral_provenance']
    assert r['kernel'] == 'erfc' and r['omega'] == .6 and r['zero_vectors'] == 0
    assert r['source_identity'] == result.receipt.source_identity
    assert r['boundary'] == 'erfc-finite-q0-on-explicit-support-no-madelung'


@pytest.mark.parametrize('field,plan_field', [('memory_bytes','modeled_peak_bytes'), ('shell_calls','shell_calls'),
    ('primitive_quartets','primitive_pairs'), ('scalar_terms','scalar_terms'), ('kernel_terms','kernel_terms')])
def test_screened_zero_mode_aggregate_budgets_are_checked_before_density_snapshot(field, plan_field):
    basis = _basis((0, 1)); density = _density(basis)
    plan = manta.plan_periodic_exchange(basis, len(LABELS), len(CELLS), len(MOMENTA), len(density),
        4, kernel='erfc', omega=.6)
    budget = manta.ResourceBudget(); setattr(budget, field, getattr(plan, plan_field))
    _build('erfc', .6, basis=basis, density=density, zero=True, budget=budget)
    setattr(budget, field, getattr(plan, plan_field)-1); density[:] = np.nan
    with pytest.raises((ValueError, RuntimeError), match='budget'):
        _build('erfc', .6, basis=basis, density=density, zero=True, budget=budget)
