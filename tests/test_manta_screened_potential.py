"""Range-separated nuclear attraction and fixed-range center derivatives."""
import copy
from decimal import Decimal, localcontext

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta
from tests.test_manta_one_electron import _basis, _request, _derivative, _shift_basis


def _potential(bra, ket=None, *, kernel='erfc', omega=.7, charges=None, budget=None):
    request = _request('potential', kernel=kernel, omega=omega,
                       charges=[(1.2, [.3, -.2, .6]), (-.7, [-.4, .1, .2])] if charges is None else charges)
    return manta.compute_one_electron(bra, bra if ket is None else ket, request,
                                      manta.ResourceBudget() if budget is None else budget)


@pytest.mark.parametrize('pure', [False, True])
@pytest.mark.parametrize('omega', [0., 1e-8, .03, .7, 5., 1e5])
def test_range_partition_and_zero_limits(pure, omega):
    a, b = _basis((0, 2), pure), _basis((1, 3), not pure, shift=(.2, -.1, .4))
    full = _potential(a, b, kernel='coulomb', omega=0.)
    long = _potential(a, b, kernel='erf', omega=omega)
    short = _potential(a, b, omega=omega)
    np.testing.assert_allclose(long.values+short.values, full.values, atol=3e-12, rtol=3e-12)
    np.testing.assert_allclose(_potential(b, a, omega=omega).values.T, short.values, atol=3e-12)
    assert short.receipt.kernel == 'erfc' and short.receipt.omega == omega
    assert len({r.receipt.source_identity for r in (full, long, short)}) == 3
    if omega == 0:
        np.testing.assert_array_equal(long.values, 0.)
        np.testing.assert_array_equal(short.values, full.values)


@pytest.mark.parametrize('omega', [1e-12, .2, 1., 100., 1e8, 1e16])
@pytest.mark.parametrize('kernel', ['erf', 'erfc'])
def test_s_tail_against_high_precision_gaussian_integral(kernel, omega):
    # Integrate the convergent scalar Gaussian power series at 90 decimal
    # digits. This neither calls the Boys evaluator nor subtracts FP64
    # full/erf matrices, and requires no optional high-precision package.
    with localcontext() as context:
        context.prec = 90
        d = Decimal.from_float
        pi = Decimal('3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534')
        basis = _basis((0,), contracted=False); s = basis.shells()[0]
        p = 2*d(s.exponents[0]); charge = [.3, -.2, .4]
        t = p*sum((d(a)-d(b))**2 for a, b in zip(s.origin, charge))
        w = d(omega); theta = w/(p+w*w).sqrt()
        lo, hi = (Decimal(0), theta) if kernel == 'erf' else (theta, Decimal(1))
        integral, coefficient = Decimal(0), Decimal(1)
        for n in range(150):
            term = coefficient*(hi**(2*n+1)-lo**(2*n+1))/(2*n+1)
            integral += term
            if abs(term) < abs(integral)*Decimal('1e-80'):
                break
            coefficient *= -t/(n+1)
        else:
            pytest.fail('independent decimal Gaussian series did not converge')
        expected = -d(1.3)*2*pi/p*d(s.coefficients[0])**2*integral
    value = _potential(basis, kernel=kernel, omega=omega, charges=[(1.3, charge)]).values[0, 0]
    assert value == pytest.approx(float(expected), rel=3e-12, abs=1e-300)


@pytest.mark.parametrize('pure', [False, True])
@pytest.mark.parametrize('l', [0, 2, 6])
@pytest.mark.parametrize('omega', [.03, .7, 5.])
def test_erf_against_optional_independent_gaussian_charge_libcint(pure, l, omega):
    gto = pytest.importorskip('pyscf.gto')
    basis = _basis((l,), pure, contracted=False); s = basis.shells()[0]; origin = [.2, .4, -.6]
    mol = gto.M(atom=[('He', s.origin)], basis={'He': [[l, [s.exponents[0], 1.]]]}, unit='Bohr', cart=not pure, verbose=0)
    with mol.with_rinv_origin(origin), mol.with_rinv_zeta(omega**2):
        expected = -1.7*mol.intor('int1e_rinv')
    if not pure and l >= 2: expected *= (2*l+1)/(4*np.pi)
    actual = _potential(basis, kernel='erf', omega=omega, charges=[(1.7, origin)])
    np.testing.assert_allclose(actual.values, expected, atol=3e-11, rtol=3e-11)


@pytest.mark.parametrize('pure', [False, True])
@pytest.mark.parametrize('omega', [.03, .7, 5.])
def test_erfc_against_native_libint_nuclear_lattice_with_one_image(pure, omega):
    basis = _basis(pure=pure)
    atoms = [vq.Atom(2, s.origin) for s in basis.shells()]
    system = vq.PeriodicSystem(3, np.eye(3)*40., atoms)
    opts = vq.LatticeSumOptions(); opts.cutoff_bohr = 4.; opts.nuclear_cutoff_bohr = 4.
    opts.pair_complete_1e = False
    reference = vq.compute_nuclear_erfc_lattice(basis, system, omega, opts)
    assert len(reference.cells) == 1
    expected = np.asarray(reference.blocks[0])
    actual = _potential(basis, omega=omega, charges=[(2., s.origin) for s in basis.shells()])
    np.testing.assert_allclose(actual.values, expected, atol=3e-12, rtol=3e-12)


@pytest.mark.parametrize('kernel', ['erf', 'erfc'])
@pytest.mark.parametrize('pure', [False, True])
def test_fixed_range_basis_charge_derivatives_and_translation_sum(kernel, pure):
    a, b = _basis((0, 2), pure), _basis((1, 3), not pure, shift=(.2, -.1, .4))
    charges = [(1.2, [.3, -.2, .6]), (-.7, [-.4, .1, .2])]
    request = _request('potential', kernel=kernel, omega=.7, charges=charges)
    h = 2e-5
    for axis in range(3):
        total = np.zeros((a.nbasis, b.nbasis))
        for target in ['bra', 'ket', 'charge']:
            for index in (range(len(charges)) if target == 'charge' else [0]):
                derivative = _derivative(target, axis, index)
                result = manta.compute_one_electron_derivative(a, b, request, derivative)
                def value(step):
                    sa, sb, shifted = a, b, copy.deepcopy(charges)
                    if target == 'bra': sa = _shift_basis(a, axis, step)
                    elif target == 'ket': sb = _shift_basis(b, axis, step)
                    else: shifted[index][1][axis] += step
                    return _potential(sa, sb, kernel=kernel, omega=.7, charges=shifted).values
                fd = (-value(2*h)+8*value(h)-8*value(-h)+value(-2*h))/(12*h)
                np.testing.assert_allclose(result.values, fd, atol=3e-10, rtol=3e-9)
                total += result.values
        np.testing.assert_allclose(total, 0., atol=3e-12)


@pytest.mark.parametrize('kernel', ['erf', 'erfc'])
@pytest.mark.parametrize('field, plan_field', [
    ('memory_bytes', 'modeled_peak_bytes'), ('shell_calls', 'shell_calls'),
    ('primitive_quartets', 'primitive_evaluations_upper_bound'), ('kernel_terms', 'kernel_terms'),
])
def test_screened_potential_aggregate_admission(kernel, field, plan_field):
    basis = _basis(); request = _request('potential', kernel=kernel, omega=.7, charges=[(2., [.1, .3, .2])])
    for derivative in (None, _derivative('bra', 2), _derivative('charge', 2)):
        if derivative is None:
            plan = manta.plan_one_electron(basis, basis, request)
            compute = lambda budget: manta.compute_one_electron(basis, basis, request, budget)
        else:
            plan = manta.plan_one_electron_derivative(basis, basis, request, derivative)
            compute = lambda budget: manta.compute_one_electron_derivative(basis, basis, request, derivative, budget)
        budget = manta.ResourceBudget(); setattr(budget, field, getattr(plan, plan_field))
        result = compute(budget)
        assert result.receipt.modeled_peak_bytes == plan.modeled_peak_bytes
        setattr(budget, field, getattr(plan, plan_field)-1)
        with pytest.raises(ValueError, match='budget'):
            compute(budget)


@pytest.mark.parametrize('fields', [
    {'kernel': 'other'}, {'kernel': 'erfc', 'omega': -.1}, {'kernel': 'erf', 'omega': np.inf},
    {'kernel': 'erfc', 'omega': np.nan}, {'kernel': 'coulomb', 'omega': .3},
])
def test_invalid_radial_requests_refused(fields):
    request = _request('potential', **fields)
    with pytest.raises(ValueError):
        manta.compute_one_electron(_basis(), _basis(), request)


@pytest.mark.parametrize('operation', ['overlap', 'kinetic', 'moment'])
def test_nonpotential_operations_refuse_radial_options(operation):
    request = _request(operation, kernel='erf', omega=.3)
    with pytest.raises(ValueError, match='potential'):
        manta.compute_one_electron(_basis(), _basis(), request)


def test_screened_source_identity_tracks_kernel_range_charge_and_signed_zero():
    basis = _basis((0,))
    a, b = _potential(basis, omega=.7), _potential(basis, omega=.8)
    assert a.receipt.source_identity != b.receipt.source_identity
    assert a.receipt.source_identity != _potential(basis, omega=.7, kernel='erf').receipt.source_identity
    assert _potential(basis, omega=0.).receipt.source_identity == _potential(basis, omega=-0.).receipt.source_identity
    assert _potential(basis, charges=[]).values[0, 0] == 0.


@pytest.mark.parametrize('operation, expected', [
    ('overlap', '0d69ecd05340c826fe024e384b62fd541c9e3b77033b0edd9c80945482e03c7e'),
    ('kinetic', '413740d7a9cd13ab2c1febe913141f5ca707fd21a261c729872f2856fcb0857e'),
    ('potential', 'aa8cde71d4582fad8ac4c5e39a7a0e6249c0cc0705abc857f1bad50c664efd5d'),
    ('moment', '4152545db4f73062c3480647faac960e2e954244f66a1d237b674c4332100c67'),
])
def test_existing_full_range_identity_bytes_are_preserved(operation, expected):
    # Fixed synthetic fixture identities captured before radial requests
    # were added. Existing full-Coulomb SCF receipts retain their source.
    basis = _basis((0,))
    request = _request(operation, charges=[(1., [.2, -.3, .1])] if operation == 'potential' else [])
    assert manta.compute_one_electron(basis, basis, request).receipt.source_identity == expected
