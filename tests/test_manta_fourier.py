"""Batched Gaussian Fourier panels with independent integration and admission."""
import gc

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta
from vibeqc import _vibeqc_core as core


def _basis(angular=(0, 1, 2), pure=True, shift=(0., 0., 0.)):
    positions = np.array([[.1, -.2, .3], [.4, .2, -.1], [-.2, .5, .7]]) + shift
    mol = vq.Molecule([vq.Atom(2, p.tolist()) for p in positions], 0, 1)
    shells = []
    for i, l in enumerate(angular):
        shell = core.ShellInfo()
        shell.atom_index, shell.l, shell.pure, shell.origin = i, l, pure, positions[i].tolist()
        shell.exponents, shell.coefficients = [.7+.1*i, 1.6+.2*i], [.8, -.18]
        shells.append(shell)
    return vq.BasisSet(mol, shells, "manta-fourier", False)


def _cart(l):
    return np.array([(x, y, l-x-y) for x in range(l, -1, -1) for y in range(l-x, -1, -1)])


def _quadrature(a, b, vectors):
    # Separable Gauss-Hermite quadrature evaluates Cartesian polynomials
    # directly. No Hermite derivative recurrence or Fourier seed is reused.
    nodes, weights = np.polynomial.hermite.hermgauss(36)
    ca, cb = _cart(a.l), _cart(b.l) if b is not None else np.zeros((1, 3), dtype=int)
    A, B = np.array(a.origin), np.array(b.origin) if b is not None else np.zeros(3)
    result = np.zeros((len(vectors), len(ca), len(cb)), dtype=complex)
    bp = zip(b.exponents, b.coefficients, strict=True) if b is not None else [(0., 1.)]
    bp = list(bp)
    for alpha, ac in zip(a.exponents, a.coefficients, strict=True):
        for beta, bc in bp:
            p = alpha+beta
            center = (alpha*A+beta*B)/p
            value = np.ones_like(result)
            for axis in range(3):
                x = nodes/np.sqrt(p)+center[axis]
                pa = (x[None, :]-A[axis])**ca[:, axis, None]
                pb = (x[None, :]-B[axis])**cb[:, axis, None]
                oscillation = np.exp(-1j*vectors[:, axis, None]*x[None, :])
                value *= np.einsum("in,jn,qn,n->qij", pa, pb, oscillation, weights)/np.sqrt(p)
            result += ac*bc*np.exp(-alpha*beta/p*np.sum((A-B)**2))*value
    # These exact polynomial transforms are independently checked against
    # spherical harmonics through l=7 in the primitive MD suite.
    if a.pure:
        result = np.einsum("ai,qij->qaj", manta.cartesian_to_spherical(a.l), result)
    if b is not None and b.pure:
        result = np.einsum("qij,bj->qib", result, manta.cartesian_to_spherical(b.l))
    return result if b is not None else result[:, :, 0]


VECTORS = np.array([[0., 0., 0.], [.4, -.7, .2], [-.4, .7, -.2], [1.1, .3, -.8]])


@pytest.mark.parametrize("l", [0, 2, 4, 6])
@pytest.mark.parametrize("pure", [False, True])
def test_complete_pair_panels_against_quadrature(l, pure):
    bra, ket = _basis((l,), pure), _basis((l,), pure, (.3, -.4, .7))
    expected = _quadrature(bra.shells()[0], ket.shells()[0], VECTORS)
    actual = manta.compute_fourier_panel(bra, ket, VECTORS).values
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("l", range(8))
@pytest.mark.parametrize("pure", [False, True])
def test_auxiliary_panels_against_quadrature_and_independent_native_backend(l, pure):
    basis = _basis((l,), pure)
    actual = manta.compute_auxiliary_fourier_panel(basis, VECTORS).values
    expected = _quadrature(basis.shells()[0], None, VECTORS)
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)
    if l == 0 or (pure and l <= 6):
        # The older independent backend admits only spherical l<=6 (and s).
        reference = core._periodic_auxiliary_gaussian_fourier_panel(basis, VECTORS, 2**20).T
        np.testing.assert_allclose(actual, reference, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("auxiliary", [False, True])
def test_zero_vector_conjugation_translation_and_panel_batching(auxiliary):
    bra, ket = _basis(), _basis((1, 2), False, (.3, -.4, .7))
    shift = np.array([.4, -.6, .8])
    translated_bra = _basis(shift=shift)
    translated_ket = _basis((1, 2), False, shift+(.3, -.4, .7))
    compute = (lambda a, b, q: manta.compute_auxiliary_fourier_panel(a, q)) if auxiliary else manta.compute_fourier_panel
    result = compute(bra, ket, VECTORS)
    actual = result.values
    np.testing.assert_allclose(actual[1].conj(), actual[2], atol=2e-13)
    phase = np.exp(-1j*VECTORS@shift).reshape((-1,)+(1,)*(actual.ndim-1))
    np.testing.assert_allclose(compute(translated_bra, translated_ket, VECTORS).values,
                               phase*actual, atol=2e-12, rtol=2e-12)
    separate = [compute(bra, ket, VECTORS[i:i+1]) for i in range(len(VECTORS))]
    np.testing.assert_allclose(np.concatenate([r.values for r in separate]), actual, atol=2e-13)
    assert result.receipt.primitive_pairs == separate[0].receipt.primitive_pairs
    assert result.receipt.vector_primitive_products == len(VECTORS)*result.receipt.primitive_pairs
    assert result.receipt.shell_calls == separate[0].receipt.shell_calls
    assert result.receipt.integral_elements == actual.size
    assert result.receipt.convention == "exp(-i*q.r)"
    if not auxiliary:
        np.testing.assert_allclose(actual[0], vq.compute_overlap_two_basis(bra, ket), atol=2e-12)
        reverse = compute(ket, bra, VECTORS).values
        np.testing.assert_allclose(actual, reverse.transpose(0, 2, 1), atol=2e-12)
        assert np.max(np.abs(actual.imag)) > .01


def test_derivative_at_zero_is_negative_imaginary_position_moment():
    basis = _basis()
    h = 1e-5
    for axis in range(3):
        q = np.zeros((2, 3)); q[0, axis] = h; q[1, axis] = -h
        values = manta.compute_fourier_panel(basis, basis, q).values
        request = manta.OneElectronRequest(); request.operation = "moment"
        powers = [0, 0, 0]; powers[axis] = 1; request.powers = powers
        expected = -1j*manta.compute_one_electron(basis, basis, request).values
        np.testing.assert_allclose((values[0]-values[1])/(2*h), expected, atol=2e-10, rtol=2e-10)


def test_auxiliary_s_panel_analytic_decay_across_vector_scales():
    basis = _basis((0,))
    shell = basis.shells()[0]
    q = np.zeros((30, 3)); q[:, 0] = np.geomspace(1e-12, 1e4, 30)
    expected = sum(c*(np.pi/a)**1.5*np.exp(-np.sum(q*q, axis=1)/(4*a))
                   for a, c in zip(shell.exponents, shell.coefficients, strict=True))
    expected = expected*np.exp(-1j*q@np.array(shell.origin))
    np.testing.assert_allclose(manta.compute_auxiliary_fourier_panel(basis, q).values[:, 0],
                               expected, atol=1e-14, rtol=2e-14)


@pytest.mark.parametrize("auxiliary", [False, True])
def test_input_layout_source_identity_and_owned_exports(auxiliary):
    basis = _basis()
    compute = (lambda q: manta.compute_auxiliary_fourier_panel(basis, q)) if auxiliary else (lambda q: manta.compute_fourier_panel(basis, basis, q))
    result = compute(VECTORS)
    same = compute(np.asfortranarray(VECTORS))
    np.testing.assert_array_equal(result.values, same.values)
    assert result.receipt.source_identity == same.receipt.source_identity
    signed_zero = VECTORS.copy(); signed_zero[0] = -0.
    assert result.receipt.source_identity == compute(signed_zero).receipt.source_identity
    assert result.receipt.source_identity != compute(VECTORS[::-1].copy()).receipt.source_identity
    # Misalignment is legal ndarray storage, handled by byte copies.
    unaligned = np.ndarray(VECTORS.shape, dtype=np.float64, buffer=bytearray(VECTORS.nbytes+1), offset=1)
    unaligned[:] = VECTORS
    np.testing.assert_array_equal(result.values, compute(unaligned).values)
    values, before = result.values, result.values.copy()
    assert values.flags.owndata and values.flags.c_contiguous and values.dtype == np.complex128
    del result; gc.collect()
    compute(VECTORS*2)
    np.testing.assert_array_equal(values, before)
    exported = same.values; exported[:] = 0
    np.testing.assert_array_equal(same.values, before)


@pytest.mark.parametrize("auxiliary", [False, True])
@pytest.mark.parametrize("field,plan_field", [
    ("memory_bytes", "modeled_peak_bytes"), ("shell_calls", "shell_calls"),
    ("primitive_quartets", "primitive_pairs"), ("kernel_terms", "kernel_terms"),
    ("scalar_terms", None),
])
def test_exact_resource_boundaries(auxiliary, field, plan_field):
    basis = _basis()
    plan = manta.plan_auxiliary_fourier(basis, len(VECTORS)) if auxiliary else manta.plan_fourier(basis, basis, len(VECTORS))
    compute = (lambda budget: manta.compute_auxiliary_fourier_panel(basis, VECTORS, budget)) if auxiliary else (lambda budget: manta.compute_fourier_panel(basis, basis, VECTORS, budget))
    limit = getattr(plan, plan_field) if plan_field else np.prod(plan.shape).item()
    budget = manta.ResourceBudget(); setattr(budget, field, limit)
    result = compute(budget)
    assert result.receipt.modeled_peak_bytes == plan.modeled_peak_bytes
    assert result.receipt.shell_calls == plan.shell_calls
    assert result.receipt.primitive_pairs == plan.primitive_pairs
    setattr(budget, field, limit-1)
    with pytest.raises(ValueError, match="budget"):
        compute(budget)


@pytest.mark.parametrize("vectors", [
    VECTORS.astype(np.float32), VECTORS.astype(complex), np.zeros((3, 2)),
    np.zeros(3), np.zeros((2, 2, 3)), np.zeros((8, 3))[::2],
    np.full((2, 3), np.nan), np.full((2, 3), np.inf),
])
def test_invalid_vectors_refused(vectors):
    basis = _basis((0,))
    with pytest.raises(ValueError):
        manta.compute_fourier_panel(basis, basis, vectors)
    with pytest.raises(ValueError):
        manta.compute_auxiliary_fourier_panel(basis, vectors)


def test_budget_refuses_before_vector_payload_scan_and_large_count_overflow():
    basis = _basis((0,))
    budget = manta.ResourceBudget(); budget.memory_bytes = 1
    with pytest.raises(ValueError, match="budget"):
        manta.compute_fourier_panel(basis, basis, np.full((2, 3), np.nan), budget)
    with pytest.raises((ValueError, OverflowError)):
        manta.plan_fourier(basis, basis, 2**63)
    with pytest.raises((ValueError, OverflowError)):
        manta.plan_auxiliary_fourier(basis, 2**63-1)


@pytest.mark.parametrize("auxiliary", [False, True])
def test_empty_panel_has_no_evaluation_work(auxiliary):
    basis = _basis()
    budget = manta.ResourceBudget()
    budget.shell_calls = budget.primitive_quartets = budget.kernel_terms = budget.scalar_terms = 0
    q = np.empty((0, 3))
    plan = manta.plan_auxiliary_fourier(basis, 0, budget) if auxiliary else manta.plan_fourier(basis, basis, 0, budget)
    result = manta.compute_auxiliary_fourier_panel(basis, q, budget) if auxiliary else manta.compute_fourier_panel(basis, basis, q, budget)
    assert result.values.shape == ((0, basis.nbasis) if auxiliary else (0, basis.nbasis, basis.nbasis))
    assert result.values.size == plan.output_bytes == plan.kernel_terms == plan.tile_elements == 0
    assert result.receipt.shell_calls == result.receipt.primitive_pairs == result.receipt.vector_primitive_products == 0
    assert result.receipt.integral_elements == 0


def test_role_specific_angular_limits():
    auxiliary, orbital = _basis((7,)), _basis((0,))
    manta.compute_auxiliary_fourier_panel(auxiliary, VECTORS)
    for bra, ket in [(auxiliary, orbital), (orbital, auxiliary)]:
        with pytest.raises(ValueError, match="angular"):
            manta.compute_fourier_panel(bra, ket, VECTORS)
    with pytest.raises(ValueError, match="angular"):
        manta.compute_auxiliary_fourier_panel(_basis((8,)), VECTORS)


def _decimal_phase(momentum,translation):
    """Independent 800-digit pi/argument reduction and scalar Taylor oracle."""
    from decimal import Decimal as D,localcontext,ROUND_HALF_EVEN
    # AGM pi iteration is independent of the native binary phase squaring.
    # Reuse its value across calls without adding a numerical dependency.
    if not hasattr(_decimal_phase,'pi'):
        with localcontext() as ctx:
            ctx.prec=850;a=D(1);b=1/D(2).sqrt();t=D(1)/4;p=D(1)
            for _ in range(13):
                mean=(a+b)/2;b=(a*b).sqrt();t-=p*(a-mean)**2;a=mean;p*=2
            _decimal_phase.pi=+(a+b)**2/(4*t)
    with localcontext() as ctx:
        ctx.prec=800
        angle=sum(D(float(a))*D(float(b)) for a,b in zip(momentum,translation))
        two_pi=2*_decimal_phase.pi
        x=angle-(angle/two_pi).to_integral_value(rounding=ROUND_HALF_EVEN)*two_pi
        cosine=D(1);sine=x;ct=D(1);st=x
        for j in range(1,2000):
            ct*=-x*x/D((2*j-1)*(2*j));st*=-x*x/D((2*j)*(2*j+1))
            cosine+=ct;sine+=st
            if max(abs(ct),abs(st))<D('1e-790'):break
        else:raise AssertionError('phase reference did not converge')
        return +cosine,+sine


@pytest.mark.parametrize('momentum,translation',[
    ([0.,0.,0.],[1.,-2.,3.]),([.3,-.7,1.1],[.4,-.2,6.]),
    ([1e16,1.,-1e16],[1.,.1,1.]),([1e100,1e-100,-1e100],[1.,1.,1.]),
    ([1e-200,0.,0.],[1e-200,0.,0.]),([np.nextafter(0.,1.),0.,0.],[1.,0.,0.]),
    ([2.**500,0.,0.],[2.**500,0.,0.]),([1.,0.,0.],[np.pi,0.,0.]),
    ([1.,0.,0.],[np.nextafter(np.pi,0.),0.,0.]),([1.,0.,0.],[np.nextafter(np.pi,np.inf),0.,0.]),
    ([.0625,0.,0.],[1.,0.,0.]),([-.0625,0.,0.],[1.,0.,0.]),
])
def test_bloch_phase_enclosure_independent_transcendental(momentum,translation):
    from decimal import Decimal as D,localcontext
    q=np.asarray(momentum);r=np.asarray(translation);evaluated=np.exp(1j*np.dot(q,r))
    c=manta.certify_bloch_phase(q,r,evaluated);re,im=_decimal_phase(q,r)
    assert D(c.real_lower)<=re<=D(c.real_upper) and D(c.imag_lower)<=im<=D(c.imag_upper)
    with localcontext() as ctx:
        ctx.prec=800;error=abs(D(float(evaluated.real))-re)+abs(D(float(evaluated.imag))-im)
        assert error<=D(c.error_bound)
    assert -1<=c.real_lower<=c.real_upper<=1 and -1<=c.imag_lower<=c.imag_upper<=1
    assert 0<=c.squarings<=1028
    if not np.any(q):assert c.error_bound==0. and c.real_lower==c.real_upper==1.
    # The diagnostic encloses arbitrary returned witnesses, not an assumed
    # correctly-rounded library implementation.
    other=manta.certify_bloch_phase(q,r,evaluated+.03-.02j)
    assert other.error_bound>=.049 or c.error_bound>.001
    with pytest.raises(AttributeError):c.real_lower=0.


@pytest.mark.parametrize('field,quantity',[('memory_bytes','modeled_peak_bytes'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_bloch_phase_budget_and_nonfinite_refusals(field,quantity):
    b=manta.ResourceBudget();p=manta.plan_bloch_phase(b)
    setattr(b,field,getattr(p,quantity)-1)
    with pytest.raises((ValueError,RuntimeError),match='budget'):manta.certify_bloch_phase(np.full(3,np.nan),np.ones(3),1.,b)
    setattr(b,field,getattr(p,quantity));assert manta.certify_bloch_phase(np.zeros(3),np.ones(3),1.,b).error_bound==0.
    with pytest.raises(ValueError,match='finite'):manta.certify_bloch_phase(np.full(3,np.nan),np.ones(3),1.,b)
    with pytest.raises(ValueError,match='finite'):manta.certify_bloch_phase(np.ones(3),np.ones(3),complex(np.inf,0),b)
    with pytest.raises((OverflowError,RuntimeError),match='overflow'):manta.certify_bloch_phase(np.full(3,1e308),np.ones(3),1.,b)


def _decimal_auxiliary_fourier(basis,vectors):
    """Closed finite Gaussian-moment sum, independent of native recurrence."""
    from decimal import Decimal as D,localcontext
    from math import factorial
    _decimal_phase([0.,0.,0.],[0.,0.,0.]);rows=[]
    def mul(a,b):return (a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0])
    with localcontext() as ctx:
        ctx.prec=250
        for wave in vectors:
            q=[D(float(v)) for v in wave];row=[]
            for shell in basis.shells():
                powers=_cart(shell.l);cart=[(D(0),D(0)) for _ in powers]
                phase=_decimal_phase(-np.asarray(wave),shell.origin)
                for a,c in zip(shell.exponents,shell.coefficients):
                    a=D(float(a));c=D(float(c));ratio=_decimal_phase.pi/a
                    seed=ratio*ratio.sqrt()*(-sum(x*x for x in q)/(4*a)).exp()*c
                    for i,power in enumerate(powers):
                        value=(seed*phase[0],seed*phase[1])
                        for axis,n in enumerate(power):
                            re,im=D(0),D(0)
                            for k in range(int(n)//2+1):
                                degree=int(n)-2*k
                                term=D(factorial(int(n)))/D(factorial(k)*factorial(degree))*(1/(4*a))**k*(D(1) if degree==0 else (-q[axis]/(2*a))**degree)
                                if degree%4==0:re+=term
                                elif degree%4==1:im+=term
                                elif degree%4==2:re-=term
                                else:im-=term
                            value=mul(value,(re,im))
                        cart[i]=(cart[i][0]+value[0],cart[i][1]+value[1])
                from tests.test_manta_md import _decimal_representation
                transform=_decimal_representation(shell.l,shell.pure)
                for coefficients in transform:
                    row.append((sum(t*z[0] for t,z in zip(coefficients,cart)),sum(t*z[1] for t,z in zip(coefficients,cart))))
            rows.append(row)
    return rows


def _check_fourier_construction(basis,vectors):
    from decimal import Decimal as D,localcontext
    c=manta.certify_auxiliary_fourier_panel(basis,vectors);actual=c.evaluated.values
    np.testing.assert_array_equal(actual,manta.compute_auxiliary_fourier_panel(basis,vectors).values)
    reference=_decimal_auxiliary_fourier(basis,vectors)
    bounds=[np.asarray(getattr(c,k)).reshape(actual.shape) for k in ('real_lower','real_upper','imag_lower','imag_upper','error_bounds')]
    with localcontext() as ctx:
        ctx.prec=250
        for g,row in enumerate(reference):
            for i,(re,im) in enumerate(row):
                assert D(float(bounds[0][g,i]))<=re<=D(float(bounds[1][g,i]))
                assert D(float(bounds[2][g,i]))<=im<=D(float(bounds[3][g,i]))
                error=abs(D(float(actual[g,i].real))-re)+abs(D(float(actual[g,i].imag))-im)
                assert error<=D(float(bounds[4][g,i]))
    assert np.all(np.isfinite(bounds[4]))
    with pytest.raises(AttributeError):c.error_bounds=[]


@pytest.mark.parametrize('l',range(8))
@pytest.mark.parametrize('pure',[False,True])
@pytest.mark.parametrize('scale',[1e-30,1.,1e30])
def test_auxiliary_fourier_construction_decimal_all_angular(l,pure,scale):
    basis=_basis((l,),pure);shell=basis.shells()[0];shell.coefficients=[x*scale for x in shell.coefficients]
    basis=vq.BasisSet(vq.Molecule([vq.Atom(2,shell.origin)]),[shell],'construction',True)
    _check_fourier_construction(basis,VECTORS)


@pytest.mark.parametrize('kind',['diffuse','tight','cancellation'])
def test_auxiliary_fourier_construction_extremes(kind):
    shell=_basis((3,),True).shells()[0]
    if kind=='cancellation':
        shell.exponents=[.7,.7]*24+[1.6];shell.coefficients=[1.,-1.]*24+[1e-15];vectors=VECTORS
    else:
        exponent=1e-6 if kind=='diffuse' else 1e6
        shell.exponents=[exponent,2*exponent];shell.coefficients=[.8,-.7];vectors=VECTORS*np.sqrt(exponent)
    basis=vq.BasisSet(vq.Molecule([vq.Atom(2,shell.origin)]),[shell],'construction-extreme',True)
    _check_fourier_construction(basis,vectors)


@pytest.mark.parametrize('field,quantity',[('memory_bytes','modeled_peak_bytes'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms'),('shell_calls','shell_calls'),('primitive_quartets','primitive_products')])
def test_auxiliary_fourier_construction_caps(field,quantity):
    basis=_basis((2,),True);b=manta.ResourceBudget();p=manta.plan_auxiliary_fourier_construction(basis,len(VECTORS),b)
    setattr(b,field,getattr(p,quantity)-1)
    with pytest.raises((ValueError,RuntimeError),match='budget'):manta.certify_auxiliary_fourier_panel(basis,np.full_like(VECTORS,np.nan),b)
    setattr(b,field,getattr(p,quantity));assert manta.certify_auxiliary_fourier_panel(basis,VECTORS,b).evaluated.values.shape==(len(VECTORS),basis.nbasis)
    with pytest.raises(ValueError,match='finite'):manta.certify_auxiliary_fourier_panel(basis,np.full_like(VECTORS,np.nan),b)


@pytest.mark.parametrize('exponent',[1e-200,1e200])
def test_auxiliary_fourier_construction_representable_scale_limits(exponent):
    shell=_basis((0,),False).shells()[0];shell.exponents=[exponent];shell.coefficients=[1.]
    basis=vq.BasisSet(vq.Molecule([vq.Atom(2,shell.origin)]),[shell],'construction-scale',True)
    _check_fourier_construction(basis,VECTORS*np.sqrt(exponent))
