"""Streamed MD Coulomb metric and three-center tiles against independent references."""
from itertools import product
from math import pi

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta
from vibeqc._vibeqc_core import ShellInfo


def _basis(angular=(0, 1, 2), pure=True, contracted=True, shift=(0., 0., 0.)):
    positions = np.array([[0., .1, -.2], [.2, -.3, 1.1], [-.4, .5, .7]])+shift
    mol = vq.Molecule([vq.Atom(2, p.tolist()) for p in positions], 0, 1)
    shells = []
    for i, l in enumerate(angular):
        shell = ShellInfo()
        shell.atom_index, shell.l, shell.pure, shell.origin = i, l, pure, positions[i].tolist()
        shell.exponents = [.7+.13*i, 1.6+.1*i] if contracted else [.7+.13*i]
        shell.coefficients = [.8, -.18] if contracted else [1.]
        shells.append(shell)
    return vq.BasisSet(mol, shells, "manta-fitting", False)


def _slices(basis):
    offsets = [0]
    for shell in basis.shells():
        offsets.append(offsets[-1]+(2*shell.l+1 if shell.pure else (shell.l+1)*(shell.l+2)//2))
    return [slice(a, b) for a, b in zip(offsets[:-1], offsets[1:], strict=True)]


@pytest.mark.parametrize("pure", [False, True])
@pytest.mark.parametrize("arity", [2, 3])
def test_complete_contracted_fitting_tiles_against_libint(pure, arity):
    aux, orbital = _basis((0, 1, 3), pure), _basis((0, 1, 2), pure, shift=(.1, -.2, .3))
    axes = [aux, aux] if arity == 2 else [aux, orbital, orbital]
    provider = manta.FittingTiles(axes)
    expected = vq.compute_2c_eri(aux) if arity == 2 else vq.compute_3c_eri(orbital, aux)
    actual = np.empty_like(expected)
    slices = [_slices(basis) for basis in axes]
    calls = primitives = 0
    saved = None
    for indices in product(*(range(b.nshells) for b in axes)):
        result = provider.evaluate(indices)
        assert result["provider"] == "manta" and result["null_tiles"] == 0
        assert result["source_identity"] == provider.source_identity
        assert result["normalization"] == "libint-shell-axial-v1"
        actual[tuple(axis[i] for axis, i in zip(slices, indices, strict=True))] = result["values"]
        calls += result["shell_calls"]
        primitives += result["primitive_products_requested"]
        if saved is None:
            saved = result["values"], result["values"].copy()
    assert calls == provider.plan.shell_calls
    assert primitives == provider.plan.primitive_products
    assert actual.size == provider.plan.integral_elements
    np.testing.assert_allclose(actual, expected, atol=3e-11, rtol=5e-12)
    del provider
    np.testing.assert_array_equal(*saved)
    if arity == 2:
        np.testing.assert_allclose(actual, actual.T, atol=3e-12)
        assert np.linalg.eigvalsh(actual).min() > 0
    else:
        np.testing.assert_allclose(actual, actual.swapaxes(1, 2), atol=3e-12)


def _libcint_reference(bases, pure):
    gto = pytest.importorskip("pyscf.gto")
    shells = [b.shells()[0] for b in bases]
    mol = gto.M(atom=[(f"He{i}", s.origin) for i, s in enumerate(shells)],
                basis={f"He{i}": [[s.l, [s.exponents[0], 1.]]] for i, s in enumerate(shells)},
                unit="Bohr", cart=not pure, verbose=0)
    name = "int2c2e" if len(bases) == 2 else "int3c2e"
    # libcint places the orbital pair before the auxiliary center in 3c.
    indices = (0, 1) if len(bases) == 2 else (1, 2, 0)
    reference = mol.intor_by_shell(name+('_sph' if pure else '_cart'), indices)
    if len(bases) == 3:
        reference = reference.transpose(2, 0, 1)
    for axis, shell in enumerate(shells):
        if pure and shell.l == 1:
            reference = np.take(reference, [1, 2, 0], axis=axis)
        elif not pure and shell.l >= 2:
            reference *= np.sqrt((2*shell.l+1)/(4*pi))
    return reference


@pytest.mark.parametrize("angular", [(7, 7), (7, 0), (7, 6, 6), (7, 1, 5), (3, 6, 2)])
@pytest.mark.parametrize("pure", [False, True])
def test_complete_high_angular_fitting_tiles_against_optional_libcint(angular, pure):
    bases = [_basis((l,), pure, False, shift=(.1*i, -.2*i, .3*i)) for i, l in enumerate(angular)]
    reference = _libcint_reference(bases, pure)
    provider = manta.FittingTiles(bases)
    actual = provider.evaluate((0,)*len(bases))["values"]
    np.testing.assert_allclose(actual, reference, atol=4e-11, rtol=2e-10)


@pytest.mark.parametrize("arity", [2, 3])
def test_s_fitting_against_analytic_formula(arity):
    bases = [_basis((0,), False, False, shift=(.3*i, -.2*i, .1*i)) for i in range(arity)]
    shells = [b.shells()[0] for b in bases]
    p = shells[0].exponents[0]
    a = np.array(shells[0].origin)
    if arity == 2:
        q = shells[1].exponents[0]
        b = np.array(shells[1].origin)
        prefactor = 1.
    else:
        beta, gamma = shells[1].exponents[0], shells[2].exponents[0]
        q = beta+gamma
        b = (beta*np.array(shells[1].origin)+gamma*np.array(shells[2].origin))/q
        prefactor = np.exp(-beta*gamma/q*np.sum((np.array(shells[1].origin)-shells[2].origin)**2))
    # Independent numerical integral definition F0(t)=integral_0^1 exp(-t*x*x) dx.
    nodes, weights = np.polynomial.legendre.leggauss(48)
    t = p*q/(p+q)*np.sum((a-b)**2)
    f0 = .5*np.dot(weights, np.exp(-t*((nodes+1)*.5)**2))
    reference = prefactor*2*pi**2.5/(p*q*np.sqrt(p+q))*f0*np.prod([s.coefficients[0] for s in shells])
    actual = manta.FittingTiles(bases).evaluate((0,)*arity)["values"].item()
    assert actual == pytest.approx(reference, rel=2e-13)


@pytest.mark.parametrize("arity", [2, 3])
def test_distinct_axes_covariance_and_source_identity(arity):
    bases = [_basis((l,), pure=False, contracted=True, shift=(.3*i, -.2*i, .1*i)) for i, l in enumerate([2, 1, 3][:arity])]
    provider = manta.FittingTiles(bases)
    permutation = [1, 0] if arity == 2 else [0, 2, 1]
    reverse = manta.FittingTiles([bases[i] for i in permutation])
    actual = provider.evaluate((0,)*arity)["values"]
    np.testing.assert_allclose(actual, reverse.evaluate((0,)*arity)["values"].transpose(permutation), atol=3e-12)
    assert provider.source_identity != reverse.source_identity
    translated = []
    for basis in bases:
        shells = basis.shells()
        for shell in shells:
            shell.origin = (np.array(shell.origin)+[.7, -.3, .4]).tolist()
        mol = vq.Molecule([vq.Atom(2, s.origin) for s in shells], 0, 1)
        translated.append(vq.BasisSet(mol, shells, "translated", True))
    shifted = manta.FittingTiles(translated)
    np.testing.assert_allclose(actual, shifted.evaluate((0,)*arity)["values"], atol=3e-12)
    assert provider.source_identity != shifted.source_identity


@pytest.mark.parametrize("field,plan_field", [
    ("shell_calls", "shell_calls"), ("primitive_quartets", "primitive_products"),
    ("scalar_terms", "integral_elements"), ("kernel_terms", "kernel_terms"),
    ("memory_bytes", "modeled_peak_bytes"),
])
@pytest.mark.parametrize("arity", [2, 3])
def test_fitting_resource_boundaries(field, plan_field, arity):
    axes = [_basis((0, 1))]*arity
    plan = manta.plan_fitting(axes)
    budget = manta.ResourceBudget()
    setattr(budget, field, getattr(plan, plan_field))
    provider = manta.FittingTiles(axes, budget)
    provider.evaluate((0,)*arity)
    setattr(budget, field, getattr(plan, plan_field)-1)
    with pytest.raises(ValueError, match="budget"):
        manta.FittingTiles(axes, budget)


def test_fitting_axis_arity_indices_and_role_limits():
    basis = _basis((0,))
    for axes in [[], [basis], [basis]*4]:
        with pytest.raises(ValueError, match="two or three"):
            manta.FittingTiles(axes)
    provider = manta.FittingTiles([basis]*3)
    with pytest.raises(ValueError, match="arity"):
        provider.evaluate((0, 0))
    with pytest.raises(IndexError):
        provider.evaluate((0, 0, 1))
    high = _basis((7,), contracted=False)
    manta.FittingTiles([high, high])
    manta.FittingTiles([high, basis, basis])
    with pytest.raises(ValueError, match="angular capability"):
        manta.FittingTiles([basis, high, basis])
    with pytest.raises(ValueError, match="angular capability"):
        manta.FittingTiles([_basis((8,), contracted=False), basis])


def test_fitting_storage_is_per_tile_not_full_tensor():
    basis = _basis()
    provider = manta.FittingTiles([basis]*3)
    assert provider.plan.tile_elements < provider.plan.integral_elements
    assert provider.plan.shape == [basis.nbasis]*3
    assert provider.plan.shell_extents == [basis.nshells]*3
    before = provider.evaluate((0, 0, 0))["values"]
    provider.evaluate((2, 2, 2))
    assert before.flags.owndata


def test_temporary_basis_sequence_is_retained_during_admission_and_copy():
    class TemporaryBases:
        def __len__(self):
            return 2

        def __getitem__(self, index):
            if index >= 2:
                raise IndexError(index)
            return _basis((index,), contracted=False)

        def __iter__(self):
            raise AssertionError("Only the admitted number of axes may be accessed")

    provider = manta.FittingTiles(TemporaryBases())
    plan = manta.plan_fitting(TemporaryBases())
    reference = manta.FittingTiles([_basis((i,), contracted=False) for i in range(2)])
    assert plan.shape == reference.plan.shape
    assert provider.source_identity == reference.source_identity
    np.testing.assert_array_equal(provider.evaluate((0, 0))["values"], reference.evaluate((0, 0))["values"])


def test_zero_auxiliary_channel_and_nonfinite_payload():
    basis = _basis((0,), contracted=False)
    mol = vq.Molecule([vq.Atom(2, s.origin) for s in basis.shells()], 0, 1)
    shells = basis.shells()
    shells[0].coefficients = [0.]
    zero = vq.BasisSet(mol, shells, "zero-channel", True)
    tile = manta.FittingTiles([zero, basis, basis]).evaluate((0, 0, 0))
    np.testing.assert_array_equal(tile["values"], np.zeros((1, 1, 1)))
    assert tile["null_tiles"] == 0
    shells[0].coefficients = [np.nan]
    invalid = vq.BasisSet(mol, shells, "invalid-channel", True)
    with pytest.raises(ValueError, match="finite"):
        manta.FittingTiles([invalid, basis])
    budget = manta.ResourceBudget()
    budget.shell_calls = 0
    with pytest.raises(ValueError, match="shell-call budget"):
        manta.FittingTiles([invalid, basis], budget)


def _decimal_metric_construction(bra,ket,kernel,omega,precision=180):
    """Independent finite Cartesian derivative sum and alternating Boys integral."""
    from decimal import Decimal as D,localcontext
    from math import factorial
    from functools import lru_cache
    from tests.test_manta_fourier import _decimal_phase,_cart
    _decimal_phase([0.,0.,0.],[0.,0.,0.])
    def pw(x,n):return D(1) if n==0 else x**n
    with localcontext() as ctx:
        ctx.prec=precision;pi=+_decimal_phase.pi;out=[]
        for a in bra.shells():
            rows=[[] for _ in range(2*a.l+1 if a.pure else (a.l+1)*(a.l+2)//2)]
            for b in ket.shells():
                ca,cb=_cart(a.l),_cart(b.l);cart=[[D(0) for _ in cb] for _ in ca]
                delta=[D(float(x))-D(float(y)) for x,y in zip(a.origin,b.origin)]
                for pa,wa in zip(a.exponents,a.coefficients):
                    for pb,wb in zip(b.exponents,b.coefficients):
                        pa,pb,wa,wb=map(lambda x:D(float(x)),(pa,pb,wa,wb));rho=pa*pb/(pa+pb);t=rho*sum(x*x for x in delta)
                        theta=D(0) if kernel=='coulomb' else D(float(omega))/(D(float(omega))**2+rho).sqrt()
                        low,high=(D(0),theta) if kernel=='erf' else (theta,D(1))
                        seeds=[]
                        for n in range(a.l+b.l+1):
                            total=D(0);term=D(1)
                            for k in range(4000):
                                degree=2*n+2*k+1
                                value=term*(pw(high,degree)-pw(low,degree))/degree;total+=value
                                if k>int(t)+40 and abs(value)<D('1e-150'):break
                                term*=-t/(k+1)
                            else:raise AssertionError('independent radial series exhausted')
                            seeds.append(total)
                        @lru_cache(None)
                        def derivative(x,y,z):
                            total=D(0)
                            for i in range(x//2+1):
                                for j in range(y//2+1):
                                    for k in range(z//2+1):
                                        degree=x+y+z-i-j-k
                                        term=pw(-2*rho,degree)*seeds[degree]
                                        for order,pair,d in zip((x,y,z),(i,j,k),delta):
                                            term*=D(factorial(order))/D(2**pair*factorial(pair)*factorial(order-2*pair))*pw(d,order-2*pair)
                                        total+=term
                            return total
                        def terms(powers,exponent):
                            powers=tuple(map(int,powers));terms=[]
                            for x in range(powers[0]%2,powers[0]+1,2):
                                for y in range(powers[1]%2,powers[1]+1,2):
                                    for z in range(powers[2]%2,powers[2]+1,2):
                                        v=D(1)
                                        for n,h in zip(powers,(x,y,z)):
                                            v*=D(factorial(n))/D(factorial((n-h)//2)*factorial(h)*2**n)/pw(exponent,(n+h)//2)
                                        terms.append(((x,y,z),v))
                            return terms
                        ea=[terms(p,pa) for p in ca];eb=[terms(p,pb) for p in cb]
                        scale=2*pi*pi*pi.sqrt()/(pa*pb*(pa+pb).sqrt())*wa*wb
                        for i,ta in enumerate(ea):
                            for j,tb in enumerate(eb):
                                cart[i][j]+=scale*sum(u*v*((-1)**sum(q))*derivative(*(p[d]+q[d] for d in range(3))) for p,u in ta for q,v in tb)
                from tests.test_manta_md import _decimal_representation
                ta=_decimal_representation(a.l,a.pure)
                tb=_decimal_representation(b.l,b.pure)
                intermediate=[[sum(t*v for t,v in zip(row,col)) for col in zip(*cart)] for row in ta]
                for i,row in enumerate(intermediate):rows[i].extend(sum(t*v for t,v in zip(coeff,row)) for coeff in tb)
            out.extend(rows)
        return out


def _check_metric_construction(a,b,kernel='erfc',omega=.6,precision=180):
    from decimal import Decimal as D
    request=manta.KernelRequest();request.centers=2;request.kernel=kernel;request.omega=omega if kernel!='coulomb' else 0.
    c=manta.certify_metric_construction(a,b,request)
    source=manta.FittingTiles([a,b],request=request);actual=np.empty_like(c.evaluated)
    for i,si in enumerate(_slices(a)):
        for j,sj in enumerate(_slices(b)):actual[si,sj]=source.evaluate([i,j])['values']
    np.testing.assert_array_equal(c.evaluated,actual)
    expected=_decimal_metric_construction(a,b,kernel,omega,precision=precision)
    for i,row in enumerate(expected):
        for j,z in enumerate(row):
            assert D(float(c.lower[i,j]))<=z<=D(float(c.upper[i,j])),(i,j,z,c.lower[i,j],c.upper[i,j])
            assert abs(D(float(c.evaluated[i,j]))-z)<=D(float(c.error_bounds[i,j]))
    return c


@pytest.mark.parametrize('l',range(8))
@pytest.mark.parametrize('pure',[False,True])
@pytest.mark.parametrize('kernel',['coulomb','erf','erfc'])
def test_metric_construction_closed_derivative_witness(l,pure,kernel):
    _check_metric_construction(_basis((l,),pure),_basis((1,),pure,shift=(.3,-.2,.4)),kernel)


@pytest.mark.parametrize('angular,scale,shift,omega',[
    ((7,7),1.,(.1,-.2,.3),.6),((3,4),1e-20,(.7,-.3,1.1),1e3),
    ((2,3),1e20,(.2,.3,.4),1e-8),((0,0),1.,(12.,0.,0.),.6),
    ((0,0),1.,(40.,0.,0.),.6),((1,2),1.,(0.,0.,0.),0.)])
def test_metric_construction_cancellation_and_large_argument(angular,scale,shift,omega):
    a=_basis((angular[0],),True,contracted=False);b=_basis((angular[1],),True,contracted=False,shift=shift)
    shell=a.shells()[0];shell.coefficients=[x*scale for x in shell.coefficients]
    a=vq.BasisSet(vq.Molecule([vq.Atom(2,shell.origin)]),[shell],'metric-proof',True)
    if shift[0]>=40.:
        # Positive erfc panels include an explicit high-T omitted-tail bound.
        request=manta.KernelRequest();request.centers=2;request.kernel='erfc';request.omega=omega
        c=manta.certify_metric_construction(a,b,request)
        assert np.all(c.lower<=c.upper) and np.all(c.upper>=0.) and np.all(np.isfinite(c.error_bounds))
        from decimal import Decimal as D
        exact=_decimal_metric_construction(a,b,'erfc',omega,precision=800)[0][0]
        assert exact>0 and D(float(c.lower[0,0]))<=exact<=D(float(c.upper[0,0]))
        assert abs(D(float(c.evaluated[0,0]))-exact)<=D(float(c.error_bounds[0,0]))
    else:_check_metric_construction(a,b,'erfc',omega)


@pytest.mark.parametrize('field,quantity',[('memory_bytes','modeled_peak_bytes'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms'),('shell_calls','shell_calls'),('primitive_quartets','primitive_products')])
def test_metric_construction_exact_resource_caps(field,quantity):
    a=_basis((1,),True);p=manta.plan_metric_construction(a,a);b=manta.ResourceBudget();setattr(b,field,getattr(p,quantity))
    manta.certify_metric_construction(a,a,budget=b)
    setattr(b,field,getattr(p,quantity)-1)
    with pytest.raises((ValueError,RuntimeError),match='budget'):manta.certify_metric_construction(a,a,budget=b)


@pytest.mark.parametrize('kind',['diffuse','tight','signed-cancellation'])
def test_metric_construction_primitive_scales_and_signed_contraction(kind):
    a=_basis((2,),True,contracted=False).shells()[0];b=_basis((3,),True,contracted=False,shift=(.2,.1,-.3)).shells()[0]
    if kind=='signed-cancellation':
        a.exponents=[.7,.7]*8+[1.6];a.coefficients=[1.,-1.]*8+[1e-14]
    else:
        exponent=1e-6 if kind=='diffuse' else 1e6
        for shell in (a,b):
            shell.exponents=[exponent,2*exponent];shell.coefficients=[.8,-.18]
            shell.origin=[x/np.sqrt(exponent) for x in shell.origin]
    def basis(shell):return vq.BasisSet(vq.Molecule([vq.Atom(2,shell.origin)]),[shell],'metric-construction-extreme',True)
    _check_metric_construction(basis(a),basis(b),'erfc',.6 if kind=='signed-cancellation' else .6*np.sqrt(exponent))


@pytest.mark.parametrize('distance,omega,relative_limit',[(0.,1e3,1e-10),(0.,1e8,1e-10),(12.,.6,1e-9),(40.,.6,1e-8)])
def test_positive_radial_cancellation_remains_relative(distance,omega,relative_limit):
    a=_basis((0,),False,False);b=_basis((0,),False,False,shift=(distance,0.,0.))
    c=_check_metric_construction(a,b,'erfc',omega,precision=800 if distance>=40. else 180)
    assert c.lower[0,0]>0 and c.error_bounds[0,0]<relative_limit*abs(c.evaluated[0,0])


@pytest.mark.parametrize('distance,omega',[(200.,1e-4),(2e50,1e-60)])
def test_positive_radial_large_argument_against_gamma_limit(distance,omega):
    from decimal import Decimal as D,localcontext
    from tests.test_manta_fourier import _decimal_phase
    a=_basis((0,),False,False);b=_basis((0,),False,False,shift=(distance,0.,0.))
    request=manta.KernelRequest();request.centers=2;request.kernel='erfc';request.omega=omega
    c=manta.certify_metric_construction(a,b,request)
    _decimal_phase([0.,0.,0.],[0.,0.,0.])
    with localcontext() as ctx:
        ctx.prec=180;pi=+_decimal_phase.pi;sa,sb=a.shells()[0],b.shells()[0]
        p,q=D(float(sa.exponents[0])),D(float(sb.exponents[0]));rho=p*q/(p+q)
        t=rho*sum((D(float(x))-D(float(y)))**2 for x,y in zip(sa.origin,sb.origin))
        w=D(omega);kappa=w*w/(w*w+rho);theta=kappa.sqrt();small=kappa*t
        # Independent half-line Gaussian integral minus a small-argument
        # alternating series; omitted full-Gaussian tail <=exp(-T)/(2T).
        series=D(0);term=D(1)
        for k in range(100):
            series+=term/(2*k+1);term*=-small/(k+1)
        seed=pi.sqrt()/(2*t.sqrt())-theta*series
        scale=2*pi*pi*pi.sqrt()/(p*q*(p+q).sqrt())*D(float(sa.coefficients[0]))*D(float(sb.coefficients[0]))
        upper=scale*seed
        # For these T values the full-Gaussian tail is below 1e-1000;
        # exp of the second argument exceeds Decimal's exponent range.
        assert t>D(10000)
        lower=upper-abs(scale)*D('1e-1000')
        assert D(float(c.lower[0,0]))<=lower<=upper<=D(float(c.upper[0,0]))
        assert abs(D(float(c.evaluated[0,0]))-upper)<=D(float(c.error_bounds[0,0]))
    assert c.lower[0,0]>0 and c.error_bounds[0,0]<1e-9*abs(c.evaluated[0,0])


@pytest.mark.parametrize('target',[8.,16.,32.,64.])
def test_positive_radial_panel_boundary(target):
    # Perturb geometry across the count-only positive-series panel boundaries.
    rho=.35;complement=rho/(rho+.6**2);distance=np.sqrt(target/(rho*complement))
    a=_basis((0,),False,False);request=manta.KernelRequest();request.centers=2;request.kernel='erfc';request.omega=.6
    for direction in (0.,np.inf):
        d=distance
        for _ in range(5):
            d=np.nextafter(d,direction);b=_basis((0,),False,False,shift=(d,0.,0.))
            c=manta.certify_metric_construction(a,b,request)
            assert 0<c.lower[0,0]<=c.upper[0,0] and np.isfinite(c.error_bounds[0,0])


def _decimal_three_center(shells,kernel='coulomb',omega=0.,selected=None,precision=180):
    """Closed polynomial/Hermite coefficients and finite derivative sum.

    Independent of the producer's Hermite raising and Coulomb recurrence.
    selected limits high-angular Cartesian witnesses without using native data.
    """
    from decimal import Decimal as D,localcontext
    from math import comb,factorial
    from functools import lru_cache
    from tests.test_manta_fourier import _decimal_phase,_cart
    from tests.test_manta_md import _decimal_representation
    _decimal_phase([0.],[0.]);a,b,c=shells
    def pw(x,n):return D(1) if n==0 else x**n
    with localcontext() as ctx:
        ctx.prec=precision;pi=+_decimal_phase.pi
        carts=[_cart(s.l) for s in shells];shape=tuple(len(v) for v in carts)
        assert selected is None or not any(s.pure for s in shells)
        indices=list(np.ndindex(shape)) if selected is None else selected
        cart={idx:D(0) for idx in indices};A,B,C=[list(map(D,s.origin)) for s in shells]
        for ia,ib,ic in product(*(range(len(s.exponents)) for s in shells)):
            alpha,beta,gamma=[D(s.exponents[i]) for s,i in zip(shells,(ia,ib,ic))];p=beta+gamma;rho=alpha*p/(alpha+p)
            center=[(beta*x+gamma*y)/p for x,y in zip(B,C)];delta=[x-y for x,y in zip(A,center)]
            pa=[x-y for x,y in zip(center,B)];pb=[x-y for x,y in zip(center,C)]
            t=rho*sum(x*x for x in delta)
            theta=D(0) if kernel=='coulomb' or omega==0. else D(1) if np.isinf(omega) else D(omega)/(D(omega)**2+rho).sqrt()
            low,high=(D(0),theta) if kernel=='erf' else (theta,D(1))
            seeds=[]
            for n in range(sum(s.l for s in shells)+1):
                total=D(0);term=D(1)
                for k in range(4000):
                    degree=2*n+2*k+1;value=term*(pw(high,degree)-pw(low,degree))/degree;total+=value
                    if k>int(t)+40 and abs(value)<D('1e-150'):break
                    term*=-t/(k+1)
                else:raise AssertionError('three-center reference radial series exhausted')
                seeds.append(total)
            @lru_cache(None)
            def derivative(x,y,z):
                value=D(0)
                for i,j,k in product(range(x//2+1),range(y//2+1),range(z//2+1)):
                    degree=x+y+z-i-j-k;term=pw(-2*rho,degree)*seeds[degree]
                    for order,pairs,d in zip((x,y,z),(i,j,k),delta):
                        term*=D(factorial(order))/D(2**pairs*factorial(pairs)*factorial(order-2*pairs))*pw(d,order-2*pairs)
                    value+=term
                return value
            @lru_cache(None)
            def axis(i,j,h,dimension,auxiliary):
                if auxiliary:
                    if h>i or (i-h)%2:return D(0)
                    return D(factorial(i))/D(factorial((i-h)//2)*factorial(h)*2**i)/pw(alpha,(i+h)//2)
                value=D(0)
                for u in range(i+1):
                    for v in range(j+1):
                        n=u+v
                        if n<h or (n-h)%2:continue
                        coefficient=D(comb(i,u)*comb(j,v)*factorial(n))/D(factorial((n-h)//2)*factorial(h)*2**n)/pw(p,(n+h)//2)
                        value+=coefficient*pw(pa[dimension],i-u)*pw(pb[dimension],j-v)
                return value
            @lru_cache(None)
            def terms(i,j,auxiliary):
                first=carts[0 if auxiliary else 1][i];second=(0,0,0) if auxiliary else carts[2][j];result=[]
                for h in product(*(range(int(x+y)+1) for x,y in zip(first,second))):
                    weight=D(1)
                    for d,(x,y,t0) in enumerate(zip(first,second,h)):weight*=axis(int(x),int(y),t0,d,auxiliary)
                    if weight:result.append((h,weight))
                return result
            scale=2*pi*pi*pi.sqrt()/(alpha*p*(alpha+p).sqrt())*(-beta*gamma/p*sum((x-y)**2 for x,y in zip(B,C))).exp()
            for s,i in zip(shells,(ia,ib,ic)):scale*=D(s.coefficients[i])
            for idx in indices:
                i,j,k=idx
                cart[idx]+=scale*sum(u*v*((-1)**sum(q))*derivative(*(h[d]+q[d] for d in range(3))) for h,u in terms(i,0,True) for q,v in terms(j,k,False))
        if selected is not None:return cart
        array=np.empty(shape,dtype=object)
        for idx,value in cart.items():array[idx]=value
        for ax,shell in enumerate(shells):
            transform=np.array(_decimal_representation(shell.l,shell.pure),dtype=object)
            array=np.moveaxis(np.tensordot(transform,array,axes=(1,ax)),0,ax)
        return {idx:array[idx] for idx in np.ndindex(array.shape)}


def _check_three_center_construction(bases,kernel='erfc',omega=.6,selected=None,precision=180):
    from decimal import Decimal as D,localcontext
    request=manta.KernelRequest();request.centers=3;request.kernel=kernel;request.omega=omega if kernel!='coulomb' else 0.
    proof=manta.ThreeCenterConstructionTiles(*bases,request=request);source=manta.FittingTiles(bases,request=request)
    assert proof.source_identity==source.source_identity
    for idx in product(*(range(b.nshells) for b in bases)):
        tile=proof.evaluate(idx);actual=source.evaluate(idx)
        np.testing.assert_array_equal(tile['values'],actual['values']);assert tile['source_identity']==source.source_identity
        expected=_decimal_three_center([b.shells()[i] for b,i in zip(bases,idx)],kernel,omega,selected,precision)
        with localcontext() as ctx:
            ctx.prec=precision
            for key,value in expected.items():
                assert D(float(tile['lower'][key]))<=value<=D(float(tile['upper'][key])),(idx,key,value,tile['lower'][key],tile['upper'][key])
                assert abs(D(float(tile['values'][key]))-value)<=D(float(tile['error_bounds'][key]))
        assert np.all(np.isfinite(tile['error_bounds']))
    return proof


@pytest.mark.parametrize('angular',[(l,1,1) for l in range(8)]+[(1,l,1) for l in range(2,7)]+[(1,1,l) for l in range(2,7)])
@pytest.mark.parametrize('pure',[False,True])
def test_three_center_construction_independent_angular(angular,pure):
    bases=[_basis((l,),pure,False,shift=(.13*i,-.21*i,.32*i)) for i,l in enumerate(angular)]
    _check_three_center_construction(bases)


@pytest.mark.parametrize('kernel,omega',[('coulomb',0.),('erf',.6),('erfc',.6),('erf',0.),('erfc',0.),('erf',float('inf')),('erfc',float('inf')),('erfc',1e8),('erf',1e-8)])
def test_three_center_construction_contracted_ranges(kernel,omega):
    bases=[_basis((l,),pure,True,shift=(.2*i,-.1*i,.3*i)) for i,(l,pure) in enumerate([(2,True),(1,False),(2,True)])]
    if np.isinf(omega):
        request=manta.KernelRequest();request.centers=3;request.kernel=kernel;request.omega=omega
        with pytest.raises(ValueError,match='finite'):manta.ThreeCenterConstructionTiles(*bases,request=request)
    else:_check_three_center_construction(bases,kernel,omega)


@pytest.mark.parametrize('angular',[(7,6,6),(6,6,6)])
def test_three_center_construction_maximum_orders(angular):
    bases=[_basis((l,),False,False,shift=(.03*i,-.02*i,.01*i)) for i,l in enumerate(angular)]
    # Fully evaluate the maximum-order tile; independent selected Cartesian
    # witnesses exercise radial/derivative orders through 19 without rounded E.
    selected=[(0,0,0),(1,2,3),tuple(b.nbasis-1 for b in bases)]
    _check_three_center_construction(bases,selected=selected)


@pytest.mark.parametrize('kind',['signed','coincident','diffuse','tight','underflow'])
def test_three_center_construction_extremes(kind):
    shells=[_basis((l,),True,False,shift=(.2*i,-.1*i,.3*i)).shells()[0] for i,l in enumerate((1,2,1))]
    omega=.6
    if kind=='signed':shells[1].exponents=[.7,.7]*4+[1.6];shells[1].coefficients=[1.,-1.]*4+[1e-14]
    elif kind=='coincident':
        for s in shells:s.origin=[0.,0.,0.]
    elif kind=='underflow':
        shells[0].origin=[50.,0.,0.];shells[1].origin=[0.,0.,0.];shells[2].origin=[100.,0.,0.]
    else:
        exponent=1e-6 if kind=='diffuse' else 1e6;omega*=np.sqrt(exponent)
        for s in shells:s.exponents=[exponent];s.coefficients=[1.];s.origin=[x/np.sqrt(exponent) for x in s.origin]
    bases=[vq.BasisSet(vq.Molecule([vq.Atom(2,s.origin)]),[s],'three-center-proof',True) for s in shells]
    _check_three_center_construction(bases,omega=omega)


@pytest.mark.parametrize('field,quantity',[('memory_bytes','modeled_peak_bytes'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms'),('shell_calls','shell_calls'),('primitive_quartets','primitive_products')])
def test_three_center_construction_exact_caps(field,quantity):
    a=_basis((1,),True);p=manta.plan_three_center_construction(a,a,a);b=manta.ResourceBudget();setattr(b,field,getattr(p,quantity))
    manta.ThreeCenterConstructionTiles(a,a,a,budget=b).evaluate([0,0,0]);setattr(b,field,getattr(p,quantity)-1)
    with pytest.raises((ValueError,RuntimeError),match='budget'):manta.ThreeCenterConstructionTiles(a,a,a,budget=b)


def test_three_center_construction_streaming_lifetime_and_refusals():
    a=_basis((0,1,2));proof=manta.ThreeCenterConstructionTiles(a,a,a);single=_basis((2,))
    p=manta.plan_three_center_construction(single,single,single)
    assert proof.plan.modeled_peak_bytes<2*p.modeled_peak_bytes # metadata scales, never full global tensor
    saved=proof.evaluate([0,1,2]);original=saved['values'].copy();proof.evaluate([2,1,0]);np.testing.assert_array_equal(saved['values'],original)
    with pytest.raises(AttributeError):proof.source_identity='wrong'
    for indices in ([0,0],[0,0,0,0]):
        with pytest.raises(ValueError,match='arity'):proof.evaluate(indices)
    with pytest.raises(IndexError):proof.evaluate([0,0,99])
    request=manta.KernelRequest();request.centers=2
    with pytest.raises(ValueError,match='center'):manta.plan_three_center_construction(a,a,a,request)
    request.centers=3;request.derivative=1
    with pytest.raises(ValueError):manta.plan_three_center_construction(a,a,a,request)
    with pytest.raises(ValueError,match='angular'):manta.plan_three_center_construction(a,_basis((7,),False),a)
    with pytest.raises(ValueError,match='angular'):manta.plan_three_center_construction(_basis((8,),False),a,a)
    shell=_basis((0,),False,False).shells()[0];shell.coefficients=[float('nan')]
    invalid=vq.BasisSet(vq.Molecule([vq.Atom(2,shell.origin)]),[shell],'invalid',True)
    with pytest.raises(ValueError):manta.ThreeCenterConstructionTiles(invalid,a,a)
    del proof;np.testing.assert_array_equal(saved['values'],original)


@pytest.mark.parametrize('arity',[2,3])
def test_fitting_scalar_producer_refuses_derivative_request(arity):
    a=_basis((0,));request=manta.KernelRequest();request.centers=arity;request.derivative=1
    for operation in (lambda:manta.plan_fitting([a]*arity,request=request),lambda:manta.FittingTiles([a]*arity,request=request)):
        with pytest.raises(ValueError,match='derivative=0'):operation()
    if arity==2:
        with pytest.raises(ValueError,match='derivative=0'):manta.certify_metric_construction(a,a,request)
    else:
        with pytest.raises(ValueError,match='derivative=0'):manta.ThreeCenterConstructionTiles(a,a,a,request)
    # The existing derivative provider still owns first derivatives.
    derivative=manta.DerivativeTiles([a]*arity,request=request).evaluate([0]*arity,0,0)
    assert np.all(np.isfinite(derivative['values']))
