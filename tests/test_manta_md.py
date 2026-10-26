"""MD kernels against independent quadrature, analytic values and libint."""
from decimal import Decimal, localcontext
from itertools import product
from math import factorial, pi

import numpy as np
import pytest

from vibeqc import Atom, BasisSet, Molecule, compute_eri, manta
from vibeqc._vibeqc_core import ShellInfo


def _basis(angular=(0, 1, 2), pure=True, contracted=True):
    positions = [[0., 0., 0.], [.2, -.3, 1.1], [-.4, .5, .7], [.7, .1, -.2]]
    mol = Molecule([Atom(2, p) for p in positions], 0, 1)
    shells = []
    for atom, l in enumerate(angular):
        s = ShellInfo()
        s.atom_index = atom
        s.l, s.pure, s.origin = l, pure, positions[atom]
        s.exponents = [.65 + .13*atom, 1.6 + .1*atom] if contracted else [.65 + .13*atom]
        s.coefficients = [.8, -.18] if contracted else [1.]
        shells.append(s)
    return BasisSet(mol, shells, "manta-md", False)


def _cart(l):
    return [(x, y, l-x-y) for x in range(l, -1, -1) for y in range(l-x, -1, -1)]


@pytest.mark.parametrize("n", [0, 1, 4, 12, 24, 32])
def test_boys_against_95_digit_reference(n):
    for t in [0., 1e-20, 1e-8, .1, 1., 12., 30., 31., 42., 54., 62., 100., 117., 500.]:
        actual = manta.boys(n, t)
        with localcontext() as ctx:
            ctx.prec = 95
            x = Decimal(t)
            for m, value in enumerate(actual):
                term = total = Decimal(1)
                for k in range(1, 2000):
                    term *= 2*x/Decimal(2*m+2*k+1)
                    total += term
                    if term < total*Decimal("1e-90"):
                        break
                reference = float((-x).exp()*total/Decimal(2*m+1))
                assert value == pytest.approx(reference, rel=1e-13, abs=0.)
    for t in [1e4, 1e100, 1e300]:
        actual = manta.boys(n, t)
        assert actual[0] == pytest.approx(.5*np.sqrt(pi/t), rel=1e-14)
        assert all(np.isfinite(actual)) and all(v >= 0 for v in actual)


@pytest.mark.parametrize("n,t", [(-1, 1.), (33, 1.), (0, -1.), (0, np.nan), (0, np.inf)])
def test_boys_refuses_unadmitted_inputs(n, t):
    with pytest.raises(ValueError):
        manta.boys(n, t)


@pytest.mark.parametrize("l", range(8))
def test_exact_polynomial_transform_against_spherical_harmonics(l):
    special = pytest.importorskip("scipy.special")
    rng = np.random.default_rng(6300+l)
    xyz = rng.normal(size=(30, 3))
    radius = np.linalg.norm(xyz, axis=1)
    theta = np.arccos(xyz[:, 2]/radius)
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    cart = np.array([np.prod(xyz**p, axis=1) for p in _cart(l)])
    actual = manta.cartesian_to_spherical(l) @ cart
    expected = []
    for m in range(-l, l+1):
        harmonic = special.sph_harm_y(l, abs(m), theta, phi)
        radial = radius**l*np.sqrt(4*pi/(2*l+1))
        expected.append(radial*(np.sqrt(2)*(-1)**abs(m) if m else 1)
                        *(harmonic.imag if m < 0 else harmonic.real))
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=1e-12)


@pytest.mark.parametrize("pure", [False, True])
def test_all_contracted_spd_tiles_and_jk_against_libint(pure):
    basis = _basis(pure=pure)
    direct, reference = manta.DirectJKBuilder(basis), manta.ReferenceJKBuilder(basis)
    md_session, li_session = direct.session(), reference.session()
    for quartet in product(range(basis.nshells), repeat=4):
        result = md_session.evaluate(quartet)
        assert result["provider"] == "manta"
        assert result["null_tiles"] == 0
        assert result["primitive_quartets_requested"] == 16
        np.testing.assert_allclose(result["values"], li_session.evaluate(quartet)["values"], atol=2e-12, rtol=1e-11)
    rng = np.random.default_rng(6301)
    d = rng.normal(size=(basis.nbasis, basis.nbasis)); d += d.T
    eri = np.asarray(compute_eri(basis))
    result = direct.build(d)
    np.testing.assert_allclose(result.J, np.einsum("abcd,cd->ab", eri, d), atol=5e-12, rtol=1e-11)
    np.testing.assert_allclose(result.K, np.einsum("abcd,bd->ac", eri, d), atol=5e-12, rtol=1e-11)
    assert result.receipt.provider == "manta"
    assert result.receipt.provider_version == "md-fp64-v1"
    assert result.receipt.source_identity != reference.source_identity
    assert result.receipt.null_tiles == 0
    assert result.receipt.integral_elements_consumed == basis.nbasis**4
    assert result.receipt.primitive_quartets_requested == direct.plan.primitive_quartets_upper_bound
    j, k = result.J.copy(), result.K.copy()
    direct.build(np.zeros_like(d))
    np.testing.assert_array_equal(result.J, j)
    np.testing.assert_array_equal(result.K, k)


@pytest.mark.parametrize("l", range(2, 7))
@pytest.mark.parametrize("pure", [False, True])
def test_higher_angular_tiles_against_generated_libint(l, pure):
    if l > manta.capability().maximum_l:
        pytest.skip("The independently generated libint build lacks this orbital angular class")
    for angular in [(l, 0, 0, 0), (l, 1, l, 0), (l, l, 0, 0)]:
        basis = _basis(angular, pure, contracted=False)
        md = manta.DirectJKBuilder(basis).evaluate((0, 1, 2, 3))["values"]
        li = manta.ReferenceJKBuilder(basis).evaluate((0, 1, 2, 3))["values"]
        np.testing.assert_allclose(md, li, atol=1e-11, rtol=1e-10)


@pytest.mark.parametrize("angular", [(6, 0, 6, 0), (6, 1, 5, 2), (6, 6, 6, 6)])
@pytest.mark.parametrize("pure", [False, True])
def test_complete_l6_tiles_against_optional_libcint(angular, pure):
    gto = pytest.importorskip("pyscf.gto")  # optional reference, never a runtime backend
    basis = _basis(angular, pure=pure, contracted=False)
    shells = basis.shells()
    reference_molecule = gto.M(
        atom=[(f"He{i}", s.origin) for i, s in enumerate(shells)],
        basis={f"He{i}": [[s.l, [s.exponents[0], 1.]]] for i, s in enumerate(shells)},
        unit="Bohr", cart=not pure, verbose=0,
    )
    name = "int2e_sph" if pure else "int2e_cart"
    reference = reference_molecule.intor_by_shell(name, (0, 1, 2, 3))
    if pure:
        # PySCF p functions are x,y,z; the provider's m=-l..l order is y,z,x.
        for axis, l in enumerate(angular):
            if l == 1:
                reference = np.take(reference, [1, 2, 0], axis=axis)
    else:
        # The reference uses radial normalization for Cartesian d and higher.
        # Convert each shell to the common axial primitive normalization.
        reference *= np.prod([np.sqrt((2*l+1)/(4*pi)) if l >= 2 else 1. for l in angular])
    actual = manta.DirectJKBuilder(basis).evaluate((0, 1, 2, 3))["values"]
    np.testing.assert_allclose(actual, reference, atol=2e-11, rtol=2e-10)


@pytest.mark.parametrize("scale", [1e-8, 1e-4, 1., 1e4, 1e8])
@pytest.mark.parametrize("pure", [False, True])
def test_diffuse_tight_coulomb_scaling_and_signed_contractions(scale, pure):
    reference_basis = _basis((0, 1, 2, 1), pure=pure)
    baseline = manta.DirectJKBuilder(reference_basis).evaluate((0, 1, 2, 3))["values"]
    shells = reference_basis.shells()
    for shell in shells:
        shell.origin = np.asarray(shell.origin)/np.sqrt(scale)
        shell.exponents = np.asarray(shell.exponents)*scale
        shell.coefficients = [.8, -.18]  # raw coefficients; renormalize once
    molecule = Molecule([Atom(2, s.origin) for s in shells], 0, 1)
    basis = BasisSet(molecule, shells, "manta-scaled", False)
    actual = manta.DirectJKBuilder(basis).evaluate((0, 1, 2, 3))["values"]
    reference = manta.ReferenceJKBuilder(basis).evaluate((0, 1, 2, 3))["values"]
    # Normalized Gaussian functions scale as length^-3/2 and 1/r as length^-1.
    np.testing.assert_allclose(actual/np.sqrt(scale), baseline, atol=2e-12, rtol=1e-11)
    np.testing.assert_allclose(actual/np.sqrt(scale), reference/np.sqrt(scale), atol=2e-12, rtol=1e-11)


def _laplace_component(shells, components):
    """Independent 1D Laplace quadrature of correlated Gaussian moments.

    Uses 1/r = 2/sqrt(pi) integral_0^infinity exp(-u^2*r^2) du.
    Cartesian moments follow the Gaussian covariance identity, without MD E,
    Hermite Coulomb auxiliaries, Boys functions or the native integral engine.
    """
    quad = pytest.importorskip("scipy.integrate").quad
    alpha, beta, gamma, delta = [s.exponents[0] for s in shells]
    a, b, c, d = [np.asarray(s.origin) for s in shells]
    p, q = alpha+beta, gamma+delta
    P, Q = a+beta/p*(b-a), c+delta/q*(d-c)
    prefactor = np.exp(-alpha*beta/p*np.sum((a-b)**2)-gamma*delta/q*np.sum((c-d)**2))
    prefactor *= np.prod([s.coefficients[0] for s in shells])

    def integrand(angle):
        u = np.tan(angle); u2 = u*u
        det = p*q+(p+q)*u2
        ma, mb = P+q*u2/det*(Q-P), Q+p*u2/det*(P-Q)
        var_a, var_b, cov = (q+u2)/(2*det), (p+u2)/(2*det), u2/(2*det)
        value = 2*pi**2.5/det**1.5*np.exp(-p*q*u2/det*np.sum((P-Q)**2))/np.cos(angle)**2
        for axis in range(3):
            def polynomial(first, second, powers, mean):
                left = np.array([factorial(powers[0])/(factorial(i)*factorial(powers[0]-i))
                                 *(mean-first)**(powers[0]-i) for i in range(powers[0]+1)])
                right = np.array([factorial(powers[1])/(factorial(i)*factorial(powers[1]-i))
                                  *(mean-second)**(powers[1]-i) for i in range(powers[1]+1)])
                return np.polynomial.polynomial.polymul(left, right)
            ca = polynomial(a[axis], b[axis], [components[0][axis], components[1][axis]], ma[axis])
            cb = polynomial(c[axis], d[axis], [components[2][axis], components[3][axis]], mb[axis])
            moments = np.zeros((len(ca), len(cb))); moments[0, 0] = 1
            for j in range(2, len(cb)):
                moments[0, j] = (j-1)*var_b*moments[0, j-2]
            for i in range(1, len(ca)):
                for j in range(len(cb)):
                    moments[i, j] = ((i-1)*var_a*moments[i-2, j] if i > 1 else 0)
                    if j:
                        moments[i, j] += j*cov*moments[i-1, j-1]
            value *= ca @ moments @ cb
        return value*prefactor
    result, error = quad(integrand, 0., pi/2, epsabs=1e-13, epsrel=1e-12, limit=200)
    assert error < 1e-10
    return result


@pytest.mark.parametrize("angular", [(0, 0, 0, 0), (2, 1, 2, 1), (6, 0, 6, 0), (6, 6, 0, 0)])
def test_coulomb_against_independent_laplace_quadrature(angular):
    basis = _basis(angular, pure=False, contracted=False)
    tile = manta.DirectJKBuilder(basis).evaluate((0, 1, 2, 3))["values"]
    components = [_cart(l) for l in angular]
    for indices in [tuple(0 for _ in angular), tuple(len(c)//2 for c in components), tuple(len(c)-1 for c in components)]:
        reference = _laplace_component(basis.shells(), [c[i] for c, i in zip(components, indices, strict=True)])
        assert tile[indices] == pytest.approx(reference, abs=2e-11, rel=1e-10)


@pytest.mark.parametrize("field,plan_field", [
    ("memory_bytes", "modeled_peak_bytes"), ("shell_calls", "shell_calls"),
    ("primitive_quartets", "primitive_quartets_upper_bound"), ("scalar_terms", "scalar_terms"),
    ("kernel_terms", "kernel_terms"),
])
def test_md_resource_caps_before_build(field, plan_field):
    basis = _basis()
    plan = manta.md_plan(basis)
    assert plan.provider == "manta"
    budget = manta.ResourceBudget()
    setattr(budget, field, getattr(plan, plan_field))
    manta.DirectJKBuilder(basis, budget=budget).build(np.eye(basis.nbasis))
    setattr(budget, field, getattr(plan, plan_field)-1)
    with pytest.raises(ValueError, match="budget"):
        manta.DirectJKBuilder(basis, budget=budget)


def test_md_builder_in_native_rhf_and_citation_route():
    import vibeqc as vq
    from vibeqc.output.citations import load_default_database
    mol = Molecule([Atom(1, [0., 0., 0.]), Atom(1, [0., 0., 1.4])], 0, 1)
    basis = BasisSet(mol, "sto-3g")
    overlap = vq.compute_overlap(basis)
    core = vq.compute_kinetic(basis) + vq.compute_nuclear(basis, mol)
    options = vq.RHFOptions()
    options.conv_tol_energy = 1e-11
    options.conv_tol_grad = 1e-9
    reference = vq.run_rhf_scf_with_jk(basis, 2, overlap, core, 1/1.4,
                                      manta.ReferenceJKBuilder(basis), options)
    actual = vq.run_rhf_scf_with_jk(basis, 2, overlap, core, 1/1.4,
                                   manta.DirectJKBuilder(basis), options)
    assert actual.converged and reference.converged
    assert actual.energy == pytest.approx(reference.energy, abs=1e-11)
    np.testing.assert_allclose(actual.density, reference.density, atol=1e-10)
    references = load_default_database().assemble(method="RHF", extra_methods=["manta_md"])
    assert {"mcmurchie_davidson_1978", "neese_shark_2023"} <= {c.key for c in references}


@pytest.mark.parametrize("provider", ["ReferenceJKBuilder", "DirectJKBuilder"])
@pytest.mark.parametrize("method", ["build_J_slot", "build_K_slot", "build_K_pair"])
@pytest.mark.parametrize("kind", ["complex", "float32", "strided"])
def test_spin_entries_refuse_implicit_density_conversion(provider, method, kind):
    basis = _basis((0, 1), contracted=False)
    builder = getattr(manta, provider)(basis)
    n = basis.nbasis
    valid = np.eye(n)
    bad = (np.eye(n, dtype=np.complex128)*(1+1j) if kind == "complex"
           else np.eye(n, dtype=np.float32) if kind == "float32" else np.eye(2*n)[::2, ::2])
    entry = getattr(builder, method)
    if method == "build_K_pair":
        with pytest.raises(ValueError):
            entry(bad, valid)
        with pytest.raises(ValueError):
            entry(valid, bad)
    else:
        with pytest.raises(ValueError):
            entry(bad, 0)


@pytest.mark.parametrize("provider", ["ReferenceJKBuilder", "DirectJKBuilder"])
def test_basis_label_storage_is_admitted_before_provider_copy(provider):
    mol = Molecule([Atom(1, [0., 0., 0.]), Atom(1, [0., 0., 1.4])], 0, 1)
    basis = BasisSet(mol, "sto-3g")
    constructor = getattr(manta, provider)
    original = constructor(basis)
    renamed = BasisSet(mol, basis.shells(), "n"*65536, True)
    budget = manta.ResourceBudget()
    budget.memory_bytes = original.plan.modeled_peak_bytes
    with pytest.raises(ValueError, match="budget"):
        constructor(renamed, budget=budget)
    expanded = constructor(renamed)
    assert expanded.source_identity == original.source_identity
    assert expanded.plan.basis_payload_bytes > original.plan.basis_payload_bytes + 65000


@pytest.mark.parametrize("provider", ["ReferenceJKBuilder", "DirectJKBuilder"])
def test_fused_result_refuses_nonfinite_arithmetic(provider):
    basis = _basis((0,), contracted=False)
    builder = getattr(manta, provider)(basis)
    with pytest.raises(OverflowError, match="fused"):
        builder.build_g_rhf(np.array([[1e100]]), 1e308)
