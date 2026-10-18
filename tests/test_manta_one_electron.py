"""Bounded MD one-electron adapters against analytic and independent references."""
from itertools import product
from math import erf, pi, sqrt

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta
from vibeqc._vibeqc_core import ShellInfo


def _basis(angular=(0, 1, 2), pure=True, shift=(0., 0., 0.), contracted=True):
    positions = np.array([[0., .1, -.2], [.2, -.3, 1.1], [-.4, .5, .7]]) + shift
    mol = vq.Molecule([vq.Atom(2, p.tolist()) for p in positions], 0, 1)
    shells = []
    for i, l in enumerate(angular):
        shell = ShellInfo()
        shell.atom_index, shell.l, shell.pure, shell.origin = i, l, pure, positions[i].tolist()
        shell.exponents = [.7+.13*i, 1.6+.1*i] if contracted else [.7+.13*i]
        shell.coefficients = [.8, -.18] if contracted else [1.]
        shells.append(shell)
    return vq.BasisSet(mol, shells, "manta-one-electron", False)


def _request(operation, **fields):
    request = manta.OneElectronRequest()
    request.operation = operation
    for name, value in fields.items():
        setattr(request, name, value)
    return request


def _compute(bra, ket=None, operation="overlap", **fields):
    return manta.compute_one_electron(bra, bra if ket is None else ket, _request(operation, **fields))


@pytest.mark.parametrize("pure", [False, True])
def test_contracted_matrices_against_libint(pure):
    basis = _basis(pure=pure)
    positions, charges = [[.3, .2, -.7], [0., 0., 1.2]], [1.2, -.6]
    for operation, expected, fields in [
        ("overlap", vq.compute_overlap(basis), {}),
        ("kinetic", vq.compute_kinetic(basis), {}),
        ("potential", vq.compute_nuclear_with_charges(basis, positions, charges),
         {"charges": list(zip(charges, positions, strict=True))}),
    ]:
        result = _compute(basis, operation=operation, **fields)
        np.testing.assert_allclose(result.values, expected, atol=2e-12, rtol=3e-12)
        assert result.receipt.provider == "manta"
        assert result.receipt.operation == operation
        assert result.receipt.shell_calls == basis.nshells**2
        assert result.receipt.primitive_pairs == (2*basis.nshells)**2
        assert result.receipt.integral_elements == basis.nbasis**2
    origin = [.1, -.3, .2]
    dipole = vq.compute_dipole(basis, origin)
    for axis, expected in enumerate([dipole.x, dipole.y, dipole.z]):
        powers = [0, 0, 0]; powers[axis] = 1
        np.testing.assert_allclose(_compute(basis, operation="moment", origin=origin, powers=powers).values,
                                   expected, atol=2e-12, rtol=3e-12)
    np.testing.assert_array_equal(_compute(basis, operation="potential").values, np.zeros_like(expected))


@pytest.mark.parametrize("operation", ["overlap", "kinetic", "potential", "moment"])
def test_cross_basis_transpose_and_owned_results(operation):
    bra = _basis((0, 2), pure=False)
    ket = _basis((1,), pure=True, shift=(.4, -.2, .1))
    fields = {"charges": [(2., [.3, -.7, 1.])] } if operation == "potential" else {}
    if operation == "moment":
        fields = {"origin": [.2, .5, -.3], "powers": [2, 1, 1]}
    result = _compute(bra, ket, operation, **fields)
    reverse = _compute(ket, bra, operation, **fields)
    np.testing.assert_allclose(result.values, reverse.values.T, atol=3e-12, rtol=3e-12)
    assert result.values.shape == (bra.nbasis, ket.nbasis)
    assert result.receipt.source_identity != reverse.receipt.source_identity
    before = result.values.copy()
    _compute(bra)
    np.testing.assert_array_equal(result.values, before)
    if operation == "overlap":
        np.testing.assert_allclose(result.values, vq.compute_overlap_two_basis(bra, ket), atol=2e-12, rtol=3e-12)


def _cart(l):
    return np.array([(x, y, l-x-y) for x in range(l, -1, -1) for y in range(l-x, -1, -1)])


def _quadrature(a, b, operation, origin, powers):
    # Product-Gaussian Gauss-Hermite integration. It evaluates polynomials
    # and their Cartesian gradients directly, without Hermite recurrences.
    nodes, weights = np.polynomial.hermite.hermgauss(12)
    grid = np.array(list(product(nodes, repeat=3)))
    weight = np.prod(np.array(list(product(weights, repeat=3))), axis=1)
    ca, cb = _cart(a.l), _cart(b.l)
    result = np.zeros((len(ca), len(cb)))
    for alpha, ac in zip(a.exponents, a.coefficients, strict=True):
        for beta, bc in zip(b.exponents, b.coefficients, strict=True):
            p = alpha+beta
            center = (alpha*np.array(a.origin)+beta*np.array(b.origin))/p
            xyz = grid/np.sqrt(p)+center
            xa, xb = xyz-np.array(a.origin), xyz-np.array(b.origin)
            pa = np.prod(xa[None, :, :]**ca[:, None, :], axis=2)
            pb = np.prod(xb[None, :, :]**cb[:, None, :], axis=2)
            if operation == "kinetic":
                value = np.zeros_like(result)
                for axis in range(3):
                    da, db = ca.copy(), cb.copy()
                    da[:, axis] = np.maximum(0, da[:, axis]-1)
                    db[:, axis] = np.maximum(0, db[:, axis]-1)
                    ga = ca[:, axis, None]*np.prod(xa[None, :, :]**da[:, None, :], axis=2)-2*alpha*xa[None, :, axis]*pa
                    gb = cb[:, axis, None]*np.prod(xb[None, :, :]**db[:, None, :], axis=2)-2*beta*xb[None, :, axis]*pb
                    value += .5*(ga*weight)@gb.T
            else:
                multiplier = np.prod((xyz-origin)**powers, axis=1) if operation == "moment" else 1.
                value = (pa*(weight*multiplier))@pb.T
            factor = ac*bc*np.exp(-alpha*beta/p*np.sum((np.array(a.origin)-b.origin)**2))/p**1.5
            result += factor*value
    if a.pure:
        result = manta.cartesian_to_spherical(a.l)@result
    if b.pure:
        result = result@manta.cartesian_to_spherical(b.l).T
    return result


@pytest.mark.parametrize("l", [0, 2, 4, 6])
@pytest.mark.parametrize("pure", [False, True])
@pytest.mark.parametrize("operation", ["overlap", "kinetic", "moment"])
def test_complete_higher_angular_matrices_by_independent_quadrature(l, pure, operation):
    bra = _basis((l,), pure, contracted=False)
    ket = _basis((l,), pure, shift=(.3, -.4, .7), contracted=False)
    origin, powers = np.array([.2, -.5, .3]), np.array([2, 1, 3])
    fields = {"origin": origin.tolist(), "powers": powers.tolist()} if operation == "moment" else {}
    expected = _quadrature(bra.shells()[0], ket.shells()[0], operation, origin, powers)
    result = _compute(bra, ket, operation, **fields)
    np.testing.assert_allclose(result.values, expected, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("pure", [False, True])
def test_l6_potential_against_optional_libcint(pure):
    gto = pytest.importorskip("pyscf.gto")
    basis = _basis((6,), pure, contracted=False)
    shell = basis.shells()[0]
    mol = gto.M(atom=[("He", shell.origin)], basis={"He": [[6, [shell.exponents[0], 1.]]]},
                unit="Bohr", cart=not pure, verbose=0)
    position = [.2, .4, -.6]
    with mol.with_rinv_origin(position):
        reference = -1.7*mol.intor("int1e_rinv")
    if not pure:
        reference *= 13/(4*pi)
    actual = _compute(basis, operation="potential", charges=[(1.7, position)])
    np.testing.assert_allclose(actual.values, reference, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("field,plan_field", [
    ("memory_bytes", "modeled_peak_bytes"), ("shell_calls", "shell_calls"),
    ("primitive_quartets", "primitive_pairs"), ("kernel_terms", "kernel_terms"),
])
def test_one_electron_exact_budget_boundary(field, plan_field):
    basis = _basis()
    request = _request("potential", charges=[(1., [0., 0., 0.])])
    plan = manta.plan_one_electron(basis, basis, request)
    budget = manta.ResourceBudget()
    setattr(budget, field, getattr(plan, plan_field))
    manta.compute_one_electron(basis, basis, request, budget)
    setattr(budget, field, getattr(plan, plan_field)-1)
    with pytest.raises(ValueError, match="budget"):
        manta.compute_one_electron(basis, basis, request, budget)


@pytest.mark.parametrize("operation,fields", [
    ("unsupported", {}), ("overlap", {"charges": [(1., [0., 0., 0.])]}),
    ("kinetic", {"origin": [1., 0., 0.]}), ("overlap", {"powers": [1, 0, 0]}),
    ("moment", {"powers": [3, 3, 1]}), ("moment", {"powers": [-1, 0, 0]}),
    ("moment", {"origin": [np.nan, 0., 0.]}),
    ("potential", {"charges": [(np.inf, [0., 0., 0.])]}),
    ("potential", {"charges": [(1., [0., np.nan, 0.])]}),
])
def test_invalid_operator_parameters_refused(operation, fields):
    basis = _basis((0,))
    with pytest.raises(ValueError):
        _compute(basis, operation=operation, **fields)


def test_request_identity_and_basis_label_independence():
    basis = _basis((0,))
    a = _compute(basis, operation="moment", powers=[1, 0, 0])
    b = _compute(basis, operation="moment", powers=[1, 0, 0], origin=[.1, 0., 0.])
    c = _compute(basis, operation="moment", powers=[0, 1, 0])
    assert len({a.receipt.source_identity, b.receipt.source_identity, c.receipt.source_identity}) == 3
    mol = vq.Molecule([vq.Atom(2, s.origin) for s in basis.shells()], 0, 1)
    renamed = vq.BasisSet(mol, basis.shells(), "renamed", True)
    assert _compute(renamed).receipt.source_identity == _compute(basis).receipt.source_identity
    assert _compute(basis, operation="moment", origin=[-0., 0., 0.]).receipt.source_identity == _compute(basis, operation="moment").receipt.source_identity


def test_one_electron_admission_is_independent_of_quartic_jk_cost():
    basis = _basis()
    budget = manta.ResourceBudget()
    budget.shell_calls = basis.nshells**2
    budget.primitive_quartets = (2*basis.nshells)**2
    manta.compute_one_electron(basis, basis, manta.OneElectronRequest(), budget)
    with pytest.raises(ValueError, match="budget"):
        manta.DirectJKBuilder(basis, budget=budget)


@pytest.mark.parametrize("alpha,beta", [(1e-8, 1e8), (1e8, 1e-8), (1e-6, 2e-6), (3e4, 2e4), (.7, 1.1)])
def test_s_matrices_against_analytic_diffuse_tight_formulas(alpha, beta):
    bases = []
    for exponent, position in [(alpha, [0., 0., 0.]), (beta, [.2, -.1, .3])]:
        mol = vq.Molecule([vq.Atom(2, position)], 0, 1)
        shell = ShellInfo()
        shell.atom_index, shell.l, shell.pure, shell.origin = 0, 0, False, position
        shell.exponents, shell.coefficients = [exponent], [1.]
        bases.append(vq.BasisSet(mol, [shell], "analytic-s", False))
    a, b = (basis.shells()[0] for basis in bases)
    p, mu = alpha+beta, alpha*beta/(alpha+beta)
    ab2 = np.sum((np.array(a.origin)-b.origin)**2)
    overlap = a.coefficients[0]*b.coefficients[0]*(pi/p)**1.5*np.exp(-mu*ab2)
    assert _compute(*bases).values[0, 0] == pytest.approx(overlap, rel=2e-13, abs=0.)
    kinetic = mu*(3-2*mu*ab2)*overlap
    assert _compute(*bases, operation="kinetic").values[0, 0] == pytest.approx(kinetic, rel=2e-13, abs=0.)
    center = (alpha*np.array(a.origin)+beta*np.array(b.origin))/p
    for position in [center, center+[.4, -.5, .7]]:
        argument = p*np.sum((center-position)**2)
        f0 = 1. if argument == 0 else .5*sqrt(pi/argument)*erf(sqrt(argument))
        potential = -1.3*2*sqrt(p/pi)*overlap*f0
        actual = _compute(*bases, operation="potential", charges=[(1.3, position.tolist())]).values[0, 0]
        assert actual == pytest.approx(potential, rel=2e-13, abs=0.)


def test_count_admission_precedes_charge_payload_scan():
    basis = _basis((0,))
    request = _request("potential", charges=[(np.nan, [0., 0., 0.])]*100)
    budget = manta.ResourceBudget()
    budget.kernel_terms = 1
    with pytest.raises(ValueError, match="kernel-work budget"):
        manta.compute_one_electron(basis, basis, request, budget)
    with pytest.raises(ValueError, match="finite"):
        manta.compute_one_electron(basis, basis, request)


def test_output_and_live_storage_admission():
    basis = _basis()
    request = manta.OneElectronRequest()
    plan = manta.plan_one_electron(basis, basis, request)
    budget = manta.ResourceBudget()
    budget.scalar_terms = basis.nbasis**2-1
    with pytest.raises(ValueError, match="element budget"):
        manta.compute_one_electron(basis, basis, request, budget)
    budget.scalar_terms += 1
    budget.memory_bytes = plan.modeled_peak_bytes
    budget.other_live_bytes = 1
    with pytest.raises(ValueError, match="memory budget"):
        manta.compute_one_electron(basis, basis, request, budget)


def test_one_electron_orbital_limit_and_basis_memory():
    with pytest.raises(ValueError, match="angular capability"):
        _compute(_basis((7,), contracted=False))
    basis = _basis((0,))
    mol = vq.Molecule([vq.Atom(2, s.origin) for s in basis.shells()], 0, 1)
    large_label = vq.BasisSet(mol, basis.shells(), "n"*65536, True)
    budget = manta.ResourceBudget()
    budget.memory_bytes = manta.plan_one_electron(basis, basis).modeled_peak_bytes
    with pytest.raises(ValueError, match="memory budget"):
        manta.compute_one_electron(large_label, basis, budget=budget)


def _derivative(target="bra", axis=0, charge_index=0):
    derivative = manta.OneElectronDerivative()
    derivative.target, derivative.axis, derivative.charge_index = target, axis, charge_index
    return derivative


def _shift_basis(basis, axis, amount):
    shells = basis.shells()
    for shell in shells:
        position = list(shell.origin); position[axis] += amount; shell.origin = position
    mol = vq.Molecule([vq.Atom(2, s.origin) for s in shells], 0, 1)
    return vq.BasisSet(mol, shells, "translated-basis", True)


@pytest.mark.parametrize("operation", ["overlap", "kinetic", "potential", "moment"])
@pytest.mark.parametrize("pure", [False, True])
def test_one_electron_derivatives_against_five_point_differences_and_translation(operation, pure):
    bra, ket = _basis((0, 2), pure), _basis((1, 3), not pure, shift=(.3, -.2, .4))
    fields = {"charges": [(1.7, [.2, -.4, .6]), (-.3, [.7, .1, -.5])]} if operation == "potential" else {}
    if operation == "moment":
        fields = {"origin": [.2, .1, -.3], "powers": [2, 1, 3]}
    request = _request(operation, **fields)
    selectors = [("bra", 0), ("ket", 0)]
    if operation == "potential":
        selectors += [("charge", 0), ("charge", 1)]
    if operation == "moment":
        selectors += [("origin", 0)]
    sources = set()
    for axis in range(3):
        derivatives = []
        for target, index in selectors:
            derivative = _derivative(target, axis, index)
            result = manta.compute_one_electron_derivative(bra, ket, request, derivative)
            shifted = []
            step = 2e-4
            for multiplier in [1, -1, 2, -2]:
                a, b, changed = bra, ket, _request(operation, **fields)
                h = step*multiplier
                if target == "bra":
                    a = _shift_basis(a, axis, h)
                elif target == "ket":
                    b = _shift_basis(b, axis, h)
                elif target == "origin":
                    origin = list(changed.origin); origin[axis] += h; changed.origin = origin
                else:
                    charges = changed.charges
                    value, position = charges[index]; position[axis] += h
                    charges[index] = (value, position); changed.charges = charges
                shifted.append(manta.compute_one_electron(a, b, changed).values)
            expected = (8*(shifted[0]-shifted[1])-(shifted[2]-shifted[3]))/(12*step)
            np.testing.assert_allclose(result.values, expected, atol=2e-9, rtol=2e-8)
            assert result.receipt.derivative_target == target and result.receipt.derivative_axis == axis
            assert result.receipt.derivative_charge_index == index
            assert result.receipt.primitive_pairs == 16
            assert result.receipt.primitive_integrals_evaluated == {"bra": 24, "ket": 32}.get(target, 16)
            reverse_target = {"bra": "ket", "ket": "bra"}.get(target, target)
            reverse = manta.compute_one_electron_derivative(ket, bra, request, _derivative(reverse_target, axis, index))
            np.testing.assert_allclose(result.values, reverse.values.T, atol=4e-12, rtol=4e-12)
            sources.add(result.receipt.source_identity)
            derivatives.append(result.values.copy())
        np.testing.assert_allclose(sum(derivatives), 0., atol=5e-12, rtol=0)
    assert len(sources) == 3*len(selectors)
    assert manta.compute_one_electron(bra, ket, request).receipt.source_identity not in sources


@pytest.mark.parametrize("l", [0, 1, 3, 6])
@pytest.mark.parametrize("pure", [False, True])
@pytest.mark.parametrize("operation", ["overlap", "kinetic", "potential"])
def test_one_electron_derivatives_against_independent_libcint(l, pure, operation):
    gto = pytest.importorskip("pyscf.gto")
    bra = _basis((l,), pure, contracted=False)
    ket = _basis((l,), pure, shift=(.3, -.4, .7), contracted=False)
    shells = [bra.shells()[0], ket.shells()[0]]
    mol = gto.M(atom=[(f"He{i}", s.origin) for i, s in enumerate(shells)],
                basis={f"He{i}": [[l, [s.exponents[0], 1.]]] for i, s in enumerate(shells)},
                unit="Bohr", cart=not pure, verbose=0)
    position, charge = [.2, -.4, .6], 1.7
    fields = {"charges": [(charge, position)]} if operation == "potential" else {}
    request = _request(operation, **fields)
    with mol.with_rinv_origin(position):
        name = {"overlap": "int1e_ipovlp", "kinetic": "int1e_ipkin", "potential": "int1e_iprinv"}[operation]
        expected = mol.intor_by_shell(name+("_sph" if pure else "_cart"), (0, 1))
        charge_expected = None
        if operation == "potential":
            reverse = mol.intor_by_shell(name+("_sph" if pure else "_cart"), (1, 0))
            charge_expected = -charge*(expected+reverse.transpose(0, 2, 1))
        expected *= charge if operation == "potential" else -1
    def convert(value):
        if pure and l == 1:
            return value[:, [1, 2, 0], :][:, :, [1, 2, 0]]
        return value*(2*l+1)/(4*pi) if not pure and l >= 2 else value
    expected = convert(expected)
    if charge_expected is not None:
        charge_expected = convert(charge_expected)
    for axis in range(3):
        result = manta.compute_one_electron_derivative(bra, ket, request, _derivative(axis=axis))
        np.testing.assert_allclose(result.values, expected[axis], atol=3e-11, rtol=3e-11)
        if charge_expected is not None:
            result = manta.compute_one_electron_derivative(bra, ket, request, _derivative("charge", axis))
            np.testing.assert_allclose(result.values, charge_expected[axis], atol=3e-11, rtol=3e-11)


@pytest.mark.parametrize("target", ["bra", "ket", "charge", "origin"])
@pytest.mark.parametrize("field,plan_field", [
    ("memory_bytes", "modeled_peak_bytes"), ("shell_calls", "shell_calls"),
    ("primitive_quartets", "primitive_evaluations_upper_bound"), ("kernel_terms", "kernel_terms"),
])
def test_one_electron_derivative_exact_budget_boundary(target, field, plan_field):
    basis = _basis()
    request = (_request("moment", powers=[2, 1, 3]) if target == "origin" else
               _request("potential", charges=[(1., [.2, .3, .4])]))
    derivative = _derivative(target)
    plan = manta.plan_one_electron_derivative(basis, basis, request, derivative)
    budget = manta.ResourceBudget()
    setattr(budget, field, getattr(plan, plan_field))
    result = manta.compute_one_electron_derivative(basis, basis, request, derivative, budget)
    assert result.receipt.primitive_integrals_evaluated <= plan.primitive_evaluations_upper_bound
    setattr(budget, field, getattr(plan, plan_field)-1)
    with pytest.raises(ValueError, match="budget"):
        manta.compute_one_electron_derivative(basis, basis, request, derivative, budget)


@pytest.mark.parametrize("target,axis,index,operation", [
    ("nucleus", 0, 0, "potential"), ("bra", -1, 0, "overlap"), ("bra", 3, 0, "overlap"),
    ("bra", 0, 1, "overlap"), ("charge", 0, 0, "overlap"),
    ("charge", 0, 2, "potential"), ("origin", 0, 0, "kinetic"),
])
def test_malformed_one_electron_derivative_refused(target, axis, index, operation):
    basis = _basis((0,))
    fields = {"charges": [(1., [0., 0., 0.])]} if operation == "potential" else {}
    with pytest.raises(ValueError):
        manta.compute_one_electron_derivative(basis, basis, _request(operation, **fields), _derivative(target, axis, index))


def test_derivative_zero_moment_origin_ownership_and_count_admission():
    basis = _basis((2,))
    result = manta.compute_one_electron_derivative(basis, basis, _request("moment"), _derivative("origin"))
    np.testing.assert_array_equal(result.values, 0.)
    assert result.receipt.primitive_integrals_evaluated == 0
    request = _request("potential", charges=[(np.nan, [0., 0., 0.])])
    budget = manta.ResourceBudget(); budget.kernel_terms = 1
    with pytest.raises(ValueError, match="kernel-work budget"):
        manta.compute_one_electron_derivative(basis, basis, request, budget=budget)
    with pytest.raises(ValueError, match="finite"):
        manta.compute_one_electron_derivative(basis, basis, request)
    request = _request("moment", powers=[2, 1, 3])
    old = manta.compute_one_electron_derivative(basis, basis, request)
    before = old.values.copy()
    manta.compute_one_electron_derivative(basis, basis, request, _derivative("ket", 2))
    np.testing.assert_array_equal(old.values, before)
    assert manta.compute_one_electron(basis, basis, request).receipt.derivative_target == "none"


@pytest.mark.parametrize("pure", [False, True])
def test_l6_moment_derivatives_by_independent_polynomial_quadrature(pure):
    bra = _basis((6,), pure, contracted=False)
    ket = _basis((6,), pure, shift=(.3, -.4, .7), contracted=False)
    a, b = bra.shells()[0], ket.shells()[0]
    alpha, beta = a.exponents[0], b.exponents[0]
    powers, origin = np.array([2, 1, 3]), np.array([.2, -.5, .3])
    nodes, weights = np.polynomial.hermite.hermgauss(12)
    grid = np.array(list(product(nodes, repeat=3)))
    weight = np.prod(np.array(list(product(weights, repeat=3))), axis=1)
    p = alpha+beta
    xyz = grid/np.sqrt(p)+(alpha*np.array(a.origin)+beta*np.array(b.origin))/p
    xa, xb = xyz-a.origin, xyz-b.origin
    ca, cb = _cart(a.l), _cart(b.l)
    pa, pb = (np.prod(x[None, :, :]**c[:, None, :], axis=2) for x, c in [(xa, ca), (xb, cb)])
    multiplier = np.prod((xyz-origin)**powers, axis=1)
    factor = a.coefficients[0]*b.coefficients[0]*np.exp(-alpha*beta/p*np.sum((np.array(a.origin)-b.origin)**2))/p**1.5
    request = _request("moment", origin=origin.tolist(), powers=powers.tolist())
    for axis in range(3):
        da, db = ca.copy(), cb.copy()
        da[:, axis] = np.maximum(0, da[:, axis]-1)
        db[:, axis] = np.maximum(0, db[:, axis]-1)
        ga = 2*alpha*xa[None, :, axis]*pa-ca[:, axis, None]*np.prod(xa[None, :, :]**da[:, None, :], axis=2)
        gb = 2*beta*xb[None, :, axis]*pb-cb[:, axis, None]*np.prod(xb[None, :, :]**db[:, None, :], axis=2)
        lower = powers.copy(); lower[axis] -= 1
        origin_multiplier = -powers[axis]*np.prod((xyz-origin)**lower, axis=1)
        for target, expected in [("bra", (ga*(weight*multiplier))@pb.T),
                                 ("ket", (pa*(weight*multiplier))@gb.T),
                                 ("origin", (pa*(weight*origin_multiplier))@pb.T)]:
            expected *= factor
            if pure:
                expected = manta.cartesian_to_spherical(a.l)@expected@manta.cartesian_to_spherical(b.l).T
            actual = manta.compute_one_electron_derivative(bra, ket, request, _derivative(target, axis))
            np.testing.assert_allclose(actual.values, expected, atol=4e-11, rtol=4e-11)
