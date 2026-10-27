"""Provider capability inventory by kernel, center count, derivative and axis role.

Every supported cell of the capability query is exercised at its angular
limit and one order above it against the actual plan admission, including
the kernel-free producers (Fourier panels, moments, extents) and the
consumers that census their own axes (BUPO fitting coupling, RI Coulomb
gradient). Unsupported requests refuse before any basis census or budget
evaluation. Deep signed
contractions at the orbital l=6 and auxiliary l=7 limits are witnessed by
exact contraction linearity over single-primitive tiles and by the optional
libcint reference; a zero-charge center carrying shells runs through the
selected molecular HF adapter against the libint route.
"""
from itertools import product
from math import fsum, pi

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta
from vibeqc._vibeqc_core import ShellInfo

ORBITAL_LIMIT, AUXILIARY_LIMIT = 6, 7
KERNELS = ["coulomb", "erf", "erfc"]
DEEP_EXPONENTS = [.11, .3, .8, 2.1, 5.5, 14.]
DEEP_COEFFICIENTS = [.9, -.75, .6, -.45, .3, -.15]
POSITIONS = np.array([[0., .1, -.2], [.4, -.3, 1.1], [-.5, .6, .7], [.9, .2, -.4]])


def _shell(l, origin, exponents, coefficients, pure, atom=0):
    shell = ShellInfo()
    shell.atom_index, shell.l, shell.pure, shell.origin = atom, l, pure, list(origin)
    shell.exponents, shell.coefficients = list(exponents), list(coefficients)
    return shell


def _single(l, pure=True, deep=False, index=0):
    """One shell on its own center: deep signed or a two-term signed contraction."""
    origin = POSITIONS[index]
    exponents = DEEP_EXPONENTS if deep else [.7+.13*index, 1.6+.1*index]
    coefficients = DEEP_COEFFICIENTS if deep else [.8, -.18]
    mol = vq.Molecule([vq.Atom(2, origin.tolist())], 0, 1)
    return vq.BasisSet(mol, [_shell(l, origin, exponents, coefficients, pure)], "manta-inventory", False)


def _merged(bases):
    shells = [b.shells()[0] for b in bases]
    for i, shell in enumerate(shells):
        shell.atom_index = i
    mol = vq.Molecule([vq.Atom(2, s.origin) for s in shells], 0, 1)
    return vq.BasisSet(mol, shells, "manta-inventory-merged", True)


def _request(kernel="coulomb", centers=4, derivative=0, omega=None):
    request = manta.KernelRequest()
    request.kernel, request.centers, request.derivative = kernel, centers, derivative
    request.omega = (0. if kernel == "coulomb" else .55) if omega is None else omega
    return request


def _roles(centers):
    if centers == 2:
        return ["auxiliary", "auxiliary"]
    if centers == 3:
        return ["auxiliary", "orbital", "orbital"]
    return ["orbital"]*4


def _generous():
    """Work and memory caps far above any l=6/l=7 plan, so only angular admission decides."""
    budget = manta.ResourceBudget()
    budget.memory_bytes = 8 << 30
    budget.kernel_terms = budget.scalar_terms = budget.primitive_quartets = 1 << 60
    return budget


def _admit(bases, request, budget=None):
    budget = _generous() if budget is None else budget
    if request.derivative == 1:
        return manta.DerivativeTiles(bases, budget, request)
    if request.centers == 4:
        return manta.md_plan(_merged(bases), request, budget)
    return manta.FittingTiles(bases, budget, request)


@pytest.mark.parametrize("pure", [False, True])
@pytest.mark.parametrize("derivative", [0, 1])
@pytest.mark.parametrize("centers", [2, 3, 4])
@pytest.mark.parametrize("kernel", KERNELS)
def test_md_capability_limits_match_plan_admission_by_axis_role(kernel, centers, derivative, pure):
    request = _request(kernel, centers, derivative)
    capability = manta.md_capability(request)
    assert capability.supported and not capability.reason
    assert (capability.maximum_l, capability.maximum_auxiliary_l) == (
        AUXILIARY_LIMIT if centers == 2 else ORBITAL_LIMIT,
        ORBITAL_LIMIT if centers == 4 else AUXILIARY_LIMIT)
    roles = _roles(centers)
    limits = [capability.maximum_auxiliary_l if role == "auxiliary" else capability.maximum_l for role in roles]
    assert limits == [AUXILIARY_LIMIT if role == "auxiliary" else ORBITAL_LIMIT for role in roles]
    _admit([_single(l, pure, index=i) for i, l in enumerate(limits)], request)
    for axis in range(centers):
        raised = list(limits)
        raised[axis] += 1
        with pytest.raises(ValueError, match="angular capability"):
            _admit([_single(l, pure, index=i) for i, l in enumerate(raised)], request)


def test_reference_capability_matches_plan_admission():
    request = _request()
    capability = manta.capability(request)
    assert capability.supported and not capability.reason
    assert 0 < capability.maximum_l <= ORBITAL_LIMIT
    assert capability.maximum_auxiliary_l == capability.maximum_l
    for pure in [False, True]:
        manta.plan(_merged([_single(capability.maximum_l, pure, index=i) for i in range(4)]), request)
        with pytest.raises(ValueError, match="exceeds provider capability"):
            manta.plan(_merged([_single(capability.maximum_l+1, pure, index=0)]+[_single(0, pure, index=i) for i in range(1, 4)]), request)
    for centers, derivative in [(2, 0), (3, 0), (4, 1)]:
        refused = manta.capability(_request("coulomb", centers, derivative))
        assert not refused.supported and "4c" in refused.reason
    for kernel in ["erf", "erfc"]:
        assert manta.capability(_request(kernel)).supported
        refused = manta.capability(_request(kernel, omega=-.5))
        assert not refused.supported and "finite and nonnegative" in refused.reason
    refused = manta.capability(_request("coulomb", omega=.5))
    assert not refused.supported and "zero for full Coulomb" in refused.reason
    with pytest.raises(ValueError, match="zero for full Coulomb"):
        manta.plan(_merged([_single(0, index=i) for i in range(4)]), _request("coulomb", omega=.5))


@pytest.mark.parametrize("centers", [2, 3, 4])
@pytest.mark.parametrize("field,value,fragment", [
    ("scalar", "complex", "real scalars"),
    ("kernel", "yukawa", "coulomb, erf or erfc"),
    ("omega", -1., "finite and nonnegative"),
    ("omega", float("nan"), "finite and nonnegative"),
    ("images", [[1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], "periodic image"),
])
def test_unsupported_md_requests_refuse_before_setup(centers, field, value, fragment):
    for derivative in [0, 1]:
        request = _request("erf" if field == "omega" else "coulomb", centers, derivative)
        setattr(request, field, value)
        capability = manta.md_capability(request)
        assert not capability.supported and fragment in capability.reason
        starved = manta.ResourceBudget()
        starved.memory_bytes = starved.shell_calls = starved.kernel_terms = 1
        with pytest.raises(ValueError, match=fragment):
            _admit([_single(0, index=i) for i in range(centers)], request, starved)


def test_coulomb_with_range_and_unadmitted_orders_refuse():
    request = _request("coulomb", omega=.3)
    capability = manta.md_capability(request)
    assert not capability.supported and "zero for full Coulomb" in capability.reason
    with pytest.raises(ValueError, match="zero for full Coulomb"):
        manta.md_plan(_merged([_single(0, index=i) for i in range(4)]), request)
    for centers in [0, 1, 5]:
        capability = manta.md_capability(_request(centers=centers))
        assert not capability.supported and "two-, three- and four-center" in capability.reason
    # The query answers fitting and derivative cells; the four-center builder
    # itself still refuses anything but zeroth-order four-center requests.
    for centers, derivative in [(2, 0), (3, 0), (4, 1), (2, 1)]:
        assert manta.md_capability(_request(centers=centers, derivative=derivative)).supported
        with pytest.raises(ValueError, match="four-center builder"):
            manta.md_plan(_merged([_single(0, index=i) for i in range(4)]), _request(centers=centers, derivative=derivative))
        with pytest.raises(ValueError, match="four-center builder"):
            manta.DirectJKBuilder(_merged([_single(0, index=i) for i in range(4)]), _request(centers=centers, derivative=derivative))
    for arity, centers in [(2, 3), (3, 2), (2, 4)]:
        with pytest.raises(ValueError, match="center count must match"):
            manta.FittingTiles([_single(0, index=i) for i in range(arity)], manta.ResourceBudget(), _request(centers=centers))
        with pytest.raises(ValueError, match="center count must match"):
            manta.DerivativeTiles([_single(0, index=i) for i in range(arity)], manta.ResourceBudget(), _request(centers=centers, derivative=1))
    for derivative in [-1, 2]:
        capability = manta.md_capability(_request(derivative=derivative))
        assert not capability.supported and "zero and one" in capability.reason
        with pytest.raises(ValueError):
            manta.md_plan(_merged([_single(0, index=i) for i in range(4)]), _request(derivative=derivative))
        with pytest.raises(ValueError):
            manta.DerivativeTiles([_single(0), _single(0, index=1)], manta.ResourceBudget(), _request(centers=2, derivative=derivative))


def _one_electron(operation, **fields):
    request = manta.OneElectronRequest()
    request.operation = operation
    for name, value in fields.items():
        setattr(request, name, value)
    return request


@pytest.mark.parametrize("operation", ["overlap", "kinetic", "potential", "moment"])
@pytest.mark.parametrize("pure", [False, True])
def test_one_electron_capability_matches_plan_admission(operation, pure):
    fields = {"charges": [(1.5, [.2, -.4, .6])]} if operation == "potential" else {}
    if operation == "moment":
        fields = {"origin": [.1, -.2, .3], "powers": [2, 1, 3]}
    request = _one_electron(operation, **fields)
    capability = manta.one_electron_capability(request)
    assert capability.supported and not capability.reason
    assert capability.maximum_l == capability.maximum_auxiliary_l == ORBITAL_LIMIT
    bra, ket = _single(ORBITAL_LIMIT, pure), _single(ORBITAL_LIMIT, not pure, index=1)
    manta.plan_one_electron(bra, ket, request)
    for over in [(_single(ORBITAL_LIMIT+1, pure), ket), (bra, _single(ORBITAL_LIMIT+1, pure, index=1))]:
        with pytest.raises(ValueError, match="angular capability"):
            manta.plan_one_electron(*over, request)
    if operation == "potential":
        for kernel in ["erf", "erfc"]:
            assert manta.one_electron_capability(_one_electron(operation, kernel=kernel, omega=.4, **fields)).supported


@pytest.mark.parametrize("operation,fields,fragment", [
    ("dipole", {}, "overlap, kinetic, potential or moment"),
    ("kinetic", {"kernel": "erf", "omega": .4}, "only to the potential"),
    ("overlap", {"charges": [(1., [0., 0., 0.])]}, "only to the potential"),
    ("overlap", {"powers": [1, 0, 0]}, "only to polynomial moments"),
    ("kinetic", {"origin": [.1, 0., 0.]}, "only to polynomial moments"),
    ("moment", {"powers": [3, 3, 1]}, "exceeds six"),
    ("moment", {"powers": [7, 0, 0]}, "between zero and six"),
    ("potential", {"kernel": "erfc", "omega": -.2}, "finite and nonnegative"),
    ("potential", {"kernel": "coulomb", "omega": .2}, "zero for full Coulomb"),
])
def test_unsupported_one_electron_requests_refuse_before_setup(operation, fields, fragment):
    request = _one_electron(operation, **fields)
    capability = manta.one_electron_capability(request)
    assert not capability.supported and fragment in capability.reason
    starved = manta.ResourceBudget()
    starved.memory_bytes = starved.shell_calls = 1
    with pytest.raises(ValueError, match=fragment):
        manta.plan_one_electron(_single(0), _single(0, index=1), request, starved)


def _slices(basis):
    offsets = [0]
    for shell in basis.shells():
        offsets.append(offsets[-1]+(2*shell.l+1 if shell.pure else (shell.l+1)*(shell.l+2)//2))
    return [slice(a, b) for a, b in zip(offsets[:-1], offsets[1:], strict=True)]


def _primitive_expansion(bases):
    """Single-primitive shells with the effective libint coefficients taken as given."""
    expanded, weights = [], []
    for basis in bases:
        shell = basis.shells()[0]
        members = []
        for exponent, coefficient in zip(shell.exponents, shell.coefficients, strict=True):
            mol = vq.Molecule([vq.Atom(2, shell.origin)], 0, 1)
            members.append(vq.BasisSet(mol, [_shell(shell.l, shell.origin, [exponent], [1.], shell.pure)], "manta-primitive", True))
        expanded.append(members)
        weights.append(list(shell.coefficients))
    return expanded, weights


def _linear_witness(bases, request):
    """Contraction linearity: sum of weighted single-primitive tiles, exactly rounded per element."""
    expanded, weights = _primitive_expansion(bases)
    terms = []
    for choice in product(*(range(len(members)) for members in expanded)):
        selected = [expanded[axis][index] for axis, index in enumerate(choice)]
        weight = np.prod([weights[axis][index] for axis, index in enumerate(choice)])
        if request.centers == 4:
            tile = manta.DirectJKBuilder(_merged(selected), request, _generous()).evaluate((0, 1, 2, 3))["values"]
        else:
            tile = manta.FittingTiles(selected, _generous(), request).evaluate((0,)*len(selected))["values"]
        terms.append(weight*np.asarray(tile, dtype=np.float64))
    stacked = np.stack(terms)
    flat = np.array([fsum(column) for column in stacked.reshape(len(terms), -1).T])
    return flat.reshape(stacked.shape[1:]), np.abs(stacked).sum(axis=0)


def _libcint_witness(bases, pure, arity):
    gto = pytest.importorskip("pyscf.gto")  # optional reference, never a runtime backend
    shells = [b.shells()[0] for b in bases]
    raw = [list(zip(DEEP_EXPONENTS, DEEP_COEFFICIENTS, strict=True)) if len(s.exponents) == len(DEEP_EXPONENTS)
           else [(.7+.13*i, .8), (1.6+.1*i, -.18)] for i, s in enumerate(shells)]
    mol = gto.M(atom=[(f"He{i}", s.origin) for i, s in enumerate(shells)],
                basis={f"He{i}": [[s.l]+[[e, c] for e, c in raw[i]]] for i, s in enumerate(shells)},
                unit="Bohr", cart=not pure, verbose=0)
    if arity == 4:
        reference = mol.intor_by_shell("int2e_sph" if pure else "int2e_cart", (0, 1, 2, 3))
    else:
        name = "int2c2e" if arity == 2 else "int3c2e"
        indices = (0, 1) if arity == 2 else (1, 2, 0)
        reference = mol.intor_by_shell(name+("_sph" if pure else "_cart"), indices)
        if arity == 3:
            reference = reference.transpose(2, 0, 1)
    for axis, shell in enumerate(shells):
        if pure and shell.l == 1:
            reference = np.take(reference, [1, 2, 0], axis=axis)
        elif not pure and shell.l >= 2:
            reference *= np.sqrt((2*shell.l+1)/(4*pi))
    return reference


@pytest.mark.parametrize("angular,deep", [
    ((6, 0, 6, 0), (True, False, True, False)),
    ((6, 1, 2, 6), (True, True, False, True)),
    ((7, 7), (True, True)),
    ((7, 6, 1), (True, True, True)),
    ((3, 6, 0), (False, True, True)),
])
@pytest.mark.parametrize("pure", [False, True])
def test_deep_signed_contractions_at_angular_limits(angular, deep, pure):
    bases = [_single(l, pure, deep=d, index=i) for i, (l, d) in enumerate(zip(angular, deep, strict=True))]
    request = _request(centers=len(angular))
    if len(angular) == 4:
        actual = manta.DirectJKBuilder(_merged(bases), request, _generous()).evaluate((0, 1, 2, 3))["values"]
    else:
        actual = manta.FittingTiles(bases, _generous(), request).evaluate((0,)*len(bases))["values"]
    actual = np.asarray(actual, dtype=np.float64)
    witness, magnitude = _linear_witness(bases, request)
    assert witness.shape == actual.shape
    # Signed deep contractions cancel: the term-magnitude sum exceeds the result.
    assert magnitude.max() > np.abs(witness).max()
    np.testing.assert_allclose(actual, witness, atol=1e-11, rtol=5e-12)
    reference = _libcint_witness(bases, pure, len(angular))
    np.testing.assert_allclose(actual, reference, atol=1e-11+1e-13*magnitude.max(), rtol=5e-12)


@pytest.mark.parametrize("kernel", KERNELS)
def test_deep_signed_contracted_derivative_tiles_at_auxiliary_limit(kernel):
    bases = [_single(AUXILIARY_LIMIT, True, deep=True, index=i) for i in range(2)]
    request = _request(kernel, 2, 1)
    provider = manta.DerivativeTiles(bases, manta.ResourceBudget(), request)
    expanded, weights = _primitive_expansion(bases)
    for center, axis in [(0, 0), (1, 2)]:
        actual = np.asarray(provider.evaluate((0, 0), center, axis)["values"], dtype=np.float64)
        terms = []
        for i, j in product(range(len(expanded[0])), range(len(expanded[1]))):
            single = manta.DerivativeTiles([expanded[0][i], expanded[1][j]], manta.ResourceBudget(), request)
            terms.append(weights[0][i]*weights[1][j]*np.asarray(single.evaluate((0, 0), center, axis)["values"], dtype=np.float64))
        stacked = np.stack(terms)
        witness = np.array([fsum(column) for column in stacked.reshape(len(terms), -1).T]).reshape(actual.shape)
        np.testing.assert_allclose(actual, witness, atol=1e-11, rtol=5e-12)


def test_ghost_center_shells_through_selected_molecular_hf():
    hydrogen = vq.Molecule([vq.Atom(1, [0., 0., 0.]), vq.Atom(1, [0., 0., 1.4])], 0, 1)
    shells = vq.BasisSet(hydrogen, "sto-3g").shells()
    ghost_origin = [.6, -.3, .7]
    shells.append(_shell(0, ghost_origin, [.3, 1.1], [.7, .4], True, atom=2))
    shells.append(_shell(1, ghost_origin, [.5], [1.], True, atom=2))
    mol = vq.Molecule([vq.Atom(1, [0., 0., 0.]), vq.Atom(1, [0., 0., 1.4]), vq.Atom(0, ghost_origin)], 0, 1)
    basis = vq.BasisSet(mol, shells, "manta-ghost", True)
    assert mol.n_electrons() == 2 and basis.nbasis == 6
    options = vq.RHFOptions()
    options.initial_guess = vq.InitialGuess.HCORE
    options.conv_tol_energy, options.conv_tol_grad = 1e-12, 1e-10
    options.stability_check = False
    selected = vq.run_rhf(mol, basis, options, integral_engine=vq.IntegralEngineOptions(backend="manta"))
    legacy = vq.run_rhf(mol, basis, options)
    assert selected.converged and legacy.converged
    assert selected.energy == pytest.approx(legacy.energy, abs=3e-10)
    np.testing.assert_allclose(selected.density, legacy.density, atol=3e-9, rtol=3e-9)
    assert manta.molecular_engine_provenance(selected)["one_electron_provider"] == "manta"
    # The zero charge contributes nothing: the potential with and without it agrees exactly.
    with_ghost = _one_electron("potential", charges=[(float(a.Z), list(a.xyz)) for a in mol.atoms])
    without = _one_electron("potential", charges=[(float(a.Z), list(a.xyz)) for a in mol.atoms if a.Z])
    np.testing.assert_array_equal(manta.compute_one_electron(basis, basis, with_ghost).values,
                                  manta.compute_one_electron(basis, basis, without).values)


def _extent_options(order):
    options = manta.ExtentOptions()
    options.moment_order = order
    return options


def _producer(producer, order=0):
    request = manta.ProducerRequest()
    request.producer, request.order = producer, order
    return request


PRODUCERS = {  # producer: (axis role of its basis, planner at the given order)
    "fourier": ("orbital", lambda basis, order: manta.plan_fourier(basis, basis, 2, _generous())),
    "auxiliary-fourier": ("auxiliary", lambda basis, order: manta.plan_auxiliary_fourier(basis, 2, _generous())),
    "pair-moments": ("orbital", lambda basis, order: manta.plan_pair_moments(basis, [0, 0], order, _generous())),
    "auxiliary-moments": ("auxiliary", lambda basis, order: manta.plan_auxiliary_moments(basis, 0, order, _generous())),
    "pair-extents": ("orbital", lambda basis, order: manta.plan_pair_extents(basis, _extent_options(order), _generous())),
    "auxiliary-extents": ("auxiliary",
                          lambda basis, order: manta.plan_auxiliary_extents(basis, _extent_options(order), _generous())),
}


@pytest.mark.parametrize("pure", [False, True])
@pytest.mark.parametrize("producer", sorted(PRODUCERS))
def test_producer_capability_limits_match_plan_admission(producer, pure):
    role, admit = PRODUCERS[producer]
    order = 0 if "fourier" in producer else 12
    capability = manta.md_producer_capability(_producer(producer, order))
    assert capability.supported and not capability.reason
    assert (capability.maximum_l, capability.maximum_auxiliary_l) == (ORBITAL_LIMIT, AUXILIARY_LIMIT)
    limit = capability.maximum_auxiliary_l if role == "auxiliary" else capability.maximum_l
    admit(_single(limit, pure), order)
    with pytest.raises(ValueError, match="angular capability"):
        admit(_single(limit+1, pure), order)


def test_producer_capability_agrees_with_the_kernel_query():
    orbital, auxiliary = manta.md_capability(_request(centers=4)), manta.md_capability(_request(centers=2))
    for producer in PRODUCERS:
        capability = manta.md_producer_capability(_producer(producer))
        assert (capability.maximum_l, capability.maximum_auxiliary_l) == (orbital.maximum_l, auxiliary.maximum_auxiliary_l)


@pytest.mark.parametrize("producer,order,fragment", [
    ("fourier", 1, "no moment order"),
    ("auxiliary-fourier", -1, "no moment order"),
    ("pair-moments", 13, "pair moment order"),
    ("auxiliary-moments", -1, "auxiliary moment order"),
    ("pair-extents", 13, "extent moment order"),
    ("auxiliary-extents", -1, "extent moment order"),
    ("moments", 0, "producer must be"),
])
def test_unsupported_producer_requests_refuse_before_setup(producer, order, fragment):
    capability = manta.md_producer_capability(_producer(producer, order))
    assert not capability.supported and fragment in capability.reason
    assert (capability.maximum_l, capability.maximum_auxiliary_l) == (ORBITAL_LIMIT, AUXILIARY_LIMIT)
    if producer in PRODUCERS and "fourier" not in producer:
        with pytest.raises(ValueError, match="order"):
            PRODUCERS[producer][1](_single(0), order)


def test_consumers_take_their_axis_limits_from_the_kernel_query():
    """BUPO fitting coupling and the RI Coulomb gradient census their axes with the three-center limits."""
    capability = manta.md_capability(_request(centers=3))
    molecule = vq.Molecule([vq.Atom(2, POSITIONS[0].tolist())], 0, 1)
    orbital = _single(capability.maximum_l)
    for raised in (0, 1):
        auxiliary = _single(capability.maximum_auxiliary_l+raised)
        if raised:
            with pytest.raises(ValueError, match="angular capability"):
                manta.BupoFittingCoupling(orbital, auxiliary, manta.BupoFittingOptions(), _generous())
            with pytest.raises(ValueError, match="angular capability"):
                manta.plan_ri_coulomb_gradient(molecule, orbital, auxiliary, manta.RIOptions(), _request(), _generous())
        else:
            coupling = manta.BupoFittingCoupling(orbital, auxiliary, manta.BupoFittingOptions(), _generous())
            assert coupling.plan.n_auxiliary == auxiliary.nbasis
            plan = manta.plan_ri_coulomb_gradient(molecule, orbital, auxiliary, manta.RIOptions(), _request(), _generous())
            assert plan.n_auxiliary == auxiliary.nbasis
