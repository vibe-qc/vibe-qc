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
