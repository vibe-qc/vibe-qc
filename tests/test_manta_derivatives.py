"""Analytic shell-center derivatives, independently checked before force assembly."""
from math import pi
from itertools import product

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta
from vibeqc._vibeqc_core import ShellInfo


def _bases(angular, pure=True, contracted=True):
    positions = [[.1, -.2, .3], [.4, .2, -.1], [-.2, .5, .7], [.8, -.3, .4]]
    bases = []
    for i, l in enumerate(angular):
        mol = vq.Molecule([vq.Atom(2, positions[i])], 0, 1)
        shell = ShellInfo()
        shell.atom_index, shell.l, shell.pure, shell.origin = 0, l, pure, positions[i]
        shell.exponents = [.7+.1*i, 1.6+.1*i] if contracted else [.7+.1*i]
        shell.coefficients = [.8, -.18] if contracted else [1.]
        bases.append(vq.BasisSet(mol, [shell], "manta-derivative", False))
    return bases


def _request(arity, kernel="coulomb", omega=None, derivative=1):
    request = manta.KernelRequest()
    request.centers, request.derivative, request.kernel = arity, derivative, kernel
    request.omega = (0. if kernel == "coulomb" else .6) if omega is None else omega
    return request


def _values(bases, kernel):
    request = _request(len(bases), kernel, derivative=0)
    if len(bases) != 4:
        return manta.FittingTiles(bases, request=request).evaluate((0,)*len(bases))["values"]
    shells = [b.shells()[0] for b in bases]
    mol = vq.Molecule([vq.Atom(2, s.origin) for s in shells], 0, 1)
    for i, shell in enumerate(shells):
        shell.atom_index = i
    merged = vq.BasisSet(mol, shells, "manta-derivative-reference", True)
    return manta.DirectJKBuilder(merged, request).evaluate((0, 1, 2, 3))["values"]


def _shifted(bases, center, axis, step):
    shifted = list(bases)
    shell = bases[center].shells()[0]
    position = list(shell.origin); position[axis] += step; shell.origin = position
    mol = vq.Molecule([vq.Atom(2, position)], 0, 1)
    shifted[center] = vq.BasisSet(mol, [shell], "shifted-derivative-reference", True)
    return shifted


@pytest.mark.parametrize("angular", [(1, 2), (2, 1, 2), (1, 0, 2, 1)])
@pytest.mark.parametrize("pure", [False, True])
@pytest.mark.parametrize("kernel", ["coulomb", "erf", "erfc"])
def test_all_center_components_against_five_point_differences_and_translation(angular, pure, kernel):
    bases = _bases(angular, pure)
    provider = manta.DerivativeTiles(bases, request=_request(len(bases), kernel))
    step = 2e-4
    for axis in range(3):
        derivatives = []
        for center in range(len(bases)):
            tile = provider.evaluate((0,)*len(bases), center, axis)
            plus, minus, plus2, minus2 = [_values(_shifted(bases, center, axis, h*step), kernel)
                                         for h in [1, -1, 2, -2]]
            expected = (8*(plus-minus)-(plus2-minus2))/(12*step)
            np.testing.assert_allclose(tile["values"], expected, atol=4e-9, rtol=3e-8)
            assert tile["derivative_order"] == 1
            assert tile["derivative_center"] == center and tile["derivative_axis"] == axis
            assert tile["primitive_products_requested"] == 2**len(bases)
            assert tile["primitive_integrals_evaluated"] == 2**len(bases)*(2 if angular[center] else 1)
            derivatives.append(tile["values"])
        np.testing.assert_allclose(sum(derivatives), np.zeros_like(derivatives[0]), atol=3e-12, rtol=0)


def _libcint_first_center(bases, kernel, pure):
    gto = pytest.importorskip("pyscf.gto")
    shells = [b.shells()[0] for b in bases]
    mol = gto.M(atom=[(f"He{i}", s.origin) for i, s in enumerate(shells)],
                basis={f"He{i}": [[s.l, [s.exponents[0], 1.]]] for i, s in enumerate(shells)},
                unit="Bohr", cart=not pure, verbose=0)
    arity = len(bases)
    name = {2: "int2c2e_ip1", 3: "int3c2e_ip2", 4: "int2e_ip1"}[arity]
    indices = (1, 2, 0) if arity == 3 else tuple(range(arity))
    omega = 0. if kernel == "coulomb" else .6 if kernel == "erf" else -.6
    with mol.with_range_coulomb(omega):
        # libcint differentiates the orbital in electron coordinates; a
        # physical basis-center derivative has the opposite sign.
        reference = -mol.intor_by_shell(name+("_sph" if pure else "_cart"), indices)
    if arity == 3:
        reference = reference.transpose(0, 3, 1, 2)
    for axis, shell in enumerate(shells, start=1):
        if pure and shell.l == 1:
            reference = np.take(reference, [1, 2, 0], axis=axis)
        elif not pure and shell.l >= 2:
            reference *= np.sqrt((2*shell.l+1)/(4*pi))
    return reference


@pytest.mark.parametrize("angular", [(7, 7), (7, 6, 6), (6, 6, 6, 6)])
@pytest.mark.parametrize("pure", [False, True])
@pytest.mark.parametrize("kernel", ["coulomb", "erf", "erfc"])
def test_high_angular_derivatives_against_independent_libcint(angular, pure, kernel):
    bases = _bases(angular, pure, contracted=False)
    provider = manta.DerivativeTiles(bases, request=_request(len(bases), kernel))
    reference = _libcint_first_center(bases, kernel, pure)
    for axis in range(3):
        actual = provider.evaluate((0,)*len(bases), 0, axis)["values"]
        np.testing.assert_allclose(actual, reference[axis], atol=5e-11, rtol=4e-10)


@pytest.mark.parametrize("arity", [2, 3, 4])
def test_axis_permutations_owned_tiles_and_range_identity(arity):
    angular = [2, 1, 0, 1][:arity]
    bases = _bases(angular)
    full = manta.DerivativeTiles(bases)
    lr = manta.DerivativeTiles(bases, request=_request(arity, "erf"))
    sr = manta.DerivativeTiles(bases, request=_request(arity, "erfc"))
    assert len({p.source_identity for p in [full, lr, sr]}) == 3
    original = full.evaluate((0,)*arity, 0, 2)["values"]
    before = original.copy()
    np.testing.assert_allclose(original, lr.evaluate((0,)*arity, 0, 2)["values"]+sr.evaluate((0,)*arity, 0, 2)["values"], atol=3e-12)
    permutation = [1, 0] if arity == 2 else [0, 2, 1] if arity == 3 else [2, 3, 0, 1]
    reverse = manta.DerivativeTiles([bases[i] for i in permutation])
    changed = reverse.evaluate((0,)*arity, permutation.index(0), 2)["values"]
    np.testing.assert_allclose(changed.transpose(np.argsort(permutation)), original, atol=3e-12)
    full.evaluate((0,)*arity, 1, 1)
    del full, bases
    np.testing.assert_array_equal(original, before)


@pytest.mark.parametrize("field,plan_field", [("memory_bytes", "modeled_peak_bytes"),
    ("shell_calls", "shell_calls"), ("primitive_quartets", "primitive_evaluations_upper_bound"),
    ("scalar_terms", "integral_elements"), ("kernel_terms", "kernel_terms")])
def test_derivative_complete_pass_resource_boundaries(field, plan_field):
    bases = _bases((2, 1, 0))
    plan = manta.plan_derivatives(bases)
    assert plan.components == plan.shell_calls == 9
    assert plan.primitive_products == 72 and plan.primitive_evaluations_upper_bound == 144
    budget = manta.ResourceBudget(); setattr(budget, field, getattr(plan, plan_field))
    manta.DerivativeTiles(bases, budget).evaluate((0, 0, 0), 1, 2)
    setattr(budget, field, getattr(plan, plan_field)-1)
    with pytest.raises(ValueError, match="budget"):
        manta.DerivativeTiles(bases, budget)


@pytest.mark.parametrize("field,value", [("derivative", 0), ("derivative", 2), ("centers", 4),
    ("scalar", "complex"), ("kernel", "unknown"), ("omega", .3)])
def test_invalid_derivative_requests(field, value):
    request = _request(2); setattr(request, field, value)
    with pytest.raises(ValueError):
        manta.DerivativeTiles(_bases((0, 0)), request=request)


@pytest.mark.parametrize("indices,center,axis", [((0,), 0, 0), ((1, 0), 0, 0),
    ((0, 0), -1, 0), ((0, 0), 2, 0), ((0, 0), 0, -1), ((0, 0), 0, 3)])
def test_bad_shell_and_derivative_component_indices(indices, center, axis):
    provider = manta.DerivativeTiles(_bases((0, 0)))
    with pytest.raises((ValueError, IndexError)):
        provider.evaluate(indices, center, axis)


def test_role_specific_angular_envelope_and_temporary_axis_ownership():
    with pytest.raises(ValueError, match="angular"):
        manta.DerivativeTiles(_bases((8, 0)))
    with pytest.raises(ValueError, match="angular"):
        manta.DerivativeTiles(_bases((0, 7, 0)))
    with pytest.raises(ValueError, match="angular"):
        manta.DerivativeTiles(_bases((7, 0, 0, 0)))

    class TemporaryAxes:
        def __len__(self):
            return 4

        def __getitem__(self, index):
            if index >= 4:
                raise IndexError(index)
            return _bases((0, 1, 2, 0))[index]

    provider = manta.DerivativeTiles(TemporaryAxes())
    assert np.isfinite(provider.evaluate((0, 0, 0, 0), 3, 0)["values"]).all()


def test_fixed_density_two_electron_energy_derivative_sums_every_moving_slot():
    positions = np.array([[.1, -.2, .3], [.4, .2, 1.5]])

    def basis_at(xyz):
        mol = vq.Molecule([vq.Atom(1, p.tolist()) for p in xyz], 0, 1)
        return vq.BasisSet(mol, "sto-3g")

    basis = basis_at(positions)
    density = np.array([[.8, -.2], [-.2, -.3]])
    derivative = manta.DerivativeTiles([basis]*4)

    def energy(xyz):
        jk = manta.DirectJKBuilder(basis_at(xyz)).build(density)
        return .5*np.einsum("ij,ji", density, jk.J)-.25*np.einsum("ij,ji", density, jk.K)

    for axis in range(3):
        analytic = 0.
        for indices in product(range(2), repeat=4):
            a, b, c, d = indices
            weight = .5*density[a, b]*density[c, d]-.25*density[a, c]*density[b, d]
            for center, atom in enumerate(indices):
                if atom == 0:
                    analytic += weight*derivative.evaluate(indices, center, axis)["values"].item()
        step = np.zeros_like(positions); step[0, axis] = 2e-4
        fd = (8*(energy(positions+step)-energy(positions-step))
              -(energy(positions+2*step)-energy(positions-2*step)))/(12*2e-4)
        assert analytic == pytest.approx(fd, abs=2e-10)
