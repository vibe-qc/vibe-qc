"""Finite scalar ECP operator checks, independent of SCF and fitted ERIs.

Shaw--Hill, doi:10.1063/1.4986887, Eqs. 1--3; CRYSTAL23 Sec. 3.2,
Eqs. 3.16--3.19. These checks establish operator contracts, not scientific
acceptance of an ECP calculation or convergence of finite image cutoffs.
"""

from itertools import product

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core


def _block(*, local=.37, s_projector=1.2, p_projector=-.4):
    block = core.ECPPrimitiveBlock()
    # Highest am is the local channel, retained even when its coefficient
    # is zero. ns=2 means literal r**0, not r**2, in libecpint's convention.
    block.n_primitive = 3
    block.exponents = [.45, .8, 1.3]
    block.coefficients = [local, s_projector, p_projector]
    block.ams = [2, 0, 1]
    block.ns = [2, 2, 2]
    return block


def _basis(positions, angular=(0,), *, charges=None):
    charges = [1] * len(positions) if charges is None else charges
    molecule = core.Molecule(
        [core.Atom(z, list(p)) for z, p in zip(charges, positions)],
        0, 1 + sum(charges) % 2)
    shells = [core.ShellInfo(i, ell, True, [.7 - .15*ell], [1.], list(p))
              for i, p in enumerate(positions) for ell in angular]
    return core.BasisSet(molecule, shells, 'scalar-ecp-test', False)


def _system(positions, lattice=None):
    lattice = np.diag([18., 19., 20.]) if lattice is None else lattice
    return core.PeriodicSystem(
        3, np.asarray(lattice), [core.Atom(1, list(p)) for p in positions],
        0, 1 + len(positions) % 2)


def _options(pair_cutoff=3.2, projector_cutoff=2.3):
    options = core.LatticeSumOptions()
    options.pair_complete_1e = True
    options.cutoff_bohr = pair_cutoff
    options.nuclear_cutoff_bohr = projector_cutoff
    return options


def _physical(basis, system, options, centers, blocks):
    return core.compute_ecp_lattice_from_primitives(
        basis, system, options, [list(p) for p in centers], blocks,
        physical_pair_support=True)


def _fold(lattice, k):
    # Independent positive ket-character sum; do not Hermitize the result.
    return sum(np.exp(1j*np.dot(k, cell.r_cart))*np.asarray(block)
               for cell, block in zip(lattice.cells, lattice.blocks))


def _brute_pair_block(basis, system, options, centers, primitives, translation):
    """Independent finite image assembly through molecular integrals.

    Enumerate an oversized fixed integer cube, then use physical distances.
    Each two-shell molecular integral supplies only its bra/ket cross block.
    No native periodic cell or nuclear-source selector is used for membership.
    """
    shells = basis.shells()
    sizes = [2*s.l + 1 for s in shells]
    offsets = np.cumsum([0] + sizes)
    expected = np.zeros((basis.nbasis, basis.nbasis))
    lattice = np.asarray(system.lattice)
    for i, bra in enumerate(shells):
        for j, ket in enumerate(shells):
            a = np.asarray(bra.origin)
            b = np.asarray(ket.origin) + translation
            if np.linalg.norm(a-b) > options.cutoff_bohr:
                continue
            midpoint = (a+b)/2
            source_centers, source_blocks = [], []
            for n in product(range(-3, 4), repeat=3):
                for center, primitive in zip(centers, primitives):
                    position = np.asarray(center) + lattice@np.asarray(n)
                    if np.linalg.norm(position-midpoint) <= options.nuclear_cutoff_bohr + 1e-12:
                        source_centers.extend(position)
                        source_blocks.append(primitive)
            molecule = core.Molecule([core.Atom(1, list(a)), core.Atom(1, list(b))], 0, 1)
            pair = core.BasisSet(molecule, [
                core.ShellInfo(0, bra.l, True, bra.exponents, bra.coefficients, list(a)),
                core.ShellInfo(1, ket.l, True, ket.exponents, ket.coefficients, list(b)),
            ], 'two-shell-oracle', True)
            molecular = np.asarray(core.compute_ecp_matrix_from_primitives(
                pair, source_centers, source_blocks))
            expected[offsets[i]:offsets[i+1], offsets[j]:offsets[j+1]] = molecular[:sizes[i], sizes[i]:]
    return expected


@pytest.mark.parametrize('part', ['local', 'nonlocal', 'both'])
def test_scalar_channels_match_analytic_atomic_gaussians_and_molecular_limit(part):
    positions = [[0., 0., 0.]]
    basis = _basis(positions, angular=(0, 1))
    primitive = _block(local=0. if part == 'nonlocal' else .37,
                       s_projector=0. if part == 'local' else 1.2,
                       p_projector=0. if part == 'local' else -.4)
    lattice = _physical(basis, _system(positions), _options(), positions, [primitive])
    matrix = _fold(lattice, np.zeros(3))
    molecular = core.compute_ecp_matrix_from_primitives(basis, positions[0], [primitive])
    np.testing.assert_allclose(matrix, molecular, atol=3e-12, rtol=0)
    expected = []
    for ell, exponent in [(0, .7), (1, .55)]:
        value = primitive.coefficients[0]*(2*exponent/(2*exponent+.45))**(ell+1.5)
        value += primitive.coefficients[ell+1]*(2*exponent/(2*exponent+primitive.exponents[ell+1]))**(ell+1.5)
        expected.extend([value]*(2*ell+1))
    np.testing.assert_allclose(core.compute_overlap(basis), np.eye(4), atol=2e-14, rtol=0)
    np.testing.assert_allclose(matrix, np.diag(expected), atol=3e-12, rtol=0)


def test_short_range_operator_contains_no_coulomb_tail_or_nuclear_charge():
    positions = [[0., 0., 0.]]
    light = _basis(positions, angular=(0, 1))
    heavy = _basis(positions, angular=(0, 1), charges=[30])
    system = _system(positions)
    options = _options()
    zero = _block(local=0., s_projector=0., p_projector=0.)
    np.testing.assert_array_equal(_fold(_physical(light, system, options, positions, [zero]), np.zeros(3)), 0.)
    primitive = _block()
    full = _fold(_physical(light, system, options, positions, [primitive]), np.zeros(3))
    np.testing.assert_allclose(
        _fold(_physical(heavy, system, options, positions, [primitive]), np.zeros(3)), full,
        atol=1e-14, rtol=0)
    local = _fold(_physical(light, system, options, positions,
                           [_block(s_projector=0., p_projector=0.)]), np.zeros(3))
    nonlocal_ = _fold(_physical(light, system, options, positions, [_block(local=0.)]), np.zeros(3))
    np.testing.assert_allclose(local+nonlocal_, full, atol=3e-12, rtol=0)
    # Coulomb attraction is separately linear in the effective ionic charge;
    # adding the short-range ECP cannot add a second copy of that charge.
    v1 = np.asarray(core.compute_nuclear_with_charges(light, positions, [1.]))
    v2 = np.asarray(core.compute_nuclear_with_charges(light, positions, [2.]))
    np.testing.assert_allclose((v2+full)-(v1+full), v1, atol=2e-14, rtol=0)


def test_physical_pairs_and_projector_images_match_explicit_molecular_assembly():
    lattice = np.array([[3.4, .3, .2], [0., 4.1, .1], [0., 0., 4.6]])
    positions = [[.2, .1, 0.], [1.35, .35, -.2]]
    system, basis, options = _system(positions, lattice), _basis(positions), _options()
    primitives = [_block(), _block(local=-.15, s_projector=.6)]
    result = _physical(basis, system, options, positions, primitives)
    saw_asymmetric, saw_screened_pair = False, False
    for cell, block in zip(result.cells, result.blocks):
        expected = _brute_pair_block(basis, system, options, positions, primitives,
                                    np.asarray(cell.r_cart))
        np.testing.assert_allclose(block, expected, atol=3e-12, rtol=0)
        saw_asymmetric |= np.max(np.abs(np.asarray(block)-np.asarray(block).T)) > 1e-5
        for i, a in enumerate(positions):
            for j, b in enumerate(positions):
                if np.linalg.norm(np.asarray(a)-np.asarray(b)-cell.r_cart) > options.cutoff_bohr:
                    assert block[i, j] == 0.
                    saw_screened_pair = True
    assert saw_asymmetric and saw_screened_pair


def test_complex_character_and_translation_covariance():
    lattice = np.array([[3.4, .3, .2], [0., 4.1, .1], [0., 0., 4.6]])
    positions = np.array([[.2, .1, 0.], [1.35, .35, -.2]])
    options = _options()
    primitives = [_block(), _block(local=-.15, s_projector=.6)]
    system = _system(positions, lattice)
    k = 2*np.pi*np.linalg.inv(lattice).T@np.array([.23, .17, .09])
    original = _physical(_basis(positions), system, options, positions, primitives)
    value = _fold(original, k)
    assert np.max(np.abs(value.imag)) > 1e-4
    np.testing.assert_allclose(value, value.conj().T, atol=3e-12, rtol=0)
    np.testing.assert_allclose(_fold(original, -k), value.conj(), atol=3e-12, rtol=0)
    translated = positions + [2.7, -1.8, .6]
    moved = _physical(_basis(translated), _system(translated, lattice), options, translated, primitives)
    np.testing.assert_allclose(_fold(moved, k), value, atol=3e-12, rtol=0)
    shifts = np.array([lattice@np.array([1, -1, 0]), np.zeros(3)])
    relabelled = positions + shifts
    moved = _physical(_basis(relabelled), _system(relabelled, lattice), options, relabelled, primitives)
    phase = np.exp(-1j*(shifts@k))
    np.testing.assert_allclose(_fold(moved, k), phase.conj()[:, None]*value*phase[None, :],
                               atol=3e-12, rtol=0)
    # Relabel only one projector, leaving the AO frame fixed.
    moved = _physical(_basis(positions), system, options, relabelled, primitives)
    np.testing.assert_allclose(_fold(moved, k), value, atol=3e-12, rtol=0)


def test_historical_support_is_default_and_physical_support_is_explicit():
    positions = [[.2, .1, 0.], [1.35, .35, -.2]]
    system, basis, options = _system(positions), _basis(positions), _options()
    args = (basis, system, options, positions, [_block(), _block()])
    default = core.compute_ecp_lattice_from_primitives(*args)
    historical = core.compute_ecp_lattice_from_primitives(*args, physical_pair_support=False)
    for a, b in zip(default.blocks, historical.blocks):
        np.testing.assert_array_equal(a, b)
    options.pair_complete_1e = False
    with pytest.raises(ValueError, match='physical ECP support'):
        _physical(basis, system, options, positions, [_block(), _block()])


@pytest.mark.parametrize('bad', ['dimension', 'pair_cutoff', 'projector_cutoff', 'primitive', 'count'])
def test_physical_operator_rejects_unsupported_or_invalid_inputs(bad):
    positions = [[0., 0., 0.]]
    system, basis, options, primitive = _system(positions), _basis(positions), _options(), _block()
    if bad == 'dimension':
        system = core.PeriodicSystem(2, np.asarray(system.lattice), list(system.unit_cell))
    elif bad == 'pair_cutoff':
        options.cutoff_bohr = float('nan')
    elif bad == 'projector_cutoff':
        options.nuclear_cutoff_bohr = -1.
    elif bad == 'primitive':
        primitive.ns = [-1, 2, 2]
    else:
        primitive.n_primitive = 2
    with pytest.raises(ValueError, match='physical ECP support|invalid periodic ECP|matching positive lengths'):
        _physical(basis, system, options, positions, [primitive])
