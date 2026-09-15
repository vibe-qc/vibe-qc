"""Admission gates for the dormant far-field reconstruction prototype.

Bare orbit metadata does not establish operator/tensor covariance. Keep the
unreduced kernel as the numerical reference and refuse unqualified reuse.
"""
from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc._vibeqc_core import LatticeSumOptions, compute_multipole_moments_lattice
from vibeqc.bipole_pair_moments import pair_center_moments
from vibeqc.bipole_spherical_moment_buffer import build_spherical_moment_buffer
from vibeqc.bipole_dispatch import PenetrationDispatchParameters
from vibeqc.bipole_symmetry_dispatch import (
    build_symmetry_reduced_penetration_dispatch,
    build_symmetry_fock_reconstruction_map,
    SymmetryFockReconstructionMap,
)
from vibeqc.bipole_far_field_kernel import (
    FarFieldFockKernel,
    build_far_field_fock_kernel,
    apply_far_field_fock_kernel,
    apply_far_field_fock_kernel_with_symmetry,
)

ANG2BOHR = 1.0 / 0.529177210903


def _build_infrastructure(system, basis, cutoff=6.0, L_max=2):
    """Build the spherical moment buffer and a permutation-reduced dispatch."""
    lo = LatticeSumOptions()
    lo.cutoff_bohr = cutoff

    # Compute Cartesian moments at L=2 or 3.
    mom_L = min(L_max, 3)
    cart = compute_multipole_moments_lattice(basis, system, lo, mom_L, (0.0, 0.0, 0.0))
    pair_mom = pair_center_moments(cart, basis, L_target=mom_L)
    sph_buf = build_spherical_moment_buffer(pair_mom, basis, L_max=L_max)

    # Build permutation-reduced dispatch with lower dispatch slope
    # to ensure we get some far-field quartets for this test.
    from vibeqc.bipole_bravais_utils import cell_volume_bohr
    vol = cell_volume_bohr(system)
    params = PenetrationDispatchParameters(
        maximum_multipole_order=L_max,
        dispatch_slope=1.0,  # lower = more far-field quartets
        cell_length_scale_inv_bohr=float(vol) ** (-1.0/3.0),
        overlap_drop_threshold=1e-6,
    )
    j_result, _k = build_symmetry_reduced_penetration_dispatch(
        basis, list(cart.cells), params, compute_exchange=False,
    )
    reduced_dispatch = j_result.dispatch
    # Build unfolded dispatch for comparison.
    from vibeqc.bipole_symmetry_dispatch import unfold_symmetry_reduced_dispatch

    n_sh = len(list(basis.shells()))
    unfolded_dispatch = unfold_symmetry_reduced_dispatch(reduced_dispatch, n_sh)

    return sph_buf, reduced_dispatch, unfolded_dispatch, list(cart.cells)


def _square_block_fixture():
    cell = (0, 0, 0)
    bra = (0, 2, 2, 4)
    transpose = (2, 4, 0, 2)
    matrix = np.diag([1., 2., 3., 4.])
    density = np.eye(4)*0.5
    density[:2, 2:] = np.array([[1., 2.], [3., 4.]])/20
    density[2:, :2] = density[:2, 2:].T
    entry = (cell, bra, cell, bra, matrix, 0)
    return cell, bra, transpose, density, entry


def test_identity_metadata_preserves_unreduced_fock_energy_and_count():
    cell, bra, transpose, density, entry = _square_block_fixture()
    # Independently supplied full kernel: second block is the explicit bra
    # transpose, whose action cannot be inferred from its equal dimensions.
    second = (cell, transpose, cell, bra, entry[4][[0, 2, 1, 3]], 1)
    kernel = FarFieldFockKernel([entry, second], 4)
    identity = SymmetryFockReconstructionMap([[entry[:4]], [second[:4]]], 1)
    full = apply_far_field_fock_kernel(kernel, {cell: density})
    result = apply_far_field_fock_kernel_with_symmetry(kernel, identity, {cell: density})
    block = (entry[4] @ density[:2, 2:].ravel()).reshape(2, 2)
    expected = np.zeros((4, 4))
    expected[:2, 2:] = block
    expected[2:, :2] = block.T
    np.testing.assert_allclose(full.fock_blocks[cell], expected, atol=1e-14, rtol=0)
    np.testing.assert_array_equal(result.fock_blocks[cell], full.fock_blocks[cell])
    assert result.e_coulomb_far == pytest.approx(0.5*np.sum(density*expected), abs=1e-14, rel=0)
    assert result.e_coulomb_far == full.e_coulomb_far
    assert result.n_quartets == full.n_quartets == 2


@pytest.mark.parametrize("shape", [(2, 2, 2, 2), (2, 3, 2, 2), (1, 3, 2, 3)])
@pytest.mark.parametrize("storage", ["C", "F"])
def test_native_pair_packing_matches_existing_python_reference(monkeypatch, shape, storage):
    import vibeqc.bipole_far_field_kernel as implementation

    n1, n2, n3, n4 = shape
    nbf = max(n1+n2, n3+n4)
    cell = (0, 0, 0)
    rng = np.random.default_rng(762)
    a = rng.normal(size=(nbf, nbf))
    density = a @ a.T
    density /= 2*np.max(np.linalg.eigvalsh(density))
    tensor = rng.normal(size=shape)
    matrix = np.array(tensor.reshape(n1*n2, n3*n4), order=storage)
    entry = (cell, (0, n1, n1, n1+n2), cell, (0, n3, n3, n3+n4), matrix, 0)
    kernel = FarFieldFockKernel([entry], nbf)
    native = implementation._apply_far_field_fock_kernel_native(kernel, {cell: density})
    expected = np.zeros((nbf, nbf))
    expected[:n1, n1:n1+n2] = np.einsum("abcd,cd->ab", tensor, density[:n3, n3:n3+n4])
    def native_unavailable(*args):
        raise ImportError("exercise the existing Python unreduced reference")
    monkeypatch.setattr(implementation, "_apply_far_field_fock_kernel_native", native_unavailable)
    python = implementation.apply_far_field_fock_kernel(kernel, {cell: density})
    np.testing.assert_allclose(native.fock_blocks[cell], expected, atol=1e-14, rtol=0)
    np.testing.assert_allclose(python.fock_blocks[cell], expected, atol=1e-14, rtol=0)
    np.testing.assert_array_equal(matrix, tensor.reshape(n1*n2, n3*n4))
    assert native.e_coulomb_far == pytest.approx(0.5*np.sum(density*expected), abs=1e-14, rel=0)
    assert native.e_coulomb_far == pytest.approx(python.e_coulomb_far, abs=1e-14, rel=0)
    assert native.n_quartets == python.n_quartets == 1


@pytest.mark.parametrize("native_builder", [False, True])
def test_both_kernel_builders_preserve_pair_index_convention(native_builder):
    from types import SimpleNamespace
    from vibeqc._vibeqc_core import multipole_interaction_tensor
    from vibeqc.bipole_fock_kernel_native import build_far_field_fock_kernel_native
    from vibeqc.bipole_quartet_far_field import QuartetBipolarDispatch
    from vibeqc.bipole_spherical_moment_buffer import SphericalMomentBuffer

    # Synthetic moments isolate index transport from multipole accuracy.
    # Exercise the native builder as well as manually stored kernel matrices.
    rng = np.random.default_rng(762)
    bra = rng.normal(size=(2, 3, 4))
    ket = rng.normal(size=(3, 2, 4))
    separation = np.array([4., 2., 1.])
    cell = SimpleNamespace(index=(0, 0, 0), r_cart=np.zeros(3))
    buffer = SphericalMomentBuffer(1, 4, [cell], [(0, 2), (2, 3), (5, 3), (8, 2)],
        blocks={(0, 1, 0): bra, (2, 3, 0): ket},
        centers={(0, 1, 0): np.zeros(3), (2, 3, 0): separation})
    dispatch = QuartetBipolarDispatch()
    dispatch.add_quartet(0, 1, 0, 2, 3, 0, 1, 1., 1.)
    builder = build_far_field_fock_kernel_native if native_builder else build_far_field_fock_kernel
    kernel = builder(buffer, dispatch, nbf=10)
    tensor = np.asarray(multipole_interaction_tensor(1, 1, *separation))
    expected_kernel = np.einsum("ijP,PQ,klQ->ijkl", bra, tensor, ket).reshape(6, 6)
    assert len(kernel.contributions) == 1
    np.testing.assert_allclose(kernel.contributions[0][4], expected_kernel, atol=1e-14, rtol=0)
    a = rng.normal(size=(10, 10))
    density = a @ a.T / 100
    result = apply_far_field_fock_kernel(kernel, {(0, 0, 0): density})
    expected = np.zeros((10, 10))
    expected[:2, 2:5] = np.einsum("ijP,PQ,klQ,kl->ij", bra, tensor, ket, density[5:8, 8:10])
    np.testing.assert_allclose(result.fock_blocks[(0, 0, 0)], expected, atol=1e-14, rtol=0)
    assert result.e_coulomb_far == pytest.approx(0.5*np.sum(density*expected), abs=1e-14, rel=0)
    assert result.n_quartets == 1


def test_equal_rank_nonidentity_orbit_refused_before_contraction(monkeypatch):
    import vibeqc.bipole_far_field_kernel as implementation

    cell, bra, transpose, density, entry = _square_block_fixture()
    occupations = np.linalg.eigvalsh(density)
    assert np.all((occupations > 0) & (occupations < 1))
    np.testing.assert_array_equal(density, density.T)
    orbit = SymmetryFockReconstructionMap(
        [[entry[:4], (cell, transpose, cell, bra)]], 1,
    )
    def unexpected_contraction(*args):
        pytest.fail("unqualified orbit reached numerical contraction")
    monkeypatch.setattr(implementation, "apply_far_field_fock_kernel", unexpected_contraction)
    with pytest.raises(NotImplementedError, match="unqualified"):
        apply_far_field_fock_kernel_with_symmetry(FarFieldFockKernel([entry], 4), orbit, {cell: density})


@pytest.mark.parametrize("index", [None, -1, 1, True, 0.5])
def test_invalid_dispatch_index_is_not_silently_skipped(index):
    cell, _, _, density, entry = _square_block_fixture()
    kernel = FarFieldFockKernel([(*entry[:5], index)], 4)
    identity = SymmetryFockReconstructionMap([[entry[:4]]], 1)
    with pytest.raises(ValueError, match="dispatch index"):
        apply_far_field_fock_kernel_with_symmetry(kernel, identity, {cell: density})


def test_missing_dispatch_index_is_not_silently_skipped():
    cell, _, _, density, entry = _square_block_fixture()
    with pytest.raises(ValueError, match="dispatch indices"):
        apply_far_field_fock_kernel_with_symmetry(
            FarFieldFockKernel([entry[:5]], 4),
            SymmetryFockReconstructionMap([[entry[:4]]], 1), {cell: density},
        )


def test_identity_label_does_not_override_mismatched_positions():
    cell, bra, transpose, density, entry = _square_block_fixture()
    with pytest.raises(ValueError, match="differs from stored kernel positions"):
        apply_far_field_fock_kernel_with_symmetry(
            FarFieldFockKernel([entry], 4),
            SymmetryFockReconstructionMap([[(cell, transpose, cell, bra)]], 1),
            {cell: density},
        )


def test_nontrivial_stabilizer_count_does_not_authorize_reconstruction():
    cell, _, _, density, entry = _square_block_fixture()
    with pytest.raises(NotImplementedError, match="unqualified"):
        apply_far_field_fock_kernel_with_symmetry(
            FarFieldFockKernel([entry], 4),
            SymmetryFockReconstructionMap([[entry[:4]]], 2), {cell: density},
        )


def test_empty_orbit_is_malformed_even_when_no_kernel_entry_was_built():
    with pytest.raises(ValueError, match="empty orbit"):
        apply_far_field_fock_kernel_with_symmetry(
            FarFieldFockKernel([], 4), SymmetryFockReconstructionMap([[]], 1), {},
        )


def test_mgo_sto3g_l2_unqualified_reconstruction_refused():
    # Historical xfail source fixture. Its permutation inventory never proved
    # finite-moment operator covariance; do not treat its discrepancy as noise.
    a = 4.21 * ANG2BOHR
    lattice = (a / 2) * np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
    system = vq.PeriodicSystem(
        3, lattice, [vq.Atom(12, [0, 0, 0]), vq.Atom(8, [a/2]*3)],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    buffer, reduced, _, cells = _build_infrastructure(system, basis, cutoff=6, L_max=2)
    assert len(reduced) > 0
    kernel = build_far_field_fock_kernel(buffer, reduced, ewald_omega=0., nbf=basis.nbasis)
    reconstruction = build_symmetry_fock_reconstruction_map(
        reduced, system, basis, cells, list(buffer.shell_slices),
    )
    assert any(len(orbit) > 1 for orbit in reconstruction.orbit_entries)
    density = {tuple(c.index): np.eye(basis.nbasis) for c in cells}
    with pytest.raises(NotImplementedError, match="unqualified"):
        apply_far_field_fock_kernel_with_symmetry(kernel, reconstruction, density)
