"""Cell-resolved periodic DF integral blocks.

The native multi-k GDF path needs translation-resolved 2c/3c blocks so
Bloch phases can be applied after the expensive libint shell loops. These
tests pin the storage boundary without implementing the full multi-k J/K
loop yet: summing the blocks must exactly reproduce the existing Γ-summed
periodic DF tensors, and inactive lattice axes must stay pinned for 1D/2D.
"""
from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.aux_basis import (
    build_lpq_bloch_native,
    build_lpq_native,
    make_aux_basis_set,
)


def _h2_dimensional_box(dim: int, box: float = 4.0, vacuum: float = 14.0):
    if dim == 1:
        lattice = np.diag([box, vacuum, vacuum])
        centre = np.array([0.5 * box, 0.5 * vacuum, 0.5 * vacuum])
    elif dim == 2:
        lattice = np.column_stack([
            [box, 0.0, 0.0],
            [0.5 * box, 0.5 * np.sqrt(3.0) * box, 0.0],
            [0.0, 0.0, vacuum],
        ])
        centre = lattice @ np.array([0.5, 0.5, 0.5])
    elif dim == 3:
        lattice = np.eye(3) * box
        centre = np.array([0.5 * box, 0.5 * box, 0.5 * box])
    else:
        raise ValueError(f"dim must be 1, 2, or 3; got {dim}")
    system = vq.PeriodicSystem(
        dim,
        lattice,
        [
            vq.Atom(1, (centre + np.array([0.0, 0.0, -0.6])).tolist()),
            vq.Atom(1, (centre + np.array([0.0, 0.0, 0.6])).tolist()),
        ],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    aux = make_aux_basis_set(
        system.unit_cell_molecule(),
        aux_name="def2-svp-jk",
    )
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = box
    return system, basis, aux, opts


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_2c_lattice_blocks_sum_to_gamma_metric(dim):
    system, _, aux, opts = _h2_dimensional_box(dim)

    indices, vectors, blocks = vq.compute_2c_eri_lattice_blocks(
        aux, system, opts,
    )
    summed = np.sum(blocks, axis=0)
    bloch_gamma = vq.bloch_sum_2c_eri_blocks(vectors, blocks, [0.0, 0.0, 0.0])
    reference = vq.compute_2c_eri_lattice(aux, system, opts)

    assert indices.shape == vectors.shape
    assert indices.shape[1] == 3
    assert blocks.shape == (indices.shape[0], aux.nbasis, aux.nbasis)
    assert indices.shape[0] > 1
    if dim < 3:
        np.testing.assert_array_equal(indices[:, dim:], 0)
    np.testing.assert_allclose(summed, reference, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(bloch_gamma, reference, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_3c_lattice_blocks_sum_to_gamma_tensor(dim):
    system, basis, aux, opts = _h2_dimensional_box(dim)

    indices, vectors, blocks = vq.compute_3c_eri_lattice_blocks(
        basis, aux, system, opts,
    )
    summed = np.sum(blocks, axis=0)
    summed_gamma = 0.5 * (summed + np.swapaxes(summed, 1, 2))
    bloch_gamma = vq.bloch_sum_3c_eri_blocks(vectors, blocks, [0.0, 0.0, 0.0])
    bloch_gamma = 0.5 * (bloch_gamma + np.swapaxes(bloch_gamma, 1, 2))
    reference = vq.compute_3c_eri_lattice(basis, aux, system, opts)

    assert indices.shape == vectors.shape
    assert indices.shape[1] == 3
    assert blocks.shape == (
        indices.shape[0],
        aux.nbasis,
        basis.nbasis,
        basis.nbasis,
    )
    assert indices.shape[0] > 1
    if dim < 3:
        np.testing.assert_array_equal(indices[:, dim:], 0)
    np.testing.assert_allclose(summed_gamma, reference, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        bloch_gamma,
        reference,
        atol=1e-12,
        rtol=1e-12,
    )


def test_2c_lattice_block_bloch_sum_has_expected_hermitian_pair():
    system, _, aux, opts = _h2_dimensional_box(1)
    _, vectors, blocks = vq.compute_2c_eri_lattice_blocks(aux, system, opts)
    k_cart = 0.2 * np.asarray(system.reciprocal_lattice())[:, 0]

    metric_plus = vq.bloch_sum_2c_eri_blocks(vectors, blocks, k_cart)
    np.testing.assert_allclose(
        metric_plus,
        metric_plus.conj().T,
        atol=1e-12,
        rtol=1e-12,
    )


def test_gdf_block_bloch_sum_rejects_shape_mismatch():
    vectors = np.zeros((2, 3))
    blocks = np.zeros((3, 2, 2))

    with pytest.raises(ValueError, match="cell vector count"):
        vq.bloch_sum_2c_eri_blocks(vectors, blocks, [0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="blocks must have shape"):
        vq.bloch_sum_3c_eri_blocks(vectors, np.zeros((2, 2, 2)), [0, 0, 0])


def test_bloch_lpq_builder_matches_gamma_native_lpq():
    system, basis, aux, opts = _h2_dimensional_box(1)

    lpq_gamma = build_lpq_bloch_native(
        system,
        basis,
        aux,
        [0.0, 0.0, 0.0],
        lat_opts=opts,
        linear_dep_thr=1e-9,
    )
    lpq_reference = build_lpq_native(
        system,
        basis,
        aux,
        lat_opts=opts,
        linear_dep_thr=1e-9,
    )

    eri_gamma = np.einsum(
        "Pij,Pkl->ijkl",
        lpq_gamma,
        lpq_gamma.conj(),
        optimize=True,
    )
    eri_reference = np.einsum(
        "Pij,Pkl->ijkl",
        lpq_reference,
        lpq_reference,
        optimize=True,
    )

    assert lpq_gamma.shape == lpq_reference.shape
    np.testing.assert_allclose(eri_gamma.real, eri_reference, atol=1e-10)
    np.testing.assert_allclose(eri_gamma.imag, 0.0, atol=1e-12)


def test_bloch_lpq_builder_accepts_nonzero_k_vector():
    system, basis, aux, opts = _h2_dimensional_box(1)
    k_cart = 0.2 * np.asarray(system.reciprocal_lattice())[:, 0]

    lpq = build_lpq_bloch_native(
        system,
        basis,
        aux,
        k_cart,
        lat_opts=opts,
        linear_dep_thr=1e-9,
    )

    assert lpq.shape[1:] == (basis.nbasis, basis.nbasis)
    assert np.iscomplexobj(lpq)
    assert np.all(np.isfinite(lpq.real))
    assert np.all(np.isfinite(lpq.imag))


@pytest.mark.parametrize("change_vectors", [False, True])
def test_aux_image_ket_phases_reuse_exact_cell_vectors(monkeypatch, change_vectors):
    """Reuse phases, while changed/reordered or mutated native vectors invalidate."""
    import types
    import vibeqc.aux_basis as ab

    images = np.array([[1., 0., 0.], [2., 0., 0.], [3., 0., 0.]])
    plan = types.SimpleNamespace(offsets=[0, 1], images=images,
                                 members=[[0]] * 3, radius=3., remainder=1e-10)
    vectors = np.array([[0., 0., 0.], [1., 0., 0.]])
    original = vectors.copy()
    blocks = np.array([[[[2.]]], [[[3.]]]])
    pairs = [(np.array([.1, 0., 0.]), np.array([.3, 0., 0.])),
             (np.array([.2, 0., 0.]), np.array([.3, 0., 0.]))]
    monkeypatch.setattr(ab, "_aux_image_plan_cached", lambda *a: plan)
    monkeypatch.setattr(ab, "_translated_fused_basis", lambda *a: None)
    monkeypatch.setattr(ab, "_member_rows", lambda *a: np.array([0]))
    calls = []
    def native(*args):
        if change_vectors and len(calls) == 1:
            vectors[:] = vectors[::-1].copy()
        calls.append(vectors.copy())
        return None, vectors, blocks
    monkeypatch.setattr(ab, "compute_3c_eri_lattice_blocks", native)
    exp = np.exp
    matrix_exps = []
    def counted(values):
        if np.asarray(values).ndim == 2:
            matrix_exps.append(np.asarray(values).copy())
        return exp(values)
    monkeypatch.setattr(ab.np, "exp", counted)
    got = ab.aux_image_3c(None, types.SimpleNamespace(nbasis=1), None, None,
                           precision=1e-8, k_pairs=pairs)
    assert len(matrix_exps) == (2 if change_vectors else 1)
    assert len(calls) == 3
    for (bra, ket), actual in zip(pairs, got):
        expected = 0j
        for image, cells in zip(images, calls):
            expected += exp(-1j * ((ket-bra) @ image)) * (
                exp(1j * (ket @ cells.T)) @ blocks.reshape(2))
        np.testing.assert_allclose(actual.item(), expected, atol=1e-14, rtol=0)
    np.testing.assert_array_equal(calls[0], original)
