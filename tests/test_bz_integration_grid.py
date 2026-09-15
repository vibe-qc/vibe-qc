"""Tests for the per-k <-> full-BZ-grid scatter/gather used by the GR kernel.

Pure-numpy synthetic meshes (no k-mesh object / SCF), so these are independent
of the periodic core: round-trip fidelity, order-independence, electron
conservation through the convenience wrapper, and the full-mesh guards.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.bz_integration import (
    eigenvalues_to_full_grid,
    expand_ibz_eigenvalues,
    fractional_kpoints,
    gilat_occupations_on_kmesh,
    gilat_raubenheimer_occupations,
    grid_to_kpoints,
    kpoint_grid_indices,
)


def test_expand_ibz_eigenvalues_matches_full_mesh():
    # Expanding IBZ-reduced eigenvalues to the full BZ must reproduce a direct
    # full-mesh evaluation of a symmetry-invariant band -- validating that the
    # C++ ir_mapping is indexed in the unreduced-mesh order we assume.
    a = 5.0
    sysp = vq.PeriodicSystem(3, np.eye(3) * a, [vq.Atom(1, [0, 0, 0])])
    attach = getattr(vq, "attach_symmetry", None)
    if attach is None:
        pytest.skip("attach_symmetry unavailable")
    attach(sysp)
    mesh = [4, 4, 4]
    ibz = vq.monkhorst_pack(sysp, mesh, use_symmetry=True)
    full = vq.monkhorst_pack(sysp, mesh, use_symmetry=False)
    if len(ibz) >= len(full):
        pytest.skip("mesh was not symmetry-reduced for this system")

    def eps_at(kc):
        kc = np.asarray(kc, dtype=float)
        return np.array([np.cos(kc[0] * a) + np.cos(kc[1] * a) + np.cos(kc[2] * a)])

    eps_ibz = [eps_at(k) for k in np.asarray(ibz.kpoints).reshape(-1, 3)]
    full_cart, eps_full = expand_ibz_eigenvalues(sysp, ibz, eps_ibz)

    eps_direct = [eps_at(k) for k in full_cart]
    assert np.allclose(
        np.array(eps_full).ravel(), np.array(eps_direct).ravel(), atol=1e-9
    )

    # the expanded full mesh feeds the GR kernel and conserves electrons
    frac = fractional_kpoints(sysp.reciprocal_lattice(), full_cart)
    occ, ef = gilat_occupations_on_kmesh(frac, tuple(mesh), eps_full, 1.0, 2.0)
    total = float(np.sum([o.sum() for o in occ])) / (4 * 4 * 4)
    assert total == pytest.approx(1.0, abs=1e-6)


def test_fractional_kpoints_roundtrip():
    # k_cart = B @ k_frac  ->  fractional_kpoints(B, k_cart) recovers k_frac.
    B = np.diag([2 * np.pi / 7.0, 2 * np.pi / 5.0, 2 * np.pi / 9.0])
    frac = _full_mesh_frac((3, 3, 3))
    cart = (B @ frac.T).T
    assert np.allclose(fractional_kpoints(B, cart), frac)


def _full_mesh_frac(mesh):
    """All Gamma-centred fractional k-points of a regular mesh, plus a band fn."""
    n1, n2, n3 = mesh
    fracs = []
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                fracs.append([i / n1, j / n2, k / n3])
    return np.asarray(fracs, dtype=float)


def _bands_at(frac, nband=2):
    """Synthetic dispersive bands as a function of fractional k."""
    cx, cy, cz = np.cos(2 * np.pi * frac.T)
    b0 = -(cx + cy + cz)
    b1 = 3.0 + 0.5 * (cx + cy + cz)
    cols = [b0, b1][:nband]
    return np.stack(cols, axis=-1)  # (N, nband)


def test_round_trip_scatter_gather():
    mesh = (4, 5, 3)
    frac = _full_mesh_frac(mesh)
    eps = _bands_at(frac)  # (N, 2)
    eps_per_k = [eps[i] for i in range(eps.shape[0])]

    grid = eigenvalues_to_full_grid(frac, mesh, eps_per_k)
    assert grid.shape == (4, 5, 3, 2)
    assert np.all(np.isfinite(grid))

    back = grid_to_kpoints(grid, frac, mesh)  # (N, 2)
    assert np.allclose(back, eps)


def test_scatter_is_order_independent():
    mesh = (3, 3, 2)
    frac = _full_mesh_frac(mesh)
    eps = _bands_at(frac)
    eps_per_k = [eps[i] for i in range(eps.shape[0])]

    grid_ref = eigenvalues_to_full_grid(frac, mesh, eps_per_k)

    rng = np.random.default_rng(3)
    perm = rng.permutation(frac.shape[0])
    grid_shuf = eigenvalues_to_full_grid(
        frac[perm], mesh, [eps_per_k[p] for p in perm]
    )
    assert np.allclose(grid_ref, grid_shuf)


def test_grid_indices_handles_symmetric_convention():
    # frac in [-1/2, 1/2) must land on the same cells as [0, 1) via the mod.
    mesh = (4, 1, 1)
    frac = np.array([[0.0, 0, 0], [0.25, 0, 0], [-0.5, 0, 0], [-0.25, 0, 0]])
    idx = kpoint_grid_indices(frac, mesh)
    assert idx[:, 0].tolist() == [0, 1, 2, 3]


def test_off_grid_kpoints_raise():
    mesh = (4, 4, 4)
    frac = np.array([[0.1, 0.0, 0.0]])  # 0.1*4 = 0.4, not integer
    with pytest.raises(ValueError, match="not on a regular mesh"):
        kpoint_grid_indices(frac, mesh)


def test_partial_mesh_raises():
    # Fewer k-points than prod(mesh) -> must raise (IBZ must be expanded first).
    mesh = (4, 4, 4)
    frac = _full_mesh_frac(mesh)[:10]
    eps_per_k = [np.zeros(2) for _ in range(10)]
    with pytest.raises(ValueError, match="full mesh"):
        eigenvalues_to_full_grid(frac, mesh, eps_per_k)


def test_wrapper_matches_kernel_and_conserves_electrons():
    mesh = (6, 6, 6)
    frac = _full_mesh_frac(mesh)
    eps = _bands_at(frac)
    eps_per_k = [eps[i] for i in range(eps.shape[0])]
    n_elec = 1.3  # partial filling -> a real Fermi surface in band 0
    n_cells = 6 * 6 * 6

    occ_per_k, ef = gilat_occupations_on_kmesh(frac, mesh, eps_per_k, n_elec)

    # electron conservation with uniform mesh weights 1/n_cells
    total = float(np.sum([o.sum() for o in occ_per_k])) / n_cells
    assert total == pytest.approx(n_elec, abs=1e-6)

    # identical to scatter -> kernel -> gather done by hand
    grid = eigenvalues_to_full_grid(frac, mesh, eps_per_k)
    occ_grid, ef2 = gilat_raubenheimer_occupations(grid, n_elec)
    assert ef == pytest.approx(ef2, abs=1e-9)
    back = grid_to_kpoints(occ_grid, frac, mesh)
    assert np.allclose(np.stack(occ_per_k, axis=0), back)
