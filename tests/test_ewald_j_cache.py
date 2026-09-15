"""Regression: the EWALD_3D analytic-FT Hartree-J cache is bit-identical
to the per-iteration rebuild (E2, ``docs/pbc_audit_2026-06.md``).

The Γ and multi-k EWALD_3D SCF drivers used to rebuild the analytic AO-pair
Fourier transform (``pair_ft`` / ``pair_at_cells``), the dense G-mesh and the
``4π/G²`` kernel **every** SCF iteration — the dominant Γ-Ewald cost. The E2
fix caches those iteration-invariant pieces once and redoes only the
density-dependent contraction per iteration. These tests pin that the cached
contraction returns **byte-for-byte** the same J as the uncached path, so the
speedup cannot silently change any periodic-SCF energy.

See ``vibeqc.ewald_composed`` (Γ: ``build_j_ewald_3d_ft_gamma_cache`` /
``make_ewald_3d_gamma_j_builder``; lattice: ``build_j_ewald_3d_ft_lattice_cache``)
and ``vibeqc.periodic_fock_multi_k.make_ewald_3d_lattice_j_cache``.
"""

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.ewald_composed import (
    _ewald_j_ft_cell_chunk_size,
    build_j_ewald_3d,
    build_j_ewald_3d_ft_gamma_cache,
    build_j_ewald_3d_ft_lattice_cache,
    compute_j_ewald_3d_ft_gamma,
    compute_j_ewald_3d_ft_k_density_to_cells,
    compute_j_ewald_3d_ft_lattice,
    make_ewald_3d_gamma_j_builder,
)
from vibeqc.periodic_fock_multi_k import (
    build_periodic_fock_ewald3d_k,
    build_periodic_j_ewald3d_k_from_k_density,
    ewald_3d_j_blocks,
    make_ewald_3d_lattice_j_cache,
)
from vibeqc.periodic_k_density import real_space_density_from_per_k_density

# Small box + modest ke keeps the G-mesh small so these are sub-second J
# builds (no SCF). The bit-identicalness is independent of box/ke.
_KE = 120.0
_OMEGA = 0.5


def _h2(box: float = 12.0):
    c = box / 2
    atoms = [vq.Atom(1, [c, c, c - 0.7]), vq.Atom(1, [c, c, c + 0.7])]
    sysp = vq.PeriodicSystem(3, np.eye(3) * box, atoms)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = 8.0
    return sysp, basis, opts


def _densities(nbf: int):
    """A few physically-plausible symmetric densities + the zero edge case."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((nbf, nbf))
    return [a + a.T, 2.0 * np.eye(nbf), np.zeros((nbf, nbf))]


def _gamma_only_d_real(sysp, basis, opts, D_gamma):
    """LatticeMatrixSet with D_gamma in the home cell, zero elsewhere."""
    D_real = vq.compute_overlap_lattice(basis, sysp, opts)  # cells + block template
    for i in range(len(D_real.cells)):
        D_real.blocks[i] = (D_gamma if i == 0 else np.zeros_like(D_gamma)).copy()
    return D_real


# ---------------------------------------------------------------------------
# Γ path
# ---------------------------------------------------------------------------


def test_gamma_j_cache_bit_identical():
    sysp, basis, opts = _h2()
    cache = build_j_ewald_3d_ft_gamma_cache(
        basis, sysp, lattice_opts=opts, ke_cutoff=_KE
    )
    for D in _densities(basis.nbasis):
        J_uncached = compute_j_ewald_3d_ft_gamma(
            basis, sysp, D, _OMEGA, lattice_opts=opts, ke_cutoff=_KE
        )
        J_cached = compute_j_ewald_3d_ft_gamma(
            basis, sysp, D, _OMEGA, lattice_opts=opts, ke_cutoff=_KE, cache=cache
        )
        assert np.array_equal(J_uncached, J_cached)


def test_gamma_builder_matches_build_j_ewald_3d():
    """The driver-facing closure == calling build_j_ewald_3d each iteration."""
    sysp, basis, opts = _h2()
    j_build = make_ewald_3d_gamma_j_builder(
        basis, sysp, omega=_OMEGA, lattice_opts=opts, ke_cutoff=_KE
    )
    for D in _densities(basis.nbasis):
        J_ref = build_j_ewald_3d(
            basis, sysp, D, omega=_OMEGA, lattice_opts=opts, ke_cutoff=_KE
        )
        assert np.array_equal(j_build(D), J_ref)


# ---------------------------------------------------------------------------
# Multi-k (lattice / per-cell) path
# ---------------------------------------------------------------------------


def test_lattice_j_cache_bit_identical():
    sysp, basis, opts = _h2()
    for D_gamma in _densities(basis.nbasis):
        D_real0 = _gamma_only_d_real(sysp, basis, opts, D_gamma)
        J_uncached = compute_j_ewald_3d_ft_lattice(
            basis, sysp, D_real0, _OMEGA, lattice_opts=opts, ke_cutoff=_KE
        )
        cache = build_j_ewald_3d_ft_lattice_cache(
            basis, sysp, D_real0.cells, lattice_opts=opts, ke_cutoff=_KE
        )
        D_real1 = _gamma_only_d_real(sysp, basis, opts, D_gamma)
        J_cached = compute_j_ewald_3d_ft_lattice(
            basis, sysp, D_real1, _OMEGA, lattice_opts=opts, ke_cutoff=_KE,
            cache=cache,
        )
        assert len(J_uncached) == len(J_cached)
        for g in range(len(J_uncached)):
            assert np.array_equal(J_uncached[g], J_cached[g])


def test_k_density_to_cells_matches_lattice_exact_j_at_gamma():
    """The pure-RKS exact-J helper matches the existing lattice builder."""
    sysp, basis, opts = _h2()
    D_gamma = 2.0 * np.eye(basis.nbasis)
    D_real = vq.compute_overlap_lattice(basis, sysp, opts)
    for i in range(len(D_real.cells)):
        D_real.set_block(i, D_gamma.copy())
    J_lattice = compute_j_ewald_3d_ft_lattice(
        basis, sysp, D_real, _OMEGA, lattice_opts=opts, ke_cutoff=_KE
    )
    J_from_k = compute_j_ewald_3d_ft_k_density_to_cells(
        basis,
        sysp,
        [D_gamma.astype(complex)],
        [np.zeros(3)],
        [1.0],
        D_real.cells,
        _OMEGA,
        ke_cutoff=_KE,
        chunk_size=37,
    )
    assert len(J_lattice) == len(J_from_k)
    for g in range(len(J_lattice)):
        np.testing.assert_allclose(J_from_k[g], J_lattice[g], atol=1e-12)


def test_k_density_to_cells_cell_chunking_matches_dense():
    """Cell batching keeps the exact multi-k J contraction memory-bounded."""
    sysp, basis, _opts = _h2()
    cells = vq.direct_lattice_cells(sysp, 14.0)
    assert len(cells) > 1
    rng = np.random.default_rng(1)
    densities = []
    for _ in range(2):
        a = rng.standard_normal((basis.nbasis, basis.nbasis))
        a = a + 1j * rng.standard_normal((basis.nbasis, basis.nbasis))
        densities.append(a + a.conj().T)
    common = dict(
        basis=basis,
        system=sysp,
        D_k_list=densities,
        k_points=[np.zeros(3), np.array([np.pi / 12.0, 0.0, 0.0])],
        weights=[0.5, 0.5],
        output_cells=cells,
        omega=_OMEGA,
        ke_cutoff=40.0,
        chunk_size=11,
    )
    dense = compute_j_ewald_3d_ft_k_density_to_cells(
        **common,
        cell_chunk_size=len(cells),
    )
    batched = compute_j_ewald_3d_ft_k_density_to_cells(
        **common,
        cell_chunk_size=1,
    )
    assert len(dense) == len(batched)
    for dense_block, batched_block in zip(dense, batched):
        np.testing.assert_allclose(batched_block, dense_block, atol=1e-12)


def test_streamed_k_density_j_matches_legacy_lattice_j_nontrim(monkeypatch):
    """The driver-facing bounded route preserves the legacy exact J(k)."""
    monkeypatch.setenv("VIBEQC_J_EWALD3D_KE", "40.0")
    sysp, basis, opts = _h2()
    opts.cutoff_bohr = 14.0
    kmesh = vq.monkhorst_pack(sysp, [3, 1, 1])
    cells = vq.direct_lattice_cells(sysp, opts.cutoff_bohr)
    assert len(cells) > 1
    assert any(
        abs(np.sin(np.dot(k, cell.r_cart))) > 1e-8
        for k in kmesh.kpoints
        for cell in cells
    )
    rng = np.random.default_rng(4)
    densities = []
    for _ in kmesh.kpoints:
        a = rng.standard_normal((basis.nbasis, basis.nbasis))
        a = a + 1j * rng.standard_normal((basis.nbasis, basis.nbasis))
        densities.append(a @ a.conj().T)

    D_real = real_space_density_from_per_k_density(
        densities,
        kmesh,
        cells,
    )
    legacy = build_periodic_fock_ewald3d_k(
        basis,
        sysp,
        D_real,
        _OMEGA,
        kmesh.kpoints,
        lattice_opts=opts,
        exchange_scale=0.0,
    )
    streamed = build_periodic_j_ewald3d_k_from_k_density(
        basis,
        sysp,
        densities,
        kmesh.kpoints,
        kmesh.weights,
        cells,
        _OMEGA,
        reciprocal_chunk_size=11,
        cell_chunk_size=1,
    )
    for got, expected in zip(streamed, legacy):
        np.testing.assert_allclose(got, expected, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize(
    ("n_cells", "nbf", "n_g", "minimum_cache_gib"),
    [
        pytest.param(55, 36, 36_612, 38.8, id="P15-Si-def2SVP"),
        pytest.param(7, 132, 163_322, 296.8, id="P10-NaCl-def2SVP"),
    ],
)
def test_streamed_j_batches_release_paper_cache_shapes(
    n_cells,
    nbf,
    n_g,
    minimum_cache_gib,
):
    """P15/P10 exact cache shapes are huge; the live batch stays bounded."""
    full_cache_gib = (
        n_cells
        * nbf
        * nbf
        * n_g
        * np.dtype(np.complex128).itemsize
        / 1024**3
    )
    assert full_cache_gib >= minimum_cache_gib

    cache_cell_batch = _ewald_j_ft_cell_chunk_size(
        nbf,
        n_g,
        target_mib=4096.0,
    )
    assert cache_cell_batch < n_cells

    reciprocal_batch = min(512, n_g)
    stream_cell_batch = _ewald_j_ft_cell_chunk_size(
        nbf,
        reciprocal_batch,
        target_mib=512.0,
    )
    pair_batch_mib = (
        stream_cell_batch
        * nbf
        * nbf
        * reciprocal_batch
        * np.dtype(np.complex128).itemsize
        / 1024**2
    )
    assert stream_cell_batch < n_cells
    assert pair_batch_mib <= 256.0


def test_k_density_to_cells_auto_cell_chunk_avoids_gan_scale_tensor():
    """BUG-PER-001 scale: do not materialise all output cells at once."""
    nbf = 46
    n_g = 49
    n_cells = 41_640
    chunk = _ewald_j_ft_cell_chunk_size(nbf, n_g, target_mib=256.0)
    nominal_chunk = _ewald_j_ft_cell_chunk_size(nbf, 512, target_mib=256.0)
    assert 1 <= chunk < n_cells
    assert chunk > nominal_chunk

    full_tensor_gib = (
        n_cells
        * nbf
        * nbf
        * n_g
        * np.dtype(np.complex128).itemsize
        / 1024**3
    )
    chunk_tensor_mib = (
        chunk
        * nbf
        * nbf
        * n_g
        * np.dtype(np.complex128).itemsize
        / 1024**2
    )
    assert full_tensor_gib > 60.0
    assert chunk_tensor_mib <= 128.0


def test_lattice_j_blocks_threads_cache_bit_identical():
    """ewald_3d_j_blocks(j_cache=...) threads the cache and stays identical."""
    sysp, basis, opts = _h2()
    D_real = _gamma_only_d_real(sysp, basis, opts, 2.0 * np.eye(basis.nbasis))
    cache = make_ewald_3d_lattice_j_cache(
        basis, sysp, D_real.cells, lattice_opts=opts
    )
    J_uncached = ewald_3d_j_blocks(basis, sysp, D_real, _OMEGA, lattice_opts=opts)
    J_cached = ewald_3d_j_blocks(
        basis, sysp, D_real, _OMEGA, lattice_opts=opts, j_cache=cache
    )
    assert len(J_uncached) == len(J_cached)
    for g in range(len(J_uncached)):
        assert np.array_equal(J_uncached[g], J_cached[g])


def test_lattice_cache_cell_count_guard():
    """A cache built for a different cell list is rejected, not silently used."""
    sysp, basis, opts = _h2()
    D_real = _gamma_only_d_real(sysp, basis, opts, np.eye(basis.nbasis))
    wrong = build_j_ewald_3d_ft_lattice_cache(
        basis, sysp, [], lattice_opts=opts, ke_cutoff=_KE
    )  # zero cells
    with pytest.raises(ValueError, match="cells"):
        compute_j_ewald_3d_ft_lattice(
            basis, sysp, D_real, _OMEGA, lattice_opts=opts, ke_cutoff=_KE,
            cache=wrong,
        )


# ---------------------------------------------------------------------------
# Backend handling
# ---------------------------------------------------------------------------


def test_grid_backend_is_not_cached(monkeypatch):
    """The diagnostic grid backend stays uncached (helpers signal it)."""
    monkeypatch.setenv("VIBEQC_J_EWALD3D_BACKEND", "grid")
    sysp, basis, opts = _h2()
    D_real = _gamma_only_d_real(sysp, basis, opts, np.eye(basis.nbasis))
    # Lattice helper returns None (no cache) for the grid backend.
    assert (
        make_ewald_3d_lattice_j_cache(basis, sysp, D_real.cells, lattice_opts=opts)
        is None
    )
    # Γ factory still returns a usable closure (it falls back to
    # build_j_ewald_3d per call); we only assert it is callable to avoid the
    # slow FFT-Poisson grid build here.
    j_build = make_ewald_3d_gamma_j_builder(
        basis, sysp, omega=_OMEGA, lattice_opts=opts
    )
    assert callable(j_build)
