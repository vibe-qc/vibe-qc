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


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("empty_only", [False, True])
def test_driver_cache_omits_unsupported_cells_before_allocation(monkeypatch, reverse, empty_only):
    import vibeqc._aopair_ft as ft
    from vibeqc.lattice_screening import ao_pair_support_mask

    system, basis, opts = _h2(box=4.)
    opts.pair_complete_1e = True
    opts.cutoff_bohr = 3.2
    cells = list(vq.direct_lattice_cells(system, 9.))
    if empty_only:
        cells = [c for c in cells if not ao_pair_support_mask(basis, c.r_cart, 3.2).any()]
    if reverse:
        cells.reverse()
    active = tuple(i for i, c in enumerate(cells)
                   if ao_pair_support_mask(basis, c.r_cart, 3.2).any())
    assert len(active) < len(cells)
    dense = build_j_ewald_3d_ft_lattice_cache(
        basis, system, cells, lattice_opts=opts, ke_cutoff=2.,
    )
    original = ft.ao_pair_fourier_transform_at_cells
    allocated = []

    def observed(b, g, shifts):
        allocated.append(len(shifts))
        assert all(ao_pair_support_mask(b, r, 3.2).any() for r in shifts)
        return original(b, g, shifts)

    with monkeypatch.context() as patch:
        patch.setattr(ft, "ao_pair_fourier_transform_at_cells", observed)
        patch.setenv("VIBEQC_J_EWALD3D_KE", "2.0")
        compact = make_ewald_3d_lattice_j_cache(basis, system, cells, lattice_opts=opts)
    assert compact.active_cell_indices == active
    assert compact.n_cells == len(cells)
    assert compact.cell_keys == dense.cell_keys
    assert allocated == ([len(active)] if active else [])
    assert compact.pair_at_cells.nbytes * len(cells) == dense.pair_at_cells.nbytes * len(active)
    np.testing.assert_array_equal(compact.pair_at_cells, dense.pair_at_cells[list(active)])
    core = vq._vibeqc_core
    for matrix in _densities(basis.nbasis):
        blocks = [np.cos(np.dot(c.r_cart, [.47, .21, .15])) * matrix for c in cells]
        density = core.make_lattice_matrix_set(basis.nbasis, cells, blocks)
        expected = compute_j_ewald_3d_ft_lattice(
            basis, system, density, .4, lattice_opts=opts, cache=dense,
        )
        actual = compute_j_ewald_3d_ft_lattice(
            basis, system, density, .4, lattice_opts=opts, cache=compact,
        )
        np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=0)
        for i in set(range(len(cells))) - set(active):
            np.testing.assert_array_equal(actual[i], np.zeros_like(matrix))
    changed = core.make_lattice_matrix_set(basis.nbasis, cells[::-1], blocks[::-1])
    with pytest.raises(ValueError, match="cell ordering"):
        compute_j_ewald_3d_ft_lattice(basis, system, changed, .4, lattice_opts=opts, cache=compact)
    changed_opts = vq.LatticeSumOptions()
    changed_opts.pair_complete_1e = True
    changed_opts.cutoff_bohr = 3.1
    with pytest.raises(ValueError, match="support"):
        compute_j_ewald_3d_ft_lattice(basis, system, density, .4, lattice_opts=changed_opts, cache=compact)


@pytest.mark.parametrize("layout", ["c", "fortran", "strided"])
@pytest.mark.parametrize("compact", [False, True])
def test_lattice_contraction_does_not_conjugate_fourier_tensor(layout, compact):
    """Keep the complex finite sum exactly, without a cache-sized temporary."""
    from types import SimpleNamespace
    from vibeqc.ewald_composed import EwaldJFTLatticeCache

    class NoTensorConjugate(np.ndarray):
        def conj(self, *args, **kwargs):
            raise AssertionError("the Fourier tensor must not be conjugated")

        conjugate = conj

    rng = np.random.default_rng(1793)
    ncell, nbf, ng = 5, 3, 37
    active = (0, 2, 4) if compact else None
    nstored = len(active) if compact else ncell
    shape = (nstored, nbf, nbf, ng * (2 if layout == "strided" else 1))
    pair = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    if layout == "fortran":
        pair = np.asfortranarray(pair)
    elif layout == "strided":
        pair = pair[..., ::-2]
    before = pair.copy()
    guarded = pair.view(NoTensorConjugate)
    guarded.setflags(write=False)
    kernel = rng.uniform(.1, 10., size=ng)
    cells = [SimpleNamespace(index=(i, 0, 0)) for i in range(ncell)]
    density = rng.normal(size=(ncell, nbf, nbf))
    selected = density[list(active)] if compact else density
    rho = np.einsum("gmn,gmnk->k", selected, pair)
    assert np.max(np.abs(rho.imag)) > 1.
    expected = .17 * np.real(np.einsum("k,k,gmnk->gmn", kernel, rho, pair.conj()))
    cache = EwaldJFTLatticeCache(guarded, kernel, .17, ncell, nbf,
                                tuple(c.index for c in cells), active_cell_indices=active)
    actual = compute_j_ewald_3d_ft_lattice(
        None, None, SimpleNamespace(cells=cells, blocks=density), .4, cache=cache,
    )
    if compact:
        np.testing.assert_array_equal(np.asarray(actual)[list(active)], expected)
        np.testing.assert_array_equal(np.asarray(actual)[[1, 3]], 0.)
    else:
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(guarded, before)


@pytest.mark.parametrize("basis_name", ["sto-3g", "cc-pvdz"])
@pytest.mark.parametrize("compact", [False, True])
def test_lattice_cache_construction_batches_reciprocal_workspace(monkeypatch, basis_name, compact):
    import vibeqc.ewald_composed as ew
    import vibeqc._aopair_ft as ft

    system, _, opts = _h2(box=4.)
    basis = vq.BasisSet(system.unit_cell_molecule(), basis_name)
    opts.pair_complete_1e = True
    opts.cutoff_bohr = 3.2
    cells = list(vq.direct_lattice_cells(system, 7.))[::-1]
    with monkeypatch.context() as patch:
        patch.setattr(ew, "_ewald_j_ft_cell_chunk_size", lambda *args: 10**9)
        expected = build_j_ewald_3d_ft_lattice_cache(
            basis, system, cells, lattice_opts=opts, ke_cutoff=4.,
            compact_physical_support=compact,
        )
    original = ft.ao_pair_fourier_transform_at_cells
    batches = []

    def observed(b, vectors, shifts):
        batches.append(len(vectors))
        assert 0 < len(vectors) <= 7
        return original(b, vectors, shifts)

    with monkeypatch.context() as patch:
        patch.setattr(ew, "_ewald_j_ft_cell_chunk_size", lambda *args: 7)
        patch.setattr(ft, "ao_pair_fourier_transform_at_cells", observed)
        actual = build_j_ewald_3d_ft_lattice_cache(
            basis, system, cells, lattice_opts=opts, ke_cutoff=4.,
            compact_physical_support=compact,
        )
    assert len(batches) > 1
    assert sum(batches) == len(actual.v_G)
    assert actual.active_cell_indices == expected.active_cell_indices
    assert actual.cell_keys == expected.cell_keys
    np.testing.assert_array_equal(actual.v_G, expected.v_G)
    np.testing.assert_array_equal(actual.pair_at_cells, expected.pair_at_cells)


# ---------------------------------------------------------------------------
# Retained-cache preflight and streamed fallback (#640)
# ---------------------------------------------------------------------------


def _modulated_lattice_density(core, basis, cells, matrix):
    blocks = [np.cos(np.dot(c.r_cart, [.47, .21, .15])) * matrix for c in cells]
    return core.make_lattice_matrix_set(basis.nbasis, cells, blocks), blocks


@pytest.mark.parametrize("physical", [False, True], ids=["dense", "physical-support"])
def test_lattice_cache_preflight_streams_above_budget(monkeypatch, physical):
    """A budget the retained tensor exceeds yields a stream plan whose J
    matches the retained cache to roundoff; a budget equal to the retained
    size still retains, and the stream keeps every guard of the cache."""
    import vibeqc._aopair_ft as ft
    from vibeqc import ewald_composed as ec
    from vibeqc.ewald_composed import (
        EwaldJFTLatticeCache,
        EwaldJFTLatticeStream,
        ewald_j_ft_lattice_cache_retained_bytes,
    )

    system, basis, opts = _h2(box=4.)
    cutoff = 3.2
    if physical:
        opts.pair_complete_1e = True
        opts.cutoff_bohr = cutoff
    cells = list(vq.direct_lattice_cells(system, 9.))
    kw = dict(lattice_opts=opts, ke_cutoff=2., compact_physical_support=physical)
    retained = build_j_ewald_3d_ft_lattice_cache(basis, system, cells, **kw)
    assert isinstance(retained, EwaldJFTLatticeCache)
    nbytes = retained.pair_at_cells.nbytes
    stored = retained.pair_at_cells.shape[0]
    assert nbytes == ewald_j_ft_lattice_cache_retained_bytes(
        stored, basis.nbasis, retained.pair_at_cells.shape[-1],
    )

    exact = build_j_ewald_3d_ft_lattice_cache(
        basis, system, cells, retained_cache_max_bytes=nbytes, **kw,
    )
    assert isinstance(exact, EwaldJFTLatticeCache)
    np.testing.assert_array_equal(exact.pair_at_cells, retained.pair_at_cells)

    original = ft.ao_pair_fourier_transform_at_cells
    calls = []

    def observed(b, g, shifts):
        calls.append((len(g), len(shifts)))
        return original(b, g, shifts)

    with monkeypatch.context() as patch:
        patch.setattr(ft, "ao_pair_fourier_transform_at_cells", observed)
        stream = build_j_ewald_3d_ft_lattice_cache(
            basis, system, cells, retained_cache_max_bytes=nbytes - 1, **kw,
        )
    assert isinstance(stream, EwaldJFTLatticeStream)
    assert calls == []  # nothing transformed at construction
    assert stream.retained_bytes == nbytes
    assert stream.budget_bytes == nbytes - 1
    assert stream.n_G == retained.pair_at_cells.shape[-1]
    assert stream.n_cells == len(cells) == retained.n_cells
    assert stream.cell_keys == retained.cell_keys
    assert stream.active_cell_indices == retained.active_cell_indices
    assert stream.pair_cutoff == retained.pair_cutoff
    assert stream.translations.shape == (stored, 3)
    np.testing.assert_array_equal(stream.v_G, retained.v_G)
    if physical:
        assert stream.pair_mask is not None and stream.pair_mask.shape == (stored, basis.nbasis, basis.nbasis)
    else:
        assert stream.pair_mask is None

    core = vq._vibeqc_core
    for matrix in _densities(basis.nbasis):
        density, blocks = _modulated_lattice_density(core, basis, cells, matrix)
        expected = compute_j_ewald_3d_ft_lattice(
            basis, system, density, .4, lattice_opts=opts, cache=retained,
        )
        calls.clear()
        with monkeypatch.context() as patch:
            patch.setattr(ft, "ao_pair_fourier_transform_at_cells", observed)
            # The chunk policy floors its target at 1 MiB, which this tiny
            # mesh never fills; force one G vector per batch instead.
            patch.setattr(ec, "_ewald_j_ft_cell_chunk_size", lambda nbf, n_g, **kw: 1)
            actual = compute_j_ewald_3d_ft_lattice(
                basis, system, density, .4, lattice_opts=opts, cache=stream,
            )
        # Forced single-vector batches: the FT is rebuilt per batch over the
        # stored cells and the batches cover the mesh exactly once.
        assert len(calls) == stream.n_G > 1 and sum(n for n, _ in calls) == stream.n_G
        assert all(m == stored for _, m in calls)
        np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=0)
        single = compute_j_ewald_3d_ft_lattice(
            basis, system, density, .4, lattice_opts=opts, cache=stream,
        )
        np.testing.assert_allclose(single, expected, atol=2e-13, rtol=0)
        if physical:
            for i in set(range(len(cells))) - set(stream.active_cell_indices or range(len(cells))):
                np.testing.assert_array_equal(actual[i], np.zeros_like(matrix))

    changed = core.make_lattice_matrix_set(basis.nbasis, cells[::-1], blocks[::-1])
    with pytest.raises(ValueError, match="cell ordering"):
        compute_j_ewald_3d_ft_lattice(basis, system, changed, .4, lattice_opts=opts, cache=stream)
    changed_opts = vq.LatticeSumOptions()
    changed_opts.pair_complete_1e = not physical
    changed_opts.cutoff_bohr = 3.1
    with pytest.raises(ValueError, match="support"):
        compute_j_ewald_3d_ft_lattice(basis, system, density, .4, lattice_opts=changed_opts, cache=stream)
    with pytest.raises(ValueError, match="cells"):
        wrong = build_j_ewald_3d_ft_lattice_cache(
            basis, system, cells[:-1], retained_cache_max_bytes=0, **kw,
        )
        compute_j_ewald_3d_ft_lattice(basis, system, density, .4, lattice_opts=opts, cache=wrong)


def test_lattice_cache_budget_policy(monkeypatch):
    """The driver's budget: the explicit MiB policy wins, otherwise half of
    the available memory, and no bound when the probe finds nothing."""
    from vibeqc.ewald_composed import (
        EwaldJFTLatticeCache,
        EwaldJFTLatticeStream,
        ewald_j_ft_lattice_cache_budget_bytes,
    )

    monkeypatch.delenv("VIBEQC_J_EWALD3D_FT_CACHE_MAX_MIB", raising=False)
    assert ewald_j_ft_lattice_cache_budget_bytes(available=4 * 1024**3) == 2 * 1024**3
    assert ewald_j_ft_lattice_cache_budget_bytes(available=0) is None
    monkeypatch.setenv("VIBEQC_J_EWALD3D_FT_CACHE_MAX_MIB", "1.5")
    assert ewald_j_ft_lattice_cache_budget_bytes(available=0) == int(1.5 * 1024**2)
    monkeypatch.setenv("VIBEQC_J_EWALD3D_FT_CACHE_MAX_MIB", "0")
    assert ewald_j_ft_lattice_cache_budget_bytes() == 0
    for bad in ("-1", "nan", "inf", "many"):
        monkeypatch.setenv("VIBEQC_J_EWALD3D_FT_CACHE_MAX_MIB", bad)
        with pytest.raises(ValueError, match="VIBEQC_J_EWALD3D_FT_CACHE_MAX_MIB"):
            ewald_j_ft_lattice_cache_budget_bytes()

    sysp, basis, opts = _h2()
    D_real = _gamma_only_d_real(sysp, basis, opts, 2.0 * np.eye(basis.nbasis))
    monkeypatch.setenv("VIBEQC_J_EWALD3D_KE", str(_KE))
    monkeypatch.setenv("VIBEQC_J_EWALD3D_FT_CACHE_MAX_MIB", "0")
    stream = make_ewald_3d_lattice_j_cache(basis, sysp, D_real.cells, lattice_opts=opts)
    assert isinstance(stream, EwaldJFTLatticeStream)
    monkeypatch.delenv("VIBEQC_J_EWALD3D_FT_CACHE_MAX_MIB")
    import vibeqc.memory as memory
    monkeypatch.setattr(memory, "available_memory_bytes", lambda: 64 * 1024**3)
    cache = make_ewald_3d_lattice_j_cache(basis, sysp, D_real.cells, lattice_opts=opts)
    assert isinstance(cache, EwaldJFTLatticeCache)
    monkeypatch.setattr(memory, "available_memory_bytes", lambda: 0)
    unbounded = make_ewald_3d_lattice_j_cache(basis, sysp, D_real.cells, lattice_opts=opts)
    assert isinstance(unbounded, EwaldJFTLatticeCache)

    # The driver's J blocks agree between the two regimes to roundoff.
    J_cached = ewald_3d_j_blocks(basis, sysp, D_real, _OMEGA, lattice_opts=opts, j_cache=cache)
    J_stream = ewald_3d_j_blocks(basis, sysp, D_real, _OMEGA, lattice_opts=opts, j_cache=stream)
    assert len(J_cached) == len(J_stream)
    for g in range(len(J_cached)):
        np.testing.assert_allclose(J_stream[g], J_cached[g], atol=2e-13, rtol=0)
