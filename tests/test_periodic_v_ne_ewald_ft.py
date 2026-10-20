"""Regression: EWALD_3D V_ne crystal-symmetry preservation (non-cubic cells).

The shared :func:`vibeqc.periodic_v_ne.compute_nuclear_lattice_dispatch`
EWALD_3D path builds the long-range half of V_ne by analytical
reciprocal-space FT (:func:`compute_v_ne_ewald_3d_ft_lattice`). The
legacy Becke-Lebedev molecular-grid quadrature
(:func:`compute_nuclear_lattice_ewald`, still reachable via
``VIBEQC_VNE_EWALD3D_BACKEND=grid``) does **not** preserve the crystal
point group of a non-cubic primitive cell, so Hcore elements that must
vanish by site symmetry pick up a finite, non-symmetry-equivalent
residue. On LiH primitive FCC / sto-3g::

    (Li-1s | V_ne | Li-py/pz/px) = (-5.6e-3, +1.66e-2, -5.6e-3)   # grid (buggy)
    (Li-1s | V_ne | Li-py/pz/px) = (~2e-13 across all three)       # PySCF / analytic FT

``(Li-1s | V_ne | Li-p)`` must be 0 by atom-on-itself symmetry and equal
across px/py/pz by the FCC 3-fold (1,1,1) rotation. SCF self-consistency
compounds the per-element break into a ~4 mHa total-energy error. The
reciprocal-space sum is symmetric in ``G`` by construction, so every
per-cell block — hence the Bloch sum at every k — preserves the symmetry
exactly.

See
``compute_v_ne_ewald_3d_ft_lattice`` for the derivation + references
(Lippert-Hutter-Parrinello 1997 / Sun-Berkelbach-McClain-Chan 2017 /
McClain-Sun-Chan-Berkelbach 2017).
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc._vibeqc_core import CoulombMethod, bloch_sum
from vibeqc.periodic_v_ne import (
    compute_nuclear_lattice_dispatch,
    compute_v_ne_ewald_3d_ft_lattice,
    compute_v_ne_ewald_3d_ft_gamma,
    _vne_ft_contracted_g_chunk_size,
    _vne_ft_g_chunk_size,
)

ANG2BOHR = 1.0 / 0.529177210903


def _lih_fcc_primitive():
    """LiH primitive FCC — the canonical non-cubic cell the grid breaks."""
    A = 4.084  # Å
    lat = np.array(
        [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
    ) * A * ANG2BOHR
    atoms = [vq.Atom(3, [0, 0, 0]), vq.Atom(1, [0.5 * A * ANG2BOHR] * 3)]
    system = vq.PeriodicSystem(3, lat, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _ewald_lat_opts(cutoff: float = 18.0):
    lo = vq.LatticeSumOptions()
    lo.cutoff_bohr = cutoff
    lo.nuclear_cutoff_bohr = cutoff
    lo.coulomb_method = CoulombMethod.EWALD_3D
    return lo


# Li sto-3g AO ordering: 0=1s, 1=2s, 2=2py, 3=2pz, 4=2px ; H: 5=1s.
# (Li-1s | V_ne | Li-p) is the symmetry-diagnostic block: it must vanish
# (atom-on-itself) and be equal across the three p components (FCC C3).
_LI_S = 0
_LI_P = slice(2, 5)


def test_ewald3d_v_ne_ft_chunks_release_paper_nacl_scale():
    """P05/P10 NaCl/def2-SVP hit a 42 GiB allocation by materialising
    the full ``(132, 132, 163322)`` AO-pair FT tensor. The analytical
    V_ne path must chunk over G so dense cells stay resource-bounded.
    """
    nbf = 132
    n_g = 163_322
    chunk = _vne_ft_g_chunk_size(nbf, target_mib=256.0)
    assert 1 <= chunk < n_g

    full_tensor_gib = nbf * nbf * n_g * np.dtype(np.complex128).itemsize / 1024**3
    chunk_tensor_mib = nbf * nbf * chunk * np.dtype(np.complex128).itemsize / 1024**2
    assert full_tensor_gib > 40.0
    assert chunk_tensor_mib <= 128.0


def test_ewald3d_v_ne_lattice_ft_never_materialises_dense_pair_ft():
    """BUG-PER-001 guard: the lattice V_ne FT path must never build an
    ``(nbf, nbf, n_G)``-scale pair-FT tensor.

    The guard's *mechanism* changed when the path moved onto the
    weight-contracted per-cell kernel
    (``ao_pair_fourier_transform_weighted_per_cell``). It used to prove
    the bound by counting one-cell calls to the dense Bloch pair-FT:
    that kernel returns ``(nbf, nbf, n_G)``, so the only way to stay
    bounded was to feed it one cell and one G-chunk at a time. The
    contracted kernel folds the ``v_long(G)`` weights into its innermost
    Hermite reduction and returns per-cell ``(n_cells, nbf, nbf)``
    blocks, so no such tensor exists at any point and all cells can be
    handed over at once.

    The bound is therefore now *stronger*, and this test pins it
    directly instead of by proxy: the surviving ``n_G``-proportional
    storage is the kernel's shared ``(-iG)^t`` power tables, which are
    independent of ``nbf`` and of the cell count, and the G sweep is
    still chunked so they stay bounded on a dense mesh. A regression
    that reintroduced the dense Bloch pair-FT here would fail the
    no-dense-kernel assertion below.
    """
    import vibeqc._aopair_ft as aopair_mod

    system, basis = _lih_fcc_primitive()
    lo = _ewald_lat_opts(cutoff=18.0)

    dense_calls: list[int] = []
    real_dense = aopair_mod.ao_pair_fourier_transform_bloch

    def _tracking_dense(*args, **kwargs):
        dense_calls.append(1)
        return real_dense(*args, **kwargs)

    aopair_mod.ao_pair_fourier_transform_bloch = _tracking_dense
    try:
        V_lat = compute_v_ne_ewald_3d_ft_lattice(
            basis, system, lo, ke_cutoff=40.0, screen_rel=0.0,
        )
    finally:
        aopair_mod.ao_pair_fourier_transform_bloch = real_dense

    assert len(V_lat.cells) > 1
    # The dense (nbf, nbf, n_G) kernel is not on this path at all.
    assert dense_calls == []


def test_vne_contracted_chunk_bound_is_nbf_independent():
    """The contracted kernel's G chunk bounds ``n_G``-proportional
    storage without reference to ``nbf`` -- the P05/P10 NaCl/def2-SVP
    case that once asked for 42 GiB.

    Sizing that chunk with the *dense*-tensor heuristic would be badly
    wrong in the other direction: at ``nbf = 132`` it yields a chunk of
    a few hundred G-points, so the shared power tables would be rebuilt
    hundreds of times for no memory benefit.
    """
    nbf = 132
    n_g = 163_322

    dense_chunk = _vne_ft_g_chunk_size(nbf, target_mib=256.0)
    contracted_chunk = _vne_ft_contracted_g_chunk_size(2, target_mib=256.0)

    # Same nbf-independent bound whatever the basis size.
    assert contracted_chunk == _vne_ft_contracted_g_chunk_size(
        2, target_mib=256.0
    )
    # Strictly less chunking than the dense heuristic would impose here.
    assert contracted_chunk > dense_chunk
    # Still bounded: the shared (-iG)^t tables stay within the target.
    rows = 3 * (2 * 2 + 1)
    tables_mib = (
        rows * min(contracted_chunk, n_g)
        * np.dtype(np.complex128).itemsize / 1024**2
    )
    assert tables_mib <= 256.0


def test_ewald3d_gamma_streamed_matches_dense(monkeypatch):
    """The stream_pair_ft mode (memory-lean companion of the screened
    GDF fit, the original investigation) reproduces the
    dense-bundle V_ne to numerical noise -- forced multi-chunk so the
    chunk boundary handling is actually exercised."""
    system, basis = _lih_fcc_primitive()
    lat_opts = _ewald_lat_opts()
    V_dense = compute_v_ne_ewald_3d_ft_gamma(
        basis, system, lat_opts, ke_cutoff=100.0
    )
    monkeypatch.setenv("VIBEQC_VNE_EWALD3D_FT_CHUNK_MIB", "0.05")
    V_stream = compute_v_ne_ewald_3d_ft_gamma(
        basis, system, lat_opts, ke_cutoff=100.0, stream_pair_ft=True
    )
    np.testing.assert_allclose(V_stream, V_dense, atol=1e-11, rtol=0.0)


def test_ewald3d_gamma_stream_conflicts_with_shared_bundle():
    """stream_pair_ft (never materialise the dense pair FT) plus a
    precomputed dense bundle is a contradiction -- must raise."""
    from vibeqc.aux_basis import _rsgdf_dense_pair_ft

    system, basis = _lih_fcc_primitive()
    lat_opts = _ewald_lat_opts()
    bundle = _rsgdf_dense_pair_ft(basis, system, 100.0, lat_opts)
    with pytest.raises(ValueError, match="stream_pair_ft"):
        compute_v_ne_ewald_3d_ft_gamma(
            basis,
            system,
            lat_opts,
            ke_cutoff=100.0,
            pair_ft_shared=bundle,
            stream_pair_ft=True,
        )


def test_ewald3d_v_ne_preserves_fcc_p_symmetry():
    """The default (analytic-FT) EWALD_3D dispatch yields equal, ~0
    (Li-1s|V_ne|Li-px/py/pz) on LiH primitive FCC — the symmetry the
    molecular-grid quadrature breaks. Symmetry is mesh- and cell-count-
    independent (the FT sum is symmetric in G per-cell), so a small
    ke / cutoff suffices to exercise it cheaply."""
    system, basis = _lih_fcc_primitive()
    lo = _ewald_lat_opts()
    V_lat = compute_nuclear_lattice_dispatch(basis, system, lo, ke_cutoff=80.0)
    V = np.real(bloch_sum(V_lat, np.zeros(3)))
    V = 0.5 * (V + V.T)
    p = V[_LI_S, _LI_P]
    assert abs(float(p.max() - p.min())) < 1e-8, (
        f"(Li-1s|V_ne|Li-px/py/pz) must be equal by FCC C3 symmetry; got "
        f"{p} (spread {float(p.max() - p.min()):.3e}). The analytic-FT "
        "V_long path should preserve crystal symmetry exactly."
    )
    assert float(np.max(np.abs(p))) < 1e-8, (
        f"(Li-1s|V_ne|Li-p) must vanish (atom-on-itself); got {p}."
    )


def test_ewald3d_lattice_bloch_sum_matches_gamma():
    """The per-cell builder behind the dispatch Bloch-sums at Γ to the
    proven Γ-only routine ``compute_v_ne_ewald_3d_ft_gamma`` (the one
    validated to µHa vs PySCF GDF on LiH). Ties the lattice
    generalisation to the reference; guards against per-cell-decomposition
    drift."""
    system, basis = _lih_fcc_primitive()
    lo = _ewald_lat_opts()
    ke = 120.0
    V_lat = compute_nuclear_lattice_dispatch(basis, system, lo, ke_cutoff=ke)
    V_bloch = np.real(bloch_sum(V_lat, np.zeros(3)))
    V_bloch = 0.5 * (V_bloch + V_bloch.T)
    V_gamma = compute_v_ne_ewald_3d_ft_gamma(basis, system, lo, ke_cutoff=ke)
    assert np.max(np.abs(V_bloch - V_gamma)) < 1e-9, (
        "bloch_sum(per-cell lattice V_ne, k=0) must reproduce the Γ-only "
        f"routine; max abs diff {np.max(np.abs(V_bloch - V_gamma)):.3e}."
    )


def test_ewald3d_grid_backend_breaks_p_symmetry(monkeypatch):
    """Characterise the bug the analytic default fixes: with the legacy
    grid backend forced on, the FCC p-symmetry breaks by ~1e-2 (the
    documented ~mHa-scale defect). This guards that the default is NOT
    the grid path — if a future change silently routes the dispatch back
    to the grid, this test (the bug) would start passing while the fix
    test above would fail."""
    monkeypatch.setenv("VIBEQC_VNE_EWALD3D_BACKEND", "grid")
    system, basis = _lih_fcc_primitive()
    lo = _ewald_lat_opts()
    V_lat = compute_nuclear_lattice_dispatch(basis, system, lo)
    V = np.real(bloch_sum(V_lat, np.zeros(3)))
    V = 0.5 * (V + V.T)
    p = V[_LI_S, _LI_P]
    assert abs(float(p.max() - p.min())) > 1e-3, (
        "The legacy grid backend is expected to break the FCC p-symmetry "
        f"by ~1e-2; got spread {float(p.max() - p.min()):.3e}. If this is "
        "now small, the grid path may have been fixed/rerouted — update "
        "this characterisation test."
    )


def _lih_rocksalt_conventional():
    """LiH rocksalt — conventional **cubic** 8-atom cell (4 formula units)."""
    a = 4.084 * ANG2BOHR
    uc = []
    for fx, fy, fz in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
        uc.append(vq.Atom(3, [fx * a, fy * a, fz * a]))  # Li
    for fx, fy, fz in [(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]:
        uc.append(vq.Atom(1, [fx * a, fy * a, fz * a]))  # H
    system = vq.PeriodicSystem(3, np.diag([a, a, a]), uc)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def test_ewald3d_v_ne_cubic_rocksalt_also_symmetric():
    """The grid V_long break is per-atom **L≥1 site symmetry**, NOT cell
    shape — so it also afflicts a conventional **cubic** multi-atom cell
    (LiH rocksalt), and the analytic-FT path must fix it there too. This
    guards against the "non-cubic primitive only" misconception: on LiH
    rocksalt the grid gives `(Li-1s|V_ne|Li-p)` a ~1.1e-2 spread (Li-2p
    site symmetry broken even though the cell is cubic); the analytic FT
    restores it to machine precision."""
    system, basis = _lih_rocksalt_conventional()
    lo = _ewald_lat_opts(cutoff=15.0)
    V_lat = compute_nuclear_lattice_dispatch(basis, system, lo, ke_cutoff=80.0)
    V = np.real(bloch_sum(V_lat, np.zeros(3)))
    V = 0.5 * (V + V.T)
    # First Li: AO 0 = 1s, AOs 2,3,4 = 2p (py, pz, px). (Li-1s | V_ne |
    # Li-p) must vanish + be equal by the octahedral (O_h) site symmetry.
    p = V[0, 2:5]
    assert abs(float(p.max() - p.min())) < 1e-8, (
        f"LiH rocksalt (cubic) (Li-1s|V_ne|Li-px/py/pz) must be equal by "
        f"O_h site symmetry; got {p} (spread {float(p.max()-p.min()):.3e}). "
        "The fix must preserve L≥1 site symmetry on cubic multi-atom cells "
        "too, not only non-cubic primitives."
    )
    assert float(np.max(np.abs(p))) < 1e-8, (
        f"LiH rocksalt (cubic) (Li-1s|V_ne|Li-p) must vanish; got {p}."
    )


# --- weight-contracted per-cell kernel -------------------------------------
# ao_pair_fourier_transform_weighted_per_cell folds the reciprocal weights
# into the innermost Hermite reduction and returns per-cell blocks, so it
# must reproduce the dense per-cell Bloch pair-FT contracted in Python.
# Re() commutes out to the innermost accumulator only because every outer
# factor (Hermite E coefficients, cart-to-sph tables) is real -- these pin
# that, and the cart-to-sph path through L = 3.


@pytest.mark.parametrize(
    "Z_pair, basis_name",
    [
        ((3, 1), "sto-3g"),      # max_l = 1
        ((3, 1), "def2-svp"),    # max_l = 1, contracted
        ((14, 14), "def2-svp"),  # max_l = 2 (d)
        ((14, 14), "def2-tzvp"),  # max_l = 3 (f)
    ],
)
def test_weighted_per_cell_kernel_matches_dense_reference(Z_pair, basis_name):
    from vibeqc._vibeqc_core import (
        ao_pair_fourier_transform_weighted_per_cell,
        direct_lattice_cells,
    )
    from vibeqc._aopair_ft import ao_pair_fourier_transform_bloch
    from vibeqc.aux_basis import rsgdf_dense_g_mesh

    a = 10.26
    lat = 0.5 * a * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    ).T
    system = vq.PeriodicSystem(
        3, lat,
        [vq.Atom(Z_pair[0], [0.0, 0.0, 0.0]),
         vq.Atom(Z_pair[1], [0.25 * a, 0.25 * a, 0.25 * a])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), basis_name)
    cells = direct_lattice_cells(system, vq.LatticeSumOptions().cutoff_bohr)
    R = np.ascontiguousarray(
        np.array([list(c.r_cart) for c in cells], dtype=float)[:5]
    )
    G = rsgdf_dense_g_mesh(system, 12.0)
    G = np.ascontiguousarray(G[(G ** 2).sum(axis=1) > 0])
    rng = np.random.default_rng(11)
    w = rng.normal(size=len(G)) + 1j * rng.normal(size=len(G))

    got = ao_pair_fourier_transform_weighted_per_cell(basis, G, R, w)
    assert got.shape == (len(R), basis.nbasis, basis.nbasis)

    for c in range(len(R)):
        ft = ao_pair_fourier_transform_bloch(
            basis, G, R[c:c + 1], k_cart=np.zeros(3)
        )
        want = np.real(np.einsum("g,mng->mn", w, ft.conj(), optimize=True))
        np.testing.assert_allclose(got[c], want, atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize("cache_bytes", [0, 16384 + 17 * 8, 64 * 1024**2])
@pytest.mark.parametrize("screen_tol", [0.0, 1e-12])
def test_weighted_radial_reuse_is_call_local_and_matches_scalar(cache_bytes, screen_tol):
    """Changed G, origins, coefficients and exponents cannot reuse stale rows."""
    from vibeqc._vibeqc_core import ao_pair_fourier_transform_weighted_per_cell

    system, original = _lih_fcc_primitive()
    shells = original.shells()
    exponents = list(shells[0].exponents)
    exponents[0] = np.nextafter(exponents[0], np.inf)
    shells[0].exponents = exponents
    coefficients = list(shells[-1].coefficients)
    coefficients[0] *= -0.75
    shells[-1].coefficients = coefficients
    changed = vq.BasisSet(system.unit_cell_molecule(), shells, "changed-radials", True)
    rng = np.random.default_rng(299)
    vectors = np.ascontiguousarray(rng.normal(size=(17, 3)))
    weights = rng.normal(size=17) + 1j * rng.normal(size=17)
    cells = np.array([[0., 0., 0.], [2.1, .3, -.2], [-2.1, -.3, .2]])
    for basis, scale in [(original, 1.0), (changed, 1.125), (original, 1.0)]:
        g = np.ascontiguousarray(vectors * scale)
        expected = ao_pair_fourier_transform_weighted_per_cell(
            basis, g, cells, weights, screen_tol, radial_cache_bytes=0,
        )
        actual = ao_pair_fourier_transform_weighted_per_cell(
            basis, g, cells, weights, screen_tol, radial_cache_bytes=cache_bytes,
        )
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("basis_name", ["def2-svp", "def2-tzvp"])
def test_weighted_hermite_reuse_keeps_primitive_cell_and_call_dependence(basis_name):
    """Contracted d/f pairs share moments only within one primitive and cell."""
    from vibeqc._vibeqc_core import ao_pair_fourier_transform_weighted_per_cell
    from vibeqc._aopair_ft import ao_pair_fourier_transform_bloch

    molecule = vq.Molecule([
        vq.Atom(14, [0.2, -0.1, 0.3]),
        vq.Atom(14, [2.1, 0.4, -0.2]),
    ])
    original = vq.BasisSet(molecule, basis_name)
    shells = original.shells()
    coefficients = list(shells[-1].coefficients)
    coefficients[0] *= -0.75
    shells[-1].coefficients = coefficients
    changed = vq.BasisSet(molecule, shells, "changed-contraction", True)
    rng = np.random.default_rng(382)
    vectors = np.ascontiguousarray(rng.normal(size=(9, 3)))
    weights = rng.normal(size=9) + 1j * rng.normal(size=9)
    cells = np.array([[0., 0., 0.], [1.4, -.3, .2], [-1.4, .3, -.2]])

    for basis, scale in [(original, 1.0), (changed, 1.125), (original, 1.0)]:
        g = np.ascontiguousarray(vectors * scale)
        w = np.ascontiguousarray(weights * complex(scale, 1.0 - scale))
        actual = ao_pair_fourier_transform_weighted_per_cell(basis, g, cells, w)
        for c in range(len(cells)):
            single = ao_pair_fourier_transform_weighted_per_cell(
                basis, g, cells[c:c + 1], w,
            )
            np.testing.assert_array_equal(actual[c], single[0])
            dense = ao_pair_fourier_transform_bloch(
                basis, g, cells[c:c + 1], k_cart=np.zeros(3),
            )
            expected = np.real(np.einsum("g,mng->mn", w, dense.conj(), optimize=True))
            np.testing.assert_allclose(actual[c], expected, atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize("target_mib", [1.0, 4.0, 256.0, 1024.0])
@pytest.mark.parametrize("threads", [1, 3, 64])
def test_weighted_fourier_workspace_reserves_radials_and_both_worker_rows(
    monkeypatch, target_mib, threads,
):
    from vibeqc import _vibeqc_core
    from vibeqc.periodic_v_ne import _vne_ft_contracted_workspace

    monkeypatch.setattr(_vibeqc_core, "get_num_threads", lambda: threads)
    chunk, cache = _vne_ft_contracted_workspace(3, target_mib=target_mib)
    per_g = (3 * 7 + 2 * threads + 3) * 16
    assert chunk * per_g + cache <= int(target_mib * 1024**2)
    assert 0 <= cache <= min(64 * 1024**2, int(target_mib * 1024**2) // 4)
    assert (chunk + 1) * per_g + cache > int(target_mib * 1024**2)


def test_weighted_per_cell_kernel_is_chunk_additive():
    """Splitting the G sweep must add exactly -- the driver accumulates
    the kernel's per-chunk results."""
    from vibeqc._vibeqc_core import (
        ao_pair_fourier_transform_weighted_per_cell,
        direct_lattice_cells,
    )
    from vibeqc.aux_basis import rsgdf_dense_g_mesh

    system, basis = _lih_fcc_primitive()
    cells = direct_lattice_cells(system, 12.0)
    R = np.ascontiguousarray(
        np.array([list(c.r_cart) for c in cells], dtype=float)[:6]
    )
    G = rsgdf_dense_g_mesh(system, 30.0)
    G = np.ascontiguousarray(G[(G ** 2).sum(axis=1) > 0])
    rng = np.random.default_rng(5)
    w = rng.normal(size=len(G)) + 1j * rng.normal(size=len(G))

    whole = ao_pair_fourier_transform_weighted_per_cell(basis, G, R, w)
    half = len(G) // 2
    split = ao_pair_fourier_transform_weighted_per_cell(
        basis, np.ascontiguousarray(G[:half]), R,
        np.ascontiguousarray(w[:half]),
    ) + ao_pair_fourier_transform_weighted_per_cell(
        basis, np.ascontiguousarray(G[half:]), R,
        np.ascontiguousarray(w[half:]),
    )
    np.testing.assert_allclose(split, whole, atol=1e-12, rtol=1e-10)


# ---------------------------------------------------------------------------
# #196: the widened one-electron AO cutoff must not drive the nuclear sums.
#
# ``_gdf_oneel_cutoff_bound`` widens the one-electron cutoff so the S/T image
# tail is converged (the S(k) metric needs it, #179). On LiH/STO-3G that takes
# the direct cell list from 135 to 12527 cells. S and T cost 0.008 s over the
# whole widened list; the libint erfc nuclear sum costs ~23 ms per cell, so
# inheriting the same domain put one V_ne build at 286 s on a two-atom cell.
# The builders now derive their own domain from the per-cell overlap norms.
# ---------------------------------------------------------------------------


def _widened_oneel_opts(system, basis):
    """A flat ball at the S/T image-tail bound.

    The k-point GDF driver resolved to this ball until its bulk domain became
    pair complete and screened (#438). A ``"flat"`` strategy with a cutoff as
    wide still hands the builders such a domain, so the screen below keeps
    its #196 fixture.
    """
    from vibeqc.lattice_screening import RcutStrategy, make_lattice_opts
    from vibeqc.periodic_k_gdf import _gdf_oneel_cutoff_bound

    base = vq.LatticeSumOptions()
    base.coulomb_method = CoulombMethod.EWALD_3D
    return make_lattice_opts(
        basis, strategy=RcutStrategy.FLAT, base_opts=base,
        cutoff_bohr=_gdf_oneel_cutoff_bound(system, basis),
    )


def test_screened_domain_radius_covers_every_cell_it_keeps():
    """The helper's contract: the radius it returns contains every cell
    whose overlap block is above the tolerance."""
    from vibeqc._vibeqc_core import compute_overlap_lattice
    from vibeqc.periodic_v_ne import _screened_domain_cutoff_bohr

    system, basis = _lih_fcc_primitive()
    opts = _widened_oneel_opts(system, basis)
    S = compute_overlap_lattice(basis, system, opts)
    norms = np.array([float(np.linalg.norm(np.asarray(b))) for b in S.blocks])
    radii = np.array(
        [float(np.linalg.norm(np.asarray(c.r_cart))) for c in S.cells]
    )

    for tol in (1e-12, 1e-16, 1e-20):
        r = _screened_domain_cutoff_bohr(S.cells, norms, tol_rel=tol)
        keep = norms > tol * norms.max()
        assert r is not None
        assert radii[keep].max() <= r, (
            f"tol {tol:g}: radius {r} excludes a cell it must keep"
        )

    # A non-positive tolerance means "keep everything" -- the escape hatch
    # the pair-complete route relies on.
    assert _screened_domain_cutoff_bohr(S.cells, norms, tol_rel=0.0) is None


def test_widened_one_electron_cutoff_does_not_drive_the_nuclear_domain(
    monkeypatch,
):
    """#196 regression guard. The erfc lattice sum must be asked for a
    strictly smaller domain than the widened S/T cutoff, while the returned
    LatticeMatrixSet still spans the full cell list its consumers expect.

    ``screen_rel=1.0`` empties the V_long active set so the test costs no
    reciprocal-space work; the erfc call is spied on and answered with the
    (cheap) overlap sum on whatever domain it asked for.
    """
    from vibeqc import periodic_v_ne as vne
    from vibeqc._vibeqc_core import compute_overlap_lattice, direct_lattice_cells

    system, basis = _lih_fcc_primitive()
    opts = _widened_oneel_opts(system, basis)
    n_full = len(direct_lattice_cells(system, float(opts.cutoff_bohr)))
    assert n_full > 10_000, (
        "fixture no longer reproduces the widened domain #196 is about "
        f"(got {n_full} cells at {opts.cutoff_bohr:.2f} bohr)"
    )

    seen = {}

    def _spy(basis_, system_, alpha_, opts_):
        seen["cutoff"] = float(opts_.cutoff_bohr)
        return compute_overlap_lattice(basis_, system_, opts_)

    monkeypatch.setattr(vne, "compute_nuclear_erfc_lattice", _spy)
    V = vne.compute_v_ne_ewald_3d_ft_lattice(
        basis, system, opts, ke_cutoff=120.0, screen_rel=1.0,
    )

    assert "cutoff" in seen, "the erfc lattice sum was never called"
    n_nuc = len(direct_lattice_cells(system, seen["cutoff"]))
    assert seen["cutoff"] < float(opts.cutoff_bohr), (
        "the nuclear sum inherited the widened S/T cutoff again (#196): "
        f"{seen['cutoff']:.2f} vs {opts.cutoff_bohr:.2f} bohr"
    )
    assert n_nuc * 3 < n_full, (
        f"the screened nuclear domain ({n_nuc} cells) is not materially "
        f"smaller than the widened one ({n_full} cells)"
    )
    assert len(V.cells) == n_full, (
        "the per-cell V_ne set must still span the full one-electron cell "
        f"list ({len(V.cells)} vs {n_full})"
    )

    # ``domain_rel=0`` restores the old, unscreened domain exactly.
    seen.clear()
    vne.compute_v_ne_ewald_3d_ft_lattice(
        basis, system, opts, ke_cutoff=120.0, screen_rel=1.0, domain_rel=0.0,
    )
    assert seen["cutoff"] == float(opts.cutoff_bohr)


def test_domain_screen_is_a_no_op_at_a_converged_cutoff():
    """At a cutoff that is not over-wide, every cell carries AO-pair density,
    so the screen must change nothing at all -- bit for bit."""
    system, basis = _lih_fcc_primitive()
    lo = _ewald_lat_opts()

    screened = compute_v_ne_ewald_3d_ft_lattice(basis, system, lo, ke_cutoff=120.0)
    full = compute_v_ne_ewald_3d_ft_lattice(
        basis, system, lo, ke_cutoff=120.0, domain_rel=0.0,
    )
    assert len(screened.cells) == len(full.cells)
    a = np.array([np.asarray(x) for x in screened.blocks])
    b = np.array([np.asarray(x) for x in full.blocks])
    assert np.array_equal(a, b), (
        "the domain screen must be inert where nothing can be dropped; "
        f"max abs diff {np.abs(a - b).max():.3e}"
    )


def test_dropped_cells_carry_no_nuclear_attraction():
    """The physical claim behind the screen: V_short(g) is bounded by the
    AO-pair overlap of cell g, so cells below the tolerance carry nothing.

    Loosened to 3e-3 so the screen actually bites on an 18 bohr domain, where
    the outermost blocks still carry 8.6e-4 of the home cell. The residual must
    track the tolerance, not the matrix scale: measured C = 0.03, i.e. the
    overlap norm overestimates the nuclear attraction it stands in for by ~30x,
    which is the headroom the shipped 1e-16 default banks on.
    """
    from vibeqc._vibeqc_core import compute_overlap_lattice

    system, basis = _lih_fcc_primitive()
    lo = _ewald_lat_opts()
    tol = 3e-3

    S = compute_overlap_lattice(basis, system, lo)
    s_ref = max(float(np.linalg.norm(np.asarray(b))) for b in S.blocks)

    screened = compute_v_ne_ewald_3d_ft_lattice(
        basis, system, lo, ke_cutoff=120.0, domain_rel=tol,
    )
    full = compute_v_ne_ewald_3d_ft_lattice(
        basis, system, lo, ke_cutoff=120.0, domain_rel=0.0,
    )
    a = np.array([np.asarray(x) for x in screened.blocks])
    b = np.array([np.asarray(x) for x in full.blocks])
    residual = np.abs(a - b).max()
    assert residual > 0.0, (
        "the loosened tolerance did not drop any cell, so this test proves "
        "nothing -- raise it"
    )
    assert residual < 10.0 * tol * s_ref, (
        "V_short on a dropped cell exceeded the overlap bound the screen "
        f"relies on: {residual:.3e} vs {10.0 * tol * s_ref:.3e}"
    )


# ---------------------------------------------------------------------------
# Per-product nuclear source radius of the pair-complete erfc sum (#196).
# Each AO product sums the nuclear images within its own erfc range instead of
# every image within nuclear_cutoff_bohr of its midpoint; the native bound
# keeps the dropped attraction under source_tol per matrix element.
# ---------------------------------------------------------------------------


def _pair_complete_opts(cutoff: float, nuclear_cutoff: float):
    lo = vq.LatticeSumOptions()
    lo.cutoff_bohr = cutoff
    lo.nuclear_cutoff_bohr = nuclear_cutoff
    lo.pair_complete_1e = True
    return lo


def _diamond_631gs():
    """Diamond, 6-31G*: d shells exercise the angular part of the bound."""
    a = 3.567 * ANG2BOHR
    lat = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]) * a
    atoms = [vq.Atom(6, [0, 0, 0]), vq.Atom(6, [0.25 * a] * 3)]
    system = vq.PeriodicSystem(3, lat, atoms)
    return system, vq.BasisSet(system.unit_cell_molecule(), "6-31g*")


def _erfc_blocks(system, basis, opts, source_tol=0.0):
    from vibeqc._vibeqc_core import compute_nuclear_erfc_lattice

    out = compute_nuclear_erfc_lattice(basis, system, 2.0, opts, source_tol)
    return out.cells, np.array([np.asarray(b) for b in out.blocks])


@pytest.mark.parametrize("fixture", [_lih_fcc_primitive, _diamond_631gs])
def test_nuclear_source_radius_bounds_the_dropped_attraction(fixture):
    """Every element moves by at most the tolerance (plus rounding), at a
    tolerance loose enough that the screen measurably drops images, and the
    shipped default changes the elements only by summation order."""
    system, basis = fixture()
    opts = _pair_complete_opts(10.0, 16.0)
    cells, full = _erfc_blocks(system, basis, opts)
    rounding = 64.0 * np.finfo(float).eps * float(np.abs(full).max())

    for tol in (1e-4, 1e-6):
        screened_cells, screened = _erfc_blocks(system, basis, opts, tol)
        assert [tuple(c.index) for c in screened_cells] == [tuple(c.index) for c in cells]
        moved = float(np.abs(screened - full).max())
        assert moved > rounding, (
            f"tol {tol:g} dropped no measurable image, so this proves nothing"
        )
        assert moved <= tol + rounding, (
            f"tol {tol:g}: an element moved by {moved:.3e}, above the bound"
        )

    from vibeqc.periodic_v_ne import DEFAULT_NUCLEAR_SOURCE_TOL

    _, shipped = _erfc_blocks(system, basis, opts, DEFAULT_NUCLEAR_SOURCE_TOL)
    assert float(np.abs(shipped - full).max()) <= DEFAULT_NUCLEAR_SOURCE_TOL + rounding


def test_nuclear_source_radius_needs_a_pair_complete_bulk_sum():
    """The origin-centred sum shares one nuclear list across every product,
    so a per-product tolerance has nothing to act on there: refused, not
    silently ignored."""
    from vibeqc._vibeqc_core import compute_nuclear_erfc_lattice

    system, basis = _lih_fcc_primitive()
    opts = _pair_complete_opts(8.0, 10.0)
    opts.pair_complete_1e = False
    with pytest.raises(ValueError, match="pair_complete_1e"):
        compute_nuclear_erfc_lattice(basis, system, 2.0, opts, 1e-12)
    with pytest.raises(ValueError, match="nonnegative"):
        compute_nuclear_erfc_lattice(basis, system, 2.0, opts, -1.0)


def test_lattice_builder_passes_the_source_tolerance_only_to_pair_complete_sums(
    monkeypatch,
):
    """The V_ne builders hand the erfc kernel the shipped tolerance for a
    pair-complete domain and the historical call otherwise."""
    from vibeqc import periodic_v_ne as vne
    from vibeqc._vibeqc_core import compute_overlap_lattice

    seen = []

    def _spy(basis_, system_, alpha_, opts_, *tol):
        seen.append(tol)
        return compute_overlap_lattice(basis_, system_, opts_)

    monkeypatch.setattr(vne, "compute_nuclear_erfc_lattice", _spy)
    system, basis = _lih_fcc_primitive()
    for pair_complete, expected in ((True, (vne.DEFAULT_NUCLEAR_SOURCE_TOL,)), (False, ())):
        opts = _pair_complete_opts(8.0, 10.0)
        opts.pair_complete_1e = pair_complete
        seen.clear()
        vne.compute_v_ne_ewald_3d_ft_lattice(basis, system, opts, ke_cutoff=60.0, screen_rel=1.0)
        assert seen == [expected]
    opts = _pair_complete_opts(8.0, 10.0)
    seen.clear()
    vne.compute_v_ne_ewald_3d_ft_lattice(
        basis, system, opts, ke_cutoff=60.0, screen_rel=1.0, source_tol=0.0)
    assert seen == [()]
