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

See ``handovers/HANDOVER_GDF_V0_11_2026_05_29.md`` (2026-05-31 entry) and
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
    GDF fit, handovers/HANDOVER_GDF_FIT_SCREENING.md) reproduces the
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
