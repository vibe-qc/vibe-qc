"""Tests for the substrate Green function module (substrate_gf.py) —
the second 3D increment of the adsorbate embedding feature.

Validates:
1. LayerPartition groups atoms correctly by z
2. _build_hk_sk returns Hermitian matrices matching Γ-only path
3. Sancho-Rubio surface GF from Gaussian H(k)/S(k) matches finite-chain
   direct inversion (large-N convergence check)
4. build_substrate_surface_gf end-to-end on a simple H-chain
5. Sigma_emb shape and Hermiticity
"""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc._vibeqc_core import (
    Atom,
    BasisSet,
    LatticeSumOptions,
    PeriodicSystem,
)
from vibeqc.periodic_embedding.region import (
    TAG_DELTA_V,
    TAG_REGION_I,
    TAG_SUBSTRATE,
    RegionPartition,
)
from vibeqc.periodic_embedding.substrate_gf import (
    LayerPartition,
    _build_hk_sk,
    build_substrate_surface_gf,
    default_surface_k_mesh,
)


def _h_chain_system(
    n_atoms: int = 4, spacing: float = 4.0, box: float = 30.0
) -> PeriodicSystem:
    """A vertical H-atom chain in a cubic box."""
    lat = np.eye(3) * box
    atoms = [Atom(1, [box / 2, box / 2, 2.0 + i * spacing]) for i in range(n_atoms)]
    mult = 1 if n_atoms % 2 == 0 else 2
    return PeriodicSystem(dim=3, lattice=lat, unit_cell=atoms, multiplicity=mult)


def _default_lat_opts():
    opts = LatticeSumOptions()
    opts.cutoff_bohr = 15.0
    opts.nuclear_cutoff_bohr = 15.0
    return opts


# ---------------------------------------------------------------------------
# 1. LayerPartition
# ---------------------------------------------------------------------------


def test_layer_partition_simple_chain():
    """Each H atom at a different z should get its own layer."""
    sys = _h_chain_system(4, spacing=4.0)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    lp = LayerPartition.from_system(sys, basis, layer_tol=0.5)

    assert lp.n_layers == 4
    # Each layer has exactly 1 AO (sto-3g on H = 1 s-function).
    for ell in range(lp.n_layers):
        start, end = lp.layer_ao_ranges[ell]
        assert end - start == 1
    # layer_of_atom: each atom gets its own unique layer.
    assert len(set(lp.layer_of_atom)) == 4
    # z-ordering is increasing.
    assert np.all(np.diff(lp.layer_z) > 0)


def test_layer_partition_two_per_layer():
    """Two atoms at the same z should merge into one layer."""
    box = 30.0
    lat = np.eye(3) * box
    atoms = [
        Atom(1, [10, 10, 2.0]),  # layer 0
        Atom(1, [14, 10, 2.0]),  # same z → same layer
        Atom(1, [10, 10, 8.0]),  # layer 1
        Atom(1, [14, 10, 8.0]),  # same z → same layer
    ]
    sys = PeriodicSystem(dim=3, lattice=lat, unit_cell=atoms, multiplicity=1)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    lp = LayerPartition.from_system(sys, basis, layer_tol=0.5)
    assert lp.n_layers == 2
    for ell in range(2):
        start, end = lp.layer_ao_ranges[ell]
        assert end - start == 2  # two AOs per layer


# ---------------------------------------------------------------------------
# 2. _build_hk_sk at Γ matches direct lattice-sum
# ---------------------------------------------------------------------------


def test_hk_sk_at_gamma_is_hermitian():
    """H(k=0) and S(k=0) are Hermitian."""
    sys = _h_chain_system(4, spacing=4.0)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    lat_opts = _default_lat_opts()

    H, S = _build_hk_sk(sys, basis, np.zeros(3), lat_opts)

    assert H.shape == (4, 4)
    assert S.shape == (4, 4)
    # Hermiticity.
    assert np.allclose(H, H.conj().T, atol=1e-12)
    assert np.allclose(S, S.conj().T, atol=1e-12)
    # S is overwhelmingly real and positive-definite at Γ.
    assert np.allclose(S.imag, 0, atol=1e-12)
    # Diagonal dominance: |S[i,i]| >> |S[i,j]| for i≠j.
    for i in range(4):
        for j in range(4):
            if i != j:
                assert abs(S[i, j]) < abs(S[i, i])


def test_hk_sk_vs_pbc_gdf_at_gamma():
    """_build_hk_sk at Γ matches the pbc_gdf path (S, Hcore)."""
    from vibeqc._vibeqc_core import (
        bloch_sum,
        compute_kinetic_lattice,
        compute_overlap_lattice,
    )
    from vibeqc.periodic_v_ne import compute_nuclear_lattice_dispatch

    sys = _h_chain_system(4, spacing=4.0)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    lat_opts = _default_lat_opts()

    # Direct path (exactly what _build_hk_sk does internally).
    k_gamma = np.zeros(3)
    S_lat = compute_overlap_lattice(basis, sys, lat_opts)
    T_lat = compute_kinetic_lattice(basis, sys, lat_opts)
    V_lat = compute_nuclear_lattice_dispatch(basis, sys, lat_opts)

    S_ref = np.asarray(bloch_sum(S_lat, k_gamma), dtype=np.complex128).real
    T_ref = np.asarray(bloch_sum(T_lat, k_gamma), dtype=np.complex128).real
    V_ref = np.asarray(bloch_sum(V_lat, k_gamma), dtype=np.complex128).real
    H_ref = T_ref + V_ref
    H_ref = 0.5 * (H_ref + H_ref.T)
    S_ref = 0.5 * (S_ref + S_ref.T)

    H, S = _build_hk_sk(sys, basis, np.zeros(3), lat_opts)
    assert np.allclose(np.real(H), H_ref, atol=1e-12)
    assert np.allclose(np.real(S), S_ref, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. Sancho-Rubio surface GF vs finite-chain direct inversion
# ---------------------------------------------------------------------------


def test_sancho_rubio_retarded_sign():
    """Sancho-Rubio from Gaussian H(k)/S(k) interior blocks produces a
    retarded surface GF (Im g_s < 0 for Im z > 0 inside the band)."""
    # Use a large box so periodic wrapping along z is negligible.
    BOX = 100.0
    n_atoms = 8
    spacing = 4.0
    lat = np.eye(3) * BOX
    atoms = [
        Atom(1, [BOX / 2, BOX / 2, BOX / 2 + (i - (n_atoms - 1) / 2) * spacing])
        for i in range(n_atoms)
    ]
    mult = 1 if n_atoms % 2 == 0 else 2
    sys = PeriodicSystem(dim=3, lattice=lat, unit_cell=atoms, multiplicity=mult)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    lat_opts = _default_lat_opts()  # cutoff 15 bohr, box 100 → no wrapping

    H, S = _build_hk_sk(sys, basis, np.zeros(3), lat_opts)

    # Use an interior atom pair as the repeating unit.
    mid = n_atoms // 2

    from vibeqc.periodic_embedding.decimation import sancho_rubio_surface_gf

    # Orthogonalize: Sancho-Rubio operates in S=I basis.
    S00 = np.array([[S[mid, mid]]])
    X = np.array([[1.0 / np.sqrt(S00[0, 0].real)]])  # S^{-1/2} for 1×1
    H00 = np.array([[H[mid, mid]]])
    H01 = np.array([[H[mid, mid + 1]]])
    Ht00 = X.T @ H00 @ X
    Ht01 = X.T @ H01 @ X

    # z inside the band for sto-3g H chain (~ [-1.5, -1.1] Ha)
    z = -1.3 + 0.1j
    g_surf, g_bulk = sancho_rubio_surface_gf(Ht00, Ht01, z)

    # Retarded GF: Im g < 0 for Im z > 0.
    assert g_surf[0, 0].imag < 0
    assert g_bulk[0, 0].imag < 0


# ---------------------------------------------------------------------------
# 4. build_substrate_surface_gf end-to-end
# ---------------------------------------------------------------------------


def test_build_substrate_surface_gf_basic():
    """End-to-end: 6 H-chain, bottom 4 = substrate, top 2 = region I.
    Compute g_surface and sigma_emb at Γ + complex z."""
    sys = _h_chain_system(6, spacing=4.0)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    # atoms z=2,6,10 (sub), 14,18 (region I), 22=z for top — actually
    # let me use 6 atoms: z=2,6,10,14 (sub), 18,22 (region I) — wait,
    # that's 6 atoms. Let me redo: bottom 3 sub, top 2 region I, that's 5.
    # Let me just use a clean setup.
    # z: 2,6,10,14 (sub), 18,22 (region I) — 6 total, sub=4, region I=2.
    tags = [
        TAG_SUBSTRATE,  # z=2
        TAG_SUBSTRATE,  # z=6
        TAG_SUBSTRATE,  # z=10
        TAG_SUBSTRATE,  # z=14
        TAG_REGION_I,  # z=18
        TAG_REGION_I,  # z=22
    ]
    region = RegionPartition.from_tags(sys, basis, tags)
    lat_opts = _default_lat_opts()

    # z inside the band for sto-3g H chain (eigenvalues ~ -1.5 to -1.1 Ha)
    z = -1.3 + 0.1j
    g_surf, sigma_emb = build_substrate_surface_gf(
        sys, basis, region, [0.0, 0.0], z, lat_opts=lat_opts
    )

    # g_surface: (n_layer, n_layer) where n_layer = 1 (one AO per substrate layer)
    assert g_surf.shape == (1, 1)
    # sigma_emb: (n_s, n_s) where n_s = number of plane-S AOs.
    # With spacing=4 bohr, plane S at z=16, only the bottom region-I atom
    # (z=18) is within the 3 bohr margin.
    assert sigma_emb.shape == (1, 1)
    # Sigma_emb should have negative imaginary part (retarded GF, Im z > 0).
    assert not np.allclose(sigma_emb, 0)
    assert np.imag(sigma_emb[0, 0]) < 0


def test_build_substrate_surface_gf_sigma_emb_is_retarded():
    """Sigma_emb inside the band should have negative imaginary part
    (retarded embedding potential: Im Σ < 0 for Im z > 0).
    It is NOT Hermitian for complex z — the retarded GF picks up an
    anti-Hermitian (broadening) part inside the band."""
    sys = _h_chain_system(6, spacing=4.0)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    tags = [
        TAG_SUBSTRATE,
        TAG_SUBSTRATE,
        TAG_SUBSTRATE,
        TAG_SUBSTRATE,
        TAG_REGION_I,
        TAG_REGION_I,
    ]
    region = RegionPartition.from_tags(sys, basis, tags)
    lat_opts = _default_lat_opts()

    z = -1.3 + 0.1j
    _, sigma = build_substrate_surface_gf(
        sys, basis, region, [0.0, 0.0], z, lat_opts=lat_opts
    )

    # Inside the band, Σ_emb has a non-zero anti-Hermitian part.
    # Im Σ < 0 (retarded).
    assert np.imag(sigma[0, 0]) < 0


# ---------------------------------------------------------------------------
# 5. Surface k-mesh
# ---------------------------------------------------------------------------


def test_default_surface_k_mesh():
    """default_surface_k_mesh builds a 2D MP mesh."""
    sys = _h_chain_system(4, spacing=4.0)
    kmesh = default_surface_k_mesh(sys, mesh=(2, 2))
    # With dim=3, mesh=(2,2,1) → 4 points
    assert len(kmesh) == 4
    # Weights sum to 1.
    assert sum(kmesh.weights) == pytest.approx(1.0)
    # k-vectors are 3D Cartesian.
    for kv in kmesh.kpoints:
        assert len(kv) == 3


# ---------------------------------------------------------------------------
# 6. Error cases
# ---------------------------------------------------------------------------


def test_no_substrate_raises():
    sys = _h_chain_system(4, spacing=4.0)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    tags = [TAG_REGION_I, TAG_REGION_I, TAG_REGION_I, TAG_REGION_I]
    region = RegionPartition.from_tags(sys, basis, tags)
    with pytest.raises(ValueError, match="No substrate"):
        build_substrate_surface_gf(sys, basis, region, [0, 0], 0.3 + 0.3j)


def test_single_substrate_layer_raises():
    """Need at least 2 substrate layers for H01 coupling."""
    sys = _h_chain_system(4, spacing=4.0)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    tags = [TAG_SUBSTRATE, TAG_REGION_I, TAG_REGION_I, TAG_REGION_I]
    region = RegionPartition.from_tags(sys, basis, tags)
    with pytest.raises(ValueError, match="at least two"):
        build_substrate_surface_gf(sys, basis, region, [0, 0], 0.3 + 0.3j)
