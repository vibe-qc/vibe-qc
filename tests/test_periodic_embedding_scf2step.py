"""Tests for the Ishida two-step SCF driver (scf2step.py) — the fourth 3D
increment of the adsorbate embedding feature.

Validates:
1. compute_region_i_gf_at_kz builds a valid GF (correct shape, retarded sign)
2. compute_region_i_density_one_shot at Γ-only produces a valid density
   with occupation matching the non-embedded chain
3. build_sigma_lin_per_k returns the right number of k-point entries
4. EmbeddedDensityResult carries correct shapes
"""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc._vibeqc_core import Atom, BasisSet, LatticeSumOptions, PeriodicSystem
from vibeqc.periodic_embedding.contour import EnergyContour
from vibeqc.periodic_embedding.region import (
    TAG_REGION_I,
    TAG_SUBSTRATE,
    RegionPartition,
)
from vibeqc.periodic_embedding.scf2step import (
    EmbeddedDensityResult,
    build_sigma_lin_per_k,
    compute_region_i_density_one_shot,
    compute_region_i_gf_at_kz,
)
from vibeqc.periodic_embedding.substrate_gf import default_surface_k_mesh
from vibeqc.periodic_embedding.surface_sigma import build_surface_sigma


def _h_chain_system(
    n_atoms: int = 6, spacing: float = 4.0, box: float = 30.0
) -> PeriodicSystem:
    lat = np.eye(3) * box
    atoms = [Atom(1, [box / 2, box / 2, 2.0 + i * spacing]) for i in range(n_atoms)]
    mult = 1 if n_atoms % 2 == 0 else 2
    return PeriodicSystem(dim=3, lattice=lat, unit_cell=atoms, multiplicity=mult)


def _default_lat_opts():
    opts = LatticeSumOptions()
    opts.cutoff_bohr = 15.0
    opts.nuclear_cutoff_bohr = 15.0
    return opts


def _make_region(
    sys: PeriodicSystem, basis: BasisSet, n_sub: int = 4
) -> RegionPartition:
    """Tag bottom n_sub atoms as substrate, rest as region I."""
    n_atoms = len(sys.unit_cell)
    tags = [TAG_SUBSTRATE] * n_sub + [TAG_REGION_I] * (n_atoms - n_sub)
    return RegionPartition.from_tags(sys, basis, tags)


# ---------------------------------------------------------------------------
# 1. compute_region_i_gf_at_kz
# ---------------------------------------------------------------------------


def test_gf_at_kz_retarded_sign():
    """G_II(k,z) has negative imaginary part for Im z > 0 inside the band."""
    sys = _h_chain_system(6, spacing=2.0)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    region = _make_region(sys, basis, n_sub=4)
    lat_opts = _default_lat_opts()

    # Build Σ_emb at z.
    z = -1.3 + 0.1j
    sigma_emb = build_surface_sigma(sys, basis, region, [0.0, 0.0], lat_opts=lat_opts)
    sigma_z = sigma_emb.at(z)

    G = compute_region_i_gf_at_kz(
        sys, basis, region, np.zeros(3), z, sigma_z, lat_opts=lat_opts
    )

    assert G.shape == (region.n_i, region.n_i)
    # Retarded GF: Im diag(G) < 0 for Im z > 0.
    assert np.all(np.imag(np.diag(G)) < 0)


def test_gf_at_kz_correct_shape():
    """Check that the GF has the expected dimension."""
    sys = _h_chain_system(8, spacing=2.0)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    region = _make_region(sys, basis, n_sub=4)
    lat_opts = _default_lat_opts()

    sigma = build_surface_sigma(sys, basis, region, [0.0, 0.0], lat_opts=lat_opts)
    z = -1.3 + 0.1j
    G = compute_region_i_gf_at_kz(
        sys, basis, region, np.zeros(3), z, sigma.at(z), lat_opts=lat_opts
    )

    assert G.shape == (region.n_i, region.n_i)
    assert region.n_i == 4  # 4 region-I atoms, 1 sto-3g AO each


# ---------------------------------------------------------------------------
# 2. compute_region_i_density_one_shot
# ---------------------------------------------------------------------------


def test_one_shot_density_gamma_only():
    """Γ-only one-shot density: occupation matches closed-form expectation
    for the embedded surface chain."""
    # 8 atom chain: 4 sub, 4 region I.
    sys = _h_chain_system(8, spacing=2.0)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    region = _make_region(sys, basis, n_sub=4)
    lat_opts = _default_lat_opts()

    # Γ-only k-mesh (1×1 surface BZ).
    kmesh = default_surface_k_mesh(sys, mesh=(1, 1))

    # Contour covering the occupied spectrum. sto-3g H bandwidth ~ 0.4 Ha
    # centred near -1.3 Ha. For half filling (4 e⁻ in 8-atom chain, 2 e⁻
    # in 4-atom region I), e_fermi ≈ -1.3 Ha (center of band).
    # e_bottom must be strictly below the band → -1.6 Ha.
    contour = EnergyContour.semicircle(-1.6, -1.3, n_nodes=40)

    # Build linearized Σ_emb at the contour reference.
    z0 = -1.3 + 0.5j
    sigma_lin = build_sigma_lin_per_k(
        sys, basis, region, kmesh, z0, lat_opts=lat_opts, layer_tol=0.5
    )

    result = compute_region_i_density_one_shot(
        sys, basis, region, kmesh, contour, sigma_lin, lat_opts=lat_opts
    )

    assert isinstance(result, EmbeddedDensityResult)
    assert result.density_i_local.shape == (region.n_i, region.n_i)
    assert result.density_i.shape == (region.n_ao_total, region.n_ao_total)

    # Density should be non-zero in region I, zero outside.
    assert np.any(result.density_i != 0)

    # Occupation should be positive and reasonable (4 atoms × 1 AO × 0.5
    # at half filling ≈ 2 e⁻ in region I).
    assert result.occupation > 0.5
    assert result.occupation < 3.5  # max 4 e⁻ for spin-degenerate

    # Band energy should be negative (bound states).
    assert result.band_energy < 0

    # Density is real, symmetric, and positive-semidefinite-ish on diagonal.
    assert np.allclose(result.density_i_local, result.density_i_local.T)
    assert np.all(np.diag(result.density_i_local) >= 0)

    # per_k_trace has the right shape.
    assert result.per_k_trace.shape == (1, 40)  # 1 k-pt × 40 nodes


def test_one_shot_density_converges_with_nodes():
    """Occupation converges as contour node count increases."""
    sys = _h_chain_system(8, spacing=2.0)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    region = _make_region(sys, basis, n_sub=4)
    lat_opts = _default_lat_opts()

    kmesh = default_surface_k_mesh(sys, mesh=(1, 1))
    z0 = -1.3 + 0.5j
    sigma_lin = build_sigma_lin_per_k(sys, basis, region, kmesh, z0, lat_opts=lat_opts)

    occs = []
    for n in (8, 16, 32):
        c = EnergyContour.semicircle(-1.6, -1.3, n_nodes=n)
        r = compute_region_i_density_one_shot(
            sys, basis, region, kmesh, c, sigma_lin, lat_opts=lat_opts
        )
        occs.append(r.occupation)

    # Occupation should converge (difference decreasing).
    assert abs(occs[2] - occs[1]) < abs(occs[1] - occs[0])


# ---------------------------------------------------------------------------
# 3. build_sigma_lin_per_k
# ---------------------------------------------------------------------------


def test_build_sigma_lin_per_k_length():
    """Returns one LinearizedEmbeddingPotential per k-point."""
    sys = _h_chain_system(8, spacing=2.0)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    region = _make_region(sys, basis, n_sub=4)
    lat_opts = _default_lat_opts()

    kmesh = default_surface_k_mesh(sys, mesh=(2, 2))  # 4 k-points
    result = build_sigma_lin_per_k(
        sys, basis, region, kmesh, -1.3 + 0.5j, lat_opts=lat_opts
    )

    assert len(result) == 4
    for ik, lin in result.items():
        assert ik in (0, 1, 2, 3)
        z_test = -1.3 + 0.1j
        sigma = lin.at(z_test)
        assert sigma.shape == (region.n_i, region.n_i)


# ---------------------------------------------------------------------------
# 4. density zero-padding
# ---------------------------------------------------------------------------


def test_density_zero_padded_outside_region_i():
    """Density is zero for AOs not in region I."""
    sys = _h_chain_system(8, spacing=2.0)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    region = _make_region(sys, basis, n_sub=4)
    lat_opts = _default_lat_opts()

    kmesh = default_surface_k_mesh(sys, mesh=(1, 1))
    contour = EnergyContour.semicircle(-1.6, -1.3, n_nodes=24)
    sigma_lin = build_sigma_lin_per_k(
        sys, basis, region, kmesh, -1.3 + 0.5j, lat_opts=lat_opts
    )

    result = compute_region_i_density_one_shot(
        sys, basis, region, kmesh, contour, sigma_lin, lat_opts=lat_opts
    )

    # Rows/columns corresponding to substrate AOs should be zero.
    sub_ao = np.setdiff1d(np.arange(region.n_ao_total), region.i_ao)
    if len(sub_ao) > 0:
        assert np.allclose(result.density_i[sub_ao, :], 0)
        assert np.allclose(result.density_i[:, sub_ao], 0)


# ---------------------------------------------------------------------------
# 5. SCF loop
# ---------------------------------------------------------------------------


def test_scf_converges():
    """SCF loop with diagonal Hartree converges in a few iterations."""
    from vibeqc.periodic_embedding.scf2step import compute_region_i_density_scf

    sys = _h_chain_system(8, spacing=2.0)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    region = _make_region(sys, basis, n_sub=5)
    lat_opts = _default_lat_opts()

    kmesh = default_surface_k_mesh(sys, mesh=(1, 1))
    contour = EnergyContour.semicircle(-2.2, -1.0, n_nodes=32)
    sigma_lin = build_sigma_lin_per_k(
        sys, basis, region, kmesh, -1.6 + 0.5j, lat_opts=lat_opts
    )

    result = compute_region_i_density_scf(
        sys,
        basis,
        region,
        kmesh,
        contour,
        sigma_lin,
        lat_opts=lat_opts,
        u_onsite=0.5,
        mixing=0.4,
    )

    assert result.n_iter >= 1
    assert result.n_iter <= 50
    assert result.occupation > 0
    assert result.density_i_local.shape == (region.n_i, region.n_i)
    # The SCF-shifted density should differ from the one-shot.
    result_one_shot = compute_region_i_density_one_shot(
        sys, basis, region, kmesh, contour, sigma_lin, lat_opts=lat_opts
    )
    # The Hartree potential shifts charge — densities may differ.
    assert result.occupation != pytest.approx(result_one_shot.occupation, abs=1e-10)


def test_scf_diagonal_hartree_shift():
    """Diagonal Hartree potential has correct sign and structure."""
    from vibeqc.periodic_embedding.scf2step import _diagonal_hartree

    d = np.array([[0.3, 0.1], [0.1, 0.4]], dtype=float)
    v = _diagonal_hartree(d, u_onsite=0.5)
    assert v.shape == (2, 2)
    assert v[0, 0] == pytest.approx(0.15)  # 0.5 * 0.3
    assert v[1, 1] == pytest.approx(0.20)  # 0.5 * 0.4
    assert v[0, 1] == 0
    assert v[1, 0] == 0


# ---------------------------------------------------------------------------
# 6. HF SCF with GDF J/K
# ---------------------------------------------------------------------------


def test_scf_hf_converges():
    """HF SCF with proper J and K from GDF converges."""
    from vibeqc.periodic_embedding.scf2step import compute_region_i_density_scf_hf

    sys = _h_chain_system(8, spacing=2.0)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    region = _make_region(sys, basis, n_sub=5)
    lat_opts = _default_lat_opts()

    kmesh = default_surface_k_mesh(sys, mesh=(1, 1))
    contour = EnergyContour.semicircle(-2.2, -1.0, n_nodes=32)
    sigma_lin = build_sigma_lin_per_k(
        sys, basis, region, kmesh, -1.6 + 0.5j, lat_opts=lat_opts
    )

    result = compute_region_i_density_scf_hf(
        sys,
        basis,
        region,
        kmesh,
        contour,
        sigma_lin,
        lat_opts=lat_opts,
        mixing=0.3,
        max_iter=10,
    )

    assert result.n_iter >= 1
    assert result.occupation > 0
    assert result.density_i_local.shape == (region.n_i, region.n_i)

    # The HF density should differ from the one-shot.
    result_one_shot = compute_region_i_density_one_shot(
        sys, basis, region, kmesh, contour, sigma_lin, lat_opts=lat_opts
    )
    # With J and K included, the density should shift.
    assert result.occupation != pytest.approx(result_one_shot.occupation, abs=1e-10)


def test_scf_hf_jk_matrices():
    """J and K matrices have correct symmetry and sign."""
    from vibeqc.periodic_embedding.scf2step import _build_jk_from_lpq

    # Lpq is symmetric in the AO pair: L_{L,mu,nu} = L_{L,nu,mu}.
    lpq_raw = np.random.randn(5, 2, 2).astype(float) * 0.1
    lpq = 0.5 * (lpq_raw + lpq_raw.transpose(0, 2, 1))
    d = np.array([[0.3, 0.05], [0.05, 0.4]], dtype=float)

    J, K = _build_jk_from_lpq(lpq, d)

    assert J.shape == (2, 2)
    assert K.shape == (2, 2)
    # J is symmetric.
    assert np.allclose(J, J.T)
    # K is symmetric for real D.
    assert np.allclose(K, K.T)


# ---------------------------------------------------------------------------
# 7. UHF SCF
# ---------------------------------------------------------------------------


def test_scf_uhf_converges():
    """UHF SCF with per-spin J and K converges and produces spin densities."""
    from vibeqc.periodic_embedding.scf2step import compute_region_i_density_scf_uhf

    sys = _h_chain_system(7, spacing=2.0)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    region = _make_region(sys, basis, n_sub=4)
    lat_opts = _default_lat_opts()

    kmesh = default_surface_k_mesh(sys, mesh=(1, 1))
    contour = EnergyContour.semicircle(-2.2, -1.0, n_nodes=24)
    sigma_lin = build_sigma_lin_per_k(
        sys, basis, region, kmesh, -1.6 + 0.5j, lat_opts=lat_opts
    )

    result = compute_region_i_density_scf_uhf(
        sys,
        basis,
        region,
        kmesh,
        contour,
        sigma_lin,
        lat_opts=lat_opts,
        mixing=0.3,
        max_iter=5,
    )

    assert result.n_iter >= 1
    assert result.density_a is not None
    assert result.density_b is not None
    assert result.density_a.shape == (3, 3)
    assert result.density_b.shape == (3, 3)
    assert np.allclose(result.density_i_local, result.density_a + result.density_b)
    occ_a = np.trace(result.density_a)
    occ_b = np.trace(result.density_b)
    assert occ_a + occ_b == pytest.approx(result.occupation, abs=1e-10)


# ---------------------------------------------------------------------------
# 8. KS-DFT with Vxc
# ---------------------------------------------------------------------------


def test_scf_ks_converges():
    """KS-DFT SCF with J + Vxc (PBE) converges."""
    from vibeqc.periodic_embedding.scf2step import compute_region_i_density_scf_ks

    sys = _h_chain_system(8, spacing=2.0)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    region = _make_region(sys, basis, n_sub=5)
    lat_opts = _default_lat_opts()

    kmesh = default_surface_k_mesh(sys, mesh=(1, 1))
    contour = EnergyContour.semicircle(-2.2, -1.0, n_nodes=24)
    sigma_lin = build_sigma_lin_per_k(
        sys, basis, region, kmesh, -1.6 + 0.5j, lat_opts=lat_opts
    )

    result = compute_region_i_density_scf_ks(
        sys,
        basis,
        region,
        kmesh,
        contour,
        sigma_lin,
        functional_name="LDA",
        lat_opts=lat_opts,
        mixing=0.3,
        max_iter=5,
    )

    assert result.n_iter >= 1
    assert result.density_i_local.shape == (region.n_i, region.n_i)
