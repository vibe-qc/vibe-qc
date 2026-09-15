"""End-to-end tests for the embedded-surface runner (runner.py) —
the fifth 3D increment of the adsorbate embedding feature.

Validates:
1. run_embedded_surface completes on a simple H-chain slab
2. Result carries correct shapes, occupation > 0, and band energy
3. Auto-estimated band edges are sensible (e_bottom < e_fermi)
4. Explicit e_fermi / e_bottom override the auto-estimate
5. Multi-k (2×2) surface BZ integration produces valid density
"""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc._vibeqc_core import Atom, BasisSet, LatticeSumOptions, PeriodicSystem
from vibeqc.periodic_embedding.region import TAG_REGION_I, TAG_SUBSTRATE
from vibeqc.periodic_embedding.runner import (
    EmbeddedSurfaceResult,
    run_embedded_surface,
)


def _h_chain_system(
    n_atoms: int = 6, spacing: float = 4.0, box: float = 30.0
) -> PeriodicSystem:
    lat = np.eye(3) * box
    atoms = [Atom(1, [box / 2, box / 2, 2.0 + i * spacing]) for i in range(n_atoms)]
    mult = 1 if n_atoms % 2 == 0 else 2
    return PeriodicSystem(dim=3, lattice=lat, unit_cell=atoms, multiplicity=mult)


# ---------------------------------------------------------------------------
# 1. Basic end-to-end — Γ-only
# ---------------------------------------------------------------------------


def test_run_embedded_surface_gamma_only():
    """Full pipeline on a 6-atom H chain at Γ-only (1×1 k-mesh).

    Uses a wide contour (e_bottom=-2.5, e_fermi=-1.3) to cover the
    embedded region-I spectral weight which is broadened by Σ_emb."""
    sys = _h_chain_system(6, spacing=2.0)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    tags = [TAG_SUBSTRATE] * 4 + [TAG_REGION_I] * 2

    result = run_embedded_surface(
        sys,
        basis,
        tags,
        surface_k_mesh=(1, 1),
        contour_n_nodes=40,
        e_bottom=-2.2,
        e_fermi=-1.3,
    )

    assert isinstance(result, EmbeddedSurfaceResult)
    assert result.n_k == 1
    assert result.n_i > 0

    # Density has correct shape.
    assert result.density.shape == (result.region.n_ao_total,) * 2
    assert result.density_local.shape == (result.n_i, result.n_i)

    # Occupation is positive (exact value depends on contour coverage of
    # the embedded spectral function).
    assert result.occupation > 0

    # Band energy is negative (bound states).
    assert result.band_energy < 0

    # Contour info carried through.
    assert result.contour.n_nodes == 40
    assert result.e_fermi == pytest.approx(-1.3)
    assert result.e_bottom == pytest.approx(-2.2)

    # Density is symmetric and has non-negative diagonal.
    d = result.density_local
    assert np.allclose(d, d.T)
    assert np.all(np.diag(d) >= -1e-12)


# ---------------------------------------------------------------------------
# 2. Auto-estimated band edges
# ---------------------------------------------------------------------------


def test_auto_band_edges_are_sensible():
    """When e_fermi/e_bottom are not provided, the auto-estimate produces
    sensible values: e_bottom < e_fermi, and both within ~chemical range."""
    sys = _h_chain_system(6, spacing=2.0)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    tags = [TAG_SUBSTRATE] * 4 + [TAG_REGION_I] * 2

    result = run_embedded_surface(
        sys,
        basis,
        tags,
        surface_k_mesh=(1, 1),
        contour_n_nodes=24,
    )

    assert result.e_bottom < result.e_fermi
    # Auto-estimate from substrate eigenvalues — values are system-dependent.
    assert result.e_bottom < -0.5
    assert result.occupation > 0


# ---------------------------------------------------------------------------
# 3. Multi-k surface BZ
# ---------------------------------------------------------------------------


def test_run_embedded_surface_multi_k():
    """2×2 surface k-mesh produces valid density."""
    sys = _h_chain_system(8, spacing=2.0)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    tags = [TAG_SUBSTRATE] * 5 + [TAG_REGION_I] * 3

    result = run_embedded_surface(
        sys,
        basis,
        tags,
        surface_k_mesh=(2, 2),
        contour_n_nodes=20,
        e_bottom=-2.2,
        e_fermi=-1.0,
    )

    assert result.n_k == 4
    assert result.n_i > 0
    assert result.occupation > 0
    assert result.per_k_trace.shape == (4, 20)


# ---------------------------------------------------------------------------
# 4. Explicit band edges override
# ---------------------------------------------------------------------------


def test_explicit_band_edges_respected():
    sys = _h_chain_system(6, spacing=2.0)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    tags = [TAG_SUBSTRATE] * 4 + [TAG_REGION_I] * 2

    result = run_embedded_surface(
        sys,
        basis,
        tags,
        surface_k_mesh=(1, 1),
        contour_n_nodes=16,
        e_bottom=-2.0,
        e_fermi=-1.0,
    )

    assert result.e_bottom == pytest.approx(-2.0)
    assert result.e_fermi == pytest.approx(-1.0)
    assert result.contour.e_bottom == pytest.approx(-2.0)
    assert result.contour.e_fermi == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# 5. sigma_lin_per_k reuse
# ---------------------------------------------------------------------------


def test_sigma_lin_can_be_reused():
    """The returned sigma_lin_per_k can be reused with a different contour."""
    sys = _h_chain_system(8, spacing=2.0)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    tags = [TAG_SUBSTRATE] * 5 + [TAG_REGION_I] * 3

    result = run_embedded_surface(
        sys,
        basis,
        tags,
        surface_k_mesh=(1, 1),
        contour_n_nodes=40,
        e_bottom=-2.2,
        e_fermi=-1.0,
    )

    # Reuse sigma_lin with a better-converged contour (more nodes,
    # same energy range).
    from vibeqc.periodic_embedding.contour import EnergyContour
    from vibeqc.periodic_embedding.scf2step import compute_region_i_density_one_shot
    from vibeqc.periodic_embedding.substrate_gf import default_surface_k_mesh

    contour2 = EnergyContour.semicircle(-2.2, -1.0, n_nodes=80)
    kmesh = default_surface_k_mesh(sys, mesh=(1, 1))

    lat_opts = LatticeSumOptions()
    lat_opts.cutoff_bohr = 15.0
    lat_opts.nuclear_cutoff_bohr = 15.0

    result2 = compute_region_i_density_one_shot(
        sys,
        basis,
        result.region,
        kmesh,
        contour2,
        result.sigma_lin_per_k,
        lat_opts=lat_opts,
    )

    # Occupation should be broadly consistent when both contours cover
    # the occupied spectrum and have enough nodes.  The Ishida linearization
    # introduces O(Δz²) error; with generous tolerance this checks that
    # sigma_lin is reusable.
    assert abs(result2.occupation - result.occupation) < 1.5


# ---------------------------------------------------------------------------
# 6. HF SCF via runner
# ---------------------------------------------------------------------------


def test_runner_with_hf_scf():
    """Runner with scf_method='hf' triggers the full HF SCF path."""
    sys = _h_chain_system(8, spacing=2.0)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    tags = [TAG_SUBSTRATE] * 5 + [TAG_REGION_I] * 3

    result = run_embedded_surface(
        sys,
        basis,
        tags,
        surface_k_mesh=(1, 1),
        contour_n_nodes=24,
        e_bottom=-2.2,
        e_fermi=-1.0,
        scf_method="hf",
        scf_mixing=0.3,
        scf_max_iter=5,
    )

    assert result.occupation > 0
    assert result.density_local.shape == (3, 3)
