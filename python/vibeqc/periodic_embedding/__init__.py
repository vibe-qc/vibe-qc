"""Green's-function embedding for adsorbates on periodic slabs.

This subpackage implements the dilute-limit adsorbate-embedding scheme:
an adsorbate plus a small patch of substrate is treated as a localized,
non-periodic perturbation embedded into the laterally periodic,
semi-infinite clean surface. Two conceptually separate layers:

* **Layer A -- surface-normal embedding (Inglesfield).** Replace the
  finite slab in *z* with a true semi-infinite substrate by adding an
  energy-dependent, non-local embedding potential ``Sigma_emb`` on a
  dividing plane *S*. Lateral (x, y) periodicity is kept via Bloch
  k-points in the 2D surface Brillouin zone. The substrate Green
  function feeding ``Sigma_emb`` is built by principal-layer decimation
  (:mod:`~vibeqc.periodic_embedding.decimation`).

* **Layer B -- lateral impurity embedding (Dyson).** Take the clean
  surface Green function ``G0`` from Layer A and introduce the adsorbate
  + perturbed surface atoms as a localized ``Delta_V`` confined to a
  finite real-space region. Solve ``G = G0 + G0 Delta_V G`` on that
  region (:mod:`~vibeqc.periodic_embedding.dyson`). This breaks lateral
  translational symmetry locally and reaches the isolated adsorbate.

Numerical backbone (shared by both layers):

* density by complex-energy contour integration
  (:mod:`~vibeqc.periodic_embedding.contour`),
* energy-linearized embedding potential
  (:mod:`~vibeqc.periodic_embedding.embedding_potential`),
* adsorption energetics through Lloyd's formula
  (:mod:`~vibeqc.periodic_embedding.lloyd`).

Status: the contour-integration + Lloyd-formula + Sancho-Rubio
machinery is validated on the analytic 1D semi-infinite tight-binding
substrate (:mod:`~vibeqc.periodic_embedding.models.tight_binding_1d`,
exercised by ``tests/test_periodic_embedding_1d.py``).  Five 3D
Gaussian-basis modules are landed and green:

* :mod:`~vibeqc.periodic_embedding.region` -- atom-tag -> AO + dividing-plane partition
* :mod:`~vibeqc.periodic_embedding.substrate_gf` -- principal-layer Sancho-Rubio GF
* :mod:`~vibeqc.periodic_embedding.surface_sigma` -- surface-projected S_emb in region-I basis
* :mod:`~vibeqc.periodic_embedding.scf2step` -- one-shot embedded density via contour + BZ
* :mod:`~vibeqc.periodic_embedding.runner` -- ``run_embedded_surface()`` entry point
* :mod:`~vibeqc.periodic_embedding.layer_b` -- Layer B Dyson + Lloyd adsorption energy

The Hartree + XC SCF loop and the ASE calculator surface are follow-ups.
The end-to-end pipeline from tags -> region -> S_emb -> G0 -> ΔV -> Lloyd is
operational.  See ``handovers/HANDOVER_GF_EMBEDDING.md``.

References
----------
* J. E. Inglesfield, J. Phys. C 14, 3795 (1981), doi:10.1088/0022-3719/14/26/015.
* H. Ishida, Phys. Rev. B 63, 165409 (2001), doi:10.1103/PhysRevB.63.165409.
* M. P. Lopez Sancho et al., J. Phys. F 15, 851 (1985), doi:10.1088/0305-4608/15/4/009.
"""

from __future__ import annotations

from .contour import EnergyContour
from .decimation import sancho_rubio_surface_gf
from .dyson import dyson_solve
from .embedded_localise import EmbeddedLocalisedResult, embedded_localise
from .embedding_potential import EmbeddingPotential, LinearizedEmbeddingPotential
from .layer_b import (
    LayerBResult,
    compute_layer_b_delta_v,
    compute_layer_b_mock_delta_v,
    make_ghost_clean_system,
)
from .ldos_grid import compute_ldos_grid, save_embedded_ldos
from .lloyd import (
    lloyd_band_energy_change,
    lloyd_friedel_sum,
    lloyd_integrated_dos_change,
)
from .region import TAG_DELTA_V, TAG_REGION_I, TAG_SUBSTRATE, RegionPartition
from .ase_calc import EmbeddedSurfaceCalculator
from .runner import (
    EmbeddedSurfaceExperimentalWarning,
    EmbeddedSurfaceResult,
    run_embedded_surface,
)
from .scf2step import (
    EmbeddedDensityResult,
    SubstrateMeanField,
    build_sigma_lin_per_k,
    build_substrate_scf_potential,
    compute_region_i_density_one_shot,
    compute_region_i_density_scf,
    compute_region_i_density_scf_hf,
    compute_region_i_density_scf_uhf,
    compute_region_i_gf_at_kz,
)
from .substrate_gf import (
    LayerPartition,
    build_substrate_surface_gf,
    default_surface_k_mesh,
)
from .surface_sigma import build_surface_sigma, linearize_surface_sigma
from .xc import build_vxc_from_density

__all__ = [
    "EnergyContour",
    "sancho_rubio_surface_gf",
    "dyson_solve",
    "EmbeddingPotential",
    "LinearizedEmbeddingPotential",
    "lloyd_integrated_dos_change",
    "lloyd_band_energy_change",
    "lloyd_friedel_sum",
    "RegionPartition",
    "TAG_SUBSTRATE",
    "TAG_REGION_I",
    "TAG_DELTA_V",
    "LayerPartition",
    "build_substrate_surface_gf",
    "default_surface_k_mesh",
    "build_surface_sigma",
    "linearize_surface_sigma",
    "EmbeddedDensityResult",
    "SubstrateMeanField",
    "build_sigma_lin_per_k",
    "build_substrate_scf_potential",
    "compute_region_i_density_one_shot",
    "compute_region_i_density_scf",
    "compute_region_i_density_scf_hf",
    "compute_region_i_density_scf_uhf",
    "compute_region_i_gf_at_kz",
    "EmbeddedSurfaceResult",
    "EmbeddedSurfaceExperimentalWarning",
    "EmbeddedSurfaceCalculator",
    "run_embedded_surface",
    "LayerBResult",
    "compute_layer_b_mock_delta_v",
    "compute_layer_b_delta_v",
    "make_ghost_clean_system",
    "compute_ldos_grid",
    "save_embedded_ldos",
    "build_vxc_from_density",
    "EmbeddedLocalisedResult",
    "embedded_localise",
]
