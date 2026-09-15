"""GPW Hartree-J builder -- Gaussian-and-plane-waves periodic SCF.

The :class:`GpwJBuilder` orchestrates *Gaussian-density
collocation* -> *FFT-Poisson* -> *AO-basis projection* into a
Hartree-J matrix on a smooth real-space grid.  The module also
provides the full GPW SCF entries (:func:`run_periodic_rhf_gpw` /
:func:`run_periodic_rks_gpw_multi_k` /
:func:`run_periodic_uhf_gpw` / :func:`run_periodic_uks_gpw`),
reachable end-to-end via ``PeriodicJKMethod.GPW``
(``run_periodic_job(jk_method="gpw")``).

The route reproduces the molecular RHF limit to sub-microHa on
vacuum-padded cells.  All-electron CP2K parity on tight ionic
crystals awaits the GAPW augmentation (the open roadmap item in
``docs/design_periodic_gapw.md``).

The Hartree-J build factors into three steps -- Lippert, Hutter &
Parrinello, *Mol. Phys.* **92**, 477 (1997),
doi:10.1080/00268979709482119:

1. **Density collocation.** Eq. (12) of Lippert-Hutter 1997:
   ``rho(r) = Sum_munu D_munu chi_mu(r) chi_nu(r)`` on every grid
   point via C++ ``evaluate_ao`` with periodic-image folding.

2. **FFT-Poisson.** Eq. (13): V_H(G) = 4pi/G^2 rho(G) with
   V(G=0) pinned to zero (periodic-neutral convention).  Solved
   via the C++ ``solve_poisson_coulomb`` kernel.

3. **AO-basis projection.** Eq. (14):
   ``J_munu = dV Sum_g chi_mu(r_g) chi_nu(r_g) V_H(r_g)``
   where dV = V_cell / N_grid.  The GAPW augmentation adds a
   per-atom radial x Lebedev correction on top for all-electron
   accuracy (``periodic_gapw_augment``, behind
   ``GAPWExperimentalWarning``).

Shipped SCF surface (v0.12):

* Gamma-only RHF / RKS / UHF / UKS with DIIS (default-on),
  density damping, analytic forces, finite-difference Hessians.
* Multi-k pure-DFT RKS via ``run_periodic_rks_gpw_multi_k``.
* C++ SCF host via ``periodic_gapw_cpp_host`` (EDIIS+DIIS).
* Both V_ne conventions: ``"ewald"`` (default) and
  ``"smeared_erfc"`` (CP2K-style).
* D3(BJ) dispersion, DFT+U, Fermi-Dirac smearing.

Not yet shipped (all behind ``GAPWExperimentalWarning``):

* Range-separated hybrids
  (``periodic_gapw_range_sep``).
* Orbital Transformation (OT) SCF solver
  (``periodic_gapw_ot``).
* Per-AO-pair compact-support optimisation (needed for
  production-scale cells).

See :mod:`vibeqc.periodic_gapw_grid` and
``docs/user_guide/gapw.md`` for the user-facing documentation.
"""

from __future__ import annotations

from .guess import periodic_guess_capabilities

from .guess import periodic_result_selection

import hashlib
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np

from . import _vibeqc_core as _core
from .memory import (
    _COMPACT_BLOCH_AO_BATCH_BYTES,
    _COMPACT_BLOCH_AO_CACHE_BYTES,
)
from .periodic_gapw_grid import GAPWExperimentalWarning, PlaneWaveGrid
from .periodic_screened_exchange import reject_unscreened_range_separated
from .progress import ProgressLogger, resolve_progress
from .smearing import (
    SmearingOptions as _SmearingOptions,
)
from .smearing import (
    apply_smearing as _apply_smearing,
)

__all__ = [
    "GpwJBuilder",
    "GpwCollocationCache",
    "build_gpw_collocation_cache",
    "GpwEnergyBreakdown",
    "GpwScfResult",
    "GpwMultiKScfResult",
    "collocate_density_on_grid",
    "project_potential_to_ao",
    "bloch_ao_on_grid",
    "collocate_bloch_density_on_grid",
    "compute_bloch_kinetic_energy_density",
    "project_potential_to_bloch_ao",
    "project_vtau_to_bloch_ao",
    "compute_j_via_gpw",
    "evaluate_gpw_energy",
    "run_periodic_rhf_gpw",
    "run_periodic_rks_gpw",
    "run_periodic_rks_gpw_multi_k",
]


def _warn_experimental(detail: str) -> None:
    warnings.warn(
        f"GPW J build: {detail}. "
        f"See docs/design_periodic_gapw.md and docs/user_guide/gapw.md.",
        category=GAPWExperimentalWarning,
        stacklevel=3,
    )


# Number of periodic image shells summed when evaluating AOs on the
# real-space grid. ``1`` -> the 3^3 = 27 cells centred on the home cell,
# which captures the full Gaussian tail of any STO-3G / valence AO whose
# centre sits anywhere in a cell of side ≳ 6 bohr (a tail crossing two
# cell boundaries is ∝ exp(-a.(2L)^2) and negligible). Diffuse bases in
# very tight cells may need 2; the GPW route is experimental and the
# compact-support optimisation (M2c) will revisit the truncation.
_AO_IMAGE_RADIUS = 1

# Multi-k GPW has two collocation regimes. Vacuum-padded / molecular-limit
# cells fold to a single Gamma real-space density on the smooth grid (the AO
# tails do not overlap across cells, so the Gamma-summed density matrix is the
# whole story). Compact crystals -- where AOs in neighbouring cells overlap --
# need the full Bloch real-space density
#     rho(r) = sum_k w_k sum_mn D_mn(k) chi_{m,k}(r) chi_{n,k}(r)*
# and a per-k projection of the smooth potential back onto Bloch AOs. Using the
# Gamma-folded path on a compact cell drops the inter-cell Bloch phases and
# drives an unphysical Hartree/XC field (the P07 Si (4,4,4) runaway). The
# thresholds below classify which regime a cell falls in.
_MULTIK_GPW_VACUUM_AXIS_BOHR = 20.0
_MULTIK_GPW_MOLECULAR_LIMIT_BOHR3_PER_ATOM = 800.0

# Density screen for meta-GGA XC on the uniform FFT grid: grid points with
# rho below this contribute zero exc AND zero potential in every channel.
# See the screen block in _evaluate_xc_on_grid for the rationale (uniform
# voxels have no Becke-weight suppression of near-vacuum libxc divergences).
_MGGA_DENSITY_SCREEN = 1e-8

# Fail-closed stiffness bounds for the meta-GGA potential on the uniform
# grid (post-screen). Physical Fock-scale potentials sit at |v_tau| ~ O(1)
# and |v_sigma| well below 1e8 even at screened-boundary vacuum points
# (r2SCAN/H2 measures ~5e7); TPSS-class iso-orbital divergence lands at
# ~1e17 / ~1e7. See the guard in _evaluate_xc_on_grid.
_MGGA_VSIGMA_STIFF_MAX = 1e12
_MGGA_VTAU_STIFF_MAX = 1e4

# Linear-dependence threshold for the compact Bloch path's canonical
# orthogonaliser. Diffuse molecular bases (def2-SVP) on a dense crystal drive
# the Bloch overlap S(k) singular/indefinite; directions with a
# diag-normalised S(k) eigenvalue below this are projected out (PySCF's
# canonical_orth_ default is 1e-7, but compact all-electron cells need a looser
# cut to shed the genuinely negative-definite tail from truncated overlap
# lattice sums).
_COMPACT_GPW_LINDEP_THRESHOLD = 1e-4


# Any NEGATIVE eigenvalue of the Bloch overlap is a bug, never a basis-set
# property: S(k) is the Gram matrix of the Bloch orbitals
# phi_mu^k(r) = sum_R exp(i k . R) chi_mu(r - R), so it is positive
# semidefinite whenever the lattice sum is complete. This tolerance is
# roundoff on the diag-normalised matrix, not a physics knob.
_BLOCH_OVERLAP_PSD_TOL = 1e-8


def _assert_bloch_overlap_psd(S_k, k_cart, cutoff_bohr: float) -> None:
    """Fail closed when the Bloch overlap at one k is not positive definite.

    ``S(k) = sum_R exp(i k . R) S(R)`` is the Gram matrix of the Bloch
    orbitals ``phi_mu^k(r) = sum_R exp(i k . R) chi_mu(r - R)``:

        <phi_mu^k | phi_nu^k> = N * sum_T exp(i k . T) S_{mu nu}(T)

    so it is positive semidefinite for every real k when the T sum is
    complete. A negative eigenvalue is therefore a truncation defect, never
    a property of the basis. Eigenvalues are taken on the diag-normalised
    matrix so the tolerance is scale free and matches what the canonical
    orthogonaliser thresholds on.

    NOT YET WIRED into the SCF drivers. With the shipped flat 15-bohr
    overlap lattice sum this guard refuses every compact-crystal GPW run,
    correctly but unhelpfully; it becomes the right gate once the cutoff is
    sized from the basis
    (:func:`vibeqc.lattice_screening.bloch_overlap_cutoff_bohr`), which is
    blocked on reconciling the GPW/GAPW stack's other real-space
    truncations. Tested in
    ``tests/test_periodic_gpw_compact_multik.py``; see
    ``handovers/HANDOVER_GPW_MULTIK_OVERLAP_PSD.md``.
    """
    d = np.real(np.asarray(S_k).diagonal())
    if np.all(d > 0.0):
        scale = 1.0 / np.sqrt(d)
        S_test = np.asarray(S_k) * np.outer(scale, scale)
    else:
        S_test = np.asarray(S_k)
    e_min = float(np.linalg.eigvalsh(S_test).min())
    if e_min >= -_BLOCH_OVERLAP_PSD_TOL:
        return
    raise RuntimeError(
        "multi-k GPW: the Bloch overlap S(k) is not positive definite at "
        f"k = {np.asarray(k_cart)} (min eigenvalue {e_min:+.6e}, tolerance "
        f"{-_BLOCH_OVERLAP_PSD_TOL:+.1e}). S(k) is a Gram matrix, so this is "
        "a truncation defect in the real-space sum "
        "S(k) = sum_R exp(i k . R) S(R), not a basis-set property. The "
        f"overlap lattice sum was cut at {float(cutoff_bohr):.2f} bohr. "
        "Every band energy and occupation derived from an indefinite metric "
        "is meaningless."
    )


def multik_one_electron_lattice_options(basis, system):
    """:class:`LatticeSumOptions` for the multi-k GPW one-electron sums.

    Named so the Bloch metric produced by ``run_periodic_rks_gpw_multi_k``
    can be exercised directly by a regression test without running an SCF:
    the overlap / kinetic lattice sums built with these options are what
    ``S(k) = sum_R exp(i k . R) S(R)`` is assembled from, and a positive
    definite S(k) at every k is a precondition for the whole driver.

    The cutoff is the basis-derived
    :func:`vibeqc.lattice_screening.bloch_overlap_cutoff_bohr` -- the
    overlap-magnitude criterion (Pisani & Dovesi, Int. J. Quantum Chem.
    17, 501 (1980), doi:10.1002/qua.560170311, Sec. 4: series truncation
    governed by the pair-overlap tolerance) -- NOT the bare flat 15-bohr
    ``LatticeSumOptions`` default, which is far inside the tail of a
    diffuse molecular basis on a compact crystal and made S(k) INDEFINITE
    away from the zone centre (measured min eig on the (4,4,4) mesh at
    def2-SVP: LiF -1.506940, MgO -0.198158, Si -0.155202 -- the
    GPW-MULTIK-OVERLAP-NOT-PSD defect).

    Every operator entering the multi-k pencil ``H(k) C = S(k) C eps``
    must be lattice-summed over this SAME (or an equally converged) R-set;
    build the V_ne options with :func:`multik_v_ne_lattice_options` and
    never from a fresh ``LatticeSumOptions``. See
    :func:`multik_v_ne_lattice_options` for why mixing R-sets collapses
    the total energy (GPW-SK-HK-RSET-INCONSISTENT).
    """
    from .lattice_screening import bloch_overlap_cutoff_bohr

    opts = _core.LatticeSumOptions()
    opts.cutoff_bohr = bloch_overlap_cutoff_bohr(basis, system)
    return opts


def multik_v_ne_lattice_options(basis, system):
    """:class:`LatticeSumOptions` for the multi-k GPW nuclear-attraction sum.

    Same ``cutoff_bohr`` as :func:`multik_one_electron_lattice_options`,
    with the EWALD_3D dispatch selected. The shared cutoff is load-bearing,
    not stylistic: the far-field one-electron lattice terms become
    proportional to the overlap block, "the coefficient of proportionality
    being independent of omega_1, omega_2 and g. Such terms do not affect
    eigenvectors and displace eigenvalues by a constant inessential
    quantity" (Pisani & Dovesi, Int. J. Quantum Chem. 17, 501 (1980),
    doi:10.1002/qua.560170311, p. 510; their Eq. (30) sums T^g, Z^g and
    the far-field Delta^g over the SAME g set as the Fock Bloch sum).
    That cancellation holds exactly when H(k) and S(k) drop the same tail
    -- vibe-qc's own EWALD_3D V_ne carries the structure explicitly, as
    the G = 0 correction ``-v_short(G=0) * S_mu_nu(g)`` over V's cell
    list (:func:`vibeqc.periodic_v_ne.compute_v_ne_ewald_3d_ft_lattice`).

    Truncating V_ne on its own shorter R-set while S/T are converged makes
    the pencil inconsistent: measured on LiF/def2-SVP (2,2,2), the Hcore
    band-sum proxy collapses by -17.6 Ha (S/T at 51.53 bohr, V at 15)
    against -0.02 Ha between the two uniformly-summed pencils, and the
    full SCF reproduces it as the cutoff-independent -18.2 Ha over-binding
    of the 2026-08-05 discriminator (vq job e7635f8e7d05). Pinned by
    ``tests/test_periodic_gpw_rset_consistency.py``.
    """
    opts = multik_one_electron_lattice_options(basis, system)
    opts.coulomb_method = _core.CoulombMethod.EWALD_3D
    return opts


def _multik_gpw_is_molecular_limit(system) -> bool:
    """Classify a cell as molecular-limit (True) vs compact crystal (False).

    Molecular-limit cells have a vacuum-scale lattice vector or a large
    volume-per-atom, so AOs in neighbouring cells do not overlap on the
    smooth grid and the Gamma-summed density matrix collocated with periodic
    (image-folded) Gamma AOs is exact. Compact crystals fail both tests and
    are routed to the Bloch real-space density / per-k projection path
    (:func:`collocate_bloch_density_on_grid` /
    :func:`project_potential_to_bloch_ao`).
    """
    lattice = np.asarray(system.lattice, dtype=float)
    # ``system.lattice`` columns are the Cartesian lattice vectors (C++
    # ``PeriodicSystem.lattice``: "Columns = Cartesian lattice vectors"), so
    # the axis lengths are the column norms -- ``axis=0``. This must match how
    # the grid interprets the same matrix: ``make_grid`` /
    # ``PlaneWaveGrid.axis_lengths_bohr`` / ``_ao_values_on_grid`` all read the
    # columns as a1,a2,a3. Using ``axis=1`` (row norms) agrees only for a
    # symmetric lattice matrix; on a skewed slab/rod or a general triclinic
    # cell the row norms differ from the column norms, which mis-classifies a
    # compact crystal as molecular-limit and routes it to the Gamma-folded
    # density path -- the unphysical Hartree/XC runaway documented above (the
    # P07 Si (4,4,4) runaway).
    lengths = np.linalg.norm(lattice, axis=0)
    atoms = list(system.unit_cell)
    volume = float(abs(np.linalg.det(lattice)))
    volume_per_atom = volume / max(len(atoms), 1)
    return (
        bool(np.any(lengths > _MULTIK_GPW_VACUUM_AXIS_BOHR))
        or volume_per_atom > _MULTIK_GPW_MOLECULAR_LIMIT_BOHR3_PER_ATOM
    )


def _ao_values_on_grid(
    basis,
    points: np.ndarray,
    lattice_bohr: np.ndarray,
    *,
    image_radius: int = _AO_IMAGE_RADIUS,
) -> np.ndarray:
    """Evaluate the AO basis on ``points`` summed over periodic images.

    Returns the Γ-point periodic AO values

        ``chi_mu(r_g) = S_R chi_mu^mol(r_g - R)``

    where ``R`` runs over the lattice vectors with each fractional
    component in ``[-image_radius, +image_radius]`` and ``chi_mu^mol`` is
    the bare molecular AO. Evaluating ``chi_mu^mol`` at the shifted point
    ``r_g - R`` is identical to evaluating the ``+R`` periodic image of
    AO ``mu`` at ``r_g`` (shifting the sample point by ``-R`` == shifting
    the AO centre by ``+R``), so this needs only the lattice -- no
    :class:`PeriodicSystem`. It is the grid-native twin of
    :func:`vibeqc.ewald_j.evaluate_ao_periodic`.

    Why this matters: the bare molecular :func:`evaluate_ao` lets an AO
    whose centre sits near a cell face leak its Gaussian tail out of the
    box, so a density collocated from it integrates to *less* than
    ``tr(D.S)`` electrons -- the v0.10.x GPW translation-invariance break
    (∫r ≈ 0.4 e instead of 2 e for H₂/STO-3G placed at a cell corner).
    The image sum folds the escaping tail back in, so ∫r = tr(D.S) = N
    for any placement of the atoms in the cell.

    ``image_radius = 0`` reproduces the bare molecular evaluator (the
    unfixed behaviour) and is kept only as an escape hatch.
    """
    pts = np.asarray(points, dtype=float)
    chi = _core.evaluate_ao(basis, pts)  # home cell, image R = 0
    if image_radius <= 0:
        return chi
    L = np.asarray(lattice_bohr, dtype=float)  # columns a1, a2, a3
    for ix in range(-image_radius, image_radius + 1):
        for iy in range(-image_radius, image_radius + 1):
            for iz in range(-image_radius, image_radius + 1):
                if ix == 0 and iy == 0 and iz == 0:
                    continue
                shift = ix * L[:, 0] + iy * L[:, 1] + iz * L[:, 2]
                chi += _core.evaluate_ao(basis, pts - shift)
    return chi


@dataclass(frozen=True)
class GpwCollocationCache:
    """Iteration-invariant grid machinery for the GPW/GAPW Hartree-J + XC
    build -- cached once per SCF, contracted with the density per iteration.

    Holds the two pieces that depend only on ``(basis, grid)`` and never on
    the density:

    * ``chi`` -- optional ``(n_grid, n_basis)`` periodic AO values
      ``chi_mu(r_g) = S_R chi_mu^mol(r_g - R)`` (see :func:`_ao_values_on_grid`).
      Building it costs 27 image ``evaluate_ao`` calls over the whole grid --
      profiled at 95-100 % of a single ``collocate``/``project`` (the
      density contraction is ~1 % on top). It is iteration-invariant, so an
      SCF that rebuilds it every iteration (the XC path did: r collocate,
      V_xc LDA + GGA projections, ~3-4x per iter) wastes all but the first.
    * ``recip`` -- ``(nx, ny, nz, 3)`` reciprocal-space G-mesh
      (:meth:`PlaneWaveGrid.reciprocal_vectors`), reused by the GGA flux
      divergence, the meta-GGA kinetic-energy density, and the spectral
      density gradient.

    Mirrors :class:`vibeqc.ewald_composed.EwaldJFTGammaCache` (the EWALD E2
    analytic-FT J cache) and the in-builder ``GpwJBuilder._chi_cache``: build
    once via :func:`build_gpw_collocation_cache`, then every primitive that
    would otherwise call :func:`_ao_values_on_grid` /
    :meth:`PlaneWaveGrid.reciprocal_vectors` takes ``cache=`` and skips the
    rebuild. The contraction is **bit-identical** to the uncached path -- the
    cached ``chi`` is the same float array the inline call would produce, and
    ``cache=None`` everywhere falls back to building inline.

    The closed-shell compact multi-k path sets ``chi=None`` and uses only
    ``recip`` because its complex Bloch AOs are point-batched instead of
    retained. Lifetime is one SCF run on a fixed geometry. It is valid only for
    the ``(basis, grid)`` it was built from; a finite-difference gradient that
    displaces atoms (a fresh basis) MUST build a new cache -- never reuse one
    across geometries.
    """

    chi: Optional[np.ndarray]
    recip: np.ndarray


def build_gpw_collocation_cache(
    basis,
    grid: PlaneWaveGrid,
    *,
    image_radius: int = _AO_IMAGE_RADIUS,
) -> GpwCollocationCache:
    """Build the iteration-invariant :class:`GpwCollocationCache` for
    ``(basis, grid)`` -- the AO-on-grid table chi and the G-mesh.

    Runs the expensive front half (the 27-image ``evaluate_ao`` collocation)
    exactly once. Pass the result into :func:`collocate_density_on_grid`,
    :func:`project_potential_to_ao`, :class:`GpwJBuilder`, and the XC
    projectors via their ``cache=`` argument to reuse it across SCF
    iterations. Bit-identical to the uncached path.
    """
    r_flat = grid.cartesian_coords().reshape(-1, 3)
    chi = _ao_values_on_grid(
        basis, r_flat, grid.lattice_bohr, image_radius=image_radius
    )
    recip = grid.reciprocal_vectors()
    return GpwCollocationCache(chi=chi, recip=recip)


def _resolve_chi(
    cache: Optional[GpwCollocationCache],
    basis,
    grid: PlaneWaveGrid,
) -> np.ndarray:
    """Return the cached chi if a cache is supplied, else build it inline.

    The inline branch is the exact computation every grid primitive used
    before caching, so ``cache=None`` is bit-identical to the pre-cache
    behaviour and one-shot callers are unaffected.
    """
    if cache is not None and cache.chi is not None:
        return cache.chi
    r_flat = grid.cartesian_coords().reshape(-1, 3)
    return _ao_values_on_grid(basis, r_flat, grid.lattice_bohr)


def _resolve_recip(
    cache: Optional[GpwCollocationCache],
    grid: PlaneWaveGrid,
) -> np.ndarray:
    """Return the cached G-mesh if a cache is supplied, else build it inline."""
    if cache is not None:
        return cache.recip
    return grid.reciprocal_vectors()


def collocate_density_on_grid(
    basis,
    density_matrix: np.ndarray,
    grid: PlaneWaveGrid,
    *,
    cache: Optional[GpwCollocationCache] = None,
) -> np.ndarray:
    """Build r(r) on ``grid`` from an AO density matrix.

    Parameters
    ----------
    basis
        :class:`vibeqc.BasisSet`. The same object the density was
        derived from; the AO ordering must match (vibe-qc's
        standard shell-major ordering).
    density_matrix
        ``(n_basis, n_basis)`` AO density matrix. For closed-shell
        RHF / RKS this is ``2 . C_occ @ C_occ.T``; for open-shell
        UHF / UKS the caller would pass ``D_a + D_b``.
    grid
        :class:`PlaneWaveGrid` defining the cell + sampling. Same
        lattice the basis was built on; mismatched lattices give
        unphysical results.

    Returns
    -------
    rho
        ``(nx, ny, nz)`` real-valued density on the grid, in
        ``e / bohr^3``. Integrates to ``tr(D @ S)`` (the total number
        of electrons for a closed-shell density) to within the
        finite-grid quadrature error -- for *any* placement of the
        atoms in the cell, including AOs centred on a face / edge /
        corner, because the AO values are summed over periodic images
        (see :func:`_ao_values_on_grid`). Before that image sum was
        wired in, an edge-centred AO leaked its Gaussian tail out of
        the box and the density under-integrated (∫r ≈ 0.4 e for
        H₂/STO-3G at a corner instead of 2 e).

    Notes
    -----
    Cost is ``O(n_basis^2 . N_grid)`` for the density contraction plus
    ``O((2.image_radius+1)^3 . n_basis . N_grid)`` for the image-summed
    AO evaluation -- no truncation around the shell-pair centre yet. M2c
    follows up with the compact-support optimisation; this is the
    reference implementation that the optimisation is validated against.
    """
    D = np.asarray(density_matrix, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(
            f"density_matrix must be a square (n_basis, n_basis) "
            f"array; got shape {D.shape}"
        )
    n_basis = D.shape[0]
    if n_basis != basis.nbasis:
        raise ValueError(
            f"density_matrix has n_basis={n_basis} but the supplied "
            f"basis has nbasis={basis.nbasis}"
        )

    # Periodic AO image sum -- NOT the bare molecular evaluate_ao -- so
    # the Gaussian tail of an edge-centred AO is folded back into the
    # cell and r integrates to tr(D.S) = N for any atom placement.
    # ``cache`` (a GpwCollocationCache) reuses chi across SCF iterations;
    # ``cache=None`` rebuilds it inline (bit-identical, one-shot path).
    chi = _resolve_chi(cache, basis, grid)  # (n_grid, n_basis)
    # r(r_g) = S_muν D_muν . chi_mu(r_g) . chi_ν(r_g)
    #       = einsum('gm,mn,gn->g', chi, D, chi)
    # The contraction order ('gm,mn->gn' then 'gn,gn->g') keeps
    # intermediate (n_grid, n_basis), which for the M1 demo systems
    # (nbasis <= ~50, N_grid <= ~64^3 = 262 144) stays well inside RAM.
    rho_flat = np.einsum("gm,mn,gn->g", chi, D, chi, optimize=True)
    return rho_flat.reshape(grid.shape)


def project_potential_to_ao(
    basis,
    potential: np.ndarray,
    grid: PlaneWaveGrid,
    *,
    cache: Optional[GpwCollocationCache] = None,
) -> np.ndarray:
    """Project a real-space potential ``V(r)`` back onto the AO
    basis to produce a one-electron matrix.

    Used as the second half of the GPW J build:

        ``J_muν = ∫ chi_mu(r) chi_ν(r) V_H(r) dr``

    where ``V_H`` is the Hartree potential :func:`solve_poisson_coulomb`
    returned. The same primitive also computes ``V_xc`` projections
    (the M3 work uses it).

    Parameters
    ----------
    basis
        :class:`vibeqc.BasisSet`. AO ordering matches
        :func:`collocate_density_on_grid`.
    potential
        ``(nx, ny, nz)`` real potential on the grid (Hartree).
    grid
        :class:`PlaneWaveGrid` defining the cell + sampling.

    Returns
    -------
    M_ao
        ``(n_basis, n_basis)`` symmetric matrix of integrated AO
        products against the potential.
    """
    V = np.asarray(potential, dtype=float)
    if V.shape != grid.shape:
        raise ValueError(f"potential shape {V.shape} doesn't match grid {grid.shape}")
    # Periodic AO image sum (matches collocate_density_on_grid) so the
    # projected J_muν = ∫ chi_mu chi_ν V uses the same periodic AOs that built
    # r -- otherwise an edge-centred AO truncates J for off-centre atoms.
    # ``cache`` reuses chi across iterations; ``cache=None`` rebuilds inline.
    chi = _resolve_chi(cache, basis, grid)  # (n_grid, n_basis)
    V_flat = V.reshape(-1)
    # M_muν = ΔV . S_g chi_mu(g) chi_ν(g) V(g)
    #     = einsum('gm,gn,g->mn', chi, chi, V) . ΔV
    # Symmetric by construction (mu <-> ν permutation of the trace).
    M = (
        np.einsum("gm,gn,g->mn", chi, chi, V_flat, optimize=True)
        * grid.voxel_volume_bohr3
    )
    # Force exact symmetry -- floating-point sums may produce a
    # ~1e-15 asymmetry that callers like a symmetric eigensolver
    # would amplify.
    return 0.5 * (M + M.T)


# ---------------------------------------------------------------------------
# Bloch (compact multi-k) real-space density collocation + per-k projection
# ---------------------------------------------------------------------------
#
# The Gamma primitives above collapse the multi-k density to a single real
# Gamma-summed density matrix -- correct only when neighbouring cells do not
# overlap on the grid (the molecular limit). For compact crystals the density
# is the full Bloch sum over the k-mesh, built from the per-k Bloch AOs
#     chi_{m,k}(r) = sum_T exp(i k.T) chi_m(r - T)
# (the ``evaluate_bloch_ao`` C++ kernel; same +i k.T convention as
# ``bloch_sum`` so D(k), S(k) and the projected potential share one gauge).


def bloch_ao_on_grid(basis, grid_points, k_cart, lattice_translations):
    """Bloch-summed AO table chi_{m,k}(r_g) on a flat (n_grid, 3) point set.

    Thin wrapper over the C++ ``evaluate_bloch_ao`` kernel

        chi_{m,k}(r_g) = sum_T exp(i k.T) chi_m(r_g - T)

    returning the complex ``(n_grid, n_basis)`` table for one k-point. The
    +i k.T phase matches :func:`bloch_sum` (used for T(k), S(k), V_ne(k)), so
    the projection :func:`project_potential_to_bloch_ao` of a constant unit
    potential reproduces the Bloch overlap S(k) -- i.e. the density matrix
    D(k), the metric S(k) and the projected Fock share a single gauge.
    """
    pts = np.ascontiguousarray(grid_points, dtype=float)
    k = np.asarray(k_cart, dtype=float).reshape(3)
    Ts = np.ascontiguousarray(lattice_translations, dtype=float)
    return _core.evaluate_bloch_ao(basis, pts, k, Ts)


def collocate_bloch_density_on_grid(chi_k_list, density_matrices_k, weights, grid):
    """Real-space Bloch density on ``grid`` from the per-k density matrices.

        rho(r) = sum_k w_k sum_mn D_mn(k) chi_{m,k}(r) chi_{n,k}(r)*

    Parameters
    ----------
    chi_k_list
        Length-``n_k`` list of complex ``(n_grid, n_basis)`` Bloch AO tables
        (:func:`bloch_ao_on_grid`), one per k-point.
    density_matrices_k
        Length-``n_k`` list of complex Hermitian ``(n_basis, n_basis)`` AO
        density matrices ``D(k) = sum_i f_{i,k} C_{i,k} C_{i,k}^H``.
    weights
        Length-``n_k`` k-point weights (sum to 1 over the full BZ).
    grid
        :class:`PlaneWaveGrid`. ``rho`` is reshaped to ``grid.shape``.

    Returns
    -------
    rho
        ``(nx, ny, nz)`` real density in ``e / bohr^3``. Integrates to the
        electron count ``sum_k w_k tr(D(k) S(k))`` to within the finite-grid
        quadrature error, for a compact crystal where the Gamma-folded
        collocation would drop the inter-cell Bloch phases.
    """
    n_grid = chi_k_list[0].shape[0]
    rho_flat = np.zeros(n_grid, dtype=float)
    for chi_k, D_k, w in zip(chi_k_list, density_matrices_k, weights):
        # rho_k(g) = sum_mn chi[g,m] D_mn chi[g,n]* ; D Hermitian -> real part.
        left = chi_k @ np.asarray(D_k, dtype=complex)  # (g,n) = sum_m chi[g,m] D_mn
        rho_flat += float(w) * np.real(
            np.einsum("gn,gn->g", left, np.conj(chi_k), optimize=True)
        )
    return rho_flat.reshape(grid.shape)


def project_potential_to_bloch_ao(chi_k, potential, grid):
    """Project a real grid potential onto Bloch AOs at one k-point.

        V_mn(k) = int chi_{m,k}(r)* v(r) chi_{n,k}(r) dr

    Returns a complex Hermitian ``(n_basis, n_basis)`` matrix -- the per-k
    local Fock block (Hartree + XC local potential) in the same gauge as
    ``bloch_sum(S_lat, k)``. For ``v == 1`` this reduces to S(k).
    """
    V = np.asarray(potential, dtype=float)
    if V.shape != grid.shape:
        raise ValueError(
            f"potential shape {V.shape} doesn't match grid {grid.shape}"
        )
    v_flat = V.reshape(-1)
    # M_mn = dV . sum_g conj(chi[g,m]) v[g] chi[g,n]
    scaled = np.conj(chi_k) * v_flat[:, None]  # (g, m)
    M = (scaled.T @ chi_k) * grid.voxel_volume_bohr3
    # Enforce exact Hermiticity against floating-point asymmetry.
    return 0.5 * (M + M.conj().T)


def _compact_bloch_batch_points(n_basis: int) -> int:
    """Maximum grid points in one bounded complex Bloch-AO table."""
    bytes_per_point = max(1, int(n_basis)) * np.dtype(np.complex128).itemsize
    return max(1, _COMPACT_BLOCH_AO_BATCH_BYTES // bytes_per_point)


def _iter_bloch_ao_grid_batches(
    basis,
    grid_points,
    k_cart,
    lattice_translations,
):
    """Yield ``(slice, chi_k_batch)`` without a full grid x AO allocation."""
    points = np.asarray(grid_points, dtype=float)
    batch_points = _compact_bloch_batch_points(basis.nbasis)
    for start in range(0, points.shape[0], batch_points):
        stop = min(start + batch_points, points.shape[0])
        point_slice = slice(start, stop)
        yield point_slice, bloch_ao_on_grid(
            basis,
            points[point_slice],
            k_cart,
            lattice_translations,
        )


def _collocate_bloch_density_streaming(
    basis,
    grid_points,
    kpoints,
    lattice_translations,
    density_matrices_k,
    weights,
    grid,
):
    """Bloch density collocation with storage bounded independently of k."""
    rho_flat = np.zeros(len(grid_points), dtype=float)
    for k, D_k, w in zip(kpoints, density_matrices_k, weights):
        Dc = np.asarray(D_k, dtype=complex)
        for point_slice, chi_batch in _iter_bloch_ao_grid_batches(
            basis, grid_points, k, lattice_translations
        ):
            # Same GPW density contraction as Lippert-Hutter-Parrinello
            # (1997), Eqs. 16-18, evaluated on a point subset:
            # rho_k(g) = sum_mn chi_m,k(g) D_mn(k) chi_n,k(g)*.
            # A one-batch call is reference-identical to the dense helper.
            left = chi_batch @ Dc
            rho_flat[point_slice] += float(w) * np.real(
                np.einsum(
                    "gn,gn->g", left, np.conj(chi_batch), optimize=True
                )
            )
            # Loop variables otherwise retain the old arrays while the
            # generator constructs the next C++ batch, doubling the peak.
            del left, chi_batch
    return rho_flat.reshape(grid.shape)


def _project_potential_to_bloch_ao_streaming(
    basis,
    grid_points,
    k_cart,
    lattice_translations,
    potential,
    grid,
):
    """Project a local potential with a bounded Bloch-AO point batch."""
    V = np.asarray(potential, dtype=float)
    if V.shape != grid.shape:
        raise ValueError(
            f"potential shape {V.shape} doesn't match grid {grid.shape}"
        )
    v_flat = V.reshape(-1)
    M = np.zeros((basis.nbasis, basis.nbasis), dtype=complex)
    for point_slice, chi_batch in _iter_bloch_ao_grid_batches(
        basis, grid_points, k_cart, lattice_translations
    ):
        # Lippert-Hutter-Parrinello (1997), Eqs. 16-18, projected in
        # point batches: V_mn(k) = dV sum_g chi_m,k(g)* v(g) chi_n,k(g).
        # With v(g) == 1 the reference value is the finite-grid S(k).
        scaled = np.conj(chi_batch) * v_flat[point_slice, None]
        M += scaled.T @ chi_batch
        del scaled, chi_batch
    M *= grid.voxel_volume_bohr3
    return 0.5 * (M + M.conj().T)


def _bloch_ao_gradients_fft(chi_k, k_cart, grid_points, grid, *, recip=None):
    """Yield the three Cartesian gradient tables of the Bloch AOs at one k.

    chi_k is Bloch-periodic, chi_k(r + L) = exp(i k.L) chi_k(r), NOT
    lattice-periodic -- so differentiating its raw FFT is wrong (the FFT
    implicitly assumes periodicity and would alias the boundary phase).
    Unwrap the phase first: u_k(r) = exp(-i k.r) chi_k(r) IS lattice-
    periodic, so

        gradchi_k = exp(i k.r) (grad u_k + i k u_k)

    with grad u_k taken spectrally per AO column (batched, one direction
    at a time -- peak footprint one extra complex (n_grid, n_basis) table).
    """
    shape = grid.shape
    n_grid, n_basis = chi_k.shape
    k = np.asarray(k_cart, dtype=float).reshape(3)
    G = grid.reciprocal_vectors() if recip is None else recip
    phase = np.exp(1j * (grid_points @ k))  # (n_grid,)
    u = chi_k * np.conj(phase)[:, None]
    u_g = np.fft.fftn(u.reshape(shape + (n_basis,)), axes=(0, 1, 2))
    for d in range(3):
        du = np.fft.ifftn(
            1j * G[..., d, None] * u_g, axes=(0, 1, 2)
        ).reshape(n_grid, n_basis)
        yield phase[:, None] * (du + 1j * k[d] * u)


def compute_bloch_kinetic_energy_density(
    chi_k_list,
    density_matrices_k,
    weights,
    kpoints,
    grid_points,
    grid,
    *,
    recip=None,
) -> np.ndarray:
    """Bloch kinetic-energy density on the grid for a compact multi-k cell.

        t(r) = 1/2 sum_k w_k sum_mn D_mn(k) gradchi_{m,k}(r) . gradchi_{n,k}(r)*

    The multi-k sibling of :func:`_compute_kinetic_energy_density_fft`,
    consistent with :func:`collocate_bloch_density_on_grid`'s density
    convention (same D(k), weights, and chi tables; gradients via
    :func:`_bloch_ao_gradients_fft`).
    """
    n_grid = chi_k_list[0].shape[0]
    tau_flat = np.zeros(n_grid, dtype=float)
    for chi_k, D_k, w, k in zip(chi_k_list, density_matrices_k, weights, kpoints):
        Dc = np.asarray(D_k, dtype=complex)
        for gd in _bloch_ao_gradients_fft(chi_k, k, grid_points, grid, recip=recip):
            left = gd @ Dc  # (g, n) = sum_m gd[g,m] D_mn
            tau_flat += (0.5 * float(w)) * np.real(
                np.einsum("gn,gn->g", left, np.conj(gd), optimize=True)
            )
        del left, gd, chi_k
    return tau_flat.reshape(grid.shape)


def _compute_bloch_kinetic_energy_density_streaming(
    basis,
    density_matrices_k,
    weights,
    kpoints,
    grid_points,
    lattice_translations,
    grid,
    *,
    recip=None,
) -> np.ndarray:
    """Bloch meta-GGA ``tau`` retaining at most one complete k table.

    Spectral AO derivatives couple the whole FFT grid, so they cannot use the
    point batches of the LDA/GGA contractions.  Streaming over k still removes
    the former ``n_k`` memory factor while preserving the dense FFT operation.
    """
    tau_flat = np.zeros(len(grid_points), dtype=float)
    for D_k, w, k in zip(density_matrices_k, weights, kpoints):
        chi_k = bloch_ao_on_grid(
            basis, grid_points, k, lattice_translations
        )
        Dc = np.asarray(D_k, dtype=complex)
        for gd in _bloch_ao_gradients_fft(
            chi_k, k, grid_points, grid, recip=recip
        ):
            left = gd @ Dc
            tau_flat += (0.5 * float(w)) * np.real(
                np.einsum("gn,gn->g", left, np.conj(gd), optimize=True)
            )
        del left, gd, chi_k
    return tau_flat.reshape(grid.shape)


def project_vtau_to_bloch_ao(chi_k, k_cart, v_tau_grid, grid_points, grid, *, recip=None):
    """Project the meta-GGA t-potential onto Bloch AOs at one k-point.

        V_tau(k)[mn] = 1/2 int v_tau(r) gradchi_{m,k}(r)* . gradchi_{n,k}(r) dr

    Complex Hermitian; the Bloch sibling of :func:`_project_vtau_to_ao`
    (same generalized-KS matrix element, same 1/2 factor), in the same
    gauge as :func:`project_potential_to_bloch_ao`.
    """
    V = np.asarray(v_tau_grid, dtype=float)
    if V.shape != grid.shape:
        raise ValueError(
            f"v_tau shape {V.shape} doesn't match grid {grid.shape}"
        )
    w = (0.5 * grid.voxel_volume_bohr3) * V.reshape(-1)
    n_basis = chi_k.shape[1]
    M = np.zeros((n_basis, n_basis), dtype=complex)
    for gd in _bloch_ao_gradients_fft(chi_k, k_cart, grid_points, grid, recip=recip):
        M += (np.conj(gd) * w[:, None]).T @ gd
    return 0.5 * (M + M.conj().T)


def _xc_effective_potential_grid(rho_grid, grid, functional, *, tau=None, cache=None):
    """Local XC potential v_xc(r) as a single scalar grid field, plus E_xc.

    Assembles the LDA piece ``v_rho`` and -- for GGAs -- the
    integrated-by-parts gradient correction ``-div(2 v_sigma grad_rho)`` into
    one real potential on the grid. This is exactly the local multiplicative
    operator that :func:`_project_vxc_to_ao` projects onto real Gamma AOs, but
    returned as a field so the Bloch path can project it per-k alongside the
    Hartree potential in a single pass.

    Returns ``(e_xc, v_eff, v_tau_grid_or_None)``. For meta-GGA the caller
    MUST pass a precomputed ``tau=`` (on the compact Bloch path that is
    :func:`compute_bloch_kinetic_energy_density`) and is responsible for
    projecting the returned ``v_tau_grid`` with the gradient kernel
    (:func:`project_vtau_to_bloch_ao`) -- v_tau is NOT part of the local
    multiplicative ``v_eff``.
    """
    e_xc, v_xc_grid, v_sigma_grid, grad_rho, v_tau_grid = _evaluate_xc_on_grid(
        rho_grid, grid, functional, tau=tau, cache=cache
    )
    if v_tau_grid is not None and tau is None:
        raise NotImplementedError(
            "compact multi-k Bloch GPW: meta-GGA needs the Bloch kinetic "
            "energy density passed as tau= (per-k Bloch AO gradients); the "
            "Gamma t builder is not valid for a compact multi-k cell."
        )
    v_eff = np.array(v_xc_grid, dtype=float)
    if v_sigma_grid is not None and grad_rho is not None:
        # GGA gradient correction: v_gga(r) = -div(2 v_sigma grad_rho),
        # computed spectrally (matches _project_vxc_to_ao's integration by
        # parts, but kept as a grid field instead of projected on Gamma AOs).
        flux = 2.0 * v_sigma_grid[..., None] * grad_rho
        divergence = np.zeros(grid.shape, dtype=float)
        G = _resolve_recip(cache, grid)
        for d in range(3):
            flux_k = np.fft.fftn(flux[..., d])
            divergence += np.real(np.fft.ifftn(1j * G[..., d] * flux_k))
        v_eff -= divergence
    return e_xc, v_eff, v_tau_grid


def compute_j_via_gpw(
    basis,
    density_matrix: np.ndarray,
    grid: PlaneWaveGrid,
    *,
    quiet: bool = False,
) -> np.ndarray:
    """One-shot GPW Hartree-J build: ``D -> r -> V -> J``.

    Composes :func:`collocate_density_on_grid`,
    :func:`solve_poisson_coulomb`, and :func:`project_potential_to_ao`.
    Equivalent to ``GpwJBuilder(basis, grid).build_J(D)`` but
    convenient for one-shot tests.

    Emits :class:`GAPWExperimentalWarning` (the route is production-quality
    for vacuum-padded / pseudopotential-regime cells; all-electron CP2K
    parity on tight crystals awaits the GAPW augmentation). Pass
    ``quiet=True`` to suppress.
    """
    if not quiet:
        _warn_experimental(f"compute_j_via_gpw on {grid.nx}x{grid.ny}x{grid.nz} grid")
    # GPW Hartree-J -- Lippert, Hutter & Parrinello, Mol. Phys. 92, 477
    # (1997), doi:10.1080/00268979709482119:
    #   r(r) = S_muν D_muν chi_mu(r) chi_ν(r)      (collocate on the grid)
    #   grad^2V(r) = -4pi r(r)                    (FFT-Poisson)
    #   J_muν   = ∫ chi_mu(r) chi_ν(r) V(r) dr     (project back to AO basis)
    rho = collocate_density_on_grid(basis, density_matrix, grid)
    V = _core.solve_poisson_coulomb(rho, grid.lattice_bohr)
    return project_potential_to_ao(basis, V, grid)


class GpwJBuilder:
    """Hartree-J builder for the M2 GPW route.

    Holds the (basis, grid) pair so the AO-on-grid values can be
    cached once and reused across SCF iterations. Each
    :meth:`build_J` call still pays the O(n_basis^2 . N_grid)
    density-collocation cost; the cached chi saves only the
    evaluate-AO call.

    The interface mirrors the C++ :class:`JKBuilder` API enough that
    a future :class:`GapwJKBuilder` (M3) can compose it with an
    augmentation correction and surface as a drop-in
    :class:`PeriodicJKMethod.GAPW` kernel.

    Cache lifetime: ``GpwJBuilder`` is meant to live for the
    duration of one SCF run. The cached chi is invalidated if
    ``basis`` or ``grid`` is replaced (the dataclass is frozen, so
    instantiation is cheap -- just spin up a new builder when the
    geometry changes).
    """

    def __init__(
        self,
        basis,
        grid: PlaneWaveGrid,
        *,
        cache_ao_values: bool = True,
        mpi_aware: bool = False,
        collocation_cache: Optional[GpwCollocationCache] = None,
    ) -> None:
        self.basis = basis
        self.grid = grid
        self._cache_ao = cache_ao_values
        # Seed chi from a shared per-SCF GpwCollocationCache when supplied, so
        # the J build and the XC projections contract against ONE resident chi
        # (built once via build_gpw_collocation_cache) instead of each path
        # rebuilding it. Bit-identical: the cached chi is the inline build.
        self._collocation_cache = collocation_cache
        self._chi_cache: Optional[np.ndarray] = (
            collocation_cache.chi if collocation_cache is not None else None
        )
        self._mpi_aware = mpi_aware
        _warn_experimental(
            f"GpwJBuilder on {grid.nx}x{grid.ny}x{grid.nz} grid, "
            f"{basis.nbasis} basis functions" + (" (MPI-aware)" if mpi_aware else "")
        )

    def _chi(self) -> np.ndarray:
        """Return ``(n_grid, n_basis)`` AO values on the grid."""
        if self._chi_cache is not None:
            return self._chi_cache
        r_flat = self.grid.cartesian_coords().reshape(-1, 3)
        # Periodic AO image sum (see _ao_values_on_grid): the cached chi
        # used for both the density collocation and the V->AO projection
        # in build_J must be periodic, or build_J leaks charge for atoms
        # near a cell face exactly as the standalone collocator did.
        chi = _ao_values_on_grid(self.basis, r_flat, self.grid.lattice_bohr)
        if self._cache_ao:
            self._chi_cache = chi
        return chi

    def chi(self) -> np.ndarray:
        """Return the cached ``(n_grid, n_basis)`` periodic AO-on-grid table.

        Public accessor over the builder's chi cache. A driver can reuse the
        SAME chi for the XC-side collocate / project calls -- wrap it in a
        :class:`GpwCollocationCache` and thread that through the primitives --
        so J and XC contract against one resident table built once per SCF.
        """
        return self._chi()

    def build_J(self, density_matrix: np.ndarray) -> np.ndarray:
        """Build the Hartree-J matrix from an AO density matrix.

        Lippert-Hutter 1997 Eqs. (12)-(14): collocate -> FFT-Poisson -> project.
        When ``mpi_aware=True``, density collocation and potential projection
        are distributed across MPI ranks via z-slab domain decomposition.

        Returns a symmetrised ``(n_basis, n_basis)`` matrix.
        """
        D = np.asarray(density_matrix, dtype=float)
        if D.ndim != 2 or D.shape != (self.basis.nbasis, self.basis.nbasis):
            raise ValueError(
                f"density_matrix shape {D.shape} doesn't match basis "
                f"({self.basis.nbasis} functions)"
            )

        if self._mpi_aware:
            return self._build_J_mpi(D)
        return self._build_J_serial(D)

    def _build_J_serial(self, D: np.ndarray) -> np.ndarray:
        """Serial GPW J build (original path)."""
        chi = self._chi()
        # Lippert-Hutter 1997 Eq. (12)
        rho_flat = np.einsum("gm,mn,gn->g", chi, D, chi, optimize=True)
        rho = rho_flat.reshape(self.grid.shape)
        # Lippert-Hutter 1997 Eq. (13)
        V = _core.solve_poisson_coulomb(rho, self.grid.lattice_bohr)
        # Lippert-Hutter 1997 Eq. (14)
        V_flat = V.reshape(-1)
        J = (
            np.einsum("gm,gn,g->mn", chi, chi, V_flat, optimize=True)
            * self.grid.voxel_volume_bohr3
        )
        return 0.5 * (J + J.T)

    def _build_J_mpi(self, D: np.ndarray) -> np.ndarray:
        """MPI-aware GPW J build with z-slab domain decomposition.

        Each rank collocates r on its own z-slab (purely local -- see
        :func:`vibeqc.mpi.collocate_density_mpi`), the slabs are
        assembled into the full grid by zero-pad + allreduce so every
        rank holds the complete density, the FFT-Poisson solve is
        replicated on the full grid, and each rank projects its slab
        of V back to the AO basis with a final allreduce over the
        disjoint slab contributions. Must reproduce
        :meth:`_build_J_serial` to float roundoff at any world size --
        pinned by ``tests/test_mpi_gpw_parity.py``.
        """
        from .mpi import (
            collocate_density_mpi,
            grid_slab_partition,
            mpi_allreduce,
            project_potential_mpi,
        )

        # Partition grid into z-slabs
        slab = grid_slab_partition(self.grid.nx, self.grid.ny, self.grid.nz)
        nx, ny = slab.nx, slab.ny

        # Evaluate AOs on the local slab only. cartesian_coords() is a
        # C-ordered (nx, ny, nz, 3) array -- z is the FASTEST axis of
        # its flattening -- so the z-slab points must be sliced on
        # axis 2. (A flat [z_start.nx.ny : z_end.nx.ny] slice selects
        # x-planes, not z-planes, and scrambles the slab reshape.)
        r_local = self.grid.cartesian_coords()[
            :, :, slab.z_start : slab.z_end, :
        ].reshape(-1, 3)

        chi_local = _ao_values_on_grid(
            self.basis,
            r_local,
            self.grid.lattice_bohr,
        )

        # Collocate on the local slab -- no reduction; slab densities
        # are complete as computed.
        rho_local = collocate_density_mpi(chi_local, D)

        # Assemble the full density for FFT-Poisson (needs the full
        # grid): zero-pad the local slab into the full grid, then
        # allreduce the FULL grid so every rank holds the complete r.
        # An allgatherv of the slabs would avoid shipping the zero
        # padding; the allreduce is the simple correct assembly.
        rho_full = np.zeros(self.grid.shape, dtype=float)
        rho_full[:, :, slab.z_start : slab.z_end] = rho_local.reshape(
            nx, ny, slab.nz_local
        )
        rho_full = mpi_allreduce(rho_full, op="sum")

        # FFT-Poisson on the full grid (replicated across ranks)
        V = _core.solve_poisson_coulomb(rho_full, self.grid.lattice_bohr)

        # Extract local slab of potential
        V_local = V[:, :, slab.z_start : slab.z_end].reshape(-1)

        # MPI-aware projection: per-rank partial sums over disjoint
        # slabs, allreduced -- each grid point counted exactly once.
        dV = self.grid.voxel_volume_bohr3
        J = project_potential_mpi(chi_local, V_local, dV)
        return 0.5 * (J + J.T)

    def hartree_energy(self, density_matrix: np.ndarray) -> float:
        """Return ``1/2 tr(D . J)`` -- the Hartree energy at ``D``.

        Convenience for diagnostics + the M1e-style internal-
        consistency tests.
        """
        J = self.build_J(density_matrix)
        return 0.5 * float(np.einsum("ij,ij->", density_matrix, J))

    # Future-API hooks (M3): build_K, build_J_slot/build_K_slot,
    # build_K_erf -- left out at M2 to keep the surface tight.


# ============================================================
# Single-point GPW energy breakdown (M2c)
# ============================================================


@dataclass(frozen=True)
class GpwEnergyBreakdown:
    """Per-term breakdown of a single-point GPW periodic energy.

    All values in Hartree. The total is the algebraic sum of the
    five term fields.

    Convention matches vibe-qc's molecular SCF: ``e_hf_exchange``
    is the negative-signed exchange contribution
    ``-(1/4) tr(D . K)`` for a closed-shell RHF density. The total
    is ``e_kinetic + e_nuclear_attraction + e_hartree +
    e_hf_exchange + e_nuclear_repulsion``.

    Attributes
    ----------
    e_kinetic
        ``tr(D . T)`` -- periodic kinetic energy via the lattice
        builder + Bloch sum at Γ (B3). Numerically equivalent to
        the molecular-limit T on vacuum-padded cells; correct on
        tight cells where image-AO kinetic coupling matters.
    e_nuclear_attraction
        ``tr(D . V_ne)`` -- electron-nucleus attraction with the
        periodic Ewald V_ne (G = 0 dropped, v_bg jellium shift
        applied) so the gauge matches the FFT-Poisson Hartree-J.
        Lifted from the molecular limit at M3a; see
        :func:`vibeqc.periodic_v_ne.compute_nuclear_lattice_dispatch`
        for the gauge contract.
    e_hartree
        ``1/2 tr(D . J_gpw)`` -- periodic Hartree energy via the
        GPW J on the smooth real-space grid. Includes the
        Madelung self-image shift on charged cells (the cell is
        neutralised by a uniform background; the shift is the
        cost of that background).
    e_hf_exchange
        ``-(1/4) tr(D . K)`` -- molecular Hartree-Fock exchange.
    e_nuclear_repulsion
        Ewald Madelung nuclear-nuclear repulsion via
        :func:`vibeqc._vibeqc_core.ewald_nuclear_repulsion`.
    e_total
        Algebraic sum of the five term fields above. The single
        number the SCF would report.
    grid
        :class:`PlaneWaveGrid` used to build the Hartree-J.
        Carried in the result for audit / reproducibility.
    """

    e_kinetic: float
    e_nuclear_attraction: float
    e_hartree: float
    e_hf_exchange: float
    e_nuclear_repulsion: float
    e_total: float
    grid: PlaneWaveGrid
    # M3d: DFT support. Zero on the HF path.
    e_xc: float = 0.0
    functional: Optional[str] = None
    # v0.12-prep: D3-BJ periodic dispersion add-on. Zero unless the
    # caller explicitly requests dispersion= on the SCF entry point.
    e_dispersion: float = 0.0
    # v0.12 R1: DFT+U (Dudarev) on-site correction. Non-zero only when
    # the caller passed dft_plus_u_sites=[HubbardSite(...)] on the SCF
    # entry point. Closed-shell GPW reports 2 x the per-spin Dudarev
    # energy; open-shell GPW reports E_U_alpha + E_U_beta. Already
    # folded into e_total -- kept as a separate field for the breakdown.
    e_dft_plus_u: float = 0.0


def evaluate_gpw_energy(
    system,
    basis,
    density_matrix: np.ndarray,
    *,
    grid: Optional[PlaneWaveGrid] = None,
    cutoff_ha: Optional[float] = None,
    omega: float = 0.0,
    hf_sr_fraction: Optional[float] = None,
    hf_lr_fraction: Optional[float] = None,
    v_ne_convention: str = "ewald",
    smearing_alpha: Optional[float] = None,
    functional: Optional[str] = None,
    quiet: bool = False,
    collocation_cache: Optional[GpwCollocationCache] = None,
    external_xc_grid_options=None,
    external_xc_image_radius_bohr: float = 10.0,
    external_xc_lattice_options=None,
    _external_xc_context=None,
) -> GpwEnergyBreakdown:
    """Compute a single-point periodic-energy breakdown using the
    GPW Hartree-J at a *given* AO density matrix.

    > **Experimental, diagnostic.** This is the M2c entry point
    > (V_ne lifted to the periodic Ewald gauge at M3a; T and S lifted
    > to the lattice path at B3): a one-shot evaluator that composes
    > the Γ-point periodic T / S / V_ne lattice builders, the
    > Γ-only periodic K via direct AO-image sum (cutoff 25 bohr),
    > and the GPW Hartree-J on the smooth FFT grid. **Not a full
    > periodic SCF** -- the density is supplied by the caller
    > (typically the converged molecular SCF on the same system in
    > vacuum). Useful for sanity-checking the GPW recipe against a
    > reference density. The full iterative SCF lives at
    > :func:`run_periodic_rhf_gpw`; the C++-hosted production
    > driver is M3b work.

    Parameters
    ----------
    system
        A :class:`vibeqc._vibeqc_core.PeriodicSystem` describing
        the cell (lattice + atoms in Cartesian bohr). Drives the
        Ewald nuclear repulsion, the Ewald V_ne (M3a lift), and
        the FFT-Poisson cell; the atoms must match the basis's
        atomic centres.
    basis
        :class:`vibeqc.BasisSet` for the system.
    density_matrix
        ``(n_basis, n_basis)`` AO density. For closed-shell RHF
        this is ``2 . C_occ @ C_occ.T``.
    grid
        Pre-built :class:`PlaneWaveGrid`. Pass ``None`` to let
        the helper construct one via :func:`make_grid` from
        ``cutoff_ha``.
    cutoff_ha
        Plane-wave kinetic-energy cutoff in Hartree, used when
        ``grid`` is None. Defaults to 300 Ha (~600 Ry in CP2K's
        Ry convention; gives ~10⁻⁴ Ha grid convergence on compact
        contracted Gaussians).
    omega
        Range-separation parameter in bohr^-1 for the diagnostic Hartree-J
        builder. ``0.0`` uses the unscreened GPW Coulomb path. Positive
        values route J through the existing short-range plus long-range
        range-separated GPW builder; the two pieces recombine to the full
        Hartree J and provide the nonzero-omega energy surface used by the
        range-separated GAPW diagnostics.
    hf_sr_fraction, hf_lr_fraction
        Exact-exchange fractions applied to the short-range and long-range
        K pieces when ``omega > 0``. If omitted, range-separated functionals
        known to the range-separated lookup supply their tabulated fractions;
        otherwise the short-range fraction defaults to the functional's
        global HF-exchange fraction (or ``1.0`` for Hartree-Fock) and the
        long-range fraction defaults to zero.
    v_ne_convention
        Which periodic V_ne convention to build:
        ``"ewald"`` (default) -> M3a path,
        :func:`vibeqc.periodic_gapw_j._ewald_v_ne_gamma`;
        ``"smeared_erfc"`` -> M3b CP2K-style erfc/erf split,
        :func:`vibeqc.periodic_gapw_smearing.smeared_v_ne_full_gamma`.
        The two are physically equivalent on neutral cells (parity
        pinned to < 1 mHa on He / H2 STO-3G); ``"smeared_erfc"``
        is the convention the GAPW augmentation will plug into.
    smearing_alpha
        Smearing exponent (bohr⁻¹) when ``v_ne_convention =
        "smeared_erfc"``. Default ``None`` -> auto-pick from the
        grid via
        :func:`vibeqc.periodic_gapw_smearing.default_smearing_alpha_from_grid`.
        Ignored when ``v_ne_convention = "ewald"``.
    quiet
        Pass True to suppress the GAPWExperimentalWarning.

    Returns
    -------
    breakdown
        :class:`GpwEnergyBreakdown` with the per-term components
        and the grid the J was built on.
    """
    if not quiet:
        _warn_experimental(
            "evaluate_gpw_energy: single-point, not a full SCF; "
            "use run_periodic_rhf_gpw for iterative self-consistency"
        )
    omega = float(omega)
    if omega < 0.0:
        raise ValueError(f"evaluate_gpw_energy: omega must be >= 0; got {omega!r}")

    if functional is not None:
        func = _core.Functional(functional, 1)
    else:
        func = None
    if func is not None and bool(getattr(func, "is_external", False)):
        if int(system.dim) != 3:
            raise NotImplementedError(
                "evaluate_gpw_energy: external XC is supported only for a "
                "3D periodic GPW Hamiltonian"
            )
        if float(system.charge) != 0.0:
            raise NotImplementedError(
                "evaluate_gpw_energy: charged-cell external XC is not "
                "supported because the GPW Coulomb background convention "
                "has not been validated; use a neutral cell"
            )

        from .pbc_bipole_common import reject_bipole_ecp_options

        try:
            reject_bipole_ecp_options(
                object(),
                driver="evaluate_gpw_energy external XC",
                basis=basis,
                system=system,
            )
        except NotImplementedError as exc:
            raise NotImplementedError(
                "evaluate_gpw_energy: ECP-bearing bases are not implemented "
                "with external XC; use an all-electron basis"
            ) from exc

    D = np.asarray(density_matrix, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(
            f"density_matrix must be a square (n_basis, n_basis) "
            f"array; got shape {D.shape}"
        )
    if D.shape[0] != basis.nbasis:
        raise ValueError(
            f"density_matrix has n_basis={D.shape[0]} but the supplied "
            f"basis has nbasis={basis.nbasis}"
        )

    # Cell-neutrality preflight. The FFT-Poisson Hartree and the Ewald
    # V_ne both drop the G = 0 reciprocal mode and rely on the silent
    # uniform-jellium background to make a charged cell sum to a finite
    # value. The total-energy cancellation between (Hartree, V_ne, E_nn)
    # is exact only when ``tr(D . S) == S_a Z_a`` (neutral cell). On a
    # charged cell the returned ``e_total`` carries an unphysical
    # jellium-class shift; CP2K's GAPW driver has the same property and
    # makes the same caveat explicit. We warn rather than refuse
    # because evaluating a slightly-non-neutral D is occasionally useful
    # diagnostically (e.g., pre-converged guess densities).
    _warn_if_non_neutral(D, basis, system, quiet=quiet)

    # Build the FFT grid if not supplied. Default cutoff 300 Ha is
    # tuned for the M2c smoke set (He / H2 in a ~16-bohr cube). A
    # caller benchmarking convergence should construct the grid
    # externally so the (Nx, Ny, Nz) is auditable.
    if grid is None:
        from .periodic_gapw_grid import make_grid as _make_grid

        if cutoff_ha is None:
            cutoff_ha = 300.0
        grid = _make_grid(np.asarray(system.lattice, dtype=float), cutoff_ha=cutoff_ha)

    # ---- One-electron integrals -------------------------------------
    # T, S, V_ne all use the periodic lattice builders + Bloch sum at
    # Γ. Kinetic + overlap decay exponentially with shell separation,
    # so the lattice path is bit-equivalent to the molecular limit on
    # vacuum-padded boxes; on tighter cells it stays correct without
    # silently mis-converging (B3, audit-driven).
    #
    # V_ne convention (M3b): two paths, both gauge-aligned with the
    # FFT-Poisson J on neutral cells.
    #  * "ewald" (M3a, default): _ewald_v_ne_gamma -- Ewald lattice
    #    sum + v_bg jellium shift. Production-tested, no external
    #    parameter.
    #  * "smeared_erfc" (M3b, CP2K-style): erfc/erf split via
    #    FFT-Poisson on smeared r_core + libint erfc one-electron
    #    + a-correction. Routes every Hartree-class quantity
    #    through the same Poisson solver -- the convention the
    #    GAPW augmentation will hook into.
    T = _kinetic_lattice_gamma(basis, system)
    V_ne = _build_v_ne(
        basis,
        system,
        v_ne_convention,
        smearing_alpha,
        grid,
    )
    e_kinetic = float(np.einsum("ij,ij->", D, T))
    e_nuclear_attraction = float(np.einsum("ij,ij->", D, V_ne))

    # ---- Periodic K via the existing AO-image-summed builder --------
    # M3d: when a functional is given, scale HF exchange by the
    # functional's exact-exchange fraction (1.0 for HF; 0.0 for pure
    # DFT; 0.2 for B3LYP; etc.).
    lo = _core.LatticeSumOptions()
    lo.cutoff_bohr = 25.0
    if func is not None:
        ex_frac = float(func.hf_exchange_fraction)
    else:
        ex_frac = 1.0
    if omega > 0.0:
        default_sr = ex_frac
        default_lr = 0.0
        if functional is not None:
            try:
                from .periodic_gapw_range_sep import (
                    omega_for_functional as _omega_for_functional,
                )

                _default_omega, default_sr, default_lr = _omega_for_functional(
                    functional
                )
            except ValueError:
                # Non-RSH hybrid with an explicit omega: treat omega as a
                # screened short-range exact-exchange request.
                pass
        c_sr = default_sr if hf_sr_fraction is None else float(hf_sr_fraction)
        c_lr = default_lr if hf_lr_fraction is None else float(hf_lr_fraction)
        if c_sr != 0.0 or c_lr != 0.0:
            jk_sr = _core.build_jk_gamma_molecular_limit(
                basis,
                system,
                lo,
                D,
                omega=omega,
            )
            K_sr = np.asarray(jk_sr.K)
            if c_lr != 0.0:
                jk_full = _core.build_jk_gamma_molecular_limit(
                    basis,
                    system,
                    lo,
                    D,
                    omega=0.0,
                )
                K_lr = np.asarray(jk_full.K) - K_sr
                K_eff = c_sr * K_sr + c_lr * K_lr
            else:
                K_eff = c_sr * K_sr
            e_hf_exchange = -0.25 * float(np.einsum("ij,ij->", D, K_eff))
        else:
            e_hf_exchange = 0.0
    elif ex_frac > 0.0:
        jk_mol = _core.build_jk_gamma_molecular_limit(basis, system, lo, D)
        K = np.asarray(jk_mol.K)
        e_hf_exchange = -0.25 * ex_frac * float(np.einsum("ij,ij->", D, K))
    else:
        e_hf_exchange = 0.0

    # ---- GPW Hartree J on the smooth grid --------------------------
    # Build chi (+ G-mesh) once and share it between the J build and the XC
    # collocate below, so this single-point evaluation pays one chi build.
    # A caller that already built the cache for this (basis, grid) -- e.g.
    # run_periodic_rhf_gpw computing its final breakdown -- passes it in to
    # skip the 27-image evaluate_ao collocation (the dominant SCF cost on a
    # fine grid); the cache is geometry/grid-specific so reuse is exact.
    if collocation_cache is None:
        collocation_cache = build_gpw_collocation_cache(basis, grid)
    if omega == 0.0:
        builder = GpwJBuilder(basis, grid, collocation_cache=collocation_cache)
    else:
        from .periodic_gapw_range_sep import RsJBuilder

        builder = RsJBuilder(
            basis,
            grid,
            omega,
            quiet=True,
            collocation_cache=collocation_cache,
        )
    J_hartree = builder.build_J(D)
    e_hartree = 0.5 * float(np.einsum("ij,ij->", D, J_hartree))

    # ---- XC ---------------------------------------------------------
    if func is not None:
        if bool(getattr(func, "is_external", False)):
            external_xc = _external_xc_context
            if external_xc is None:
                from .periodic_external_xc import PeriodicExternalXC

                external_xc = PeriodicExternalXC.prepare(
                    basis,
                    system,
                    func,
                    grid_options=external_xc_grid_options,
                    image_radius_bohr=external_xc_image_radius_bohr,
                    lattice_options=(
                        external_xc_lattice_options
                        if external_xc_lattice_options is not None
                        else lo
                    ),
                )
            e_xc, _v_xc = external_xc.build_gamma(D)
        else:
            rho_grid = collocate_density_on_grid(
                basis, D, grid, cache=collocation_cache
            )
            # basis + density_matrix let meta-GGA functionals build t(r);
            # LDA/GGA ignore them.
            (
                e_xc,
                _v_xc_grid,
                _v_sigma_grid,
                _grad_rho,
                _v_tau_grid,
            ) = _evaluate_xc_on_grid(
                rho_grid,
                grid,
                func,
                basis=basis,
                density_matrix=D,
                cache=collocation_cache,
            )
    else:
        e_xc = 0.0

    # ---- Ewald nuclear repulsion -----------------------------------
    if system.dim == 3:
        e_nuclear_repulsion = float(
            _core.ewald_nuclear_repulsion(system, _core.EwaldOptions())
        )
    else:
        e_nuclear_repulsion = 0.0  # 1D/2D Ewald is M3+

    e_total = (
        e_kinetic
        + e_nuclear_attraction
        + e_hartree
        + e_hf_exchange
        + e_xc
        + e_nuclear_repulsion
    )
    return GpwEnergyBreakdown(
        e_kinetic=e_kinetic,
        e_nuclear_attraction=e_nuclear_attraction,
        e_hartree=e_hartree,
        e_hf_exchange=e_hf_exchange,
        e_nuclear_repulsion=e_nuclear_repulsion,
        e_total=e_total,
        grid=grid,
        e_xc=e_xc,
        functional=functional,
    )


def _wrap_molecule(system):
    """Wrap a PeriodicSystem's unit cell into a Molecule for the
    molecular-integral builders. Charge / multiplicity are not
    consumed by the integral routines; the wrapper is kept around for
    callers that still want the molecular limit (e.g., pre-M3a
    diagnostics) -- the M3a SCF path itself takes Ewald V_ne via
    :func:`_ewald_v_ne_gamma`."""
    from .molecule import Molecule

    return Molecule(list(system.unit_cell), 0, 1)


def _warn_if_non_neutral(
    D: np.ndarray,
    basis,
    system,
    *,
    quiet: bool = False,
    tol: float = 1e-3,
) -> None:
    """Emit a :class:`GAPWExperimentalWarning` if ``|tr(D.S) - S Z|``
    exceeds ``tol`` electrons.

    On a charged cell the GPW total energy carries a silent jellium-
    class shift; the FFT-Poisson Hartree and the Ewald V_ne both drop
    the G = 0 mode, and the cross-term cancellation that closes the
    bookkeeping is exact only when the cell is neutral. CP2K's GAPW
    driver has the same property -- its developer notes pin "energy of
    a +1 cell is not a quantity CP2K-GAPW returns directly". We
    inherit that limitation and surface it as a warning.
    """
    if quiet:
        return
    try:
        S = np.asarray(_core.compute_overlap(basis))
        n_elec = float(np.einsum("ij,ij->", D, S))
        z_total = float(sum(int(a.Z) for a in system.unit_cell))
    except Exception:
        # Best-effort diagnostic; never crash the SCF over the check.
        return
    excess = n_elec - z_total
    if abs(excess) > tol:
        warnings.warn(
            f"GPW: cell is non-neutral (tr(D.S) = {n_elec:.4f}, "
            f"S Z = {z_total:.4f}, excess = {excess:+.4f} e). The "
            f"FFT-Poisson G = 0 = 0 + Ewald-V_ne v_bg jellium "
            f"convention only cancels on a neutral cell; the returned "
            f"e_total carries an unphysical jellium-class shift. See "
            f"docs/design_periodic_gapw.md.",
            category=GAPWExperimentalWarning,
            stacklevel=4,
        )


def _gamma_one_e(lattice_matrix_set) -> np.ndarray:
    """Bloch-sum a real-valued one-electron LatticeMatrixSet at Γ.

    The Γ-point Bloch sum of an integer-translation real-symmetric
    integral is real and symmetric in any AO basis. We take ``np.real``
    to drop a ~1e-15 imaginary drift from the FFT-style sum and
    symmetrise to defend the downstream eigensolver from the same
    rounding-error asymmetry.
    """
    M_k = _core.bloch_sum(lattice_matrix_set, np.zeros(3))
    M = np.real(M_k)
    return 0.5 * (M + M.T)


def _kinetic_lattice_gamma(basis, system) -> np.ndarray:
    """Γ-point lattice kinetic-energy matrix.

    The kinetic integrand ``-1/2 grad^2`` between AO products decays
    exponentially with shell separation, so for vacuum-padded boxes
    the lattice T is numerically identical to the molecular T. We
    still use the lattice path (B3, audit-driven) so the SCF stays
    correct on tighter cells without silently mis-converging.
    """
    lat_opts = _core.LatticeSumOptions()
    T_lat = _core.compute_kinetic_lattice(basis, system, lat_opts)
    return _gamma_one_e(T_lat)


def _overlap_lattice_gamma(basis, system) -> np.ndarray:
    """Γ-point lattice overlap matrix.

    The overlap integrand decays exponentially with shell separation,
    same family as kinetic. Lifted to the lattice path at B3 for the
    same reason as :func:`_kinetic_lattice_gamma`.
    """
    lat_opts = _core.LatticeSumOptions()
    S_lat = _core.compute_overlap_lattice(basis, system, lat_opts)
    return _gamma_one_e(S_lat)


def _compute_density_gradient_fft(
    rho: np.ndarray,
    grid: PlaneWaveGrid,
    *,
    recip: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return ``(nx, ny, nz, 3)`` gradient ``gradr(r)`` via FFT.

    Uses the standard spectral method: ``gradr(r) = IFFT(i G r̂(G))``,
    where ``r̂`` is the forward FFT of r and ``G`` is the
    reciprocal-space wavevector on the FFT-wrapped grid. Higher-
    order quadrature is unnecessary because the spectral
    derivative is exact for band-limited inputs (and r from a
    Gaussian basis is band-limited at the grid's resolution).
    """
    rho_k = np.fft.fftn(rho)
    G = grid.reciprocal_vectors() if recip is None else recip  # (nx, ny, nz, 3)
    grad = np.empty(grid.shape + (3,), dtype=float)
    for d in range(3):
        rho_k_d = 1j * G[..., d] * rho_k
        grad[..., d] = np.real(np.fft.ifftn(rho_k_d))
    return grad


def _ao_gradients_fft(
    chi: np.ndarray,
    grid: PlaneWaveGrid,
    *,
    recip: Optional[np.ndarray] = None,
):
    """Yield the three Cartesian AO-gradient tables ``gradchi_d`` on the grid.

    ``chi`` is the (n_grid, n_basis) periodic AO table (real, Gamma). Each
    yielded array is the (n_grid, n_basis) real table of
    ``d chi_mu / d r_d (r_g)`` computed spectrally per AO column
    (``IFFT(i G_d FFT(chi_mu))``), batched over the basis in one 3D FFT per
    direction. Yielding one direction at a time keeps the peak footprint to a
    single extra (n_grid, n_basis) table instead of three.
    """
    shape = grid.shape
    n_grid, n_basis = chi.shape
    G = grid.reciprocal_vectors() if recip is None else recip
    chi_g = np.fft.fftn(chi.reshape(shape + (n_basis,)), axes=(0, 1, 2))
    for d in range(3):
        gd = np.real(
            np.fft.ifftn(1j * G[..., d, None] * chi_g, axes=(0, 1, 2))
        )
        yield gd.reshape(n_grid, n_basis)


def _project_vtau_to_ao(
    basis,
    v_tau_grid: np.ndarray,
    grid: PlaneWaveGrid,
    *,
    cache: Optional[GpwCollocationCache] = None,
) -> np.ndarray:
    """Project the meta-GGA t-potential onto the AO basis.

    The t contribution to the generalized-KS Fock matrix is

        ``V_tau[muν] = dE_xc/dD_muν |_tau
                     = 1/2 ∫ v_tau(r) gradchi_mu(r) . gradchi_ν(r) dr``

    (from ``t(r) = 1/2 S_muν D_muν gradchi_mu . gradchi_ν``), matching the
    ``w_vtau = 0.5 w v_tau; V += da^T diag(w_vtau) ds`` convention of the C++
    real-space periodic XC builder (``cpp/src/periodic_xc.cpp``). It is NOT
    the multiplicative projection ``∫ chi_mu chi_ν v_tau dr`` -- projecting
    v_tau like a local potential gives a Fock that is not the derivative of
    E_xc and converges the SCF to the wrong density.
    """
    V = np.asarray(v_tau_grid, dtype=float)
    if V.shape != grid.shape:
        raise ValueError(
            f"v_tau shape {V.shape} doesn't match grid {grid.shape}"
        )
    chi = _resolve_chi(cache, basis, grid)
    G = _resolve_recip(cache, grid)
    w = (0.5 * grid.voxel_volume_bohr3) * V.reshape(-1)
    n_basis = chi.shape[1]
    M = np.zeros((n_basis, n_basis), dtype=float)
    for gd in _ao_gradients_fft(chi, grid, recip=G):
        M += (gd * w[:, None]).T @ gd
    return 0.5 * (M + M.T)


def _compute_kinetic_energy_density_fft(
    basis,
    density_matrix: np.ndarray,
    grid: PlaneWaveGrid,
    *,
    cache: Optional[GpwCollocationCache] = None,
) -> np.ndarray:
    """Compute the kinetic energy density t(r) = 1/2 S_i |gradpsi_i(r)|^2
    on the FFT grid from an AO density matrix.

    Uses the AO gradient on the grid: t(r) = 1/2 S_muν D_muν . gradchi_mu(r) . gradchi_ν(r).

    Parameters
    ----------
    basis
        :class:`vibeqc.BasisSet`.
    density_matrix
        ``(n_basis, n_basis)`` AO density matrix.
    grid
        :class:`PlaneWaveGrid`.

    Returns
    -------
    tau
        ``(nx, ny, nz)`` kinetic energy density in Hartree.
    """
    # chi + G-mesh from the per-SCF cache when supplied (cache=None rebuilds
    # inline, bit-identical).
    chi = _resolve_chi(cache, basis, grid)  # (n_grid, n_basis)
    D = np.asarray(density_matrix, dtype=float)
    # t(r) = 1/2 S_muν D_muν . gradchi_mu(r) . gradchi_ν(r), with the AO
    # gradients taken spectrally (gradchi_mu = IFFT(iG FFT(chi_mu))). The
    # per-direction table gd is (n_grid, n_basis), so each direction is one
    # batched FFT pair + one density contraction -- the historical
    # implementation FFT'd chi_ν inside a mu/ν double loop (O(n_basis^2)
    # grid FFTs) and crashed on a flat-vs-grid shape mismatch before ever
    # producing a t; this is the working O(3 n_basis) replacement.
    n_grid = chi.shape[0]
    G = _resolve_recip(cache, grid)  # (nx, ny, nz, 3)
    tau_flat = np.zeros(n_grid, dtype=float)
    for gd in _ao_gradients_fft(chi, grid, recip=G):
        tau_flat += np.einsum("gm,mn,gn->g", gd, D, gd, optimize=True)
    return (0.5 * tau_flat).reshape(grid.shape)


def _evaluate_xc_on_grid(
    rho: np.ndarray,
    grid: PlaneWaveGrid,
    functional,
    *,
    basis=None,
    density_matrix=None,
    tau: Optional[np.ndarray] = None,
    cache: Optional[GpwCollocationCache] = None,
) -> tuple:
    """Evaluate XC on the FFT grid via libxc's ``Functional``.

    Returns ``(e_xc_total, v_xc_grid, v_sigma_grid_or_None, grad_or_None,
    v_tau_grid_or_None)`` where:

    * ``e_xc_total`` = ``S_g r(g) . e_xc(g) . ΔV`` -- the total
      exchange-correlation energy.
    * ``v_xc_grid`` = the LDA piece of the XC potential
      ``dE_xc / dr`` on the grid (shape ``(nx, ny, nz)``).
    * ``v_sigma_grid_or_None`` = for GGAs, ``2 . dE_xc / ds .
      gradr`` term (None for LDA).
    * ``grad_or_None`` = ``gradr`` on the grid (None for LDA).
    * ``v_tau_grid_or_None`` = for meta-GGAs, ``dE_xc / dt`` term
      (None for LDA/GGA).

    For meta-GGA functionals the potential has an additional ``v_tau``
    piece coupling the kinetic energy density t(r). The caller passes
    ``basis`` and ``density_matrix`` when t is needed, or a precomputed
    ``tau=`` grid (the compact multi-k Bloch path builds t from per-k
    Bloch AO gradients instead of the Gamma builder). The von Weizsacker
    floor, constrained chain rule, density screen, and stiffness guard
    are applied to the supplied t exactly as to the built one.
    """
    # Clip negative densities to a small positive floor -- finite-grid
    # interpolation of contracted Gaussians can give a few negative
    # cells where the basis has high-l components; libxc misbehaves on
    # negative r.
    rho_flat = np.maximum(rho.reshape(-1), 0.0)
    kind = functional.kind
    is_mgga = kind == _core.XCKind.MGGA
    is_gga = kind != _core.XCKind.LDA and not is_mgga

    grad = None
    sigma_flat = np.zeros_like(rho_flat)
    tau_flat = None

    if is_gga or is_mgga:
        grad = _compute_density_gradient_fft(
            rho, grid, recip=_resolve_recip(cache, grid) if cache else None
        )
        sigma_flat = (grad**2).sum(axis=-1).reshape(-1)

    if is_mgga:
        if tau is None:
            if basis is None or density_matrix is None:
                raise ValueError(
                    "_evaluate_xc_on_grid: meta-GGA functionals require "
                    "basis and density_matrix to compute t(r) (or a "
                    "precomputed tau=)."
                )
            tau = _compute_kinetic_energy_density_fft(
                basis,
                density_matrix,
                grid,
                cache=cache,
            )
        # Enforce the von Weizsacker lower bound tau >= tau_W = sigma/(8 rho)
        # (tau = 0 where rho <= 0). The grid tau at diffuse points can dip
        # below tau_W (spectral-gradient roundoff on the AO tails), where
        # unregularised meta-GGAs (TPSS, M06-L, ...) are SINGULAR: libxc
        # returns v_sigma / v_tau ~ 1e16 which contaminates the projected
        # V_xc while the energy integral stays ~correct. Clamping to 0 is NOT
        # enough (libxc still diverges at tau = 0 with finite sigma); the
        # bound must be tau_W. Mirrors enforce_von_weizsaecker_floor in
        # cpp/src/periodic_xc.cpp -- the same clamped tau feeds the energy
        # AND the Fock so the two stay consistent.
        tau_raw = np.asarray(tau.reshape(-1), dtype=float)
        tau_w = np.where(rho_flat > 0.0, sigma_flat / (8.0 * rho_flat), 0.0)
        clamp_active = (tau_raw < tau_w) & (rho_flat > 0.0)
        tau_flat = np.where(rho_flat > 0.0, np.maximum(tau_raw, tau_w), 0.0)
        exc, v_rho, v_sigma, v_tau = functional.eval_unpolarised_mgga(
            rho_flat,
            sigma_flat,
            tau_flat,
        )
        exc = np.asarray(exc, dtype=float)
        v_rho = np.asarray(v_rho, dtype=float)
        v_sigma = np.asarray(v_sigma, dtype=float)
        v_tau = np.asarray(v_tau, dtype=float)
        # Constrained chain rule at clamp-active points. There the energy
        # is E(rho, sigma, tau_W(rho, sigma)), so the derivative w.r.t. the
        # density flows through tau_W = sigma/(8 rho):
        #     dE/dsigma_eff = v_sigma + v_tau / (8 rho)
        #     dE/drho_eff   = v_rho   - v_tau sigma / (8 rho^2)
        #     dE/dtau_eff   = 0
        # Without this the projected V_xc keeps a spurious 1/2 v_tau
        # gradchi.gradchi from points whose tau no longer responds to D --
        # a systematic Fock-vs-energy inconsistency (~5e-3 on the FD pin
        # for single-orbital H2, where roughly half the grid rides the
        # tau == tau_W iso-orbital boundary).
        if np.any(clamp_active):
            rho_c = np.maximum(rho_flat, 1e-300)
            v_sigma = np.where(
                clamp_active, v_sigma + v_tau / (8.0 * rho_c), v_sigma
            )
            v_rho = np.where(
                clamp_active,
                v_rho - v_tau * sigma_flat / (8.0 * rho_c * rho_c),
                v_rho,
            )
            v_tau = np.where(clamp_active, 0.0, v_tau)
        # Density screen for the uniform grid. Unlike the Becke atomic
        # grids of the C++ real-space builder, FFT voxels carry full weight
        # everywhere -- so libxc's near-vacuum meta-GGA derivatives (v_sigma
        # can reach ~1e12 at rho ~ 1e-12 even for self-regularised r2SCAN)
        # would enter the projected Fock at full strength. Zero ALL outputs
        # below the cutoff, identically in the energy and every potential
        # channel, so V_xc stays the exact derivative of the screened E_xc.
        # exc at the screened points is O(rho^{4/3}) ~ 1e-11 -- the energy
        # change is far below the grid quadrature error.
        screen = rho_flat < _MGGA_DENSITY_SCREEN
        exc[screen] = 0.0
        v_rho[screen] = 0.0
        v_sigma[screen] = 0.0
        v_tau[screen] = 0.0
        # Stiffness guard (fail-closed). Non-self-regularising meta-GGAs
        # (TPSS, M06-L, ...) have divergent partials dz/dsigma = 1/(8 rho t)
        # at orbital critical points (sigma -> 0, t -> 0 at finite rho).
        # Becke atomic grids rarely sample exactly there; the uniform FFT
        # grid does, at full voxel weight -- the projected Fock is then
        # garbage-dominated (|v_sigma| ~ 1e17) and the SCF converges to a
        # spurious stationary point while the energy integral still looks
        # plausible. Refuse to continue rather than return a wrong number.
        # Self-regularising functionals (SCAN / rSCAN / r2SCAN) stay orders
        # of magnitude below these bounds.
        max_vs = float(np.abs(v_sigma).max()) if v_sigma.size else 0.0
        max_vt = float(np.abs(v_tau).max()) if v_tau.size else 0.0
        if max_vs > _MGGA_VSIGMA_STIFF_MAX or max_vt > _MGGA_VTAU_STIFF_MAX:
            raise NotImplementedError(
                "GPW meta-GGA: the XC potential on the uniform grid is "
                f"stiffness-dominated (max|v_sigma| = {max_vs:.2e}, "
                f"max|v_tau| = {max_vt:.2e}) -- this functional's "
                "iso-orbital-boundary derivatives diverge at orbital "
                "critical points, which the FFT grid samples at full "
                "weight. Use a self-regularising meta-GGA (r2scan / scan) "
                "on the GPW route, or jk_method='gdf' for this functional."
            )
        v_tau_grid = v_tau.reshape(grid.shape)
    elif is_gga:
        exc, v_rho, v_sigma = functional.eval_unpolarised(rho_flat, sigma_flat)
        v_tau_grid = None
    else:
        exc, v_rho, v_sigma = functional.eval_unpolarised(rho_flat, sigma_flat)
        v_tau_grid = None

    dv = grid.voxel_volume_bohr3
    e_xc_total = float(exc.sum() * dv)

    v_xc_grid = v_rho.reshape(grid.shape)
    if is_gga or is_mgga:
        v_sigma_grid = v_sigma.reshape(grid.shape)
    else:
        v_sigma_grid = None

    return e_xc_total, v_xc_grid, v_sigma_grid, grad, v_tau_grid


def _project_vxc_to_ao(
    basis,
    v_xc_grid: np.ndarray,
    grid: PlaneWaveGrid,
    *,
    v_sigma_grid: Optional[np.ndarray] = None,
    grad_rho: Optional[np.ndarray] = None,
    cache: Optional[GpwCollocationCache] = None,
) -> np.ndarray:
    """Project ``v_xc(r)`` onto an AO matrix.

    For LDA: ``V_xc[muν] = ∫ chi_mu chi_ν . v_rho dr`` -- same primitive
    as :func:`project_potential_to_ao`.

    For GGA: adds the gradient correction term

        ``V_xc_GGA[muν] = 2 . ∫ (v_sigma . gradr) . grad(chi_mu chi_ν) dr``

    Spectrally integrating by parts, this becomes

        ``V_xc_GGA[muν] = - ∫ chi_mu chi_ν . grad.(2 v_sigma gradr) dr``

    which we implement by FFT'ing the divergence and projecting
    via the same AO primitive. Symmetry is enforced at the end.
    """
    V_lda = project_potential_to_ao(basis, v_xc_grid, grid, cache=cache)
    if v_sigma_grid is None or grad_rho is None:
        return V_lda
    # GGA: divergence of (2 v_sigma . gradr) -- compute via FFT for
    # spectral accuracy.
    flux = 2.0 * v_sigma_grid[..., None] * grad_rho  # (nx, ny, nz, 3)
    divergence = np.zeros(grid.shape, dtype=float)
    G = _resolve_recip(cache, grid)
    for d in range(3):
        flux_k = np.fft.fftn(flux[..., d])
        divergence += np.real(np.fft.ifftn(1j * G[..., d] * flux_k))
    V_gga = -project_potential_to_ao(basis, divergence, grid, cache=cache)
    V_total = V_lda + V_gga
    return 0.5 * (V_total + V_total.T)


def _build_v_ne(
    basis,
    system,
    convention: str,
    smearing_alpha: Optional[float],
    grid: Optional[PlaneWaveGrid],
) -> np.ndarray:
    """Dispatch V_ne build by convention. Internal helper for the
    GPW entry points.

    Conventions
    -----------
    ``"ewald"`` (default)
        :func:`_ewald_v_ne_gamma` -- M3a path, Ewald lattice sum
        with the v_bg jellium shift.
    ``"smeared_erfc"``
        :func:`vibeqc.periodic_gapw_smearing.smeared_v_ne_full_gamma`
        -- M3b CP2K-style erfc/erf split with the a-correction.
        Auto-picks a from the grid cutoff when
        ``smearing_alpha`` is ``None``.
    """
    if convention == "ewald":
        return _ewald_v_ne_gamma(basis, system)
    if convention == "smeared_erfc":
        from .periodic_gapw_smearing import smeared_v_ne_full_gamma

        if grid is None:
            raise ValueError(
                "_build_v_ne: smeared_erfc convention requires a "
                "PlaneWaveGrid for the FFT-Poisson on r_core. "
                "Pass grid= or let the entry point auto-build one."
            )
        return smeared_v_ne_full_gamma(
            basis,
            system,
            alpha=smearing_alpha,
            grid=grid,
            quiet=True,
        )
    raise ValueError(
        f"_build_v_ne: unknown convention {convention!r}; "
        f"valid: 'ewald', 'smeared_erfc'."
    )


def _ewald_v_ne_gamma_fingerprint(basis, system) -> bytes:
    """Content fingerprint of the ``(basis, system)`` V_ne depends on.

    Deliberately **content**, not identity. An ``id()``-keyed cache is a
    correctness trap here: a geometry optimiser that mutates a system in
    place would keep the same object identity while moving the nuclei,
    and the memo would hand back a V_ne for the previous geometry -- a
    wrong answer rather than a slow one. Hashing the actual lattice,
    nuclei and shell parameters costs O(n_atoms + n_shells) against a
    build that costs seconds, so the safe choice is also the cheap one.
    """
    h = hashlib.blake2b(digest_size=32)
    h.update(np.ascontiguousarray(
        np.asarray(system.lattice, dtype=float)).tobytes())
    h.update(np.int64(int(system.dim)).tobytes())
    for atom in system.unit_cell:
        h.update(np.int64(int(atom.Z)).tobytes())
        h.update(np.ascontiguousarray(
            np.asarray(atom.xyz, dtype=float)).tobytes())
    for shell in basis.shells():
        h.update(np.int64(int(shell.l)).tobytes())
        h.update(np.int64(int(bool(shell.pure))).tobytes())
        h.update(np.ascontiguousarray(
            np.asarray(shell.origin, dtype=float)).tobytes())
        h.update(np.ascontiguousarray(
            np.asarray(shell.exponents, dtype=float)).tobytes())
        h.update(np.ascontiguousarray(
            np.asarray(shell.coefficients, dtype=float)).tobytes())
    return h.digest()


# Small LRU of Gamma V_ne matrices, keyed by content fingerprint. A GPW
# run asks for the same one three times -- once in the SCF, once in
# evaluate_gpw_energy, once in the runner adapter -- which measured 1.72 s
# of a 4.64 s LiH/STO-3G/PBE job (37% of wall, ~25% of it pure repeat).
# Depth 2 covers that plus one alternating neighbour (e.g. a finite
# difference stepping between two geometries) while holding at most two
# (nbf, nbf) float arrays.
_V_NE_GAMMA_CACHE: "OrderedDict[bytes, np.ndarray]" = OrderedDict()
_V_NE_GAMMA_CACHE_DEPTH = 2


def _ewald_v_ne_gamma(basis, system) -> np.ndarray:
    """Real, symmetric Γ-point V_ne in the FFT-Poisson gauge.

    Builds the periodic electron-nucleus attraction with the Ewald
    convention that drops the G = 0 reciprocal mode and applies the
    v_bg jellium shift, matching the gauge of the FFT-Poisson Hartree
    builder. This is the M3a lift: cells with significant nuclear
    Coulomb tails (ionic crystals; charged cells) now compute
    ``tr(D . V_ne) + 1/2 tr(D . J)`` without the Madelung-class shift
    that the molecular-limit V_ne introduced (the convention caveat
    on :class:`GpwEnergyBreakdown` pre-M3a).

    The lattice cutoffs default to the same values the other Ewald-3D
    periodic SCF drivers use (``LatticeSumOptions`` defaults:
    ``cutoff_bohr = 15.0``, ``nuclear_cutoff_bohr = 25.0``). The M3a
    test systems are 16-bohr cubes on light atoms, well inside these
    cutoffs; callers needing tighter / larger cutoffs would build
    their own ``LatticeSumOptions``, but the GPW driver does not
    surface that knob at M3a -- production-strength control lives on
    the C++-hosted SCF entry that M3b will swap to.
    """
    from .periodic_v_ne import compute_nuclear_lattice_dispatch

    key = _ewald_v_ne_gamma_fingerprint(basis, system)
    cached = _V_NE_GAMMA_CACHE.get(key)
    if cached is not None:
        _V_NE_GAMMA_CACHE.move_to_end(key)
        # Hand back a copy: callers own the array they receive, and one
        # of them scaling or symmetrising it in place must not corrupt
        # the next caller's V_ne.
        return cached.copy()

    lat_opts = _core.LatticeSumOptions()
    lat_opts.coulomb_method = _core.CoulombMethod.EWALD_3D
    V_lat = compute_nuclear_lattice_dispatch(basis, system, lat_opts)
    V_ne = _gamma_one_e(V_lat)

    _V_NE_GAMMA_CACHE[key] = V_ne.copy()
    while len(_V_NE_GAMMA_CACHE) > _V_NE_GAMMA_CACHE_DEPTH:
        _V_NE_GAMMA_CACHE.popitem(last=False)
    return V_ne


# ============================================================
# Minimal iterative GPW RHF SCF (M2-full)
# ============================================================


@dataclass(frozen=True)
class GpwScfResult:
    """Result of a converged single-point GPW periodic RHF SCF.

    Attributes
    ----------
    energy
        Total periodic energy in Hartree (the converged
        ``e_total`` from the breakdown).
    breakdown
        :class:`GpwEnergyBreakdown` evaluated at the converged
        density matrix.
    density
        ``(n_basis, n_basis)`` converged AO density matrix
        ``D = 2 . C_occ . C_occ.T``.
    mo_coeffs
        ``(n_basis, n_basis)`` MO coefficient matrix (canonical
        orthonormal AO basis).
    mo_energies
        ``(n_basis,)`` MO energies (Hartree).
    converged
        True iff the SCF reached both ``conv_tol_energy`` and
        ``conv_tol_density`` within ``max_iter`` iterations.
    n_iter
        Number of SCF iterations actually executed.
    grid
        :class:`PlaneWaveGrid` used for the Hartree-J build.
    """

    energy: float
    breakdown: GpwEnergyBreakdown
    density: np.ndarray
    mo_coeffs: np.ndarray
    mo_energies: np.ndarray
    converged: bool
    n_iter: int
    grid: PlaneWaveGrid
    # M3d: per-iter trace. Each entry is a dict with keys
    # 'iter', 'energy', 'delta_e', 'grad_norm', 'e_xc'.
    scf_trace: tuple = ()
    # v0.12-prep: D3-BJ periodic dispersion contribution to total
    # energy. Zero unless dispersion= was passed.
    e_dispersion: float = 0.0
    # v0.12 R1: DFT+U Dudarev on-site energy at the converged density.
    # Zero unless dft_plus_u_sites= was passed.
    e_dft_plus_u: float = 0.0
    # v0.12 R1: Fermi-Dirac smearing surface. ``smearing_temperature``
    # is the k_B T (Hartree) actually used; 0.0 means the integer-
    # occupation Aufbau path. ``smearing_entropy`` is the
    # dimensionless electronic entropy S/k_B per unit cell (zero at
    # T = 0). ``fermi_level`` is the chemical potential mu in
    # Hartree (midgap for T = 0 insulators). ``occupations`` is the
    # final ``(n_basis,)`` occupation array in [0, 2].
    smearing_temperature: float = 0.0
    smearing_entropy: float = 0.0
    fermi_level: float = 0.0
    occupations: tuple = ()
    # The converged Fock actually diagonalized to produce
    # ``mo_coeffs``/``mo_energies`` (post-DIIS extrapolation), and the
    # Gamma lattice overlap it was diagonalized against. Carried so the
    # runner adapter can surface the SCF's own operator instead of
    # re-deriving an HF-shaped ``Hcore + J - K/2`` from scratch (wrong
    # for KS/GAPW, and ~36% of a small GPW job's wall time). ``None``
    # on producers that predate the field; the adapter falls back to
    # the reconstruction in that case.
    fock: Optional[np.ndarray] = None
    overlap: Optional[np.ndarray] = None

    @property
    def free_energy(self) -> float:
        """Mermin free energy ``F = E - T.S``.

        Equals ``energy`` exactly when ``smearing_temperature ==
        0.0``. With smearing on, this is the quantity that is
        variational with respect to the occupations (the entropy
        term comes from the per-orbital ``f log f + (1-f) log(1-f)``
        bookkeeping on the closed-shell occupation in ``[0, 2]``).
        """
        return float(self.energy) - float(self.smearing_temperature) * float(
            self.smearing_entropy
        )

    restart_basis: object = None
    restart_lattice: object = None
    guess_selection: object = None

    restart_kpoints: object = None
    restart_weights: object = None


def _compute_d3_correction(
    system,
    dispersion: Optional[str],
    dispersion_functional: Optional[str],
) -> float:
    """Compute the periodic D3-BJ dispersion energy for ``system``.

    Thin shim over :func:`vibeqc.dispersion_periodic.compute_d3bj_periodic`
    used by the GAPW SCF entry points. Returns 0.0 when ``dispersion``
    is ``None`` / empty; otherwise normalises the request and dispatches
    with ``backend="dftd3"`` pinned (the bit-exact lattice sum). The
    approximate builtin supercell backend is never substituted here:
    if the dftd3 bindings are missing, this raises
    ``NotImplementedError`` with a remediation hint instead of silently
    degrading the SCF total energy.

    Parameters
    ----------
    system
        The :class:`PeriodicSystem` (positions + species drive the
        dispersion sum; the SCF method / functional do not).
    dispersion
        Dispersion model identifier. Currently only ``"d3-bj"`` /
        ``"d3bj"`` is wired; future models (D4, ...) would add here.
        ``None`` / ``""`` / ``"none"`` returns 0.0.
    dispersion_functional
        Functional name passed to the damping-parameter lookup
        (e.g. ``"pbe"``, ``"b3lyp"``). Required when ``dispersion``
        is set.

    Returns
    -------
    float
        Dispersion energy in Hartree (per unit cell).
    """
    if dispersion is None:
        return 0.0
    key = str(dispersion).strip().lower()
    if key in ("", "none", "false"):
        return 0.0
    if key not in ("d3-bj", "d3bj", "d3"):
        raise NotImplementedError(
            f"_compute_d3_correction: dispersion={dispersion!r} is "
            f"not wired on the GAPW path. Only 'd3-bj' is supported."
        )
    if not dispersion_functional:
        raise ValueError(
            "_compute_d3_correction: dispersion='d3-bj' requires "
            "dispersion_functional=<name> (e.g. 'pbe') so the D3-BJ "
            "damping parameters can be looked up."
        )
    try:
        from .dispersion_periodic import compute_d3bj_periodic
    except ImportError as exc:  # pragma: no cover
        raise NotImplementedError(
            "GAPW dispersion: vibeqc.dispersion_periodic is "
            "unavailable. Install the dispersion extra: "
            "`pip install -e '.[dispersion]'`."
        ) from exc
    # backend="dftd3" is pinned deliberately: the default "auto" would
    # silently fall back to the approximate builtin supercell backend
    # when the dftd3 bindings are missing, degrading the SCF total
    # energy with no diagnostic. A missing dependency must be loud.
    try:
        result = compute_d3bj_periodic(
            system,
            "d3bj",
            functional=dispersion_functional,
            backend="dftd3",
        )
    except ImportError as exc:
        raise NotImplementedError(
            "GAPW dispersion: the 'dftd3' python bindings are "
            "not installed. Run `pip install dftd3` (or "
            "`pip install -e '.[dispersion]'`) and retry."
        ) from exc
    return float(result.energy)


def run_periodic_rhf_gpw(
    system,
    basis,
    *,
    grid: Optional[PlaneWaveGrid] = None,
    cutoff_ha: Optional[float] = None,
    max_iter: int = 50,
    conv_tol_energy: float = 1e-9,
    conv_tol_density: float = 1e-7,
    damping: float = 0.0,
    initial_density: Optional[np.ndarray] = None,
    initial_guess: Optional[Union[str, _core.InitialGuess]] = "AUTO",
    v_ne_convention: str = "ewald",
    smearing_alpha: Optional[float] = None,
    functional: Optional[str] = None,
    use_diis: bool = True,
    diis_subspace_size: int = 8,
    diis_start_iter: int = 2,
    dispersion: Optional[str] = None,
    dispersion_functional: Optional[str] = None,
    smearing_temperature: float = 0.0,
    smearing_method: str = "fermi-dirac",
    dft_plus_u_sites=None,
    quiet: bool = False,
    use_davidson: bool = False,
    davidson_min_dim: int = 100,
    external_xc_grid_options=None,
    external_xc_image_radius_bohr: float = 10.0,
    external_xc_lattice_options=None,
    progress: Union[bool, ProgressLogger, None] = None,
) -> GpwScfResult:
    """Minimal closed-shell GPW RHF SCF on a periodic cell.

    > **Experimental.** Every GPW SCF emits ``GAPWExperimentalWarning``;
    > the route is production-quality for vacuum-padded /
    > pseudopotential-regime cells (it reproduces the molecular RHF limit
    > to sub-µHa) but does not yet reach all-electron CP2K parity on tight
    > ionic crystals -- the GAPW augmentation is the open roadmap item.
    > This Python SCF iterates the AO density to self-consistency under
    > the GPW Hartree-J kernel with **Pulay DIIS** (``use_diis=True`` by
    > default) and optional linear ``damping``; pass ``functional=`` for
    > RKS-DFT (or use :func:`run_periodic_rks_gpw`). A C++ SCF host
    > (:func:`run_rhf_scf_gpw_cpp` via the ``PyGpwJKBuilder`` trampoline)
    > is also available for the production C++ SCF (EDIIS+DIIS / dynamic
    > damping / level-shift).

    The build is hybrid: J via :class:`GpwJBuilder` on the smooth
    real-space grid (the M2 GPW kernel), K via
    :func:`build_jk_gamma_molecular_limit` -- despite the historical
    name, this is the Γ-only periodic K via direct AO-image sum at
    a lattice cutoff of 25 bohr, not a true molecular-limit K. T
    and S use the periodic lattice builders + Bloch sum at Γ (B3).
    V_ne uses the periodic Ewald gauge (M3a lift via
    :func:`_ewald_v_ne_gamma`) so the FFT-Poisson J and V_ne live
    in the same gauge -- the converged total energy no longer
    carries the "2x Madelung" shift the M2-full driver did.

    Closed-shell only. The number of doubly-occupied orbitals is
    derived from ``system.unit_cell`` as
    ``sum(atom.Z for atom in system.unit_cell) // 2`` and **must**
    be an integer (i.e., the cell must have an even number of
    electrons). Open-shell + DFT lands at M3 / M4.

    Parameters
    ----------
    system
        :class:`vibeqc._vibeqc_core.PeriodicSystem`.
    basis
        :class:`vibeqc.BasisSet` for the system.
    grid
        Pre-built :class:`PlaneWaveGrid`. Pass ``None`` to let
        the helper construct one via :func:`make_grid` from
        ``cutoff_ha``.
    cutoff_ha
        PW cutoff (Ha) when ``grid`` is None. Default 300 Ha.
    max_iter
        Hard cap on SCF iterations.
    conv_tol_energy
        ``|E - E_prev| < conv_tol_energy`` plus the density
        criterion below define convergence.
    conv_tol_density
        ``||D - D_prev||_F < conv_tol_density`` is the density
        criterion.
    damping
        Simple linear-density damping factor in ``[0, 1)``:
        ``D_new = (1 - damping) . D_new + damping . D_prev``.
        Default 0.0 (no damping) -- fine for compact systems;
        bump to 0.3-0.5 for tight ionic crystals.
    initial_density
        ``(n_basis, n_basis)`` initial AO density. Default
        ``None`` falls back to the Hcore guess (diagonalise
        Hcore in the orthonormal AO basis).
    v_ne_convention
        ``"ewald"`` (default) -> M3a Ewald lattice V_ne;
        ``"smeared_erfc"`` -> M3b CP2K-style erfc/erf split.
        Physically equivalent on neutral cells (parity pinned to
        < 1 mHa on He / H2 STO-3G across a in [0.5, 2.5] bohr⁻¹).
    smearing_alpha
        Smearing exponent (bohr⁻¹) when
        ``v_ne_convention = "smeared_erfc"``. Default ``None`` ->
        auto-pick from the grid via
        :func:`vibeqc.periodic_gapw_smearing.default_smearing_alpha_from_grid`.
        Ignored when ``v_ne_convention = "ewald"``.
    dispersion, dispersion_functional
        Optional D3-BJ dispersion correction, added to the total
        energy after convergence (system-only; does not enter the
        Fock loop). Requires the ``dftd3`` python bindings.
    quiet
        Suppress the GAPWExperimentalWarning.
    progress
        Live per-iteration progress passthrough (``True`` / a
        :class:`~vibeqc.progress.ProgressLogger` / ``None``). Each SCF
        cycle calls ``plog.iteration(it, energy=..., dE=..., grad=...)``,
        which is what the periodic runner's QVF checkpointer wraps to fire
        a per-iteration checkpoint snapshot.

    Returns
    -------
    result
        :class:`GpwScfResult` with the converged energy, density,
        MO coefficients, breakdown, and convergence flag.

    Raises
    ------
    ValueError
        If the cell has an odd number of electrons (open shell)
        or if ``system.dim != 3``.
    NotImplementedError
        If ``dispersion=`` is requested but the ``dftd3`` python
        bindings are not installed.
    """
    input_restart_supplied = initial_density is not None
    from .guess import _coerce_periodic_driver_guess

    requested_initial_guess = initial_guess
    initial_guess = _coerce_periodic_driver_guess(
        initial_guess,
        driver="run_periodic_rhf_gpw",
        supported=periodic_guess_capabilities('gpw', 'RHF', dim=getattr(system, "dim", 3), multi_k=False, transport='k'),
        restart_supplied=initial_density is not None,
    )
    if initial_density is not None:
        from .guess_read import _real_density
        initial_density = _real_density(initial_density)

    if not quiet:
        _warn_experimental(
            "run_periodic_rhf_gpw: GPW SCF with Pulay DIIS; "
            "the route is production-quality for vacuum-padded / "
            "pseudopotential-regime cells (sub-µHa vs molecular RHF)"
        )

    plog = resolve_progress(progress)

    if system.dim != 3:
        raise ValueError(
            f"run_periodic_rhf_gpw: only dim == 3 is supported "
            f"(got dim={system.dim}); 1D/2D Ewald is M3+ work"
        )

    if functional is not None:
        requested_func = _core.Functional(functional, 1)
        if (
            bool(getattr(requested_func, "is_external", False))
            and float(system.charge) != 0.0
        ):
            raise NotImplementedError(
                "run_periodic_rhf_gpw: charged-cell external XC is not "
                "supported because the GPW Coulomb background and electron-"
                "count convention are not validated; use a neutral cell"
            )
        if bool(getattr(requested_func, "is_external", False)):
            from .pbc_bipole_common import reject_bipole_ecp_options
            from .periodic_external_xc import (
                _require_zero_temperature_external_xc,
            )

            _require_zero_temperature_external_xc(
                smearing_temperature,
                where="run_periodic_rhf_gpw",
            )

            try:
                reject_bipole_ecp_options(
                    object(),
                    driver="run_periodic_rhf_gpw external XC",
                    basis=basis,
                    system=system,
                )
            except NotImplementedError as exc:
                raise NotImplementedError(
                    "run_periodic_rhf_gpw: ECP-bearing bases are not "
                    "implemented with external XC; use an all-electron basis"
                ) from exc

    n_elec = int(sum(int(a.Z) for a in system.unit_cell))
    if n_elec % 2 != 0:
        raise ValueError(
            f"run_periodic_rhf_gpw: cell has {n_elec} electrons "
            f"(odd); closed-shell RHF needs an even count. UHF / "
            f"UKS support is M3+ work."
        )
    n_occ = n_elec // 2

    # v0.12 R1: Fermi-Dirac smearing knobs. ``smearing_temperature``
    # is k_B T in Hartree; 0.0 reproduces the integer-occupation
    # Aufbau path bit-for-bit. The flavor is fixed to "fermi-dirac"
    # at v0.12 -- Methfessel-Paxton / cold smearing plug in via the
    # shared apply_smearing dispatcher later.
    smearing_T = float(smearing_temperature)
    if smearing_T < 0.0:
        raise ValueError(
            f"run_periodic_rhf_gpw: smearing_temperature must be >= 0; got {smearing_T}"
        )
    if smearing_T > 0.0:
        smearing_opts = _SmearingOptions(
            temperature=smearing_T,
            flavor=str(smearing_method),
        )
    else:
        smearing_opts = None

    # Build grid if needed.
    if grid is None:
        from .periodic_gapw_grid import make_grid as _make_grid

        if cutoff_ha is None:
            cutoff_ha = 300.0
        grid = _make_grid(np.asarray(system.lattice, dtype=float), cutoff_ha=cutoff_ha)

    # One-electron integrals. T, S, V_ne all use the periodic lattice
    # builders + Bloch sum at Γ. T/S decay exponentially with shell
    # separation so the lattice path matches the molecular limit on
    # vacuum-padded cells and stays correct on tighter ones (B3). V_ne
    # is built per the requested convention (M3a "ewald" by default
    # or M3b "smeared_erfc") -- both gauge-aligned with the FFT-Poisson
    # J on neutral cells.
    T = _kinetic_lattice_gamma(basis, system)
    V_ne = _build_v_ne(
        basis,
        system,
        v_ne_convention,
        smearing_alpha,
        grid,
    )
    S = _overlap_lattice_gamma(basis, system)
    Hcore = T + V_ne

    # Ewald nuclear repulsion.
    E_nn = float(_core.ewald_nuclear_repulsion(system, _core.EwaldOptions()))

    # Build the iteration-invariant AO-on-grid table chi (+ G-mesh) ONCE for
    # this (basis, grid) and share it across the whole SCF: the J builder
    # and every XC-side collocate / project below contract against it
    # instead of rebuilding chi (~95-100 % of a collocate) each iteration.
    collocation_cache = build_gpw_collocation_cache(basis, grid)

    # JK builders (J via GPW, K via molecular limit).
    gpw = GpwJBuilder(basis, grid, collocation_cache=collocation_cache)
    lo = _core.LatticeSumOptions()
    lo.cutoff_bohr = 25.0

    # Initial density. Every density-mode guess goes through the shared
    # periodic adapter; HCORE deliberately returns ``None`` so this route can
    # retain its own one-electron Hamiltonian and orthogonalisation. An
    # explicitly supplied density (restart) always takes precedence.
    if initial_density is None:
        from .guess import initial_density_closed_shell

        initial_density = initial_density_closed_shell(
            system.unit_cell_molecule(),
            basis,
            n_occ,
            _core.InitialGuess.SAD if initial_guess == _core.InitialGuess.PATOM else initial_guess,
            is_periodic=True,
            periodic_system=system,
            lattice_opts=_core.LatticeSumOptions(),
            overlap=S,
        )
    if initial_density is None:
        # Hcore guess: diagonalise Hcore in the canonical-orthonormal
        # AO basis (S^{-1/2} Hcore S^{-1/2}).
        s_eigs, U = _eigh_safe(S)
        S_half_inv = U @ np.diag(1.0 / np.sqrt(s_eigs)) @ U.T
        e, C_orth = _eigh_safe(S_half_inv @ Hcore @ S_half_inv)
        C = S_half_inv @ C_orth
        D = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T
    else:
        D = np.asarray(initial_density, dtype=float).copy()
        if D.shape != (basis.nbasis, basis.nbasis):
            raise ValueError(
                f"initial_density shape {D.shape} doesn't match "
                f"basis ({basis.nbasis} functions)"
            )

    # Pre-build orthogonalizer S^{-1/2} once (S is fixed across SCF).
    s_eigs, U = _eigh_safe(S)
    S_half_inv = U @ np.diag(1.0 / np.sqrt(s_eigs)) @ U.T

    from .guess import normalize_density_guess
    D = normalize_density_guess(D, S, 2 * n_occ)

    if initial_guess == _core.InitialGuess.PATOM and not input_restart_supplied:
        J_seed = gpw.build_J(D)
        K_seed = np.asarray(_core.build_jk_gamma_molecular_limit(basis, system, lo, D).K)
        _, C_seed = _eigh_safe(S_half_inv @ (Hcore + J_seed - 0.5 * K_seed) @ S_half_inv)
        C_seed = S_half_inv @ C_seed
        D = 2 * C_seed[:, :n_occ] @ C_seed[:, :n_occ].T

    use_dav = use_davidson and S.shape[0] >= davidson_min_dim
    dav_opts = None
    if use_dav:
        from vibeqc._vibeqc_core import DavidsonOptions

        dav_opts = DavidsonOptions()

    # Optional functional.
    if functional is not None:
        func = _core.Functional(functional, 1)
        # The GPW grid K is full-range only; screened hybrids must not
        # silently run as their full-range twins.
        reject_unscreened_range_separated(
            func, where="run_periodic_rhf_gpw"
        )
        ex_frac = float(func.hf_exchange_fraction)
    else:
        func = None
        ex_frac = 1.0  # Pure HF

    external_xc = None
    if func is not None and bool(getattr(func, "is_external", False)):
        from .periodic_external_xc import PeriodicExternalXC

        external_xc = PeriodicExternalXC.prepare(
            basis,
            system,
            func,
            grid_options=external_xc_grid_options,
            image_radius_bohr=external_xc_image_radius_bohr,
            lattice_options=(
                external_xc_lattice_options
                if external_xc_lattice_options is not None
                else lo
            ),
        )

    # Pulay-DIIS state (M3d).
    diis = _make_diis(use_diis, diis_subspace_size)

    # v0.12 R1: DFT+U (Dudarev) on-site setup. Closed-shell pattern
    # mirrors run_pbc_bipole_rhf: build the AO-index groups once,
    # then per iter compute n_s from P_s = P_total/2, the Dudarev
    # V_AO_s = U_eff (1/2I - n_s), and add S V_AO_s S to F (per spin ->
    # the same matrix on both halves of the closed-shell Fock).
    # E_total_+U = 2 x E_s.
    from .dft_plus_u import (
        _v_ao_per_spin as _dftu_v_ao_per_spin,
    )
    from .dft_plus_u import (
        ao_group_indices as _dftu_ao_group_indices,
    )
    from .dft_plus_u import (
        compute_dudarev_energy as _dftu_compute_dudarev_energy,
    )
    from .dft_plus_u import (
        compute_occupation_matrices as _dftu_compute_occupation_matrices,
    )

    _dftu_sites = list(dft_plus_u_sites or ())
    if _dftu_sites:
        _dftu_ao_groups = _dftu_ao_group_indices(basis)
        for _site in _dftu_sites:
            _key = (_site.atom_index, _site.l)
            if _key not in _dftu_ao_groups:
                raise ValueError(
                    f"run_periodic_rhf_gpw: HubbardSite{_key} has no "
                    f"AOs in the basis. Available channels: "
                    f"{sorted(_dftu_ao_groups.keys())}"
                )
    else:
        _dftu_ao_groups = {}
    e_dft_plus_u = 0.0

    # Per-iter SCF trace (M3d task #21).
    scf_trace: list[dict] = []

    # SCF iteration.
    E_prev = 0.0
    converged = False
    n_iter = 0
    F = None
    C = np.zeros((basis.nbasis, basis.nbasis))
    e = np.zeros(basis.nbasis)
    # v0.12 R1: occupations from smearing dispatcher (or hard Aufbau).
    occ_final = np.zeros(basis.nbasis)
    occ_final[:n_occ] = 2.0
    fermi_level = 0.0
    entropy = 0.0
    for it in range(1, max_iter + 1):
        J = gpw.build_J(D)
        if ex_frac > 0.0:
            jk_mol = _core.build_jk_gamma_molecular_limit(
                basis,
                system,
                lo,
                D,
            )
            K = np.asarray(jk_mol.K)
        else:
            K = np.zeros_like(D)
        F = Hcore + J - 0.5 * ex_frac * K

        # v0.12 R1: DFT+U Fock contribution. Closed-shell: P_s = D/2,
        # build V_AO_s via the same _v_ao_per_spin primitive the
        # molecular and BIPOLE paths use; add S V_AO_s S to F (the
        # per-spin Fock -- same on a and b for closed shell, so it just
        # adds once into the closed-shell F). The energy is the spin-
        # summed Dudarev energy = 2 x E_s.
        e_dft_plus_u = 0.0
        V_U_fock = None
        if _dftu_sites:
            P_sigma = 0.5 * D
            n_sigma_map = _dftu_compute_occupation_matrices(
                _dftu_sites,
                P_sigma,
                S,
                _dftu_ao_groups,
            )
            e_sigma = _dftu_compute_dudarev_energy(
                _dftu_sites,
                n_sigma_map,
            )
            e_dft_plus_u = 2.0 * float(e_sigma)
            V_AO_sigma = _dftu_v_ao_per_spin(
                _dftu_sites,
                P_sigma,
                S,
                _dftu_ao_groups,
            )
            V_U_fock = S @ V_AO_sigma @ S
            F = F + V_U_fock

        # XC piece (M3d). All grid primitives reuse the per-SCF chi cache.
        if func is not None:
            if external_xc is not None:
                e_xc_iter, V_xc = external_xc.build_gamma(D)
            else:
                rho_grid = collocate_density_on_grid(
                    basis, D, grid, cache=collocation_cache
                )
                # basis + density_matrix let meta-GGA functionals build t(r);
                # LDA/GGA ignore them.
                (
                    e_xc_iter,
                    v_xc_grid,
                    v_sigma_grid,
                    grad_rho,
                    v_tau_grid,
                ) = _evaluate_xc_on_grid(
                    rho_grid,
                    grid,
                    func,
                    basis=basis,
                    density_matrix=D,
                    cache=collocation_cache,
                )
                V_xc = _project_vxc_to_ao(
                    basis,
                    v_xc_grid,
                    grid,
                    v_sigma_grid=v_sigma_grid,
                    grad_rho=grad_rho,
                    cache=collocation_cache,
                )
                # Meta-GGA: t contribution to the generalized-KS Fock,
                # V_tau[muν] = 1/2 ∫ v_tau gradchi_mu.gradchi_ν dr (see
                # _project_vtau_to_ao -- the multiplicative ∫ chichi v_tau
                # projection is NOT the derivative of E_xc).
                if v_tau_grid is not None:
                    V_tau = _project_vtau_to_ao(
                        basis, v_tau_grid, grid, cache=collocation_cache
                    )
                    V_xc = V_xc + V_tau
            F = F + V_xc
        else:
            e_xc_iter = 0.0

        # Energy: E_elec = 0.5 tr(D.(Hcore + F - V_xc)) + E_xc.
        # For pure HF this collapses to the familiar 0.5 tr(D.(Hcore+F)).
        # v0.12 R1: when DFT+U is on, F includes V_U_fock = S V_AO_s S
        # which would contribute 0.5.tr(D.V_U_fock) under the standard
        # formula; we drop that and add the explicit Dudarev energy
        # e_dft_plus_u so the bookkeeping matches the BIPOLE convention
        # (E_total = E_elec_no_U + E_nn + e_dft_plus_u).
        if func is None:
            E_elec = 0.5 * float(np.einsum("ij,ij->", D, Hcore + F))
            if V_U_fock is not None:
                E_elec -= 0.5 * float(np.einsum("ij,ij->", D, V_U_fock))
            E = E_elec + E_nn + e_dft_plus_u
        else:
            # Re-derive electronic energy in the DFT convention. The
            # +U piece never enters E_elec here, so just add it on top.
            E_elec = (
                float(np.einsum("ij,ij->", D, Hcore))
                + 0.5 * float(np.einsum("ij,ij->", D, J))
                - 0.25 * ex_frac * float(np.einsum("ij,ij->", D, K))
                + e_xc_iter
            )
            E = E_elec + E_nn + e_dft_plus_u

        # DIIS error r = FDS - SDF (in the orthonormal AO basis). ``D`` is the
        # density that built ``F``; see tests/test_diis_error_vector_source.py.
        if diis is not None and it >= diis_start_iter:
            err = F @ D @ S - S @ D @ F
            err = S_half_inv @ err @ S_half_inv
            F = diis.extrapolate(F, err)

        # Solve F C = e S C.
        e, C_orth = _eigh_safe(
            S_half_inv @ F @ S_half_inv, use_dav=use_dav, dav_opts=dav_opts
        )
        C = S_half_inv @ C_orth

        # v0.12 R1: build D from fractional occupations when smearing
        # is on. Γ-only single-k path -> weights = [1.0]. At T = 0 the
        # dispatcher reproduces the integer-Aufbau path bit-for-bit.
        sm_result = _apply_smearing(
            [e],
            weights=[1.0],
            n_electrons_per_cell=float(n_elec),
            n_occ_each=n_occ,
            smearing=smearing_opts,
        )
        occ_final = np.asarray(sm_result.occupations_per_k[0], dtype=float)
        fermi_level = float(sm_result.mu)
        entropy = float(sm_result.entropy)
        if smearing_T > 0.0:
            D_new = (C * occ_final[None, :]) @ C.T
        else:
            D_new = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T
        if damping > 0.0:
            D_new = (1.0 - damping) * D_new + damping * D

        dE = E - E_prev
        dD = float(np.linalg.norm(D_new - D))
        n_iter = it
        scf_trace.append(
            {
                "iter": it,
                "energy": float(E),
                "delta_e": float(dE),
                "grad_norm": float(dD),
                "e_xc": float(e_xc_iter),
            }
        )
        # Per-iteration progress hook. When the periodic runner passes a
        # checkpoint-wrapped ``plog``, this drives the live QVF cadence
        # snapshot (energy read from the ``energy`` field); a plain logger
        # just prints the iteration row.
        plog.iteration(
            it,
            energy=float(E),
            dE=float(dE),
            grad=float(dD),
            diis=(diis.subspace_size if diis is not None else 0),
        )
        if abs(dE) < conv_tol_energy and dD < conv_tol_density and it > 1:
            converged = True
            D = D_new
            break
        D = D_new
        E_prev = E

    # Compute final breakdown at the converged D. Reuse the SCF's collocation
    # cache (same basis + grid) so the breakdown does not repeat the 27-image
    # evaluate_ao collocation -- ~halves the single-point GPW cost on a fine
    # grid (the chi build dominates), which is what the FD gradient/Hessian
    # tests pay per displaced geometry.
    breakdown = evaluate_gpw_energy(
        system,
        basis,
        D,
        grid=grid,
        quiet=True,
        functional=functional,
        collocation_cache=collocation_cache,
        external_xc_grid_options=external_xc_grid_options,
        external_xc_image_radius_bohr=external_xc_image_radius_bohr,
        external_xc_lattice_options=external_xc_lattice_options,
        _external_xc_context=external_xc,
    )

    # v0.12-prep: D3-BJ periodic dispersion add-on. System-only
    # correction (positions + species drive it, not the SCF method),
    # added after convergence rather than inside the Fock loop because
    # it produces no AO matrix contribution.
    e_disp = _compute_d3_correction(
        system,
        dispersion,
        dispersion_functional,
    )
    if e_disp != 0.0:
        from dataclasses import replace as _dc_replace

        breakdown = _dc_replace(
            breakdown,
            e_total=breakdown.e_total + e_disp,
            e_dispersion=e_disp,
        )

    # v0.12 R1: DFT+U bookkeeping -- recompute at the converged density
    # so the breakdown reports a stable +U energy. The SCF-loop value
    # is from the second-to-last F build, which differs from this one
    # by the same residual that drives ΔE < tol.
    e_dft_plus_u_final = 0.0
    if _dftu_sites:
        P_sigma_final = 0.5 * D
        n_sigma_final = _dftu_compute_occupation_matrices(
            _dftu_sites,
            P_sigma_final,
            S,
            _dftu_ao_groups,
        )
        e_dft_plus_u_final = 2.0 * float(
            _dftu_compute_dudarev_energy(_dftu_sites, n_sigma_final)
        )
        from dataclasses import replace as _dc_replace

        breakdown = _dc_replace(
            breakdown,
            e_total=breakdown.e_total + e_dft_plus_u_final,
            e_dft_plus_u=e_dft_plus_u_final,
        )
    total_energy = breakdown.e_total

    if external_xc is not None:
        # Return the physical KS operator belonging to the same accepted
        # density as the final nonlinear energy.  The loop-local F/C/e may
        # contain a preceding density generation and a DIIS extrapolate,
        # which is useful for iteration but is not a valid result operator.
        J_final = gpw.build_J(D)
        if ex_frac > 0.0:
            jk_final = _core.build_jk_gamma_molecular_limit(
                basis,
                system,
                lo,
                D,
            )
            K_final = np.asarray(jk_final.K)
        else:
            K_final = np.zeros_like(D)
        F_final = Hcore + J_final - 0.5 * ex_frac * K_final
        if _dftu_sites:
            V_AO_sigma_final = _dftu_v_ao_per_spin(
                _dftu_sites,
                0.5 * D,
                S,
                _dftu_ao_groups,
            )
            F_final = F_final + S @ V_AO_sigma_final @ S
        _e_xc_final, V_xc_final = external_xc.build_gamma(D)
        F_final = F_final + V_xc_final
        F_final = 0.5 * (F_final + F_final.T)
        e, C_orth = _eigh_safe(
            S_half_inv @ F_final @ S_half_inv,
            use_dav=use_dav,
            dav_opts=dav_opts,
        )
        C = S_half_inv @ C_orth
        final_smearing = _apply_smearing(
            [e],
            weights=[1.0],
            n_electrons_per_cell=float(n_elec),
            n_occ_each=n_occ,
            smearing=smearing_opts,
        )
        occ_final = np.asarray(
            final_smearing.occupations_per_k[0], dtype=float
        )
        fermi_level = float(final_smearing.mu)
        entropy = float(final_smearing.entropy)
        F = F_final

    return GpwScfResult(
               restart_kpoints=np.zeros((1, 3)), restart_weights=np.ones(1),
               guess_selection=periodic_result_selection(system, requested_initial_guess, restarted=input_restart_supplied),
               restart_basis=basis, restart_lattice=np.asarray(system.lattice).copy(),
        energy=total_energy,
        breakdown=breakdown,
        density=D,
        mo_coeffs=C,
        mo_energies=e,
        converged=converged,
        n_iter=n_iter,
        grid=grid,
        scf_trace=tuple(scf_trace),
        e_dispersion=e_disp,
        e_dft_plus_u=e_dft_plus_u_final,
        smearing_temperature=smearing_T,
        smearing_entropy=entropy,
        fermi_level=fermi_level,
        occupations=tuple(occ_final.tolist()),
        fock=F,
        overlap=S,
    )


@dataclass(frozen=True)
class GpwMultiKScfResult:
    """Result of a converged multi-k GPW periodic RKS SCF (M3e).

    Same field shape as :class:`GpwScfResult` except the orbital,
    occupation, and accepted-density blocks carry per-k lists. ``energy``
    is the total band-summed BZ-averaged energy.

    Attributes
    ----------
    energy
        Total energy (Hartree). Equals ``S_k w_k . E_band(k) +
        E_xc + E_nn`` for the converged density (RKS pure-DFT).
    breakdown
        :class:`GpwEnergyBreakdown` at the converged density.
    mo_coeffs_k
        List of ``(n_basis, n_basis)`` complex MO coefficient
        matrices, one per k-point.
    mo_energies_k
        List of ``(n_basis,)`` real MO eigenvalue arrays per k.
    density
        ``(n_basis, n_basis)`` Γ-summed AO density.

        **Only the molecular-limit regime builds the Hartree-J and
        V_xc from this field.** On a compact crystal the driver takes
        the Bloch branch of ``_collocate_multik`` and builds them from
        the per-k tables instead, and this Γ-fold is then a diagnostic
        summary, not the object the Fock was made from. It is *not*
        consumable by the Γ-folded helpers: feeding a compact-crystal
        value to :func:`collocate_density_on_grid` or
        :func:`evaluate_gpw_energy` yields nonsense (measured on
        LiF/def2-SVP/(2,2,2): 2.2e5 electrons and +8.9e7 Ha, against a
        driver-reported -125.4 Ha for the same state), because the
        near-null Bloch components that cancel under the per-k metric
        do not cancel under a Γ fold. Use the per-k quantities for any
        post-hoc analysis of a compact-crystal run.
    density_k
        Accepted ``(n_basis, n_basis)`` complex AO density matrix at each
        k-point.  This is the authoritative density used for compact-cell
        Hartree/XC reconstruction; it is intentionally preserved when final
        canonical orbitals are reported after an unconverged iteration-limit
        exit.
    converged
        SCF convergence flag.
    n_iter
        Number of SCF iterations executed.
    grid
        :class:`PlaneWaveGrid` for the FFT path.
    kmesh
        The :class:`BlochKMesh` used.
    fock_k
        Physical terminal KS matrices evaluated at ``density_k``.  These are
        populated for external-XC calculations so post-SCF consumers see the
        same operator whose canonical eigenpairs are returned, rather than a
        loop-local DIIS extrapolate.
    overlap_k
        Per-k overlap matrices paired with ``fock_k``.
    hcore_k
        Per-k one-electron Hamiltonians paired with ``fock_k``.
    scf_trace
        Per-iter SCF records.
    """

    energy: float
    breakdown: "GpwEnergyBreakdown"
    mo_coeffs_k: tuple
    mo_energies_k: tuple
    density: np.ndarray
    density_k: tuple
    converged: bool
    n_iter: int
    grid: PlaneWaveGrid
    kmesh: object
    scf_trace: tuple = ()
    # v0.12-prep: D3-BJ periodic dispersion contribution. Zero unless
    # dispersion= was passed on the SCF entry point.
    e_dispersion: float = 0.0
    # v0.12 R1: DFT+U Dudarev on-site energy at the converged density.
    # Zero unless dft_plus_u_sites= was passed.
    e_dft_plus_u: float = 0.0
    # CRYSTAL-style Fock/KS matrix mixing used by the multi-k SCF loop.
    fock_mixing: float = 0.0
    # v0.12 R1: Fermi-Dirac smearing surface. ``occupations_k`` is a
    # tuple of per-k ``(n_basis,)`` occupation arrays in [0, 2],
    # built from the BZ-shared chemical potential ``fermi_level``
    # via the standard Fermi-Dirac distribution.
    smearing_temperature: float = 0.0
    smearing_entropy: float = 0.0
    fermi_level: float = 0.0
    occupations_k: tuple = ()
    # Keep newly added optional fields at the end so positional construction
    # of the pre-existing public result shape remains backward compatible.
    fock_k: tuple = ()
    overlap_k: tuple = ()
    hcore_k: tuple = ()

    @property
    def free_energy(self) -> float:
        """Mermin free energy ``F = E - T.S``.

        Equals ``energy`` exactly when ``smearing_temperature ==
        0.0``.
        """
        return float(self.energy) - float(self.smearing_temperature) * float(
            self.smearing_entropy
        )

    restart_basis: object = None
    restart_lattice: object = None
    guess_selection: object = None

    restart_kpoints: object = None
    restart_weights: object = None


def _normalise_initial_density_k(
    initial_density_k: Sequence[np.ndarray],
    *,
    n_k: int,
    n_basis: int,
    label: str,
) -> list[np.ndarray]:
    """Validate caller-supplied per-k density blocks."""
    blocks = list(initial_density_k)
    if len(blocks) != int(n_k):
        raise ValueError(
            f"{label}: initial_density_k has {len(blocks)} blocks; "
            f"expected {int(n_k)} for the target k-mesh."
        )
    out: list[np.ndarray] = []
    for ik, block in enumerate(blocks):
        D = np.asarray(block, dtype=complex)
        if D.shape != (int(n_basis), int(n_basis)):
            raise ValueError(
                f"{label}: initial_density_k[{ik}] has shape {D.shape}; "
                f"expected {(int(n_basis), int(n_basis))}."
            )
        out.append(0.5 * (D + D.conj().T))
    return out


def _weighted_real_density_from_k(
    density_k: Sequence[np.ndarray],
    weights: Sequence[float],
    n_basis: int,
) -> np.ndarray:
    """Return the real Gamma-summed density from per-k AO density blocks."""
    D_total = np.zeros((int(n_basis), int(n_basis)), dtype=float)
    for weight, Dk in zip(weights, density_k):
        D_total += float(weight) * np.real(np.asarray(Dk, dtype=complex))
    return 0.5 * (D_total + D_total.T)


def run_periodic_rks_gpw(system, basis, *, functional, **kwargs):
    """Closed-shell Γ-only RKS-DFT SCF on a periodic cell via the GPW route.

    Thin, explicit alias for :func:`run_periodic_rhf_gpw` with a **required**
    ``functional=``: the Γ-only RHF and RKS paths share one SCF loop (the
    runner already routes ``method="RKS"`` through ``run_periodic_rhf_gpw``
    with a functional), and this name makes the DFT entry discoverable --
    mirroring :func:`run_periodic_rks_gpw_multi_k`. All other keyword
    arguments are forwarded unchanged; returns the same
    :class:`GpwScfResult`.

    Example
    -------
    >>> result = run_periodic_rks_gpw(system, basis, functional="lda")
    """
    if not functional:
        raise ValueError(
            "run_periodic_rks_gpw requires a functional= (e.g. 'lda', "
            "'pbe', 'b3lyp'); for Hartree-Fock use run_periodic_rhf_gpw()."
        )
    return run_periodic_rhf_gpw(system, basis, functional=functional, **kwargs)


def run_periodic_rks_gpw_multi_k(
    system,
    basis,
    kmesh,
    *,
    functional: str,
    grid: Optional[PlaneWaveGrid] = None,
    cutoff_ha: Optional[float] = None,
    max_iter: int = 80,
    conv_tol_energy: float = 1e-8,
    conv_tol_density: float = 1e-6,
    v_ne_convention: str = "ewald",
    smearing_alpha: Optional[float] = None,
    dispersion: Optional[str] = None,
    dispersion_functional: Optional[str] = None,
    smearing_temperature: float = 0.0,
    smearing_method: str = "fermi-dirac",
    fock_mixing: Optional[float] = None,
    dft_plus_u_sites=None,
    initial_density_k: Optional[Sequence[np.ndarray]] = None,
    initial_guess: Optional[Union[str, _core.InitialGuess]] = "AUTO",
    quiet: bool = False,
    external_xc_grid_options=None,
    external_xc_image_radius_bohr: float = 10.0,
    external_xc_lattice_options=None,
    progress: Union[bool, ProgressLogger, None] = None,
) -> "GpwMultiKScfResult":
    """Multi-k GPW periodic RKS SCF (closed-shell, pure DFT only).

    M3e closure of the "multi-k GPW" open item. Restrictions:

    * **Closed-shell RKS only** -- no HF / UHF / UKS. HF exchange
      would need per-k K builders which our existing
      ``build_jk_gamma_molecular_limit`` doesn't supply; the
      pure-DFT path is the natural multi-k destination because
      V_xc + J both come from the Γ-summed total density on the
      FFT grid.
    * **Pure DFT functionals only** -- ``hf_exchange_fraction``
      must be 0. Hybrids raise ``NotImplementedError``.
    * **Two collocation regimes** -- vacuum-padded / molecular-limit
      cells fold the density to a single Gamma-summed density matrix
      on the smooth grid; compact crystals take the Bloch path (true
      Bloch real-space density + per-k smooth-potential projection,
      linear-dependence projection, per-k Pulay DIIS). Dispatched by
      :func:`_multik_gpw_is_molecular_limit`.
    * **Meta-GGA works on both regimes** -- molecular-limit cells build
      t(r) from the Gamma-summed density; compact cells build the Bloch
      t from per-k Bloch AO gradients
      (:func:`compute_bloch_kinetic_energy_density`) and add the per-k
      ``1/2 v_tau gradchi_k*.gradchi_k`` Fock block
      (:func:`project_vtau_to_bloch_ao`). The von Weizsacker floor,
      density screen, and fail-closed stiffness guard apply on both
      (TPSS-class functionals whose iso-orbital derivatives diverge on
      the uniform grid raise; use r2scan / scan).

    The SCF loop:
      1. Build T_lat, S_lat, V_ne_lat once.
      2. For each iter:
         a. Bloch-sum T, S, V_ne, F per k.
         b. Eigen-solve F(k) C(k) = e S(k) C(k) per k.
         c. Build r(r) on the FFT grid from C(k), w_k -- Gamma-folded
            in the molecular limit, full Bloch sum for compact cells.
         d. FFT-Poisson J and libxc V_xc on r.
         e. Project the local potential to AO -- once on Gamma AOs in
            the molecular limit (J + V_xc are k-independent there), or
            per k onto Bloch AOs for compact cells.
         f. Update F(k) = Hcore(k) + local block (+U if requested).
      3. Convergence: ΔE < tol AND ||r_new - r_old||_F < tol.

    ``dispersion=`` / ``dispersion_functional=`` add a system-only
    D3-BJ correction to the total energy after convergence.
    Requires the ``dftd3`` python bindings; raises
    ``NotImplementedError`` with a remediation hint when missing.
    """
    input_restart_supplied = initial_density_k is not None
    from .guess import _coerce_periodic_driver_guess

    requested_initial_guess = initial_guess
    initial_guess = _coerce_periodic_driver_guess(
        initial_guess,
        driver="run_periodic_rks_gpw_multi_k",
        supported=periodic_guess_capabilities('gpw', 'RKS', dim=getattr(system, "dim", 3), multi_k=True, transport='k'),
        restart_supplied=initial_density_k is not None,
    )
    if not quiet:
        warnings.warn(
            "run_periodic_rks_gpw_multi_k: M3e experimental -- "
            "pure-DFT multi-k GPW. RHF / hybrids / open-shell "
            "remain Γ-only.",
            category=GAPWExperimentalWarning,
            stacklevel=2,
        )

    plog = resolve_progress(progress)

    if system.dim != 3:
        raise ValueError(
            f"run_periodic_rks_gpw_multi_k: dim == 3 only; got dim={system.dim}"
        )

    func = _core.Functional(functional, 1)
    if (
        bool(getattr(func, "is_external", False))
        and float(system.charge) != 0.0
    ):
        raise NotImplementedError(
            "run_periodic_rks_gpw_multi_k: charged-cell external XC is not "
            "supported because the GPW Coulomb background and electron-count "
            "convention are not validated; use a neutral cell"
        )
    if bool(getattr(func, "is_external", False)):
        from .pbc_bipole_common import reject_bipole_ecp_options
        from .periodic_external_xc import (
            _require_zero_temperature_external_xc,
        )

        _require_zero_temperature_external_xc(
            smearing_temperature,
            where="run_periodic_rks_gpw_multi_k",
        )

        try:
            reject_bipole_ecp_options(
                object(),
                driver="run_periodic_rks_gpw_multi_k external XC",
                basis=basis,
                system=system,
            )
        except NotImplementedError as exc:
            raise NotImplementedError(
                "run_periodic_rks_gpw_multi_k: ECP-bearing bases are not "
                "implemented with external XC; use an all-electron basis"
            ) from exc
    if float(func.hf_exchange_fraction) != 0.0:
        raise NotImplementedError(
            f"run_periodic_rks_gpw_multi_k: hybrids "
            f"(hf_exchange_fraction = "
            f"{func.hf_exchange_fraction}) are not supported "
            f"on the multi-k path -- per-k K builders are not "
            f"wired. Use a pure functional like 'lda' or 'pbe'."
        )

    # k-mesh. Classify the cell before the GPW grid/cache build so the
    # multi-k density regime is fixed up front.
    kpoints = np.asarray(kmesh.kpoints, dtype=float)
    weights = np.asarray(kmesh.weights, dtype=float)
    n_k = kpoints.shape[0]
    # Compact crystals (AOs overlap across cells) need the Bloch real-space
    # density + per-k projection; vacuum-padded / molecular-limit cells fold
    # to the Gamma-summed smooth-grid density. Single-k (n_k == 1) always
    # uses the Gamma path (Bloch AO at Gamma == periodic Gamma AO).
    use_bloch_compact = n_k > 1 and not _multik_gpw_is_molecular_limit(system)

    # Set up grid.
    if grid is None:
        from .periodic_gapw_grid import make_grid as _make_grid

        if cutoff_ha is None:
            cutoff_ha = 300.0
        grid = _make_grid(np.asarray(system.lattice, dtype=float), cutoff_ha=cutoff_ha)

    # Molecular-limit cells cache the iteration-invariant Gamma AO table and
    # G mesh. Compact multi-k cells retain only the G mesh: their complex
    # Bloch AOs are streamed in bounded point batches, so neither a redundant
    # Gamma table nor n_k complete Bloch tables occupy memory.
    if use_bloch_compact:
        collocation_cache = GpwCollocationCache(
            chi=None, recip=grid.reciprocal_vectors()
        )
    else:
        collocation_cache = build_gpw_collocation_cache(basis, grid)

    n_elec = int(sum(int(a.Z) for a in system.unit_cell))
    if n_elec % 2 != 0:
        raise ValueError(
            f"run_periodic_rks_gpw_multi_k: cell has {n_elec} "
            f"electrons (odd); RKS needs even count."
        )
    n_occ = n_elec // 2
    n_basis = basis.nbasis

    # v0.12 R1: Fermi-Dirac smearing for the multi-k path. mu is found
    # by BZ-wide bisection (shared chemical potential across k); the
    # entropy term feeds the Mermin free energy.
    smearing_T = float(smearing_temperature)
    if smearing_T < 0.0:
        raise ValueError(
            "run_periodic_rks_gpw_multi_k: smearing_temperature must be "
            f">= 0; got {smearing_T}"
        )
    if smearing_T > 0.0:
        smearing_opts = _SmearingOptions(
            temperature=smearing_T,
            flavor=str(smearing_method),
        )
    else:
        smearing_opts = None

    if fock_mixing is None:
        # Multi-k GPW uses a modest Roothaan stabiliser on the
        # molecular-limit path; keep the single-k path bit-for-bit unchanged.
        fock_mixing_value = 0.30 if n_k > 1 else 0.0
    else:
        fock_mixing_value = float(fock_mixing)
    if not (0.0 <= fock_mixing_value < 1.0):
        raise ValueError(
            "run_periodic_rks_gpw_multi_k: fock_mixing must be in [0, 1); "
            f"got {fock_mixing_value}"
        )

    # One-electron lattice integrals (built once). S, T and V_ne MUST share
    # one R-set: the pencil H(k) C = S(k) C eps is only consistent when
    # every operator drops the same lattice tail (Pisani & Dovesi 1980,
    # doi:10.1002/qua.560170311, p. 510 -- see multik_v_ne_lattice_options
    # for the mechanism and the measured -18 Ha failure mode of mixing).
    lat_opts = multik_one_electron_lattice_options(basis, system)
    T_lat = _core.compute_kinetic_lattice(basis, system, lat_opts)
    S_lat = _core.compute_overlap_lattice(basis, system, lat_opts)
    from .periodic_v_ne import compute_nuclear_lattice_dispatch

    lat_opts_v = multik_v_ne_lattice_options(basis, system)
    V_lat = compute_nuclear_lattice_dispatch(basis, system, lat_opts_v)

    external_xc = None
    if bool(getattr(func, "is_external", False)):
        from .periodic_external_xc import PeriodicExternalXC

        external_xc = PeriodicExternalXC.prepare(
            basis,
            system,
            func,
            grid_options=external_xc_grid_options,
            image_radius_bohr=external_xc_image_radius_bohr,
            lattice_options=(
                external_xc_lattice_options
                if external_xc_lattice_options is not None
                else lat_opts
            ),
        )

    # Nuclear repulsion (system-level, k-independent).
    E_nn = float(_core.ewald_nuclear_repulsion(system, _core.EwaldOptions()))

    # v0.12 R1: DFT+U (Dudarev) on-site setup, multi-k pattern. Mirrors
    # run_pbc_bipole_rks: translate user-facing HubbardSites to the C++
    # _HubbardSiteCxx + parallel ao_groups list, then per iter call the
    # multi-k per-spin C++ kernel to get E_s and V_AO_s (k-independent),
    # add S(k) V_AO_s S(k) to F(k). E_total_+U = 2 x E_s for closed
    # shell.
    _dftu_sites = list(dft_plus_u_sites or ())
    _dftu_sites_cxx: list = []
    _dftu_ao_groups_list: list = []
    if _dftu_sites:
        from ._vibeqc_core import _HubbardSiteCxx as _HSC
        from .dft_plus_u import ao_group_indices as _dftu_ao_group_indices

        _ao_map = _dftu_ao_group_indices(basis)
        for _site in _dftu_sites:
            _key = (_site.atom_index, _site.l)
            if _key not in _ao_map:
                raise ValueError(
                    f"run_periodic_rks_gpw_multi_k: HubbardSite{_key} "
                    f"has no AOs in the basis. Available channels: "
                    f"{sorted(_ao_map.keys())}"
                )
            _dftu_sites_cxx.append(
                _HSC(
                    _site.atom_index,
                    _site.l,
                    _site.U_eff_hartree,
                )
            )
            _dftu_ao_groups_list.append(_ao_map[_key])
    e_dft_plus_u = 0.0

    # Pre-bloch-sum to T(k), S(k), V_ne(k) at every k.
    T_k: list = []
    S_k: list = []
    V_ne_k: list = []
    X_k: list = []  # canonical orthogonalizer per k
    if use_bloch_compact:
        from .periodic_rhf_multi_k_ewald import (
            _canonical_orthogonalizer_complex,
        )
    n_lindep_dropped = 0
    for ik in range(n_k):
        k = kpoints[ik]
        Tk = np.asarray(_core.bloch_sum(T_lat, k))
        Sk = np.asarray(_core.bloch_sum(S_lat, k))
        Vk = np.asarray(_core.bloch_sum(V_lat, k))
        # Hermitise the bloch sums against tiny noise.
        Sk = 0.5 * (Sk + Sk.conj().T)
        Tk = 0.5 * (Tk + Tk.conj().T)
        Vk = 0.5 * (Vk + Vk.conj().T)
        if use_bloch_compact:
            # Fail closed on an indefinite metric: S(k) is a Gram matrix,
            # so a negative eigenvalue is ALWAYS a truncation defect of the
            # overlap lattice sum, never a basis property. With the shared
            # basis-derived cutoff above this never fires; if it does, the
            # run must stop rather than absorb the defect into the
            # linear-dependence projection below (CLAUDE.md section 7).
            _assert_bloch_overlap_psd(Sk, k, lat_opts.cutoff_bohr)
            # Compact crystals with a molecular (diffuse) basis can still
            # leave S(k) near-singular (genuine near-linear-dependence,
            # smallest eigenvalues ~1e-8..1e-6 at def2-SVP). The plain
            # S^{-1/2} below floors those modes to 1e-12 and amplifies
            # them by ~1e6, which is the def2-SVP GPW runaway. The
            # canonical orthogonaliser instead PROJECTS OUT every
            # direction with S(k) eigenvalue below the threshold, giving a
            # rectangular X(k) (n_bf, n_kept) with X^+ S X = I. n_kept varies
            # per k; the aufbau / smearing occupation code already handles a
            # per-k band count, and D(k) = C f C^+ stays (n_bf, n_bf).
            X, n_kept = _canonical_orthogonalizer_complex(
                Sk, threshold=_COMPACT_GPW_LINDEP_THRESHOLD
            )
            n_lindep_dropped += Sk.shape[0] - n_kept
        else:
            # Canonical orthogonalizer X = S^{-1/2} (molecular-limit path,
            # unchanged: vacuum cells keep a well-conditioned S(k)).
            s_eigs, U = np.linalg.eigh(Sk)
            X = U @ np.diag(1.0 / np.sqrt(np.maximum(s_eigs, 1e-12))) @ U.conj().T
        T_k.append(Tk)
        S_k.append(Sk)
        V_ne_k.append(Vk)
        X_k.append(X)
    if use_bloch_compact and n_lindep_dropped and not quiet:
        warnings.warn(
            f"run_periodic_rks_gpw_multi_k: dropped {n_lindep_dropped} "
            f"linearly-dependent Bloch-AO direction(s) across the k-mesh "
            f"(S(k) eigenvalue < {_COMPACT_GPW_LINDEP_THRESHOLD:.0e}); "
            "compact crystal + molecular basis over-completeness.",
            category=GAPWExperimentalWarning,
            stacklevel=2,
        )

    # Bloch-AO evaluator inputs for the compact-crystal density build.  The
    # old path retained one complex (n_grid, n_basis) table per k-point here;
    # P16 therefore requested 464 GiB for 64 x 128^3 x 232 complex values.
    # Keep that iteration-saving cache only when every table fits under a fixed
    # 1 GiB cap. Larger jobs make separate bounded density/projection passes.
    # Meta-GGA spectral derivatives retain one whole-grid table at a time
    # because the FFT couples all points, but the overall peak stays under a
    # fixed bound instead of growing without limit with n_k.
    direct_cells = _core.direct_lattice_cells(system, 25.0)
    lattice_translations = np.array([c.r_cart for c in direct_cells], dtype=float)
    grid_pts = grid.cartesian_coords().reshape(-1, 3)
    chi_k_list: Optional[list[np.ndarray]] = None
    compact_cache_bytes = (
        n_k
        * len(grid_pts)
        * n_basis
        * np.dtype(np.complex128).itemsize
    )
    if (
        use_bloch_compact
        and compact_cache_bytes <= _COMPACT_BLOCH_AO_CACHE_BYTES
    ):
        chi_k_list = [
            bloch_ao_on_grid(
                basis, grid_pts, kpoints[ik], lattice_translations
            )
            for ik in range(n_k)
        ]

    def _build_D_k(C_list, occ_list):
        """Per-k Hermitian AO density matrices D(k) = C f_k C^H (occupations
        folded in; the closed-shell factor of 2 lives in occ_list). The BZ
        weights are applied by the density collocator, not here."""
        return [
            (C_list[ik] * occ_list[ik][None, :]) @ C_list[ik].conj().T
            for ik in range(n_k)
        ]

    def _collocate_multik(D_total_real, D_k_list):
        """Route the real-space density build by regime: Bloch sum for
        compact crystals, Gamma-folded collocation for the molecular limit."""
        if use_bloch_compact:
            if chi_k_list is not None:
                return collocate_bloch_density_on_grid(
                    chi_k_list, D_k_list, weights, grid
                )
            return _collocate_bloch_density_streaming(
                basis,
                grid_pts,
                kpoints,
                lattice_translations,
                D_k_list,
                weights,
                grid,
            )
        return collocate_density_on_grid(
            basis, D_total_real, grid, cache=collocation_cache
        )

    # Fock-mode guesses keep this driver's existing per-k occupation
    # semantics. Density-mode guesses are built once by the shared adapter and
    # injected as the g=0 block at every k point. Explicit restart blocks win.
    guess_fock_k = None
    if initial_density_k is None and initial_guess == _core.InitialGuess.PATOM:
        from .guess import patom_full_hf_focks_k
        guess_fock_k, _ = patom_full_hf_focks_k(
            system, basis, kmesh, [t + v for t, v in zip(T_k, V_ne_k)],
            S_k, n_occ, n_occ, lat_opts)
    if initial_density_k is None and guess_fock_k is None:
        from .guess import initial_density_closed_shell, periodic_fock_guess_k

        guess_fock_k = periodic_fock_guess_k(
            system,
            basis,
            kpoints,
            initial_guess,
            lattice_opts=lat_opts,
            kinetic_lattice=T_lat,
            overlap_lattice=S_lat,
        )
        if guess_fock_k is None:
            density_g0 = initial_density_closed_shell(
                system.unit_cell_molecule(),
                basis,
                n_occ,
                initial_guess,
                is_periodic=True,
                periodic_system=system,
                lattice_opts=lat_opts,
                overlap=S_k, weights=weights,
            )
            if density_g0 is not None:
                initial_density_k = [
                    np.asarray(density_g0, dtype=complex).copy()
                    for _ in range(n_k)
                ]

    # Build initial density from the selected guess Fock and accumulate rho.
    D_total = np.zeros((n_basis, n_basis), dtype=float)
    C_k: list = []
    e_k: list = []
    eps_per_k_init: list = []
    for ik in range(n_k):
        guess_fock = (
            guess_fock_k[ik]
            if guess_fock_k is not None
            else T_k[ik] + V_ne_k[ik]
        )
        eps, C_orth = np.linalg.eigh(
            X_k[ik].conj().T @ guess_fock @ X_k[ik]
        )
        C = X_k[ik] @ C_orth
        C_k.append(C)
        e_k.append(eps)
        eps_per_k_init.append(np.asarray(np.real(eps), dtype=float))
        # k-resolved 1-RDM contribution: D(k) = 2 . C_occ C_occ^+ . w_k
        # Sum on Γ to get a real-valued AO density matrix.
        Dk = 2.0 * (C[:, :n_occ] @ C[:, :n_occ].conj().T)
        D_total += weights[ik] * np.real(Dk)
    D_total = 0.5 * (D_total + D_total.T)

    # v0.12 R1: initial smearing pass -- only matters when smearing_T > 0
    # so the very first r already carries fractional occupations.
    if smearing_T > 0.0:
        sm_init = _apply_smearing(
            eps_per_k_init,
            weights=list(weights),
            n_electrons_per_cell=float(n_elec),
            n_occ_each=n_occ,
            smearing=smearing_opts,
        )
        occ_per_k_curr = [np.asarray(o, dtype=float) for o in sm_init.occupations_per_k]
        D_total = np.zeros((n_basis, n_basis), dtype=float)
        for ik in range(n_k):
            Dk = (C_k[ik] * occ_per_k_curr[ik][None, :]) @ C_k[ik].conj().T
            D_total += weights[ik] * np.real(Dk)
        D_total = 0.5 * (D_total + D_total.T)
    else:
        # Size occupations by each k-point's MO count. With the square
        # molecular-limit X this is n_basis; on the compact path the canonical
        # orthogonaliser can return fewer columns (n_kept) after dropping
        # linearly-dependent Bloch-AO directions.
        occ_per_k_curr = [
            np.concatenate(
                [2.0 * np.ones(n_occ), np.zeros(C_k[ik].shape[1] - n_occ)]
            )
            for ik in range(n_k)
        ]

    # Per-k density matrices consistent with (C_k, occ_per_k_curr). Used by
    # the compact Bloch density collocator; the molecular path ignores them
    # and uses the Gamma-summed D_total.
    D_k_curr = _build_D_k(C_k, occ_per_k_curr)
    if initial_density_k is not None:
        from .guess import normalize_density_k_guess
        D_k_curr = normalize_density_k_guess(initial_density_k, S_k, weights, n_elec)
        D_total = _weighted_real_density_from_k(D_k_curr, weights, n_basis)

    scf_trace: list = []
    E_prev = 0.0
    converged = False
    n_iter = 0
    fermi_level = 0.0
    entropy = 0.0
    F_prev_k: Optional[list[np.ndarray]] = None
    # Pulay commutator DIIS for the compact Bloch path (per-k Fock list).
    if use_bloch_compact:
        from .periodic_rhf_multi_k_ewald import _MultiKPulayDIIS

        compact_diis = _MultiKPulayDIIS(max_subspace=8)

    for it in range(1, max_iter + 1):
        # Real-space density: Bloch sum over the k-mesh for compact crystals,
        # Gamma-folded collocation in the molecular limit. All grid primitives
        # reuse per-SCF caches (Gamma chi table or per-k Bloch chi tables).
        rho_grid = _collocate_multik(D_total, D_k_curr)
        V_H = _core.solve_poisson_coulomb(rho_grid, grid.lattice_bohr)

        # Local Fock block (Hartree + XC). Two regimes:
        #   * compact Bloch: the whole local operator is a single real grid
        #     potential v_local(r) = V_H + v_xc (GGA gradient term folded in
        #     by parts), projected per k onto Bloch AOs -> V_loc_per_k[k].
        #   * molecular limit: J and V_xc are k-independent, projected once
        #     onto the periodic Gamma AOs (unchanged legacy path).
        # V_loc_per_k[k] is the per-k local block on the Bloch path; J_ao /
        # V_xc are the shared k-independent blocks on the molecular path.
        V_loc_per_k: list = [None] * n_k
        J_ao = None
        V_xc = None
        V_xc_external_k = None
        if external_xc is not None:
            e_xc_iter, V_xc_external_k = external_xc.build_kpoints(
                D_k_curr, kmesh
            )
        if use_bloch_compact:
            e_hartree = 0.5 * float(
                np.sum(rho_grid * V_H) * grid.voxel_volume_bohr3
            )
            # Meta-GGA: t from the per-k Bloch AO gradients (the Gamma t
            # builder drops the inter-cell Bloch phases on a compact cell,
            # exactly like the density collocation it mirrors).
            if external_xc is not None:
                # The FFT grid remains the GPW Hartree representation.  A
                # full-grid provider receives the complete AO density on its
                # separate atom-centred grid and returns V_xc(k) above.
                v_local_grid = np.asarray(V_H, dtype=float)
                v_tau_grid_c = None
            else:
                tau_bloch = None
                if func.kind == _core.XCKind.MGGA:
                    if chi_k_list is not None:
                        tau_bloch = compute_bloch_kinetic_energy_density(
                            chi_k_list,
                            D_k_curr,
                            weights,
                            kpoints,
                            grid_pts,
                            grid,
                            recip=collocation_cache.recip,
                        )
                    else:
                        tau_bloch = (
                            _compute_bloch_kinetic_energy_density_streaming(
                                basis,
                                D_k_curr,
                                weights,
                                kpoints,
                                grid_pts,
                                lattice_translations,
                                grid,
                                recip=collocation_cache.recip,
                            )
                        )
                (
                    e_xc_iter,
                    v_eff_xc,
                    v_tau_grid_c,
                ) = _xc_effective_potential_grid(
                    rho_grid,
                    grid,
                    func,
                    tau=tau_bloch,
                    cache=collocation_cache,
                )
                v_local_grid = np.asarray(V_H, dtype=float) + v_eff_xc
            for ik in range(n_k):
                if chi_k_list is not None:
                    chi_k = chi_k_list[ik]
                    V_loc_per_k[ik] = project_potential_to_bloch_ao(
                        chi_k, v_local_grid, grid
                    )
                    if v_tau_grid_c is not None:
                        V_loc_per_k[ik] = (
                            V_loc_per_k[ik]
                            + project_vtau_to_bloch_ao(
                                chi_k,
                                kpoints[ik],
                                v_tau_grid_c,
                                grid_pts,
                                grid,
                                recip=collocation_cache.recip,
                            )
                        )
                elif v_tau_grid_c is None:
                    V_loc_per_k[ik] = (
                        _project_potential_to_bloch_ao_streaming(
                            basis,
                            grid_pts,
                            kpoints[ik],
                            lattice_translations,
                            v_local_grid,
                            grid,
                        )
                    )
                else:
                    # Spectral meta-GGA gradients require the complete grid,
                    # but only this k-point's table is live at once. Reuse it
                    # for the local and v_tau projections before releasing it.
                    chi_k = bloch_ao_on_grid(
                        basis,
                        grid_pts,
                        kpoints[ik],
                        lattice_translations,
                    )
                    V_loc_per_k[ik] = project_potential_to_bloch_ao(
                        chi_k, v_local_grid, grid
                    )
                    # t Fock term per k: 1/2 int v_tau gradchi_k* . gradchi_k
                    # (see project_vtau_to_bloch_ao).
                    V_loc_per_k[ik] = V_loc_per_k[ik] + project_vtau_to_bloch_ao(
                        chi_k,
                        kpoints[ik],
                        v_tau_grid_c,
                        grid_pts,
                        grid,
                        recip=collocation_cache.recip,
                    )
                    del chi_k
        else:
            J_ao = project_potential_to_ao(basis, V_H, grid, cache=collocation_cache)
            e_hartree = 0.5 * float(np.einsum("ij,ij->", D_total, J_ao))

            if external_xc is None:
                # XC. basis + density_matrix let meta-GGA functionals build
                # t(r) from the Gamma-summed density (molecular-limit cells
                # only -- the compact Bloch branch gates meta-GGA upstream).
                (
                    e_xc_iter,
                    v_xc_grid,
                    v_sigma_grid,
                    grad_rho,
                    v_tau_grid,
                ) = _evaluate_xc_on_grid(
                    rho_grid,
                    grid,
                    func,
                    basis=basis,
                    density_matrix=D_total,
                    cache=collocation_cache,
                )
                V_xc = _project_vxc_to_ao(
                    basis,
                    v_xc_grid,
                    grid,
                    v_sigma_grid=v_sigma_grid,
                    grad_rho=grad_rho,
                    cache=collocation_cache,
                )
                if v_tau_grid is not None:
                    # t Fock term: 1/2 ∫ v_tau gradchi.gradchi (see
                    # _project_vtau_to_ao).
                    V_tau = _project_vtau_to_ao(
                        basis, v_tau_grid, grid, cache=collocation_cache
                    )
                    V_xc = V_xc + V_tau

        # v0.12 R1: DFT+U per-spin per-k Fock contribution. Built from
        # the *current* per-k density matrices (constructed from the
        # previous iter's C_k + occ_per_k_curr -- i.e. the densities
        # that produced D_total). Mirrors run_pbc_bipole_rks: P_s(k)
        # = 1/2 P_total(k); V_AO_s is k-independent; per-k Fock gets
        # S(k) V_AO_s S(k).
        e_dft_plus_u = 0.0
        V_U_per_k = [None] * n_k
        if _dftu_sites_cxx:
            from ._vibeqc_core import (
                _compute_dft_plus_u_multi_k_per_spin_cxx,
            )

            P_sigma_k: list = []
            for ik in range(n_k):
                # Per-k 1-RDM (alpha = beta for closed shell -> 1/2 total).
                Pk = (C_k[ik] * occ_per_k_curr[ik][None, :]) @ C_k[ik].conj().T
                P_sigma = 0.5 * Pk
                P_sigma = 0.5 * (P_sigma + P_sigma.conj().T)
                P_sigma_k.append(np.asarray(P_sigma, dtype=complex))
            S_k_cplx = [np.asarray(s, dtype=complex) for s in S_k]
            E_sigma, V_AO = _compute_dft_plus_u_multi_k_per_spin_cxx(
                _dftu_sites_cxx,
                _dftu_ao_groups_list,
                S_k_cplx,
                P_sigma_k,
                list(weights),
            )
            e_dft_plus_u = 2.0 * float(E_sigma)
            V_AO_c = np.asarray(V_AO, dtype=complex)
            for ik in range(n_k):
                V_U_per_k[ik] = S_k[ik] @ V_AO_c @ S_k[ik]

        # Per-k Fock + diagonalisation.
        new_C_k = []
        new_e_k = []
        eps_per_k_new: list = []

        # Raw per-k Fock (Hcore + local block + optional +U), Hermitised.
        F_raw_k: list[np.ndarray] = []
        for ik in range(n_k):
            Hcore_k = T_k[ik] + V_ne_k[ik]
            if use_bloch_compact:
                Fk = Hcore_k + V_loc_per_k[ik]
            else:
                Fk = Hcore_k + J_ao
                if V_xc is not None:
                    Fk = Fk + V_xc
            if V_xc_external_k is not None:
                Fk = Fk + V_xc_external_k[ik]
            if V_U_per_k[ik] is not None:
                Fk = Fk + V_U_per_k[ik]
            Fk = 0.5 * (Fk + Fk.conj().T)
            F_raw_k.append(np.asarray(Fk, dtype=complex))

        # Convergence acceleration. Compact crystals have small gaps and a
        # tight all-electron core on the smooth grid, so plain linear Fock
        # mixing converges only geometrically (ratio ~0.9). The compact path
        # runs Pulay commutator DIIS entirely in the ORTHONORMAL kept
        # subspace: F~(k) = X(k)^+ F(k) X(k) with the S = I commutator error
        #     e(k) = F~(k) D~(k) - D~(k) F~(k),  D~(k) = X^+ S D(k) S X.
        # Doing DIIS in this basis (rather than the raw-AO F D S - S D F) is
        # what makes it robust to the linear-dependent directions X projects
        # out: the error carries no null-space pollution, so the Pulay B
        # matrix stays well-conditioned and a diffuse-basis compact crystal
        # (def2-SVP Si) converges instead of charge-sloshing. The
        # molecular-limit path keeps the historical linear Fock mixing
        # bit-for-bit.
        diis_err = 0.0
        if use_bloch_compact:
            Forth_list: list[np.ndarray] = []
            err_list: list[np.ndarray] = []
            for ik in range(n_k):
                X = X_k[ik]
                Sk = S_k[ik]
                Forth = X.conj().T @ F_raw_k[ik] @ X
                Forth = 0.5 * (Forth + Forth.conj().T)
                D_orth = X.conj().T @ (Sk @ D_k_curr[ik] @ Sk) @ X
                D_orth = 0.5 * (D_orth + D_orth.conj().T)
                err = Forth @ D_orth - D_orth @ Forth
                Forth_list.append(Forth)
                err_list.append(err)
                diis_err = max(
                    diis_err,
                    float(np.max(np.abs(err))) if err.size else 0.0,
                )
            Forth_use = compact_diis.extrapolate(
                Forth_list, err_list, list(weights)
            )
            for ik in range(n_k):
                Fo = 0.5 * (Forth_use[ik] + Forth_use[ik].conj().T)
                eps, C_orth = np.linalg.eigh(Fo)
                C = X_k[ik] @ C_orth
                new_C_k.append(C)
                new_e_k.append(eps)
                eps_per_k_new.append(np.asarray(np.real(eps), dtype=float))
        else:
            F_use_k = []
            for ik in range(n_k):
                Fk = F_raw_k[ik]
                if fock_mixing_value != 0.0 and F_prev_k is not None:
                    Fk = (
                        (1.0 - fock_mixing_value) * Fk
                        + fock_mixing_value * F_prev_k[ik]
                    )
                    Fk = 0.5 * (Fk + Fk.conj().T)
                F_use_k.append(np.asarray(Fk, dtype=complex))
            if fock_mixing_value != 0.0:
                F_prev_k = [F.copy() for F in F_use_k]

            for ik in range(n_k):
                Fk = F_use_k[ik]
                eps, C_orth = np.linalg.eigh(X_k[ik].conj().T @ Fk @ X_k[ik])
                C = X_k[ik] @ C_orth
                new_C_k.append(C)
                new_e_k.append(eps)
                eps_per_k_new.append(np.asarray(np.real(eps), dtype=float))

        # v0.12 R1: smear -> fractional occupations. At T = 0 this
        # collapses to the historical hard-Aufbau path bit-for-bit.
        sm_iter = _apply_smearing(
            eps_per_k_new,
            weights=list(weights),
            n_electrons_per_cell=float(n_elec),
            n_occ_each=n_occ,
            smearing=smearing_opts,
        )
        occ_per_k_new = [np.asarray(o, dtype=float) for o in sm_iter.occupations_per_k]
        fermi_level = float(sm_iter.mu)
        entropy = float(sm_iter.entropy)

        # Build D from fractional occupations:
        #   D(k) = S_i f_i,k . C_i,k . C_i,k+
        # The per-k density matrices feed both the Gamma-summed D_new and the
        # compact Bloch density collocation on the next iteration.
        D_k_new = _build_D_k(new_C_k, occ_per_k_new)
        D_new = np.zeros_like(D_total)
        for ik in range(n_k):
            D_new += weights[ik] * np.real(D_k_new[ik])
        D_new = 0.5 * (D_new + D_new.T)

        # Total energy (RKS unified formula):
        # E_total = S_k w_k . tr(D(k).Hcore(k)) + 0.5 . tr(D.J) +
        #           E_xc + E_nn
        e_hcore = 0.0
        for ik in range(n_k):
            Hcore_k = T_k[ik] + V_ne_k[ik]
            e_hcore += weights[ik] * float(
                np.real(np.einsum("ij,ji->", D_k_curr[ik], Hcore_k))
            )
        E = e_hcore + e_hartree + e_xc_iter + E_nn + e_dft_plus_u
        occ_per_k_curr = occ_per_k_new

        dE = E - E_prev
        dD = float(np.linalg.norm(D_new - D_total))
        n_iter = it
        scf_trace.append(
            {
                "iter": it,
                "energy": float(E),
                "delta_e": float(dE),
                "grad_norm": float(dD),
                "e_xc": float(e_xc_iter),
                "fock_mixing": float(fock_mixing_value),
                "diis_error": float(diis_err),
            }
        )
        # Per-iteration progress hook -> live QVF checkpoint cadence when the
        # runner passes a checkpoint-wrapped ``plog`` (see run_periodic_rhf_gpw).
        plog.iteration(it, energy=float(E), dE=float(dE), grad=float(dD))
        if abs(dE) < conv_tol_energy and dD < conv_tol_density and it > 1:
            converged = True
            D_total = D_new
            D_k_curr = D_k_new
            C_k = new_C_k
            e_k = new_e_k
            break
        D_total = D_new
        D_k_curr = D_k_new
        C_k = new_C_k
        e_k = new_e_k
        E_prev = E

    # Build a breakdown at the returned density.  The last SCF iteration
    # evaluates its Hartree, XC, and +U terms at the input density before
    # accepting ``D_k_new``.  Reusing those values here would combine the
    # returned one-electron density with stale nonlinear energy terms (most
    # visibly for a deliberately non-converged return).  Re-evaluate every
    # density-functional component from the accepted per-k density.
    #
    # The kinetic + nuclear-attraction terms MUST be accumulated from the
    # per-k Bloch-summed Hcore(k) -- the same sum that produced the reported
    # total `E` (e_hcore in the loop) -- NOT from a single Γ (k=0)
    # re-evaluation. tr(D_total . T^Γ) drops the per-k Bloch phases and
    # disagrees with S_k w_k tr(D(k).T(k)) by tens of mHa on a dispersive
    # mesh (e.g. -60.8 mHa for [2,1,1] H2/STO-3G in a 6-bohr cell): the
    # v0.12.x "GPW multi-k breakdown != total" audit finding.  The Hartree
    # and XC terms are rebuilt immediately below; e_hf_exchange is 0 on the
    # pure-DFT multi-k path (hybrids are refused upstream).
    e_kinetic_mk = 0.0
    e_ne_mk = 0.0
    for ik in range(n_k):
        Dk = np.asarray(D_k_curr[ik], dtype=complex)
        e_kinetic_mk += weights[ik] * float(
            np.real(np.einsum("ij,ji->", Dk, T_k[ik]))
        )
        e_ne_mk += weights[ik] * float(
            np.real(np.einsum("ij,ji->", Dk, V_ne_k[ik]))
        )

    rho_grid_final = _collocate_multik(D_total, D_k_curr)
    V_H_final = _core.solve_poisson_coulomb(
        rho_grid_final, grid.lattice_bohr
    )
    J_ao_final = None
    if use_bloch_compact:
        e_hartree_final = 0.5 * float(
            np.sum(rho_grid_final * V_H_final) * grid.voxel_volume_bohr3
        )
    else:
        J_ao_final = project_potential_to_ao(
            basis,
            V_H_final,
            grid,
            cache=collocation_cache,
        )
        e_hartree_final = 0.5 * float(
            np.einsum("ij,ij->", D_total, J_ao_final)
        )

    V_xc_final_k = None
    terminal_fock_k: tuple = ()
    terminal_overlap_k = tuple(np.asarray(s, dtype=complex).copy() for s in S_k)
    terminal_hcore_k: tuple = ()
    if external_xc is not None:
        e_xc_final, V_xc_final_k = external_xc.build_kpoints(
            D_k_curr, kmesh
        )
    elif use_bloch_compact:
        tau_bloch_final = None
        if func.kind == _core.XCKind.MGGA:
            if chi_k_list is not None:
                tau_bloch_final = compute_bloch_kinetic_energy_density(
                    chi_k_list,
                    D_k_curr,
                    weights,
                    kpoints,
                    grid_pts,
                    grid,
                    recip=collocation_cache.recip,
                )
            else:
                tau_bloch_final = (
                    _compute_bloch_kinetic_energy_density_streaming(
                        basis,
                        D_k_curr,
                        weights,
                        kpoints,
                        grid_pts,
                        lattice_translations,
                        grid,
                        recip=collocation_cache.recip,
                    )
                )
        e_xc_final, _, _ = _xc_effective_potential_grid(
            rho_grid_final,
            grid,
            func,
            tau=tau_bloch_final,
            cache=collocation_cache,
        )
    else:
        e_xc_final, *_ = _evaluate_xc_on_grid(
            rho_grid_final,
            grid,
            func,
            basis=basis,
            density_matrix=D_total,
            cache=collocation_cache,
        )
    e_xc_final = float(e_xc_final)

    e_dft_plus_u_final = 0.0
    V_AO_final = None
    if _dftu_sites_cxx:
        from ._vibeqc_core import (
            _compute_dft_plus_u_multi_k_per_spin_cxx,
        )

        P_sigma_k_final = [
            0.5 * np.asarray(Dk, dtype=complex) for Dk in D_k_curr
        ]
        E_sigma_final, V_AO_final_raw = (
            _compute_dft_plus_u_multi_k_per_spin_cxx(
                _dftu_sites_cxx,
                _dftu_ao_groups_list,
                [np.asarray(s, dtype=complex) for s in S_k],
                P_sigma_k_final,
                list(weights),
            )
        )
        e_dft_plus_u_final = 2.0 * float(E_sigma_final)
        V_AO_final = np.asarray(V_AO_final_raw, dtype=complex)

    if external_xc is not None:
        # Return canonical orbitals of the physical KS operator evaluated at
        # the same accepted density as the final nonlinear energy.  The last
        # loop orbitals may diagonalise a DIIS/Fock-mixed operator belonging
        # to the preceding density generation.
        if use_bloch_compact:
            J_final_k = []
            for ik in range(n_k):
                if chi_k_list is not None:
                    Jk = project_potential_to_bloch_ao(
                        chi_k_list[ik], V_H_final, grid
                    )
                else:
                    Jk = _project_potential_to_bloch_ao_streaming(
                        basis,
                        grid_pts,
                        kpoints[ik],
                        lattice_translations,
                        V_H_final,
                        grid,
                    )
                J_final_k.append(np.asarray(Jk, dtype=complex))
        else:
            assert J_ao_final is not None
            J_final_k = [
                np.asarray(J_ao_final, dtype=complex) for _ in range(n_k)
            ]

        assert V_xc_final_k is not None
        C_final_k = []
        e_final_k = []
        F_final_k = []
        for ik in range(n_k):
            Fk = (
                np.asarray(T_k[ik] + V_ne_k[ik], dtype=complex)
                + J_final_k[ik]
                + np.asarray(V_xc_final_k[ik], dtype=complex)
            )
            if V_AO_final is not None:
                Fk = Fk + S_k[ik] @ V_AO_final @ S_k[ik]
            Fk = 0.5 * (Fk + Fk.conj().T)
            F_final_k.append(Fk)
            eps, C_orth = np.linalg.eigh(
                X_k[ik].conj().T @ Fk @ X_k[ik]
            )
            C_final_k.append(X_k[ik] @ C_orth)
            e_final_k.append(np.asarray(np.real(eps), dtype=float))

        final_smearing = _apply_smearing(
            e_final_k,
            weights=list(weights),
            n_electrons_per_cell=float(n_elec),
            n_occ_each=n_occ,
            smearing=smearing_opts,
        )
        C_k = C_final_k
        e_k = e_final_k
        occ_per_k_curr = [
            np.asarray(o, dtype=float)
            for o in final_smearing.occupations_per_k
        ]
        fermi_level = float(final_smearing.mu)
        entropy = float(final_smearing.entropy)
        terminal_fock_k = tuple(
            np.asarray(Fk, dtype=complex) for Fk in F_final_k
        )
        terminal_overlap_k = tuple(
            np.asarray(Sk, dtype=complex) for Sk in S_k
        )
        terminal_hcore_k = tuple(
            np.asarray(T_k[ik] + V_ne_k[ik], dtype=complex)
            for ik in range(n_k)
        )

    E = (
        e_kinetic_mk
        + e_ne_mk
        + e_hartree_final
        + e_xc_final
        + E_nn
    )
    breakdown = GpwEnergyBreakdown(
        e_kinetic=e_kinetic_mk,
        e_nuclear_attraction=e_ne_mk,
        e_hartree=e_hartree_final,
        e_hf_exchange=0.0,
        e_nuclear_repulsion=E_nn,
        e_total=E,
        grid=grid,
        e_xc=e_xc_final,
        functional=functional,
    )

    # v0.12-prep: D3-BJ periodic dispersion add-on (system-only).
    e_disp = _compute_d3_correction(
        system,
        dispersion,
        dispersion_functional,
    )
    if e_disp != 0.0:
        from dataclasses import replace as _dc_replace

        breakdown = _dc_replace(
            breakdown,
            e_total=breakdown.e_total + e_disp,
            e_dispersion=e_disp,
        )
        E = E + e_disp

    # v0.12 R1: DFT+U bookkeeping into breakdown.  The final value above was
    # recomputed from the accepted per-k density, not the preceding iterate.
    if _dftu_sites_cxx:
        from dataclasses import replace as _dc_replace

        E = E + e_dft_plus_u_final
        breakdown = _dc_replace(
            breakdown,
            e_total=breakdown.e_total + e_dft_plus_u_final,
            e_dft_plus_u=e_dft_plus_u_final,
        )

    return GpwMultiKScfResult(
               restart_kpoints=np.asarray(kmesh.kpoints).copy(), restart_weights=np.asarray(kmesh.weights).copy(),
               guess_selection=periodic_result_selection(system, requested_initial_guess, restarted=input_restart_supplied),
               restart_basis=basis, restart_lattice=np.asarray(system.lattice).copy(),
        energy=E,
        breakdown=breakdown,
        mo_coeffs_k=tuple(C_k),
        mo_energies_k=tuple(e_k),
        density=D_total,
        density_k=tuple(np.asarray(Dk, dtype=complex) for Dk in D_k_curr),
        converged=converged,
        n_iter=n_iter,
        grid=grid,
        kmesh=kmesh,
        fock_k=terminal_fock_k,
        overlap_k=terminal_overlap_k,
        hcore_k=terminal_hcore_k,
        scf_trace=tuple(scf_trace),
        e_dispersion=e_disp,
        e_dft_plus_u=e_dft_plus_u_final,
        fock_mixing=fock_mixing_value,
        smearing_temperature=smearing_T,
        smearing_entropy=entropy,
        fermi_level=fermi_level,
        occupations_k=tuple(tuple(o.tolist()) for o in occ_per_k_curr),
    )


def _eigh_safe(M: np.ndarray, use_dav: bool = False, dav_opts=None):
    """Symmetric eigendecomposition with a fallback to scipy when
    numpy refuses. ``M`` is symmetrised before the solve so
    rounding-error asymmetry doesn't break the eigh.

    When ``use_dav`` is True and ``dav_opts`` is provided, uses the
    Davidson iterative solver for the leading eigenvalues/vectors
    instead of a dense diagonalisation."""
    M_sym = 0.5 * (M + M.T)
    if use_dav and dav_opts is not None:
        from vibeqc._vibeqc_core import davidson_solve

        if dav_opts.n_eig == 0:
            dav_opts.n_eig = M_sym.shape[0]
        if dav_opts.guess_vectors is not None:
            pass  # already set from previous iteration
        dres = davidson_solve(M_sym, dav_opts)
        if not dres.converged:
            raise RuntimeError(f"Davidson did not converge after {dres.n_iter} iters")
        dav_opts.guess_vectors = dres.eigenvectors
        return dres.eigenvalues, dres.eigenvectors
    try:
        return np.linalg.eigh(M_sym)
    except np.linalg.LinAlgError:
        import scipy.linalg as sla

        return sla.eigh(M_sym)


def _make_diis(use_diis: bool, diis_subspace_size: int):
    """The canonical Pulay accelerator for the GAPW / GPW / RSGAPW drivers.

    Returns ``vibeqc::DIIS`` (``cpp/src/diis.cpp``), the one DIIS shared with
    every other vibe-qc SCF backend, or ``None`` when the caller has turned
    extrapolation off.

    A subspace below two iterates is also ``None``: the Pulay solve needs two
    points, so ``DIIS`` raises on ``max_subspace < 2``. The private numpy
    history this replaces instead degenerated silently to ``F`` unchanged
    there, and preserving that no-op keeps a previously legal (if useless)
    ``diis_subspace_size=1`` from becoming a hard error.
    """
    if not use_diis or int(diis_subspace_size) < 2:
        return None
    return _core.DIIS(int(diis_subspace_size))
