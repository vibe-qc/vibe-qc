"""High-level entry point for embedded-surface calculations.

Ties the four 3D embedding modules -- region partition, substrate GF,
surface-projected S_emb, and contour density -- into a single call.

Usage::

    from vibeqc.periodic_embedding.runner import run_embedded_surface

    result = run_embedded_surface(
        slab_system, basis,
        tags=[0, 0, 0, 0, 1, 1],          # 4 sub + 2 region I
        surface_k_mesh=(4, 4),
    )

The result carries the embedded region-I density, occupation, band
energy, and the per-k linearized embedding potentials for downstream
Layer-B work.

Status: one-shot (non-self-consistent) density from Hcore + S_emb.
The Hartree + XC self-consistency loop and the ASE calculator surface
are follow-ups.  Gated as experimental until validated against a
thick-slab GDF limit (Sec.7).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .._vibeqc_core import (
    BasisSet,
    LatticeSumOptions,
    PeriodicSystem,
)
from .contour import EnergyContour
from .embedding_potential import LinearizedEmbeddingPotential
from .region import RegionPartition
from .scf2step import (
    build_sigma_lin_per_k,
    build_substrate_scf_potential,
    compute_region_i_density_one_shot,
    compute_region_i_density_scf,
    compute_region_i_density_scf_hf,
    compute_region_i_density_scf_uhf,
)
from .substrate_gf import default_surface_k_mesh


class EmbeddedSurfaceExperimentalWarning(UserWarning):
    """Emitted when the Green's-function surface-embedding driver is used.

    The embedding subpackage (Inglesfield surface-normal embedding +
    Sancho-Rubio substrate Green function + Ishida energy linearization +
    Lloyd energetics) is **experimental**.  Its analytic 1D foundation and
    the 3D Gaussian pipeline are validated and regression-tested, but the
    method has not yet been shown to reproduce a thick-slab GDF reference
    to tight tolerance (CLAUDE.md Sec.7); treat absolute energies and
    densities as qualitative until that parity lands.  Filter via
    ``warnings.filterwarnings("ignore",
    category=EmbeddedSurfaceExperimentalWarning)`` for quiet output.
    """


def _warn_experimental() -> None:
    warnings.warn(
        "Green's-function surface embedding is experimental and not yet "
        "validated to thick-slab GDF parity; treat results as qualitative. "
        "See docs/user_guide/surface_embedding.md.",
        EmbeddedSurfaceExperimentalWarning,
        stacklevel=3,
    )


@dataclass
class EmbeddedSurfaceResult:
    """Complete result of an embedded-surface calculation (Layer A).

    Attributes
    ----------
    region
        The region partition used.
    density
        Region-I density matrix (full AO basis, zero-padded outside I).
    density_local
        Region-I density matrix restricted to ``region.i_ao``.
    occupation
        Integrated occupied charge in region I.
    band_energy
        Band-structure energy from the contour.
    contour
        The complex-energy contour used.
    sigma_lin_per_k
        Per-k linearized embedding potentials (for downstream reuse).
    e_fermi
        Fermi level used for the contour (Ha).
    e_bottom
        Band-bottom estimate used for the contour (Ha).
    per_k_trace
        Trace of G(z) per (k, node) for LDOS diagnostics.
    """

    region: RegionPartition
    density: NDArray[np.float64]
    density_local: NDArray[np.float64]
    occupation: float
    band_energy: float
    contour: EnergyContour
    sigma_lin_per_k: Dict[int, LinearizedEmbeddingPotential]
    e_fermi: float
    e_bottom: float
    per_k_trace: NDArray[np.float64]

    @property
    def n_k(self) -> int:
        return len(self.sigma_lin_per_k)

    @property
    def n_i(self) -> int:
        return self.region.n_i


def _estimate_band_edges(
    system: PeriodicSystem,
    basis: BasisSet,
    region: RegionPartition,
    *,
    lat_opts: Optional[LatticeSumOptions] = None,
    e_fermi: Optional[float] = None,
    e_bottom: Optional[float] = None,
    e_bottom_margin: float = 0.5,
) -> Tuple[float, float]:
    """Estimate the Fermi level and band bottom from the substrate's
    S⁻¹H eigenvalues at Γ.

    If ``e_fermi`` or ``e_bottom`` are provided, they are used as-is.
    Otherwise they are estimated from the substrate's core-Hamiltonian
    spectrum.

    Returns ``(e_fermi, e_bottom)`` in Ha.
    """
    if e_fermi is not None and e_bottom is not None:
        return e_fermi, e_bottom

    # Build H(k=0) and S(k=0) for the substrate atoms only.
    from .substrate_gf import _build_hk_sk

    lat_opts = lat_opts or _default_lat_opts()
    H_full, S_full = _build_hk_sk(system, basis, np.zeros(3), lat_opts)

    # Restrict to substrate AOs for the eigenvalue estimate.
    sub_ao = np.setdiff1d(np.arange(region.n_ao_total), region.i_ao).astype(int)
    if len(sub_ao) == 0:
        # No substrate -- fall back to region I.
        sub_ao = region.i_ao

    H_sub = H_full[np.ix_(sub_ao, sub_ao)]
    S_sub = S_full[np.ix_(sub_ao, sub_ao)]

    # Generalized eigenvalue problem: H.C = S.C.e
    try:
        from scipy.linalg import eigh as eigh_gen

        evals = eigh_gen(
            np.real(H_sub) if np.allclose(H_sub.imag, 0) else H_sub,
            np.real(S_sub) if np.allclose(S_sub.imag, 0) else S_sub,
            eigvals_only=True,
        )
    except (np.linalg.LinAlgError, ValueError):
        # Fallback: use region I's on-site.
        H_i = H_full[np.ix_(region.i_ao, region.i_ao)]
        S_i = S_full[np.ix_(region.i_ao, region.i_ao)]
        Ht = np.linalg.solve(np.linalg.cholesky(np.real(S_i)), np.real(H_i))
        Ht = 0.5 * (Ht + Ht.T)
        evals = np.linalg.eigvalsh(Ht)

    evals = np.sort(np.asarray(evals, dtype=float))

    # Estimate band bottom (lowest eigenvalue with margin).
    est_bottom = float(evals[0]) - float(e_bottom_margin)

    # Estimate Fermi level: for a semiconductor with integer occupation,
    # take the HOMO eigenvalue. For a metal, use the middle of the band.
    # Simple heuristic: half-filling of the substrate.
    n_elec_sub = sum(
        a.Z for i, a in enumerate(system.unit_cell) if i not in region.i_atoms
    )
    # Each state holds 2 electrons (spin-degenerate at this level).
    n_occ = max(1, (n_elec_sub + 1) // 2)
    n_occ = min(n_occ, len(evals) - 1)
    est_fermi = float(evals[n_occ - 1])

    if e_fermi is not None:
        est_fermi = e_fermi
    if e_bottom is not None:
        est_bottom = e_bottom

    return est_fermi, est_bottom


def _default_lat_opts() -> LatticeSumOptions:
    opts = LatticeSumOptions()
    opts.cutoff_bohr = 30.0
    opts.nuclear_cutoff_bohr = 30.0
    return opts


def run_embedded_surface(
    system: PeriodicSystem,
    basis: BasisSet,
    tags: Sequence[int],
    *,
    surface_k_mesh: Tuple[int, int] = (2, 2),
    contour_n_nodes: int = 40,
    e_fermi: Optional[float] = None,
    e_bottom: Optional[float] = None,
    e_bottom_margin: float = 0.5,
    z0: Optional[complex] = None,
    lat_opts: Optional[LatticeSumOptions] = None,
    layer_tol: float = 0.5,
    scf_method: str = "one_shot",
    scf_mixing: float = 0.3,
    scf_max_iter: int = 30,
    sigma_scf: bool = False,
    sigma_scf_aux: Optional[str] = None,
    label: str = "embedded-surface",
) -> EmbeddedSurfaceResult:
    """Run a Layer-A embedded-surface calculation.

    Builds the embedding potential S_emb(k, z) from the clean substrate,
    sets up a complex-energy contour, and computes the region-I density
    via k-space contour integration.  This is the one-shot (Hcore + S_emb)
    path; the Hartree + XC self-consistency is a follow-up.

    Parameters
    ----------
    system
        Clean slab system (no adsorbate).  ``system.dim`` should be 2 or 3.
    basis
        Orbital basis set (must match the atoms in ``system``).
    tags
        Per-atom integer tags: 0 = substrate, 1 = region I.
        Must have one entry per atom in ``system.unit_cell``.
    surface_k_mesh
        In-plane Monkhorst-Pack subdivisions ``(nx, ny)``.  Default (2,2).
    contour_n_nodes
        Number of Gauss-Legendre nodes on the semicircular contour.
    e_fermi, e_bottom
        Fermi level and band-bottom estimate in Ha.  If not provided,
        estimated from the substrate's S⁻¹H eigenvalues at Γ.
    e_bottom_margin
        Extra margin below the lowest eigenvalue for ``e_bottom`` (Ha).
    z0
        Reference complex energy for Ishida linearization.  Default:
        ``(e_bottom + e_fermi)/2 + 0.5j``, the contour centre.
    lat_opts
        Lattice-sum options.  Default: cutoff 30 bohr, direct truncated.
    layer_tol
        z-tolerance for layer grouping (bohr).
    scf_method
        Density method: ``"one_shot"`` (Hcore + S_emb only),
        ``"hartree"`` (diagonal Hartree SCF), ``"hf"`` (full
        Hartree-Fock with GDF J and K), ``"uhf"`` (spin-polarized UHF).
        Default: ``"one_shot"``.
    scf_mixing
        Linear mixing fraction for SCF.
    scf_max_iter
        Maximum SCF iterations.
    sigma_scf
        If ``True``, build the embedding potential S_emb from the
        **SCF-converged** substrate Fock (Ishida two-step, step 1) rather
        than from the bare Hcore.  A periodic GDF RHF is run once on the
        clean slab to obtain the converged mean-field potential
        (:func:`~vibeqc.periodic_embedding.scf2step.build_substrate_scf_potential`),
        which then enters S_emb at every contour node.  Exact for a Γ-only
        surface mesh; the k-independent mean-field approximation otherwise.
        Default ``False`` (Hcore S_emb, the original behaviour).
    sigma_scf_aux
        Optional density-fitting auxiliary basis for the substrate SCF
        potential (defaults to the orbital basis's canonical aux).
    label
        Human-readable tag for log messages (future).

    Returns
    -------
    EmbeddedSurfaceResult
        Density, occupation, band energy, contour, and per-k S_emb
        for possible Layer-B reuse.
    """
    _warn_experimental()
    lat_opts = lat_opts or _default_lat_opts()

    # --- 1. Region partition -------------------------------------------
    region = RegionPartition.from_tags(system, basis, tags)

    # --- 2a. Substrate mean field (Ishida step 1, optional) -----------
    # Built first because V_eff shifts the substrate spectrum, and the
    # contour window must enclose the *shifted* (SCF) bands -- estimating
    # it from the Hcore spectrum would silently miss them (zero occupation).
    v_eff_substrate: Optional[NDArray[np.float64]] = None
    substrate_mf = None
    if sigma_scf:
        substrate_mf = build_substrate_scf_potential(
            system,
            basis,
            lat_opts=lat_opts,
            aux_name=sigma_scf_aux,
        )
        v_eff_substrate = substrate_mf.v_eff

    # --- 2b. Fermi-level / band-edge estimation -----------------------
    if sigma_scf and (e_fermi is None or e_bottom is None):
        # SCF-level S_emb: take the window from the SCF substrate spectrum.
        est_fermi = substrate_mf.e_homo
        est_bottom = float(substrate_mf.mo_energies[0]) - float(e_bottom_margin)
        ef = e_fermi if e_fermi is not None else est_fermi
        eb = e_bottom if e_bottom is not None else est_bottom
    else:
        ef, eb = _estimate_band_edges(
            system,
            basis,
            region,
            lat_opts=lat_opts,
            e_fermi=e_fermi,
            e_bottom=e_bottom,
            e_bottom_margin=e_bottom_margin,
        )

    # --- 3. Contour ----------------------------------------------------
    contour = EnergyContour.semicircle(eb, ef, n_nodes=contour_n_nodes)

    # --- 4. Surface k-mesh ---------------------------------------------
    kmesh = default_surface_k_mesh(system, mesh=surface_k_mesh)

    # --- 5. Linearized S_emb at each k-point --------------------------
    if z0 is None:
        z0 = complex(0.5 * (eb + ef), 0.5)
    sigma_lin = build_sigma_lin_per_k(
        system,
        basis,
        region,
        kmesh,
        z0,
        lat_opts=lat_opts,
        layer_tol=layer_tol,
        v_eff_full=v_eff_substrate,
    )

    # --- 6. Density ---------------------------------------------------
    if scf_method == "one_shot":
        density_result = compute_region_i_density_one_shot(
            system,
            basis,
            region,
            kmesh,
            contour,
            sigma_lin,
            lat_opts=lat_opts,
        )
    elif scf_method == "hartree":
        density_result = compute_region_i_density_scf(
            system,
            basis,
            region,
            kmesh,
            contour,
            sigma_lin,
            lat_opts=lat_opts,
            mixing=scf_mixing,
            max_iter=scf_max_iter,
        )
    elif scf_method == "hf":
        density_result = compute_region_i_density_scf_hf(
            system,
            basis,
            region,
            kmesh,
            contour,
            sigma_lin,
            lat_opts=lat_opts,
            mixing=scf_mixing,
            max_iter=scf_max_iter,
        )
    elif scf_method == "uhf":
        density_result = compute_region_i_density_scf_uhf(
            system,
            basis,
            region,
            kmesh,
            contour,
            sigma_lin,
            lat_opts=lat_opts,
            mixing=scf_mixing,
            max_iter=scf_max_iter,
        )
    else:
        raise ValueError(
            f"Unknown scf_method={scf_method!r}; "
            "choose 'one_shot', 'hartree', 'hf', or 'uhf'"
        )

    return EmbeddedSurfaceResult(
        region=region,
        density=density_result.density_i,
        density_local=density_result.density_i_local,
        occupation=density_result.occupation,
        band_energy=density_result.band_energy,
        contour=contour,
        sigma_lin_per_k=sigma_lin,
        e_fermi=ef,
        e_bottom=eb,
        per_k_trace=density_result.per_k_trace,
    )
