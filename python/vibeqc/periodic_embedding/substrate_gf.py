"""Substrate Green function for Layer A -- principal-layer construction.

Builds the semi-infinite substrate Green function ``g_surface(k_2d, z)``
and the surface-projected embedding potential ``Sigma_emb(k_2d, z)`` on the
dividing plane *S* from the periodic Gaussian-basis Hamiltonian of the clean
slab.

Principal-layer construction
----------------------------
The unit-cell AO indices are partitioned into layers by atom *z*-coordinate
(surface normal). For each layer index ``ℓ``, the blocks

    H00(ℓ) = H(k)[ℓ, ℓ]       (intra-layer)
    H01(ℓ) = H(k)[ℓ, ℓ+1]     (coupling to next layer deeper into bulk)

and the corresponding overlap blocks ``S00`` / ``S01`` are extracted from the
Bloch-summed matrices ``H(k) = T(k) + V_ne(k)`` and ``S(k)`` at the 2D surface
*k*-point ``k_2d``.

The effective "Hamiltonian" for Sancho-Rubio decimation in a non-orthogonal
basis is ``h00 = z*S00 - H00``, ``h01 = z*S01 - H01``. The decimation is run
starting from the topmost substrate layer (the layer just below the dividing
plane *S*); the result ``g_surface`` is the retarded surface Green function of
the semi-infinite stack at complex energy ``z``.

The embedding potential on the dividing-plane AOs (indices ``s_ao`` from
:class:`~vibeqc.periodic_embedding.region.RegionPartition`) is

    Sigma_emb(s, s'; k, z) = sum_{a,b} H_cpl[s, a] * g_surface[a, b] * H_cpl[s', b]^*

where ``H_cpl = z*S_cpl - H_cpl`` is the effective coupling between the
plane-*S* AOs and the topmost substrate layer.

References
----------
* M. P. Lopez Sancho et al., J. Phys. F 15, 851 (1985), doi:10.1088/0305-4608/15/4/009.
* J. E. Inglesfield, J. Phys. C 14, 3795 (1981), doi:10.1088/0022-3719/14/26/015.
* H. Ishida, Phys. Rev. B 63, 165409 (2001), doi:10.1103/PhysRevB.63.165409.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .._vibeqc_core import (
    BasisSet,
    BlochKMesh,
    LatticeSumOptions,
    PeriodicSystem,
    bloch_sum,
    compute_kinetic_lattice,
    compute_overlap_lattice,
    monkhorst_pack,
)
from ..periodic_v_ne import compute_nuclear_lattice_dispatch
from .decimation import sancho_rubio_surface_gf
from .region import RegionPartition, _ao_indices_for_atoms, _AtomAOMap


@dataclass(frozen=True)
class LayerPartition:
    """AO index partition of the unit cell into *z*-layers.

    Each layer is a contiguous block of AOs belonging to atoms whose
    *z*-coordinate falls into the same bin. Layers are numbered from
    bottom (smallest *z*) to top (largest *z*), matching the substrate's
    natural stacking direction.

    Attributes
    ----------
    layer_ao_ranges
        ``(n_layers, 2)`` -- ``[start, end)`` AO index ranges per layer.
    layer_z
        Mean *z*-coordinate (bohr) of each layer.
    n_layers
        Number of layers.
    layer_of_atom
        ``(n_atoms,)`` -- which layer each atom belongs to.
    """

    layer_ao_ranges: NDArray[np.int64]  # (n_layers, 2)
    layer_z: NDArray[np.float64]  # (n_layers,)
    layer_of_atom: NDArray[np.int64]  # (n_atoms,)

    @property
    def n_layers(self) -> int:
        return int(self.layer_ao_ranges.shape[0])

    @classmethod
    def from_system(
        cls,
        system: PeriodicSystem,
        basis: BasisSet,
        *,
        layer_tol: float = 0.5,
    ) -> "LayerPartition":
        """Group atoms (and their AOs) into layers by *z*-coordinate.

        Parameters
        ----------
        system
            Slab system. Atoms are binned by their Cartesian *z* position.
        basis
            Orbital basis set (determines the AO->atom mapping).
        layer_tol
            Tolerance in bohr for considering two atoms to be in the same
            layer. Atoms whose *z* differ by less than ``layer_tol`` are
            binned together.

        Returns
        -------
        LayerPartition
            AO ranges, mean layer *z*, and per-atom layer assignment.
        """
        n_atoms = len(system.unit_cell)
        z_all = np.array([a.xyz[2] for a in system.unit_cell], dtype=float)

        # Sort atoms by z and bin.
        order = np.argsort(z_all)
        z_sorted = z_all[order]
        layer_of_atom_unsorted = np.empty(n_atoms, dtype=np.int64)
        layer_idx = 0
        i = 0
        while i < n_atoms:
            j = i + 1
            while j < n_atoms and z_sorted[j] - z_sorted[i] <= layer_tol:
                j += 1
            for k in range(i, j):
                layer_of_atom_unsorted[order[k]] = layer_idx
            layer_idx += 1
            i = j
        n_layers = layer_idx

        # Build AO ranges: AO indices are ordered by atom, so layers get
        # contiguous AO ranges *for atoms in the same layer*.
        ao_map = _AtomAOMap.from_basis(basis, n_atoms)
        starts = ao_map.atom_ao_start

        layer_ao_ranges = np.empty((n_layers, 2), dtype=np.int64)
        layer_z_mean = np.empty(n_layers, dtype=np.float64)
        for ell in range(n_layers):
            mask = layer_of_atom_unsorted == ell
            atom_indices = np.flatnonzero(mask)
            # AO range: from the first AO of the first atom to the last AO
            # of the last atom. Since AO ranges are contiguous per atom and
            # atoms in the same layer are consecutive in z, their AO ranges
            # should be contiguous in practice. We union the ranges.
            ao_start = int(starts[atom_indices[0]])
            ao_end = int(starts[atom_indices[-1] + 1])
            layer_ao_ranges[ell] = [ao_start, ao_end]
            layer_z_mean[ell] = float(z_all[atom_indices].mean())

        return cls(
            layer_ao_ranges=layer_ao_ranges,
            layer_z=layer_z_mean,
            layer_of_atom=layer_of_atom_unsorted,
        )


def _build_hk_sk(
    system: PeriodicSystem,
    basis: BasisSet,
    k_cart: NDArray[np.float64],
    lat_opts: LatticeSumOptions,
    *,
    v_eff_full: Optional[NDArray[np.float64]] = None,
) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Build Bloch-summed H(k) and S(k) at k-point ``k_cart``.

    The one-electron core is ``Hcore(k) = T(k) + V_ne(k)``.  When
    ``v_eff_full`` is given it is *added* to the core, promoting H(k) to
    the **SCF Fock** ``F(k) = T(k) + V_ne(k) + V_eff`` -- the Ishida
    two-step path, where ``V_eff`` is the converged substrate mean-field
    potential (see
    :func:`~vibeqc.periodic_embedding.scf2step.build_substrate_scf_potential`).

    Uses the same lattice-sum machinery as
    :func:`~vibeqc.pbc_gdf._pbc_gdf_gamma_setup`, but at arbitrary *k* rather
    than Γ only. Returns Hermitian (up to machine precision) matrices.

    Parameters
    ----------
    v_eff_full
        Optional ``(n_ao, n_ao)`` real effective potential added to the
        core Hamiltonian.  It is the **Γ-point** converged mean field;
        adding it at every *k* is exact for a Γ-only surface mesh and is
        the (dominant) k-independent mean-field approximation otherwise.
        ``None`` (default) keeps the bare Hcore (the original behaviour).
        Must be built in the embedding's own DF convention (no exxdiv) so
        it is consistent with *this* Hcore's nuclear-attraction gauge
        (the external pbc-gdf Hcore differs by a Madelung-scale shift; Sec.7).
    """
    _k = np.asarray(k_cart, dtype=float).reshape(3)
    S_lat = compute_overlap_lattice(basis, system, lat_opts)
    T_lat = compute_kinetic_lattice(basis, system, lat_opts)
    V_lat = compute_nuclear_lattice_dispatch(basis, system, lat_opts)

    S_k = np.asarray(bloch_sum(S_lat, _k), dtype=np.complex128)
    T_k = np.asarray(bloch_sum(T_lat, _k), dtype=np.complex128)
    V_k = np.asarray(bloch_sum(V_lat, _k), dtype=np.complex128)

    H_k = T_k + V_k
    if v_eff_full is not None:
        H_k = H_k + np.asarray(v_eff_full, dtype=float)
    # Explicit Hermitian symmetrization (the Bloch sum is Hermitian in
    # exact arithmetic; machine rounding can introduce tiny anti-Hermitian
    # noise).
    H_k = 0.5 * (H_k + H_k.conj().T)
    S_k = 0.5 * (S_k + S_k.conj().T)
    return H_k, S_k


def _extract_block(
    mat: NDArray[np.complex128],
    rows: Tuple[int, int],
    cols: Tuple[int, int],
) -> NDArray[np.complex128]:
    """Extract a block ``mat[rows[0]:rows[1], cols[0]:cols[1]]``."""
    return mat[rows[0] : rows[1], cols[0] : cols[1]]


def build_substrate_surface_gf(
    system: PeriodicSystem,
    basis: BasisSet,
    region: RegionPartition,
    k_2d: Sequence[float],
    z: complex,
    *,
    lat_opts: Optional[LatticeSumOptions] = None,
    layer_tol: float = 0.5,
    tol: float = 1e-12,
    max_iter: int = 64,
    v_eff_full: Optional[NDArray[np.float64]] = None,
) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Build the substrate surface Green function and embedding potential at
    one (k_2d, z) point.

    Parameters
    ----------
    system
        Clean slab system (no adsorbate). Layer A operates here.
    basis
        Orbital basis set.
    region
        Region partition from :class:`RegionPartition`. Defines which atoms
        are region I (above plane *S*) and which are substrate (below).
    k_2d
        2D surface k-vector in Cartesian bohr⁻¹. The third component is
        set to 0 internally (k is in-plane only).
    z
        Complex energy at which to evaluate the substrate GF. Must be off
        the real axis for convergence (use a contour node or ``E + i*eta``).
    lat_opts
        Lattice-sum options. If ``None``, a default with
        ``cutoff_bohr=30.0`` and ``CoulombMethod.DIRECT_TRUNCATED`` is used.
    layer_tol
        *z* tolerance for grouping atoms into layers (bohr).
    tol, max_iter
        Forwarded to :func:`~vibeqc.periodic_embedding.decimation.sancho_rubio_surface_gf`.
    v_eff_full
        Optional ``(n_ao, n_ao)`` converged substrate mean-field potential
        (Ishida step 1).  When given, the substrate principal-layer blocks
        and the coupling block are built from the SCF Fock ``Hcore + V_eff``
        instead of the bare Hcore, so ``Sigma_emb`` is at the SCF level.
        ``None`` (default) keeps the Hcore embedding potential.

    Returns
    -------
    (g_surface, sigma_emb_s)
        ``g_surface`` is the ``(n_layer, n_layer)`` surface Green function of
        the semi-infinite substrate at its topmost layer. ``sigma_emb_s`` is
        the ``(n_s, n_s)`` embedding self-energy on the plane-*S* AOs (indices
        ``region.s_ao``).

    Raises
    ------
    ValueError
        If there are no substrate atoms (tag 0) in the system, or if no
        substrate layers sit below the plane *S*.
    DecimationNotConvergedError
        If Sancho-Rubio fails to converge (``z`` too close to real axis).
    """
    if region.n_substrate_atoms == 0:
        raise ValueError(
            "No substrate atoms (tag=0) in the system; cannot build substrate GF."
        )

    if lat_opts is None:
        lat_opts = LatticeSumOptions()
        lat_opts.cutoff_bohr = 30.0
        lat_opts.nuclear_cutoff_bohr = 30.0

    # -- 3D k-vector: pad k_2d with 0 for the vacuum direction ----------
    _k = np.zeros(3, dtype=float)
    k2 = np.asarray(k_2d, dtype=float).reshape(-1)
    if k2.shape[0] < 1 or k2.shape[0] > 3:
        raise ValueError(f"k_2d must have 1-3 components, got {k2.shape[0]}")
    _k[: len(k2)] = k2

    # -- Build H(k), S(k) -----------------------------------------------
    # With v_eff_full set, H_k is the SCF Fock F(k) (Ishida step 1), so the
    # principal-layer blocks and the region-I/substrate coupling below are
    # extracted at the SCF level rather than from the bare Hcore.
    H_k, S_k = _build_hk_sk(system, basis, _k, lat_opts, v_eff_full=v_eff_full)

    # -- Layer partition -------------------------------------------------
    layers = LayerPartition.from_system(system, basis, layer_tol=layer_tol)

    # Determine which layers are substrate (below plane S), and which is the
    # topmost substrate layer (the one immediately below S, coupled to
    # region I).
    sub_mask = region.substrate_atoms
    layer_of_atom = layers.layer_of_atom
    sub_layers = sorted(set(int(layer_of_atom[a]) for a in sub_mask))
    if not sub_layers:
        raise ValueError("No substrate layers found")

    # Find the topmost substrate layer -- the one with highest z that is
    # still below the plane S (z < region.s_plane_z).
    sub_layer_z = layers.layer_z[sub_layers]
    below_plane = sub_layer_z < region.s_plane_z
    if not below_plane.any():
        raise ValueError(
            f"No substrate layers below plane S (z={region.s_plane_z:.3f}); "
            f"substrate layer z's: {sub_layer_z}"
        )
    top_sub_layer = int(sub_layers[int(np.argmax(sub_layer_z))])

    # -- Extract principal-layer blocks for the substrate ----------------
    # The substrate principal layers are the top_sub_layer and all deeper
    # layers (higher layer index = higher z, but the top_sub_layer is the
    # highest-z substrate layer). We need layers top_sub_layer, top_sub_layer+1, ...
    # going deeper into the bulk. Since the substrate is semi-infinite,
    # we iterate downward: all substrate layers from top_sub_layer downward.

    # For Sancho-Rubio: H00 is the intra-layer block of the top substrate
    # layer; H01 is the coupling from top_sub_layer to top_sub_layer+1
    # (deeper into bulk). Since the Sancho-Rubio iteration assumes
    # translational invariance (identical layers), we take H00/H01 from the
    # first pair and assume all deeper pairs are identical.
    # For a clean slab, this holds for the interior bulk layers; the surface
    # layer H00 may differ slightly due to surface relaxation. For now we
    # use the topmost substrate layer's blocks directly.

    # -- Extract energy-independent blocks for the principal layer ---------
    # The substrate's topmost layer defines the principal-layer unit.
    rng_top = tuple(int(x) for x in layers.layer_ao_ranges[top_sub_layer])
    n_top = rng_top[1] - rng_top[0]

    # Layers are numbered from bottom (z_min) to top (z_max). The next
    # layer deeper into the bulk has a *smaller* layer index (lower z).
    deeper_layers = [ell for ell in sub_layers if ell < top_sub_layer]
    if not deeper_layers:
        raise ValueError(
            "Only one substrate layer found below plane S; need at least "
            "two for inter-layer coupling H01. Add more substrate layers "
            "to the slab."
        )
    next_deeper = max(deeper_layers)  # closest in z to top_sub_layer
    rng_deeper = tuple(int(x) for x in layers.layer_ao_ranges[next_deeper])
    n_deeper = rng_deeper[1] - rng_deeper[0]
    if n_top != n_deeper:
        raise ValueError(
            f"Substrate layers have mismatched sizes: layer {top_sub_layer} has "
            f"{n_top} AOs, layer {next_deeper} has {n_deeper}. "
            "Sancho-Rubio requires identical principal-layer shapes. "
            "Check layer_tol and basis set assignments."
        )

    # Extract bare (energy-independent) H and S blocks.
    H00 = _extract_block(H_k, rng_top, rng_top)
    H01 = _extract_block(H_k, rng_top, rng_deeper)
    S00 = _extract_block(S_k, rng_top, rng_top)
    S01 = _extract_block(S_k, rng_top, rng_deeper)

    # -- Orthogonalize the principal-layer basis ---------------------------
    # Sancho-Rubio operates in an orthogonal basis (S=I). We transform
    # H -> H̃ = X^T H X with X = S00^{-1/2}, run the decimation in the
    # orthogonal basis, and transform the GF back.  Assumes translational
    # invariance: the deeper layer has the same S00 (same atom types).
    S00_eigh = np.linalg.eigh(S00)
    # Filter small eigenvalues to avoid blow-up (should not happen for
    # well-conditioned sto-3g, but be safe).
    s_vals = np.maximum(S00_eigh.eigenvalues, 1e-10)
    X = S00_eigh.eigenvectors @ np.diag(1.0 / np.sqrt(s_vals)) @ S00_eigh.eigenvectors.T
    # Ensure X is real (S00 is real-symmetric at Γ; X is real).
    X = np.asarray(X.real, dtype=float)

    Ht00 = X.T @ H00 @ X
    Ht01 = X.T @ H01 @ X  # uses X from surface layer

    # -- Sancho-Rubio decimation (orthogonal basis) -----------------------
    g_surf_orth, _g_bulk_orth = sancho_rubio_surface_gf(
        Ht00, Ht01, complex(z), tol=tol, max_iter=max_iter
    )
    # Transform surface GF back to the original (non-orthogonal) basis.
    g_surface = X @ g_surf_orth @ X.T

    # -- Embedding potential on plane-S AOs ------------------------------
    # Coupling between plane-S AOs and the top substrate layer, in the
    # original non-orthogonal basis.
    s_ao = region.s_ao
    if len(s_ao) == 0:
        raise ValueError("No AOs on plane S (region.s_ao is empty).")

    # Build the s -> top_sub_layer coupling block.
    s_rng = (int(s_ao[0]), int(s_ao[-1]) + 1)  # contiguous by construction
    H_cpl = H_k[s_rng[0] : s_rng[1], rng_top[0] : rng_top[1]]  # (n_s, n_top)
    S_cpl = S_k[s_rng[0] : s_rng[1], rng_top[0] : rng_top[1]]  # (n_s, n_top)

    # Orthogonalize the plane-S AOs too, so the coupling is consistently
    # in the orthogonal basis for the Dyson-like embedding formula.
    S_ss = S_k[s_rng[0] : s_rng[1], s_rng[0] : s_rng[1]]
    S_ss_eigh = np.linalg.eigh(S_ss)
    ss_vals = np.maximum(S_ss_eigh.eigenvalues, 1e-10)
    Xs = (
        S_ss_eigh.eigenvectors
        @ np.diag(1.0 / np.sqrt(ss_vals))
        @ S_ss_eigh.eigenvectors.T
    )
    Xs = np.asarray(Xs.real, dtype=float)

    # Effective coupling in the orthogonal basis: H̃_cpl = Xs^T . H_cpl . X
    Ht_cpl = Xs.T @ H_cpl @ X  # (n_s, n_top)

    # S_emb in the orthogonal basis: S̃ = H̃_cpl . g̃_surf . H̃_cpl^T
    # (Sancho-Rubio g_surf_orth is the GF of the semi-infinite substrate
    # at its top layer in the orthogonal basis.)
    sigma_orth = Ht_cpl @ g_surf_orth @ Ht_cpl.T

    # Transform back: S = Xs . S̃ . Xs^T
    sigma_emb_s = Xs @ sigma_orth @ Xs.T

    return g_surface, sigma_emb_s


def default_surface_k_mesh(
    system: PeriodicSystem,
    mesh: Tuple[int, int] = (4, 4),
) -> BlochKMesh:
    """Build a 2D surface Monkhorst-Pack mesh.

    Uses ``monkhorst_pack`` with ``mesh=(nx, ny, 1)`` -- only the
    in-plane directions are sampled. For ``dim=2`` systems the third
    mesh entry is ignored.

    Parameters
    ----------
    system
        Periodic slab system (dim=2 or 3).
    mesh
        In-plane MP subdivisions ``(nx, ny)``.

    Returns
    -------
    BlochKMesh
        k-point mesh with Cartesian k-vectors (bohr⁻¹) and weights.
    """
    full_mesh = (int(mesh[0]), int(mesh[1]), 1)
    return monkhorst_pack(system, full_mesh, use_symmetry=False)
