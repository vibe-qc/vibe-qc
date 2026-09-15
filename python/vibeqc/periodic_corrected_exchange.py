"""Corrected-Ewald multi-k exact exchange --- the one implementation.

On a finite k mesh the SCF density is Born-von-Karman-torus periodic, so
``D(g)`` does **not** decay with image distance. Contracting a bare
``1/r`` kernel against it over a radial image ball therefore does not
converge: adding image shells keeps adding non-decaying contributions and
drives the energy down without limit. That is not a gauge choice, it is a
divergent sum, and it is the CLAUDE.md § 7 symptom.

The convergent construction --- the same one PySCF's ``exxdiv='ewald'``
implements and the one the BIPOLE/ROHF/ROKS routes were built on --- splits
the Coulomb kernel with the Ewald parameter ``alpha``::

    K_full = K_SR(erfc(alpha r)/r)          real-space, short-ranged, convergent
           + K_LR(erf(alpha r)/r)           reciprocal, one channel per q = k - k'
           + xi_G0 . S(k) D(k) S(k)         the excluded q + G = 0 term

``K_SR`` is a lattice-block quantity built by
:func:`build_jk_2e_real_space` with the Ewald ``alpha`` as its screening
argument; the caller Bloch-folds it like any other block. ``K_LR`` and the
``q + G = 0`` correction are k-space objects and are added per k by
:meth:`CorrectedEwaldExchange.k_space_terms_at_k`.

``xi_G0 = xi_M(BvK supercell) - pi / (alpha^2 V N_k)`` is the probe-charge
Madelung constant of the supercell spanned by the mesh, less the
self-interaction of the Gaussian screening charge. q -> 0 treatment per
Gygi & Baldereschi, Phys. Rev. B 34, 4405(R) (1986),
doi:10.1103/PhysRevB.34.4405; probe-charge Ewald form per Sundararaman &
Arias, Phys. Rev. B 87, 165122 (2013), doi:10.1103/PhysRevB.87.165122;
Bloch AO-pair FT factorisation per Sun, Berkelbach, McClain & Chan,
J. Chem. Phys. 147, 164119 (2017), doi:10.1063/1.4998644.

The arbitrary-lattice generalisation of Gygi-Baldereschi is Carrier, Rohra
& Görling, Phys. Rev. B 75, 205126 (2007), doi:10.1103/PhysRevB.75.205126.
Its abstract and § II give the property this class is built on: the
singularity correction *"depend[s] only on the total number and the
positions of k points and on the lattice vectors, in particular the unit
cell volume, but not on the particular positions of atoms within the unit
cell"*. That is why ``xi_G0`` is one scalar, computed once per SCF from
the mesh and the lattice alone, and reused at every k regardless of the
geometry --- pinned by
``tests/test_periodic_corrected_exchange.py::test_exxdiv_constant_is_independent_of_atomic_positions``.

**This applies to the full-Coulomb (``c_full``) arm only.** A screened
hybrid's ``c_sr`` arm is ``erfc(omega_screen r)/r``, which is already
short-ranged and convergent in real space, so it stays on
:func:`vibeqc.periodic_screened_exchange.build_exchange_blocks`. HSE06 has
``c_full = 0`` and is therefore untouched by this module.

Preconditions, all fail closed rather than falling back to the divergent
sum: ``dim == 3`` (the reciprocal cache is 3D-only), and a uniform-weight
Monkhorst-Pack mesh carrying its dimensions (the torus density list
:func:`bvk_torus_density_matrices` needs one). Symmetry-reduced (IBZ) and
Gilat-Raubenheimer meshes do not qualify.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ._vibeqc_core import BasisSet, PeriodicSystem

__all__ = [
    "CorrectedEwaldExchange",
    "corrected_ewald_reciprocal_cutoff",
    "corrected_exchange_unavailable_reason",
    "is_unsupported_gauge",
]


def is_unsupported_gauge(system: PeriodicSystem, *, slab_mode: bool) -> bool:
    """True when the corrected split does not exist for this Coulomb gauge.

    2D slab and ``dim != 3`` cells: the reciprocal q-channel cache is
    3D-only. Distinct from a mesh this code merely cannot handle yet ---
    see :func:`corrected_exchange_unavailable_reason`.
    """
    return bool(slab_mode) or int(system.dim) != 3


def corrected_ewald_reciprocal_cutoff(
    volume: float,
    alpha: float,
    precision: float = 1.0e-8,
) -> float:
    """Reciprocal envelope shared by corrected-Ewald energy and gradient.

    The volume-based CRYSTAL cutoff is calibrated to its volume-matched
    default alpha.  A larger explicit alpha broadens
    ``exp(-K^2 / (4 alpha^2))``, so resolve that Gaussian to ``precision``
    and retain the existing 1.5 safety factor.  Returning the larger cutoff
    preserves CRYSTAL parity at the default while making explicit-alpha
    energy/gradient pairs use identical reciprocal domains.
    """
    volume_f = float(volume)
    alpha_f = float(alpha)
    precision_f = float(precision)
    if volume_f <= 0.0:
        raise ValueError(f"volume must be positive; got {volume!r}")
    if alpha_f <= 0.0:
        raise ValueError(f"alpha must be positive; got {alpha!r}")
    if not 0.0 < precision_f < 1.0:
        raise ValueError(
            f"precision must lie strictly between 0 and 1; got {precision!r}"
        )

    from .bipole_ext_el_pole import crystal_ewald_reciprocal_cutoff

    k_max_alpha = (
        2.0 * alpha_f * float(np.sqrt(-np.log(precision_f))) * 1.5
    )
    return max(
        float(crystal_ewald_reciprocal_cutoff(volume_f)),
        k_max_alpha,
    )


def corrected_exchange_unavailable_reason(
    system: PeriodicSystem,
    kmesh,
    *,
    slab_mode: bool = False,
) -> Optional[str]:
    """Why the corrected split cannot serve this job, or ``None`` if it can.

    Returned string is a sentence fragment naming the precondition that
    fails, for the caller to embed in its own error or log line.

    Callers must distinguish two classes of failure, because the right
    response differs:

    * **A mesh the construction cannot serve** --- an *explicit* k list
      (no Monkhorst-Pack metadata, no ``ir_mapping``). There is no star
      structure to unfold from, so these **fail closed**.
    * **A gauge the construction does not exist for** --- 2D slab, or
      ``dim != 3``. The reciprocal q-channel cache is 3D-only. These are
      live routes that pre-date the corrected split, so they keep the
      historical kernel and say so, rather than being deleted.

    A **symmetry-reduced (IBZ) Monkhorst-Pack mesh carrying non-empty
    ``ir_mapping``** is accepted since 2026-07-29: the split serves it by
    unfolding the wedge densities to the full BZ via the star transport
    (``periodic_k_symmetry.KMeshUnfolding``; density law per Pisani,
    Dovesi & Roetti 1988, Eq. II.7.21). Weighting the irreducible
    representatives without the rotation is still wrong (measured
    +1.389 Ha on MgO (2,2,2)) and is never done.

    :func:`is_unsupported_gauge` separates the failure classes.
    """
    if slab_mode:
        return (
            "the 2D slab Coulomb gauge has no corrected-Ewald exchange "
            "split (the reciprocal q-channel cache is 3D-only)"
        )
    if int(system.dim) != 3:
        return (
            f"the corrected-Ewald exchange split requires dim=3; "
            f"got dim={int(system.dim)}"
        )
    n_k = len(list(kmesh.kpoints))
    mesh = tuple(int(x) for x in getattr(kmesh, "mesh", (1, 1, 1)))
    n_full = int(np.prod(mesh))
    ir_mapping = np.asarray(
        getattr(kmesh, "ir_mapping", []), dtype=int
    ).reshape(-1)
    if n_full != n_k:
        # Not a full mesh. A symmetry-reduced wedge is fine when it
        # carries its star metadata and the system's symmetry is attached
        # (the unfolding needs the operations).
        if ir_mapping.size == n_full and n_k < n_full:
            ops = getattr(
                getattr(system, "symmetry", None), "operations", None
            )
            if not ops:
                return (
                    "this symmetry-reduced k-mesh carries no attached "
                    "symmetry operations to unfold with; call "
                    "vq.attach_symmetry(system) before building the mesh"
                )
            return None
        return (
            "the corrected-Ewald exchange split requires a Monkhorst-Pack "
            "mesh carrying its dimensions (full, or symmetry-reduced with "
            f"ir_mapping star metadata); got mesh={mesh} for {n_k} "
            "k-points with no usable star map, which is an explicit "
            "k-point list"
        )
    weights = np.asarray(kmesh.weights, dtype=float)
    if n_k and not np.allclose(weights, 1.0 / float(n_k), atol=1e-12):
        return (
            "the corrected-Ewald exchange split requires uniform k-point "
            "weights on a full mesh (the Born-von-Karman torus unfolding "
            f"assumes w_k = 1/{n_k}); got a non-uniform weight set"
        )
    return None


@dataclass
class CorrectedEwaldExchange:
    """Prebuilt k-space arms of the corrected exchange split.

    Build once per SCF via :meth:`build`; call
    :meth:`k_space_terms_at_k` inside the k loop. The short-range arm is
    *not* held here --- it is an ordinary lattice-block quantity the caller
    already builds and folds (pass ``alpha`` as the screening argument of
    :func:`build_jk_2e_real_space`, not ``0.0``).
    """

    k_cache: object
    exchange_g0: float
    alpha: float
    mesh: Tuple[int, int, int]
    # The FULL-BZ quadrature, always: on a symmetry-reduced input mesh
    # these are the regenerated full mesh's points and uniform weights,
    # never the wedge (the reciprocal channels q = k - k' and the q+G=0
    # constant both live on the full mesh).
    k_points: List[np.ndarray]
    weights: np.ndarray
    # Symmetry-reduced input only: the star bookkeeping that unfolds
    # wedge densities to the full BZ (None on a full-mesh input), and
    # per wedge point its own index in the full-mesh list.
    unfolding: object = None
    target_full_indices: Optional[List[int]] = None
    # False until one full serial pass has populated the lazily-built
    # channel_tables caches; see k_space_terms_all_k.
    _cache_is_warm: bool = False

    @classmethod
    def build(
        cls,
        basis: BasisSet,
        system: PeriodicSystem,
        cells_r_cart: np.ndarray,
        kmesh,
        alpha: float,
        *,
        where: str,
        slab_mode: bool = False,
        precision: float = 1.0e-8,
        lattice_opts=None,
    ) -> "CorrectedEwaldExchange":
        """Construct the reciprocal cache and the ``q + G = 0`` constant.

        ``alpha`` is the **Ewald split parameter** the driver already uses
        for its Hartree J and nuclear gauge --- not a functional's
        range-separation omega. Raises :class:`NotImplementedError` when a
        precondition fails, so no caller can silently fall back to the
        divergent bare sum.
        """
        reason = corrected_exchange_unavailable_reason(
            system, kmesh, slab_mode=slab_mode
        )
        if reason is not None:
            raise NotImplementedError(f"{where}: {reason}.")
        if float(alpha) <= 0.0:
            raise ValueError(
                f"{where}: corrected-Ewald exchange needs a positive Ewald "
                f"split alpha; got {alpha!r}."
            )

        from .bipole_fock_ewald import (
            _build_j_long_range_cache,
            build_k_exchange_long_range_cache,
            probe_charge_madelung_supercell,
            exchange_q0_gauge_constant,
        )

        mesh = tuple(int(x) for x in getattr(kmesh, "mesh", (1, 1, 1)))
        n_input = len(list(kmesh.kpoints))
        unfolding = None
        target_full_indices: Optional[List[int]] = None
        if int(np.prod(mesh)) != n_input:
            # Symmetry-reduced wedge (the gate above already vetted the
            # metadata). Regenerate the full mesh and build the star
            # transport; the reciprocal channels, weights, and the q+G=0
            # constant all live on the FULL mesh, while the caller keeps
            # diagonalising on the wedge.
            from .periodic_k_symmetry import KMeshUnfolding

            try:
                unfolding = KMeshUnfolding.build(system, basis, kmesh)
            except ValueError as exc:
                raise NotImplementedError(
                    f"{where}: this symmetry-reduced mesh cannot be "
                    f"unfolded to its full Brillouin zone ({exc}). Use "
                    "the full mesh (use_symmetry=False)."
                ) from exc
            kmesh_quad = unfolding.kmesh_full
            target_full_indices = list(unfolding.rep_full_index)
        else:
            kmesh_quad = kmesh
        k_points = [
            np.asarray(k, dtype=float).reshape(3) for k in kmesh_quad.kpoints
        ]
        weights = np.asarray(kmesh_quad.weights, dtype=float)
        n_k = len(k_points)
        volume = float(
            abs(np.linalg.det(np.asarray(system.lattice, dtype=float)))
        )
        # Reciprocal cutoff. CRYSTAL's volume-based default is tuned to the
        # volume-matched Ewald alpha; a caller that supplies its own (larger)
        # alpha widens the Gaussian envelope exp(-K^2 / 4 alpha^2) and that
        # default then truncates the sum while the summand is still large.
        # Measured on a 30-bohr box at alpha = 0.5 the neglected envelope is
        # 0.53 -- i.e. more than half the peak -- and even a 12-bohr box costs
        # 1.1e-4 Ha. So take whichever cutoff is tighter: the CRYSTAL default
        # or the one that actually resolves this alpha to `precision`.
        #   exp(-K_max^2 / 4 alpha^2) <= precision  =>  K_max >= 2 alpha sqrt(-ln precision)
        k_max = corrected_ewald_reciprocal_cutoff(
            volume, float(alpha), float(precision)
        )
        j_cache = _build_j_long_range_cache(
            basis,
            system,
            np.asarray(cells_r_cart, dtype=float),
            float(alpha),
            float(precision),
            K_max=k_max,
            lattice_opts=lattice_opts,
        )
        k_cache = build_k_exchange_long_range_cache(
            basis, system, j_cache, K_max=k_max
        )
        xi_madelung = probe_charge_madelung_supercell(system, mesh)
        exchange_g0 = exchange_q0_gauge_constant(
            xi_madelung, alpha, volume, n_k
        )
        return cls(
            k_cache=k_cache,
            exchange_g0=float(exchange_g0),
            alpha=float(alpha),
            mesh=mesh,  # type: ignore[arg-type]
            k_points=k_points,
            weights=weights,
            unfolding=unfolding,
            target_full_indices=target_full_indices,
        )

    def torus_densities(self, density) -> List[np.ndarray]:
        """Per-k Born-von-Karman torus density matrices for ``density``.

        Reconstructs ``D(k)`` from a real-space :class:`LatticeMatrixSet`.
        This needs the cell list to *cover* the BvK torus and raises
        naming the missing cell otherwise --- a real constraint, since a
        30-bohr cell at a 12-bohr lattice cutoff carries exactly one cell
        and covers a ``(2,1,1)`` torus not at all.

        **Prefer passing the driver's own per-k densities** to
        :meth:`k_space_terms_at_k` when it has them. Every multi-k SCF
        driver here carries ``D(k) = C(k) f(k) C(k)^H`` in lock-step with
        the real-space density it folds from them, so the round trip is
        avoidable: it is the *derived* quantity, not the primary one.
        Using the per-k list directly is both cheaper and free of the
        coverage constraint. This method exists for callers holding only
        a lattice density.
        """
        from .pbc_bipole_common import bvk_torus_density_matrices

        return bvk_torus_density_matrices(density, self.k_points, self.mesh)

    def k_space_terms_at_k(
        self,
        idx: int,
        S_k: np.ndarray,
        D_k_list: Sequence[np.ndarray],
    ) -> np.ndarray:
        """``K_LR(k_idx) + xi_G0 . S(k) D(k) S(k)``.

        The two k-space arms of the split. Add to the Bloch-folded
        short-range blocks to obtain the full corrected ``K(k)``; the
        caller applies its own ``c_full`` and per-spin/closed-shell
        prefactor.
        """
        from .bipole_fock_ewald import compute_K_long_range_at_k

        k_lr = compute_K_long_range_at_k(
            self.k_cache,
            self.k_points[idx],
            self.k_points,
            self.weights,
            D_k_list,
        )
        return k_lr + self.exchange_g0 * (S_k @ D_k_list[idx] @ S_k)

    def k_space_terms_all_k(
        self,
        S_k_list: Sequence[np.ndarray],
        D_k_list: Sequence[np.ndarray],
    ) -> List[np.ndarray]:
        """:meth:`k_space_terms_at_k` at every k, threaded over k.

        The ``q = k - k'`` channel loop inside
        :func:`compute_K_long_range_at_k` makes this **O(n_k^2)** work and
        it is the dominant per-iteration cost of a multi-k HF or hybrid
        SCF: measured at 16.5 s per iteration for a 32-k mesh on an
        18-function cell, and 23 s at 64 k.

        Each k is independent given the shared cache and the full density
        list, so the loop farms over k with no communication until the
        Fock assembly. The contractions spend most of their time inside
        BLAS with the GIL released, so plain threads scale: measured
        **7.1x on 16 threads** at 32 k (16.5 s -> 2.3 s), bit-identical to
        the serial result because each k performs exactly the same
        operations in the same order.

        The first call runs **serial on purpose**. ``channel_tables``
        populates its ``q_channels`` / ``b_per_k`` caches lazily, and a
        cold parallel first pass would have every thread miss, recompute
        the same ``B`` tensors, and race to store them --- last writer
        wins, so the losers' entries are orphaned and recomputed forever
        after. One serial pass fills the cache; every later call is a pure
        read and threads cleanly.

        This is also an MPI seam, and a **different one** from the k-point
        farming already landed under ``HANDOVER_MPI.md`` next-step 4: that
        partitions the GDF cderi q-groups once per SCF *setup* and has
        nothing per iteration. This loop runs every iteration and is where
        the O(n_k^2) cost actually lives, so it is the one that rewards
        multi-node farming. Same decomposition as here, with the per-k
        results gathered rather than shared.

        ``VIBEQC_EXCHANGE_K_THREADS`` overrides the worker count (``1``
        forces serial).

        **Symmetry-reduced input** (``self.unfolding`` set): ``S_k_list``
        and ``D_k_list`` are the *wedge* lists the driver holds. The
        wedge densities are unfolded to the full BZ once per call
        (Pisani Eq. II.7.21 star transport), the k' quadrature runs over
        the full mesh, and terms are computed **only at the wedge
        representatives** --- the SCF never diagonalises elsewhere. That
        is the efficiency of the wedge: the target loop shrinks by the
        star multiplicity (4.5x at a cubic ``(3,3,3)``) while each
        target's channel sum keeps its full-BZ quadrature.
        """
        if self.unfolding is not None:
            D_full = self.unfolding.unfold(D_k_list)
            targets = list(self.target_full_indices)
        else:
            D_full = D_k_list
            targets = list(range(len(self.k_points)))
        if len(S_k_list) != len(targets):
            raise ValueError(
                "k_space_terms_all_k: S_k_list has "
                f"{len(S_k_list)} entries but there are {len(targets)} "
                "target k-points"
            )
        workers = self._k_worker_count(len(targets))
        if workers <= 1 or not self._cache_is_warm:
            out = [
                self.k_space_terms_at_k(j, S_k_list[i], D_full)
                for i, j in enumerate(targets)
            ]
            self._cache_is_warm = True
            return out

        from concurrent.futures import ThreadPoolExecutor

        # Keep this pool free of user-facing output. ``vibeqc.output``'s
        # active channel lives in a ContextVar, which a worker thread does
        # not inherit, so a ``write()`` anywhere below here would be
        # dropped silently. The chain is pure numerics today and
        # ``tests/test_output_channel.py`` allowlists this one site on that
        # basis; adding an emitter below it needs the threading.local
        # fallback in ``vibeqc.output.channel`` first.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(
                pool.map(
                    lambda ij: self.k_space_terms_at_k(
                        ij[1], S_k_list[ij[0]], D_full
                    ),
                    enumerate(targets),
                )
            )

    def fold_density(self, D_k_wedge: Sequence[np.ndarray], lattice_opts):
        """Exact real-space fold of wedge densities (IBZ input only).

        The weighted wedge fold uses each representative in place of its
        orbit average and is wrong at multi-cell cutoffs; the orbit sum
        (Pisani Eq. II.7.48) needs the rotated star members. Exchange is
        linear in the density, so on the ``c_full != 0`` path the driver
        must feed J / K_SR / XC this fold rather than its wedge fold.
        """
        if self.unfolding is None:
            raise ValueError(
                "fold_density: only meaningful for a symmetry-reduced "
                "input mesh; the driver's own fold is already exact on a "
                "full mesh"
            )
        return self.unfolding.fold_density(D_k_wedge, lattice_opts)

    @staticmethod
    def _k_worker_count(n_k: int) -> int:
        """Threads to farm the k loop over; 1 means run serial."""
        import os

        raw = os.environ.get("VIBEQC_EXCHANGE_K_THREADS", "").strip()
        if raw:
            try:
                requested = int(raw)
            except ValueError:
                requested = 0
            if requested > 0:
                return max(1, min(requested, n_k))
        try:
            from ._vibeqc_core import get_num_threads

            available = int(get_num_threads())
        except Exception:
            available = int(os.cpu_count() or 1)
        # Below a handful of k-points the pool costs more than it saves.
        if n_k < 4 or available <= 1:
            return 1
        return max(1, min(available, n_k))
