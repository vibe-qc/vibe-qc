"""Surface-projected embedding potential for region I.

Wraps the per-(k, z) embedding potential ``Sigma_emb(k, z)`` computed by
:func:`~vibeqc.periodic_embedding.substrate_gf.build_substrate_surface_gf`
into an :class:`~vibeqc.periodic_embedding.embedding_potential.EmbeddingPotential`
that can be energy-linearized (Ishida) and applied to the full region-I
Hamiltonian during the two-step SCF.

The substrate GF machinery returns ``Sigma_emb`` as an ``(n_s, n_s)`` matrix
in the plane-*S* AO basis (indices ``region.s_ao``).  This module zero-pads
it to the full region-I basis ``(n_i, n_i)`` so it can be added directly to
``H_I(k)`` during the contour SCF and Dyson solves.

References
----------
* H. Ishida, Phys. Rev. B 63, 165409 (2001), doi:10.1103/PhysRevB.63.165409.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from .._vibeqc_core import BasisSet, LatticeSumOptions, PeriodicSystem
from .embedding_potential import EmbeddingPotential, LinearizedEmbeddingPotential
from .region import RegionPartition
from .substrate_gf import build_substrate_surface_gf


def build_surface_sigma(
    system: PeriodicSystem,
    basis: BasisSet,
    region: RegionPartition,
    k_2d: Sequence[float],
    *,
    lat_opts: Optional[LatticeSumOptions] = None,
    layer_tol: float = 0.5,
    tol: float = 1e-12,
    max_iter: int = 64,
    v_eff_full: Optional[NDArray[np.float64]] = None,
    label: str = "",
) -> EmbeddingPotential:
    """Build the energy-dependent embedding potential ``Sigma_emb(z)``
    for region I at a fixed 2D surface k-point.

    The returned :class:`EmbeddingPotential` is a callable ``z -> S(z)``
    where ``S(z)`` is an ``(n_i, n_i)`` complex matrix -- the embedding
    potential zero-padded from the plane-*S* AOs to the full region-I
    AO basis.  It is ready for energy linearization via
    :meth:`EmbeddingPotential.linearize` and for contour integration.

    Parameters
    ----------
    system
        Clean slab system (Layer A).
    basis
        Orbital basis set.
    region
        Region partition.
    k_2d
        2D surface k-vector in Cartesian bohr⁻¹.
    lat_opts
        Lattice-sum options forwarded to
        :func:`~vibeqc.periodic_embedding.substrate_gf.build_substrate_surface_gf`.
    layer_tol, tol, max_iter
        Forwarded to the same.
    v_eff_full
        Optional converged substrate mean-field potential (Ishida step 1);
        forwarded to ``build_substrate_surface_gf`` so ``Sigma_emb`` is
        built from the SCF Fock rather than the bare Hcore.
    label
        Human-readable label for the embedding potential (e.g. ``"Γ"``).

    Returns
    -------
    EmbeddingPotential
        Callable ``z -> S(z)``, shape ``(n_i, n_i)``.
    """
    k = np.asarray(k_2d, dtype=float)

    def _sigma_fn(z: complex) -> NDArray[np.complex128]:
        _gs, sigma_s = build_substrate_surface_gf(
            system,
            basis,
            region,
            k,
            z,
            lat_opts=lat_opts,
            layer_tol=layer_tol,
            tol=tol,
            max_iter=max_iter,
            v_eff_full=v_eff_full,
        )
        return _zero_pad_s_to_i(sigma_s, region)

    return EmbeddingPotential(_sigma_fn, label=label)


def _zero_pad_s_to_i(
    sigma_s: NDArray[np.complex128],
    region: RegionPartition,
) -> NDArray[np.complex128]:
    """Zero-pad a plane-*S* matrix ``(n_s, n_s)`` to region-I shape ``(n_i, n_i)``.

    ``sigma_s[i, j]`` (in the ``s_ao`` ordering) maps to
    ``Sigma_II[s_loc[i], s_loc[j]]`` where ``s_loc`` is the index of each
    ``s_ao`` element within ``i_ao``.
    """
    n_i = region.n_i
    sigma_i = np.zeros((n_i, n_i), dtype=np.complex128)
    if region.n_s == 0:
        return sigma_i

    # Build a lookup: s_ao_idx -> position within i_ao.
    i_to_pos = {int(idx): pos for pos, idx in enumerate(region.i_ao)}
    s_loc = np.array([i_to_pos[int(idx)] for idx in region.s_ao], dtype=int)

    sigma_i[np.ix_(s_loc, s_loc)] = sigma_s
    return sigma_i


def linearize_surface_sigma(
    system: PeriodicSystem,
    basis: BasisSet,
    region: RegionPartition,
    k_2d: Sequence[float],
    z0: complex,
    *,
    lat_opts: Optional[LatticeSumOptions] = None,
    layer_tol: float = 0.5,
    tol: float = 1e-12,
    max_iter: int = 64,
    v_eff_full: Optional[NDArray[np.float64]] = None,
    step: float = 1e-5,
    label: str = "",
) -> LinearizedEmbeddingPotential:
    """Build and immediately linearize the embedding potential at ``z0``.

    Convenience wrapper that calls :func:`build_surface_sigma` then
    :meth:`EmbeddingPotential.linearize` in one step.  Returns an
    :class:`LinearizedEmbeddingPotential` for cheap evaluation at every
    contour node.

    Parameters
    ----------
    z0
        Reference complex energy for the Ishida linearization (typically
        the contour centre or a representative node).
    step
        Finite-difference step along the real axis for ``dS/dz``.
    """
    sigma = build_surface_sigma(
        system,
        basis,
        region,
        k_2d,
        lat_opts=lat_opts,
        layer_tol=layer_tol,
        tol=tol,
        max_iter=max_iter,
        v_eff_full=v_eff_full,
        label=label,
    )
    return sigma.linearize(z0, step=step)
