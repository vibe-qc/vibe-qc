"""Post-SCF helpers for the GAPW periodic route -- DOS + band structure.

Provides GPAW-equivalent surface for the v0.11.x GPW driver:

* :func:`gaussian_dos` -- Gaussian-broadened density of states from a
  flat array of eigenvalues + per-state weights;
* :func:`compute_dos_from_result` -- wrapper that pulls eigenvalues
  out of a converged :class:`GpwScfResult` or :class:`GpwMultiKScfResult`
  and broadens them with closed-shell occupation weights;
* :func:`compute_homo_lumo` -- HOMO / LUMO / gap from a converged
  result, supporting both single-k (Γ) and multi-k results;
* :func:`band_path_eigenvalues` -- solve ``F(k) C(k) = e S(k) C(k)``
  at a list of user-supplied k-points using a FIXED converged
  density matrix; the per-k F is built with the same lattice
  builders + Bloch sum + GAPW J/V_xc projection that
  :func:`run_periodic_rks_gpw_multi_k` uses, so band-structure
  eigenvalues are gauge-consistent with the SCF.

The GPAW reference shape is ``calc.get_dos()`` and
``calc.band_structure()``. The vibe-qc equivalents are deliberately
plain functions on the result rather than methods on a calculator --
matches the rest of the GAPW post-SCF surface (cube writer, QVF
emitter).

> **Experimental.** Same caveat as the rest of the GPW route:
> the underlying SCF emits :class:`GAPWExperimentalWarning` at
> construction; these helpers don't add their own warning, but
> consume results from a driver that does.

See :mod:`vibeqc.periodic_gapw_j` for the underlying SCF, the
breakdown shape, and the GPAW-comparable parts of the API.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence, Tuple

import numpy as np

from . import _vibeqc_core as _core
from .periodic_gapw_grid import GAPWExperimentalWarning, PlaneWaveGrid

__all__ = [
    "band_path_eigenvalues",
    "compute_dos_from_result",
    "compute_homo_lumo",
    "gaussian_dos",
]


# ============================================================
# 1. Gaussian-broadened DOS on a flat eigenvalue array
# ============================================================


def gaussian_dos(
    mo_energies: np.ndarray,
    *,
    e_grid: Optional[np.ndarray] = None,
    sigma: float = 0.01,
    e_range: Optional[Tuple[float, float]] = None,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gaussian-broadened density of states.

    Builds ``DOS(E) = S_i w_i . g(E - e_i; s)`` where ``g`` is a
    unit-norm Gaussian of width ``s``.

    Parameters
    ----------
    mo_energies
        ``(n_states,)`` array of eigenvalues in Hartree. Pass a
        flat 1D array; multi-k callers stack per-k eigenvalues
        before calling (see :func:`compute_dos_from_result`).
    e_grid
        ``(n_points,)`` energy grid in Hartree at which to
        evaluate the DOS. When ``None`` the helper auto-builds
        one -- either from ``e_range`` (linearly spaced over the
        range with 800 points) or from the eigenvalue extent
        with ``±10.s`` padding.
    sigma
        Gaussian width in Hartree. Default ``0.01`` Ha ≈ 0.27 eV
        -- a sensible plotting default that's narrow enough to
        resolve well-separated MOs and wide enough to avoid the
        zero-width singular limit.
    e_range
        Optional ``(e_lo, e_hi)`` Hartree window used to build
        ``e_grid`` when ``e_grid`` is None. Ignored when an
        explicit ``e_grid`` is passed.
    weights
        ``(n_states,)`` per-state weights. Defaults to 1.0 per
        state (raw spectral count). For a closed-shell DOS the
        caller passes ``2 . occupied`` (electronic count).

    Returns
    -------
    (e_grid, dos)
        Both ``(n_points,)`` arrays -- energies in Hartree, DOS
        in states / Hartree. ``np.trapz(dos, e_grid)`` recovers
        ``S_i w_i`` (the integral over the grid equals the total
        weight) to within the grid's truncation error at the
        Gaussian tails.
    """
    eps = np.asarray(mo_energies, dtype=float).ravel()
    if eps.size == 0:
        raise ValueError("gaussian_dos: mo_energies is empty")
    if sigma <= 0.0:
        raise ValueError(f"gaussian_dos: sigma must be positive (got {sigma!r})")
    if weights is None:
        w = np.ones_like(eps)
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.shape != eps.shape:
            raise ValueError(
                f"gaussian_dos: weights shape {w.shape} doesn't match "
                f"mo_energies shape {eps.shape}"
            )

    if e_grid is None:
        if e_range is not None:
            e_lo, e_hi = float(e_range[0]), float(e_range[1])
            if not e_hi > e_lo:
                raise ValueError(
                    f"gaussian_dos: e_range must satisfy e_hi > e_lo (got {e_range!r})"
                )
        else:
            pad = 10.0 * sigma
            e_lo = float(eps.min()) - pad
            e_hi = float(eps.max()) + pad
            if not e_hi > e_lo:
                # Degenerate single-eigenvalue case -- pad explicitly.
                e_hi = e_lo + max(20.0 * sigma, 1e-6)
        e_grid_arr = np.linspace(e_lo, e_hi, 800)
    else:
        e_grid_arr = np.asarray(e_grid, dtype=float).ravel()
        if e_grid_arr.size == 0:
            raise ValueError("gaussian_dos: e_grid is empty")

    # Vectorised Gaussian sum.
    # DOS(E) = S_i w_i . (1 / (s √(2pi))) . exp(-(E - e_i)^2 / (2s^2))
    norm = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    delta = e_grid_arr[:, None] - eps[None, :]  # (n_points, n_states)
    g = norm * np.exp(-0.5 * (delta / sigma) ** 2)
    dos = (g * w[None, :]).sum(axis=1)
    return e_grid_arr, dos


# ============================================================
# 2. DOS from a converged GPW result
# ============================================================


def compute_dos_from_result(
    result,
    *,
    system=None,
    n_elec: Optional[int] = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gaussian-broadened DOS from a converged GPW SCF result.

    Pulls eigenvalues from ``result.mo_energies`` (single-k) or
    ``result.mo_energies_k`` (multi-k) and computes the closed-
    shell DOS by treating each occupied orbital as carrying weight
    ``2`` and each virtual orbital as carrying weight ``0``:

    * Single-k: ``weights[i] = result.occupations[i]`` if available,
      else ``weights[i] = 2.0`` for ``i < n_occ`` and ``0.0``
      otherwise (n_occ derived from ``result.density`` .
      ``result.overlap`` when present, falling back to
      ``S Z`` // 2 if the system is on the result).
    * Multi-k: ``weights[i, k] = w_k . 2`` for occupied,
      ``w_k . 0`` for virtual. The result is the integrated
      density of states summed over the BZ sample.

    Any extra ``**kwargs`` are forwarded to :func:`gaussian_dos`
    (e.g., ``sigma=``, ``e_grid=``, ``e_range=``).

    Parameters
    ----------
    result
        :class:`vibeqc.periodic_gapw_j.GpwScfResult` (Γ-only) or
        :class:`vibeqc.periodic_gapw_j.GpwMultiKScfResult`
        (multi-k). Duck-typed on the presence of
        ``mo_energies_k`` (multi-k) vs ``mo_energies`` (single-k).

    Returns
    -------
    (e_grid, dos)
        See :func:`gaussian_dos`.
    """
    # Multi-k path: stack per-k eigenvalues with k-weighted occupation.
    if hasattr(result, "mo_energies_k") and result.mo_energies_k:
        eigs_k = result.mo_energies_k
        kmesh = getattr(result, "kmesh", None)
        if kmesh is not None and hasattr(kmesh, "weights"):
            k_weights = np.asarray(kmesh.weights, dtype=float)
        else:
            # Uniform fallback if no kmesh attached.
            k_weights = np.full(len(eigs_k), 1.0 / len(eigs_k))

        n_occ = _infer_n_occ(result, system=system, n_elec=n_elec)

        flat_eps = []
        flat_w = []
        for ik, eps_k in enumerate(eigs_k):
            eps_k = np.asarray(eps_k, dtype=float).ravel()
            n_basis = eps_k.size
            w = np.zeros(n_basis, dtype=float)
            # Closed shell: occupied carry 2 electrons. The k-weight
            # captures the BZ-integration measure so summing across
            # k reproduces 2 . n_occ electrons.
            w[:n_occ] = 2.0 * k_weights[ik]
            flat_eps.append(eps_k)
            flat_w.append(w)
        eps_all = np.concatenate(flat_eps)
        w_all = np.concatenate(flat_w)
        return gaussian_dos(eps_all, weights=w_all, **kwargs)

    # Single-k (Γ) path.
    eps = np.asarray(result.mo_energies, dtype=float).ravel()
    occ = getattr(result, "occupations", None)
    if occ is not None:
        w = np.asarray(occ, dtype=float).ravel()
        if w.shape != eps.shape:
            # Mismatch (e.g., result was hand-built with a stub
            # occupations); fall through to n_occ derivation.
            w = None
    else:
        w = None

    if w is None:
        n_occ = _infer_n_occ(result, system=system, n_elec=n_elec)
        w = np.zeros_like(eps)
        w[:n_occ] = 2.0

    return gaussian_dos(eps, weights=w, **kwargs)


def _infer_n_occ(result, *, system=None, n_elec: Optional[int] = None) -> int:
    """Best-effort recovery of the number of doubly-occupied orbitals.

    Tries (in order):

    * explicit ``n_elec``;
    * S Z from the user-supplied ``system``;
    * tr(D . S) when both ``result.density`` and ``result.overlap`` are
      present (the runner-shape wrapper, not the bare GPW result);
    * S Z from ``result.system`` if any;
    * ``result.occupations`` cell-count heuristic.

    Always returns a non-negative integer. Falls back to 0 if every
    channel fails -- the DOS will then carry zero electron weight,
    which is the most conservative failure mode.
    """
    if n_elec is not None:
        return max(int(n_elec) // 2, 0)
    if system is not None and hasattr(system, "unit_cell"):
        try:
            n = int(sum(int(a.Z) for a in system.unit_cell))
            return max(n // 2, 0)
        except (ValueError, TypeError, AttributeError):
            pass

    D = getattr(result, "density", None)
    S = getattr(result, "overlap", None)
    if D is not None and S is not None:
        try:
            D_arr = np.asarray(D, dtype=float)
            S_arr = np.asarray(S, dtype=float)
            n = int(round(float(np.einsum("ij,ij->", D_arr, S_arr))))
            return max(n // 2, 0)
        except (ValueError, TypeError):
            pass

    sys2 = getattr(result, "system", None)
    if sys2 is not None and hasattr(sys2, "unit_cell"):
        try:
            n = int(sum(int(a.Z) for a in sys2.unit_cell))
            return max(n // 2, 0)
        except (ValueError, TypeError, AttributeError):
            pass

    occ = getattr(result, "occupations", None)
    if occ is not None:
        try:
            occ_arr = np.asarray(occ, dtype=float).ravel()
            return int((occ_arr > 1e-6).sum())
        except (ValueError, TypeError):
            pass

    return 0


# ============================================================
# 3. HOMO / LUMO / gap
# ============================================================


def compute_homo_lumo(
    result,
    *,
    n_elec: Optional[int] = None,
    system=None,
) -> Tuple[float, float, float]:
    """Return ``(e_homo, e_lumo, gap)`` for a converged GPW SCF.

    For single-k results ``e_homo = e[n_occ - 1]`` and
    ``e_lumo = e[n_occ]``.

    For multi-k results ``e_homo = max_k e_k[n_occ - 1]`` and
    ``e_lumo = min_k e_k[n_occ]`` -- the standard band-structure
    convention (top of valence and bottom of conduction). ``gap =
    e_lumo - e_homo``; a negative gap signals a metal (the bands
    overlap across the BZ sample).

    Parameters
    ----------
    result
        :class:`GpwScfResult` or :class:`GpwMultiKScfResult`.
    n_elec
        Electron count override. When ``None`` the helper sums
        ``Z`` over ``result.system.unit_cell`` (single-k carries
        the system via the wrapper; multi-k via ``result.kmesh``
        + the cube path). Pass an explicit int for ECP cells
        where the formal electron count differs from S Z.

    Returns
    -------
    (e_homo, e_lumo, gap)
        All values in Hartree. ``gap = e_lumo - e_homo`` (signed).
    """
    if n_elec is None:
        n_elec = _infer_n_elec(result, system=system)
    if n_elec is None or n_elec <= 0:
        raise ValueError(
            "compute_homo_lumo: couldn't infer electron count from result. "
            "Pass n_elec= explicitly."
        )
    if n_elec % 2 != 0:
        raise ValueError(
            f"compute_homo_lumo: open-shell systems not supported "
            f"(n_elec={n_elec} is odd). Use the open-shell post-SCF "
            f"path once it lands."
        )
    n_occ = n_elec // 2

    if hasattr(result, "mo_energies_k") and result.mo_energies_k:
        # Multi-k: HOMO = max over k of e_k[n_occ-1]; LUMO = min over k.
        e_homo_per_k = []
        e_lumo_per_k = []
        for eps_k in result.mo_energies_k:
            eps_k = np.asarray(eps_k, dtype=float).ravel()
            if eps_k.size <= n_occ:
                raise ValueError(
                    f"compute_homo_lumo: n_occ={n_occ} >= n_basis "
                    f"({eps_k.size}); LUMO not in the basis."
                )
            e_homo_per_k.append(float(eps_k[n_occ - 1]))
            e_lumo_per_k.append(float(eps_k[n_occ]))
        e_homo = max(e_homo_per_k)
        e_lumo = min(e_lumo_per_k)
    else:
        eps = np.asarray(result.mo_energies, dtype=float).ravel()
        if eps.size <= n_occ:
            raise ValueError(
                f"compute_homo_lumo: n_occ={n_occ} >= n_basis "
                f"({eps.size}); LUMO not in the basis."
            )
        e_homo = float(eps[n_occ - 1])
        e_lumo = float(eps[n_occ])

    return e_homo, e_lumo, e_lumo - e_homo


def _infer_n_elec(result, *, system=None) -> Optional[int]:
    """Pull total electron count from the result + user-supplied system.

    Tries (in order): user-supplied ``system`` S Z; tr(D . S) when
    the runner-shape wrapper provides ``overlap`` on the result;
    S Z from a system attached to the result. Returns None if no
    channel works.
    """
    if system is not None and hasattr(system, "unit_cell"):
        try:
            return int(sum(int(a.Z) for a in system.unit_cell))
        except (ValueError, TypeError, AttributeError):
            pass

    D = getattr(result, "density", None)
    S = getattr(result, "overlap", None)
    if D is not None and S is not None:
        try:
            D_arr = np.asarray(D, dtype=float)
            S_arr = np.asarray(S, dtype=float)
            return int(round(float(np.einsum("ij,ij->", D_arr, S_arr))))
        except (ValueError, TypeError):
            pass

    sys2 = getattr(result, "system", None)
    if sys2 is not None and hasattr(sys2, "unit_cell"):
        try:
            return int(sum(int(a.Z) for a in sys2.unit_cell))
        except (ValueError, TypeError, AttributeError):
            pass

    return None


# ============================================================
# 4. Band-path eigenvalues at a fixed density
# ============================================================


def band_path_eigenvalues(
    system,
    basis,
    density_matrix: np.ndarray,
    k_path: Sequence,
    *,
    functional: Optional[str] = None,
    cutoff_ha: float = 300.0,
    v_ne_convention: str = "ewald",
) -> np.ndarray:
    """Per-k eigenvalues along an arbitrary k-path at a fixed density.

    Builds ``F(k) = T(k) + V_ne(k) + J + V_xc`` at every k along
    ``k_path`` using:

    * the lattice ``T_lat``, ``S_lat``, ``V_ne_lat`` builders +
      Bloch sum (same machinery as
      :func:`vibeqc.periodic_gapw_j.run_periodic_rks_gpw_multi_k`);
    * the GPW Hartree-J built once from the supplied density on
      the FFT grid (``J + V_xc`` are Γ-summed quantities that
      don't depend on k);
    * the GAPW V_xc projection when ``functional`` is given.

    Each per-k Fock is solved as a generalised eigenvalue problem
    ``F(k) C(k) = e(k) S(k) C(k)``; the eigenvalues are returned.

    > Density is **fixed** at the supplied ``density_matrix``; this
    > is a non-self-consistent post-SCF probe along the band path
    > (the standard band-structure recipe -- converge the SCF on a
    > MP mesh, then diagonalise at a denser k-path under the same
    > converged density).

    Parameters
    ----------
    system
        :class:`vibeqc._vibeqc_core.PeriodicSystem`.
    basis
        :class:`vibeqc.BasisSet`.
    density_matrix
        ``(n_basis, n_basis)`` AO density. Typically the
        ``result.density`` from a converged
        :func:`run_periodic_rks_gpw_multi_k`.
    k_path
        Sequence of 3-vectors in Cartesian bohr⁻¹ (the convention
        :func:`_core.bloch_sum` consumes). Pass a list of tuples,
        a list of arrays, or an ``(n_k, 3)`` ndarray.
    functional
        Optional XC functional name (libxc). When given, the same
        GAPW projection chain used by the multi-k SCF is applied;
        defaults to None (pure HF, no V_xc).
    cutoff_ha
        PW kinetic-energy cutoff for the FFT grid (Ha). Default
        300 Ha -- the standard GAPW default.
    v_ne_convention
        ``"ewald"`` (default) or ``"smeared_erfc"``. Matches the
        :func:`evaluate_gpw_energy` convention selector.

    Returns
    -------
    eigenvalues
        ``(n_k, n_basis)`` real array of eigenvalues per k-point,
        sorted ascending at each k (the standard ``eigh``
        ordering).

    Notes
    -----
    HF exchange is **not** included on the band path -- the
    multi-k SCF doesn't support per-k K builders (see
    :func:`run_periodic_rks_gpw_multi_k` docstring), and the
    band path inherits that restriction. Pure-DFT band
    structures are the supported scope; hybrid bands need the
    M4+ per-k K wiring.
    """
    from .periodic_gapw_grid import make_grid as _make_grid
    from .periodic_gapw_j import (
        _build_v_ne,
        _evaluate_xc_on_grid,
        _project_vxc_to_ao,
        collocate_density_on_grid,
        project_potential_to_ao,
    )
    from .bands import require_periodic_system
    from .periodic_v_ne import compute_nuclear_lattice_dispatch

    require_periodic_system(system, feature="band_path_eigenvalues")
    D = np.asarray(density_matrix, dtype=float)
    if D.ndim != 2 or D.shape != (basis.nbasis, basis.nbasis):
        raise ValueError(
            f"band_path_eigenvalues: density_matrix shape {D.shape} "
            f"doesn't match basis ({basis.nbasis} functions)"
        )

    # k-path normalisation. Accept (n_k, 3) array or sequence of 3-tuples.
    k_path_arr = np.asarray(k_path, dtype=float)
    if k_path_arr.ndim == 1 and k_path_arr.shape == (3,):
        k_path_arr = k_path_arr.reshape(1, 3)
    if k_path_arr.ndim != 2 or k_path_arr.shape[1] != 3:
        raise ValueError(
            f"band_path_eigenvalues: k_path must be a sequence of "
            f"3-vectors; got shape {k_path_arr.shape}"
        )

    # Build the FFT grid.
    grid = _make_grid(
        np.asarray(system.lattice, dtype=float),
        cutoff_ha=cutoff_ha,
    )

    # One-electron lattice integrals (built once).
    lat_opts = _core.LatticeSumOptions()
    T_lat = _core.compute_kinetic_lattice(basis, system, lat_opts)
    S_lat = _core.compute_overlap_lattice(basis, system, lat_opts)
    lat_opts_v = _core.LatticeSumOptions()
    lat_opts_v.coulomb_method = _core.CoulombMethod.EWALD_3D
    V_lat = compute_nuclear_lattice_dispatch(basis, system, lat_opts_v)

    # GAPW J + V_xc at the fixed density (Γ-summed quantities, same
    # at every k). The smearing path needs the grid threaded in.
    rho_grid = collocate_density_on_grid(basis, D, grid)
    V_H = _core.solve_poisson_coulomb(rho_grid, grid.lattice_bohr)
    J_ao = project_potential_to_ao(basis, V_H, grid)

    if functional is not None:
        func = _core.Functional(functional, 1)
        _e_xc, v_xc_grid, v_sigma_grid, grad_rho, v_tau_grid = _evaluate_xc_on_grid(
            rho_grid,
            grid,
            func,
        )
        V_xc = _project_vxc_to_ao(
            basis,
            v_xc_grid,
            grid,
            v_sigma_grid=v_sigma_grid,
            grad_rho=grad_rho,
        )
    else:
        V_xc = np.zeros_like(J_ao)

    # v_ne_convention selector mirrors the SCF entry's contract.
    # The "ewald" path doesn't actually use the k-independent
    # lattice V_lat we built above (that's a Γ-only convenience);
    # the per-k V_ne(k) is Bloch-summed from V_lat below.
    if v_ne_convention not in ("ewald", "smeared_erfc"):
        raise ValueError(
            f"band_path_eigenvalues: unknown v_ne_convention "
            f"{v_ne_convention!r}; valid: 'ewald', 'smeared_erfc'"
        )
    if v_ne_convention == "smeared_erfc":
        # The smeared-erfc V_ne is built Γ-only in our infrastructure;
        # for the band path we fall back to the Ewald lattice V_ne
        # (gauge-equivalent on neutral cells). Warn the caller.
        warnings.warn(
            "band_path_eigenvalues: 'smeared_erfc' V_ne is Γ-only; "
            "falling back to Ewald V_ne for per-k Bloch sum. "
            "Equivalent on neutral cells.",
            category=GAPWExperimentalWarning,
            stacklevel=2,
        )

    # Per-k diagonalisation loop.
    n_k = k_path_arr.shape[0]
    n_basis = basis.nbasis
    eigs = np.empty((n_k, n_basis), dtype=float)
    for ik in range(n_k):
        k = k_path_arr[ik]
        T_k = np.asarray(_core.bloch_sum(T_lat, k))
        S_k = np.asarray(_core.bloch_sum(S_lat, k))
        V_ne_k = np.asarray(_core.bloch_sum(V_lat, k))
        # Hermitise.
        T_k = 0.5 * (T_k + T_k.conj().T)
        S_k = 0.5 * (S_k + S_k.conj().T)
        V_ne_k = 0.5 * (V_ne_k + V_ne_k.conj().T)
        # J + V_xc are k-independent (built from Γ-summed r).
        F_k = T_k + V_ne_k + J_ao + V_xc
        F_k = 0.5 * (F_k + F_k.conj().T)
        # Canonical orthogonalisation via S^{-1/2}.
        s_eigs, U = np.linalg.eigh(S_k)
        X = U @ np.diag(1.0 / np.sqrt(np.maximum(s_eigs, 1e-12))) @ U.conj().T
        eps, _ = np.linalg.eigh(X.conj().T @ F_k @ X)
        eigs[ik, :] = np.real(eps)
    return eigs
