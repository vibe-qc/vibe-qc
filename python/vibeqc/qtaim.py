"""QTAIM — topological analysis of the electron density.

Critical-point search and bond-path tracing following Bader's *Atoms in
Molecules* (1990).  Uses analytic second derivatives from the
``evaluate_ao_with_hessian`` C++ binding.

Public API
----------

``qtaim_analysis(result, basis, molecule, ...)``
    Full QTAIM analysis: find critical points, classify by Hessian
    eigenvalue sign pattern, and trace bond paths. Returns a
    :class:`QTAIMResult`.

``qtaim_result_to_qvf(result: QTAIMResult) -> dict``
    Convert to the dict shape the QVF writer expects as
    ``context["qtaim_data"]``.

Algorithm sketch
----------------

1. Build a uniform grid around the molecule.
2. Evaluate ρ(r), ∇ρ(r), and H_ρ(r) via analytic C++ AO Hessian.
3. Find seeds: voxels where |∇ρ| is a local minimum (≤ all 26 neighbours).
4. Newton-Raphson refine each seed to a stationary point.
5. Classify by Hessian eigenvalues (3,-3) → ncp, (3,-1) → bcp, etc.
6. For each BCP, trace gradient paths to the two connected atoms.

The module is MPL 2.0 licensed.  QTAIM is an open method; the algorithm
follows the standard approach described in Bader (1990) and the MultiWFN
manual (Lu & Chen, J. Comput. Chem. 2012).

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ._vibeqc_core import BasisSet, Molecule, evaluate_ao_with_hessian
from .cube import CubeGrid, make_uniform_grid
from .spin_channels import spin_densities

# e/bohr³ → e/Å³  (1 bohr = 0.529177210903 Å)
_BOHR3_TO_ANG3 = 1.0 / (0.529177210903**3)
# e/bohr⁵ → e/Å⁵
_BOHR5_TO_ANG5 = 1.0 / (0.529177210903**5)
# bohr → Å
_BOHR_TO_ANG = 0.529177210903
_ANG_TO_BOHR = 1.0 / _BOHR_TO_ANG

# Upper-triangular Hessian component indices.
_H_IDX = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]

__all__ = [
    "BondPath",
    "CriticalPoint",
    "QTAIMResult",
    "qtaim_analysis",
    "qtaim_result_to_qvf",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class CriticalPoint:
    """One stationary point of the electron density.

    Attributes
    ----------
    type
        ``"ncp"`` (nuclear, (3,-3)), ``"bcp"`` (bond, (3,-1)),
        ``"rcp"`` (ring, (3,+1)), or ``"ccp"`` (cage, (3,+3)).
    position
        Cartesian coordinates in Angstrom.
    rho
        Electron density at the CP, e/Å³.
    laplacian
        Trace of the density Hessian, e/Å⁵.
    ellipticity
        Bond ellipticity ε = λ₁/λ₂ − 1; only meaningful for BCPs.
        ``None`` for other CP types.
    atom_pair
        Indices of the two atoms this BCP connects; ``None`` for non-BCPs.
    """

    type: str
    position: np.ndarray  # (3,) Angstrom
    rho: float
    laplacian: float
    ellipticity: Optional[float] = None
    atom_pair: Optional[tuple[int, int]] = None


@dataclass
class BondPath:
    """Gradient path connecting two atoms through a bond critical point.

    Attributes
    ----------
    atoms
        ``(i, j)`` — the two connected atom indices.
    path
        ``(N, 3)`` array of points (Angstrom) tracing the bond path.
    """

    atoms: tuple[int, int]
    path: np.ndarray  # (N, 3) Angstrom


@dataclass
class QTAIMResult:
    """Complete QTAIM topological analysis.

    Attributes
    ----------
    critical_points
        All stationary points found.
    bond_paths
        Gradient paths connecting atoms through BCPs.  Empty if no
        bond paths were traced (e.g. on a coarse grid).
    grid_spacing
        The grid spacing used, in bohr.
    """

    critical_points: list[CriticalPoint] = field(default_factory=list)
    bond_paths: list[BondPath] = field(default_factory=list)
    grid_spacing: float = 0.1


# ---------------------------------------------------------------------------
# Density + gradient + Hessian on a grid (Route A — C++ analytic Hessian)
# ---------------------------------------------------------------------------


def _density_hessian_on_grid(
    D: np.ndarray,
    basis: BasisSet,
    points: np.ndarray,  # (n_pts, 3) in bohr
    *,
    chunk_size: int = 100_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate ρ, ∇ρ, H_ρ on a set of points via analytic C++ Hessian.

    Returns
    -------
    rho : (n_pts,) float array in e/bohr³
    grad_rho : (n_pts, 3) float array in e/bohr⁴
    hess_rho : (n_pts, 3, 3) float array in e/bohr⁵
    """
    n_pts = points.shape[0]
    rho = np.empty(n_pts, dtype=np.float64)
    grad_rho = np.empty((n_pts, 3), dtype=np.float64)
    hess_rho = np.empty((n_pts, 3, 3), dtype=np.float64)

    for i in range(0, n_pts, chunk_size):
        block = points[i : i + chunk_size]
        m = block.shape[0]

        (
            chi_v,
            chi_gx,
            chi_gy,
            chi_gz,
            chi_hxx,
            chi_hxy,
            chi_hxz,
            chi_hyy,
            chi_hyz,
            chi_hzz,
        ) = evaluate_ao_with_hessian(basis, block)

        chi = np.asarray(chi_v, dtype=np.float64)
        dchi = np.stack([
            np.asarray(chi_gx, dtype=np.float64),
            np.asarray(chi_gy, dtype=np.float64),
            np.asarray(chi_gz, dtype=np.float64),
        ], axis=0)  # (3, m, n_ao)
        hao = [
            np.asarray(chi_hxx, dtype=np.float64),
            np.asarray(chi_hxy, dtype=np.float64),
            np.asarray(chi_hxz, dtype=np.float64),
            np.asarray(chi_hyy, dtype=np.float64),
            np.asarray(chi_hyz, dtype=np.float64),
            np.asarray(chi_hzz, dtype=np.float64),
        ]  # list of (m, n_ao)

        chiD = chi @ D  # (m, n_ao)

        # ρ = Σ μν χ_μ D_μν χ_ν
        rho[i : i + m] = np.einsum("mi,mi->m", chi, chiD, optimize=True)

        # ∂_c ρ = 2 Σ μν (∂_c χ_μ) D_μν χ_ν
        for c in range(3):
            grad_rho[i : i + m, c] = 2.0 * np.einsum(
                "mi,mi->m", dchi[c], chiD, optimize=True
            )

        # ∂_c∂_d ρ = 2 Σ μν [ (∂_c∂_d χ_μ) D_μν χ_ν + (∂_c χ_μ) D_μν (∂_d χ_ν) ]
        for k, (c, d) in enumerate(_H_IDX):
            term1 = np.einsum("mi,mi->m", hao[k], chiD, optimize=True)
            term2 = np.einsum("mi,mi->m", dchi[c], dchi[d] @ D, optimize=True)
            hess_rho[i : i + m, c, d] = 2.0 * (term1 + term2)

    # Symmetrize.
    hess_rho[:, 1, 0] = hess_rho[:, 0, 1]
    hess_rho[:, 2, 0] = hess_rho[:, 0, 2]
    hess_rho[:, 2, 1] = hess_rho[:, 1, 2]

    return rho, grad_rho, hess_rho


# ---------------------------------------------------------------------------
# Critical-point search
# ---------------------------------------------------------------------------


def _classify_cp(hess_eigenvalues: np.ndarray) -> str:
    """Classify a critical point by the sign pattern of its Hessian
    eigenvalues."""
    n_neg = int(np.sum(hess_eigenvalues < 0))
    if n_neg == 3:
        return "ncp"  # (3, -3) — nuclear attractor
    elif n_neg == 2:
        return "bcp"  # (3, -1) — bond critical point
    elif n_neg == 1:
        return "rcp"  # (3, +1) — ring critical point
    else:
        return "ccp"  # (3, +3) — cage critical point


def _find_critical_points(
    rho: np.ndarray,          # (nx, ny, nz)
    grad_rho: np.ndarray,     # (nx, ny, nz, 3)
    hess_rho: np.ndarray,     # (nx, ny, nz, 3, 3)
    grid: CubeGrid,
    *,
    eval_fn: object,  # callable(pts_bohr) -> (rho, grad, hess)
    gradient_tol: float = 0.2,
    cp_rho_floor: float = 1e-4,
    nr_max_iter: int = 30,
    nr_step_trust: float = 2.0,
    nr_convergence: float = 1e-7,
    dedup_tol: float = 0.2,
) -> list[tuple[str, np.ndarray, float, float, Optional[float]]]:
    """Search for critical points via local minima of |grad rho|.

    Seeds are voxels where |grad rho| is a local minimum (≤ all 26
    neighbours) and below ``gradient_tol``.  Each seed is refined via
    Newton-Raphson.

    Returns
    -------
    list of (type, position_bohr, rho, laplacian, ellipticity)
    """
    nx, ny, nz = grid.shape
    grad_norm = np.sqrt(np.sum(grad_rho**2, axis=3))

    # Local minima of |grad rho| via 3×3×3 minimum filter.
    from scipy.ndimage import minimum_filter

    gn_local = minimum_filter(grad_norm, size=3, mode="constant", cval=1e10)
    seed_mask = (
        (grad_norm < gradient_tol)
        & (grad_norm <= gn_local)
        & (rho > cp_rho_floor)
    )
    seed_ijk = np.argwhere(seed_mask)  # (n_seeds, 3) → (ix, iy, iz)

    origin = np.asarray(grid.origin, dtype=np.float64)
    spacing = np.asarray(grid.spacing, dtype=np.float64)

    raw_cps: list[tuple[str, np.ndarray, float, float, Optional[float]]] = []
    for ix, iy, iz in seed_ijk:
        r = origin + np.array([ix, iy, iz], dtype=np.float64) * spacing
        r0 = r.copy()
        converged = False
        for _ in range(nr_max_iter):
            _rh, gh, Hh = eval_fn(r[np.newaxis, :])
            g = gh[0]
            H = Hh[0]
            g_norm = float(np.linalg.norm(g))
            if g_norm < nr_convergence:
                converged = True
                break
            # Newton step.
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                step = g * 0.1
            # Accept only if it reduces |g|.
            r_try = r - step
            sn = float(np.linalg.norm(step))
            if sn > nr_step_trust:
                step *= nr_step_trust / sn
                r_try = r - step
            _rh2, gh2, _ = eval_fn(r_try[np.newaxis, :])
            if float(np.linalg.norm(gh2[0])) < g_norm:
                r = r_try.copy()
            else:
                r = r - g * 0.05
            # Trust-region.
            if float(np.linalg.norm(r - r0)) > nr_step_trust:
                r = r0 + (r - r0) * nr_step_trust / float(np.linalg.norm(r - r0))

        if converged:
            rh_f, _gh_f, Hh_f = eval_fn(r[np.newaxis, :])
            eigvals = np.linalg.eigvalsh(Hh_f[0])
            cp_type = _classify_cp(eigvals)
            rho_at = float(rh_f[0])
            if rho_at < cp_rho_floor:
                continue
            lap = float(np.trace(Hh_f[0]))
            ellip: Optional[float] = None
            # Bond ellipticity eps = lambda1/lambda2 - 1 with |lambda1| >= |lambda2|
            # (the two negative curvatures). eigvalsh returns ascending values, so
            # eigvals[0] is lambda1 (most negative) and eigvals[1] is lambda2; the
            # smaller-magnitude curvature eigvals[1] is the denominator and the
            # guard target. eps >= 0 by construction (was inverted -> eps in [-1,0]).
            if cp_type == "bcp" and abs(eigvals[1]) > 1e-14:
                ellip = float(abs(eigvals[0]) / abs(eigvals[1]) - 1.0)
            raw_cps.append((cp_type, r.copy(), rho_at, lap, ellip))

    # Deduplicate.
    deduped: list[tuple[str, np.ndarray, float, float, Optional[float]]] = []
    for cp in raw_cps:
        pos = cp[1]
        duplicate = any(
            np.linalg.norm(pos - d[1]) < dedup_tol for d in deduped
        )
        if not duplicate:
            deduped.append(cp)

    return deduped


# ---------------------------------------------------------------------------
# Bond-path tracing
# ---------------------------------------------------------------------------


def _trace_bond_path(
    start: np.ndarray,
    direction: np.ndarray,
    eval_fn: object,
    atom_positions: np.ndarray,
    *,
    step_size: float = 0.05,
    max_steps: int = 500,
    capture_radius: float = 0.3,
) -> np.ndarray:
    """Trace a gradient path from a BCP toward an atom.

    Follows steepest ascent on ρ by stepping along ∇ρ.

    Returns
    -------
    (N, 3) array of positions in bohr.
    """
    current = start.copy()
    trace = [start.copy()]

    for _ in range(max_steps):
        # Check atom capture.
        dists = np.linalg.norm(atom_positions - current, axis=1)
        nearest = int(np.argmin(dists))
        if dists[nearest] < capture_radius:
            trace.append(atom_positions[nearest].copy())
            break

        rh, gh, _ = eval_fn(current[np.newaxis, :])
        grad = gh[0]
        gn = float(np.linalg.norm(grad))
        if gn < 1e-12:
            break
        step_dir = grad / gn
        current = current + step_dir * step_size
        trace.append(current.copy())
    else:
        # Max steps — snap to nearest atom.
        dists = np.linalg.norm(atom_positions - current, axis=1)
        trace.append(atom_positions[int(np.argmin(dists))].copy())

    return np.array(trace, dtype=np.float64)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def qtaim_analysis(
    result,  # SCF result with .density or .density_alpha/.density_beta
    basis: BasisSet,
    molecule: Molecule,
    *,
    grid_spacing: float = 0.15,
    padding: float = 2.5,
    gradient_tol: float = 0.2,
    cp_rho_floor: float = 1e-4,
) -> QTAIMResult:
    """Topological analysis of the electron density.

    Computes critical points and bond paths of ρ(r) on a uniform grid
    using analytic C++ Hessians.

    Parameters
    ----------
    result
        Converged SCF result from ``run_rhf``, ``run_rks``, ``run_uhf``,
        or ``run_uks``. Must have ``.density`` (closed-shell) or
        ``.density_alpha`` + ``.density_beta`` (open-shell).
    basis
        The :class:`BasisSet` used for the SCF.
    molecule
        The :class:`Molecule` the SCF was run on.
    grid_spacing
        Voxel width in bohr.  Finer grids find more CPs but cost O(N³).
    padding
        Extra margin around the molecule in bohr.
    gradient_tol
        |∇ρ| upper bound for seed selection (local minima below this
        are refined).  Higher values (0.1–0.5) work well for 0.15 bohr
        grids.
    cp_rho_floor
        Minimum ρ at a CP; skip points in vacuum.

    Returns
    -------
    QTAIMResult
        Critical points (classified by Hessian sign pattern) and bond
        paths (gradient paths from BCPs to atoms).
    """
    # --- Density matrix ---------------------------------------------------
    alpha, beta = spin_densities(result)
    if alpha is not None:
        D = np.asarray(alpha, dtype=np.float64)
        D = D + np.asarray(beta, dtype=np.float64)
    else:
        D = np.asarray(result.density, dtype=np.float64)

    # --- Grid -------------------------------------------------------------
    grid = make_uniform_grid(molecule, spacing=grid_spacing, padding=padding)
    pts = grid.points()
    shape = grid.shape

    # --- ρ, ∇ρ, H_ρ on the grid (flat arrays) ----------------------------
    rho_flat, grad_flat, hess_flat = _density_hessian_on_grid(D, basis, pts)

    # Reshape to 3-D for local-minimum detection.
    rho_3d = rho_flat.reshape(shape)
    grad_3d = grad_flat.reshape(shape + (3,))
    hess_3d = hess_flat.reshape(shape + (3, 3))

    def _eval(pts_bohr):
        return _density_hessian_on_grid(D, basis, pts_bohr)

    # --- Critical points --------------------------------------------------
    raw_cps = _find_critical_points(
        rho_3d, grad_3d, hess_3d, grid,
        eval_fn=_eval,
        gradient_tol=gradient_tol,
        cp_rho_floor=cp_rho_floor,
    )

    # Atom positions in bohr.
    atom_pos = np.array([a.xyz for a in molecule.atoms], dtype=np.float64)

    # Filter grid NCPs: keep only those near actual atoms.
    atom_ncp_tol = 0.5  # bohr
    raw_cps = [
        cp for cp in raw_cps
        if cp[0] != "ncp"
        or np.min(np.linalg.norm(atom_pos - cp[1], axis=1)) < atom_ncp_tol
    ]

    # Insert NCPs at every atom position.
    existing_ncp_pos = np.array([
        cp[1] for cp in raw_cps if cp[0] == "ncp"
    ]).reshape(-1, 3) if any(cp[0] == "ncp" for cp in raw_cps) \
        else np.zeros((0, 3), dtype=np.float64)

    for i, pos_bohr in enumerate(atom_pos):
        if existing_ncp_pos.shape[0] > 0:
            if np.min(np.linalg.norm(existing_ncp_pos - pos_bohr, axis=1)) < 0.2:
                continue
        rh, _gh, Hh = _eval(pos_bohr[np.newaxis, :])
        rho_at = float(rh[0])
        lap = float(np.trace(Hh[0]))
        raw_cps.append(("ncp", pos_bohr.copy(), rho_at, lap, None))

    # --- Classify and assign atom pairs -----------------------------------
    critical_points: list[CriticalPoint] = []
    for cp_type, pos_bohr, rho_val, lap, ellip in raw_cps:
        pos_ang = pos_bohr * _BOHR_TO_ANG
        atom_pair = None
        if cp_type == "bcp":
            dists = np.linalg.norm(atom_pos - pos_bohr, axis=1)
            nearest = np.argsort(dists)[:2]
            atom_pair = (int(nearest[0]), int(nearest[1]))

        critical_points.append(
            CriticalPoint(
                type=cp_type,
                position=pos_ang,
                rho=rho_val * _BOHR3_TO_ANG3,
                laplacian=lap * _BOHR5_TO_ANG5,
                ellipticity=(
                    float(ellip) if ellip is not None and cp_type == "bcp" else None
                ),
                atom_pair=atom_pair,
            )
        )

    # --- Bond paths (BCPs only) -------------------------------------------
    bond_paths: list[BondPath] = []
    bcp_entries = [
        (cp, raw[1])
        for cp, raw in zip(critical_points, raw_cps)
        if cp.type == "bcp" and cp.atom_pair is not None
    ]
    for cp, pos_bohr in bcp_entries:
        i, j = cp.atom_pair  # type: ignore[misc]
        vec_i = atom_pos[i] - pos_bohr
        vec_j = atom_pos[j] - pos_bohr
        ni, nj = np.linalg.norm(vec_i), np.linalg.norm(vec_j)
        if ni < 1e-10 or nj < 1e-10:
            continue

        path_i = _trace_bond_path(
            pos_bohr, vec_i / ni, _eval, atom_pos,
            step_size=grid_spacing * 0.5,
        )
        path_j = _trace_bond_path(
            pos_bohr, vec_j / nj, _eval, atom_pos,
            step_size=grid_spacing * 0.5,
        )

        full_path = (
            np.vstack([path_j[::-1], path_i[1:]])
            if len(path_j) > 1 and len(path_i) > 1
            else np.vstack([path_j[::-1], path_i])
        )

        bond_paths.append(
            BondPath(
                atoms=(int(i), int(j)),
                path=full_path * _BOHR_TO_ANG,
            )
        )

    return QTAIMResult(
        critical_points=critical_points,
        bond_paths=bond_paths,
        grid_spacing=grid_spacing,
    )


def qtaim_result_to_qvf(result: QTAIMResult) -> dict:
    """Convert a :class:`QTAIMResult` to the dict shape the QVF writer
    expects as ``context["qtaim_data"]``.

    Usage::

        qtaim = qtaim_analysis(...)
        write_qvf(..., qtaim_data=qtaim_result_to_qvf(qtaim))
    """
    cp_list: list[dict] = []
    for cp in result.critical_points:
        d: dict = {
            "type": cp.type,
            "position": cp.position.tolist(),
            "rho": cp.rho,
            "laplacian": cp.laplacian,
        }
        if cp.ellipticity is not None:
            d["ellipticity"] = cp.ellipticity
        if cp.atom_pair is not None:
            d["atom_pair"] = list(cp.atom_pair)
        cp_list.append(d)

    bp_list: list[dict] = []
    for bp in result.bond_paths:
        bp_list.append({
            "atoms": list(bp.atoms),
            "path": bp.path.tolist(),
        })

    return {
        "critical_points": cp_list,
        "bond_paths": bp_list,
    }
