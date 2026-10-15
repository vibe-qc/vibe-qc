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
6. For each BCP, leave along plus and minus the eigenvector of the one
   positive Hessian eigenvalue and follow steepest ascent on ρ to the two
   nuclei the bond path joins.

The module is MPL 2.0 licensed.  QTAIM is an open method; the algorithm
follows the standard approach described in Bader (1990) and the MultiWFN
manual (Lu & Chen, J. Comput. Chem. 2012).

"""

from __future__ import annotations

import warnings
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
        ``(N, 3)`` array of points (Angstrom) tracing the bond path.  It runs
        from the nucleus of atom ``i``, through the bond critical point, to the
        nucleus of atom ``j``; the density falls monotonically from each
        nucleus down to the bond critical point.
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
    initial_step: float = 0.01,
) -> np.ndarray:
    """Trace half a bond path from a BCP up to a nucleus.

    ``start`` is a bond critical point, where ∇ρ vanishes, so the gradient
    there cannot say which way to go.  ``direction`` is a unit vector along
    the eigenvector of the positive Hessian eigenvalue, taken with either
    sign; the two signs give the two halves of the bond path.  The trace
    first moves ``initial_step`` off the BCP along ``direction`` and then
    follows steepest ascent on ρ in steps of ``step_size`` along ∇ρ,
    halving any step that fails to raise ρ, until it comes within
    ``capture_radius`` of a nucleus.  All lengths are in bohr.

    Returns
    -------
    (N, 3) array of positions in bohr.  Row 0 is ``start``.  If the trace
    reached a nucleus, the last row is that nucleus's position.  Otherwise
    (``max_steps`` used up, or the ascent stalled) the trace is returned as
    far as it got, and it is not a bond path.
    """
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    # A step longer than the capture sphere could hop over a nucleus.
    step_size = min(step_size, 0.5 * capture_radius)
    min_step = 1e-3 * step_size

    def _rho_grad(point: np.ndarray) -> tuple[float, np.ndarray]:
        rho, grad, _ = eval_fn(point[np.newaxis, :])
        return float(rho[0]), grad[0]

    current = start + initial_step * direction
    rho_cur, grad_cur = _rho_grad(current)
    trace = [start.copy(), current.copy()]
    step = step_size

    for _ in range(max_steps):
        dists = np.linalg.norm(atom_positions - current, axis=1)
        nearest = int(np.argmin(dists))
        if dists[nearest] < capture_radius:
            trace.append(atom_positions[nearest].copy())
            break

        gn = float(np.linalg.norm(grad_cur))
        if gn < 1e-12:
            break
        step_dir = grad_cur / gn
        while True:
            trial = current + step * step_dir
            rho_try, grad_try = _rho_grad(trial)
            if rho_try > rho_cur:
                break
            step *= 0.5
            if step < min_step:
                return np.array(trace, dtype=np.float64)
        current, rho_cur, grad_cur = trial, rho_try, grad_try
        trace.append(current.copy())
        step = min(2.0 * step, step_size)

    return np.array(trace, dtype=np.float64)


def _terminal_atom(trace: np.ndarray, atom_positions: np.ndarray) -> Optional[int]:
    """Index of the nucleus a :func:`_trace_bond_path` trace ended on, or
    ``None`` if it stopped short of one."""
    dists = np.linalg.norm(atom_positions - trace[-1], axis=1)
    nearest = int(np.argmin(dists))
    return nearest if dists[nearest] < 1e-8 else None


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

        # The density curves up along the bond, the one positive Hessian
        # eigenvalue, and down across it, so the bond path leaves the BCP
        # along plus and minus that eigenvector.
        _rho, _grad, hess = _eval(pos_bohr[np.newaxis, :])
        v3 = np.linalg.eigh(hess[0])[1][:, -1]

        halves = [
            _trace_bond_path(
                pos_bohr, sign * v3, _eval, atom_pos,
                step_size=grid_spacing * 0.5,
            )
            for sign in (1.0, -1.0)
        ]
        ends = [_terminal_atom(half, atom_pos) for half in halves]
        if None in ends or ends[0] == ends[1]:
            warnings.warn(
                f"QTAIM: the gradient paths from the bond critical point "
                f"assigned to atoms {i}-{j} do not reach two different nuclei "
                f"(they end on {ends}); no bond path was recorded for it.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        # Run the path from atom i to atom j.  The proximity rule behind
        # ``atom_pair`` can disagree with where the paths actually end; the
        # traced nuclei are the bond.
        if set(ends) == {i, j}:
            first, second = halves if ends[0] == i else halves[::-1]
        else:
            warnings.warn(
                f"QTAIM: the bond critical point assigned to atoms {i}-{j} "
                f"by proximity has gradient paths ending on atoms "
                f"{ends[0]} and {ends[1]}; the bond path joins those.",
                RuntimeWarning,
                stacklevel=2,
            )
            first, second = halves
            i, j = ends

        # Each half starts at the BCP: run one reversed, drop the repeat.
        full_path = np.vstack([first[::-1], second[1:]])

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
