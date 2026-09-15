"""Madelung point-charge embedding arrays for embedded-cluster CAS.

MR6a (target) + MR6b (charge-neutral boundary-corrected arrays).
Two constructions, both electroneutral and boundary-corrected, both
reproducing the Ewald infinite-lattice Madelung potential across the
QM cluster region to sub-mHa/e:

1. ``evjen_array`` — Evjen (1932) fractional boundary weights
   (doi:10.1103/PhysRev.39.675).  Parameter-free; converges with
   block size.

2. ``fitted_array`` — Derenzo, Klintenberg & Weber (2000)
   least-squares fit (doi:10.1063/1.480776).  Compact, fixed-size;
   matched to ~1e-4 Ha/e rms on independent test probes.

Charge bookkeeping: the neutral object is (QM-region formal charges +
embedding array).  The array carries net charge *-q_QM*, so the
combined system is neutral (the cure for the conditional divergence
of a naive charged finite shell, MR6a).

Reuses the validated in-tree Ewald summer (``ewald_point_charge_energy``,
``ewald_point_charge_potential``).  Productionised from
``studies/embedded-cluster-cas/madelung_array.py`` +
``madelung_embedding.py``.
"""

from __future__ import annotations

import numpy as np
import vibeqc as vq
from vibeqc import EwaldOptions, ewald_point_charge_potential

ANG = 1.8897259886  # angstrom -> bohr
TOL = 1e-6  # cube-face / lattice-plane tolerance (bohr)


# --------------------------------------------------------------------------- #
# Ewald helpers (MR6a).
# --------------------------------------------------------------------------- #


def _ewald_opts() -> EwaldOptions:
    o = EwaldOptions()
    o.real_cutoff_bohr = 40.0
    return o


def exact_lattice_potential(
    system: vq.PeriodicSystem,
    charge_of_Z: dict[int, float],
    eval_points: np.ndarray,
) -> np.ndarray:
    """Exact infinite-lattice Coulomb potential at *eval_points* (off-site),
    from the crystal's formal charges, via the validated Ewald summer.
    """
    lat = np.asarray(system.lattice, float)
    pos = np.column_stack([np.asarray(a.xyz, float) for a in system.unit_cell])
    q = np.array([charge_of_Z[int(a.Z)] for a in system.unit_cell])
    return np.asarray(
        ewald_point_charge_potential(
            lat, pos, q, np.asarray(eval_points, float), _ewald_opts(), True
        )
    )


def finite_array_potential(
    pos: np.ndarray, q: np.ndarray, eval_points: np.ndarray
) -> np.ndarray:
    """Plain Coulomb potential of a FINITE point-charge array (no images)."""
    pts = np.asarray(eval_points, float)
    out = np.zeros(len(pts))
    P = np.asarray(pos, float)
    Q = np.asarray(q, float)
    for k, r in enumerate(pts):
        d = np.linalg.norm(P - r, axis=1)
        out[k] = float(np.sum(Q / d))
    return out


# --------------------------------------------------------------------------- #
# Embedding target (MR6a): V(inf lattice) - V(QM formal charges).
# --------------------------------------------------------------------------- #


def embedding_target(
    system: vq.PeriodicSystem,
    qm_sites: list[tuple[int, np.ndarray]],
    formal: dict[int, float],
    probes: np.ndarray,
) -> np.ndarray:
    r"""V_target(probe) = V(infinite lattice) − V(QM-region formal charges).

    The exact potential the embedding array must reproduce over the QM
    region, evaluated at off-site *probes* (MR6a).  *qm_sites* is the
    carved cluster ``[(Z, xyz_bohr), ...]``; *formal* maps *Z* →
    formal ionic charge.
    """
    probes = np.asarray(probes, float)
    v_inf = exact_lattice_potential(system, formal, probes)
    qm_pos = [r for _, r in qm_sites]
    qm_q = [formal[Z] for Z, _ in qm_sites]
    v_qm = finite_array_potential(np.asarray(qm_pos), np.asarray(qm_q), probes)
    return np.asarray(v_inf) - np.asarray(v_qm)


# --------------------------------------------------------------------------- #
# Evjen (1932) fractionally-weighted neutral block.
# --------------------------------------------------------------------------- #


def _evjen_block(
    system: vq.PeriodicSystem,
    center: np.ndarray,
    formal: dict[int, float],
    half_side_half_a: int,
    a: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evjen (1932) fractionally-weighted finite crystal block.

    Reference: H. M. Evjen, "On the Stability of Certain Heteropolar
    Crystals", Phys. Rev. 39, 675 (1932), doi:10.1103/PhysRev.39.675.

    A cube centred on a lattice SITE with half-side
    *L = half_side_half_a · (a/2)*, so the six faces pass exactly
    through planes of lattice sites.  A site is weighted by

        w = (1/2) ** (number of cartesian components lying on a cube face)

    i.e. 1 (interior), 1/2 (face), 1/4 (edge), 1/8 (corner).  For a
    rock-salt lattice this fractionally-weighted cube is electroneutral
    by reflection symmetry and its interior potential converges to the
    Madelung potential.

    Returns ``(positions [n,3], effective_charges [n])`` (bohr, e).
    """
    center = np.asarray(center, float)
    L = half_side_half_a * (a / 2.0)
    lat = np.asarray(system.lattice, float)
    cell = [(int(at.Z), np.asarray(at.xyz, float)) for at in system.unit_cell]
    norms = np.linalg.norm(lat, axis=0)
    nmax = [int(np.ceil(L / n)) + 1 for n in norms]
    pos_list: list[np.ndarray] = []
    q_list: list[float] = []
    for n1 in range(-nmax[0], nmax[0] + 1):
        for n2 in range(-nmax[1], nmax[1] + 1):
            for n3 in range(-nmax[2], nmax[2] + 1):
                shift = lat @ np.array([n1, n2, n3], float)
                for Z, r in cell:
                    rr = r + shift
                    d = rr - center
                    if np.max(np.abs(d)) > L + TOL:
                        continue
                    n_face = int(np.sum(np.abs(np.abs(d) - L) < TOL))
                    pos_list.append(rr)
                    q_list.append(formal[Z] * 0.5**n_face)
    return np.asarray(pos_list), np.asarray(q_list)


def _drop_qm(
    pos: np.ndarray, q: np.ndarray, qm_pos: np.ndarray, tol: float = 1e-4
) -> tuple[np.ndarray, np.ndarray]:
    """Remove array entries coinciding with a QM site (treated QM)."""
    qm = np.asarray(qm_pos, float)
    keep = [
        k for k, r in enumerate(pos) if np.min(np.linalg.norm(qm - r, axis=1)) >= tol
    ]
    return pos[keep], q[keep]


def evjen_array(
    system: vq.PeriodicSystem,
    qm_sites: list[tuple[int, np.ndarray]],
    center: np.ndarray,
    formal: dict[int, float],
    half_side_half_a: int,
    a: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evjen neutral block minus the QM sites → the embedding point charges.

    The QM sites sit deep in the interior (full weight 1), so removing
    them is exact: V_array(r) = V_block(r) − V_QM(r) → V_inf(r) −
    V_QM(r) = V_target(r) as the block grows.  The returned array
    carries net charge *-q_QM* (block is neutral; the QM region's
    +q_QM has been removed).
    """
    pos, q = _evjen_block(system, center, formal, half_side_half_a, a)
    qm_pos = np.array([r for _, r in qm_sites])
    return _drop_qm(pos, q, qm_pos)


# --------------------------------------------------------------------------- #
# Crystal site enumeration.
# --------------------------------------------------------------------------- #


def _crystal_sites_within(
    system: vq.PeriodicSystem, center: np.ndarray, radius: float
) -> list[tuple[int, np.ndarray]]:
    """All periodic image sites within *radius* of *center*: ``[(Z, pos)]``."""
    center = np.asarray(center, float)
    lat = np.asarray(system.lattice, float)
    cell = [(int(at.Z), np.asarray(at.xyz, float)) for at in system.unit_cell]
    norms = np.linalg.norm(lat, axis=0)
    nmax = [int(np.ceil(radius / n)) + 1 for n in norms]
    out: list[tuple[int, np.ndarray]] = []
    for n1 in range(-nmax[0], nmax[0] + 1):
        for n2 in range(-nmax[1], nmax[1] + 1):
            for n3 in range(-nmax[2], nmax[2] + 1):
                shift = lat @ np.array([n1, n2, n3], float)
                for Z, r in cell:
                    rr = r + shift
                    if np.linalg.norm(rr - center) <= radius + TOL:
                        out.append((Z, rr))
    return out


# --------------------------------------------------------------------------- #
# Derenzo-Klintenberg-Weber (2000) least-squares fitted shell.
# --------------------------------------------------------------------------- #


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def fitted_array(
    system: vq.PeriodicSystem,
    qm_sites: list[tuple[int, np.ndarray]],
    center: np.ndarray,
    formal: dict[int, float],
    probes: np.ndarray,
    v_target: np.ndarray,
    *,
    r_exact: float,
    r_fit: float,
    ridge: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Derenzo-Klintenberg-Weber (2000) fitted embedding array.

    Reference: S. E. Derenzo, M. K. Klintenberg & M. J. Weber,
    "Determining point charge arrays that produce accurate ionic
    crystal fields for atomic cluster calculations",
    J. Chem. Phys. 112, 2074 (2000), doi:10.1063/1.480776.

    Inner zone (*r ≤ r_exact*, QM removed): formal charges, held
    fixed — the near field is reproduced exactly.  Fit shell
    (*r_exact < r ≤ r_fit*, QM removed): charge VALUES *q* fitted
    to reproduce *v_target* at *probes* by solving

        min_q ‖A q − b‖² + ridge·‖q − q_formal‖²   s.t.   1·q = Q_fit

    with A[p,f] = 1/|probe_p − R_f|, b = v_target − V_inner, and the
    neutrality constraint Q_fit = −q_QM − sum(q_inner) so the full
    array carries net charge −q_QM (QM_formal + array is neutral).
    The small Tikhonov *ridge* toward the formal charges keeps the
    weakly-constrained outer charges physical.  Solved as the KKT
    linear system.

    Returns ``(positions [n,3], charges [n], info)``.
    """
    center = np.asarray(center, float)
    qm_pos = np.array([r for _, r in qm_sites])

    def _is_qm(r: np.ndarray) -> bool:
        return bool(np.min(np.linalg.norm(qm_pos - r, axis=1)) < 1e-4)

    inner_pos: list[np.ndarray] = []
    inner_q: list[float] = []
    fit_pos: list[np.ndarray] = []
    fit_q0: list[float] = []
    for Z, r in _crystal_sites_within(system, center, r_fit):
        if _is_qm(r):
            continue
        if np.linalg.norm(r - center) <= r_exact + TOL:
            inner_pos.append(r)
            inner_q.append(float(formal[Z]))
        else:
            fit_pos.append(r)
            fit_q0.append(float(formal[Z]))
    inner_pos_a = np.asarray(inner_pos)
    inner_q_a = np.asarray(inner_q)
    fit_pos_a = np.asarray(fit_pos)
    fit_q0_a = np.asarray(fit_q0)
    nf = len(fit_q0_a)
    if nf == 0:
        raise ValueError("fitted_array: empty fit shell (raise r_fit / lower r_exact)")

    P = np.asarray(probes, float)
    v_inner = (
        finite_array_potential(inner_pos_a, inner_q_a, P)
        if len(inner_pos_a)
        else np.zeros(len(P))
    )
    b = np.asarray(v_target, float) - np.asarray(v_inner)

    # A[p, f] = 1 / |probe_p − fit_f|   (off-site → non-singular)
    A = 1.0 / np.linalg.norm(P[:, None, :] - fit_pos_a[None, :, :], axis=2)

    q_qm = sum(float(formal[Z]) for Z, _ in qm_sites)
    q_fit_total = -q_qm - float(inner_q_a.sum())  # neutrality target

    # KKT for  min ‖A q − b‖² + ridge ‖q − q0‖²  s.t.  1·q = q_fit_total
    AtA = A.T @ A + ridge * np.eye(nf)
    Atb = A.T @ b + ridge * fit_q0_a
    C = np.ones((1, nf))
    KKT = np.block([[2.0 * AtA, C.T], [C, np.zeros((1, 1))]])
    rhs = np.concatenate([2.0 * Atb, [q_fit_total]])
    sol = np.linalg.solve(KKT, rhs)
    fit_q = sol[:nf]

    if len(inner_pos_a):
        pos = np.vstack([inner_pos_a, fit_pos_a])
        q = np.concatenate([inner_q_a, fit_q])
    else:
        pos = fit_pos_a
        q = fit_q

    info: dict = dict(
        n_inner=len(inner_q),
        n_fit=nf,
        q_qm=q_qm,
        net=float(q.sum()),
        max_fit_dev=float(np.max(np.abs(fit_q - fit_q0_a))),
        cond=float(np.linalg.cond(A.T @ A + ridge * np.eye(nf))),
    )
    return pos, q, info


# --------------------------------------------------------------------------- #
# Off-site probe grids.
# --------------------------------------------------------------------------- #


def offsite_probe_ball(
    center: np.ndarray,
    all_sites: list[tuple[int, np.ndarray]],
    radius: float,
    n: int,
    *,
    d_min: float = 0.6,
    seed: int = 0,
) -> np.ndarray:
    """*n* random points in a ball of *radius* around *center* that
    are at least *d_min* bohr from every crystal site in *all_sites*
    (off-site, so the explicit point charges' 1/r spikes never corrupt
    the fit or evaluation).
    """
    rng = np.random.default_rng(seed)
    center = np.asarray(center, float)
    S = np.array([r for _, r in all_sites])
    out: list[np.ndarray] = []
    while len(out) < n:
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        u = rng.uniform() ** (1.0 / 3.0) * radius
        p = center + u * v
        if np.min(np.linalg.norm(S - p, axis=1)) >= d_min:
            out.append(p)
    return np.asarray(out)
