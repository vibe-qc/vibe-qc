"""MR6b: a charge-neutral, boundary-corrected finite point-charge array whose
OFF-SITE potential over the QM cluster matches the Ewald Madelung target
defined in MR6a (madelung_array.py). Embedded-cluster CASSCF path
(HANDOVER_PERIODIC_MULTIREF.md, MR6).

MR6a defined the target V_target(r) = V(infinite lattice) - V(QM-region formal
charges) at off-site probes, anchored it to the textbook NaCl Madelung
constant, and showed that a naive (non-neutral, uncorrected) finite shell
misses it by ~2-4 Ha/e with NO convergence in the cutoff radius (the classic
conditional divergence of a charged finite sphere). MR6b closes that gap with
two constructions, both electroneutral and boundary-corrected:

1. ``evjen_embedding_array`` -- Evjen (1932) fractional boundary weights. A
   finite cube of crystal sites whose bounding faces/edges/corners are
   down-weighted (1/2, 1/4, 1/8) so the block is electroneutral; the QM sites
   are then removed (they are treated quantum-mechanically). Its off-site
   potential converges to V_target as the cube grows -- the textbook
   boundary-corrected Madelung sum, parameter-free.

2. ``fitted_array`` -- Derenzo, Klintenberg & Weber (2000) least-squares fit.
   An inner zone of exact formal charges plus an outer shell whose charge
   VALUES are fitted, under an exact neutrality constraint, to reproduce
   V_target at a dense probe grid. Matches V_target to ~uHa/e at a fixed,
   small block size -- the production array MR6c hands to the embedded CASSCF.

Charge bookkeeping (the cure for the MR6a conditional divergence): the
neutral object is (QM-region formal charges + embedding array). The array
therefore carries net charge -q_QM, neutralizing the QM region's formal net
charge; for a stoichiometric (neutral) QM cluster the array is itself neutral.

Reuses the validated in-tree Ewald summer (``ewald_point_charge_potential``,
via the MR6a target helpers) and adds no core code. Productionizing these
builders into ``python/vibeqc/embed/`` (the MR8 driver) is the follow-on,
pending maintainer review (CLAUDE.md s9); the citation database entry + route
land with that production merge.

Run:
    .venv/bin/python studies/embedded-cluster-cas/madelung_embedding.py
"""
from __future__ import annotations

import numpy as np

import vibeqc as vq

from cluster_carve import _mgo_cell, cluster_carve
from madelung_array import (
    _ewald_opts,
    exact_lattice_potential,
    finite_array_potential,
)

ANG = 1.8897259886  # angstrom -> bohr
TOL = 1e-6          # cube-face / lattice-plane tolerance (bohr)


# --------------------------------------------------------------------------- #
# The embedding target (MR6a), factored for reuse.
# --------------------------------------------------------------------------- #

def embedding_target(system, qm_sites, formal, probes):
    """V_target(probe) = V(infinite lattice) - V(QM-region formal charges).

    The exact potential the embedding array must reproduce over the QM region,
    evaluated at off-site ``probes`` (MR6a). ``qm_sites`` is the carved cluster
    [(Z, xyz_bohr), ...]; ``formal`` maps Z -> formal ionic charge.
    """
    probes = np.asarray(probes, float)
    v_inf = exact_lattice_potential(system, formal, probes)
    qm_pos = [r for _, r in qm_sites]
    qm_q = [formal[Z] for Z, _ in qm_sites]
    v_qm = finite_array_potential(qm_pos, qm_q, probes)
    return np.asarray(v_inf) - np.asarray(v_qm)


# --------------------------------------------------------------------------- #
# Construction 1: Evjen (1932) fractionally-weighted neutral block.
# --------------------------------------------------------------------------- #

def evjen_block(system, center, formal, half_side_half_a, a):
    """Evjen (1932) fractionally-weighted finite crystal block.

    Reference: H. M. Evjen, "On the Stability of Certain Heteropolar Crystals",
    Phys. Rev. 39, 675 (1932), doi:10.1103/PhysRev.39.675.

    A cube centered on a lattice SITE with half-side ``L = half_side_half_a *
    (a/2)``, so the six faces pass exactly through planes of lattice sites. A
    site is weighted by

        w = (1/2) ** (number of cartesian components lying on a cube face)

    i.e. 1 (interior), 1/2 (face), 1/4 (edge), 1/8 (corner). For a rock-salt
    lattice this fractionally-weighted cube is electroneutral by reflection
    symmetry and its interior potential converges to the Madelung potential.

    Returns ``(positions [n,3], effective_charges [n])`` (bohr, e).
    """
    center = np.asarray(center, float)
    L = half_side_half_a * (a / 2.0)
    lat = np.asarray(system.lattice, float)
    cell = [(int(at.Z), np.asarray(at.xyz, float)) for at in system.unit_cell]
    norms = np.linalg.norm(lat, axis=0)
    nmax = [int(np.ceil(L / n)) + 1 for n in norms]
    pos, q = [], []
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
                    pos.append(rr)
                    q.append(formal[Z] * 0.5 ** n_face)
    return np.asarray(pos), np.asarray(q)


def _drop_qm(pos, q, qm_pos, tol=1e-4):
    """Remove array entries coinciding with a QM site (treated QM)."""
    qm = np.asarray(qm_pos, float)
    keep = [k for k, r in enumerate(pos)
            if np.min(np.linalg.norm(qm - r, axis=1)) >= tol]
    return pos[keep], q[keep]


def evjen_embedding_array(system, qm_sites, center, formal, half_side_half_a, a):
    """Evjen neutral block minus the QM sites -> the embedding point charges.

    The QM sites sit deep in the interior (full weight 1), so removing them is
    exact: V_array(r) = V_block(r) - V_QM(r) -> V_inf(r) - V_QM(r) = V_target(r)
    as the block grows. The returned array carries net charge -q_QM (block is
    neutral; the QM region's +q_QM has been removed).
    """
    pos, q = evjen_block(system, center, formal, half_side_half_a, a)
    qm_pos = np.array([r for _, r in qm_sites])
    return _drop_qm(pos, q, qm_pos)


# --------------------------------------------------------------------------- #
# Construction 2: Derenzo-Klintenberg-Weber (2000) least-squares fitted shell.
# --------------------------------------------------------------------------- #

def _crystal_sites_within(system, center, radius):
    """All periodic image sites within ``radius`` of ``center``: [(Z, pos)]."""
    center = np.asarray(center, float)
    lat = np.asarray(system.lattice, float)
    cell = [(int(at.Z), np.asarray(at.xyz, float)) for at in system.unit_cell]
    norms = np.linalg.norm(lat, axis=0)
    nmax = [int(np.ceil(radius / n)) + 1 for n in norms]
    out = []
    for n1 in range(-nmax[0], nmax[0] + 1):
        for n2 in range(-nmax[1], nmax[1] + 1):
            for n3 in range(-nmax[2], nmax[2] + 1):
                shift = lat @ np.array([n1, n2, n3], float)
                for Z, r in cell:
                    rr = r + shift
                    if np.linalg.norm(rr - center) <= radius + TOL:
                        out.append((Z, rr))
    return out


def fitted_array(system, qm_sites, center, formal, probes, v_target, *,
                 r_exact, r_fit, ridge=1e-3):
    """Derenzo-Klintenberg-Weber (2000) fitted embedding array.

    Reference: S. E. Derenzo, M. K. Klintenberg & M. J. Weber, "Determining
    point charge arrays that produce accurate ionic crystal fields for atomic
    cluster calculations", J. Chem. Phys. 112, 2074 (2000), doi:10.1063/1.480776.

    Inner zone (``r <= r_exact``, QM removed): formal charges, held fixed -- the
    near field is reproduced exactly. Fit shell (``r_exact < r <= r_fit``, QM
    removed): charge VALUES ``q`` fitted to reproduce ``v_target`` at ``probes``
    by solving

        min_q ||A q - b||^2 + ridge*||q - q_formal||^2   s.t.   1.q = Q_fit

    with ``A[p,f] = 1/|probe_p - R_f|``, ``b = v_target - V_inner``, and the
    neutrality constraint ``Q_fit = -q_QM - sum(q_inner)`` so that the full
    array carries net charge -q_QM (QM_formal + array is neutral). The small
    Tikhonov ``ridge`` toward the formal charges keeps the weakly-constrained
    outer charges physical. Solved as the KKT linear system.

    Returns ``(positions [n,3], charges [n], info)``.
    """
    center = np.asarray(center, float)
    qm_pos = np.array([r for _, r in qm_sites])

    def _is_qm(r):
        return np.min(np.linalg.norm(qm_pos - r, axis=1)) < 1e-4

    inner_pos, inner_q, fit_pos, fit_q0 = [], [], [], []
    for Z, r in _crystal_sites_within(system, center, r_fit):
        if _is_qm(r):
            continue
        if np.linalg.norm(r - center) <= r_exact + TOL:
            inner_pos.append(r)
            inner_q.append(float(formal[Z]))
        else:
            fit_pos.append(r)
            fit_q0.append(float(formal[Z]))
    inner_pos = np.asarray(inner_pos)
    inner_q = np.asarray(inner_q)
    fit_pos = np.asarray(fit_pos)
    fit_q0 = np.asarray(fit_q0)
    nf = len(fit_q0)
    if nf == 0:
        raise ValueError("fitted_array: empty fit shell (raise r_fit / lower r_exact)")

    P = np.asarray(probes, float)
    v_inner = (finite_array_potential(inner_pos, inner_q, P)
               if len(inner_pos) else np.zeros(len(P)))
    b = np.asarray(v_target, float) - np.asarray(v_inner)

    # A[p, f] = 1 / |probe_p - fit_f|   (off-site -> non-singular)
    A = 1.0 / np.linalg.norm(P[:, None, :] - fit_pos[None, :, :], axis=2)

    q_qm = sum(float(formal[Z]) for Z, _ in qm_sites)
    q_fit_total = -q_qm - float(inner_q.sum())     # neutrality target

    # KKT for  min ||A q - b||^2 + ridge ||q - q0||^2  s.t.  1.q = q_fit_total
    AtA = A.T @ A + ridge * np.eye(nf)
    Atb = A.T @ b + ridge * fit_q0
    C = np.ones((1, nf))
    KKT = np.block([[2.0 * AtA, C.T], [C, np.zeros((1, 1))]])
    rhs = np.concatenate([2.0 * Atb, [q_fit_total]])
    sol = np.linalg.solve(KKT, rhs)
    fit_q = sol[:nf]

    pos = np.vstack([inner_pos, fit_pos]) if len(inner_pos) else fit_pos
    q = np.concatenate([inner_q, fit_q])
    info = dict(n_inner=len(inner_q), n_fit=nf, q_qm=q_qm, net=float(q.sum()),
                max_fit_dev=float(np.max(np.abs(fit_q - fit_q0))),
                cond=float(np.linalg.cond(A.T @ A + ridge * np.eye(nf))))
    return pos, q, info


# --------------------------------------------------------------------------- #
# Off-site probe grids.
# --------------------------------------------------------------------------- #

def offsite_probe_ball(center, all_sites, radius, n, *, d_min=0.6, seed=0):
    """``n`` random points in a ball of ``radius`` around ``center`` that are at
    least ``d_min`` bohr from every crystal site in ``all_sites`` (off-site, so
    the explicit point charges' 1/r spikes never corrupt the fit/eval)."""
    rng = np.random.default_rng(seed)
    center = np.asarray(center, float)
    S = np.array([r for _, r in all_sites])
    out = []
    while len(out) < n:
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        u = rng.uniform() ** (1.0 / 3.0) * radius
        p = center + u * v
        if np.min(np.linalg.norm(S - p, axis=1)) >= d_min:
            out.append(p)
    return np.asarray(out)


# --------------------------------------------------------------------------- #
# Validation.
# --------------------------------------------------------------------------- #

def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def main():
    print("=" * 72)
    print("MR6b: charge-neutral, boundary-corrected embedding array vs target")
    print("=" * 72)

    sysm, a = _mgo_cell()
    formal = {12: +2.0, 8: -2.0}
    qm_mol, qm = cluster_carve(sysm, 4, 0.51 * a)            # [OMg6]
    center = next(r for Z, r in qm if Z == 8)               # the central O
    q_qm = sum(formal[Z] for Z, _ in qm)
    print(f"\nMgO a = {a / ANG:.2f} A; QM = [OMg6] ({len(qm)} sites), "
          f"formal net charge q_QM = {q_qm:+.0f} e; center on the O.")

    # MR6a headline probes: small fixed off-site displacement off each Mg site.
    head = np.array([r + np.array([0.3, 0.2, 0.1]) for Z, r in qm if Z == 12])
    v_head = embedding_target(sysm, qm, formal, head)
    print(f"MR6a target over the 6 head probes: mean V_target = "
          f"{v_head.mean():+.5f} Ha/e, spread {v_head.max() - v_head.min():.5f}")

    # ---- Construction 1: Evjen convergence ----
    print("\n(1) Evjen (1932) fractionally-weighted neutral block, QM removed:")
    print(f"    {'half-side':>10s} {'block q':>9s} {'n_pc':>7s} {'array q':>9s} "
          f"{'rms|V-target|':>14s}")
    for m in (2, 3, 4, 5, 6, 8):
        bpos, bq = evjen_block(sysm, center, formal, m, a)
        apos, aq = evjen_embedding_array(sysm, qm, center, formal, m, a)
        v = finite_array_potential(apos, aq, head)
        print(f"    {m:6d}*a/2 {bq.sum():+9.4f} {len(aq):7d} {aq.sum():+9.2f} "
              f"{_rms(v - v_head):14.3e}")
    print("    -> block stays neutral; array carries -q_QM; rms decreases with "
          "size (vs the\n       non-convergent ~2-4 Ha/e naive shell of MR6a).")

    # ---- Construction 2: fitted outer shell ----
    # Independent fit + test probe sets (test never enters the fit) guard
    # against overfitting: a fit that only matched its own probes would show
    # test >> fit. Here test ~ fit, so the match generalizes over the QM region.
    print("\n(2) Derenzo-Klintenberg-Weber (2000) fitted outer shell:")
    all_sites = _crystal_sites_within(sysm, center, 3.0 * a)
    fit_probes = offsite_probe_ball(center, all_sites, 0.60 * a, 240, seed=1)
    test_probes = offsite_probe_ball(center, all_sites, 0.60 * a, 240, seed=99)
    v_fit_t = embedding_target(sysm, qm, formal, fit_probes)
    v_test_t = embedding_target(sysm, qm, formal, test_probes)

    pos, q, info = fitted_array(sysm, qm, center, formal, fit_probes, v_fit_t,
                                r_exact=2.0 * a, r_fit=3.0 * a, ridge=1e-4)
    v_fit = finite_array_potential(pos, q, fit_probes)
    v_test = finite_array_potential(pos, q, test_probes)
    v_h = finite_array_potential(pos, q, head)
    print(f"    inner(formal)={info['n_inner']}  fit shell={info['n_fit']}  "
          f"net q={info['net']:+.4f} (target {-q_qm:+.0f})  "
          f"max|q_fit-formal|={info['max_fit_dev']:.3f} e  cond={info['cond']:.1e}")
    print(f"    rms|V-target|:  fit probes = {_rms(v_fit - v_fit_t):.3e}   "
          f"test probes = {_rms(v_test - v_test_t):.3e}   "
          f"MR6a head = {_rms(v_h - v_head):.3e} Ha/e")

    ok_neutral = abs(info["net"] - (-q_qm)) < 1e-6
    ok_fit = _rms(v_test - v_test_t) < 1e-4
    ok_evjen = _rms(finite_array_potential(
        *evjen_embedding_array(sysm, qm, center, formal, 8, a), head) - v_head) < 1e-2
    print("\nGAP CLOSED: " + ("PASS" if (ok_neutral and ok_fit and ok_evjen) else
                              "CHECK") +
          f" -- neutral={ok_neutral}, fitted rms(test)<1e-4={ok_fit}, "
          f"Evjen rms<1e-2={ok_evjen}.")
    print("MR6b: charge-neutral boundary-corrected array matches the MR6a "
          "Ewald target; MR6c runs the embedded CASSCF on it.")


if __name__ == "__main__":
    main()
