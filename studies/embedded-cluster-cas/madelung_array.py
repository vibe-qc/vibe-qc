"""MR6a: the exact Madelung potential, and why a naive finite charge
shell is not enough (the target + the gap that motivates MR6b).

Embedded-cluster CASSCF path (handovers/HANDOVER_PERIODIC_MULTIREF.md, MR6). The
embedding must make the QM cluster feel the crystal's Madelung
potential. This script:

1. Anchors the physics: reproduces the textbook Madelung constant from
   `ewald_point_charge_energy` (NaCl rocksalt, M = 1.7475645946).
2. Defines the embedding TARGET: the exact potential the QM region must
   feel = V(infinite lattice) - V(QM-region point charges), evaluated at
   OFF-SITE points in the cluster (non-singular), via the validated
   `ewald_point_charge_potential`.
3. Quantifies the GAP: the crude formal-charge shell from cluster_carve
   (Milestone 2) does NOT reproduce that target -- it is neither
   charge-neutral nor boundary-corrected. This is exactly what MR6b (an
   Evjen-weighted / fitted neutral array) must fix.

This is MR6a only: the exact target + the gap. MR6b (the boundary-
corrected finite array whose off-site potential matches the Ewald
target) and MR6c (the physical embedded CASSCF) are the next bricks.

Run:
    .venv/bin/python studies/embedded-cluster-cas/madelung_array.py
"""
from __future__ import annotations

import numpy as np

import vibeqc as vq
from vibeqc import EwaldOptions, ewald_point_charge_energy, \
    ewald_point_charge_potential

from cluster_carve import _mgo_cell, cluster_carve, formal_charge_shell

ANG = 1.8897259886


def _ewald_opts():
    o = EwaldOptions()
    o.real_cutoff_bohr = 40.0
    return o


def madelung_constant_nacl(a_ang=5.6):
    """Textbook anchor: M from the validated Ewald energy summer."""
    a = a_ang * ANG
    lattice = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]).T.astype(float)
    positions = np.column_stack([[0, 0, 0], 0.5 * a * np.array([1, 0, 0])])
    charges = np.array([+1.0, -1.0])
    r_nn = 0.5 * a
    energy = ewald_point_charge_energy(lattice, positions, charges, _ewald_opts())
    return -energy * r_nn, 1.7475645946


def exact_lattice_potential(system, charge_of_Z, eval_points):
    """Exact infinite-lattice Coulomb potential at eval_points (off-site),
    from the crystal's formal charges, via the validated Ewald summer."""
    lat = np.asarray(system.lattice, float)
    pos = np.column_stack([np.asarray(a.xyz, float) for a in system.unit_cell])
    q = np.array([charge_of_Z[int(a.Z)] for a in system.unit_cell])
    return np.asarray(ewald_point_charge_potential(
        lat, pos, q, np.asarray(eval_points, float), _ewald_opts(), True))


def finite_array_potential(pos, q, eval_points):
    """Plain Coulomb potential of a FINITE point-charge array (no images)."""
    pts = np.asarray(eval_points, float)
    out = np.zeros(len(pts))
    P = np.asarray(pos, float)
    Q = np.asarray(q, float)
    for k, r in enumerate(pts):
        d = np.linalg.norm(P - r, axis=1)
        out[k] = np.sum(Q / d)
    return out


def main():
    print("=" * 70)
    print("MR6a: exact Madelung potential (target) + the naive-shell gap")
    print("=" * 70)

    # 1. Textbook anchor.
    m, m_ref = madelung_constant_nacl()
    d = abs(m - m_ref)
    print(f"\n(1) NaCl Madelung constant via ewald_point_charge_energy: "
          f"{m:.10f}  (ref {m_ref:.10f}, |Δ|={d:.1e})  "
          f"{'PASS' if d < 1e-6 else 'FAIL'}")

    # 2. Embedding target over an MgO cluster region. QM = [OMg6] (carve
    #    centered on an O), formal Mg=+2 / O=-2. Evaluate at OFF-SITE
    #    points: small displacements off the 6 Mg sites (non-singular).
    sysm, a = _mgo_cell()
    formal = {12: +2.0, 8: -2.0}
    qm_mol, qm = cluster_carve(sysm, 4, 0.51 * a)        # [OMg6]
    o_site = next(r for Z, r in qm if Z == 8)
    probes = np.array([r + np.array([0.3, 0.2, 0.1])     # off-site probes
                       for Z, r in qm if Z == 12])
    v_inf = exact_lattice_potential(sysm, formal, probes)
    # QM-region point-charge contribution to remove (their own formal q):
    v_qm = finite_array_potential([r for _, r in qm],
                                  [formal[Z] for Z, _ in qm], probes)
    v_target = v_inf - v_qm                                # embedding target
    print(f"\n(2) Embedding target over [OMg6] (6 off-site probes), "
          f"V = V(inf lattice) - V(QM charges):")
    print(f"      mean V_target = {v_target.mean():+.5f} Ha/e, "
          f"spread = {v_target.max() - v_target.min():.5f}")

    # 3. The crude shell from Milestone 2 vs the target.
    for rpc in (1.5, 2.5, 3.5):
        pos, q = formal_charge_shell(sysm, qm, radius_pc=rpc * a, formal=formal)
        v_shell = finite_array_potential(pos, q, probes)
        err = v_shell - v_target
        print(f"      crude shell r={rpc:.1f}a: {len(q):4d} charges, "
              f"sum q={sum(q):+6.1f}; "
              f"mean|V-target|={np.abs(err).mean():.4f}, "
              f"rms={np.sqrt((err**2).mean()):.4f} Ha/e")

    print("\nGAP: the crude shell is non-neutral (sum q != 0) and "
          "uncorrected at the boundary, so its potential does NOT match the\n"
          "Ewald target. MR6b must build a charge-neutral, boundary-"
          "corrected (Evjen / fitted) array whose OFF-SITE potential over\n"
          "the cluster matches V_target; MR6c then runs the embedded CASSCF.")
    print("\nMR6a: target defined + anchored to the textbook constant; "
          "gap quantified.")


if __name__ == "__main__":
    main()
