"""B1.5 `cluster_carve` prototype: carve a finite cluster from a periodic
cell. The geometry foundation for the `embed_cluster` driver (MR8) and
the embedded-cluster CASSCF/CASPT2 path (handovers/HANDOVER_PERIODIC_MULTIREF.md).

Prototype in studies/ (research-first). Productionizing into
python/vibeqc/ as `vq.cluster_carve` (public export + tests + docs +
H-saturation for covalent hosts) is the follow-on, pending maintainer
review (CLAUDE.md s9).

`main()` validates the carve geometry on MgO rocksalt against the known
neighbour shells and smoke-tests the carve -> formal-point-charge embed
composition (building the embedded one-electron Hamiltonian from the
Milestone 1 enabler).

Run:
    .venv/bin/python studies/embedded-cluster-cas/cluster_carve.py
"""

from __future__ import annotations

import numpy as np
import vibeqc as vq

ANG = 1.8897259886  # angstrom -> bohr


def cluster_carve(system, center, radius, saturate=None):
    """Finite cluster carved from a periodic cell.

    Parameters
    ----------
    system : PeriodicSystem
        ``system.lattice`` is the 3x3 with columns a, b, c (bohr);
        ``system.unit_cell`` is the list of ``Atom`` (Z + cartesian bohr).
    center : int | (3,) array-like
        an atom index into ``unit_cell``, or a cartesian point (bohr).
    radius : float
        keep every periodic image atom within ``radius`` (bohr) of center.
    saturate : None | "H"
        ``None`` -> ionic carve (no capping). ``"H"`` -> cap dangling
        boundary bonds with hydrogen (covalent hosts) -- not yet
        implemented.

    Returns
    -------
    (Molecule, list[(int Z, ndarray xyz_bohr)])
        the finite cluster Molecule and its atom records.
    """
    lat = np.asarray(system.lattice, dtype=float)  # columns a, b, c
    cell = [(int(a.Z), np.asarray(a.xyz, dtype=float)) for a in system.unit_cell]
    if np.ndim(center) == 0:
        center = cell[int(center)][1].copy()
    center = np.asarray(center, dtype=float)

    # Enough cells in each lattice direction to cover the radius.
    norms = np.linalg.norm(lat, axis=0)
    nmax = [int(np.ceil(radius / n)) + 1 for n in norms]
    carved = []
    for n1 in range(-nmax[0], nmax[0] + 1):
        for n2 in range(-nmax[1], nmax[1] + 1):
            for n3 in range(-nmax[2], nmax[2] + 1):
                shift = lat @ np.array([n1, n2, n3], dtype=float)
                for Z, r in cell:
                    rr = r + shift
                    if np.linalg.norm(rr - center) <= radius + 1e-9:
                        carved.append((Z, rr))

    if saturate not in (None, "none"):
        raise NotImplementedError(
            "cluster_carve: only saturate=None (ionic, no capping) is "
            "implemented; H-capping for covalent hosts is a follow-on."
        )

    n_elec = sum(Z for Z, _ in carved)
    mult = 1 if n_elec % 2 == 0 else 2
    mol = vq.Molecule(
        [vq.Atom(Z, list(r)) for Z, r in carved], charge=0, multiplicity=mult
    )
    return mol, carved


def formal_charge_shell(system, qm_atoms, radius_pc, formal):
    """Point charges at lattice sites in a shell around the QM cluster.

    Every periodic image site within ``radius_pc`` of the cluster
    centroid that is NOT one of the QM atoms becomes a classical point
    charge with the element's formal charge (``formal`` maps Z -> q).
    A crude Madelung stand-in for the MR6 array (which will instead fit
    the charges to a periodic SCF density).
    """
    centroid = np.mean([r for _, r in qm_atoms], axis=0)
    qm_set = {(Z, tuple(np.round(r, 6))) for Z, r in qm_atoms}
    lat = np.asarray(system.lattice, dtype=float)
    cell = [(int(a.Z), np.asarray(a.xyz, dtype=float)) for a in system.unit_cell]
    norms = np.linalg.norm(lat, axis=0)
    nmax = [int(np.ceil(radius_pc / n)) + 1 for n in norms]
    pos, q = [], []
    for n1 in range(-nmax[0], nmax[0] + 1):
        for n2 in range(-nmax[1], nmax[1] + 1):
            for n3 in range(-nmax[2], nmax[2] + 1):
                shift = lat @ np.array([n1, n2, n3], dtype=float)
                for Z, r in cell:
                    rr = r + shift
                    if np.linalg.norm(rr - centroid) > radius_pc:
                        continue
                    if (Z, tuple(np.round(rr, 6))) in qm_set:
                        continue
                    pos.append(list(rr))
                    q.append(float(formal[Z]))
    return pos, q


def _mgo_cell(a_ang=4.21):
    """MgO rocksalt, conventional cubic cell (4 Mg + 4 O)."""
    a = a_ang * ANG
    lat = np.diag([a, a, a])
    frac = [
        (0, 0, 0),
        (0.5, 0.5, 0),
        (0.5, 0, 0.5),
        (0, 0.5, 0.5),  # Mg
        (0.5, 0, 0),
        (0, 0.5, 0),
        (0, 0, 0.5),
        (0.5, 0.5, 0.5),
    ]  # O
    Z = [12] * 4 + [8] * 4
    atoms = [vq.Atom(z, list(np.asarray(f) @ lat.T)) for z, f in zip(Z, frac)]
    return vq.PeriodicSystem(3, lat, atoms, charge=0, multiplicity=1), a


def main():
    print("=" * 68)
    print("B1.5 cluster_carve: validate geometry on MgO + carve->embed")
    print("=" * 68)
    sysm, a = _mgo_cell()
    # Center on an O site (index 4: fractional (1/2, 0, 0)).
    o_center = 4
    # Known rocksalt shells from an anion site (units of a):
    #   6 Mg @ 0.5a, 12 O @ a/sqrt2 (~0.707a), 8 Mg @ sqrt3/2 a (~0.866a),
    #   6 O @ a.
    shells = [(0.51, "OMg6", 1, 6), (0.72, "+12 O", 13, 6), (0.87, "+8 Mg", 13, 14)]
    print(f"\nMgO a = {a / ANG:.2f} A = {a:.4f} bohr; carve centered on an O\n")
    ok = True
    for freq, label, n_o, n_mg in shells:
        mol, carved = cluster_carve(sysm, o_center, freq * a)
        zc = [Z for Z, _ in carved]
        got_o, got_mg = zc.count(8), zc.count(12)
        hit = got_o == n_o and got_mg == n_mg
        ok &= hit
        print(
            f"  r = {freq:.2f} a ({freq * a:5.2f} bohr): {label:7s} -> "
            f"{got_o:2d} O + {got_mg:2d} Mg  (want {n_o} O + {n_mg} Mg)  "
            f"{'PASS' if hit else 'FAIL'}"
        )

    # First-shell Mg-O distance must be exactly a/2.
    mol, carved = cluster_carve(sysm, o_center, 0.51 * a)
    center = carved[0][1] if carved[0][0] == 8 else next(r for Z, r in carved if Z == 8)
    mg_d = sorted(np.linalg.norm(r - center) for Z, r in carved if Z == 12)
    d_ok = abs(mg_d[0] - a / 2) < 1e-9 and abs(mg_d[-1] - a / 2) < 1e-9
    ok &= d_ok
    print(
        f"\n  [OMg6] Mg-O distances all == a/2 = {a / 2:.4f} bohr: "
        f"{'PASS' if d_ok else 'FAIL'}  (got {mg_d[0]:.4f}..{mg_d[-1]:.4f})"
    )

    # Carve -> formal-charge embed composition (builds the embedded
    # one-electron Hamiltonian; physical ionic CASSCF is Milestone 3).
    print("\nCarve -> formal-charge embed composition (MR8 pipeline preview):")
    qm, carved = cluster_carve(sysm, o_center, 0.51 * a)  # [OMg6]
    pos, q = formal_charge_shell(
        sysm, carved, radius_pc=2.5 * a, formal={12: +2.0, 8: -2.0}
    )
    net_qm = sum({12: +2.0, 8: -2.0}[Z] for Z, _ in carved)
    print(
        f"  QM cluster: {len(carved)} atoms (formal net charge {net_qm:+.0f}); "
        f"embedding shell: {len(q)} point charges "
        f"(sum q = {sum(q):+.2f})."
    )
    from spike_external_field import embedded_pieces

    basis = vq.BasisSet(qm, "6-31g")
    S, Hcore, E_nuc, jk = embedded_pieces(qm, basis, pos, q)
    print(
        f"  embedded one-electron Hamiltonian built: S{S.shape}, "
        f"Hcore{Hcore.shape}, E_nuc(QM-QM + QM-pc) = {E_nuc:.4f} Ha  "
        f"-> carve+embed COMPOSES."
    )

    print(
        "\n"
        + (
            "B1.5 cluster_carve: PASS (geometry validated + "
            "composes with the embedding)."
            if ok
            else "SOME CHECK FAILED"
        )
    )


if __name__ == "__main__":
    main()
