"""Cluster carve: extract a finite QM cluster from a periodic crystal.

The geometry foundation for embedded-cluster CASSCF/CASPT2
(``vq.embed_cluster``).  Carves a finite cluster of atoms from a
``PeriodicSystem`` by replicating the unit cell and keeping every
image atom within a given radius of a centre site.

Productionised from ``studies/embedded-cluster-cas/cluster_carve.py``
(Milestone 2, B1.5).
"""

from __future__ import annotations

import numpy as np
import vibeqc as vq


def carve_cluster(
    system: vq.PeriodicSystem,
    center: int | np.ndarray,
    radius: float,
    saturate: str | None = None,
) -> tuple[vq.Molecule, list[tuple[int, np.ndarray]]]:
    """Carve a finite cluster from a periodic cell.

    Parameters
    ----------
    system : vq.PeriodicSystem
        ``system.lattice`` is the 3×3 matrix with columns *a*, *b*, *c*
        (bohr); ``system.unit_cell`` is the list of ``Atom`` carrying
        *Z* and cartesian position.
    center : int or (3,) array-like
        An atom index into ``unit_cell``, or a cartesian point (bohr).
    radius : float
        Keep every periodic image atom within ``radius`` (bohr) of
        *center*.
    saturate : None or ``"H"``, optional
        ``None`` (default) — ionic carve, no capping.
        ``"H"`` — cap dangling boundary bonds with hydrogen (covalent
        hosts); not yet implemented.

    Returns
    -------
    (vq.Molecule, list[(int Z, ndarray xyz_bohr)])
        The finite cluster molecule and its atom records.
    """
    lat = np.asarray(system.lattice, dtype=float)
    cell = [(int(a.Z), np.asarray(a.xyz, dtype=float)) for a in system.unit_cell]
    if np.ndim(center) == 0:
        center = cell[int(center)][1].copy()
    center = np.asarray(center, dtype=float)

    norms = np.linalg.norm(lat, axis=0)
    nmax = [int(np.ceil(radius / n)) + 1 for n in norms]
    carved: list[tuple[int, np.ndarray]] = []
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
            "carve_cluster: only saturate=None (ionic, no capping) is "
            "implemented; H-capping for covalent hosts is a follow-on."
        )

    n_elec = sum(Z for Z, _ in carved)
    mult = 1 if n_elec % 2 == 0 else 2
    mol = vq.Molecule(
        [vq.Atom(Z, list(r)) for Z, r in carved], charge=0, multiplicity=mult
    )
    return mol, carved
