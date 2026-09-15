"""Crystallographically valid B1 rocksalt(100) and B2(001) slab probes.

The superseded fixtures behind IIDs 141, 150, and 155 were neither of
these structures: the alleged B2 cell aligned successive species planes
instead of shifting the body-centred sublattice, and the alleged B1
checkerboard alternated only along x. Their energy and SCC conclusions
are intentionally not retained here.

The B1 builder uses primitive surface translations (a/2,a/2,0) and
(-a/2,a/2,0), two unlike atoms in every neutral (100) plane, and swaps
the two sites in the next plane. Its three-dimensional periodic extension
has six unlike nearest neighbours. The B2 builder uses square a x a
planes and shifts the unlike plane by (a/2,a/2); its periodic extension
has eight unlike nearest neighbours.
"""

from __future__ import annotations

import numpy as np

from vibeqc import Atom, Molecule
from vibeqc.semiempirical import run_gfn2_seccm
from vibeqc.semiempirical.seccm.topology import (
    bind_finite_group,
    build_seccm_topology,
)

BOHR = 1.8897259886
A = 4.212


def _molecule(coords_ang, zs) -> Molecule:
    return Molecule(
        [
            Atom(z, (np.asarray(c) * BOHR).tolist())
            for z, c in zip(zs, coords_ang)
        ],
        0,
        1,
    )


def _topology(coords_ang, primitives, n):
    translations = [n * np.asarray(p) for p in primitives]
    topology = build_seccm_topology(
        [np.asarray(c) * BOHR for c in coords_ang],
        [np.asarray(t) * BOHR for t in translations],
        length_unit="bohr",
        geometry_quantum=1.0e-10,
    )
    return bind_finite_group(
        topology,
        primitive_vectors=[np.asarray(p) * BOHR for p in primitives],
        replicas=(n, n, 1),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )


def b2_001_slab(n_planes: int, n: int = 2):
    """B2/CsCl(001): single-species planes shifted in both x and y."""
    if n_planes % 2 != 0:
        raise ValueError("even plane count only (stoichiometric)")
    primitives = [
        np.array([A, 0.0, 0.0]),
        np.array([0.0, A, 0.0]),
    ]
    atoms: list[np.ndarray] = []
    zs: list[int] = []
    for k in range(n_planes):
        species = 12 if k % 2 == 0 else 8
        shift = np.array([
            A / 2.0 if k % 2 else 0.0,
            A / 2.0 if k % 2 else 0.0,
            k * A / 2.0,
        ])
        for i in range(n):
            for j in range(n):
                atoms.append(i * primitives[0] + j * primitives[1] + shift)
                zs.append(species)
    return _molecule(atoms, zs), _topology(atoms, primitives, n)


def rocksalt_100_slab(n_planes: int, n: int = 2):
    """B1 rocksalt(100): neutral Mg/O checkerboard planes."""
    if n_planes < 1:
        raise ValueError("plane count must be positive")
    primitives = [
        np.array([A / 2.0, A / 2.0, 0.0]),
        np.array([-A / 2.0, A / 2.0, 0.0]),
    ]
    unlike_offset = np.array([A / 2.0, 0.0, 0.0])
    atoms: list[np.ndarray] = []
    zs: list[int] = []
    for k in range(n_planes):
        z_offset = np.array([0.0, 0.0, k * A / 2.0])
        for i in range(n):
            for j in range(n):
                home = i * primitives[0] + j * primitives[1] + z_offset
                if k % 2 == 0:
                    atoms.extend((home, home + unlike_offset))
                    zs.extend((12, 8))
                else:
                    atoms.extend((home + unlike_offset, home))
                    zs.extend((12, 8))
    return _molecule(atoms, zs), _topology(atoms, primitives, n)


def main() -> None:
    print("== geometry check ==")
    for tag, mol, _ in [
        ("B2/CsCl(001)", *b2_001_slab(2)),
        ("B1 rocksalt(100)", *rocksalt_100_slab(2)),
    ]:
        n_at = len(mol.atoms)
        print(f"  {tag}: n_atoms={n_at}")

    print("== SECCM B1 rocksalt(100) verdicts (2-plane) ==")
    mol, topo = rocksalt_100_slab(2)
    n_at = len(mol.atoms)
    for kwargs in ({}, {"madelung": True}, {"ewald_gamma": True}):
        label = kwargs.get("madelung") and "madelung" or (
            kwargs.get("ewald_gamma") and "ewald_gamma" or "unembedded"
        )
        try:
            res = run_gfn2_seccm(mol, topo, max_iter=3000, **kwargs)
            q = np.asarray(res.charges)
            print(
                f"  {label}: converged={res.converged} "
                f"physical={res.physical_basin} n_iter={res.n_iter} "
                f"E/atom={res.energy * res.group_order / n_at:.6f} "
                f"q_rms={float(np.sqrt((q ** 2).mean())):.4f}"
            )
        except RuntimeError as exc:
            print(f"  {label}: RuntimeError: {str(exc)[:80]}")


if __name__ == "__main__":
    main()
