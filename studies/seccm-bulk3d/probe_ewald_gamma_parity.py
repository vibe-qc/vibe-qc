"""Parity probe: GFN2-xTB-SECCM with ewald_gamma=True on the bulk-3D ladder.

Acceptance ladder from docs/design_seccm_gfn2_long_range_gamma.md:
1. eta->0 anchor vs bare-Coulomb Ewald (separate unit test);
2. MgO 2x2x2 keeps its minimum and charge parity (vs madelung=True and xtb);
3. corundum converges with Mulliken charges at the xtb scale;
4. fcc Cu 2x2x2 sane basin at a=3.615; compressed a=3.434 fails closed;
5. 2-D slabs use the complete Parry/de Leeuw kernel and remain in a sane
   charge basin; only nontrivial 1-D shell-gamma cells still fail closed.
"""

from __future__ import annotations

import numpy as np
from ase.spacegroup import crystal as ase_crystal

from vibeqc import Atom, Molecule
from vibeqc.semiempirical.seccm.gfn2 import run_gfn2_seccm
from vibeqc.semiempirical.seccm.topology import (
    bind_finite_group,
    build_seccm_topology,
)

BOHR = 1.8897259886
AL_Z = 0.35216
O_X = 0.30624


def rocksalt(a_angstrom: float, replicas=(2, 2, 2)):
    prim = [
        np.array([0.0, 0.5, 0.5]) * a_angstrom,
        np.array([0.5, 0.0, 0.5]) * a_angstrom,
        np.array([0.5, 0.5, 0.0]) * a_angstrom,
    ]
    basis = [np.zeros(3), np.array([a_angstrom / 2.0, 0.0, 0.0])]
    n1, n2, n3 = replicas
    atoms, zs = [], []
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                for site in range(2):
                    atoms.append(
                        i * prim[0] + j * prim[1] + k * prim[2] + basis[site]
                    )
                    zs.append(12 if site == 0 else 8)
    translations = [n1 * prim[0], n2 * prim[1], n3 * prim[2]]
    return atoms, zs, translations, prim, replicas


def corundum(a_angstrom: float, c_angstrom: float):
    atoms_ase = ase_crystal(
        ["Al", "O"],
        basis=[(0, 0, AL_Z), (O_X, 0, 0.25)],
        spacegroup=167,
        cellpar=[a_angstrom, a_angstrom, c_angstrom, 90, 90, 120],
    )
    cell_ang = atoms_ase.cell.array
    translations_ang = [row for row in cell_ang]
    coords_ang = [np.asarray(p) for p in atoms_ase.get_positions()]
    zs = [13 if s == "Al" else 8 for s in atoms_ase.get_chemical_symbols()]
    return coords_ang, zs, translations_ang


def fcc_cu(a_angstrom: float, replicas=(2, 2, 2)):
    prim = [
        np.array([0.0, 0.5, 0.5]) * a_angstrom,
        np.array([0.5, 0.0, 0.5]) * a_angstrom,
        np.array([0.5, 0.5, 0.0]) * a_angstrom,
    ]
    n1, n2, n3 = replicas
    atoms = [
        i * prim[0] + j * prim[1] + k * prim[2]
        for i in range(n1)
        for j in range(n2)
        for k in range(n3)
    ]
    zs = [29] * len(atoms)
    translations = [n1 * prim[0], n2 * prim[1], n3 * prim[2]]
    return atoms, zs, translations, prim, replicas


def _build(atoms, zs, translations, prim, replicas):
    molecule = Molecule(
        [Atom(z, (np.asarray(c) * BOHR).tolist()) for z, c in zip(zs, atoms)],
        0,
        1,
    )
    topology = build_seccm_topology(
        [np.asarray(c) * BOHR for c in atoms],
        [np.asarray(t) * BOHR for t in translations],
        length_unit="bohr",
        geometry_quantum=1.0e-10,
    )
    topology = bind_finite_group(
        topology,
        primitive_vectors=[np.asarray(p) * BOHR for p in prim],
        replicas=replicas,
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    return molecule, topology


def _report(label, r, n_atoms):
    # r.energy is per primitive cell; atoms per primitive cell =
    # n_atoms / group_order.
    per_atom = r.energy * r.group_order / n_atoms
    shell = np.asarray(r.shell_charges)
    print(
        f"{label:24s} E/atom={per_atom:12.6f} "
        f"gap={r.homo_lumo_gap * 27.2114:7.3f} eV iter={r.n_iter:5d} "
        f"q_rms={np.sqrt((np.asarray(r.charges) ** 2).mean()):6.3f} "
        f"dqsh_rms={np.sqrt((shell ** 2).mean()):6.3f} "
        f"band0={r.e_band0 * r.group_order / n_atoms:9.4f} "
        f"scc={r.e_scc * r.group_order / n_atoms:9.4f} "
        f"rep={r.e_repulsive * r.group_order / n_atoms:9.4f}"
    )


def main() -> None:
    print("== MgO 2x2x2, ewald_gamma=True ==")
    for a in (4.5, 4.4, 4.3, 4.212, 4.1, 4.0):
        atoms, zs, translations, prim, replicas = rocksalt(a)
        molecule, topology = _build(atoms, zs, translations, prim, replicas)
        try:
            r = run_gfn2_seccm(molecule, topology, ewald_gamma=True)
            _report(f"a={a}", r, len(atoms))
        except Exception as exc:  # noqa: BLE001
            print(f"a={a}: FAILED: {exc}")

    print("== MgO 2x2x2, madelung=True (reference) ==")
    atoms, zs, translations, prim, replicas = rocksalt(4.212)
    molecule, topology = _build(atoms, zs, translations, prim, replicas)
    r = run_gfn2_seccm(molecule, topology, madelung=True)
    _report("a=4.212", r, len(atoms))

    print("== corundum hex cell, ewald_gamma=True ==")
    atoms, zs, translations = corundum(4.7589, 12.991)
    molecule, topology = _build(atoms, zs, translations, translations, (1, 1, 1))
    try:
        r = run_gfn2_seccm(molecule, topology, ewald_gamma=True, max_iter=1200)
        _report("a=4.7589 c=12.991", r, len(atoms))
    except Exception as exc:  # noqa: BLE001
        print(f"corundum: FAILED: {exc}")

    print("== fcc Cu 2x2x2, ewald_gamma=True ==")
    for a in (3.615, 3.434):
        atoms, zs, translations, prim, replicas = fcc_cu(a)
        molecule, topology = _build(atoms, zs, translations, prim, replicas)
        try:
            r = run_gfn2_seccm(
                molecule, topology, ewald_gamma=True,
                electronic_temperature=0.005,
            )
            _report(f"a={a}", r, len(atoms))
        except Exception as exc:  # noqa: BLE001
            print(f"a={a}: FAILED: {exc}")

    print("== 2-D slab checks (Parry/Heyes KO kernel) ==")
    atoms, zs = [], []
    for k in range(2):
        z = k * 4.212 / 2.0
        for i in range(2):
            for j in range(2):
                atoms.append(np.array([i * 4.212, j * 4.212, z]))
                zs.append(12 if k % 2 == 0 else 8)
    translations = [
        np.array([2.0 * 4.212, 0.0, 0.0]),
        np.array([0.0, 2.0 * 4.212, 0.0]),
    ]
    prim = [
        np.array([4.212, 0.0, 0.0]),
        np.array([0.0, 4.212, 0.0]),
    ]
    molecule, topology = _build(atoms, zs, translations, prim, (2, 2, 1))
    try:
        r = run_gfn2_seccm(molecule, topology, ewald_gamma=True)
        _report("slab 2L", r, len(atoms))
    except Exception as exc:  # noqa: BLE001
        print(f"slab 2L: FAILED: {exc}")


if __name__ == "__main__":
    main()
