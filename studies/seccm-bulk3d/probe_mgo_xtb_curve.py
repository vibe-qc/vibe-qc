"""Run xtb across the MgO scan and print the parity table."""
from __future__ import annotations

import re
import subprocess

import numpy as np

from vibeqc import Atom, Molecule
from vibeqc.semiempirical.seccm.gfn2 import run_gfn2_seccm
from vibeqc.semiempirical.seccm.topology import (
    bind_finite_group,
    build_seccm_topology,
)

BOHR = 1.8897259886


def mgo_cell(a_angstrom: float):
    prim = [
        np.array([0.0, 0.5, 0.5]) * a_angstrom,
        np.array([0.5, 0.0, 0.5]) * a_angstrom,
        np.array([0.5, 0.5, 0.0]) * a_angstrom,
    ]
    basis = [np.zeros(3), np.array([a_angstrom / 2.0, 0.0, 0.0])]
    atoms, zs = [], []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for site in range(2):
                    atoms.append(
                        i * prim[0] + j * prim[1] + k * prim[2] + basis[site]
                    )
                    zs.append(12 if site == 0 else 8)
    translations = [2 * p for p in prim]
    return atoms, zs, translations, prim


def seccm_energy(a: float) -> tuple[float, float, float]:
    atoms, zs, translations, prim = mgo_cell(a)
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
        replicas=(2, 2, 2),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    r = run_gfn2_seccm(molecule, topology, ewald_gamma=True)
    q = np.asarray(r.charges)
    return r.energy / 2.0, np.sqrt((q**2).mean()), r.homo_lumo_gap * 27.2114


def xtb_energy(path: str) -> float:
    proc = subprocess.run(
        ["xtb", "input.xyz", "--gfn", "2", "--acc", "1.0", "--parallel", "1"],
        cwd=path,
        capture_output=True,
        text=True,
        timeout=600,
    )
    for line in proc.stdout.splitlines():
        m = re.search(r"total energy\s+(-?\d+\.\d+)", line)
        if m:
            return float(m.group(1)) / 16.0
    raise RuntimeError("no total energy in xtb output")


def main() -> None:
    print(f"{'a/A':>6} {'xtb E/atom':>12} {'SECCM E/atom':>13} "
          f"{'dq':>6} {'q_rms':>7} {'gap/eV':>8}")
    for a in (4.5, 4.4, 4.3, 4.212, 4.1, 4.0, 3.9):
        e_x = xtb_energy("/tmp/xtbtest")
        # regenerate the input at this a for xtb parity
        atoms, zs, translations, prim = mgo_cell(a)
        lines = ["16", "$lattice: " + " ".join(
            f"{v:.10f}" for row in [2 * p for p in prim] for v in row)]
        for s, c in zip(["Mg", "O"] * 8, atoms):
            lines.append(f"{s:2s} {c[0]:14.8f} {c[1]:14.8f} {c[2]:14.8f}")
        with open("/tmp/xtbtest/input.xyz", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        e_x = xtb_energy("/tmp/xtbtest")
        e_s, q_rms, gap = seccm_energy(a)
        print(f"{a:6.3f} {e_x:12.6f} {e_s:13.6f} {e_s - e_x:+8.3f} "
              f"{q_rms:7.3f} {gap:8.2f}")


if __name__ == "__main__":
    main()
