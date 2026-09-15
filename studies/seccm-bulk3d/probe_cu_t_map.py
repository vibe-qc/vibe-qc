"""Cu 2x2x2 faithful-kernel T/mixer map (IID 130): find the robust
production path to the sane basin with ewald_gamma=True."""

from __future__ import annotations

import sys

import numpy as np

from vibeqc import Atom, Molecule
from vibeqc.semiempirical.seccm.gfn2 import run_gfn2_seccm
from vibeqc.semiempirical.seccm.topology import (
    bind_finite_group,
    build_seccm_topology,
)

BOHR = 1.8897259886
Z = 29  # Cu


def fcc_primitive(a_angstrom: float):
    return [
        np.array([0.0, 0.5, 0.5]) * a_angstrom,
        np.array([0.5, 0.0, 0.5]) * a_angstrom,
        np.array([0.5, 0.5, 0.0]) * a_angstrom,
    ]


def build(a_angstrom, replicas):
    prim = fcc_primitive(a_angstrom)
    n1, n2, n3 = replicas
    atoms = [
        i * prim[0] + j * prim[1] + k * prim[2]
        for i in range(n1)
        for j in range(n2)
        for k in range(n3)
    ]
    translations = [n1 * prim[0], n2 * prim[1], n3 * prim[2]]
    molecule = Molecule(
        [Atom(Z, (np.asarray(c) * BOHR).tolist()) for c in atoms], 0, 1
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


def try_case(label, temperature, mixer, **extra):
    molecule, topology = build(3.615, (2, 2, 2))
    try:
        result = run_gfn2_seccm(
            molecule,
            topology,
            max_iter=3600,
            electronic_temperature=temperature,
            ewald_gamma=True,
            scc_mixer=mixer,
            **extra,
        )
        q = np.asarray(result.charges)
        dq = np.asarray(result.shell_charges)
        print(
            f"{label:40s} conv={result.converged} phys={result.physical_basin} "
            f"E/atom={result.energy:+.6f} gap={result.homo_lumo_gap:.6f} "
            f"iter={result.n_iter} q_rms={np.sqrt((q**2).mean()):.4f} "
            f"shell_rms={np.sqrt((dq**2).mean()):.4f}",
            flush=True,
        )
        return result
    except Exception as exc:  # noqa: BLE001
        print(f"{label:40s} FAILED: {exc}", flush=True)
        return None


def main():
    for temp in (0.005, 0.002, 0.001, 0.0005):
        for mixer in ("simple", "newton"):
            try_case(f"ewald T={temp} {mixer}", temp, mixer)
    # T-ladder: converge low, restart lower, confirm the sane branch
    # persists down to 300K-scale temperatures.
    print()
    prev = None
    for temp in (0.002, 0.001, 0.0005):
        molecule, topology = build(3.615, (2, 2, 2))
        label = f"ladder T={temp} simple"
        try:
            result = run_gfn2_seccm(
                molecule,
                topology,
                max_iter=3600,
                electronic_temperature=temp,
                ewald_gamma=True,
                scc_mixer="simple",
                initial_shell_charges=(
                    None
                    if prev is None
                    else np.asarray(prev.shell_charges, dtype=float)
                ),
            )
            q = np.asarray(result.charges)
            dq = np.asarray(result.shell_charges)
            print(
                f"{label:40s} "
                f"conv={result.converged} phys={result.physical_basin} "
                f"E/atom={result.energy:+.6f} gap={result.homo_lumo_gap:.6f} "
                f"iter={result.n_iter} q_rms={np.sqrt((q**2).mean()):.4f} "
                f"shell_rms={np.sqrt((dq**2).mean()):.4f}",
                flush=True,
            )
            prev = result
        except Exception as exc:  # noqa: BLE001
            print(f"{label:40s} FAILED: {exc}")


if __name__ == "__main__":
    sys.exit(main())
