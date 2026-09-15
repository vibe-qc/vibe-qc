"""compute-cluster lattice scan: fcc Cu 4x4x4 with the IID 130 metal recipe.

ewald_gamma=True (tblite-faithful image-summed shell gamma),
scc_mixer="newton", electronic_temperature=0.002 Ha.

Usage: run_cu444_ewald.py a1 a2 a3 ...  (lattice constants in Angstrom)
One line per point, flush=True; parabola minimum over the 3 lowest.
Exit 1 if any point fails.
"""

from __future__ import annotations

import sys

import numpy as np

import vibeqc
from vibeqc import Atom, Molecule
from vibeqc.semiempirical import run_gfn2_seccm
from vibeqc.semiempirical.seccm.topology import (
    bind_finite_group,
    build_seccm_topology,
)

BOHR = 1.8897259886
Z = 29  # Cu


def run_point(a_angstrom: float):
    prim = [
        np.array([0.0, 0.5, 0.5]) * a_angstrom,
        np.array([0.5, 0.0, 0.5]) * a_angstrom,
        np.array([0.5, 0.5, 0.0]) * a_angstrom,
    ]
    atoms = [
        i * prim[0] + j * prim[1] + k * prim[2]
        for i in range(4)
        for j in range(4)
        for k in range(4)
    ]
    translations = [4 * p for p in prim]
    mol = Molecule(
        [Atom(Z, (np.asarray(c) * BOHR).tolist()) for c in atoms],
        0,
        1,
    )
    topo = build_seccm_topology(
        [np.asarray(c) * BOHR for c in atoms],
        [np.asarray(t) * BOHR for t in translations],
        length_unit="bohr",
        geometry_quantum=1.0e-10,
    )
    topo = bind_finite_group(
        topo,
        primitive_vectors=[np.asarray(p) * BOHR for p in prim],
        replicas=(4, 4, 4),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    res = run_gfn2_seccm(
        mol,
        topo,
        max_iter=3600,
        ewald_gamma=True,
        scc_mixer="newton",
        electronic_temperature=0.002,
    )
    q = np.asarray(res.charges)
    dq = np.asarray(res.shell_charges)
    e_atom = res.energy * res.group_order / len(q)
    return (
        e_atom,
        res.n_iter,
        float(np.sqrt((q**2).mean())),
        float(np.sqrt((dq**2).mean())),
    )


def main() -> None:
    print(f"vibeqc {vibeqc.__version__}  python {sys.version.split()[0]}", flush=True)
    a_list = [float(a) for a in sys.argv[1:]]
    if len(a_list) < 3:
        raise SystemExit(2)
    print(
        f"{'a/A':>8} {'E/atom':>12} {'iter':>6} {'q_rms':>7} {'shell_rms':>9}",
        flush=True,
    )
    results: list[tuple[float, float]] = []
    failed = 0
    for a in a_list:
        try:
            e_atom, n_iter, q_rms, shell_rms = run_point(a)
        except RuntimeError as exc:
            print(f"{a:8.4f}  FAILED: {str(exc)[:60]}", flush=True)
            failed += 1
            continue
        print(
            f"{a:8.4f} {e_atom:12.6f} {n_iter:6d} {q_rms:7.3f} {shell_rms:9.3f}",
            flush=True,
        )
        results.append((a, e_atom))
    if len(results) < 3:
        raise SystemExit(1)
    order = sorted(results, key=lambda item: item[1])
    a_fit = np.array([item[0] for item in order[:3]])
    e_fit = np.array([item[1] for item in order[:3]])
    coeff = np.polyfit(a_fit, e_fit, 2)
    a_min = -coeff[1] / (2.0 * coeff[0])
    print(
        f"parabolic minimum: a = {a_min:.4f} A "
        f"(E/atom = {np.polyval(coeff, a_min):.6f})",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
