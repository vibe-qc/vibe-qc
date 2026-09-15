"""Corundum a-ladder: SECCM mixers vs the out-of-process xtb reference.

The Newton mixer lands in a covalent fixed point (q_rms 0.165); the
design-doc-validated simple-mixing ewald_gamma state sits at xtb-scale
charges. This probe compares both mixers against xtb on the same
a-ladder (c = 12.99 fixed). xtb only via subprocess (rule 2).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from vibeqc import Atom, Molecule
from vibeqc.semiempirical import run_gfn2_seccm
from vibeqc.semiempirical.seccm.topology import (
    bind_finite_group,
    build_seccm_topology,
)

from xtb_corundum_reference import _find_xtb, corundum_positions

BOHR = 1.8897259886
Z_BY_SYMBOL = {"Al": 13, "O": 8}
C0 = 12.99


def run_seccm(a_angstrom, mixer, temperature=0.002):
    symbols, positions, lattice = corundum_positions(a_angstrom, C0)
    coords_ang = [np.asarray(p) for p in positions]
    lattice = [np.asarray(row) for row in lattice]
    zs = [Z_BY_SYMBOL[s] for s in symbols]
    mol = Molecule(
        [Atom(z, (c * BOHR).tolist()) for z, c in zip(zs, coords_ang)], 0, 1
    )
    topo = build_seccm_topology(
        [c * BOHR for c in coords_ang],
        [t * BOHR for t in lattice],
        length_unit="bohr",
        geometry_quantum=1.0e-10,
    )
    topo = bind_finite_group(
        topo,
        primitive_vectors=[t * BOHR for t in lattice],
        replicas=(1, 1, 1),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    result = run_gfn2_seccm(
        mol,
        topo,
        max_iter=3600,
        ewald_gamma=True,
        scc_mixer=mixer,
        electronic_temperature=temperature,
    )
    q = np.asarray(result.charges)
    return (
        result.energy * result.group_order / len(q),
        result.n_iter,
        float(np.sqrt((q**2).mean())),
    )


def run_xtb(a_angstrom) -> tuple[float, float]:
    symbols, positions, lattice = corundum_positions(a_angstrom, C0)
    n = len(symbols)
    lines = [
        str(n),
        "$lattice: " + " ".join(f"{v:.10f}" for row in lattice for v in row),
    ]
    for symbol, pos in zip(symbols, positions):
        lines.append(f"{symbol:2s} {pos[0]:14.8f} {pos[1]:14.8f} {pos[2]:14.8f}")
    xyz = "\n".join(lines) + "\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "input.xyz"
        path.write_text(xyz, encoding="utf-8")
        proc = subprocess.run(
            [_find_xtb(), "input.xyz", "--gfn", "2", "--acc", "1.0",
             "--parallel", "1"],
            cwd=tmp, capture_output=True, text=True, timeout=1800,
        )
        energy = None
        for line in proc.stdout.splitlines():
            if "TOTAL ENERGY" in line:
                for token in line.replace("|", " ").split():
                    try:
                        energy = float(token)
                        break
                    except ValueError:
                        continue
        if energy is None:
            raise RuntimeError("xtb total energy not found")
        return energy / n, energy / n


def main():
    print(
        f"{'a':>7} {'xtb':>10} {'T0-simple':>10} {'d_0':>8} "
        f"{'T.002-simple':>12} {'d_2':>8} {'q_0':>6} {'q_2':>6}"
    )
    for a in (4.55, 4.65, 4.759, 4.85, 4.95):
        e_xtb, _ = run_xtb(a)
        try:
            e_t0, it_0, q_0 = run_seccm(a, "simple", temperature=0.0)
            s_t0 = f"{e_t0:10.4f}"
            d_0 = f"{e_t0 - e_xtb:+8.4f}"
        except Exception as exc:  # noqa: BLE001
            s_t0, d_0, q_0 = "FAILED", "  -", float("nan")
        try:
            e_t2, it_2, q_2 = run_seccm(a, "simple", temperature=0.002)
            s_t2 = f"{e_t2:12.4f}"
            d_2 = f"{e_t2 - e_xtb:+8.4f}"
        except Exception as exc:  # noqa: BLE001
            s_t2, d_2, q_2 = "FAILED", "  -", float("nan")
        print(
            f"{a:7.3f} {e_xtb:10.4f} {s_t0} {d_0} {s_t2} {d_2} "
            f"{q_0:6.3f} {q_2:6.3f}",
            flush=True,
        )


if __name__ == "__main__":
    sys.exit(main())
