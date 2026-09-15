"""Generate MgO 2x2x2 inputs across a scan for xtb parity.

CORRECTION (2026-08-26): the "$lattice:" xyz comment line is
IGNORED by xtb (verified: bit-identical energy without it), and
xtb 6.7.0 aborts on genuine periodic GFN2 input ("Multipoles not
available with PBC"). Inputs produced here run as FREE MOLECULAR
CLUSTERS. See handovers/HANDOVER_SECCM_BULK3D.md.
"""
from __future__ import annotations

import numpy as np


def main() -> None:
    for a in (4.5, 4.4, 4.3, 4.212, 4.1, 4.0, 3.9):
        prim = [
            np.array([0.0, 0.5, 0.5]) * a,
            np.array([0.5, 0.0, 0.5]) * a,
            np.array([0.5, 0.5, 0.0]) * a,
        ]
        basis = [np.zeros(3), np.array([a / 2.0, 0.0, 0.0])]
        atoms: list[np.ndarray] = []
        syms: list[str] = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for site in range(2):
                        atoms.append(
                            i * prim[0] + j * prim[1] + k * prim[2] + basis[site]
                        )
                        syms.append("Mg" if site == 0 else "O")
        lattice = [2 * p for p in prim]
        lines = [
            "16",
            "$lattice: "
            + " ".join(f"{v:.10f}" for row in lattice for v in row),
        ]
        for s, p in zip(syms, atoms):
            lines.append(f"{s:2s} {p[0]:14.8f} {p[1]:14.8f} {p[2]:14.8f}")
        with open(f"/tmp/xtbtest/mgo_{a}.xyz", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    print("done")


if __name__ == "__main__":
    main()
