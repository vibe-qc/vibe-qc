"""Debug the corundum hex-cell WS topology validity failure."""

from __future__ import annotations

import numpy as np

from vibeqc.semiempirical.seccm.topology import build_seccm_topology

BOHR = 1.8897259886
Z_AL = 0.352
X_O = 0.306


def corundum_hex(a: float, c: float):
    a1 = np.array([a, 0.0, 0.0])
    a2 = np.array([-0.5 * a, 0.5 * np.sqrt(3.0) * a, 0.0])
    a3 = np.array([0.0, 0.0, c])
    al_sites = [
        np.array([0.0, 0.0, Z_AL]),
        np.array([0.0, 0.0, -Z_AL]),
        np.array([0.0, 0.0, Z_AL + 0.5]),
        np.array([0.0, 0.0, -Z_AL + 0.5]),
    ]
    o_base = [
        np.array([X_O, 0.0, 0.25]),
        np.array([0.0, X_O, 0.25]),
        np.array([-X_O, -X_O, 0.25]),
    ]
    o_sites = list(o_base) + [s + np.array([0.0, 0.0, 0.5]) for s in o_base]
    r_centering = [
        np.array([0.0, 0.0, 0.0]),
        np.array([2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
        np.array([1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0]),
    ]
    frac = [s + t for s in al_sites for t in r_centering] + [
        s + t for s in o_sites for t in r_centering
    ]
    coords = [f @ np.array([a1, a2, a3]) for f in frac]
    return coords, [a1, a2, a3]


def main() -> None:
    coords, translations = corundum_hex(4.761, 12.997)
    coords_bohr = [np.asarray(x) * BOHR for x in coords]
    trans_bohr = [np.asarray(t) * BOHR for t in translations]
    print(f"natoms = {len(coords)}")
    # Pair-distance histogram to check for overlapping sites.
    dists = []
    for i in range(len(coords_bohr)):
        for j in range(i + 1, len(coords_bohr)):
            dists.append(np.linalg.norm(coords_bohr[i] - coords_bohr[j]))
    dists = np.array(dists) / BOHR
    print(f"min pair distance = {dists.min():.6f} A")
    topology = build_seccm_topology(
        coords_bohr, trans_bohr, length_unit="bohr", geometry_quantum=1.0e-10
    )
    print(f"is_valid = {topology.is_valid(len(coords))}")
    for central in range(len(coords)):
        w = topology.total_weight(central)
        if abs(round(w) - (len(coords) - 1)) > 0.5:
            print(f"  atom {central}: total weight {w:.6f} (expected 29)")
            # Which origins are missing?
            origins = {img.origin for img in topology.cells[central]}
            missing = [o for o in range(len(coords)) if o not in origins]
            print(f"    missing origins: {missing}")
    diags = topology.diagnostics()
    for d in diags:
        print(f"  diag: {d.code} {d.message[:120]}")


if __name__ == "__main__":
    main()
