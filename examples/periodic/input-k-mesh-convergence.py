"""K-mesh convergence demo — exercises every ``vibeqc.KPoints`` flavor.

Walks a small cubic Mg cell through:

  - a Γ-only sanity check
  - 2x2x2, 3x3x3, 4x4x4 Monkhorst-Pack meshes
  - the same 4x4x4 reduced to the IBZ via spglib symmetry
  - density-based KPPRA auto-mesh at three levels
  - a band-path sample (HPKOT via seekpath)

For each, runs ``run_rhf_periodic_scf`` and reports total energy +
mesh count. The IBZ-reduced energy should match the full mesh
bit-for-bit; the KPPRA scan shows monotone convergence.

Mg in a 3 Å cubic cell is small enough that everything finishes in a
few seconds while still exercising the full multi-k Ewald path.

Run with:
    python examples/periodic/input-k-mesh-convergence.py
"""

from __future__ import annotations

import time

import numpy as np

import vibeqc as vq
from vibeqc import CoulombMethod


ANGSTROM_TO_BOHR = 1.8897261339213


def make_cubic_mg(a_ang: float = 3.0) -> tuple[vq.PeriodicSystem, vq.BasisSet]:
    a = a_ang * ANGSTROM_TO_BOHR
    sys = vq.PeriodicSystem(3, np.eye(3) * a, [vq.Atom(12, [0, 0, 0])])
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    return sys, basis


def run_one(sys, basis, kpoints, *, label: str) -> float:
    """Run one SCF and report (energy, n_kpoints, time)."""
    opts = vq.PeriodicSCFOptions()
    opts.lattice_opts.coulomb_method = CoulombMethod.EWALD_3D
    opts.max_iter = 50
    opts.conv_tol_energy = 1e-8

    t0 = time.time()
    result = vq.run_rhf_periodic_scf(sys, basis, kpoints, opts)
    t = time.time() - t0
    n_k = len(kpoints) if hasattr(kpoints, "__len__") else len(kpoints.kpoints)
    flag = "✓" if getattr(result, "converged", False) else "✗"
    print(f"  {label:<35s}  n_k={n_k:>4d}   E = {result.energy:>14.8f} Ha   {flag}   {t:.2f}s")
    return float(result.energy)


def main() -> None:
    print("vibe-qc k-mesh convergence demo (cubic Mg, STO-3G, EWALD_3D)")
    print("=" * 72)
    sys, basis = make_cubic_mg(3.0)
    vq.attach_symmetry(sys, symprec=1e-4)
    print(f"  spacegroup: {sys.symmetry.international_symbol} "
          f"(SG {sys.symmetry.number}), point group "
          f"{sys.symmetry.point_group}")
    print()

    print("Γ-only and Monkhorst-Pack scan:")
    e_gamma = run_one(sys, basis, vq.KPoints.gamma(sys), label="Γ-only")
    for n in [2, 3, 4]:
        kp = vq.KPoints.monkhorst_pack(sys, [n, n, n])
        run_one(sys, basis, kp, label=f"MP {n}×{n}×{n}")

    print("\nSymmetry-reduced (IBZ via spglib):")
    full = vq.KPoints.monkhorst_pack(sys, [4, 4, 4])
    e_full = run_one(sys, basis, full, label="MP 4×4×4 (full mesh)")
    ibz = full.symmetry_reduce()
    e_ibz = run_one(sys, basis, ibz, label="MP 4×4×4 (IBZ-reduced)")
    print(f"  IBZ vs full energy diff: {abs(e_full - e_ibz):.2e} Ha "
          f"(should be ~1e-10 — symmetry-aware weights)")

    print("\nDensity-based auto-mesh (KPPRA):")
    for kppra in [100, 500, 1000]:
        kp = vq.KPoints.from_kppra(sys, kppra)
        run_one(sys, basis, kp, label=f"KPPRA={kppra} → mesh={kp.mesh}")

    print("\nBand-path sample (HPKOT, just first 10 k-points):")
    band = vq.KPoints.band_path(sys)
    print(f"  band path: {' → '.join(lbl for _, lbl in band.labels)}")
    print(f"  total k-points along path: {len(band)}")
    print(f"  example: first 3 k-points (Cartesian, bohr⁻¹):")
    for i, kc in enumerate(band.kpoints_cart[:3]):
        print(f"    k[{i}] = {kc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
