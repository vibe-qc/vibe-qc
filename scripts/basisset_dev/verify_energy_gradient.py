#!/usr/bin/env python3
"""compute-host-d verification of the Phase-0 SCF-energy-gradient real provider.

Validates ``vibeqc.basis_optimization.energy_gradient`` end-to-end on a
*built* vibe-qc: the analytic energy-weighted-density (Pulay) assembly fed
by the real ``VibeqcIntegralProvider`` (compute_overlap/kinetic/nuclear/eri
+ run_rhf density) must reproduce the full re-SCF total-energy finite
difference. Uses closed-shell H₂ (RHF) with the pob-TZVP H basis — a
genuine *multi-centre* test, which the same-centre analytic penalty kernel
does not cover.

The build-free assembly math is already covered by
``tests/basisset_dev/test_energy_gradient_assembly.py`` (mock RHF); this
script is the build-only half (run on compute-host-d/compute-host-a via vq, per CLAUDE.md § 15).

Verified 2026-06-17 on compute-host-a (compute-host-d was administratively down), vibe-qc
0.12.3.dev0: assembly vs full-energy central FD agreed to
    H diffuse-s exponent   |Δ| = 1.4e-10 Ha
    H contracted-s coeff   |Δ| = 1.5e-07 Ha

Usage (compute-host-d/compute-host-a, via vq):
    vq submit <host> --branch main scripts/basisset_dev/verify_energy_gradient.py
    <venv>/bin/python scripts/basisset_dev/verify_energy_gradient.py

Exit status is non-zero if any gradient component disagrees with the
full-energy finite difference beyond tolerance, so a vq terminal job
surfaces a regression.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def main() -> int:
    import vibeqc as vq
    from vibeqc.basis_crystal import parse_crystal_atom_basis_file
    from vibeqc.basis_optimization import VibeqcIntegralProvider, energy_gradient_fd
    from vibeqc.basis_optimization.io import TempBasisLibrary
    from vibeqc.basis_optimization.parametrise import (
        BasisParametrisation,
        FreeSpec,
        Transform,
    )

    # pob-TZVP H, with the worktree fallback used by the single_atom recipe.
    pkg_root = Path(vq.__file__).parent
    src_h = pkg_root / "basis_library" / "sources" / "pob-TZVP" / "01_H"
    if not src_h.exists():
        from vibeqc import basis_optimization as _bo

        root = Path(_bo.__file__).resolve().parents[3]
        src_h = root / "python" / "vibeqc" / "basis_library" / "sources" / "pob-TZVP" / "01_H"
    atoms = {"H": parse_crystal_atom_basis_file(src_h)}

    # Two free parameters: a valence exponent and a contraction coefficient,
    # so both derivative kinds go through the real integrals.
    parametrisation = BasisParametrisation(
        atoms=atoms,
        free=[
            FreeSpec("H", shell_idx=2, prim_idx=0, field="exponent",
                     transform=Transform.LOG, label="H_diffuse_s_exp"),
            FreeSpec("H", shell_idx=0, prim_idx=1, field="coeff",
                     transform=Transform.LINEAR, label="H_contracted_s_c1"),
        ],
    )

    # Closed-shell H2 (singlet) → RHF. ~1.4 bohr is near equilibrium.
    def molecule_factory():
        return vq.Molecule(
            [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
            multiplicity=1,
        )

    x0 = parametrisation.pack()
    delta = 1e-4   # integral-FD step inside the assembly
    h = 1e-4       # full-energy FD step for the reference
    tol = 5e-5     # Ha per unit parameter

    with TempBasisLibrary() as lib:
        provider = VibeqcIntegralProvider(
            parametrisation, molecule_factory, lib, vq.run_rhf
        )
        g_assembly = energy_gradient_fd(provider, x0, delta=delta)

        def e_total(x: np.ndarray) -> float:
            atoms_x = parametrisation.unpack(np.asarray(x, dtype=float))
            name = lib.write_g94(atoms_x, basis_name="egrad-ref")
            mol = molecule_factory()
            basis = vq.BasisSet(mol, name)
            return float(vq.run_rhf(mol, basis).energy)

        g_fd = np.zeros(len(x0))
        for i in range(len(x0)):
            xp = x0.copy(); xp[i] += h
            xm = x0.copy(); xm[i] -= h
            g_fd[i] = (e_total(xp) - e_total(xm)) / (2.0 * h)

    labels = parametrisation.labels()
    print("=" * 72)
    print("Phase-0 energy-gradient verification — H2 / pob-TZVP / RHF")
    print("=" * 72)
    print(f"{'param':<22} {'assembly':>16} {'energy-FD':>16} {'|Δ|':>12}")
    ok = True
    for i, lab in enumerate(labels):
        d = abs(g_assembly[i] - g_fd[i])
        flag = "" if d <= tol else "  <-- FAIL"
        ok = ok and d <= tol
        print(f"{lab:<22} {g_assembly[i]:>16.9f} {g_fd[i]:>16.9f} {d:>12.2e}{flag}")
    print("=" * 72)
    print("RESULT:", "PASS" if ok else "FAIL", f"(tol={tol:g} Ha)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
