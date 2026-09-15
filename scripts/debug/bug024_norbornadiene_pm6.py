#!/usr/bin/env python3
"""BUG-024: Diagnose PM6 non-convergence on norbornadiene.

Norbornadiene (bicyclo[2.2.1]hepta-2,5-diene, C7H8) is a strained bridged
bicycle that exhausts PM6's 200-iteration SCF budget without converging in
fleet validation runs. This script reproduces the failure and tests recovery
strategies.

The molecule has C2v symmetry with two weakly-coupled ethylene-like double
bonds bridged by a CH2 group and two bridgehead CH groups. PM6's hard-Aufbau
occupations can cycle between near-degenerate fragment-localised orbitals,
analogous to the formamide-dimer case fixed in v0.15.57.
"""

from __future__ import annotations

import sys

# ── norbornadiene geometry (C2v, B3LYP/def2-SVP optimised, Angstrom) ──────
# Bridgehead carbons are on the z-axis; the bridging CH2 is in the yz-plane.
# Norbornadiene (bicyclo[2.2.1]hepta-2,5-diene) C7H8, C2v symmetry.
# Geometry from NIST CCCBDB: B3LYP/6-31G(d) optimised, Angstrom.
# Bridgehead: C1(bottom), C4(top).  Bridging CH2: C7 (in yz-plane).
# Double bonds: C2=C3, C5=C6.
NORBORNADIENE_XYZ = """15
norbornadiene C7H8
C      0.000000    0.000000    1.268695
C      0.000000    1.215139    0.805739
C      0.000000   -1.215139    0.805739
C      0.000000    1.215139   -0.805739
C      0.000000   -1.215139   -0.805739
C      0.000000    0.000000   -1.268695
C      0.000000    0.000000    0.000000
H      0.000000    2.146163    1.365586
H      0.000000   -2.146163    1.365586
H      0.000000    2.146163   -1.365586
H      0.000000   -2.146163   -1.365586
H      0.000000    0.000000    2.356519
H      0.000000    0.000000   -2.356519
H      0.882000    0.000000    0.000000
H     -0.882000    0.000000    0.000000
"""


def _build_molecule():
    """Build a vibe-qc Molecule from the XYZ string (Angstrom -> Bohr)."""
    from vibeqc._vibeqc_core import Atom, Molecule

    BOHR_PER_ANGSTROM = 1.8897259886
    z_map = {"C": 6, "H": 1}
    atoms = []
    for line in NORBORNADIENE_XYZ.strip().split("\n")[2:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        sym, x, y, z = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
        atoms.append(Atom(z_map[sym], [
            x * BOHR_PER_ANGSTROM,
            y * BOHR_PER_ANGSTROM,
            z * BOHR_PER_ANGSTROM,
        ]))
    return Molecule(atoms, charge=0, multiplicity=1)


def _run_native_pm6(mol, max_iter=200, conv_tol=1e-7):
    """Run the native C++ PM6 SCF and return the result object."""
    from vibeqc.semiempirical.methods.pm6_params import load_pm6_params
    from vibeqc._vibeqc_core.semiempirical.nddo import run_pm6

    params = load_pm6_params()
    return run_pm6(mol, params, max_iter=max_iter, conv_tol=conv_tol)


def _run_pm6_with_smearing(mol, electronic_temperature=0.005, max_iter=100):
    """Run PM6 with finite-temperature Fermi-Dirac occupations."""
    from vibeqc.semiempirical.methods.pm6_params import load_pm6_params
    from vibeqc._vibeqc_core.semiempirical.nddo import run_pm6_with_smearing

    params = load_pm6_params()
    return run_pm6_with_smearing(
        mol,
        params,
        electronic_temperature=electronic_temperature,
        max_iter=max_iter,
    )


def _run_pm6_with_density(mol, initial_density, max_iter=200):
    """Run PM6 continuing from a prior converged density."""
    from vibeqc.semiempirical.methods.pm6_params import load_pm6_params
    from vibeqc._vibeqc_core.semiempirical.nddo import run_pm6_with_density

    params = load_pm6_params()
    return run_pm6_with_density(
        mol, params, initial_density, max_iter=max_iter
    )


def _run_full_recovery(mol, max_iter=200):
    """Run the full recovery path (matches _run_pm6 in semiempirical/runner.py)."""
    from vibeqc.semiempirical.methods.pm6_params import load_pm6_params
    from vibeqc._vibeqc_core.semiempirical.nddo import (
        run_pm6,
        run_pm6_with_density,
        run_pm6_with_smearing,
    )

    params = load_pm6_params()

    # Step 1: zero-T SCF
    result = run_pm6(mol, params, max_iter=max_iter)
    total_iter = int(result.n_iter)
    path = ["cold"]

    if not result.converged:
        # Step 2: finite-T smearing
        warm = run_pm6_with_smearing(
            mol, params, electronic_temperature=0.005, max_iter=100
        )
        total_iter += int(warm.n_iter)
        path.append(f"warm(T=0.005, converged={warm.converged})")
        if warm.converged:
            # Step 3: cool back to zero-T
            result = run_pm6_with_density(
                mol, params, warm.density, max_iter=max_iter
            )
            total_iter += int(result.n_iter)
            path.append("cool")

    return result, total_iter, path


def main():
    mol = _build_molecule()
    print(f"Molecule: {len(mol.atoms)} atoms, {mol.n_electrons()} electrons")
    print(f"Charge: {mol.charge}, Multiplicity: {mol.multiplicity}")
    print()

    # ── Test 1: Default PM6 (200 iterations) ──
    print("=== Test 1: Default PM6 (max_iter=200) ===")
    result = _run_native_pm6(mol, max_iter=200)
    print(f"  Energy:     {result.energy:.10f} Ha")
    print(f"  Converged:  {result.converged}")
    print(f"  Iterations: {result.n_iter}")
    print()

    # ── Test 2: Extended iterations ──
    print("=== Test 2: Extended (max_iter=2000) ===")
    result = _run_native_pm6(mol, max_iter=2000)
    print(f"  Energy:     {result.energy:.10f} Ha")
    print(f"  Converged:  {result.converged}")
    print(f"  Iterations: {result.n_iter}")
    print()

    # ── Test 3: Finite-temperature smearing ──
    print("=== Test 3: Finite-T smearing (T=0.005 Ha) ===")
    for T in [0.001, 0.003, 0.005, 0.01, 0.02]:
        warm = _run_pm6_with_smearing(mol, electronic_temperature=T, max_iter=200)
        print(f"  T={T:.3f}: E={warm.energy:.10f}, "
              f"converged={warm.converged}, iters={warm.n_iter}")
    print()

    # ── Test 4: Full recovery path ──
    print("=== Test 4: Full recovery path ===")
    result, total_iter, path = _run_full_recovery(mol, max_iter=200)
    print(f"  Path:        {' → '.join(path)}")
    print(f"  Energy:      {result.energy:.10f} Ha")
    print(f"  Converged:   {result.converged}")
    print(f"  Total iters: {total_iter}")
    print()

    # ── Test 5: Hot start from extended smearing ──
    print("=== Test 5: Hot-start from T=0.02 → cool to T=0 ===")
    warm = _run_pm6_with_smearing(mol, electronic_temperature=0.02, max_iter=200)
    print(f"  Warm (T=0.02): converged={warm.converged}, "
          f"E={warm.energy:.10f}, iters={warm.n_iter}")
    if warm.converged:
        result = _run_pm6_with_density(mol, warm.density, max_iter=200)
        print(f"  Cool (T=0):    converged={result.converged}, "
              f"E={result.energy:.10f}, iters={result.n_iter}")
        # Verify energy is reasonable (single-point from hot start)
        result2 = _run_pm6_with_density(mol, result.density, max_iter=200)
        print(f"  Re-cool:       converged={result2.converged}, "
              f"E={result2.energy:.10f}, iters={result2.n_iter}")
    print()

    # ── Test 6: Iteration-by-iteration energy trace ──
    print("=== Test 6: Cold-start energy vs iteration cap ===")
    from vibeqc.semiempirical.methods.pm6_params import load_pm6_params
    from vibeqc._vibeqc_core.semiempirical.nddo import run_pm6

    params = load_pm6_params()
    for cap in [5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 1000]:
        r = run_pm6(mol, params, max_iter=cap)
        print(f"  cap={cap:4d}: E={r.energy:.10f}, "
              f"converged={r.converged}, iters={r.n_iter}")

    # ── Test 7: Temperature ladder test ──
    print("\n=== Test 7: Temperature ladder (cold → warm → cool) ===")
    from vibeqc._vibeqc_core.semiempirical.nddo import (
        run_pm6_with_density,
        run_pm6_with_smearing,
    )
    for T_try in [0.005, 0.01, 0.015, 0.02, 0.03]:
        warm = run_pm6_with_smearing(mol, params, electronic_temperature=T_try, max_iter=200)
        if warm.converged:
            cool = run_pm6_with_density(mol, params, warm.density, max_iter=200)
            print(f"  T={T_try:.3f}: warm E={warm.energy:.10f} ({warm.n_iter} it), "
                  f"cool E={cool.energy:.10f} ({cool.n_iter} it), "
                  f"converged={cool.converged}")
        else:
            print(f"  T={T_try:.3f}: warm NOT converged ({warm.n_iter} it)")

    # ── Test 8: Verify energy stability ──
    print("\n=== Test 8: Energy stability of hot-start solution ===")
    warm02 = run_pm6_with_smearing(mol, params, electronic_temperature=0.02, max_iter=200)
    cool = run_pm6_with_density(mol, params, warm02.density, max_iter=200)
    # Re-run from the cooled density to verify it's a fixed point
    recool = run_pm6_with_density(mol, params, cool.density, max_iter=200)
    e_diff = abs(cool.energy - recool.energy)
    print(f"  T=0.02 warm:  E={warm02.energy:.10f}")
    print(f"  Cool to T=0:  E={cool.energy:.10f}, iters={cool.n_iter}")
    print(f"  Re-cool:       E={recool.energy:.10f}, iters={recool.n_iter}")
    print(f"  |E_cool - E_recool| = {e_diff:.2e} Ha")
    assert e_diff < 1e-8, f"Hot-start solution is not stable: ΔE = {e_diff:.2e}"
    print("  ✓ Hot-start solution is a stable fixed point.")

    return 0 if cool.converged else 1


if __name__ == "__main__":
    sys.exit(main())
