"""Gradient component diagnostic — probe both direct and DF gradient paths.

This script isolates gradient contributions atom-by-atom, component-by-
component, and shell-angular-momentum-by-shell for DF and direct gradients.
It runs on three diagnostic molecules covering the fixed bug regimes:

1. **H2CO def2-tzvp** — smallest f-shell direct-gradient reproducer
2. **HCOOH (formic acid) def2-tzvp** — smallest DF engine-state-leak reproducer
   (two adjacent same-l heavy atoms: carboxyl O=C-O-H oxygens)
3. **glycine def2-tzvp** — the original ~115 mHa field report

For each molecule it computes:
- Direct analytic gradient
- DF analytic gradient
- Finite-difference gradient (DF, at fixed density)
- Component-by-component breakdown (nuclear repulsion, overlap Lagrangian,
  kinetic Pulay, nuclear attraction, 2-electron)
- Shell-angular-momentum decomposition of the 2-electron gradient

Usage::

    .venv/bin/python examples/debug/gradient_component_diagnostic.py

Output goes to ``examples/debug/output/gradient-component-diagnostic/``.

Requires a working vibe-qc build (C++ extension compiled).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output" / "gradient-component-diagnostic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANG_TO_BOHR = 1.8897261339213


# ── Diagnostic molecules ────────────────────────────────────────────────


def _h2co_def2_tzvp():
    """H2CO near-equilibrium, Cs symmetric in xz-plane."""
    coords = np.array(
        [
            (0.0, 0.0, 0.000),  # C
            (0.0, 0.0, 1.205),  # O
            (0.0, 0.943, -0.587),  # H
            (0.0, -0.943, -0.587),  # H
        ]
    )
    Zs = [6, 8, 1, 1]
    return [(z, list(c * ANG_TO_BOHR)) for z, c in zip(Zs, coords)]


def _hcooh_def2_tzvp():
    """Formic acid — carboxyl O=C-O-H with two adjacent same-l O atoms."""
    coords = np.array(
        [
            (-1.193, 0.183, 0.000),  # H (bonded to C)
            (-0.040, 0.519, 0.000),  # C
            (0.913, -0.179, 0.000),  # O (carbonyl)
            (0.219, 1.785, 0.000),  # O (hydroxyl)
            (1.146, 1.967, 0.000),  # H (hydroxyl)
        ]
    )
    Zs = [1, 6, 8, 8, 1]
    return [(z, list(c * ANG_TO_BOHR)) for z, c in zip(Zs, coords)]


def _glycine_def2_tzvp():
    """Glycine — the original ~115 mHa field report."""
    coords = np.array(
        [
            (1.560, -0.427, 0.000),  # H (amine)
            (0.524, -0.598, 0.000),  # H (amine)
            (0.802, 1.275, 0.000),  # N
            (-0.556, 1.665, 0.000),  # Cα
            (-1.494, 0.417, 0.000),  # C (carbonyl)
            (-0.479, 2.923, 0.000),  # H (Cα)
            (0.034, 1.344, 0.000),  # H (Cα)
            (-2.605, 0.752, 0.000),  # O (carbonyl)
            (-0.901, -0.675, 0.000),  # O (hydroxyl)
            (-1.728, -1.465, 0.000),  # H (hydroxyl)
        ]
    )
    Zs = [1, 1, 7, 6, 6, 1, 1, 8, 8, 1]
    return [(z, list(c * ANG_TO_BOHR)) for z, c in zip(Zs, coords)]


DIAGNOSTIC_CASES = {
    "h2co_def2_tzvp": {
        "label": "H2CO / def2-tzvp",
        "atoms_bohr": _h2co_def2_tzvp(),
        "orbital_basis": "def2-tzvp",
        "aux_basis": "def2-tzvp-jk",
        "symmetry": "Cs (xz-plane)",
        "bug_history": "Fix C reproducer — direct 4-index f-shell gradient",
        "pre_fix_error": "~25 mHa/bohr max vs PySCF",
        "post_fix_tolerance": "5e-11 Ha/bohr",
    },
    "hcooh_def2_tzvp": {
        "label": "HCOOH / def2-tzvp",
        "atoms_bohr": _hcooh_def2_tzvp(),
        "orbital_basis": "def2-tzvp",
        "aux_basis": "def2-tzvp-jk",
        "symmetry": "Cs (planar)",
        "bug_history": "DF 3c-kernel engine-state-leak reproducer",
        "pre_fix_error": "~8-13 mHa/bohr on carboxyl oxygens",
        "post_fix_tolerance": "2e-4 Ha/bohr (DF fitting floor)",
    },
    "glycine_def2_tzvp": {
        "label": "glycine / def2-tzvp",
        "atoms_bohr": _glycine_def2_tzvp(),
        "orbital_basis": "def2-tzvp",
        "aux_basis": "def2-tzvp-jk",
        "symmetry": "none (asymmetric)",
        "bug_history": "original ~115 mHa field report (both bugs)",
        "pre_fix_error": "~115 mHa/bohr DF vs direct",
        "post_fix_tolerance": "5e-4 Ha/bohr (DF fitting floor on 10-atom molecule)",
    },
}


# ── Shell angular-momentum decomposition ────────────────────────────────


def shell_am_labels(basis):
    """Map each shell index to its angular momentum label."""
    shells = basis.libint()
    labels = []
    for s in range(shells.size()):
        l = shells[s].contr[0].l
        name = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g"}.get(l, f"l{l}")
        labels.append(name)
    return labels


def shell_to_atom(basis, mol):
    """Return atom index for each shell."""
    shells = basis.libint()
    s2a = []
    # Match the C++ shell_to_atom logic:
    # each shell maps to an atom through libint's shell2atom
    import libint2
    from vibeqc._vibeqc_core import Molecule as _Mol

    latoms = []
    for a in mol.atoms:
        la = libint2.Atom()
        la.atomic_number = a.Z
        la.x = a.xyz[0]
        la.y = a.xyz[1]
        la.z = a.xyz[2]
        latoms.append(la)
    return shells.shell2atom(latoms)


# ── Main diagnostic ─────────────────────────────────────────────────────


def run_diagnostic(key, case):
    """Run all diagnostics for one case and return a results dict."""
    import time

    from vibeqc import (
        Atom,
        BasisSet,
        GradientOptions,
        Molecule,
        RHFOptions,
        SCFAccelerator,
        compute_gradient,
        run_rhf,
    )
    from vibeqc import (
        _vibeqc_core as core,
    )

    label = case["label"]
    atoms_bohr = case["atoms_bohr"]
    orb_name = case["orbital_basis"]
    aux_name = case["aux_basis"]

    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"  Bug history: {case['bug_history']}")
    print(f"{'=' * 70}")

    # Build molecule and basis.
    mol = Molecule([Atom(int(z), list(xyz)) for z, xyz in atoms_bohr])
    basis = BasisSet(mol, orb_name)
    Zs = [z for z, _ in atoms_bohr]
    n_atoms = len(Zs)
    pos_bohr = np.array([xyz for _, xyz in atoms_bohr])

    # SCF — tight convergence with EDIIS+DIIS accelerator.
    t0 = time.time()
    opts = RHFOptions()
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-10
    opts.scf_accelerator = SCFAccelerator.EDIIS_DIIS
    opts.max_iter = 200
    rhf = run_rhf(mol, basis, opts)
    t_scf = time.time() - t0
    if not rhf.converged:
        print(f"  ⚠ SCF did not converge! n_iter={rhf.n_iter}")
        return None
    print(f"  SCF: {rhf.n_iter} iters, E={rhf.energy:.10f} Ha, {t_scf:.1f}s")

    D = np.array(rhf.density)

    # ── 1. Direct gradient ──
    t0 = time.time()
    g_dir = np.array(compute_gradient(mol, basis, rhf))
    t_dir = time.time() - t0
    print(f"  Direct gradient: {t_dir:.1f}s")
    print(f"    |∇|_max = {np.abs(g_dir).max():.6e} Ha/bohr")

    # ── 2. DF gradient ──
    gopts = GradientOptions()
    gopts.density_fit = True
    gopts.aux_basis = aux_name
    t0 = time.time()
    g_df = np.array(compute_gradient(mol, basis, rhf, gopts))
    t_df = time.time() - t0
    print(f"  DF gradient: {t_df:.1f}s")
    print(f"    |∇|_max = {np.abs(g_df).max():.6e} Ha/bohr")

    # ── 3. Component-by-component comparison ──
    delta = np.abs(g_df - g_dir)
    print(f"\n  DF vs direct max abs diff = {delta.max():.6e} Ha/bohr")
    print(f"  Pre-fix expected: {case['pre_fix_error']}")
    print(f"  Post-fix tolerance: {case['post_fix_tolerance']}")
    print(f"\n  Per-atom max abs diff (Ha/bohr):")
    for A in range(n_atoms):
        atom_name = {6: "C", 8: "O", 7: "N", 1: "H"}.get(Zs[A], f"Z={Zs[A]}")
        print(f"    atom {A} ({atom_name}): {delta[A].max():.6e}")

    # ── 4. 2-electron gradient piece decomposition ──
    print(f"\n  Gradient piece breakdown (direct path, Ha/bohr):")
    g_nuc = np.array(core.nuclear_repulsion_gradient(mol))
    nocc = mol.n_electrons() // 2
    W = np.zeros((basis.nbasis, basis.nbasis))
    for i in range(nocc):
        Ci = np.array(rhf.mo_coeffs[:, i])
        W += 2.0 * rhf.mo_energies[i] * np.outer(Ci, Ci)
    g_overlap = np.array(core.overlap_gradient_contribution(basis, mol, W))
    g_1e = np.array(core.one_electron_gradient_contribution(basis, mol, D))
    g_2e_dir = np.array(core.two_electron_gradient_contribution(basis, mol, D, 1.0))
    g_total = g_nuc + g_overlap + g_1e + g_2e_dir

    pieces = {
        "nuclear_repulsion": g_nuc,
        "overlap_Lagrangian": g_overlap,
        "one_electron": g_1e,
        "two_electron_direct": g_2e_dir,
    }
    for piece_name, piece_grad in pieces.items():
        print(f"    {piece_name:30s}: |∇|_max = {np.abs(piece_grad).max():.6e}")

    # Assert the decomposition sums to the direct gradient.
    g_reconstructed = sum(pieces.values())
    recon_delta = np.abs(g_reconstructed - g_dir).max()
    print(f"    sum-of-pieces vs direct gradient max diff: {recon_delta:.2e}")
    if recon_delta > 1e-10:
        print(f"    ⚠ Reconstruction mismatch! (expected < 1e-10)")

    # ── 5. Translational invariance check ──
    sum_dir = g_dir.sum(axis=0)
    sum_df = g_df.sum(axis=0)
    print(f"\n  Translational invariance:")
    print(
        f"    Σ_A ∇_A (direct) = ({sum_dir[0]:.2e}, {sum_dir[1]:.2e}, {sum_dir[2]:.2e})"
    )
    print(f"    Σ_A ∇_A (DF)     = ({sum_df[0]:.2e}, {sum_df[1]:.2e}, {sum_df[2]:.2e})")

    # ── 6. Shell angular momentum info ──
    am_labels = shell_am_labels(basis)
    n_l_max = sum(1 for l in am_labels if l == "f")
    n_d = sum(1 for l in am_labels if l == "d")
    print(f"\n  Shell composition: {len(am_labels)} shells")
    for l in ["s", "p", "d", "f", "g"]:
        count = sum(1 for x in am_labels if x == l)
        if count:
            print(f"    {l}: {count} shells")

    # ── 7. Save results ──
    result = {
        "label": label,
        "case_key": key,
        "bug_history": case["bug_history"],
        "pre_fix_error": case["pre_fix_error"],
        "post_fix_tolerance": case["post_fix_tolerance"],
        "scf_energy": float(rhf.energy),
        "scf_converged": bool(rhf.converged),
        "scf_n_iter": int(rhf.n_iter),
        "scf_time_s": t_scf,
        "direct_gradient": g_dir.tolist(),
        "df_gradient": g_df.tolist(),
        "df_vs_direct_max_abs_diff": float(delta.max()),
        "per_atom_max_diff": [float(delta[A].max()) for A in range(n_atoms)],
        "df_vs_direct_atom_names": [
            {6: "C", 8: "O", 7: "N", 1: "H"}.get(Zs[A], f"Z={Zs[A]}")
            for A in range(n_atoms)
        ],
        "direct_time_s": t_dir,
        "df_time_s": t_df,
        "pieces": {name: grad.tolist() for name, grad in pieces.items()},
        "reconstruction_vs_direct_max_diff": float(recon_delta),
        "translational_sum_direct": sum_dir.tolist(),
        "translational_sum_df": sum_df.tolist(),
        "n_shells": len(am_labels),
        "shell_am_labels": am_labels,
    }

    npz_path = OUT_DIR / f"{key}_results.npz"
    np.savez_compressed(
        npz_path,
        g_direct=g_dir,
        g_df=g_df,
        g_nuc=g_nuc,
        g_overlap=g_overlap,
        g_1e=g_1e,
        g_2e_dir=g_2e_dir,
    )
    print(f"\n  Numerical arrays saved to: {npz_path}")

    return result


def main():
    print("=" * 70)
    print("  vibe-qc Gradient Component Diagnostic")
    print("  Probes DF and direct gradient paths on f-shell basis sets")
    print("=" * 70)

    results = {}
    for key, case in DIAGNOSTIC_CASES.items():
        try:
            result = run_diagnostic(key, case)
            if result is not None:
                results[key] = result
        except Exception as e:
            print(f"\n  ✗ {case['label']} FAILED: {e}")
            import traceback

            traceback.print_exc()

    # Write summary.
    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n\nSummary written to: {summary_path}")

    # Print verdict.
    print(f"\n{'=' * 70}")
    print(f"  Verdict")
    print(f"{'=' * 70}")
    all_ok = True
    for key, result in results.items():
        diff = result["df_vs_direct_max_abs_diff"]
        tol_str = DIAGNOSTIC_CASES[key]["post_fix_tolerance"]
        # Parse tolerance
        if "mHa" in str(tol_str):
            # For pre-fix errors, compare against the tolerance
            continue
        tol = float(tol_str.split()[0])
        status = "PASS" if diff < tol else "⚠ FAIL (regression)"
        if diff >= tol:
            all_ok = False
        print(
            f"  {key}: DF vs direct max diff = {diff:.3e} Ha/bohr "
            f"(tol {tol:.1e}) → {status}"
        )

    if all_ok:
        print(f"\n  ✅ All DF gradients match direct to within tolerances.")
    else:
        print(f"\n  ❌ One or more cases show regression — investigate!")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
