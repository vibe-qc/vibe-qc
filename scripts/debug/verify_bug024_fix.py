#!/usr/bin/env python3
"""Verify BUG-024 fix: PM6 on norbornadiene now converges via temperature ladder."""
from __future__ import annotations

import math
import sys

BOHR_PER_ANGSTROM = 1.8897259886

# Norbornadiene C7H8, C2v symmetry, Angstrom
XYZ_LINES = [
    ("C", 0.0, 0.0, 1.268695),
    ("C", 0.0, 1.215139, 0.805739),
    ("C", 0.0, -1.215139, 0.805739),
    ("C", 0.0, 1.215139, -0.805739),
    ("C", 0.0, -1.215139, -0.805739),
    ("C", 0.0, 0.0, -1.268695),
    ("C", 0.0, 0.0, 0.0),
    ("H", 0.0, 2.146163, 1.365586),
    ("H", 0.0, -2.146163, 1.365586),
    ("H", 0.0, 2.146163, -1.365586),
    ("H", 0.0, -2.146163, -1.365586),
    ("H", 0.0, 0.0, 2.356519),
    ("H", 0.0, 0.0, -2.356519),
    ("H", 0.882, 0.0, 0.0),
    ("H", -0.882, 0.0, 0.0),
]


def main():
    from vibeqc._vibeqc_core import Atom, Molecule
    from vibeqc.semiempirical.runner import run_semiempirical

    Z_MAP = {"C": 6, "H": 1}
    atoms = [
        Atom(Z_MAP[sym], [x * BOHR_PER_ANGSTROM, y * BOHR_PER_ANGSTROM, z * BOHR_PER_ANGSTROM])
        for sym, x, y, z in XYZ_LINES
    ]
    mol = Molecule(atoms, charge=0, multiplicity=1)

    print("=== Norbornadiene PM6 via run_semiempirical ===")
    result = run_semiempirical("pm6", mol)

    print(f"  Energy:     {result.energy:.10f} Ha")
    print(f"  Converged:  {result.converged}")
    print(f"  Iterations: {result.n_iter}")

    if not result.converged:
        print("FAIL: PM6 did not converge on norbornadiene.")
        return 1

    # BUG-024 is the SILENT-E-ZERO symptom: PM6 returning a converged-looking
    # result whose energy is zero (or non-finite) and a job that never
    # completes.  What this diagnostic must therefore assert is health, not a
    # particular number.
    #
    # It used to carry `expected = -35.75976498`, a hot-start value from the
    # original 2026 diagnostic.  That number was retired by the #244
    # heavy-heavy NDDO multipole repair (ancestral via febe89375) and the
    # script then exited 1 on a perfectly healthy run -- 2.67 Ha "off" a
    # reference that no longer describes the operator -- before it ever
    # reached the run_job and manifest checks below (issue #469).  A
    # superseded absolute oracle that fails closed on correct output is worse
    # than no oracle: it hides the checks behind it.
    #
    # Absolute PM6 accuracy is not this script's job and is tracked where it
    # belongs -- the canonical PM6 regressions, which pin oracle parity
    # against MOPAC (issue #255: core-core at 1.000x the oracle, H2O total
    # within 5.9e-06 Ha of the oracle ETOT).  If an absolute pin is ever
    # reintroduced here it must name the operator and provenance it validates.
    if not math.isfinite(result.energy):
        print(f"FAIL: PM6 energy is not finite: {result.energy!r}")
        return 1
    if result.energy == 0.0:
        print("FAIL: PM6 returned a silent zero energy (the BUG-024 symptom).")
        return 1

    # Also verify through run_job
    print("\n=== Norbornadiene PM6 via run_job ===")
    from vibeqc import runner
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmp:
        stem = os.path.join(tmp, "norbornadiene_pm6")
        try:
            rj_result = runner.run_job(
                mol,
                method="pm6",
                output=stem,
                write_molden_file=False,
                write_xyz_file=False,
                write_population_file=False,
                citations=False,
                progress=False,
            )
            print(f"  run_job energy: {rj_result.energy:.10f} Ha")
            print(f"  run_job converged: {getattr(rj_result, 'converged', '?')}")

            # Operator-independent oracle: the two public routes must agree on
            # the same molecule whatever the current PM6 operator is. This
            # catches a route diverging from the other without pinning a value
            # that a legitimate operator repair would retire.
            route_delta = abs(rj_result.energy - result.energy)
            print(f"  run_semiempirical vs run_job: {route_delta:.2e} Ha")
            if route_delta > 1.0e-8:
                print(
                    f"FAIL: routes disagree -- run_semiempirical "
                    f"{result.energy:.10f} vs run_job {rj_result.energy:.10f}"
                )
                return 1

            # Check manifest
            import tomllib
            manifest = tomllib.loads(
                open(stem + ".system").read()
            )
            status = manifest["outputs"]["status"]
            print(f"  .system status: {status}")
            if status != "complete":
                print(f"FAIL: status={status}, expected 'complete'")
                return 1
        except RuntimeError as e:
            print(f"FAIL: run_job raised RuntimeError: {e}")
            return 1

    print("\n✓ BUG-024 FIXED: PM6 converges on norbornadiene.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
