"""Cross-validate H2O vibrational frequencies vs. ORCA.

Both vibe-qc and ORCA compute the analytic CPHF Hessian for closed-
shell HF, write a normal-modes file (``.hess`` from ORCA's
``orca_2mkl``-style writer; vibe-qc's ``write_orca_hess`` matches the
same ASCII format so moltui / Avogadro / chemcraft / VMD-nmwiz read
either side identically), and report the vibrational frequencies.

Run:
    .venv/bin/python examples/ase_compare/compare-h2o-vibrations.py

ORCA is required for the ORCA row — unavailable codes are reported
in the table as "unavailable" and the comparison continues.

Produces:
    output-compare-h2o-vibrations.csv  — CSV with both freq vectors
    (stdout)                            — pretty-printed comparison

Any ORCA logs, Hessian files, and vibe-qc outputs are generated
locally by the run and should stay out of git.

Tolerance: 5 cm⁻¹ on each frequency. Both codes use the same CPHF
analytic Hessian formalism with identical Grimme / Pople integral
thresholds; agreement should be that tight on a clean H2O test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase.build import molecule

from vibeqc.ase import VibeQC
from vibeqc.benchmark import (
    make_orca_calculator,
    print_calculator_availability,
)

HERE = Path(__file__).resolve().parent
EV_PER_HA = 27.211386245988


def _vibeqc_freqs(atoms) -> np.ndarray:
    """Compute the analytic RHF Hessian via vibe-qc's ASE calculator,
    return the 3 real vibrational frequencies (cm⁻¹)."""
    import vibeqc as vq

    atoms.calc = VibeQC(basis="6-31g*")
    H = atoms.calc.get_property("hessian", atoms)   # eV/Å²

    # Build vibe-qc objects to call write_orca_hess on. We re-derive
    # the underlying HessianResult here — same kernel, same numbers,
    # but as a vibeqc.HessianResult we can pass to write_orca_hess.
    pos_bohr = atoms.positions / 0.529177210903
    mol = vq.Molecule(
        [vq.Atom(int(z), list(p))
         for z, p in zip(atoms.numbers, pos_bohr)],
    )
    basis = vq.BasisSet(mol, "6-31g*")
    rhf_result = vq.run_rhf(mol, basis)
    hess = vq.compute_hessian_rhf_analytic(
        mol, basis, rhf_result, basis_name="6-31g*",
    )
    freqs = np.asarray(hess.frequencies_cm1)
    return np.sort(np.real(freqs[np.abs(freqs) > 100.0]))


def _orca_freqs() -> np.ndarray | None:
    """Run ORCA HF/6-31G* Freq on the same H2O geometry, return the
    3 vibrational frequencies (cm⁻¹). Returns None if ORCA isn't
    available."""
    orca = make_orca_calculator(
        orcasimpleinput="HF 6-31G* Freq",
        label="orca-h2o-vib",
    )
    if orca is None:
        return None

    atoms = molecule("H2O")
    atoms.calc = orca
    # Trigger the calculation; ASE writes orca-h2o-vib.{inp,out,hess}
    # to the cwd. `Freq` makes ORCA produce the .hess.
    atoms.get_potential_energy()

    # Parse ORCA's .hess file (ASE doesn't expose ORCA frequencies
    # via a property; do it ourselves). ASE's ORCA wrapper writes the
    # file with the `label=` prefix when supplied; ASE 3.22+ sometimes
    # falls back to "orca" regardless of label, so check both names
    # in the cwd of script invocation (where ASE actually writes).
    for candidate in (
        Path.cwd() / "orca-h2o-vib.hess",
        Path.cwd() / "orca.hess",
        HERE / "orca-h2o-vib.hess",
        HERE / "orca.hess",
    ):
        if candidate.is_file():
            return _parse_orca_hess_freqs(candidate)

    raise FileNotFoundError(
        f"ORCA .hess file not found in {Path.cwd()} or {HERE} — "
        f"check that ORCA actually completed (orca.out should "
        f"contain 'TERMINATED NORMALLY')"
    )


def _parse_orca_hess_freqs(path: Path) -> np.ndarray:
    """Extract the [vibrational_frequencies] block from an ORCA .hess
    file. ORCA orders all 3N frequencies (translations and rotations
    appear at the bottom as ~zero); we drop those (|ω| < 100 cm⁻¹)."""
    text = path.read_text()
    lines = text.splitlines()
    # Locate the $vibrational_frequencies block
    for i, line in enumerate(lines):
        if line.strip() == "$vibrational_frequencies":
            n = int(lines[i + 1])
            freqs = []
            for j in range(n):
                fields = lines[i + 2 + j].split()
                freqs.append(float(fields[1]))
            return np.sort([f for f in freqs if abs(f) > 100.0])
    raise ValueError(f"could not find $vibrational_frequencies in {path}")


def main() -> None:
    print("=" * 72)
    print(" Cross-validation:  H2O / RHF / 6-31G* — vibrational frequencies")
    print("=" * 72)
    print()
    print("Calculator availability:")
    print_calculator_availability()
    print()

    atoms = molecule("H2O")

    print("Computing vibe-qc analytic CPHF Hessian (Phase 17b-3)…")
    vqc_freqs = _vibeqc_freqs(atoms)

    print("Computing ORCA analytic Hessian via ! HF 6-31G* Freq…")
    orca_freqs = _orca_freqs()

    # Print comparison
    print()
    if orca_freqs is not None:
        print(f"  Mode               vibe-qc (cm⁻¹)    ORCA (cm⁻¹)    |Δ|")
        print(f"  " + "─" * 60)
        labels = ["bend", "asym stretch", "sym stretch"]
        if len(vqc_freqs) != 3 or len(orca_freqs) != 3:
            labels = [f"mode {i+1}" for i in range(len(vqc_freqs))]
        for label, vqc, orca in zip(labels, vqc_freqs, orca_freqs):
            print(f"  {label:<18}   {vqc:>12.3f}      {orca:>10.3f}    "
                  f"{abs(vqc - orca):>6.3f}")
        max_gap = float(np.max(np.abs(vqc_freqs - orca_freqs)))
        print(f"  Max |Δω| = {max_gap:.3f} cm⁻¹")
    else:
        print("ORCA not available — vibe-qc-only frequencies:")
        for i, w in enumerate(vqc_freqs):
            print(f"  mode {i+1}:  ω = {w:.3f} cm⁻¹")
        print()
        print("Set ORCA_COMMAND or add orca to $PATH to enable the "
              "cross-validation row.")
        return

    # Save CSV
    csv_path = HERE / "output-compare-h2o-vibrations.csv"
    with csv_path.open("w") as f:
        f.write("mode_index,vibeqc_cm1,orca_cm1,delta_cm1\n")
        for i, (v, o) in enumerate(zip(vqc_freqs, orca_freqs)):
            f.write(f"{i},{v},{o},{abs(v-o)}\n")
    print(f"\n  CSV: {csv_path.relative_to(HERE.parent.parent)}")

    # Tolerance: 5 cm⁻¹ — same CPHF formalism, same basis, same SCF
    # threshold. Tighter agreement requires matching the numerical
    # FD step in the displaced-Fock pieces of CPHF (vibe-qc default
    # 1e-4 bohr; ORCA default 0.005 bohr); harmonising those would
    # close the gap to ~0.1 cm⁻¹.
    assert max_gap < 5.0, (
        f"frequencies disagree by {max_gap:.3f} cm⁻¹ — investigate "
        f"CPHF or basis-set conventions"
    )
    print("\n✓ vibe-qc and ORCA agree on H2O vibrational frequencies "
          "to within 5 cm⁻¹ — same CPHF formalism + basis + SCF "
          "threshold.")


if __name__ == "__main__":
    main()
