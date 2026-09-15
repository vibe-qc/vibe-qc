"""Vibrational analysis of H2O via ase.vibrations.Vibrations.

ASE's ``Vibrations`` class does finite-difference Hessian on top of
the calculator's forces. With vibe-qc's analytic forces (Phase 17
gradients), this is a pure-FD-on-gradient workflow — fast and
calculator-agnostic.

Compare this to:
  * vibe-qc's own ``compute_hessian_fd`` — same approach, but pulls
    the FD orchestration into the C++/Python boundary so it can run
    in parallel and reuse density-matrix initial guesses.
  * vibe-qc's ``compute_hessian_rhf_analytic`` (Phase 17b-3) — full
    CPHF analytic Hessian, available through
    ``atoms.calc.get_property("hessian", atoms)`` (Phase A).

Run:
    .venv/bin/python examples/ase_workflows/vibrations-via-ase-vibrations.py

Wall time on water: ~3 seconds (12 displacements × small SCF).

Produces:
    vib-h2o.<*>.json    — ASE's per-displacement cache (incremental;
                          re-run resumes)
    output-vibrations-summary.txt
                       — frequencies + reduced masses + IR
                          intensities (when available)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.optimize import BFGS
from ase.vibrations import Vibrations

from vibeqc.ase import VibeQC

HERE = Path(__file__).resolve().parent

# Constants for the ASE freq → cm⁻¹ conversion is built into ASE
# (Vibrations.get_frequencies() returns cm⁻¹ already; complex values
# for imaginary modes).


def main() -> None:
    print("=" * 68)
    print(" ASE Vibrations:  H2O / RHF / 6-31G*")
    print("=" * 68)

    # Start at ASE's H2O equilibrium-ish geometry, relax tightly so
    # rot/trans modes show as ~0 cm⁻¹.
    atoms = Atoms(
        symbols=["O", "H", "H"],
        positions=[
            (0.000, 0.000,  0.000),
            (0.762, 0.000, -0.566),
            (-0.762, 0.000, -0.566),
        ],
    )
    atoms.calc = VibeQC(basis="6-31g*")

    print("\nRelaxing the geometry to the local minimum…")
    BFGS(atoms, logfile=None).run(fmax=0.001, steps=80)
    print(f"  E(min) = {atoms.get_potential_energy():.6f} eV")
    print(f"  |F|max  = {np.abs(atoms.get_forces()).max():.4e} eV/Å")

    print("\nRunning ASE Vibrations (FD on analytic gradient)…")
    vib_dir = HERE / "vib-h2o-cache"
    vib_dir.mkdir(exist_ok=True)
    vib = Vibrations(atoms, name=str(vib_dir / "vib-h2o"))
    vib.run()

    # Frequencies (cm⁻¹), real for stable modes, imaginary for
    # transition-state-like (negative-curvature) modes.
    freqs = vib.get_frequencies()
    print("\nFrequencies (cm⁻¹):")
    for i, f in enumerate(freqs):
        if np.iscomplex(f):
            print(f"  mode {i:2d}:   {abs(f.imag):8.2f} i  (imaginary)")
        else:
            print(f"  mode {i:2d}:   {float(f.real):8.2f}")

    # The 6 rotational/translational modes should be near zero;
    # the remaining 3 are the H2O bend + symmetric/antisymmetric
    # stretches (~1750, 3700, 3800 cm⁻¹ at HF/6-31G*, give or take
    # a few %).
    real_freqs = sorted(
        float(f.real) for f in freqs
        if not np.iscomplex(f)
    )
    rot_trans_max = max(abs(v) for v in real_freqs[:6]) if len(real_freqs) >= 6 else None
    vib_modes = sorted(real_freqs[-3:])
    print(f"\nMax |rot/trans freq|:  "
          f"{rot_trans_max:.2f} cm⁻¹"
          if rot_trans_max is not None else "(insufficient real modes)")
    print(f"Three vibrational modes: {vib_modes}")

    # Standard summary file
    summary = HERE / "output-vibrations-summary.txt"
    with summary.open("w") as f:
        vib.summary(log=f)
    print(f"\nFull summary: {summary.relative_to(HERE.parent.parent)}")

    # Sanity check: the three vibrational frequencies should be in a
    # specific physical range for water at HF/6-31G*. HF overbinds the
    # bonds → frequencies overshoot experiment by ~10-15 %.
    assert vib_modes[0] > 1500, f"bend mode {vib_modes[0]} cm⁻¹ out of range"
    assert 3500 < vib_modes[1] < 4500, f"sym stretch {vib_modes[1]} cm⁻¹ out of range"
    assert 3500 < vib_modes[2] < 4500, f"asym stretch {vib_modes[2]} cm⁻¹ out of range"
    print("\n✓ all three real vibrational frequencies are in the "
          "expected range for HF/6-31G* water.")


if __name__ == "__main__":
    main()
