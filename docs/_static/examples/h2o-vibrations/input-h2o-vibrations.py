"""Closed-shell RHF + analytic Hessian on H2O / 6-31G*.

Run:
    .venv/bin/python input-h2o-vibrations.py

Produces:
    output-h2o-vibrations.out      — banner, SCF trace, energies, MO table
    output-h2o-vibrations.molden   — molecular orbitals + normal modes
                                       (the [FREQ] / [FR-COORD] blocks
                                       any molden-aware viewer renders)
    output-h2o-vibrations.hess     — ORCA-format normal-modes file
                                       (consumed by moltui / chemcraft /
                                        Avogadro / VMD-nmwiz; produced by
                                        vibe-qc's vq.write_orca_hess)

The Hessian is the analytic CPHF closed-shell-RHF kernel
(Phase 17b-3); matches PySCF analytic Hessian to <1e-7 Ha/bohr².
HF/6-31G* harmonic frequencies overshoot experiment by ~10-15%
(the standard scaling factor is 0.89). Bend at ~1750 cm⁻¹, two
stretches at ~3700-3800 cm⁻¹.

Companion to ``input-h2o-rhf.py`` (energy + forces only); this
script adds the analytic Hessian + .hess export so the same
geometry can be inspected vibrationally inline in moltui::

    moltui output-h2o-vibrations.hess
"""

from pathlib import Path

import vibeqc as vq

HERE = Path(__file__).parent

# Equilibrium-ish water geometry (relaxed at HF/6-31G* to ~mHa)
mol = vq.Molecule.from_xyz(HERE.parent / "h2o.xyz")
basis = vq.BasisSet(mol, "6-31g*")

# 1. Run the SCF + write .out / .molden via run_job. This gives us
#    the standard RHF text log + orbital-only molden for free; it
#    returns the underlying RHFResult directly.
rhf_result = vq.run_job(
    mol,
    basis="6-31g*",
    method="rhf",
    output=HERE / "output-h2o-vibrations",
)

# 2. Compute the analytic Hessian on top. Reuses the same converged
#    SCF result. ``basis_name`` is required by the kernel for the
#    FD-on-Fock displaced-geometry rebuilds inside CPHF.
hess = vq.compute_hessian_rhf_analytic(
    mol, basis, rhf_result, basis_name="6-31g*",
)

# 3. Write the ORCA-format .hess for moltui / chemcraft / Avogadro
#    / VMD-nmwiz interop. ``vq.write_orca_hess`` (Phase M1) ships
#    the ASCII format with [FREQ] / [NORMAL_MODES] / [DIPOLE_DERIVATIVES]
#    blocks ORCA emits.
hess_path = HERE / "output-h2o-vibrations.hess"
vq.write_orca_hess(
    hess_path,
    mol,
    hess,
    energy_ha=float(rhf_result.energy),
    multiplicity=mol.multiplicity,
)
print(f"Wrote {hess_path}")
print(f"Frequencies (cm-1): {hess.frequencies_cm1}")
