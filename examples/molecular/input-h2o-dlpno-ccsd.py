"""Water / cc-pVDZ (aux cc-pVDZ-RI): DLPNO-CCSD, the reduced-scaling local solver.

`method="dlpno-ccsd"` runs the per-pair local coupled-cluster solver
(``vibeqc.dlpno.ccsd_local_solver``): amplitudes remain in pair-natural-
orbital bases, while the default residual is contracted in the atom-based
extended PAO domain and projected back into the target pair. No full-system
amplitude tensor is formed. In the fully untruncated, unscreened limit it
reproduces canonical closed-shell CCSD for the same active occupied space.
The example reports, rather than pre-claims, the recovery of the current
published NormalPNO default.

This script compares default truncation with a full-domain local CCSD
calculation on the same RHF reference. It does not run an independent
canonical CCSD solver and therefore does not itself establish canonical parity.
For the (T) correction use ``method="dlpno-ccsd(t)"`` (DLPNO-(T1),
which restores off-diagonal localised-Fock coupling and reaches canonical
(T) only in the full-domain limit; see ``input-h2o-dlpno-ccsd-t.py``).

(The default is Liakos NormalPNO: ``tcut_pairs=1e-4``,
``tcut_pno=3.33e-7``, ``tcut_mkn=1e-3``, and
``residual_domain="extended"``. The full-domain comparison below sets every
pair, PNO, PAO-domain, and occupied-coupling screen to zero.)

Honest scope: ``LocalCCSDOptions.coupling_radius`` (bohr) restricts each
pair's occupied coupling set to a local neighbourhood — the lever that
bounds the O(N⁴) occupied coupling. Its 12-bohr default is a bit-identical
no-op on molecules under ~12-bohr extent (this water included) and a
controlled approximation below the PNO truncation error on larger systems;
set ``coupling_radius=0`` for the exact full-coupling reference the ratchet
pins. The default extended-domain residual currently uses NumPy; the compiled
target-pair kernel serves only the explicit legacy ``residual_domain="pair"``
mode. The solver carries a generous, overridable
``LocalCCSDOptions.max_nbf`` guard. See
``docs/user_guide/dlpno_mp2.md`` and
``docs/tutorial/dlpno_local_correlation.md``.

Run:
    python input-h2o-dlpno-ccsd.py > console.log

Copy this script to a separate run directory first, and use the installed
engine's Python. Coordinates are in bohr and energies are in hartree.
The comparison table is printed to stdout; console.log captures it together
with progress messages. Existing files with the same stem are overwritten.

Outputs (next to this script):
    input-h2o-dlpno-ccsd.out     — banner + RHF trace + DLPNO-CCSD block
    input-h2o-dlpno-ccsd.molden  — orbitals at the RHF solution
    input-h2o-dlpno-ccsd.system  — provenance manifest
"""

from pathlib import Path

import vibeqc as vq
from vibeqc.dlpno.ccsd_local_solver import LocalCCSDOptions, run_local_dlpno_ccsd

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem  # → "input-h2o-dlpno-ccsd"

mol = vq.Molecule(
    [
        vq.Atom(8, [0.0, 0.00, 0.00]),
        vq.Atom(1, [0.0, 1.43, -0.98]),
        vq.Atom(1, [0.0, -1.43, -0.98]),
    ]
)

# 1. One-call DLPNO-CCSD with the auto-resolved RI fitting basis.
result = vq.run_job(
    mol,
    basis="cc-pvdz",
    method="dlpno-ccsd",
    frozen_core="published",
    dlpno_thresholds="normal",
    output=HERE / STEM,
)
cc = result.dlpno_ccsd
if not result.converged or not cc.converged:
    raise RuntimeError("RHF or DLPNO amplitudes did not converge; inspect the job log")

# 2. Compare default and full-domain local CCSD on one shared RHF reference.
basis = vq.BasisSet(mol, "cc-pvdz")
rhf_opts = vq.RHFOptions()
rhf_opts.conv_tol_energy = 1e-10
hf = vq.run_rhf(mol, basis, rhf_opts)
if not hf.converged:
    raise RuntimeError("Comparison RHF did not converge")

from vibeqc.density_fitting import DensityFitting

df = DensityFitting(basis, vq.BasisSet(mol, "cc-pvdz-ri"), aux_basis_name="cc-pvdz-ri")

n_frozen = vq.chemical_core_orbital_count(mol)
full_domain = run_local_dlpno_ccsd(
    mol,
    basis,
    hf,
    df,
    LocalCCSDOptions(
        n_frozen=n_frozen,
        tcut_pairs=0.0,
        tcut_pno=0.0,
        tcut_mkn=0.0,
        coupling_radius=0.0,
        residual_domain="full",
    ),
)
if not full_domain.converged:
    raise RuntimeError("Full-domain local CCSD did not converge")
default = run_local_dlpno_ccsd(
    mol, basis, hf, df, LocalCCSDOptions(n_frozen=n_frozen)
)
if not default.converged:
    raise RuntimeError("Default local CCSD did not converge")
avg = sum(default.pno_per_pair.values()) / max(len(default.pno_per_pair), 1)

# Keep comparison output separate from the finalized run_job artifacts.
print(f"  E(RHF)                                = {hf.energy:14.8f} Ha")
print(f"  E_corr(full-domain local CCSD)         = {full_domain.e_corr:14.8f} Ha")
print(
    f"  E_corr(default NormalPNO)             = {default.e_corr:14.8f} Ha"
    f"   ({default.e_corr / full_domain.e_corr:.4%} recovery)"
)
delta_uha = 1e6 * (default.e_corr - full_domain.e_corr)
print(f"  delta(default - full domain)           = {delta_uha:14.4f} uHa")
print(f"  average retained PNOs per pair         = {avg:.1f}")
