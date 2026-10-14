"""Water, range-separated hybrids next to a global hybrid: energies and HOMO levels.

System: H2O/def2-SVP at the fixed geometry shared by these examples (bohr).
Method: ``run_job(method="rks", functional=...)`` for wB97X and CAM-B3LYP
(long-range corrected: full exact exchange at long range), HSE06 (screened
hybrid: exact exchange only at short range) and PBE0 (global hybrid, 25
percent exact exchange everywhere) as the reference point. The script
prints the total energy, the HOMO and LUMO eigenvalues and the gap.

Files (next to this script): one set per functional,
    input-h2o-range-separated-<functional>.out / .molden / .system / ...

Success: total energies wB97X -76.334606, CAM-B3LYP -76.327213,
HSE06 -76.279685, PBE0 -76.273301 Ha, and HOMO eigenvalues -10.85, -9.74,
-7.79 and -8.21 eV. The two long-range corrected functionals push the HOMO
down toward minus the ionisation potential (experiment 12.6 eV), which is
what restoring the correct -1/r exchange potential does; HSE06 screens the
long-range exchange away and its HOMO sits above PBE0's. Tolerance: 1e-6 Ha
on the energies (vibe-qc 0.17.x development build, Apple M5 Max, about 30 s
for the four runs).

Optimisations on range-separated hybrids: the analytic RKS gradient omits
the range-separation terms, so ``run_job(optimize=True)`` and
``optimize_molecule`` fall back to finite differences of the energy for
these functionals (GitLab #571). Energies are production; the fallback is
slower, not less accurate.
"""

from pathlib import Path

import numpy as np

import vibeqc as vq

HERE = Path(__file__).parent
STEM = Path(__file__).stem
EV = 27.211386245988

mol = vq.Molecule([
    vq.Atom(8, [0.000,  0.000,  0.000]),
    vq.Atom(1, [0.000,  1.499, -1.160]),
    vq.Atom(1, [0.000, -1.499, -1.160]),
])
N_OCC = 5
FUNCTIONALS = [
    ("wb97x", "long-range corrected"),
    ("cam-b3lyp", "long-range corrected (Coulomb attenuated)"),
    ("hse06", "screened hybrid (short-range exact exchange)"),
    ("pbe0", "global hybrid, 25 percent exact exchange"),
]

rows = []
for functional, kind in FUNCTIONALS:
    result = vq.run_job(mol, basis="def2-svp", method="rks", functional=functional,
                        output=str(HERE / f"{STEM}-{functional}"))
    eps = np.asarray(result.mo_energies, dtype=float)
    rows.append((functional, kind, float(result.energy), eps[N_OCC - 1], eps[N_OCC], result.n_iter))

print("\n  RKS/def2-SVP water: range-separated versus global hybrid")
print(f"  {'functional':10s} {'E_total (Ha)':>16s} {'HOMO (eV)':>10s} {'LUMO (eV)':>10s} "
      f"{'gap (eV)':>9s} {'iters':>6s}  type")
for functional, kind, e, homo, lumo, n_iter in rows:
    print(f"  {functional:10s} {e:16.8f} {homo * EV:10.3f} {lumo * EV:10.3f} "
          f"{(lumo - homo) * EV:9.3f} {n_iter:6d}  {kind}")
print("  (experimental vertical IP of water: 12.6 eV; -HOMO approaches it only with "
      "long-range exact exchange)")
