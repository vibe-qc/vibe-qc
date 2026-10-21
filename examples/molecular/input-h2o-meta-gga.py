"""Water, the meta-GGA rungs of the ladder: TPSS, SCAN, r2SCAN against PBE.

System: H2O/def2-SVP at the fixed geometry shared by these examples (bohr).
Method: ``run_job(method="rks", functional=...)`` for the GGA PBE and the
meta-GGAs TPSS, SCAN and r2SCAN, which add the kinetic-energy density tau
to the exchange-correlation input. The script prints total energies, the
HOMO and LUMO eigenvalues and the Kohn-Sham gap; the grid is the default
``grid_level="orca-defgrid3"``; SCAN is known to be more grid-sensitive than
TPSS, which is one reason r2SCAN exists, so use ``grid_level="fine"`` for
benchmark work.

Files (next to this script): one set per functional,
    input-h2o-meta-gga-<functional>.out / .molden / .system / ...

Success: total energies PBE -76.270962, TPSS -76.358946, SCAN -76.323888,
r2SCAN -76.314754 Ha, with Kohn-Sham gaps 6.62, 7.04, 8.13 and 8.14 eV.
SCAN and r2SCAN agree with each other to 0.01 eV in the gap while differing
by 9 mHa in total energy, and all three meta-GGAs open the gap relative to
PBE. Tolerance: 1e-6 Ha on the energies (vibe-qc 0.17.x development build,
Apple M5 Max, about 20 s for the four runs).
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
    ("pbe", "GGA"),
    ("tpss", "meta-GGA"),
    ("scan", "meta-GGA, constraint-based"),
    ("r2scan", "meta-GGA, regularised SCAN"),
]

rows = []
for functional, kind in FUNCTIONALS:
    result = vq.run_job(mol, basis="def2-svp", method="rks", functional=functional,
                        output=str(HERE / f"{STEM}-{functional}"))
    eps = np.asarray(result.mo_energies, dtype=float)
    rows.append((functional, kind, float(result.energy), eps[N_OCC - 1], eps[N_OCC], result.n_iter))

print("\n  RKS/def2-SVP water: GGA versus meta-GGA")
print(f"  {'functional':10s} {'E_total (Ha)':>16s} {'HOMO (eV)':>10s} {'LUMO (eV)':>10s} "
      f"{'gap (eV)':>9s} {'iters':>6s}  rung")
for functional, kind, e, homo, lumo, n_iter in rows:
    print(f"  {functional:10s} {e:16.8f} {homo * EV:10.3f} {lumo * EV:10.3f} "
          f"{(lumo - homo) * EV:9.3f} {n_iter:6d}  {kind}")
e_pbe = rows[0][2]
for functional, _kind, e, *_rest in rows[1:]:
    print(f"  E({functional}) - E(pbe) = {(e - e_pbe) * 1000:+9.3f} mHa")
