# Lattice-scan compute-cluster payloads

compute-cluster submission drivers for the bulk-3D lattice ladder (diamond C, MgO,
corundum, fcc Cu). Each takes lattice constants in Angstrom as argv, prints
one line per point, and exits 1 if fewer than three points converge; the
three lowest-energy points are fitted to a parabola and the minimum is
reported. These are the payloads referenced by the seccm-fix-checks and
lattice-scans campaign lanes on compute-cluster (`vq`, an explicitly configured queue).

| File | System | Engine path |
|---|---|---|
| `run_diamond.py` | C diamond 2x2x2 | GFN2-SECCM plain WS kernel |
| `run_mgo.py` | MgO rocksalt 2x2x2 | GFN2-SECCM, madelung embedding |
| `run_corundum.py` | Al2O3 corundum hex cell | GFN2-SECCM, T=0.005 |
| `run_cu.py` | fcc Cu 2x2x2 | GFN2-SECCM, T=0.005 |
| `run_cu444.py` | fcc Cu 4x4x4 | GFN2-SECCM, T=0.005 |
| `run_cu444_ewald.py` | fcc Cu 4x4x4 | GFN2-SECCM, ewald_gamma + newton + T=0.002 |
| `run_corundum_ewald.py` | Al2O3 corundum hex cell | GFN2-SECCM, ewald_gamma + newton + T=0.002 |

Known scan outcomes (coarse, 2026-08-24 tree, recorded in the IID 130/301
tracker notes): MgO min 4.2885 A (+1.8% vs experiment), diamond min
3.5241 A (-1.2%), Cu monotonic or fail-closed, corundum branch-discontinuous
at the experimental lattice constant.

The 2026-08-25 metal recipe (IID 130) supersedes the plain Cu payloads:
`ewald_gamma=True + scc_mixer="newton" + electronic_temperature <= 0.002`
reaches the sane metallic basin at every scanned a. See
`../scan_cu_ewald_newton.py` and `../xtb_cu_reference.py` for the current
Cu ladder and its out-of-process xtb reference.
