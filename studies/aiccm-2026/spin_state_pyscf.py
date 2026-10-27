#!/usr/bin/env python
"""Matched PySCF reference for the per-cell doublet sector (#519 RG-14).

Li STO-3G cube (7 bohr) at (2,1,1), per-cell doublet: PySCF KUHF and
KUKS/PBE on the Gamma-centred 2x1x1 mesh with ``nelec`` pinned to
``(N_k n_alpha, N_k n_beta)`` and ``exxdiv="ewald"``, vibe-qc's AO and
auxiliary shells transported shell by shell (see ``ladder_pyscf.py``),
PySCF's range-separated GDF against vibe-qc's SR fit and PySCF's GDF
against the classic fit, on both neutral producers run in the same sector.
The XC quadrature of the KUKS rows is not matched (each code's own grid).
vibe-qc never imports PySCF; the child interpreter runs with ``-I``.
"""
import json, subprocess, sys, tempfile, time, warnings
from pathlib import Path
warnings.simplefilter("ignore")
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vibeqc as vq
from vibeqc import Atom, PeriodicSystem, PeriodicRHFOptions, PeriodicKSOptions
from vibeqc.periodic.ccm import CCMSystem, assert_comparable_neutral_results
from vibeqc.periodic.ccm.direct import run_ccm_uhf_direct, run_ccm_uks_direct, ccm_exchange_q0_madelung
from vibeqc.periodic.ccm.ri import run_ccm_uhf_gdf, run_ccm_uks_gdf
from vibeqc.periodic.ccm.fit_source import RangeSeparatedFitSource
from ladder_pyscf import _transported_bases, PYSCF_PRECISION

CHILD = r"""
import json, sys, time
import numpy as np
import pyscf
from pyscf.data.elements import ELEMENTS
from pyscf.pbc import gto, scf, dft, tools
p = json.load(open(sys.argv[1]))
sym = {int(z): ELEMENTS[int(z)] for z in p["ao_basis"]}
cell = gto.Cell(); cell.a = np.asarray(p["lattice_rows"], float)
cell.atom = [[sym[int(z)], xyz] for z, xyz in p["atoms"]]; cell.unit = "Bohr"
cell.basis = {sym[int(z)]: s for z, s in p["ao_basis"].items()}
cell.spin = p["spin_per_cell"]; cell.precision = p["precision"]; cell.verbose = 0; cell.build()
aux = {sym[int(z)]: s for z, s in p["aux_basis"].items()}
kpts = cell.make_kpts(p["mesh"]); nk = len(kpts)
out = {"pyscf": pyscf.__version__, "madelung": float(tools.pbc.madelung(cell, kpts)), "rows": {}}
for method in p["methods"]:
    for fit in p["fits"]:
        if method == "uhf":
            mf = scf.KUHF(cell, kpts, exxdiv="ewald")
        else:
            mf = dft.KUKS(cell, kpts, exxdiv="ewald"); mf.xc = "pbe"
        mf = mf.rs_density_fit(auxbasis=aux) if fit == "sr" else mf.density_fit(auxbasis=aux)
        mf.nelec = (nk * p["n_alpha"], nk * p["n_beta"])
        mf.conv_tol = 1e-11; mf.max_cycle = 200
        t0 = time.perf_counter(); e = float(mf.kernel())
        if tuple(int(n) for n in mf.nelec) != (nk * p["n_alpha"], nk * p["n_beta"]):
            raise RuntimeError("PySCF changed nelec")
        occ_a = np.concatenate([np.asarray(o) for o in mf.mo_occ[0]]); occ_b = np.concatenate([np.asarray(o) for o in mf.mo_occ[1]])
        out["rows"][f"{method}/{fit}"] = {"energy": e, "converged": bool(mf.converged), "wall_s": time.perf_counter() - t0,
            "n_alpha_total": float(occ_a.sum()), "n_beta_total": float(occ_b.sum()),
            "integer_occ": bool(np.all(np.abs(occ_a - np.round(occ_a)) < 1e-8) and np.all(np.abs(occ_b - np.round(occ_b)) < 1e-8)),
            "s2": float(mf.spin_square()[0]) if method == "uhf" else None}
print("PYSCF-RESULT:" + json.dumps(out), flush=True)
"""

cell = PeriodicSystem(3, np.diag([7.0, 7.0, 7.0]), [Atom(3, [3.5, 3.5, 3.5])], 0, 2)
mesh = (2, 1, 1)
ccm = CCMSystem(cell, mesh, "sto-3g", spin_state="per-cell")
state = ccm.spin_state
print("sector:", state.describe())
TIGHT = dict(conv_tol=1e-12, conv_tol_grad=1e-9, max_iter=300)
def bopts(ks):
    o = PeriodicKSOptions() if ks else PeriodicRHFOptions(); o.conv_tol_energy = 1e-11; o.conv_tol_grad = 1e-8; o.max_iter = 200; return o

rows = {}
for fit in ("sr", "classic"):
    fk = dict(fit_source=RangeSeparatedFitSource()) if fit == "sr" else {}
    bk = dict(fit_source="sr") if fit == "sr" else {}
    rg = run_ccm_uhf_direct(ccm, **fk, **TIGHT)
    nb = run_ccm_uhf_gdf(ccm, options=bopts(False), **bk)
    ident, st = assert_comparable_neutral_results([rg, nb])
    rows[f"uhf/{fit}"] = dict(real_gamma=rg.energy / ccm.n_cells, bloch=nb.energy, s2=rg.s_squared, ident=ident[:12], fit=rg.fit_source)
    rg = run_ccm_uks_direct(ccm, "pbe", **fk, **TIGHT)
    nb = run_ccm_uks_gdf(ccm, "pbe", options=bopts(True), **bk)
    ident, st = assert_comparable_neutral_results([rg, nb])
    rows[f"uks/{fit}"] = dict(real_gamma=rg.energy / ccm.n_cells, bloch=nb.energy, ident=ident[:12], fit=rg.fit_source)
print("vibe-qc madelung", ccm_exchange_q0_madelung(ccm))

ao_by_z, aux_by_z, aux_name = _transported_bases(cell, "sto-3g")
mol = cell.unit_cell_molecule()
payload = {"lattice_rows": np.asarray(cell.lattice, float).T.tolist(),
           "atoms": [[int(a.Z), [float(x) for x in a.xyz]] for a in mol.atoms],
           "ao_basis": {str(z): v for z, v in ao_by_z.items()}, "aux_basis": {str(z): v for z, v in aux_by_z.items()},
           "mesh": list(mesh), "spin_per_cell": int(state.per_cell[0] - state.per_cell[1]),
           "n_alpha": int(state.per_cell[0]), "n_beta": int(state.per_cell[1]),
           "methods": ["uhf", "uks"], "fits": ["sr", "classic"], "precision": PYSCF_PRECISION}
with tempfile.TemporaryDirectory() as tmp:
    spec = Path(tmp) / "spec.json"; spec.write_text(json.dumps(payload))
    proc = subprocess.run([sys.executable, "-I", "-c", CHILD, str(spec)], capture_output=True, text=True)
if proc.returncode != 0:
    print(proc.stderr[-3000:]); sys.exit(1)
res = json.loads(next(l for l in proc.stdout.splitlines() if l.startswith("PYSCF-RESULT:"))[13:])
print("pyscf", res["pyscf"], "madelung", res["madelung"], "aux", aux_name)
print(f"{'row':12s} {'real-gamma':>16s} {'bloch':>16s} {'pyscf':>16s} {'rg-pyscf uHa':>14s} {'rg-bloch':>10s}")
for key, row in rows.items():
    pr = res["rows"][key]
    print(f"{key:12s} {row['real_gamma']:16.10f} {row['bloch']:16.10f} {pr['energy']:16.10f} {(row['real_gamma']-pr['energy'])*1e6:14.3f} {(row['real_gamma']-row['bloch']):10.1e}  pyscf conv={pr['converged']} int_occ={pr['integer_occ']} na/nb={pr['n_alpha_total']}/{pr['n_beta_total']} s2={pr['s2']} vq_fit={row['fit']}")
print("vibeqc", vq.__version__, vq.__file__)
