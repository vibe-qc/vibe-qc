#!/usr/bin/env python
"""Real-Gamma ladder parity against PySCF KRHF (RG-13, GitLab #519).

The real-Gamma producer, ``run_ccm_rhf_direct``, evaluates the neutral
fitted-torus Hamiltonian on the ``nrep`` supercell at Gamma. By band folding
that is the unit cell's periodic HF on the Gamma-centred ``nrep`` k-mesh
(Theorem 1 of ``docs/aiccm2026dev_a.md``); the real-Gamma and neutral-Bloch
producers agree to 1e-10 Ha/cell in the test suite. This script measures the
cross-code step: on every rung of the ladder the real-Gamma energy per cell is
compared with PySCF's KRHF on the same mesh.

Both codes get the same Hamiltonian ingredients:

* the AO and auxiliary basis are vibe-qc's own shells, transported to PySCF
  shell by shell (libint stores contraction coefficients that already carry the
  primitive norm, which is proportional to ``alpha**((2l+3)/4)``; dividing it out
  leaves PySCF's convention up to a per-shell factor PySCF renormalises away);
* the exact-exchange ``q = 0`` treatment is ``exxdiv='ewald'`` on both sides,
  and the two Madelung constants are recorded next to each other;
* the SCF starts from SAD on the real-Gamma side and the converged state is
  checked for an RHF internal instability, so a rung never compares a saddle.

Fits. vibe-qc's SR/LR range-separated fit (``fit_source="sr"``) is compared
with PySCF's range-separated GDF (``rs_density_fit``); both implement Ye and
Berkelbach, J. Chem. Phys. 154, 131104 (2021), doi:10.1063/5.0046617. vibe-qc's
classic fit with its auto-sized high-|G| tail is compared with PySCF's GDF
(``density_fit``) on the cells where that tail is affordable: the MgO tail
(3291.6 Ha) belongs to the cost class of GitLab #258, so MgO runs the SR fit
only.

The cells are the ladder set of the production criteria
(``docs/aiccm2026dev_a_production.md``): LiH, MgO and diamond from the test set
(``testset.py``, CRYSTAL23 lattice constants) and the tail-parity tests' fcc Be,
at STO-3G, plus MgO at pob-DZVP-rev2. PySCF runs in a child interpreter;
vibe-qc never imports it.

Usage
-----
    python ladder_pyscf.py [--cases c-diamond lih-rocksalt mgo be-fcc ...]
                           [--meshes 1,1,1 2,2,2] [--out FILE]

Writes ``results_ladder_pyscf.json`` next to this script unless ``--out`` says
otherwise, and prints the ladder table.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import testset  # noqa: E402

import vibeqc as vq  # noqa: E402
from vibeqc.aux_basis import default_aux_for, make_aux_basis_set  # noqa: E402
from vibeqc.periodic.ccm import CCMSystem  # noqa: E402
from vibeqc.periodic.ccm.direct import (  # noqa: E402
    ccm_exchange_q0_madelung,
    run_ccm_rhf_direct,
)
from vibeqc.periodic.ccm.fit_source import RangeSeparatedFitSource  # noqa: E402



def _be_fcc():
    """fcc Be, a = 3.20 Angstrom: the tail-parity tests' cheapest tight-core cell."""
    a = 3.20 * testset.ANG
    lat = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    return vq.PeriodicSystem(3, lat, [vq.Atom(4, [0.0, 0.0, 0.0])])


# The ladder set of the production criteria: LiH, MgO, diamond and fcc Be.
# ``classic`` says whether the classic fit's tail is affordable on the cell.
MESHES = ((1, 1, 1), (2, 2, 2))
CASES = {
    "c-diamond": dict(build=lambda: testset.build("c-diamond"), basis="sto-3g",
                      classic=True),
    "lih-rocksalt": dict(build=lambda: testset.build("lih-rocksalt"), basis="sto-3g",
                         classic=True),
    "mgo": dict(build=lambda: testset.build("mgo"), basis="sto-3g", classic=False),
    "be-fcc": dict(build=_be_fcc, basis="sto-3g", classic=True,
                   meshes=((1, 1, 1), (2, 1, 1), (2, 2, 2))),
    "mgo/pob-dzvp-rev2": dict(build=lambda: testset.build("mgo"),
                              basis="pob-dzvp-rev2", classic=False,
                              meshes=((1, 1, 1),)),
}
SCF = dict(conv_tol=1e-10, conv_tol_grad=1e-7, max_iter=300)
PYSCF_CONV_TOL = 1e-11
# PySCF's integral precision. vibe-qc's SR fit plans from a raw integral error
# of 1e-10, so PySCF runs at the same target.
PYSCF_PRECISION = 1e-10

_PYSCF_CHILD = r"""
import json, sys, time
import numpy as np
import pyscf
from pyscf.data.elements import ELEMENTS
from pyscf.pbc import gto, scf, tools

p = json.load(open(sys.argv[1]))
sym = {int(z): ELEMENTS[int(z)] for z in p["ao_basis"]}
cell = gto.Cell()
cell.a = np.asarray(p["lattice_rows"], dtype=float)
cell.atom = [[sym[int(z)], xyz] for z, xyz in p["atoms"]]
cell.unit = "Bohr"
cell.basis = {sym[int(z)]: shells for z, shells in p["ao_basis"].items()}
cell.precision = p["precision"]
cell.verbose = 0
cell.build()
aux = {sym[int(z)]: shells for z, shells in p["aux_basis"].items()}
out = {"pyscf_version": pyscf.__version__, "nao": int(cell.nao_nr()), "rows": []}
for mesh in p["meshes"]:
    kpts = cell.make_kpts(mesh)
    row = {"mesh": list(mesh),
           "madelung": float(tools.pbc.madelung(cell, kpts))}
    for fit in p["fits"]:
        mf = scf.KRHF(cell, kpts, exxdiv="ewald")
        if fit == "rsgdf":
            mf = mf.rs_density_fit(auxbasis=aux)
        else:
            mf = mf.density_fit(auxbasis=aux)
        mf.conv_tol = p["conv_tol"]
        mf.max_cycle = 200
        t0 = time.perf_counter()
        e = float(mf.kernel())
        rec = {"energy": e, "converged": bool(mf.converged),
               "wall_s": time.perf_counter() - t0,
               "naux": int(mf.with_df.auxcell.nao_nr())}
        omega = getattr(mf.with_df, "omega", None)
        if omega is not None:
            rec["omega"] = float(omega)
        row[fit] = rec
    out["rows"].append(row)
print("PYSCF-RESULT:" + json.dumps(out), flush=True)
"""


def _pyscf_shells(basis: "vq.BasisSet", atom_index: int) -> list:
    """The shells on one atom in PySCF's format ``[l, [alpha, c], ...]``."""
    shells = []
    for s in basis.shells():
        if int(s.atom_index) != atom_index:
            continue
        l = int(s.l)
        if l >= 2 and not s.pure:
            raise ValueError("a Cartesian shell has no PySCF spherical twin")
        alpha = np.asarray(s.exponents, dtype=float)
        coef = np.asarray(s.coefficients, dtype=float) / alpha ** ((2 * l + 3) / 4.0)
        shells.append([l] + [[float(a), float(c)] for a, c in zip(alpha, coef)])
    return shells


def _transported_bases(cell: "vq.PeriodicSystem", basis_name: str):
    """vibe-qc's AO and auxiliary shells, one entry per element."""
    mol = cell.unit_cell_molecule()
    ao = vq.BasisSet(mol, basis_name)
    aux_name = default_aux_for(basis_name)
    aux = make_aux_basis_set(mol, aux_name=aux_name, drop_eta=0.0)
    ao_by_z, aux_by_z = {}, {}
    for i, atom in enumerate(mol.atoms):
        z = int(atom.Z)
        if z not in ao_by_z:
            ao_by_z[z] = _pyscf_shells(ao, i)
            aux_by_z[z] = _pyscf_shells(aux, i)
    return ao_by_z, aux_by_z, aux_name


def run_pyscf(case: dict, meshes, fits, python: str) -> dict:
    """PySCF KRHF on every mesh, in a child interpreter."""
    cell = case["build"]()
    ao_by_z, aux_by_z, aux_name = _transported_bases(cell, case["basis"])
    mol = cell.unit_cell_molecule()
    payload = {
        "lattice_rows": np.asarray(cell.lattice, dtype=float).T.tolist(),
        "atoms": [[int(a.Z), [float(x) for x in a.xyz]] for a in mol.atoms],
        "ao_basis": {str(z): v for z, v in ao_by_z.items()},
        "aux_basis": {str(z): v for z, v in aux_by_z.items()},
        "meshes": [list(m) for m in meshes],
        "fits": list(fits),
        "conv_tol": PYSCF_CONV_TOL,
        "precision": PYSCF_PRECISION,
    }
    with tempfile.TemporaryDirectory() as tmp:
        spec = Path(tmp) / "spec.json"
        spec.write_text(json.dumps(payload))
        proc = subprocess.run([python, "-I", "-c", _PYSCF_CHILD, str(spec)],
                              capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"PySCF child failed:\n{proc.stderr[-4000:]}")
    line = next(x for x in proc.stdout.splitlines() if x.startswith("PYSCF-RESULT:"))
    out = json.loads(line[len("PYSCF-RESULT:"):])
    out["aux_basis"] = aux_name
    return out


def _vibeqc_run(ccm, **kw) -> dict:
    t0 = time.perf_counter()
    res = run_ccm_rhf_direct(ccm, initial_guess="sad", stability_check=True, **kw, **SCF)
    return {
        "energy_per_cell": float(res.energy) / ccm.n_cells,
        "converged": bool(res.converged),
        "stability_eigenvalue": float(res.stability_eigenvalue),
        "internal_instability": bool(res.internal_instability),
        "fit_source": res.fit_source,
        "tail_ke_cutoff": res.rsgdf_tail_ke_cutoff,
        "neutral_source_identity": res.neutral_source.identity,
        "wall_s": time.perf_counter() - t0,
    }


def run_vibeqc(case: dict, mesh, *, classic: bool, sr: bool = True,
               storage: str = "dense") -> dict:
    """The real-Gamma producer on the ``mesh`` supercell, SR and classic fits.

    The storage of the fitted integrals does not change the operator (block
    and dense agree to 1e-10 Ha/cell in the test suite); block storage keeps
    the larger meshes in memory.
    """
    ccm = CCMSystem(case["build"](), tuple(mesh), case["basis"])
    row = {"mesh": list(mesh), "n_cells": int(ccm.n_cells), "storage": storage,
           "madelung": float(ccm_exchange_q0_madelung(ccm))}
    if sr:
        row["sr"] = _vibeqc_run(ccm, fit_source=RangeSeparatedFitSource(),
                                cderi_storage=storage)
    if classic:
        row["classic"] = _vibeqc_run(ccm, cderi_storage=storage)
    return row


def _runtime_identity() -> dict:
    ident = {"vibeqc_version": vq.__version__}
    try:
        head = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short=12", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        # Only the computing sources count: a modified page or study script
        # does not change a number.
        dirty = subprocess.run(["git", "-C", str(HERE.parents[1]), "status", "--porcelain",
                                "--untracked-files=no", "--", "python", "cpp"],
                               capture_output=True, text=True, check=True).stdout.strip()
        ident["source_commit"] = head + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        ident["source_commit"] = None
    return ident


def _merge(entry: dict, ref: dict, rows: list) -> None:
    """Fold one invocation's rows into a case entry, a mesh replacing its twin."""
    meshes = {tuple(r["mesh"]) for r in rows}
    ref_meshes = {tuple(r["mesh"]) for r in ref["rows"]}
    old_ref = entry.get("pyscf", {"rows": []})
    ref = dict(ref, rows=[r for r in old_ref["rows"] if tuple(r["mesh"]) not in ref_meshes]
               + ref["rows"])
    entry["pyscf"] = ref
    entry["vibeqc"] = ([r for r in entry.get("vibeqc", []) if tuple(r["mesh"]) not in meshes]
                       + rows)

    def order(r):
        return int(np.prod(r["mesh"])), r["mesh"]

    entry["pyscf"]["rows"].sort(key=order)
    entry["vibeqc"].sort(key=order)


def _table(results: dict) -> str:
    lines = ["| system | mesh | fit | vibe-qc real-Gamma (Ha/cell) | PySCF KRHF (Ha/cell) "
             "| difference (uHa/cell) | Madelung difference |",
             "|---|---|---|---|---|---|---|"]
    for name, sysres in results["systems"].items():
        by_mesh = {tuple(r["mesh"]): r for r in sysres["pyscf"]["rows"]}
        for row in sysres["vibeqc"]:
            ref = by_mesh[tuple(row["mesh"])]
            dmad = row["madelung"] - ref["madelung"]
            for ours, theirs in (("sr", "rsgdf"), ("classic", "gdf")):
                if ours not in row or theirs not in ref:
                    continue
                e, r = row[ours]["energy_per_cell"], ref[theirs]["energy"]
                mesh = "x".join(str(n) for n in row["mesh"])
                lines.append(f"| {name} | {mesh} | {ours} / {theirs} | {e:.10f} | {r:.10f} "
                             f"| {1e6 * (e - r):+.2f} | {dmad:+.1e} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    ap.add_argument("--meshes", nargs="+", default=None,
                    help="meshes such as 1,1,1 2,2,2 (default: each case's own)")
    ap.add_argument("--no-classic", action="store_true",
                    help="run the SR fit only on every cell")
    ap.add_argument("--no-sr", action="store_true",
                    help="run the classic fit only (where the case affords it)")
    ap.add_argument("--storage", default="dense", choices=("dense", "block"),
                    help="cderi storage of the real-Gamma runs")
    ap.add_argument("--pyscf-meshes", nargs="+", default=[],
                    help="extra meshes run by PySCF alone, the bridge from the "
                         "rungs to the CRYSTAL limit")
    ap.add_argument("--no-vibeqc", action="store_true",
                    help="run PySCF only (with --merge, the vibe-qc rows already "
                         "in --out are kept)")
    ap.add_argument("--pyscf-python", default=sys.executable,
                    help="interpreter with PySCF for the child runs")
    ap.add_argument("--out", type=Path, default=HERE / "results_ladder_pyscf.json")
    ap.add_argument("--merge", action="store_true",
                    help="add the rows to an existing --out file, replacing equal meshes")
    args = ap.parse_args()
    chosen = (None if args.meshes is None
              else [tuple(int(x) for x in m.split(",")) for m in args.meshes])

    runtime = _runtime_identity()
    results = {"scf": SCF, "pyscf_conv_tol": PYSCF_CONV_TOL,
               "pyscf_precision": PYSCF_PRECISION, "systems": {}}
    if args.merge and args.out.exists():
        results = json.loads(args.out.read_text())
        legacy = results.pop("runtime", None)
        for entry in results["systems"].values():
            for row in entry.get("vibeqc", []):
                row.setdefault("runtime", legacy)
    for name in args.cases:
        case = CASES[name]
        meshes = chosen or list(case.get("meshes", MESHES))
        classic = case["classic"] and not args.no_classic
        sr = not args.no_sr
        if not (classic or sr):
            raise SystemExit(f"{name}: --no-sr leaves no affordable fit on this case")
        fits = tuple(f for f, on in (("rsgdf", sr), ("gdf", classic)) if on)
        extra = [tuple(int(x) for x in m.split(",")) for m in args.pyscf_meshes]
        if args.no_vibeqc:
            pyscf_meshes = extra or list(meshes)
        else:
            pyscf_meshes = list(meshes) + [m for m in extra if m not in meshes]
        t0 = time.perf_counter()
        ref = run_pyscf(case, pyscf_meshes, fits, args.pyscf_python)
        print(f"{name}: PySCF done in {time.perf_counter() - t0:.0f} s", flush=True)
        rows = []
        for mesh in ([] if args.no_vibeqc else meshes):
            t0 = time.perf_counter()
            rows.append(run_vibeqc(case, mesh, classic=classic, sr=sr,
                                   storage=args.storage))
            print(f"{name} {mesh}: vibe-qc done in {time.perf_counter() - t0:.0f} s",
                  flush=True)
        for row in rows:
            row["runtime"] = runtime
        _merge(results["systems"].setdefault(name, {"basis": case["basis"]}), ref, rows)
        args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
