#!/usr/bin/env python3
"""DLPNO-CCSD(T) on crystals vs PySCF KRCCSD(T): the external reference of the
periodic GDF route (increments B1, B2 and B2b of
docs/design_dlpno_correlation_source.md).

Three measurements on a crystal: the built-in dilute H2 molecular crystal
(H2 on a simple-cubic lattice, closed shell, genuinely 3-D periodic), or any
closed-shell spec of the regression suite (``--system lih_rocksalt``,
``mgo_rocksalt``, ``nacl_rocksalt``, ``diamond_c``, ...; the same geometry
and the same Angstrom-to-bohr conversion the suite's runners use).

  PART A: full domains against canonical k-point CCSD(T). vibe-qc runs the
    engine with every truncation off on the Gamma-point GDF reference, and on
    a Gamma-centred mesh folded to the real BvK supercell
    (``vibeqc.periodic_dlpno``). PySCF runs KRHF/GDF (``exxdiv='ewald'``) and
    KRCCSD(T) on the same Gamma-centred mesh, twice. Both codes contract bare
    fitted Coulomb factors in the amplitude equations; they differ in the
    Fock operator those equations see. vibe-qc keeps the SCF's Fock, Ewald
    exchange zero-mode term included (the convention of
    ``gamma_gdf_source``); PySCF's ``KRCCSD`` keeps it only with
    ``keep_exxdiv=True`` and by default (``keep_exxdiv=False``) rebuilds the
    Fock without it, which raises every occupied orbital energy by the
    Madelung constant inside the amplitude equations. The script reports
    both: the ``keep_exxdiv=True`` column is the like-for-like comparison
    (integrals and orbital energies agree to the fit tolerance, so the gap
    there is the cross-fit gap between vibe-qc's classic RSGDF and PySCF's
    GDF), and the default column measures what the convention is worth at a
    finite mesh. Which convention the production route should carry is a
    maintainer decision recorded on the design page; both have the same
    thermodynamic limit. Both sides correlate all electrons.

  PART B: the shipped thresholds. The same cells at the engine's defaults
    (NormalPNO, Pipek-Mezey localised occupied block, all electrons) against
    the full-domain numbers of PART A: the local truncation error per cell.

  PART C: the fold identity restated from the outside. The per-cell energy on
    the (1, 1, 2) mesh equals half of the explicit Gamma supercell's.

PySCF runs in a child interpreter and is parsed back; vibe-qc never imports
it (AGENTS.md, ground rule 5).

Usage:
    python examples/regression/parity_periodic_dlpno_vs_pyscf.py
        [--system h2|<regression spec id>] [--basis sto-3g] [--aux NAME]
        [--a 6.0 --d 1.4] [--meshes 1,1,1;1,1,2;2,2,2] [--json PATH]
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903

# vibe-qc and PySCF spell the same basis sets differently.
_PYSCF_BASIS = {"sto-3g": "sto3g", "def2-svp": "def2-svp", "cc-pvdz": "ccpvdz",
                "pob-tzvp": "pob-tzvp"}
# PySCF's per-zeta aliases select its universal JKfit data. The bundled
# per-zeta records have the same exponents, coefficients and angular momenta
# for every element they cover; these aliases do not change the Vibe-QC fit.
_PYSCF_AUX = {f"def2-{zeta}-jk": f"def2-{zeta}-jkfit"
              for zeta in ("svp", "tzvp", "tzvpp", "qzvp", "qzvpp")}


def _canonical_aux_name(aux):
    """Resolve the legacy universal spelling before either backend sees it."""
    return "def2-universal-jkfit" if aux == "def2-universal-jk" else aux

_PYSCF_AUX_PREFLIGHT = r"""
import json, sys
from pyscf.gto import basis as basis_library
p = json.loads(sys.argv[1])
for symbol in sorted({atom[0] for atom in p["atoms"]}):
    try:
        shells = basis_library.load(p["aux"], symbol)
        if not shells:
            raise ValueError("no shells loaded")
    except Exception as exc:
        raise RuntimeError(
            f"PySCF auxiliary preflight failed for {p['aux']!r}, element {symbol}: {exc}. "
            "No default auxiliary substitution is allowed."
        ) from exc
"""

_PYSCF_CHILD = _PYSCF_AUX_PREFLIGHT + r"""
import numpy as np
from pyscf.pbc import gto, scf, cc
cell = gto.Cell()
cell.a = np.asarray(p["lattice"], dtype=float)
cell.atom = [[sym, list(xyz)] for sym, xyz in p["atoms"]]
cell.unit = p["unit"]; cell.basis = p["basis"]; cell.verbose = 0
cell.precision = 1e-10
cell.build()
out = {"aux": None}
for km in p["kmeshes"]:
    kpts = cell.make_kpts(km)            # Gamma-centred, no shift
    mf = scf.KRHF(cell, kpts).density_fit()
    mf.with_df.auxbasis = p["aux"]
    out["aux"] = p["aux"]
    mf.exxdiv = "ewald"
    mf.conv_tol = 1e-11
    e_hf = float(mf.kernel())
    row = {"e_hf": e_hf, "converged": bool(mf.converged)}
    for keep in (True, False):
        mycc = cc.KRCCSD(mf)
        mycc.keep_exxdiv = keep
        mycc.conv_tol = 1e-10
        mycc.conv_tol_normt = 1e-8
        e_corr, _, _ = mycc.kernel()
        e_t = float(mycc.ccsd_t())
        tag = "keep" if keep else "bare"
        row[f"e_corr_{tag}"] = float(e_corr)
        row[f"e_t_{tag}"] = e_t
        row["converged"] = bool(row["converged"] and mycc.converged)
    out["x".join(str(n) for n in km)] = row
print("PYSCF-RESULT:" + json.dumps(out))
"""


def _key(mesh):
    return "x".join(str(n) for n in mesh)


def build_system(args):
    """The vibe-qc cell and the PySCF payload geometry, from one source."""
    import vibeqc as vq

    if args.system == "h2":
        a, d = float(args.a), float(args.d)
        atoms = [("H", [a / 2, a / 2, a / 2 - d / 2]), ("H", [a / 2, a / 2, a / 2 + d / 2])]
        system = vq.PeriodicSystem(
            3, np.diag([a, a, a]), [vq.Atom(1, list(xyz)) for _, xyz in atoms],
        )
        geometry = {"lattice": np.diag([a, a, a]).tolist(), "atoms": atoms, "unit": "Bohr"}
        label = f"H2 crystal, a={a} bohr, d={d} bohr"
        return system, geometry, label
    # A regression-suite spec: the same conversion as core/runner_vibeqc.py
    # (vibe-qc) and core/runner_pyscf.py (PySCF).
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    spec = importlib.import_module(
        f"examples.regression.systems.periodic.{args.system}"
    ).SPEC
    if getattr(spec, "is_open_shell", False):
        raise SystemExit(f"{args.system}: open-shell specs need increment C")
    lat_ang = np.asarray(spec.lattice_ang, dtype=float)
    lat_bohr = lat_ang * ANGSTROM_TO_BOHR
    vq_atoms, py_atoms = [], []
    for at in spec.atoms:
        frac = np.asarray(at.frac, dtype=float)
        vq_atoms.append(vq.Atom(int(at.z), [float(x) for x in (lat_bohr @ frac)]))
        py_atoms.append((at.symbol, [float(x) for x in (lat_ang @ frac)]))
    system = vq.PeriodicSystem(dim=3, lattice=lat_bohr, unit_cell=vq_atoms)
    geometry = {"lattice": lat_ang.tolist(), "atoms": py_atoms, "unit": "A"}
    label = f"{spec.id} ({spec.family}, {len(spec.atoms)} atoms per cell)"
    return system, geometry, label


def _pyscf_payload(geometry, basis, meshes, aux):
    aux = _canonical_aux_name(aux)
    if not aux:
        raise ValueError("resolve the Vibe-QC auxiliary before preparing the reference")
    return {**geometry, "basis": _PYSCF_BASIS.get(basis, basis),
            "kmeshes": [list(m) for m in meshes], "aux": _PYSCF_AUX.get(aux, aux)}


def preflight_pyscf_aux(geometry, basis, aux):
    """Load each required reference block before either expensive calculation."""
    payload = _pyscf_payload(geometry, basis, [], aux)
    with tempfile.TemporaryDirectory(prefix="pyscf-aux-") as scratch:
        out = subprocess.run(
            [sys.executable, "-c", _PYSCF_AUX_PREFLIGHT, json.dumps(payload)],
            capture_output=True, text=True, cwd=scratch,
        )
    if out.returncode:
        raise RuntimeError("PySCF auxiliary preflight failed\n" + out.stderr[-3000:])


def pyscf_kccsd_t(geometry, basis, meshes, aux):
    payload = _pyscf_payload(geometry, basis, meshes, aux)
    # PySCF's (T) kernel writes an HDF5 scratch file into its working
    # directory; keep that out of the checkout.
    with tempfile.TemporaryDirectory(prefix="pyscf-kccsd-") as scratch:
        out = subprocess.run(
            [sys.executable, "-c", _PYSCF_CHILD, json.dumps(payload)],
            capture_output=True, text=True, cwd=scratch,
        )
    for line in out.stdout.splitlines():
        if line.startswith("PYSCF-RESULT:"):
            return json.loads(line[len("PYSCF-RESULT:"):])
    raise RuntimeError("no PySCF result\nSTDERR:\n" + out.stderr[-3000:])


def _full_options():
    from vibeqc.dlpno.ccsd_local_solver import LocalCCSDOptions

    return LocalCCSDOptions(
        localise="none", n_frozen=0, tcut_pno=0.0, tcut_mkn=0.0,
        tcut_pairs=0.0, coupling_radius=0.0, compute_triples=True,
    )


def vibeqc_runs(system, basis, meshes, aux):
    """Full-domain and shipped-default DLPNO-CCSD(T) per mesh, per cell."""
    from vibeqc.periodic_dlpno import run_periodic_dlpno_ccsd

    aux = _canonical_aux_name(aux)
    rows = {}
    for mesh in meshes:
        t0 = time.time()
        exact = run_periodic_dlpno_ccsd(
            system, basis, kmesh=mesh, with_triples=True, localise="none",
            cc_options=_full_options(), aux_basis=aux, progress=False,
        )
        t_full = time.time() - t0
        t0 = time.time()
        shipped = run_periodic_dlpno_ccsd(
            system, basis, kmesh=mesh, with_triples=True, aux_basis=aux, progress=False,
        )
        t_ship = time.time() - t0
        rows[_key(mesh)] = {
            "aux": getattr(exact.hf_result, "aux_basis_name", None),
            "n_cells": exact.n_cells,
            "n_bf": exact.n_bf, "n_occ": exact.n_occ,
            "full": {"e_hf": exact.e_hf_per_cell, "e_corr": exact.e_corr_per_cell,
                     "e_t": exact.e_t_per_cell, "converged": exact.converged,
                     "seconds": t_full},
            "shipped": {"e_hf": shipped.e_hf_per_cell, "e_corr": shipped.e_corr_per_cell,
                        "e_t": shipped.e_t_per_cell, "converged": shipped.converged,
                        "n_pairs": int(shipped.cc_result.n_pairs),
                        "localisation": shipped.localisation, "seconds": t_ship},
            "source_kind": exact.source_kind,
            "same_source": exact.same_source,
            "exchange_q0": exact.exchange_q0,
        }
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--system", default="h2",
                    help="'h2' (built-in box) or a regression spec id such as lih_rocksalt")
    ap.add_argument("--basis", default="sto-3g")
    ap.add_argument("--aux", default=None, type=_canonical_aux_name,
                    help="RI auxiliary for both codes; def2-universal-jk aliases "
                         "def2-universal-jkfit (default: the driver's)")
    ap.add_argument("--a", type=float, default=6.0, help="h2: cubic lattice constant, bohr")
    ap.add_argument("--d", type=float, default=1.4, help="h2: H-H distance, bohr")
    ap.add_argument("--meshes", default="1,1,1;1,1,2;2,2,2")
    ap.add_argument("--json", default=None, help="write the measured numbers here")
    args = ap.parse_args()
    meshes = [tuple(int(x) for x in m.split(",")) for m in args.meshes.split(";")]

    from vibeqc.periodic_toroidal_mp2 import build_toroidal_supercell

    system, geometry, label = build_system(args)
    if not args.aux:
        from vibeqc.aux_basis import default_aux_for

        args.aux = default_aux_for(args.basis)
    preflight_pyscf_aux(geometry, args.basis, args.aux)
    print(f"=== DLPNO-CCSD(T) on crystals vs PySCF KRCCSD(T): {label}, {args.basis} ===\n")

    vqr = vibeqc_runs(system, args.basis, meshes, args.aux)
    aux = next(iter(vqr.values()))["aux"]
    py = pyscf_kccsd_t(geometry, args.basis, meshes, aux)
    first = next(iter(vqr.values()))
    print(f"vibe-qc RI auxiliary: {aux}; PySCF auxiliary: {py.get('aux')}; "
          f"{first['n_bf']} basis functions and {first['n_occ']} occupied orbitals per cell "
          f"on the first mesh (all electrons correlated on both sides)\n")

    print("PART A: full domains vs PySCF KRCCSD(T), per cell (Ha); gaps vq - PySCF in µHa")
    print(f"  {'mesh':>9s} {'cells':>5s} {'E_HF vq':>16s} {'ΔE_HF':>8s} {'E_corr vq':>14s} "
          f"{'Δ keep_exxdiv':>13s} {'Δ PySCF default':>15s} {'E_(T) vq':>12s} {'Δ(T) keep':>10s}")
    a_rows = {}
    for mesh in meshes:
        k = _key(mesh)
        v, p = vqr[k]["full"], py[k]
        d_hf = (v["e_hf"] - p["e_hf"]) * 1e6
        d_keep = (v["e_corr"] - p["e_corr_keep"]) * 1e6
        d_bare = (v["e_corr"] - p["e_corr_bare"]) * 1e6
        d_t = ((v["e_t"] or 0.0) - p["e_t_keep"]) * 1e6
        a_rows[k] = {"d_hf_uha": d_hf, "d_corr_keep_exxdiv_uha": d_keep,
                     "d_corr_pyscf_default_uha": d_bare, "d_t_keep_exxdiv_uha": d_t}
        print(f"  {str(mesh):>9s} {vqr[k]['n_cells']:>5d} {v['e_hf']:16.10f} {d_hf:8.2f} "
              f"{v['e_corr']:14.10f} {d_keep:13.2f} {d_bare:15.2f} {(v['e_t'] or 0.0):12.3e} {d_t:10.3f}"
              f"   [{vqr[k]['source_kind']}, {'conv' if v['converged'] and p['converged'] else 'NOT CONV'}]")
    print("  keep_exxdiv: PySCF correlates on the same Fock as vibe-qc (Ewald exchange "
          "term kept), so that gap is the cross-fit gap on top of ΔE_HF; the PySCF "
          "default rebuilds the Fock without the term, a different finite-mesh "
          "convention with the same thermodynamic limit.\n")

    print("PART B: shipped thresholds (NormalPNO, Pipek-Mezey) vs full domains, per cell")
    print(f"  {'mesh':>9s} {'pairs':>5s} {'E_corr shipped':>16s} {'Δ vs full (µHa)':>16s} "
          f"{'E_(T) shipped':>14s} {'Δ (µHa)':>9s} {'t_full s':>9s} {'t_ship s':>9s}")
    b_rows = {}
    for mesh in meshes:
        k = _key(mesh)
        f, s = vqr[k]["full"], vqr[k]["shipped"]
        d_c = (s["e_corr"] - f["e_corr"]) * 1e6
        d_t = ((s["e_t"] or 0.0) - (f["e_t"] or 0.0)) * 1e6
        b_rows[k] = {"d_corr_uha": d_c, "d_t_uha": d_t}
        print(f"  {str(mesh):>9s} {s['n_pairs']:>5d} {s['e_corr']:16.10f} {d_c:16.2f} "
              f"{(s['e_t'] or 0.0):14.3e} {d_t:9.3f} {f['seconds']:9.1f} {s['seconds']:9.1f}")
    print()

    print("PART C: the fold from the outside: (1,1,2) per cell vs the explicit Gamma supercell / 2")
    c_row = None
    if "1x1x2" in vqr:
        from vibeqc.periodic_dlpno import run_periodic_dlpno_ccsd

        sc = run_periodic_dlpno_ccsd(
            build_toroidal_supercell(system, (1, 1, 2)), args.basis, with_triples=True,
            localise="none", cc_options=_full_options(), aux_basis=args.aux, progress=False,
        )
        v = vqr["1x1x2"]["full"]
        c_row = {"d_hf": v["e_hf"] - sc.e_hf / 2, "d_corr": v["e_corr"] - sc.e_corr / 2,
                 "d_t": (v["e_t"] or 0.0) - (sc.e_t or 0.0) / 2}
        print(f"  ΔE_HF {c_row['d_hf']:.2e}  ΔE_corr {c_row['d_corr']:.2e}  ΔE_(T) {c_row['d_t']:.2e} Ha\n")
    else:
        print("  (skipped: (1,1,2) not in --meshes)\n")

    record = {"system": {"label": label, "basis": args.basis, "aux_vibeqc": aux,
                         "aux_pyscf": py.get("aux"), "geometry": geometry},
              "vibeqc": vqr, "pyscf": {k: py[k] for k in py if k != "aux"},
              "part_a": a_rows, "part_b": b_rows, "part_c": c_row}
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(record, fh, indent=1)
        print(f"numbers written to {args.json}")
    worst_corr = max(abs(r["d_corr_keep_exxdiv_uha"]) for r in a_rows.values())
    print(f"largest |ΔE_corr| vs PySCF on the same Fock (keep_exxdiv=True): "
          f"{worst_corr:.2f} µHa per cell")
    return 0


if __name__ == "__main__":
    sys.exit(main())
