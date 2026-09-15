#!/usr/bin/env python3
"""AICCM AIC7 compute driver (2026-08-24) -- isolated, hard-capped ladders.

WHY THIS EXISTS (measured, from the 2026-08-22 ``aic-*`` wave on compute-managed):

  * ``aic-o8-dia-tz-gdf`` burned the full 23 h wall and produced ZERO rungs:
    a single SCF ran past the wall and nothing killed it.
  * ``aic-o8-mgo-dz-gdf`` spent 74 478 s (20.7 h) to land 3 of 15 rungs
    (N=1 rhf/pbe/pbe0 at ~24 700 s each).
  * ``aic-mad-mgo-pob-dzvp-rev2`` spent 26 024 s (7.2 h) inside ONE
    ``run_ccm_rhf_gdf`` call at N=1.
  * Every ``-rgam`` and every ``o9`` member died in <1 s on
    ``TypeError: ccm_neutral_cderi_fold() got an unexpected keyword argument
    'fold_threads'``.  ``fold_threads`` exists on main but NOT in the deployed
    release; 18 jobs produced no science at all.
  * ``aic-o8-lih-tz-gdf`` skipped 8 of 15 rungs on the ``n_cells**1.5``
    predictive-overrun heuristic.  That heuristic is wrong: on
    ``lih/pob-dzvp-rev2`` N=1 rhf cost 24.9 s and N=2 rhf cost 13.8 s, i.e. the
    cost is NOT monotone in the mesh, so extrapolating from N=1 refuses rungs
    that would have finished in minutes.

Three architectural consequences, all implemented here:

  1. **Every rung runs in its own child process with a hard kill deadline.**
     A rung that overruns is SIGTERM'd then SIGKILL'd.  A full per-rung cap is
     ``rung_timeout``; a global-budget-shortened allotment is
     ``budget_killed``.  The ladder continues.  An OOM or SIGSEGV in one rung
     no longer destroys the whole member either -- the child's exit signal is
     recorded and the driver walks on.
  2. **The child flushes its record after every stage**, so a killed rung still
     leaves the stages it did complete on disk.
  3. **No predictive skipping.**  A rung is attempted and measured, or refused
     because the *elapsed* budget is already gone.  Nothing is refused on an
     extrapolation.
  4. **The kill deadlines are sized from the reservation vq actually granted**
     (``VQ_WALL_TIME_SECONDS``), not from a constant.  ``--budget-s`` defaults
     to the whole wall less a finalization margin, and ``--rung-cap-s`` becomes
     a FLOOR under which each rung may take its share of what is left.  A fixed
     cap inside an unsized reservation is GitLab issue #120: it killed 206
     healthy members across 202 cases while the granted nodes stood idle.  Pass
     either flag explicitly to pin it as a hard value; with no reservation
     exported both keep their historical constants.

Plus: every call into vibe-qc goes through :func:`_call`, which drops keyword
arguments the *deployed* signature does not accept and records what it dropped.
That is what makes the real-Gamma half of this campaign runnable on the release
build without hand-patching a host checkout (forbidden, CLAUDE.md sec. 15).

Modes::

    aic7.py lad  --system S --basis B --route {gdf,real-gamma} --methods M \
                 --meshes 1,2,3 [--rung-cap-s N] [--budget-s N]
    aic7.py mad  --system S --basis B --meshes 1,2,3 [...]
    aic7.py par   --cases lih:sto-3g:1,mgo:sto-3g:2,...      (Theorem-1 points)
    aic7.py probe --cases lih:pob-tzvp-rev2:1:1,...:1:0       (nrep=1 pathology)
    aic7.py diag [--rung-cap-s N]
    aic7.py worker <specfile> <outfile>      (internal; one rung)

Output: ``$VQ_WORKDIR/aic7-<tag>/results.json`` plus one
``rungs/<id>.json`` / ``rungs/<id>.err`` pair per rung.  Nothing is ever
written next to the payload sources (they are submitted sealed 0444).

NORMALISATION (vibe-qc issue #177 / article item C8) is unchanged from the
wave-4 payloads: ``run_ccm_rhf_direct`` / ``run_ccm_rks_direct`` /
``run_ccm_rhf_scalable`` return a **supercell total**; ``run_ccm_rhf_gdf`` /
``run_ccm_rks_gdf`` return **per unit cell**.  Every record carries
``energy_raw``, ``energy_raw_normalisation``, ``n_cells`` and the derived
``energy_per_cell`` / ``energy_per_atom``.

Geometries are byte-identical to the wave-4 payloads (``studies/aiccm-2026/testset.py``,
extracted from the matching CRYSTAL23 ``.d12`` in ``qc-input-library``).
CCM reference: M. F. Peintinger and T. Bredow, J. Comput. Chem. 35, 839 (2014),
doi:10.1002/jcc.23550.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ANG = 1.0 / 0.529177210903
BUDGET_KILL_GRACE_S = 60.0
WALL_FINALIZATION_GRACE_S = 300.0

# Fallbacks used ONLY when vq exported no reservation (a bare local run).
# Under a real grant both are derived -- see ``_resolve_driver_limits`` and
# GitLab issue #120.  ``DEFAULT_BUDGET_S`` is the historical 18 h constant,
# whose help text openly admitted it was tuned "5 h under a 23 h wall".
DEFAULT_BUDGET_S = 64800.0
DEFAULT_RUNG_CAP_S = {
    "lad": 10800.0,
    "mad": 10800.0,
    "par": 2700.0,
    "probe": 3600.0,
    "itr": 7200.0,
    "diag": 2700.0,
}


# ============================================================ geometry ======
def _rocksalt(vq, np, a_ang, z_cat, z_an):
    a = a_ang * ANG
    lat = np.array([[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]])
    an = (lat @ np.array([0.5, 0.5, 0.5])).tolist()
    return vq.PeriodicSystem(3, lat, [vq.Atom(z_cat, [0, 0, 0]), vq.Atom(z_an, an)])


def _zincblende(vq, np, a_ang, z1, z2):
    a = a_ang * ANG
    lat = np.array([[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]])
    return vq.PeriodicSystem(3, lat, [vq.Atom(z1, [0, 0, 0]),
                                      vq.Atom(z2, [a / 4, a / 4, a / 4])])


SYSTEMS = {
    "lih": (4.0840, "LiH rocksalt (Fm-3m)", "rocksalt", (3, 1)),
    "mgo": (4.22389871, "MgO rocksalt (Fm-3m)", "rocksalt", (12, 8)),
    "diamond": (3.5670, "C diamond (Fd-3m)", "zincblende", (6, 6)),
}


def build_unit(vq, np, name):
    a_ang, label, kind, zz = SYSTEMS[name]
    if kind == "rocksalt":
        return _rocksalt(vq, np, a_ang, zz[0], zz[1]), a_ang, label
    return _zincblende(vq, np, a_ang, zz[0], zz[1]), a_ang, label


# ========================================================== provenance ======
def live_stamp():
    """Read the producer identity LIVE, in this process, at run time.

    Deliberately does NOT call ``pslib.version_gate()``-style helpers: several
    copies of that helper in the tree return a hard-coded string without
    probing anything (fleet brief 2026-08-24).  Everything below is read from
    the imported package, including a SHA-256 of the compiled core -- the layer
    a symbol probe cannot see, because a stale core returns wrong numbers
    rather than failing to import.
    """
    import glob
    import hashlib
    import platform
    import socket

    import numpy as np
    import vibeqc

    stamp = {
        "vibeqc_version": getattr(vibeqc, "__version__", None),
        "vibeqc_file": getattr(vibeqc, "__file__", None),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "vq_job_id": os.environ.get("VQ_JOB_ID"),
        "vq_program": os.environ.get("VQ_PROGRAM"),
        "vq_workdir": os.environ.get("VQ_WORKDIR"),
        "vq_cpus": os.environ.get("VQ_CPUS"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "vibeqc_ccm_fold_threads": os.environ.get("VIBEQC_CCM_FOLD_THREADS"),
        "read_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    cores = []
    try:
        root = os.path.dirname(os.path.abspath(vibeqc.__file__))
        seen = set()
        for pat in ("_core*.so", "_core*.pyd", "**/_core*.so"):
            for p in glob.glob(os.path.join(root, pat), recursive=True):
                if p in seen:
                    continue
                seen.add(p)
                with open(p, "rb") as fh:
                    cores.append({"path": p, "bytes": os.path.getsize(p),
                                  "sha256": hashlib.sha256(fh.read()).hexdigest()})
    except Exception as exc:                       # pragma: no cover
        stamp["native_core_error"] = repr(exc)
    stamp["native_cores"] = cores
    for attr in ("__git_sha__", "__commit__", "__source_sha__", "__build__"):
        if hasattr(vibeqc, attr):
            stamp[attr] = str(getattr(vibeqc, attr))
    try:
        # ``vibeqc.banner`` on the package root is the public banner()
        # function, not this submodule.  Import the probe directly so the
        # campaign cannot silently replace linked-library provenance with an
        # AttributeError (GitLab #477).
        from vibeqc.banner import library_versions
        stamp["library_versions"] = {str(k): str(v)
                                     for k, v in dict(library_versions()).items()}
    except Exception as exc:
        stamp["library_versions_error"] = repr(exc)
    return stamp


def _require_linked_library_provenance(producer):
    """Refuse a campaign whose live producer stamp lacks library identity."""
    if not isinstance(producer, dict):
        raise TypeError("live producer stamp is missing or malformed")
    versions = producer.get("library_versions")
    if not isinstance(versions, dict) or not versions:
        detail = producer.get("library_versions_error")
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(
            "live producer stamp has no non-empty linked-library provenance"
            + suffix
        )


def _minimal_stamp():
    import vibeqc
    return {"vibeqc_version": getattr(vibeqc, "__version__", None),
            "vibeqc_file": getattr(vibeqc, "__file__", None),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "vibeqc_ccm_fold_threads": os.environ.get("VIBEQC_CCM_FOLD_THREADS")}


# ============================================== signature-safe invocation ===
def _call(rec, fn, *args, **kw):
    """Call ``fn`` dropping kwargs the DEPLOYED signature does not accept.

    The 2026-08-22 wave lost 18 jobs to ``fold_threads``, a keyword that exists
    on main and not in the deployed release.  Anything dropped is recorded in
    ``rec['dropped_kwargs']`` so a consumer can see the call was not the one
    the source text reads like.
    """
    import inspect
    name = getattr(fn, "__name__", str(fn))
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(*args, **kw)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(*args, **kw)
    dropped = sorted(k for k in kw if k not in params)
    for k in dropped:
        kw.pop(k)
    if dropped:
        rec.setdefault("dropped_kwargs", {}).setdefault(name, []).extend(dropped)
        print(f"[aic7] NOTE {name}: dropped unsupported kwargs {dropped}",
              file=sys.stderr, flush=True)
    return fn(*args, **kw)


# ================================================================ worker ====
def _atomic_json_dump(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=1, default=str)
    os.replace(tmp, path)


def _flusher(path, rec):
    def flush():
        _atomic_json_dump(path, rec)
    return flush


def _geom_of(ccm):
    return {
        "n_cells": int(ccm.n_cells),
        "n_atoms_supercell": int(ccm.n_atoms),
        "n_atoms_unit": int(ccm.n_atoms // ccm.n_cells),
        "nbf_supercell": int(ccm.nbf),
        "wsc_inscribed_radius_bohr": float(ccm.wsc_inscribed_radius),
        "kspacing_equiv_bohr_inv": float(ccm.kspacing_equiv),
    }


def worker(specpath, outpath):
    """One rung, in its own process.  ``kind == 'stamp'`` is the live probe."""
    spec = json.load(open(specpath))
    if spec.get("kind") == "stamp":
        rec = {"kind": "stamp",
               "read_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        try:
            rec["producer_full"] = live_stamp()
            rec["status"] = "ok"
        except BaseException as exc:               # noqa: BLE001
            rec["status"] = "failed"
            rec["error"] = repr(exc)
            rec["traceback"] = traceback.format_exc()
        _atomic_json_dump(outpath, rec)
        return 0 if rec["status"] == "ok" else 1

    rec = {"spec": spec, "status": "running", "worker_pid": os.getpid(),
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "stages": []}
    flush = _flusher(outpath, rec)
    flush()

    t_all = time.time()
    try:
        import numpy as np
        import vibeqc as vq
        rec["producer"] = _minimal_stamp()
        flush()

        from vibeqc.periodic.ccm import CCMSystem
        unit, a_ang, label = build_unit(vq, np, spec["system"])
        n = int(spec["mesh"])
        t0 = time.time()
        ccm = CCMSystem(unit, (n, n, n), basis=spec["basis"])
        rec["system_build_s"] = round(time.time() - t0, 3)
        rec.update(_geom_of(ccm))
        rec["mesh"] = n
        rec["system_label"] = label
        rec["lattice_constant_ang"] = a_ang
        flush()

        kind = spec["kind"]
        if kind == "lad":
            _rung_lad(rec, flush, np, ccm, spec)
        elif kind == "mad":
            _rung_mad(rec, flush, ccm, spec)
        elif kind == "diag":
            _rung_diag(rec, flush, np, ccm, spec)
        elif kind == "par":
            _rung_par(rec, flush, ccm, spec)
        elif kind == "probe":
            _rung_probe(rec, flush, ccm, spec)
        elif kind == "itr":
            _rung_itr(rec, flush, ccm, spec)
        else:
            raise ValueError(f"unknown rung kind {kind!r}")
    except MemoryError as exc:
        rec["status"] = "memory_error"
        rec["error"] = repr(exc)
        rec["traceback"] = traceback.format_exc()
    except BaseException as exc:                   # noqa: BLE001 - record everything
        rec["status"] = "failed"
        rec["error"] = repr(exc)
        rec["traceback"] = traceback.format_exc()
    rec["worker_elapsed_s"] = round(time.time() - t_all, 3)
    rec["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if rec["status"] == "running":
        rec["status"] = "worker_end_without_status"
    flush()
    return 0 if rec["status"] in ("ok", "not_converged") else 1


def _energy_fields(rec, res, normalisation, n_cells, n_atoms_unit):
    e_raw = float(res.energy)
    rec["energy_raw"] = e_raw
    rec["energy_raw_normalisation"] = normalisation
    e_cell = e_raw if normalisation == "per_unit_cell" else e_raw / n_cells
    rec["energy_per_cell"] = e_cell
    rec["energy_per_atom"] = e_cell / n_atoms_unit
    rec["converged"] = bool(getattr(res, "converged", False))
    rec["n_iter"] = int(getattr(res, "n_iter", -1))
    rec["exchange_q0"] = str(getattr(res, "exchange_q0", ""))
    return e_cell


def _rung_lad(rec, flush, np, ccm, spec):
    from vibeqc.periodic.ccm.direct import run_ccm_rhf_direct, run_ccm_rks_direct
    from vibeqc.periodic.ccm.neutral import ccm_neutral_cderi_fold
    from vibeqc.periodic.ccm.ri import run_ccm_rhf_gdf, run_ccm_rks_gdf

    route = spec["route"]
    # A real-Gamma rung carries EVERY method for its mesh so the fold cderi --
    # measured at 275 s (N=2) / 1649 s (N=3) on lih/pob-dzvp-rev2, i.e. 97 % of
    # the rung -- is built once and reused, instead of once per method as in
    # wave 5 (aic5-rg-lihdz-dft rebuilt it 4x for 4 rungs).
    methods = spec.get("methods") or [spec["method"]]
    n_cells = rec["n_cells"]
    n_at = rec["n_atoms_unit"]
    cderi = None
    rec["methods"] = methods
    rec["per_method"] = []

    if route == "real-gamma":
        t = time.time()
        cderi = _call(rec, ccm_neutral_cderi_fold, ccm, ke_cutoff=200.0,
                      aux_basis=None, symmetry=None,
                      fold_threads=int(os.environ.get("VIBEQC_CCM_FOLD_THREADS", 1)))
        rec["cderi_seconds"] = round(time.time() - t, 3)
        arr = np.asarray(cderi)
        rec["cderi_shape"] = list(arr.shape)
        rec["cderi_gib"] = round(arr.nbytes / 2**30, 5)
        rec["n_aux_supercell"] = int(arr.shape[0])
        rec["cderi_frobenius_norm"] = float(np.linalg.norm(arr))
        print(f"[aic7] cderi {rec['cderi_seconds']}s shape={rec['cderi_shape']} "
              f"{rec['cderi_gib']} GiB", flush=True)
        flush()

    n_ok = 0
    for method in methods:
        sub = {"method": method, "route": route}
        rec["per_method"].append(sub)
        flush()
        t = time.time()
        try:
            if route == "gdf":
                if method == "rhf":
                    res = run_ccm_rhf_gdf(ccm)
                else:
                    res = run_ccm_rks_gdf(ccm, functional=method)
                sub["gdf_pair_builds"] = int(getattr(res, "gdf_pair_builds", -1))
                sub["gdf_pair_total"] = int(getattr(res, "gdf_pair_total", -1))
                e_cell = _energy_fields(sub, res, "per_unit_cell", n_cells, n_at)
            else:
                if method == "rhf":
                    res = _call(sub, run_ccm_rhf_direct, ccm, cderi=cderi,
                                exxdiv="ewald", max_iter=int(spec["max_iter"]),
                                conv_tol=float(spec["conv_tol"]))
                else:
                    res = _call(sub, run_ccm_rks_direct, ccm, functional=method,
                                cderi=cderi, exxdiv="ewald",
                                max_iter=int(spec["max_iter"]),
                                conv_tol=max(float(spec["conv_tol"]), 1e-8))
                e_cell = _energy_fields(sub, res, "per_supercell", n_cells, n_at)
            sub["scf_seconds"] = round(time.time() - t, 3)
            sub["status"] = "ok" if sub["converged"] else "not_converged"
            n_ok += 1 if sub["status"] == "ok" else 0
            print(f"[aic7] N={rec['mesh']} {method:5s} {route:10s} "
                  f"E/cell={e_cell:.9f} conv={sub['converged']} "
                  f"it={sub['n_iter']} {sub['scf_seconds']}s", flush=True)
        except BaseException as exc:               # noqa: BLE001
            sub["scf_seconds"] = round(time.time() - t, 3)
            sub["status"] = "failed"
            sub["error"] = repr(exc)
            sub["traceback"] = traceback.format_exc()
            print(f"[aic7] N={rec['mesh']} {method} FAILED after "
                  f"{sub['scf_seconds']}s: {exc!r}", flush=True)
        flush()
    # Promote the single-method case so wave-5 consumers keep working.
    if len(methods) == 1:
        rec.update({k: v for k, v in rec["per_method"][0].items()
                    if k not in ("method", "route")})
        rec["method"] = methods[0]
    rec["n_methods_ok"] = n_ok
    rec["status"] = "ok" if n_ok == len(methods) else "failed"


def _rung_mad(rec, flush, ccm, spec):
    from vibeqc.periodic.ccm.ri import run_ccm_rhf_gdf
    from vibeqc.periodic.ccm.scf import run_ccm_rhf_scalable

    c = spec["construction"]
    n_cells = rec["n_cells"]
    n_at = rec["n_atoms_unit"]
    rec["construction"] = c
    t = time.time()
    if c == "neutral":
        res = run_ccm_rhf_gdf(ccm)
        e_cell = _energy_fields(rec, res, "per_unit_cell", n_cells, n_at)
    else:
        res = _call(rec, run_ccm_rhf_scalable, ccm, method=c, four_center="direct",
                    schwarz_threshold=0.0, max_iter=int(spec["max_iter"]),
                    conv_tol=float(spec["conv_tol"]))
        e_cell = _energy_fields(rec, res, "per_supercell", n_cells, n_at)
    rec["scf_seconds"] = round(time.time() - t, 3)
    rec["status"] = "ok" if rec["converged"] else "not_converged"
    print(f"[aic7] N={rec['mesh']} {c:16s} E/cell={e_cell:.9f} "
          f"E/atom={e_cell / n_at:.9f} conv={rec['converged']} "
          f"{rec['scf_seconds']}s", flush=True)


def _rung_par(rec, flush, ccm, spec):
    """One Theorem-1 parity POINT: both production routes, same ccm, same rung.

    Paper 1 Theorem 1 is the claim that the real-Gamma and Bloch/GDF routes of
    the neutral fitted-torus control agree.  The wave-5 harvest reported
    ``theorem1_parity.ALL_POINTS_PASS=false`` with ``n_failures=0`` over
    ``n_points_with_both_routes=0`` -- a zero over an EMPTY domain, which
    cannot discriminate anything.  This mode exists to make the domain
    non-empty by construction: a rung is admitted to the gate only if BOTH
    routes returned on the same ``CCMSystem`` object, and the record says so
    explicitly in ``parity_point_admissible``.

    Wave-5 measured deltas (deployed 0.15.138, per unit cell), for calibration:
      lih/sto-3g   N=1  -1.55e-12 Ha    lih/sto-3g   N=2  +1.87e-05 Ha
      lih/pob-tzvp N=1  -5.84e-05 Ha    mgo/sto-3g   N=2  +2.75e-08 Ha
      mgo/sto-3g   N=1  -4.91e-01 Ha    <- 0.49 Ha between two routes that
                                           are supposed to be identical
    """
    from vibeqc.periodic.ccm.direct import (run_ccm_rhf_direct,
                                            run_ccm_rks_direct)
    from vibeqc.periodic.ccm.ri import run_ccm_rhf_gdf, run_ccm_rks_gdf

    n_cells = rec["n_cells"]
    n_at = rec["n_atoms_unit"]
    method = str(spec.get("method", "rhf")).lower()
    sym = spec.get("symmetry", None)      # None => whatever the route defaults to
    aux = spec.get("aux_basis", None)
    rec["method"] = method
    rec["symmetry_requested"] = sym
    rec["aux_basis"] = aux
    out = {}
    rec["routes"] = {}

    for label, fn in (("direct", "direct"), ("gdf", "gdf")):
        r = {"route": label, "method": method}
        rec["routes"][label] = r
        flush()
        t = time.time()
        try:
            if fn == "direct":
                kw = dict(exxdiv="ewald", max_iter=int(spec["max_iter"]),
                          conv_tol=float(spec["conv_tol"]))
                if aux is not None:
                    kw["aux_basis"] = aux
                if method == "rhf":
                    res = _call(r, run_ccm_rhf_direct, ccm, **kw)
                else:
                    kw["conv_tol"] = max(kw["conv_tol"], 1e-8)
                    res = _call(r, run_ccm_rks_direct, ccm,
                                functional=method, **kw)
                e_cell = _energy_fields(r, res, "per_supercell", n_cells, n_at)
            else:
                kw = {}
                if sym is not None:
                    kw["symmetry"] = bool(sym)
                if aux is not None:
                    kw["aux_basis"] = aux
                if method == "rhf":
                    res = run_ccm_rhf_gdf(ccm, **kw)
                else:
                    res = run_ccm_rks_gdf(ccm, functional=method, **kw)
                r["gdf_pair_builds"] = int(getattr(res, "gdf_pair_builds", -1))
                r["gdf_pair_total"] = int(getattr(res, "gdf_pair_total", -1))
                e_cell = _energy_fields(r, res, "per_unit_cell", n_cells, n_at)
            r["seconds"] = round(time.time() - t, 3)
            r["status"] = "ok" if r["converged"] else "not_converged"
            out[label] = e_cell
            print(f"[aic7-par] {spec['system']}/{spec['basis']} N={rec['mesh']} "
                  f"{method}/sym={sym}/aux={aux} {label:6s} "
                  f"E/cell={e_cell:.9f} conv={r['converged']} "
                  f"it={r['n_iter']} {r['seconds']}s", flush=True)
        except BaseException as exc:               # noqa: BLE001
            r["seconds"] = round(time.time() - t, 3)
            r["status"] = "failed"
            r["error"] = repr(exc)
            r["traceback"] = traceback.format_exc()
            print(f"[aic6-par] {label} FAILED after {r['seconds']}s: {exc!r}",
                  flush=True)
        flush()

    both = ("direct" in out and "gdf" in out
            and rec["routes"]["direct"]["status"] == "ok"
            and rec["routes"]["gdf"]["status"] == "ok")
    rec["parity_point_admissible"] = bool(both)
    if both:
        d = out["direct"] - out["gdf"]
        rec["direct_minus_gdf_per_cell_ha"] = d
        rec["direct_minus_gdf_per_atom_ha"] = d / n_at
        rec["abs_delta_ha_per_cell"] = abs(d)
        print(f"[aic6-par] PARITY N={rec['mesh']} direct-gdf = {d:+.6e} Ha/cell",
              flush=True)
    else:
        rec["parity_excluded_reason"] = (
            "one or both routes did not return ok on this rung; the point is "
            "NOT counted in the theorem-1 domain")
    rec["status"] = "ok" if both else "failed"


def _rung_itr(rec, flush, ccm, spec):
    """Is the nrep=(1,1,1) GDF cost paid PER ITERATION or ONCE up front?

    Wave 2 (aic6-probe-gdfn1, job 360c0c47b95b) eliminated H-A -- the pair-star
    symmetry reduction is not what degenerates at one k-point, because
    symmetry=True and symmetry=False were identical in BOTH cost and energy
    at N=1 (lih/pob-tzvp-rev2: 2635.05 s vs 2634.96 s, E/cell -8.449328671 in
    both).  Reading the producer rather than the record says why: `_ccm_gdf`
    gates the Lpq cache builder on ``int(np.prod(ccm.nrep)) > 1``
    (vibeqc/periodic/ccm/ri.py:339), so at nrep=(1,1,1) the ``symmetry``
    argument is INERT and the run loses the Lpq cache entirely -- not merely
    its symmetry reduction.  The reported ``pair_builds=0/0`` is the same gate
    showing through the reporting block: a zero over an EMPTY domain.

    Two mechanisms survive, and they predict different shapes of wall(n_iter):

      H-1  uncached => the three-centre Lpq integrals are rebuilt INSIDE the
           SCF loop, once per iteration.  wall is LINEAR in the iteration
           count with a near-zero intercept, and the slope is the whole cost.
      H-2  uncached => one unscreened up-front build, then cheap iterations.
           wall = C0 + small*n_iter, i.e. a LARGE intercept and a flat slope.

    Sweeping the iteration cap over a short ladder at fixed nrep separates
    them by the intercept.  The same sweep is run at N=2, where the cache IS
    installed, as the control: the ratio of the two slopes is the per-iteration
    penalty that the missing cache costs.
    """
    from vibeqc.periodic.ccm.ri import run_ccm_rhf_gdf

    cap = int(spec["iter_cap"])
    rec["iter_cap"] = cap
    n_cells = rec["n_cells"]
    n_at = rec["n_atoms_unit"]
    # A tolerance that is never met makes the cap bind, so n_iter == cap and
    # the regression has a clean x-axis.  If the deployed wrapper does not take
    # these kwargs the call is retried bare and the ACTUAL n_iter is regressed
    # instead -- the diagnostic degrades, it does not fail.
    t = time.time()
    try:
        res = run_ccm_rhf_gdf(ccm, max_iter=cap, conv_tol=1e-14)
        rec["iter_cap_passthrough"] = "accepted"
    except TypeError as exc:
        rec["iter_cap_passthrough"] = f"rejected: {exc!r}"
        print(f"[aic7-itr] NOTE max_iter/conv_tol rejected by the deployed "
              f"wrapper ({exc!r}); retrying bare", file=sys.stderr, flush=True)
        t = time.time()
        res = run_ccm_rhf_gdf(ccm)
    e_cell = _energy_fields(rec, res, "per_unit_cell", n_cells, n_at)
    rec["scf_seconds"] = round(time.time() - t, 3)
    rec["gdf_pair_builds"] = int(getattr(res, "gdf_pair_builds", -1))
    rec["gdf_pair_total"] = int(getattr(res, "gdf_pair_total", -1))
    rec["seconds_per_iter"] = (round(rec["scf_seconds"] / rec["n_iter"], 4)
                               if rec.get("n_iter") else None)
    # Convergence is NOT the point here; a capped run is a valid measurement.
    rec["status"] = "ok"
    print(f"[aic7-itr] {spec['system']}/{spec['basis']} N={rec['mesh']} "
          f"cap={cap} it={rec['n_iter']} conv={rec['converged']} "
          f"E/cell={e_cell:.9f} pair_builds={rec['gdf_pair_builds']}/"
          f"{rec['gdf_pair_total']} {rec['scf_seconds']}s "
          f"({rec['seconds_per_iter']} s/it)", flush=True)


def _rung_probe(rec, flush, ccm, spec):
    """Discriminate WHY run_ccm_rhf_gdf is catastrophic at nrep=(1,1,1).

    Measured on the deployed 0.15.138 in wave 5 -- the single largest consumer
    of fleet time in two waves:

      lih/pob-tzvp-rev2  gdf  N=1 3185.8 s   N=2 20.9 s   N=5 1742 s
      mgo/pob-dzvp-rev2  gdf  N=1 >25200 s   N=2 161.8 s  N=3 971.7 s
      diamond/pob-tzvp   gdf  N=1 >26000 s (still running at judge time)

    N=1 costs MORE than N=5 on the same system and basis: a 152x inversion for
    lih/pob-tzvp, >155x for mgo/pob-dzvp.  Two hypotheses:

      H-A  the RSGDF pair-star symmetry reduction degenerates at a single
           k-point -- with nothing to reduce it falls into an unscreened path.
      H-B  the k-point count itself is the trigger and symmetry is irrelevant.

    The design is a 2x2: {mesh 1, mesh 2} x {symmetry True, False} on one
    (system, basis).  It discriminates because the two hypotheses predict
    different cells:
      * H-A predicts N=1/symmetry=False is FAST (comparable to N=2).
      * H-B predicts N=1 is slow for BOTH symmetry settings.
    A result where N=1 is slow only with symmetry=True eliminates H-B; a
    result where both N=1 cells are slow eliminates H-A.  Either way the
    energies are recorded, so a symmetry setting that changes the ANSWER (not
    just the cost) is caught in the same run.
    """
    from vibeqc.periodic.ccm.ri import run_ccm_rhf_gdf

    sym = bool(spec["symmetry"])
    rec["symmetry"] = sym
    n_cells = rec["n_cells"]
    n_at = rec["n_atoms_unit"]
    t = time.time()
    res = run_ccm_rhf_gdf(ccm, symmetry=sym)
    e_cell = _energy_fields(rec, res, "per_unit_cell", n_cells, n_at)
    rec["scf_seconds"] = round(time.time() - t, 3)
    rec["gdf_pair_builds"] = int(getattr(res, "gdf_pair_builds", -1))
    rec["gdf_pair_total"] = int(getattr(res, "gdf_pair_total", -1))
    rec["status"] = "ok" if rec["converged"] else "not_converged"
    print(f"[aic7-probe] {spec['system']}/{spec['basis']} N={rec['mesh']} "
          f"symmetry={sym} E/cell={e_cell:.9f} it={rec['n_iter']} "
          f"pair_builds={rec['gdf_pair_builds']}/{rec['gdf_pair_total']} "
          f"{rec['scf_seconds']}s", flush=True)


def _rung_diag(rec, flush, np, ccm, spec):
    """Stage-resolved cost of the real-Gamma direct route, per basis.

    The question this is built to answer (coordinator, 2026-08-23): LiH/MgO
    STO-3G ``run_ccm_rhf_direct`` ran >25 min without finishing while the same
    fixture in pob-TZVP-rev2 returned in 2.4 s.  Two hypotheses:

      H1  STO-3G hits a pathological branch (its cost is real work in the
          wrong place -- e.g. a huge or ill-conditioned auto-aux set, or an
          SCF that never converges and grinds to max_iter).
      H2  the pob-TZVP path short-circuits and never does the work (returns
          an energy without building/contracting the torus RI).

    The stages below discriminate them WITHOUT reading vibe-qc internals:

      * ``n_aux_supercell`` + ``cderi_gib`` + ``cderi_frobenius_norm`` --
        a short-circuiting TZVP path has a degenerate (empty, tiny, or
        zero-norm) cderi; a pathological STO-3G path has an anomalously LARGE
        or ill-conditioned one.  Under H2 the TZVP tensor is degenerate;
        under H1 it is a normal, full-size tensor.
      * ``iter1_s`` and ``iter3_s`` -- the marginal cost of one SCF iteration,
        ``(iter3_s - iter1_s)/2``.  Under H2 the TZVP per-iteration cost is
        ~0; under H1 it is a normal nonzero cost and STO-3G's is the outlier.
      * ``gdf_energy_per_cell`` vs ``direct_energy_per_cell`` -- the two
        production routes must agree (Paper 1 Theorem 1).  A TZVP direct
        energy that matches GDF to numerical tolerance CANNOT be a
        short-circuit; a mismatch is H2 evidence.

    Every stage is flushed, so a rung killed at the cap still reports the
    stages it completed.
    """
    from vibeqc.periodic.ccm.direct import run_ccm_rhf_direct
    from vibeqc.periodic.ccm.neutral import ccm_neutral_cderi_fold
    from vibeqc.periodic.ccm.ri import run_ccm_rhf_gdf

    n_cells = rec["n_cells"]
    n_at = rec["n_atoms_unit"]
    stages = rec["stages"]

    def stage(name, fn):
        s = {"stage": name, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                         time.gmtime())}
        stages.append(s)
        flush()
        t = time.time()
        try:
            out = fn(s)
            s["seconds"] = round(time.time() - t, 3)
            s["status"] = "ok"
            print(f"[aic6-diag] {rec['spec']['system']}/{rec['spec']['basis']} "
                  f"N={rec['mesh']} {name}: {s['seconds']}s", flush=True)
            flush()
            return out
        except BaseException as exc:               # noqa: BLE001
            s["seconds"] = round(time.time() - t, 3)
            s["status"] = "failed"
            s["error"] = repr(exc)
            s["traceback"] = traceback.format_exc()
            print(f"[aic6-diag] {name} FAILED after {s['seconds']}s: {exc!r}",
                  flush=True)
            flush()
            return None

    def _cderi(s):
        c = _call(s, ccm_neutral_cderi_fold, ccm, ke_cutoff=200.0, aux_basis=None,
                  symmetry=None,
                  fold_threads=int(os.environ.get("VIBEQC_CCM_FOLD_THREADS", 1)))
        arr = np.asarray(c)
        s["cderi_shape"] = list(arr.shape)
        s["cderi_gib"] = round(arr.nbytes / 2**30, 5)
        s["n_aux_supercell"] = int(arr.shape[0])
        s["cderi_frobenius_norm"] = float(np.linalg.norm(arr))
        s["cderi_max_abs"] = float(np.max(np.abs(arr))) if arr.size else 0.0
        s["cderi_nonzero_fraction"] = (float(np.count_nonzero(arr)) / arr.size
                                       if arr.size else 0.0)
        return c

    cderi = stage("cderi_fold", _cderi)

    def _direct(maxit, use_cderi):
        def inner(s):
            s["max_iter"] = maxit
            s["cderi_supplied"] = bool(use_cderi)
            res = _call(s, run_ccm_rhf_direct, ccm,
                        cderi=(cderi if use_cderi else None),
                        exxdiv="ewald", max_iter=maxit, conv_tol=1e-9)
            e = float(res.energy)
            s["energy_raw"] = e
            s["energy_raw_normalisation"] = "per_supercell"
            s["energy_per_cell"] = e / n_cells
            s["energy_per_atom"] = e / (n_cells * n_at)
            s["converged"] = bool(getattr(res, "converged", False))
            s["n_iter"] = int(getattr(res, "n_iter", -1))
            return s["energy_per_cell"]
        return inner

    if cderi is not None:
        stage("direct_max_iter_1", _direct(1, True))
        stage("direct_max_iter_3", _direct(3, True))

    def _gdf(s):
        res = run_ccm_rhf_gdf(ccm)
        s["energy_raw"] = float(res.energy)
        s["energy_raw_normalisation"] = "per_unit_cell"
        s["energy_per_cell"] = float(res.energy)
        s["energy_per_atom"] = float(res.energy) / n_at
        s["converged"] = bool(getattr(res, "converged", False))
        s["n_iter"] = int(getattr(res, "n_iter", -1))
        return s["energy_per_cell"]

    stage("gdf_reference", _gdf)
    if cderi is not None:
        stage("direct_full", _direct(128, True))
    stage("direct_full_internal_cderi", _direct(128, False))

    # Derived discriminators, computed here so a consumer does not have to.
    by = {s["stage"]: s for s in stages if s.get("status") == "ok"}
    d = {}
    if "direct_max_iter_1" in by and "direct_max_iter_3" in by:
        d["marginal_iteration_s"] = round(
            (by["direct_max_iter_3"]["seconds"] - by["direct_max_iter_1"]["seconds"]) / 2.0, 4)
    if "direct_full" in by and "gdf_reference" in by:
        d["direct_minus_gdf_per_cell_ha"] = (by["direct_full"]["energy_per_cell"]
                                             - by["gdf_reference"]["energy_per_cell"])
    if "cderi_fold" in by:
        d["n_aux_supercell"] = by["cderi_fold"].get("n_aux_supercell")
        d["cderi_gib"] = by["cderi_fold"].get("cderi_gib")
        d["cderi_frobenius_norm"] = by["cderi_fold"].get("cderi_frobenius_norm")
        d["aux_per_ao"] = (round(by["cderi_fold"]["n_aux_supercell"] / rec["nbf_supercell"], 4)
                           if rec.get("nbf_supercell") else None)
    rec["discriminators"] = d
    rec["status"] = "ok" if by else "failed"
    rec.setdefault("error", None)


# ================================================================ driver ====
_METHOD_SEPARATOR = re.compile(r"[,+]")


def _parse_method_list(value):
    """Parse the CLI list grammar shared by method arguments and tags."""
    text = str(value).strip()
    if not text:
        raise argparse.ArgumentTypeError("method list must not be empty")
    pieces = _METHOD_SEPARATOR.split(text)
    if any(not piece.strip() for piece in pieces):
        raise argparse.ArgumentTypeError(
            "method list contains an empty entry; use comma or plus between names"
        )
    methods = [piece.strip().lower() for piece in pieces]
    if len(methods) != len(set(methods)):
        raise argparse.ArgumentTypeError("method list contains a duplicate entry")
    return methods


def _method_tag(methods):
    """Return the stable tag representation accepted by ``_parse_method_list``."""
    return "+".join(methods)


def _rung_id(index, spec):
    rid = f"{index:03d}-" + "-".join(
        str(spec.get(key))
        for key in (
            "kind",
            "system",
            "basis",
            "mesh",
            "method",
            "construction",
            "route",
        )
        if spec.get(key) is not None
    )
    return rid.replace("/", "_")


def _budget_elapsed(doc):
    return max(
        0.0,
        time.monotonic() - float(doc["_budget_started_monotonic"]),
    )


def _vq_wall_seconds(parser):
    """The reservation vq actually granted this job, or ``None``.

    ``vq`` exports ``VQ_WALL_TIME_SECONDS`` to every job it dispatches --
    local and scheduler alike (``vibe-queue/src/vq/daemon.py``,
    ``_vq_job_env``) -- whenever the submission declared a wall.  It is
    absent for a bare local run, and the driver must then keep its fixed
    defaults.
    """
    wall_text = os.environ.get("VQ_WALL_TIME_SECONDS")
    if wall_text is None:
        return None
    try:
        wall_s = float(wall_text)
    except ValueError:
        parser.error("VQ_WALL_TIME_SECONDS must be numeric")
    if not math.isfinite(wall_s) or wall_s <= 0.0:
        parser.error("VQ_WALL_TIME_SECONDS must be finite and positive")
    return wall_s


def _resolve_driver_limits(parser, args):
    """Fill unset ``--budget-s`` / ``--rung-cap-s`` from the granted wall.

    GitLab issue #120: a carrier that enforces a *fixed* per-case cap
    inside a reservation it did not size kills healthy calculations with
    the node almost entirely unused -- 206 members lost across 202 cases
    in the rp217/rp223 pass, one bundle spending 2.11 h of a 16 h grant
    and still losing two members at ~1 h.  The reservation is known: vq
    exports it, and this driver already read it to *validate* against.

    So the defaults are derived rather than fixed:

    * ``budget_s`` -- the line past which no new rung is STARTED -- becomes
      the whole granted wall less :data:`WALL_FINALIZATION_GRACE_S`.  No
      rung can outlive it: ``run_rungs`` allots at most
      ``remaining - BUDGET_KILL_GRACE_S``, so every rung completes strictly
      inside ``budget_s``.  The previous 18 h constant encoded one
      operator's 23 h wall; under a longer grant it left the tail unspent,
      and under a shorter one it refused the run outright.
    * ``cap_s`` stops being a ceiling and becomes a FLOOR: with a known
      reservation, each rung may take its share of what is left
      (``run_rungs(..., cap_is_floor=True)``).

    An explicitly passed flag always wins -- that is issue #120's ask (2),
    a declared per-wave parameter set from measured cost.  With no wall
    exported, both keep the historical constants exactly.
    """
    wall_s = _vq_wall_seconds(parser)
    args.vq_wall_s = wall_s
    args.rung_cap_is_floor = False

    if args.rung_cap_s is None:
        args.rung_cap_s = DEFAULT_RUNG_CAP_S[args.mode]
        args.rung_cap_source = "default"
        args.rung_cap_is_floor = wall_s is not None
    else:
        args.rung_cap_source = "explicit"

    if args.budget_s is None:
        if wall_s is None:
            args.budget_s = DEFAULT_BUDGET_S
            args.budget_source = "default"
        else:
            args.budget_s = max(0.0, wall_s - WALL_FINALIZATION_GRACE_S)
            args.budget_source = "vq-wall"
    else:
        args.budget_source = "explicit"


def _validate_driver_limits(parser, args):
    if not math.isfinite(args.budget_s) or args.budget_s < 0.0:
        parser.error("--budget-s must be finite and nonnegative")
    if not math.isfinite(args.rung_cap_s) or args.rung_cap_s <= 0.0:
        parser.error("--rung-cap-s must be finite and positive")
    wall_s = _vq_wall_seconds(parser)
    if wall_s is None:
        return
    if args.budget_s + WALL_FINALIZATION_GRACE_S > wall_s:
        parser.error(
            "--budget-s must leave at least "
            f"{WALL_FINALIZATION_GRACE_S:.0f}s of the vq wall for final output"
        )


def _finalize_run(doc):
    """Derive the top-level outcome from planned-rung terminal states."""
    planned = int(doc["n_rungs_planned"])
    actual = [row for row in doc["runs"] if row.get("rung_id")]
    statuses = Counter(str(row.get("status") or "missing") for row in actual)
    expected_ids = [row["rung_id"] for row in doc.get("planned_rungs", [])]
    expected_id_set = set(expected_ids)
    actual_by_id = {}
    for row in actual:
        actual_by_id.setdefault(row["rung_id"], []).append(row)
    missing_ids = [rid for rid in expected_ids if rid not in actual_by_id]
    unexpected_ids = sorted(set(actual_by_id) - expected_id_set)
    duplicate_ids = sorted(
        rid for rid, rows in actual_by_id.items() if len(rows) != 1
    )
    delivered = sum(
        len(actual_by_id.get(rid, ())) == 1
        and actual_by_id[rid][0].get("status") == "ok"
        for rid in expected_ids
    )
    attempted = sum(bool(row.get("attempted", True)) for row in actual)

    doc["n_rungs_recorded"] = len(actual)
    doc["n_rungs_delivered"] = delivered
    doc["n_rungs_ok"] = delivered
    doc["n_rungs_incomplete"] = max(0, planned - delivered)
    doc["n_rungs_unrecorded"] = len(missing_ids)
    doc["n_rungs_unexpected"] = len(unexpected_ids)
    doc["n_rungs_duplicate_ids"] = len(duplicate_ids)
    doc["n_rungs_attempted"] = attempted
    doc["n_rungs_budget_killed"] = statuses.get("budget_killed", 0)
    doc["rung_status_counts"] = dict(sorted(statuses.items()))
    doc["missing_rung_ids"] = missing_ids
    doc["unexpected_rung_ids"] = unexpected_ids
    doc["duplicate_rung_ids"] = duplicate_ids

    identities_match = (
        len(expected_ids) == planned
        and len(expected_id_set) == planned
        and not missing_ids
        and not unexpected_ids
        and not duplicate_ids
    )
    complete = (
        planned > 0
        and identities_match
        and delivered == planned
        and not doc.get("driver_error")
    )
    doc["status"] = "complete" if complete else "partial"
    return 0 if complete else 1


def _thread_layout(cpus, route):
    """BLAS vs fold-thread split, per the wave-4 measurement.

    The Bloch/GDF route never builds a fold cderi, so it gets every core as a
    BLAS thread.  The real-Gamma route splits: ``ccm_neutral_cderi_fold``'s
    per-(k_a,k_b) unit-cell fits are the CCM analogue of k-point parallelism
    (CLAUDE.md sec. 15), and its docstring says to pin BLAS first because the
    fits are themselves BLAS-heavy.  A real-Gamma DFT rung is dominated by the
    periodic-Becke XC build and the dense supercell eigenproblem instead, both
    OpenMP, so it keeps the BLAS-wide layout.
    """
    if route == "real-gamma":
        blas = max(1, cpus // 8)
        return blas, max(1, cpus // blas)
    return max(1, cpus), 1


def _prepare_outdir(tag):
    workdir = os.environ.get("VQ_WORKDIR") or os.getcwd()
    outdir = os.path.join(workdir, f"aic7-{tag}")
    try:
        os.makedirs(outdir, exist_ok=False)
    except FileExistsError:
        print(f"FATAL: {outdir} already exists; refusing to write into it.",
              file=sys.stderr, flush=True)
        raise SystemExit(2)
    except OSError as exc:
        print(f"FATAL: cannot create {outdir}: {exc!r}", file=sys.stderr, flush=True)
        raise SystemExit(3)
    probe = os.path.join(outdir, ".writable-probe")
    try:
        with open(probe, "w") as fh:
            fh.write("ok\n")
        os.unlink(probe)
    except OSError as exc:
        print(f"FATAL: {outdir} is not writable ({exc!r}); refusing to run. "
              f"VQ_WORKDIR={workdir!r}", file=sys.stderr, flush=True)
        raise SystemExit(3)
    os.makedirs(os.path.join(outdir, "rungs"), exist_ok=True)
    return workdir, outdir


def _tail(path, nbytes=6000):
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            if size > nbytes:
                fh.seek(size - nbytes)
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


def run_rungs(doc, outdir, rungs, cpus, route_for_threads, budget_s, cap_s,
              flush, cap_is_floor=False):
    """Run every rung in its own child, each under a kill deadline.

    ``cap_s`` is a hard per-rung ceiling by default.  When *cap_is_floor*
    is set -- which ``main`` does exactly when vq exported a reservation
    and the operator did not pin ``--rung-cap-s`` -- it becomes a floor
    instead: each rung may take up to its share of what is left of the
    budget, ``usable / rungs_remaining``, and never less than ``cap_s``.

    That is GitLab issue #120: enforcing a fixed cap inside a reservation
    the carrier did not size kills healthy calculations while the node
    stands idle.  Sharing is self-correcting -- a rung that finishes early
    returns its unused share to the pool, because the divisor is
    recomputed from the *actual* remaining budget at every rung.

    The ceiling on completion is unchanged either way: the allotment is
    still clipped to ``remaining - BUDGET_KILL_GRACE_S``, so no rung can
    outlive ``budget_s`` however large its share.
    """
    doc.setdefault("_budget_started_monotonic", time.monotonic())
    rungdir = os.path.join(outdir, "rungs")
    blas, fold = _thread_layout(cpus, route_for_threads)
    doc["thread_layout"] = {"cpus": cpus, "blas_threads": blas, "fold_threads": fold,
                            "route_for_threads": route_for_threads}
    child_env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        child_env[v] = str(blas)
    child_env["VIBEQC_CCM_FOLD_THREADS"] = str(fold)

    for i, spec in enumerate(rungs):
        rid = _rung_id(i, spec)
        specpath = os.path.join(rungdir, rid + ".spec.json")
        outpath = os.path.join(rungdir, rid + ".json")
        errpath = os.path.join(rungdir, rid + ".err")
        _atomic_json_dump(specpath, spec)
        elapsed = _budget_elapsed(doc)
        remaining = float(budget_s) - elapsed
        remaining_at_start = max(0.0, remaining)
        usable = max(0.0, remaining - BUDGET_KILL_GRACE_S)

        if usable <= 0.0:
            rec = {
                "spec": spec,
                "rung_id": rid,
                "status": "budget_killed",
                "attempted": False,
                "budget_s": float(budget_s),
                "elapsed_s": round(elapsed, 3),
                "remaining_at_start_s": round(remaining_at_start, 3),
                "allotted_s": 0.0,
                "driver": {
                    "attempted": False,
                    "budget_limited": True,
                    "budget_s": float(budget_s),
                    "elapsed_at_start_s": round(elapsed, 3),
                    "remaining_at_start_s": round(remaining_at_start, 3),
                    "allotted_s": 0.0,
                },
                "stderr_tail": "",
            }
            _atomic_json_dump(outpath, rec)
            doc["runs"].append(rec)
            flush()
            print(
                f"[aic7] BUDGET {rid}: elapsed={elapsed:.0f}s "
                f"remaining={remaining_at_start:.0f}s; not started",
                flush=True,
            )
            continue

        # Recompute the absolute-deadline allotment immediately after spawn:
        # both spec persistence and process creation count against the budget.
        t1 = time.monotonic()
        timed_out = False
        rc = None
        with open(errpath, "wb") as errfh:
            p = subprocess.Popen(
                [sys.executable, os.path.abspath(__file__), "worker", specpath, outpath],
                cwd=outdir, env=child_env, stdout=None, stderr=errfh,
                start_new_session=True)
            elapsed = _budget_elapsed(doc)
            remaining = float(budget_s) - elapsed
            remaining_at_start = max(0.0, remaining)
            usable = max(0.0, remaining - BUDGET_KILL_GRACE_S)
            rungs_remaining = max(1, len(rungs) - i)
            if cap_is_floor:
                # This rung's share of what the reservation has left, but
                # never below the declared cap (issue #120 ask 1).
                allotment = max(float(cap_s), usable / rungs_remaining)
                cap_basis = "reservation-share"
            else:
                allotment = float(cap_s)
                cap_basis = "fixed-cap"
            this_cap = min(allotment, usable)
            budget_limited = this_cap < allotment
            print(
                f"[aic7] RUNG {rid} cap={this_cap:.0f}s "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )
            if this_cap <= 0.0:
                timed_out = True
                print(
                    f"[aic7] RUNG {rid} exhausted budget during launch -- "
                    "killing",
                    flush=True,
                )
            else:
                try:
                    rc = p.wait(timeout=this_cap)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    print(f"[aic7] RUNG {rid} EXCEEDED {this_cap:.0f}s -- killing",
                          flush=True)
            if timed_out:
                signals = ((signal.SIGKILL,) if this_cap <= 0.0
                           else (signal.SIGTERM, signal.SIGKILL))
                for sig in signals:
                    try:
                        os.killpg(os.getpgid(p.pid), sig)
                    except (ProcessLookupError, PermissionError):
                        break
                    grace = min(
                        BUDGET_KILL_GRACE_S / 2.0,
                        max(0.0, float(budget_s) - _budget_elapsed(doc)),
                    )
                    try:
                        rc = p.wait(timeout=grace)
                        break
                    except subprocess.TimeoutExpired:
                        continue

        wall = round(time.monotonic() - t1, 3)
        rec = {}
        if os.path.exists(outpath):
            try:
                with open(outpath) as fh:
                    rec = json.load(fh)
            except (OSError, ValueError) as exc:
                rec = {"status": "rung_record_unreadable", "error": repr(exc)}
        else:
            rec = {"status": "rung_no_record"}
        rec.setdefault("spec", spec)
        rec["rung_id"] = rid
        rec["attempted"] = True
        rec["driver"] = {"returncode": rc, "timed_out": timed_out,
                         "cap_s": round(this_cap, 1),
                         "allotted_s": round(this_cap, 1),
                         "budget_limited": budget_limited,
                         "budget_s": float(budget_s),
                         # Issue #120 ask (3): a reader must be able to tell
                         # WHY a kill deadline had the value it had --
                         # a declared fixed cap, or this rung's share of the
                         # reservation -- without re-deriving it.
                         "cap_basis": cap_basis,
                         "cap_floor_s": float(cap_s),
                         "rungs_remaining_at_start": rungs_remaining,
                         "elapsed_at_start_s": round(elapsed, 3),
                         "remaining_at_start_s": round(remaining_at_start, 3),
                         "attempted": True,
                         "driver_wall_s": wall}
        if timed_out:
            if budget_limited:
                rec["status"] = "budget_killed"
                rec["budget_s"] = float(budget_s)
                rec["elapsed_s"] = round(_budget_elapsed(doc), 3)
                rec["remaining_at_start_s"] = round(remaining_at_start, 3)
                rec["allotted_s"] = round(this_cap, 1)
            else:
                rec["status"] = "rung_timeout"
        elif rec.get("status") == "ok" and rc != 0:
            rec["worker_reported_status"] = "ok"
            rec["status"] = f"child_exit_{rc}"
        elif rc not in (0, 1) and rec.get("status") in (None, "running",
                                                        "worker_end_without_status"):
            rec["status"] = f"child_exit_{rc}"
        elif rec.get("status") == "running":
            rec["status"] = "worker_died_" + str(rc)
        rec["stderr_tail"] = _tail(errpath)
        _atomic_json_dump(outpath, rec)
        doc["runs"].append(rec)
        flush()
        print(f"[aic7] DONE {rid} status={rec['status']} rc={rc} wall={wall}s",
              flush=True)


def _driver_common(ap):
    ap.add_argument("--meshes", default="1,2,3")
    ap.add_argument("--budget-s", type=float, default=None,
                    help="stop STARTING rungs past this elapsed time "
                         "(default: the whole vq reservation less "
                         f"{WALL_FINALIZATION_GRACE_S:.0f}s, or "
                         f"{DEFAULT_BUDGET_S / 3600.0:.0f} h when vq exported "
                         "no wall)")
    ap.add_argument("--rung-cap-s", type=float, default=None,
                    help="per-rung kill deadline. Under a vq reservation this "
                         "is a FLOOR: a rung may take its share of the "
                         "remaining budget. Passing it explicitly pins it as "
                         "a hard cap (default: this mode's entry in "
                         "DEFAULT_RUNG_CAP_S)")
    ap.add_argument("--conv-tol", type=float, default=1e-9)
    ap.add_argument("--max-iter", type=int, default=128)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "worker":
        return worker(argv[1], argv[2])

    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    p_lad = sub.add_parser("lad", help="HF/DFT convergence ladder (O8/O24)")
    p_lad.add_argument("--system", required=True, choices=sorted(SYSTEMS))
    p_lad.add_argument("--basis", required=True)
    p_lad.add_argument("--route", required=True, choices=("gdf", "real-gamma"))
    p_lad.add_argument("--methods", type=_parse_method_list, default="rhf")
    _driver_common(p_lad)

    p_mad = sub.add_parser("mad", help="Madelung / construction-gap ladder (O9)")
    p_mad.add_argument("--system", required=True, choices=sorted(SYSTEMS))
    p_mad.add_argument("--basis", required=True)
    p_mad.add_argument("--constructions", default="union12,aiccm2026dev-a,neutral")
    _driver_common(p_mad)

    p_par = sub.add_parser("par", help="Theorem-1 parity points (direct vs gdf)")
    p_par.add_argument("--cases", required=True,
                       help="comma-separated system:basis:mesh triples")
    p_par.add_argument("--methods", type=_parse_method_list, default="rhf",
                       help="comma- or plus-separated: rhf and/or a libxc functional "
                            "(pbe).  A pure functional carries NO exact "
                            "exchange, so a parity gap that survives it "
                            "cannot be an exchange-divergence term.")
    p_par.add_argument("--symmetry", default="default",
                       choices=("default", "0", "1"),
                       help="GDF pair-star symmetry; 'default' leaves the "
                            "route default (True) in place")
    p_par.add_argument("--aux-basis", default=None,
                       help="override the auxiliary fitting basis on BOTH "
                            "routes")
    p_par.add_argument("--budget-s", type=float, default=None)
    p_par.add_argument("--rung-cap-s", type=float, default=None)
    p_par.add_argument("--conv-tol", type=float, default=1e-9)
    p_par.add_argument("--max-iter", type=int, default=128)

    p_prb = sub.add_parser("probe", help="nrep=(1,1,1) GDF pathology 2x2 design")
    p_prb.add_argument("--cases", required=True,
                       help="comma-separated system:basis:mesh:symmetry quads "
                            "(symmetry in {1,0})")
    p_prb.add_argument("--budget-s", type=float, default=None)
    p_prb.add_argument("--rung-cap-s", type=float, default=None)
    p_prb.add_argument("--conv-tol", type=float, default=1e-9)
    p_prb.add_argument("--max-iter", type=int, default=128)

    p_itr = sub.add_parser("itr", help="max_iter sweep at fixed nrep "
                                       "(nrep=(1,1,1) cost shape)")
    p_itr.add_argument("--cases", required=True,
                       help="comma-separated system:basis:mesh triples")
    p_itr.add_argument("--iter-caps", default="1,2,3,5,8",
                       help="comma-separated iteration caps to sweep")
    p_itr.add_argument("--budget-s", type=float, default=None)
    p_itr.add_argument("--rung-cap-s", type=float, default=None)
    p_itr.add_argument("--conv-tol", type=float, default=1e-9)
    p_itr.add_argument("--max-iter", type=int, default=128)

    p_dia = sub.add_parser("diag", help="stage-resolved STO-3G vs pob-TZVP probe")
    p_dia.add_argument("--cases",
                       default="lih:sto-3g:1,lih:pob-tzvp-rev2:1,"
                               "mgo:sto-3g:1,mgo:pob-tzvp-rev2:1,"
                               "lih:sto-3g:2,mgo:sto-3g:2")
    p_dia.add_argument("--budget-s", type=float, default=None)
    p_dia.add_argument("--rung-cap-s", type=float, default=None)
    p_dia.add_argument("--conv-tol", type=float, default=1e-9)
    p_dia.add_argument("--max-iter", type=int, default=128)

    args = ap.parse_args(argv)
    _resolve_driver_limits(ap, args)
    _validate_driver_limits(ap, args)
    cpus = int(os.environ.get("VQ_CPUS") or os.environ.get("SLURM_CPUS_PER_TASK") or 1)
    t0 = time.time()
    budget_started = time.monotonic()
    root_methods = None

    if args.mode == "lad":
        methods = args.methods
        root_methods = methods
        meshes = sorted(
            (int(x) for x in args.meshes.split(",") if x.strip()),
            reverse=True,
        )
        tag = f"lad-{args.system}-{args.basis}-{args.route}-{_method_tag(methods)}"
        if args.route == "real-gamma":
            # One rung per MESH: the fold cderi dominates and is shared.
            rungs = [{"kind": "lad", "system": args.system, "basis": args.basis,
                      "route": args.route, "methods": methods, "mesh": n,
                      "max_iter": args.max_iter, "conv_tol": args.conv_tol}
                     for n in meshes]
        else:
            rungs = [{"kind": "lad", "system": args.system, "basis": args.basis,
                      "route": args.route, "method": m, "mesh": n,
                      "max_iter": args.max_iter, "conv_tol": args.conv_tol}
                     for n in meshes for m in methods]
        route_threads = args.route
        schema = "aiccm-o8-ladder/3"
        order_policy = (
            "target-first: mesh descending; method order preserved within mesh"
        )
    elif args.mode == "par":
        tag = "par-theorem1"
        sym = None if args.symmetry == "default" else bool(int(args.symmetry))
        pmethods = args.methods
        root_methods = pmethods
        rungs = []
        for case in args.cases.split(","):
            sy, b, n = case.split(":")
            for m in pmethods:
                rungs.append({"kind": "par", "system": sy.strip(),
                              "basis": b.strip(), "mesh": int(n), "method": m,
                              "symmetry": sym, "aux_basis": args.aux_basis,
                              "max_iter": args.max_iter,
                              "conv_tol": args.conv_tol})
        route_threads = "gdf"
        schema = "aiccm-theorem1-parity/1"
        order_policy = "operator case order; method order preserved within case"
    elif args.mode == "probe":
        tag = "probe-gdf-nrep1"
        rungs = []
        for case in args.cases.split(","):
            sy, b, n, sym = case.split(":")
            rungs.append({"kind": "probe", "system": sy.strip(), "basis": b.strip(),
                          "mesh": int(n), "symmetry": bool(int(sym)),
                          "max_iter": args.max_iter, "conv_tol": args.conv_tol})
        route_threads = "gdf"
        schema = "aiccm-gdf-nrep1-pathology/1"
        order_policy = "operator case order preserved"
    elif args.mode == "itr":
        tag = "itr-gdf-cost-shape"
        caps = sorted(
            (int(x) for x in args.iter_caps.split(",") if x.strip()),
            reverse=True,
        )
        rungs = []
        for case in args.cases.split(","):
            sy, b, n = case.split(":")
            for c in caps:
                rungs.append({"kind": "itr", "system": sy.strip(),
                              "basis": b.strip(), "mesh": int(n),
                              "iter_cap": c, "max_iter": args.max_iter,
                              "conv_tol": args.conv_tol})
        route_threads = "gdf"
        schema = "aiccm-gdf-nrep1-cost-shape/1"
        order_policy = (
            "target-first: operator case order; iteration caps descending "
            "within case"
        )
    elif args.mode == "mad":
        cons = [c.strip() for c in args.constructions.split(",") if c.strip()]
        meshes = sorted(
            (int(x) for x in args.meshes.split(",") if x.strip()),
            reverse=True,
        )
        tag = f"mad-{args.system}-{args.basis}"
        rungs = [{"kind": "mad", "system": args.system, "basis": args.basis,
                  "construction": c, "mesh": n,
                  "max_iter": args.max_iter, "conv_tol": args.conv_tol}
                 for n in meshes for c in cons]
        route_threads = "gdf"
        schema = "aiccm-madelung-sensitivity/2"
        order_policy = (
            "target-first: mesh descending; construction order preserved "
            "within mesh"
        )
    else:
        tag = "diag-direct-basis-cost"
        rungs = []
        for case in args.cases.split(","):
            s, b, n = case.split(":")
            rungs.append({"kind": "diag", "system": s.strip(), "basis": b.strip(),
                          "mesh": int(n), "max_iter": args.max_iter,
                          "conv_tol": args.conv_tol})
        route_threads = "real-gamma"
        schema = "aiccm-direct-basis-cost-diagnostic/1"
        order_policy = "operator case order preserved"

    if not rungs:
        ap.error("planned rung set is empty")

    workdir, outdir = _prepare_outdir(tag)
    outfile = os.path.join(outdir, "results.json")
    planned_rungs = [
        {"rung_id": _rung_id(index, spec), "spec": dict(spec)}
        for index, spec in enumerate(rungs)
    ]

    doc = {
        "schema": schema,
        "driver_schema": "aic7-driver/2",
        "wave": "aic7 (2026-08-24, wave 3)",
        "tag": tag,
        "argv": argv,
        "cpus": cpus,
        "vq_workdir": workdir,
        "outdir": outdir,
        "budget_s": args.budget_s,
        "rung_cap_s": args.rung_cap_s,
        "budget_kill_grace_s": BUDGET_KILL_GRACE_S,
        # Issue #120: record where each limit came from, so a judging pass
        # can tell "we ran out of our own patience" from "we ran out of
        # wall" without reconstructing the argv.
        "vq_wall_time_s": args.vq_wall_s,
        "budget_source": args.budget_source,
        "rung_cap_source": args.rung_cap_source,
        "rung_cap_is_floor": args.rung_cap_is_floor,
        "rung_order_policy": order_policy,
        "planned_rungs": planned_rungs,
        "rung_isolation": "one child process per rung, SIGTERM then SIGKILL at "
                          "the cap; the child flushes after every stage so a "
                          "killed rung still reports what it completed",
        "no_predictive_skip": "rungs are attempted and measured; nothing is "
                              "refused on an n_cells**1.5 extrapolation "
                              "(that heuristic skipped 8/15 rungs of "
                              "aic-o8-lih-tz-gdf on 2026-08-22 while the true "
                              "cost was non-monotone in the mesh)",
        "normalisation_note": "issue #177 / item C8: *_direct and *_scalable "
                              "return supercell totals, *_gdf returns per unit "
                              "cell; every record carries both",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_rungs_planned": len(rungs),
        "runs": [],
        "status": "running",
        "_t0": t0,
        "_budget_started_monotonic": budget_started,
    }
    if root_methods is not None:
        doc["methods"] = root_methods

    def flush():
        payload = {key: value for key, value in doc.items()
                   if not key.startswith("_")}
        _atomic_json_dump(outfile, payload)

    flush()

    # LIVE release stamp, read in a child of this very job step -- not from a
    # cached constant and not from pslib.version_gate() (fleet brief: 8 of its
    # 9 copies falsely claim to probe).  Doubles as a proof the interpreter and
    # the vibeqc import work before any rung burns wall time.
    stamppath = os.path.join(outdir, "producer.json")
    stamp_specpath = _write_stamp_spec(outdir)
    stamp_elapsed = _budget_elapsed(doc)
    stamp_remaining = max(0.0, args.budget_s - stamp_elapsed)
    stamp_usable = max(0.0, stamp_remaining - BUDGET_KILL_GRACE_S)
    stamp_cap = min(900.0, stamp_usable)
    stamp_budget_limited = stamp_cap < 900.0
    doc["stamp_driver"] = {
        "attempted": stamp_cap > 0.0,
        "budget_limited": stamp_budget_limited,
        "elapsed_at_start_s": round(stamp_elapsed, 3),
        "remaining_at_start_s": round(stamp_remaining, 3),
        "allotted_s": round(stamp_cap, 1),
    }
    if stamp_cap <= 0.0:
        doc["stamp_worker_returncode"] = "budget_killed"
    else:
        try:
            st = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "worker",
                 stamp_specpath, stamppath],
                timeout=stamp_cap, check=False)
            doc["stamp_worker_returncode"] = st.returncode
        except subprocess.TimeoutExpired:
            doc["stamp_worker_returncode"] = (
                "budget_killed" if stamp_budget_limited else "timeout"
            )
    if os.path.exists(stamppath):
        try:
            with open(stamppath) as fh:
                doc["producer"] = json.load(fh).get("producer_full")
        except (OSError, ValueError) as exc:
            doc["producer_error"] = repr(exc)
    flush()
    v = (doc.get("producer") or {}).get("vibeqc_version")
    print(f"[aic7] {tag}: LIVE vibeqc={v} cpus={cpus} rungs={len(rungs)} "
          f"budget={args.budget_s:.0f}s cap={args.rung_cap_s:.0f}s out={outfile}",
          flush=True)

    try:
        # Producer identity is a precondition, not best-effort decoration.
        # Check it before the first rung can burn compute so a broken stamp is
        # an explicit partial run rather than another unattributable result.
        if doc["stamp_driver"]["attempted"]:
            _require_linked_library_provenance(doc.get("producer"))
        run_rungs(doc, outdir, rungs, cpus, route_threads, args.budget_s,
                  args.rung_cap_s, flush, args.rung_cap_is_floor)

        if args.mode == "mad":
            _mad_gaps(doc)
        if args.mode == "par":
            _parity_rollup(doc)
    except Exception as exc:  # noqa: BLE001 - persist driver failure provenance
        doc["driver_error"] = repr(exc)
        doc["driver_traceback"] = traceback.format_exc()
        print(f"[aic7] DRIVER FAILED: {exc!r}", file=sys.stderr, flush=True)

    returncode = _finalize_run(doc)
    doc["elapsed_s"] = round(_budget_elapsed(doc), 1)
    doc["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    flush()
    print(f"[aic7] {tag}: {doc['n_rungs_delivered']}/{doc['n_rungs_planned']} "
          f"rungs delivered; status={doc['status']} in {doc['elapsed_s']}s "
          f"-> {outfile}", flush=True)
    return returncode


def _write_stamp_spec(outdir):
    path = os.path.join(outdir, "producer.spec.json")
    _atomic_json_dump(path, {"kind": "stamp"})
    return path


def _parity_rollup(doc, tol_ha_per_cell=1e-6):
    """State the Theorem-1 verdict WITH its denominator, or refuse to state one.

    A gate that reports ALL_POINTS_PASS=false on n_points=0 has not run.  This
    roll-up therefore reports ``domain_empty`` and ``all_points_pass=None``
    when no rung produced both routes, instead of a boolean that looks like a
    measurement.
    """
    pts = [r for r in doc["runs"] if r.get("parity_point_admissible")]
    fails = [r for r in pts if r.get("abs_delta_ha_per_cell", 0.0) > tol_ha_per_cell]
    doc["theorem1_parity"] = {
        "tol_ha_per_cell": tol_ha_per_cell,
        "n_rungs_attempted": sum(1 for r in doc["runs"] if r.get("spec", {}).get("kind") == "par"),
        "n_points_with_both_routes": len(pts),
        "n_failures": len(fails),
        "domain_empty": len(pts) == 0,
        "all_points_pass": (None if not pts else len(fails) == 0),
        "verdict_meaningful": len(pts) > 0,
        "failures": [{"system": r["spec"]["system"], "basis": r["spec"]["basis"],
                      "mesh": r["spec"]["mesh"],
                      "delta_ha_per_cell": r.get("direct_minus_gdf_per_cell_ha")}
                     for r in fails],
        "points": [{"system": r["spec"]["system"], "basis": r["spec"]["basis"],
                    "mesh": r["spec"]["mesh"],
                    "delta_ha_per_cell": r.get("direct_minus_gdf_per_cell_ha")}
                   for r in pts],
    }
    t = doc["theorem1_parity"]
    if t["domain_empty"]:
        print("[aic6-par] THEOREM1 DOMAIN EMPTY (0 points with both routes) -- "
              "no verdict is possible from this member", flush=True)
    else:
        print(f"[aic6-par] THEOREM1: {t['n_failures']} failures / "
              f"{t['n_points_with_both_routes']} points with both routes "
              f"at tol {tol_ha_per_cell:g} Ha/cell -> "
              f"all_points_pass={t['all_points_pass']}", flush=True)


def _mad_gaps(doc):
    """Per-mesh construction gaps, formed only between converged per-cell values."""
    per_mesh = {}
    for r in doc["runs"]:
        if r.get("status") == "ok" and r.get("construction") and "energy_per_cell" in r:
            per_mesh.setdefault(r["mesh"], {})[r["construction"]] = (
                r["energy_per_cell"], r["n_atoms_unit"])
    for n in sorted(per_mesh):
        vals = per_mesh[n]
        gaps = {}
        for a in vals:
            for b in vals:
                if a < b:
                    d = vals[a][0] - vals[b][0]
                    gaps[f"{a}_minus_{b}"] = {"per_cell_ha": d,
                                              "per_atom_ha": d / vals[a][1]}
        if gaps:
            doc["runs"].append({"mesh": n, "record": "construction_gap",
                                "gaps": gaps, "status": "derived"})
            for k, val in gaps.items():
                print(f"[aic7] N={n} GAP {k}: {val['per_atom_ha']:+.6f} Ha/atom",
                      flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
