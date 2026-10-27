"""G0-A CPU profiling, component-timing and provenance harness.

GPU programme stage G0-A (``docs/design/gpu_g0_g1m_plan.md``). It measures
the current CPU SCF so later GPU work can be chosen from measured
bottlenecks. Nothing here runs on a GPU.

Subcommands::

    manifest    print the fixed G0 workload manifest as JSON
    run-case    run one case in this process and write one raw record
    run-batch   run a batch of cases as fresh child processes (vq payload)
    submit      stage a batch under the external output root and submit it
                through ``vq``
    summarize   reduce raw records to a public-safe summary (JSON/Markdown)

The workload is fixed: the H2O, naphthalene and n-hexadecane coordinates
of ``parity_matrix_orca/cases.py`` (one Angstrom-to-Bohr conversion), the
def2-SVP orbital basis, RHF / RKS-PBE / RKS-PBE0, and four Fock-build
routes: forced direct, DF, RIJCOSX and the resolved default (AUTO). Every
option the harness sets is listed in :data:`RECIPE` and serialised with
each record; unlike the speed runner it applies no hidden tolerance clamp.

Raw records, logs, ``.out`` files and absolute paths stay under the
explicitly selected external output root (``--output-root`` or
``VIBEQC_RUNS_DIR``), never inside the checkout. Only ``summarize`` output,
which passes :func:`assert_public_safe`, is meant for the repository.

This module must stay importable from a bare ``vq`` workspace: the worker
subcommands use only the standard library and ``vibeqc``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import resource
import shutil
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

SCHEMA = "vibeqc.gpu_g0.record/1"
SUMMARY_SCHEMA = "vibeqc.gpu_g0.summary/1"
RESULT_MARKER = "VIBEQC-GPU-G0-RECORD:"

# Size label -> regression key in parity_matrix_orca/cases.py::GEOMETRIES.
SYSTEMS = {"S": "H2O", "M": "naphthalene", "L": "n-hexadecane"}
BASIS = "def2-svp"
METHODS = ("RHF", "PBE", "PBE0")
ROUTES = ("direct", "df", "rijcosx", "auto")

# RHFOptions/RKSOptions::thresh_cosx default (rhf.hpp); not settable from
# Python, so it is the value every driver call uses.
DRIVER_THRESH_COSX_DEFAULT = 1e-6

# Exact-exchange fraction a_x per method; PBE has no exact-exchange build.
EXCHANGE_FRACTION = {"RHF": 1.0, "PBE": 0.0, "PBE0": 0.25}

# The common G0 recipe. Every value here is set explicitly on the options
# object; driver defaults are recorded but never silently relied upon.
RECIPE: Dict[str, Any] = {
    "conv_tol_energy": 1e-9,
    # ||F D S - S D F||_Frobenius, the drivers' stated normalization.
    "conv_tol_grad": 1e-6,
    "max_iter": 200,
    "initial_guess": "SAD",
    "scf_mode_auto_threshold": 140,
    "cosx_grid_level": 2,
    "df_aux_basis": "def2-universal-jkfit",
    "rijcosx_aux_basis": "def2-universal-jfit",
    "xc_grid": {
        "atomic_grid_profile": "Generic",
        "n_radial": 75,
        "angular": "ProductGaussLegendre",
        "n_theta": 17,
        "n_phi": 36,
        "angular_pruning": "None",
        "partition": "Becke",
        "becke_k": 3,
    },
}

PROTOCOL = {
    "warmup_runs": 1,
    "timed_runs": 3,
    "per_run_budget_s": 3600,
    "order": "randomized with recorded seed",
    "process": "one fresh process per run",
}

# Summary grouping of phase paths by leaf name. Exclusive times of all
# groups plus "unclassified" reconcile with the SCF session wall time.
PHASE_GROUPS = {
    "df.construction": "df_construction",
    "fock.jk_fused": "jk_fused",
    "fock.j": "j",
    "df.j": "j",
    "j": "j",
    "fock.k": "k",
    "df.k": "k",
    "cosx.k": "k",
    "k": "k",
    "cosx.one_center": "k_correction",
    "xc": "ao_xc_vxc",
    "xc_grid": "grid_construction",
    "cosx_grid": "grid_construction",
    "cosx.grid_stage": "grid_construction",
    "diagonalization": "diagonalization",
    "second_order_step": "second_order_step",
    "accelerator": "accelerator",
    "density": "density",
    "stability": "stability",
    "final_rebuild": "final_rebuild_other",
}


# --------------------------------------------------------------------------
# Manifest


def _cases_module():
    try:
        from examples.regression.parity_matrix_orca import cases
    except ImportError:  # standalone checkout layout
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from examples.regression.parity_matrix_orca import cases
    return cases


def geometry_sha256(atoms: Sequence[Sequence[Any]]) -> str:
    """Content hash of ``[(Z, [x, y, z]_bohr), ...]`` (exact float repr)."""
    canon = [[int(z), [repr(float(c)) for c in xyz]] for z, xyz in atoms]
    blob = json.dumps(canon, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def route_request(route: str, method: str) -> Dict[str, Any]:
    """The requested Fock-build settings for one manifest route label."""
    if route == "direct":
        return {"density_fit": False, "cosx": False, "scf_mode": "DIRECT",
                "aux_basis": ""}
    if route == "df":
        return {"density_fit": True, "cosx": False, "scf_mode": "AUTO",
                "aux_basis": RECIPE["df_aux_basis"]}
    if route == "rijcosx":
        return {"density_fit": True, "cosx": True, "scf_mode": "AUTO",
                "aux_basis": RECIPE["rijcosx_aux_basis"]}
    if route == "auto":
        return {"density_fit": False, "cosx": False, "scf_mode": "AUTO",
                "aux_basis": ""}
    raise ValueError(f"unknown route {route!r}")


def build_manifest(
    systems: Optional[Iterable[str]] = None,
    methods: Optional[Iterable[str]] = None,
    routes: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """The fixed G0 workload, optionally filtered."""
    cases = _cases_module()
    sizes = {v: k for k, v in SYSTEMS.items()}
    wanted_systems = list(systems or SYSTEMS.values())
    for name in wanted_systems:
        if name not in sizes:
            raise ValueError(f"{name!r} is not a G0 system")
    out = []
    for system in wanted_systems:
        atoms = cases.geometry_bohr(system)
        gsha = geometry_sha256(atoms)
        for method in methods or METHODS:
            if method not in METHODS:
                raise ValueError(f"{method!r} is not a G0 method")
            for route in routes or ROUTES:
                out.append({
                    "case_id": f"{system}__{BASIS}__{method}__{route}",
                    "size": sizes[system],
                    "system": system,
                    "basis": BASIS,
                    "method": method,
                    "route": route,
                    "charge": 0,
                    "multiplicity": 1,
                    "request": route_request(route, method),
                    "atoms_bohr": [[z, list(xyz)] for z, xyz in atoms],
                    "geometry_sha256": gsha,
                })
    return out


# --------------------------------------------------------------------------
# Options and resolution (worker side; imports vibeqc)


def _enum(ns: Any, name: str) -> Any:
    return getattr(ns, name)


def make_options(case: Dict[str, Any], *, component_timing: bool):
    """(runner, options) with every recipe value set explicitly."""
    import vibeqc
    from vibeqc import _vibeqc_core as core

    method = case["method"]
    if method == "RHF":
        opts = vibeqc.RHFOptions()
        runner = vibeqc.run_rhf
    else:
        opts = vibeqc.RKSOptions()
        opts.functional = method
        runner = vibeqc.run_rks
        g = opts.grid
        spec = RECIPE["xc_grid"]
        g.atomic_grid_profile = _enum(core.AtomicGridProfile,
                                      spec["atomic_grid_profile"])
        g.n_radial = spec["n_radial"]
        g.angular = _enum(core.AngularScheme, spec["angular"])
        g.n_theta = spec["n_theta"]
        g.n_phi = spec["n_phi"]
        g.angular_pruning = _enum(core.AngularPruning,
                                  spec["angular_pruning"])
        g.partition = _enum(core.AtomicPartition, spec["partition"])
        g.becke_k = spec["becke_k"]
        opts.grid = g
    opts.conv_tol_energy = RECIPE["conv_tol_energy"]
    opts.conv_tol_grad = RECIPE["conv_tol_grad"]
    opts.max_iter = RECIPE["max_iter"]
    opts.initial_guess = _enum(vibeqc.InitialGuess, RECIPE["initial_guess"])
    opts.scf_mode_auto_threshold = RECIPE["scf_mode_auto_threshold"]
    req = case["request"]
    opts.density_fit = req["density_fit"]
    opts.cosx = req["cosx"]
    opts.aux_basis = req["aux_basis"]
    opts.scf_mode = _enum(vibeqc.SCFMode, req["scf_mode"])
    if req["cosx"]:
        opts.cosx_grid_level = RECIPE["cosx_grid_level"]
    opts.component_timing = bool(component_timing)
    return runner, opts


def serialize(obj: Any, depth: int = 0) -> Any:
    """JSON view of a pybind options object, recursively; arrays by shape."""
    if depth > 4:
        return repr(obj)
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else repr(obj)
    if isinstance(obj, (list, tuple)):
        return [serialize(x, depth + 1) for x in obj]
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        return {"array_shape": list(obj.shape)}
    if hasattr(obj, "name") and hasattr(obj, "value") and type(obj).__module__.startswith("vibeqc"):
        return str(obj.name)
    out: Dict[str, Any] = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception:  # noqa: BLE001 - recorded, not hidden
            out[name] = "<unreadable>"
            continue
        if callable(value):
            continue
        out[name] = serialize(value, depth + 1)
    return out if out else repr(obj)


def basis_sha256(basis) -> str:
    """Content hash of the resolved basis: per shell atom, l, pure, exponents,
    contraction coefficients as stored, and origin."""
    rows = []
    for sh in basis.shells():
        rows.append([
            int(sh.atom_index), int(sh.l), bool(sh.pure),
            [repr(float(x)) for x in sh.exponents],
            [repr(float(c)) for c in sh.coefficients],
            [repr(float(c)) for c in sh.origin],
        ])
    return hashlib.sha256(
        json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def resolve_settings(case: Dict[str, Any], mol, basis, opts) -> Dict[str, Any]:
    """Record what the drivers will actually do for this request."""
    import vibeqc

    req = case["request"]
    n_ao = int(basis.nbasis)
    n_elec = int(mol.n_electrons())
    a_x = EXCHANGE_FRACTION[case["method"]]
    resolved: Dict[str, Any] = {
        "n_ao": n_ao,
        "n_electrons": n_elec,
        "n_occ": n_elec // 2,
        "exchange_fraction": a_x,
        "basis_sha256": basis_sha256(basis),
    }
    if req["density_fit"]:
        aux = vibeqc.BasisSet(mol, req["aux_basis"])
        resolved["n_aux"] = int(aux.nbasis)
        resolved["aux_basis_sha256"] = basis_sha256(aux)
        resolved["scf_mode_resolved"] = "not_used"
        if req["cosx"]:
            card = vibeqc.cosx_basis_cardinality_from_name(basis.name)
            # thresh_cosx is not bound on the SCF options, so the drivers
            # always use the C++ default; resolve with that same value.
            thresh = getattr(opts, "thresh_cosx", DRIVER_THRESH_COSX_DEFAULT)
            variant = vibeqc.resolve_cosx_variant(
                opts.cosx_variant, thresh, card, opts.cosx_grid_level)
            level = vibeqc.resolve_cosx_grid_level(opts.cosx_grid_level, card)
            stages = vibeqc.cosx_grid_stages_for_level(level)
            resolved["builder"] = "ri_j_cosx_k" if a_x else "ri_j_only"
            resolved["cosx_variant"] = str(variant.name)
            resolved["thresh_cosx"] = thresh
            resolved["cosx_grid_level"] = int(level)
            resolved["cosx_stage_points"] = [
                int(vibeqc.build_grid(mol, st).points.shape[0]) for st in stages]
        else:
            resolved["builder"] = "df_j_k" if a_x else "df_j_only"
        resolved["incremental_fock_active"] = False
    else:
        mode = vibeqc._vibeqc_core.resolve_scf_mode(opts.scf_mode, n_ao,
                                       opts.scf_mode_auto_threshold)
        resolved["scf_mode_resolved"] = str(mode.name)
        base = "direct" if mode.name == "DIRECT" else "conventional_four_index"
        resolved["builder"] = base
        resolved["incremental_fock_active"] = bool(
            mode.name == "DIRECT" and opts.incremental_fock)
    resolved["k_build"] = ("not_applicable" if a_x == 0.0 else
                           ("cosx" if req["cosx"] else "exact_or_df"))
    if case["method"] != "RHF":
        resolved["xc_grid_points"] = int(
            vibeqc.build_grid(mol, opts.grid).points.shape[0])
    return resolved


def dedupe_key(case: Dict[str, Any], resolved: Dict[str, Any],
               options: Dict[str, Any]) -> str:
    """Identical keys mean identical resolved settings (e.g. AUTO == DIRECT)."""
    opts = {k: v for k, v in options.items() if k not in ("scf_mode",
                                                           "component_timing")}
    payload = {
        "system": case["system"], "basis": case["basis"],
        "method": case["method"],
        "builder": resolved.get("builder"),
        "scf_mode_resolved": resolved.get("scf_mode_resolved"),
        "options": opts,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def deduplicate_cases(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop a case whose resolved settings equal an earlier case's.

    Manifest order puts forced direct before AUTO, so an AUTO that resolves
    to DIRECT is run once under the direct label and keeps its own label in
    ``aliases``. Any difference in resolved settings keeps both runs.
    """
    import vibeqc

    kept: Dict[str, Dict[str, Any]] = {}
    for case in cases:
        mol = vibeqc.Molecule(
            [vibeqc.Atom(int(z), list(xyz)) for z, xyz in case["atoms_bohr"]],
            charge=int(case["charge"]), multiplicity=int(case["multiplicity"]))
        basis = vibeqc.BasisSet(mol, case["basis"])
        _runner, opts = make_options(case, component_timing=True)
        key = dedupe_key(case, resolve_settings(case, mol, basis, opts),
                         serialize(opts))
        if key in kept:
            kept[key].setdefault("aliases", []).append(case["case_id"])
        else:
            kept[key] = dict(case)
    return list(kept.values())


# --------------------------------------------------------------------------
# Provenance


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cpu_model() -> str:
    try:
        if sys.platform == "darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True,
                timeout=5).strip()
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or "unknown"


THREAD_ENV = ("OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES",
              "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


def runtime_provenance() -> Dict[str, Any]:
    """Raw provenance (contains local paths; stays in the external root)."""
    import vibeqc
    from vibeqc import _vibeqc_core as core

    prov: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "vibeqc_file": vibeqc.__file__,
        "native_file": core.__file__,
        "native_sha256": _sha256_file(core.__file__),
        "vibeqc_version": getattr(vibeqc, "__version__", None),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "hostname": socket.gethostname(),
        "thread_env": {k: os.environ.get(k) for k in THREAD_ENV},
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
    }
    try:
        from vibeqc.banner import build_info
        prov["build_info"] = dict(build_info())
    except Exception as exc:  # noqa: BLE001
        prov["build_info"] = {"error": repr(exc)}
    for name in ("native_build_info", "blas_info"):
        fn = getattr(core, name, None)
        prov[name] = dict(fn()) if fn else None
    for name in ("libint_version", "libxc_version", "libecpint_version"):
        fn = getattr(core, name, None)
        try:
            prov[name] = fn() if fn else None
        except Exception as exc:  # noqa: BLE001
            prov[name] = repr(exc)
    try:
        import numpy
        prov["numpy"] = numpy.__version__
    except ImportError:
        prov["numpy"] = None
    return prov


def peak_rss_bytes() -> int:
    """Process high-water RSS (ru_maxrss is bytes on macOS, KiB on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def timing_to_dict(ct) -> Dict[str, Any]:
    if ct is None or not ct.enabled:
        return {"enabled": False}

    def phases(rows):
        return [{"path": r.path, "inclusive_s": r.inclusive_s,
                 "exclusive_s": r.exclusive_s, "calls": int(r.calls)}
                for r in rows]

    return {
        "enabled": True,
        "wall_s": ct.wall_s,
        "unclassified_s": ct.unclassified_s,
        "n_segments": int(ct.n_segments),
        "phases": phases(ct.phases),
        "iterations": [{"segment": int(it.segment), "iter": int(it.iter),
                        "wall_s": it.wall_s, "phases": phases(it.phases)}
                       for it in ct.iterations],
        "memory": {m.label: m.bytes for m in ct.memory},
    }


def group_phases(timing: Dict[str, Any]) -> Dict[str, float]:
    """Exclusive seconds per summary group; adds 'setup' and 'unclassified'."""
    groups: Dict[str, float] = {}
    for row in timing.get("phases", []):
        leaf = row["path"].rsplit("/", 1)[-1]
        top = row["path"].split("/", 1)[0]
        if leaf in PHASE_GROUPS:
            key = PHASE_GROUPS[leaf]
        elif top.startswith("setup."):
            key = "setup"
        else:
            key = "other:" + leaf
        groups[key] = groups.get(key, 0.0) + row["exclusive_s"]
    groups["unclassified"] = timing.get("unclassified_s", 0.0)
    return groups


def reconcile(timing: Dict[str, Any], rel_tol: float = 1e-6,
              abs_tol: float = 1e-6) -> Dict[str, Any]:
    """Check that exclusive phases + unclassified equal the session wall."""
    total = sum(r["exclusive_s"] for r in timing.get("phases", []))
    total += timing.get("unclassified_s", 0.0)
    wall = timing.get("wall_s", 0.0)
    ok = abs(total - wall) <= max(abs_tol, rel_tol * wall)
    neg = [r["path"] for r in timing.get("phases", [])
           if r["exclusive_s"] < -abs_tol]
    unc_ok = timing.get("unclassified_s", 0.0) >= -abs_tol
    return {"sum_s": total, "wall_s": wall, "ok": bool(ok and not neg and unc_ok),
            "negative_exclusive": neg}


# --------------------------------------------------------------------------
# Worker


def _write_out_file(path: Path, case: Dict[str, Any], record: Dict[str, Any]) -> None:
    """User-facing calculation output, routed through vibeqc.output."""
    from vibeqc.output import OutputChannel, section, write

    res = record["result"]
    with OutputChannel.to_file(path):
        section("G0 CPU profiling run")
        write(f"case        {case['case_id']}\n")
        write(f"route       {case['route']}  builder={record['resolved'].get('builder')}\n")
        write(f"timing      component_timing={record['component_timing']}\n")
        write(f"converged   {res['converged']}  iterations={res['n_iter']}\n")
        write(f"energy      {res['energy']:.12f} Eh\n")
        write(f"scf wall    {record['scf_wall_s']:.6f} s\n")
        timing = record.get("timing", {})
        if timing.get("enabled"):
            section("Component timing (exclusive seconds)")
            for key, val in sorted(group_phases(timing).items()):
                write(f"  {key:24s} {val:12.6f}\n")


def run_case(case: Dict[str, Any], *, out_dir: Path, component_timing: bool,
             threads: int, run_label: str) -> Dict[str, Any]:
    import vibeqc

    t_start = time.perf_counter()
    vibeqc.set_num_threads(int(threads))
    mol = vibeqc.Molecule(
        [vibeqc.Atom(int(z), list(xyz)) for z, xyz in case["atoms_bohr"]],
        charge=int(case["charge"]), multiplicity=int(case["multiplicity"]))
    basis = vibeqc.BasisSet(mol, case["basis"])
    runner, opts = make_options(case, component_timing=component_timing)
    resolved = resolve_settings(case, mol, basis, opts)
    options_view = serialize(opts)
    rss_before = peak_rss_bytes()
    t0 = time.perf_counter()
    result = runner(mol, basis, opts)
    scf_wall = time.perf_counter() - t0
    rss_after = peak_rss_bytes()
    trace = list(result.scf_trace)
    record: Dict[str, Any] = {
        "schema": SCHEMA,
        "run_label": run_label,
        "case": {k: v for k, v in case.items() if k != "atoms_bohr"},
        "component_timing": bool(component_timing),
        "threads_requested": int(threads),
        "threads_observed": int(vibeqc.get_num_threads()),
        "xc_batch_workers_used": getattr(result, "xc_batch_workers_used", None),
        "options": options_view,
        "resolved": resolved,
        "dedupe_key": dedupe_key(case, resolved, options_view),
        "scf_wall_s": scf_wall,
        "result": {
            "energy": float(result.energy),
            "converged": bool(result.converged),
            "n_iter": int(result.n_iter),
            "final_grad_norm": float(trace[-1].grad_norm) if trace else None,
            "final_delta_e": float(trace[-1].delta_e) if trace else None,
            "stability_checked": bool(result.stability_checked),
            "stability_wall_s": float(result.stability_wall_s),
            "trace_wall_sum_s": float(sum(t.wall_s for t in trace)),
        },
        "memory": {
            "process_peak_rss_before_scf_bytes": rss_before,
            "process_peak_rss_bytes": rss_after,
        },
        "timing": timing_to_dict(getattr(result, "component_timing", None)),
        "provenance": runtime_provenance(),
    }
    if record["timing"]["enabled"]:
        record["timing"]["reconciliation"] = reconcile(record["timing"])
        mem = record["timing"]["memory"]
        record["memory"]["tracked_live_tensor_bytes"] = mem.get(
            "df.live_tracked_bytes")
        record["memory"]["tracked_construction_peak_bytes"] = mem.get(
            "df.construction_tracked_peak_bytes")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{run_label}__{case['case_id']}__ct{int(component_timing)}__t{threads}"
    t_out = time.perf_counter()
    _write_out_file(out_dir / f"{stem}.out", case, record)
    record["output_s"] = time.perf_counter() - t_out
    record["process_wall_s"] = time.perf_counter() - t_start
    (out_dir / f"{stem}.json").write_text(json.dumps(record, indent=1),
                                          encoding="utf-8")
    return record


def cmd_run_case(args: argparse.Namespace) -> int:
    case = json.loads(Path(args.case_json).read_text(encoding="utf-8"))
    try:
        rec = run_case(case, out_dir=Path(args.out_dir),
                       component_timing=bool(args.component_timing),
                       threads=int(args.threads), run_label=args.label)
    except Exception as exc:  # noqa: BLE001 - reported as INCOMPLETE evidence
        print(RESULT_MARKER + json.dumps({"status": "error",
                                          "case_id": case.get("case_id"),
                                          "note": f"{type(exc).__name__}: {exc}"}),
              flush=True)
        return 1
    print(RESULT_MARKER + json.dumps({
        "status": "ok", "case_id": case["case_id"],
        "energy": rec["result"]["energy"],
        "converged": rec["result"]["converged"],
        "scf_wall_s": rec["scf_wall_s"]}), flush=True)
    return 0


def plan_batch(cases: List[Dict[str, Any]], *, threads: Sequence[int],
               timing_modes: Sequence[bool], warmup: int, repeats: int,
               seed: int) -> List[Dict[str, Any]]:
    """Run list: per (case, threads) one warm-up, then randomized timed runs."""
    rng = random.Random(seed)
    runs: List[Dict[str, Any]] = []
    timed: List[Dict[str, Any]] = []
    for case in cases:
        for t in threads:
            for w in range(warmup):
                runs.append({"case_id": case["case_id"], "threads": t,
                             "component_timing": True, "label": f"warmup{w}"})
            for ct in timing_modes:
                for r in range(repeats):
                    timed.append({"case_id": case["case_id"], "threads": t,
                                  "component_timing": ct, "label": f"rep{r}"})
    rng.shuffle(timed)
    return runs + timed


def cmd_run_batch(args: argparse.Namespace) -> int:
    """vq payload: execute a planned batch, one fresh process per run."""
    here = Path(__file__).resolve()
    batch = json.loads(Path(args.batch_json).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = {c["case_id"]: c for c in batch["cases"]}
    case_dir = out_dir / "cases"
    case_dir.mkdir(exist_ok=True)
    for cid, case in cases.items():
        (case_dir / f"{cid}.json").write_text(json.dumps(case), encoding="utf-8")
    status = []
    for run in batch["runs"]:
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = str(run["threads"])
        for var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
            env[var] = str(batch.get("blas_threads", 1))
        cmd = [sys.executable, str(here), "run-case",
               str(case_dir / f"{run['case_id']}.json"),
               "--out-dir", str(out_dir / "records"),
               "--threads", str(run["threads"]), "--label", run["label"]]
        if run["component_timing"]:
            cmd.append("--component-timing")
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                  timeout=batch.get("per_run_budget_s", 3600))
            rc, tail = proc.returncode, (proc.stdout + proc.stderr)[-4000:]
        except subprocess.TimeoutExpired:
            rc, tail = "timeout", ""
        status.append({**run, "returncode": rc,
                       "elapsed_s": time.perf_counter() - t0, "tail": tail})
        print(f"{run['label']} {run['case_id']} ct={int(run['component_timing'])} "
              f"t={run['threads']} rc={rc}", flush=True)
    (out_dir / "batch_status.json").write_text(json.dumps(status, indent=1),
                                               encoding="utf-8")
    return 0 if all(s["returncode"] == 0 for s in status) else 1


# --------------------------------------------------------------------------
# Orchestration (local side; uses the shared regression plumbing)


def cmd_submit(args: argparse.Namespace) -> int:
    from examples.regression.core.output_paths import make_run_id, resolve_output_root
    from examples.regression.parity_matrix_orca import orca_vq

    cases = build_manifest(
        systems=args.systems.split(",") if args.systems else None,
        methods=args.methods.split(",") if args.methods else None,
        routes=args.routes.split(",") if args.routes else None)
    if not args.no_dedupe:
        cases = deduplicate_cases(cases)
    threads = [int(t) for t in args.threads.split(",")]
    timing_modes = [True, False] if args.compare_timing_off else [True]
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(1 << 30)
    runs = plan_batch(cases, threads=threads, timing_modes=timing_modes,
                      warmup=args.warmup, repeats=args.repeats, seed=seed)
    root = resolve_output_root(args.output_root, create=True)
    run_dir = root / (args.run_id or make_run_id("gpu-g0"))
    workspace = run_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), workspace / "gpu_profile.py")
    batch = {"cases": cases, "runs": runs, "seed": seed,
             "blas_threads": args.blas_threads,
             "per_run_budget_s": args.per_run_budget_s,
             "protocol": PROTOCOL, "recipe": RECIPE}
    (workspace / "batch.json").write_text(json.dumps(batch, indent=1),
                                          encoding="utf-8")
    expected = {"python": args.python}
    if args.host:
        os.environ["VIBEQC_BENCHMARK_HOST"] = args.host
    proc = orca_vq._vq("submit", "-d", str(workspace), "--cpus",
                       str(max(threads)), "--wall-time-seconds",
                       str(args.per_run_budget_s * len(runs)), "--",
                       args.python, "gpu_profile.py", "run-batch", "batch.json",
                       "--out-dir", "evidence")
    jobid = proc.stdout.strip().splitlines()[-1].strip()
    (run_dir / "submission.json").write_text(json.dumps(
        {"jobid": jobid, "expected": expected, "n_runs": len(runs),
         "seed": seed}, indent=1), encoding="utf-8")
    print(json.dumps({"jobid": jobid, "run_dir": str(run_dir),
                      "n_runs": len(runs)}))
    return 0


# --------------------------------------------------------------------------
# Summary and redaction


_ABS_PATH = re.compile(r"(?<![\w.])(?:/[\w.@+-]+){2,}/?")


def _private_terms() -> List[str]:
    terms = {socket.gethostname(), socket.gethostname().split(".")[0],
             str(Path.home())}
    user = os.environ.get("USER") or os.environ.get("LOGNAME")
    if user and len(user) >= 3:
        terms.add(user)
    return sorted(t for t in terms if t and len(t) >= 3)


def redact(value: Any, terms: Optional[Sequence[str]] = None) -> Any:
    """Strip absolute paths to their basename and mask private terms."""
    terms = _private_terms() if terms is None else list(terms)
    if isinstance(value, dict):
        return {k: redact(v, terms) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, terms) for v in value]
    if isinstance(value, str):
        out = _ABS_PATH.sub(lambda m: m.group(0).rstrip("/").rsplit("/", 1)[-1],
                            value)
        for t in terms:
            out = out.replace(t, "<redacted>")
        return out
    return value


def assert_public_safe(value: Any, terms: Optional[Sequence[str]] = None) -> None:
    """Raise if a summary still carries absolute paths or private terms."""
    terms = _private_terms() if terms is None else list(terms)
    blob = json.dumps(value)
    if _ABS_PATH.search(blob):
        raise ValueError(f"absolute path in summary: {_ABS_PATH.search(blob).group(0)}")
    for t in terms:
        if t in blob:
            raise ValueError("private term in summary")


def _stats(values: List[float]) -> Optional[Dict[str, float]]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return {"median": statistics.median(vals), "min": min(vals),
            "max": max(vals), "n": len(vals)}


def load_records(raw_dir: Path) -> List[Dict[str, Any]]:
    recs = []
    for p in sorted(raw_dir.rglob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("schema") == SCHEMA:
            recs.append(data)
    return recs


def summarize(records: List[Dict[str, Any]], *,
              expected_source_sha: Optional[str] = None,
              expected_native_sha256: Optional[str] = None,
              expected_cases: Optional[List[str]] = None,
              energy_tol: float = 1e-8) -> Dict[str, Any]:
    """Reduce raw records to public summary rows. Warm-up runs are dropped."""
    timed = [r for r in records if not r["run_label"].startswith("warmup")]
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in timed:
        key = (r["case"]["case_id"], r["threads_requested"], r["component_timing"])
        groups.setdefault(key, []).append(r)
    rows = []
    for (cid, threads, ct), recs in sorted(groups.items()):
        prov = [r["provenance"] for r in recs]
        sources = {p.get("build_info", {}).get("sha_full") for p in prov}
        natives = {p.get("native_sha256") for p in prov}
        dirty = any(p.get("build_info", {}).get("dirty") for p in prov)
        identity_ok = (len(sources) == 1 and len(natives) == 1 and not dirty
                       and None not in sources
                       and (expected_source_sha is None
                            or sources == {expected_source_sha})
                       and (expected_native_sha256 is None
                            or natives == {expected_native_sha256}))
        energies = [r["result"]["energy"] for r in recs]
        converged = all(r["result"]["converged"] for r in recs)
        iters = sorted({r["result"]["n_iter"] for r in recs})
        spread = max(energies) - min(energies)
        timing_rows = [r["timing"] for r in recs if r["timing"].get("enabled")]
        phase_groups: Dict[str, Optional[Dict[str, float]]] = {}
        if timing_rows:
            keys = sorted({k for t in timing_rows for k in group_phases(t)})
            phase_groups = {k: _stats([group_phases(t).get(k, 0.0)
                                       for t in timing_rows]) for k in keys}
        recon_ok = all(t["reconciliation"]["ok"] for t in timing_rows) if timing_rows else None
        reasons = []
        if not converged:
            reasons.append("not_converged")
        if dirty:
            reasons.append("source_tree_dirty")
        if not identity_ok and not dirty:
            reasons.append("runtime_identity_mismatch")
        if spread > energy_tol:
            reasons.append("energy_spread")
        if recon_ok is False:
            reasons.append("timing_not_reconciled")
        status = "FAIL" if reasons else "PASS"
        r0 = recs[0]
        p0 = prov[0]
        nbi = p0.get("native_build_info") or {}
        rows.append({
            "case_id": cid,
            "route": r0["case"]["route"],
            "aliases": r0["case"].get("aliases", []),
            "method": r0["case"]["method"],
            "builder": r0["resolved"].get("builder"),
            "scf_mode_resolved": r0["resolved"].get("scf_mode_resolved"),
            "k_build": r0["resolved"].get("k_build"),
            "cosx_variant": r0["resolved"].get("cosx_variant"),
            "dedupe_key": r0["dedupe_key"][:16],
            "geometry_sha256": r0["case"]["geometry_sha256"],
            "basis_sha256": r0["resolved"].get("basis_sha256"),
            "component_timing": ct,
            "threads_requested": threads,
            "threads_observed": sorted({r["threads_observed"] for r in recs}),
            "dims": {k: r0["resolved"].get(k) for k in
                     ("n_ao", "n_aux", "n_occ", "xc_grid_points",
                      "cosx_stage_points")},
            "n_runs": len(recs),
            "scf_wall_s": _stats([r["scf_wall_s"] for r in recs]),
            "output_s": _stats([r.get("output_s") for r in recs]),
            "stability_wall_s": _stats([r["result"]["stability_wall_s"] for r in recs]),
            "phase_exclusive_s": phase_groups,
            "timing_reconciled": recon_ok,
            "memory": {
                "tracked_live_tensor_bytes": recs[0]["memory"].get("tracked_live_tensor_bytes"),
                "tracked_construction_peak_bytes": recs[0]["memory"].get("tracked_construction_peak_bytes"),
                "process_peak_rss_bytes": _stats([r["memory"]["process_peak_rss_bytes"] for r in recs]),
            },
            "iterations": iters,
            "energy_eh": statistics.median(energies),
            "energy_spread_eh": spread,
            "final_grad_norm_max": max((r["result"]["final_grad_norm"] or 0.0) for r in recs),
            # A driver may report convergence with its last trace residual
            # above the recipe target (e.g. a final COSX grid stage); this
            # is recorded, not reclassified.
            "final_grad_above_recipe": any(
                (r["result"]["final_grad_norm"] or 0.0) > RECIPE["conv_tol_grad"]
                for r in recs),
            "converged": converged,
            "identity_ok": identity_ok,
            "source_sha": next(iter(sources)) if len(sources) == 1 else None,
            "native_sha256": next(iter(natives)) if len(natives) == 1 else None,
            "compiler": nbi.get("compiler_version"),
            "blas": redact((p0.get("blas_info") or {}).get("libraries")),
            "cpu_model": p0.get("cpu_model"),
            "status": status,
            "fail_reasons": reasons,
        })
    present = {row["case_id"] for row in rows}
    incomplete = sorted(set(expected_cases or []) - present)
    comparisons = compare_timing_modes(rows)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "protocol": PROTOCOL,
        "recipe": RECIPE,
        "rows": rows,
        "timing_on_off": comparisons,
        "incomplete_cases": incomplete,
    }
    summary = redact(summary)
    assert_public_safe(summary)
    return summary


def compare_timing_modes(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Instrumentation on vs off: identical science, measured overhead."""
    by_key: Dict[tuple, Dict[bool, Dict[str, Any]]] = {}
    for row in rows:
        by_key.setdefault((row["case_id"], row["threads_requested"]), {})[
            row["component_timing"]] = row
    out = []
    for (cid, threads), pair in sorted(by_key.items()):
        if True not in pair or False not in pair:
            continue
        on, off = pair[True], pair[False]
        out.append({
            "case_id": cid,
            "threads_requested": threads,
            "energy_delta_eh": on["energy_eh"] - off["energy_eh"],
            "iterations_equal": on["iterations"] == off["iterations"],
            "median_wall_on_s": on["scf_wall_s"]["median"],
            "median_wall_off_s": off["scf_wall_s"]["median"],
            "overhead_fraction": on["scf_wall_s"]["median"] / off["scf_wall_s"]["median"] - 1.0,
            "ranges_overlap": not (on["scf_wall_s"]["min"] > off["scf_wall_s"]["max"]
                                   or off["scf_wall_s"]["min"] > on["scf_wall_s"]["max"]),
        })
    return out


_MD_GROUPS = ("setup", "df_construction", "grid_construction", "jk_fused",
              "j", "k", "k_correction", "ao_xc_vxc", "accelerator",
              "diagonalization", "density", "final_rebuild_other",
              "stability", "unclassified")


def _fmt(x: Optional[float], digits: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def render_markdown(summary: Dict[str, Any]) -> str:
    """Tables of measured values from a summary; no placeholders."""
    lines = ["| Case | Threads | Builder | Status | Iter | Energy (Eh) | "
             "Last residual | SCF wall median [min, max] (s) | Peak RSS (MiB) |",
             "|---|---|---|---|---|---|---|---|---|"]
    rows = [r for r in summary["rows"] if r["component_timing"]]
    for r in rows:
        w = r["scf_wall_s"]
        rss = (r["memory"]["process_peak_rss_bytes"] or {}).get("median")
        alias = "".join(f" (= {a.rsplit('__', 1)[-1]})" for a in r["aliases"])
        lines.append(
            f"| {r['case_id']}{alias} | {r['threads_requested']} | {r['builder']} | "
            f"{r['status']} | {','.join(map(str, r['iterations']))} | "
            f"{r['energy_eh']:.10f} | {r['final_grad_norm_max']:.1e}"
            f"{' (above target)' if r['final_grad_above_recipe'] else ''} | "
            f"{_fmt(w['median'])} "
            f"[{_fmt(w['min'])}, {_fmt(w['max'])}] | "
            f"{_fmt(None if rss is None else rss / 2**20, 0)} |")
    lines += ["", "Median exclusive phase seconds (component timing on):", "",
              "| Case | Threads | " + " | ".join(_MD_GROUPS) + " |",
              "|---|---|" + "---|" * len(_MD_GROUPS)]
    for r in rows:
        g = r["phase_exclusive_s"]
        cells = [_fmt((g.get(k) or {}).get("median")) if k in g else "-"
                 for k in _MD_GROUPS]
        lines.append(f"| {r['case_id']} | {r['threads_requested']} | "
                     + " | ".join(cells) + " |")
    if summary["timing_on_off"]:
        lines += ["", "Instrumentation on versus off:", "",
                  "| Case | Threads | dE (Eh) | Same iterations | "
                  "Median on (s) | Median off (s) | Overhead | Ranges overlap |",
                  "|---|---|---|---|---|---|---|---|"]
        for c in summary["timing_on_off"]:
            lines.append(
                f"| {c['case_id']} | {c['threads_requested']} | "
                f"{c['energy_delta_eh']:.1e} | {c['iterations_equal']} | "
                f"{_fmt(c['median_wall_on_s'])} | {_fmt(c['median_wall_off_s'])} | "
                f"{100 * c['overhead_fraction']:+.1f}% | {c['ranges_overlap']} |")
    return "\n".join(lines) + "\n"


def cmd_summarize(args: argparse.Namespace) -> int:
    records = load_records(Path(args.raw_dir))
    summary = summarize(records, expected_source_sha=args.expected_source_sha,
                        expected_native_sha256=args.expected_native_sha256)
    text = json.dumps(summary, indent=1, sort_keys=True)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if args.md_out:
        Path(args.md_out).write_text(render_markdown(summary), encoding="utf-8")
    return 0


def cmd_manifest(args: argparse.Namespace) -> int:
    cases = build_manifest()
    print(json.dumps({"basis": BASIS, "systems": SYSTEMS, "methods": METHODS,
                      "routes": ROUTES, "recipe": RECIPE, "protocol": PROTOCOL,
                      "cases": [{k: v for k, v in c.items() if k != "atoms_bohr"}
                                for c in cases]}, indent=1))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("manifest").set_defaults(fn=cmd_manifest)

    rc = sub.add_parser("run-case")
    rc.add_argument("case_json")
    rc.add_argument("--out-dir", required=True)
    rc.add_argument("--threads", type=int, default=1)
    rc.add_argument("--label", default="adhoc")
    rc.add_argument("--component-timing", action="store_true")
    rc.set_defaults(fn=cmd_run_case)

    rb = sub.add_parser("run-batch")
    rb.add_argument("batch_json")
    rb.add_argument("--out-dir", required=True)
    rb.set_defaults(fn=cmd_run_batch)

    sb = sub.add_parser("submit")
    sb.add_argument("--python", required=True,
                    help="interpreter of the runtime to measure, on the queue host")
    sb.add_argument("--host", default=None)
    sb.add_argument("--output-root", default=None)
    sb.add_argument("--run-id", default=None)
    sb.add_argument("--systems", default=None)
    sb.add_argument("--methods", default=None)
    sb.add_argument("--routes", default=None)
    sb.add_argument("--threads", default="1")
    sb.add_argument("--blas-threads", type=int, default=1)
    sb.add_argument("--warmup", type=int, default=PROTOCOL["warmup_runs"])
    sb.add_argument("--repeats", type=int, default=PROTOCOL["timed_runs"])
    sb.add_argument("--seed", type=int, default=None)
    sb.add_argument("--per-run-budget-s", type=int,
                    default=PROTOCOL["per_run_budget_s"])
    sb.add_argument("--compare-timing-off", action="store_true")
    sb.add_argument("--no-dedupe", action="store_true",
                    help="run AUTO even when it resolves to the direct settings")
    sb.set_defaults(fn=cmd_submit)

    sm = sub.add_parser("summarize")
    sm.add_argument("raw_dir")
    sm.add_argument("--json-out", default=None)
    sm.add_argument("--md-out", default=None)
    sm.add_argument("--expected-source-sha", default=None)
    sm.add_argument("--expected-native-sha256", default=None)
    sm.set_defaults(fn=cmd_summarize)

    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
