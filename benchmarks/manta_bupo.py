#!/usr/bin/env python3
"""Small molecular Coulomb construction benchmark; portable JSON on stdout.

Each variant runs in a fresh single-thread process with a bounded timeout.
Basis preparation and reference validation are outside construction timings.
This benchmark does not run SCF. Process peak
RSS includes imports, the fixture and validation; it is not workspace RSS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time

VARIANTS = ("libint-direct", "libint-ordered", "manta-ordered", "manta-near",
            "manta-bupo", "libint-near-bupo")
CASES = ("water", "water-dimer", "separated-water-dimer", "diffuse-dimer")
TIMINGS = ("validation_wall_seconds", "leaf_moments_wall_seconds", "upward_wall_seconds",
           "far_admission_wall_seconds", "far_apply_wall_seconds", "near_integrals_wall_seconds",
           "near_digest_wall_seconds", "downward_wall_seconds", "reverse_moments_wall_seconds",
           "finalize_wall_seconds")


def worker(case, variant, repeats):
    import numpy as np
    import vibeqc as vq
    from vibeqc import manta
    from vibeqc import _vibeqc_core as core

    root = Path(__file__).resolve().parents[1]
    native = Path(core.__file__).resolve()
    if not native.is_relative_to(root):
        raise RuntimeError("Use this checkout's rebuilt editable native extension")
    newest = max(p.stat().st_mtime for p in (root/"cpp").rglob("*") if p.suffix in (".cpp", ".hpp", ".h"))
    if native.stat().st_mtime <= newest:
        raise RuntimeError("Rebuild the native extension before benchmarking")
    positions = np.array([[0., 0., 0.], [1.43, 0., 1.11], [-1.43, 0., 1.11]])
    charges = [8, 1, 1]
    if case != "water":
        distance = 60. if case == "separated-water-dimer" else 6.
        positions = np.vstack([positions, positions+np.array([distance, .3, -.4])])
        charges *= 2
    molecule = vq.Molecule([vq.Atom(z, p.tolist()) for z, p in zip(charges, positions, strict=True)], 0, 1)
    basis = vq.BasisSet(molecule, "sto-3g")
    if case == "diffuse-dimer":
        shells = basis.shells()
        for atom in (0, 3):
            s = core.ShellInfo(); s.atom_index = atom; s.l = 0; s.pure = True
            s.origin = positions[atom].tolist(); s.exponents = [.025]; s.coefficients = [(2*.025/np.pi)**.75]
            shells.append(s)
        basis = vq.BasisSet(molecule, shells, "sto-3g-plus-diffuse-s", True)
    # A reproducible PSD trial density normalized to the molecular electron
    # count in S, plus a signed response density. Neither requires a particular
    # SCF basin, and both timed results receive independent accuracy checks.
    overlap = vq.compute_overlap(basis)
    rng = np.random.default_rng(5612)
    x = rng.normal(size=(basis.nbasis, basis.nbasis))
    density = x@x.T
    density *= molecule.n_electrons()/np.sum(density*overlap)
    delta = rng.normal(size=density.shape); delta = .01*(delta+delta.T)
    densities = (density, delta)
    options = manta.BupoOptions(); options.order = 12; options.opening = .8
    options.integral_tolerance = 1e-11; options.bubbles.leaf_capacity = 4
    options.collect_timings = True
    if variant == "manta-near":
        options.opening = 0.
    options.near_backend = "libint" if variant == "libint-near-bupo" else "manta"
    budget = manta.ResourceBudget(); budget.memory_bytes = 512*1024**2
    budget.other_live_bytes = 128*basis.nbasis**2
    start, process = time.perf_counter(), time.process_time()
    if variant == "libint-direct":
        builder = core.make_direct_jk_builder(basis, 0.)
    elif variant == "libint-ordered":
        builder = manta.ReferenceJKBuilder(basis, budget=budget)
    elif variant == "manta-ordered":
        builder = manta.DirectJKBuilder(basis, budget=budget)
    else:
        builder = manta.BupoJBuilder(basis, options, budget)
    setup_wall, setup_cpu = time.perf_counter()-start, time.process_time()-process
    is_bupo = variant in ("manta-near", "manta-bupo", "libint-near-bupo")
    is_ordered = variant in ("libint-ordered", "manta-ordered")

    def build():
        if is_bupo:
            results = [builder.build(d) for d in densities]
            return [r.J for r in results], [r.receipt for r in results]
        if is_ordered:
            results = [builder.build_batch([d], make_j=True, make_k=False) for d in densities]
            return [r.J[0] for r in results], [r.receipt for r in results]
        return [builder.build_J(d) for d in densities], []

    build()
    elapsed, cpu, component_samples, native_elapsed = [], [], [], []
    for _ in range(repeats):
        start, process = time.perf_counter(), time.process_time()
        matrices, receipts = build()
        elapsed.append(time.perf_counter()-start); cpu.append(time.process_time()-process)
        if is_bupo:
            native_elapsed.append(sum(r.wall_seconds for r in receipts))
            component_samples.append({name: sum(getattr(r, name) for r in receipts) for name in TIMINGS})
    rss_before_reference = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    reference = core.make_direct_jk_builder(basis, 0.)
    reference_matrices = [reference.build_J(d) for d in densities]
    j_errors = [float(np.max(abs(j-ref))) for j, ref in zip(matrices, reference_matrices, strict=True)]
    energy_errors = [float(abs(.5*np.sum(d*(j-ref)))) for d, j, ref in zip(densities, matrices, reference_matrices, strict=True)]
    if max(j_errors) > 1e-8 or max(energy_errors) > 1e-8:
        raise RuntimeError(f"Accuracy gate failed: J={max(j_errors)}, EJ={max(energy_errors)}")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_scale = 1 if platform.system() == "Darwin" else 1024
    def total(name):
        return sum(getattr(r, name) for r in receipts)
    work = None
    if is_bupo:
        work = {name: total(name) for name in ("near_shell_calls", "near_primitive_quartets", "near_null_tiles", "far_blocks", "moment_panels", "moment_primitive_pairs", "traversal_visits")}
        work["near_primitive_evaluations_measured"] = receipts[0].near_primitive_evaluations_measured
    elif is_ordered:
        work = {name: total(name) for name in ("shell_calls", "primitive_quartets_requested", "density_terms_accumulated")}
        work["primitive_quartets_evaluated"] = total("primitive_quartets_evaluated") if receipts[0].primitive_evaluations_measured else None
    return {
        "case": case, "variant": variant, "n_basis": basis.nbasis, "n_shells": basis.nshells,
        "n_densities": len(densities), "repeats": repeats, "threads": 1,
        "setup_wall_s": setup_wall, "setup_cpu_s": setup_cpu,
        "build_median_wall_s": statistics.median(elapsed), "build_min_wall_s": min(elapsed),
        "build_max_wall_s": max(elapsed), "build_median_cpu_s": statistics.median(cpu),
        "cold_setup_plus_density_pair_wall_s": setup_wall+statistics.median(elapsed),
        "component_median_wall_s": {name: statistics.median(s[name] for s in component_samples) for name in TIMINGS} if is_bupo else None,
        "native_build_median_wall_s": statistics.median(native_elapsed) if is_bupo else None,
        "setup_component_wall_s": {name: getattr(builder.plan, name) for name in ("provider_setup_wall_seconds", "extent_setup_wall_seconds", "tree_setup_wall_seconds", "error_model_setup_wall_seconds", "partition_setup_wall_seconds")} if is_bupo else None,
        "peak_process_rss_bytes": rss*rss_scale,
        "peak_before_reference_rss_bytes": rss_before_reference*rss_scale,
        "modeled_peak_bytes": receipts[0].modeled_peak_bytes if receipts else None,
        "actual_work": work, "j_errors": j_errors, "coulomb_energy_errors": energy_errors,
        "analytic_j_bounds": [r.approximation_j_bound for r in receipts] if is_bupo else None,
        "partition_identity": receipts[0].partition_identity if is_bupo else None,
        "source_identity": receipts[0].source_identity if receipts else None,
        "bupo_controls": {"order": options.order, "opening": options.opening, "integral_tolerance": options.integral_tolerance, "leaf_capacity": options.bubbles.leaf_capacity} if is_bupo else None,
        "core_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
        "platform": platform.system(), "architecture": platform.machine(),
        "python": platform.python_version(), "numpy": np.__version__,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker", choices=VARIANTS)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--timeout", type=float, default=300.)
    args = parser.parse_args()
    if not 3 <= args.repeats <= 100 or not 1 <= args.timeout <= 3600:
        parser.error("Use 3..100 repeats and 1..3600 seconds timeout per variant")
    environment = os.environ.copy()
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
        environment[name] = "1"
        os.environ[name] = "1"
    if args.worker:
        if args.case is None:
            parser.error("--worker requires --case")
        print(json.dumps(worker(args.case, args.worker, args.repeats)))
        return
    rows = []
    for case in (args.case,) if args.case else CASES:
        for variant in VARIANTS:
            print(f"{case}: {variant}", file=sys.stderr, flush=True)
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", variant,
                       "--case", case, "--repeats", str(args.repeats)]
            completed = subprocess.run(command, env=environment, check=True, text=True,
                                       capture_output=True, timeout=args.timeout)
            rows.append(json.loads(completed.stdout))
    for case in (args.case,) if args.case else CASES:
        compared = [r for r in rows if r["case"] == case and r["variant"] in ("manta-bupo", "libint-near-bupo")]
        if len({r["partition_identity"] for r in compared}) != 1:
            raise RuntimeError("Near-provider comparison changed the BUPO partition")
    print(json.dumps({"schema": "manta-bupo-benchmark-v1", "results": rows}, indent=2))


if __name__ == "__main__":
    main()
