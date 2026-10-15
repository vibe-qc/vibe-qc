#!/usr/bin/env python3
"""Bounded native channel-sharing benchmark; JSON goes to stdout.

Run with the checkout's rebuilt editable Python. Every variant runs in a fresh
single-thread process. RSS includes imports, fixture and independent reference
checks; it is process peak memory, not an integral-workspace measurement.
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


def worker(case: str, variant: str, repeats: int) -> dict:
    import numpy as np
    import vibeqc as vq
    from vibeqc import manta
    from vibeqc import _vibeqc_core as core

    root = Path(__file__).resolve().parents[1]
    native = Path(core.__file__).resolve()
    if not native.is_relative_to(root):
        raise RuntimeError("Use an editable native build from this checkout")
    newest = max(p.stat().st_mtime for p in (root / "cpp").rglob("*") if p.suffix in (".cpp", ".hpp", ".h"))
    if native.stat().st_mtime <= newest:
        raise RuntimeError("Rebuild the native extension before benchmarking")

    mol = vq.Molecule([vq.Atom(2, [.2, -.3, .4])], 0, 1)
    shells = []
    for i in range(3):
        shell = core.ShellInfo()
        shell.atom_index, shell.l, shell.pure, shell.origin = 0, 2, True, [.2, -.3, .4]
        shell.exponents = ([.6, 1.3] if case == "general" else
                           [[.6, 1.3], [1.3, 2.1], [.6]][i] if case == "partial" else
                           [.6+.2*i, 1.3+.2*i])
        shell.coefficients = [[.8, -.2], [.3, .9], [-.4, .6]][i][:len(shell.exponents)]
        shells.append(shell)
    basis = vq.BasisSet(mol, shells, "manta-channel-benchmark", False)
    rng = np.random.default_rng(5129)
    densities = []
    for _ in range(2):
        d = rng.normal(size=(basis.nbasis, basis.nbasis)); d += d.T
        densities.append(d)

    start = time.perf_counter()
    if variant == "libint-direct":
        builder = core.make_direct_jk_builder(basis, 0.)
    elif variant == "libint-ordered":
        builder = manta.ReferenceJKBuilder(basis)
    else:
        builder = manta.DirectJKBuilder(basis, channel_batch=16 if variant == "manta-shared" else 1)
    setup = time.perf_counter()-start

    def build():
        if variant == "libint-direct":
            return [builder.build_J(d) for d in densities], [builder.build_K(d) for d in densities], None
        result = builder.build_batch(densities)
        return result.J, result.K, result.receipt

    build()  # One complete warm-up.
    elapsed, cpu = [], []
    for _ in range(repeats):
        start, process = time.perf_counter(), time.process_time()
        j, k, receipt = build()
        cpu.append(time.process_time()-process); elapsed.append(time.perf_counter()-start)
    # Separate same-source comparison with the existing optimized direct path.
    reference = core.make_direct_jk_builder(basis, 0.)
    error = max(float(np.max(np.abs(actual-expected)))
                for d, actual_j, actual_k in zip(densities, j, k, strict=True)
                for actual, expected in [(actual_j, reference.build_J(d)), (actual_k, reference.build_K(d))])
    if error > 3e-11:
        raise RuntimeError(f"Numerical benchmark gate failed: {error}")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() != "Darwin":
        rss *= 1024
    return {
        "case": case, "variant": variant, "n_basis": basis.nbasis,
        "n_shells": basis.nshells, "n_densities": len(densities), "repeats": repeats,
        "setup_wall_s": setup, "build_median_wall_s": statistics.median(elapsed),
        "build_min_wall_s": min(elapsed), "build_max_wall_s": max(elapsed),
        "build_median_cpu_s": statistics.median(cpu), "peak_process_rss_bytes": rss,
        "max_abs_jk_error": error,
        "actual_shell_calls": None if receipt is None else receipt.shell_calls,
        "requested_primitive_quartets": None if receipt is None else receipt.primitive_quartets_requested,
        "evaluated_primitive_quartets": None if receipt is None or not receipt.primitive_evaluations_measured else receipt.primitive_quartets_evaluated,
        "density_terms_accumulated": None if receipt is None else receipt.density_terms_accumulated,
        "modeled_peak_bytes": None if receipt is None else receipt.modeled_peak_bytes,
        "core_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
        "platform": platform.system(), "architecture": platform.machine(),
        "python": platform.python_version(), "numpy": np.__version__,
        "threads": 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--worker", choices=["libint-direct", "libint-ordered", "manta-segmented", "manta-shared"])
    parser.add_argument("--case", choices=["general", "partial", "disjoint"], default="general")
    args = parser.parse_args()
    if not 3 <= args.repeats <= 100:
        parser.error("repeats must be between 3 and 100")
    if args.worker:
        for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"]:
            os.environ[name] = "1"
        print(json.dumps(worker(args.case, args.worker, args.repeats)))
        return
    environment = os.environ.copy()
    for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        environment[name] = "1"
    rows = []
    for case in ["general", "partial", "disjoint"]:
        for variant in ["libint-direct", "libint-ordered", "manta-segmented", "manta-shared"]:
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", variant,
                       "--case", case, "--repeats", str(args.repeats)]
            completed = subprocess.run(command, env=environment, check=True, text=True, capture_output=True)
            rows.append(json.loads(completed.stdout))
    print(json.dumps({"schema": "manta-channel-benchmark-v1", "results": rows}, indent=2))


if __name__ == "__main__":
    main()
