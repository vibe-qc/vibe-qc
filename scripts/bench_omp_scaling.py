#!/usr/bin/env python3
"""OpenMP / BLAS-thread scaling harness for the per-method cores-per-job map.

Inner runner: invoked as a subprocess with OMP_NUM_THREADS and
VECLIB_MAXIMUM_THREADS / OPENBLAS_NUM_THREADS already set in the environment
(both are read at process start, so each (threads) point must be its own
process). Builds a representative system, runs the method via run_job (the
queue-facing path, file writes off), and prints `TIME=<seconds>` for the
compute call.

Usage (single point):
    OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=1 \
        python scripts/bench_omp_scaling.py <method> <system>

The orchestrator (bench_omp_scaling_sweep.sh) loops the (method, system,
omp, blas) grid and collects the times.
"""
from __future__ import annotations

import sys
import time

import vibeqc as vq
from vibeqc import Atom, Molecule

# --- Representative geometries (Angstrom) ---------------------------------
GLYCINE = [
    (6, [-2.14912117, 0.61490945, -0.04623979]),
    (6, [-0.69532565, 1.08777284, 0.46529739]),
    (8, [2.07136293, -0.20398383, -0.40288843]),
    (8, [-0.47627582, 0.75493601, 1.83890424]),
    (7, [0.25131369, 0.36737660, -0.41062234]),
    (1, [-2.42040617, 1.13995698, -0.95870224]),
    (1, [-2.26895158, -0.45332391, -0.11091677]),
    (1, [-0.61604639, 2.17228635, 0.34163710]),
    (1, [1.35683503, 1.20732242, -0.25452337]),
    (1, [0.90108897, -0.53527126, -0.96850826]),
]
WATER_TRIMER = [
    (8, [1.675, 0.0, 0.0]), (1, [0.844, 0.48, 0.0]), (1, [1.883, -0.12, 0.929]),
    (8, [-0.838, 1.45, 0.0]), (1, [-0.838, 0.49, 0.0]), (1, [-0.838, 1.69, 0.929]),
    (8, [-0.838, -1.45, 0.0]), (1, [-0.007, -0.97, 0.0]), (1, [-1.046, -1.57, 0.929]),
]
# Benzene, D6h, r(CC)=1.39, r(CH)=1.09
import math
_BENZ = []
for k in range(6):
    a = math.radians(60 * k)
    _BENZ.append((6, [1.39 * math.cos(a), 1.39 * math.sin(a), 0.0]))
    _BENZ.append((1, [2.48 * math.cos(a), 2.48 * math.sin(a), 0.0]))
BENZENE = _BENZ
OH = [(8, [0.0, 0.0, 0.0]), (1, [0.9699818276, 0.0, 0.0])]
WATER = [(8, [0.0, 0.0, 0.117]), (1, [0.0, 0.757, -0.470]), (1, [0.0, -0.757, -0.470])]
CH3 = [(6, [0.0, 0.0, 0.0]), (1, [1.079, 0.0, 0.0]),
       (1, [-0.5395, 0.9345, 0.0]), (1, [-0.5395, -0.9345, 0.0])]

# system -> (atoms, basis, charge, multiplicity)
SYSTEMS = {
    "glycine":      (GLYCINE,      "def2-svp", 0, 1),   # ~115 bf, medium DFT
    "glycine_tz":   (GLYCINE,      "def2-tzvp", 0, 1),  # ~250 bf, large DFT
    "h2o3":         (WATER_TRIMER, "cc-pvdz",  0, 1),   # ~75 bf, small CC
    "benzene":      (BENZENE,      "cc-pvdz",  0, 1),   # ~114 bf, medium CC
    "oh":           (OH,           "cc-pvdz",  0, 2),   # open-shell, small UCC
    "ch3":          (CH3,          "cc-pvdz",  0, 2),   # open-shell, UCC
    "cas_h2o":      (WATER,        "cc-pvdz",  0, 1),   # CASSCF/CASPT2 (8,8)
}

# methods needing an active space: (n_elec, n_orb)
ACTIVE_SPACE = {"casscf": (8, 8), "caspt2": (6, 6)}

# method -> run_job kwargs
METHODS = {
    "rhf":     dict(method="rhf"),
    "rks":     dict(method="rks", functional="PBE"),
    "mp2":     dict(method="mp2"),
    "ccsd":    dict(method="ccsd"),
    "ccsd_t":  dict(method="ccsd(t)"),
    "uhf":     dict(method="uhf"),
    "uccsd":   dict(method="ccsd"),       # open-shell mol -> UCCSD via ref
    "tddft":   dict(method="rks", functional="PBE", tddft_type="tda"),
    "casscf":  dict(method="casscf"),
    "caspt2":  dict(method="caspt2"),
}


def main() -> int:
    method, system = sys.argv[1], sys.argv[2]
    atoms, basis, charge, mult = SYSTEMS[system]
    mol = Molecule([Atom(z, list(xyz)) for z, xyz in atoms],
                   charge=charge, multiplicity=mult)
    kwargs = dict(METHODS[method])
    if method in ACTIVE_SPACE:
        kwargs["active_space"] = ACTIVE_SPACE[method]
    import tempfile, os
    out = os.path.join(tempfile.gettempdir(), f"bench_{method}_{system}")
    kwargs.update(
        basis=basis,
        output=out,
        write_molden_file=False,
        write_xyz_file=False,
        write_population_file=False,
        verbose=False,
    )
    t0 = time.perf_counter()
    try:
        vq.run_job(mol, **kwargs)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR={type(e).__name__}: {e}")
        return 1
    print(f"TIME={time.perf_counter() - t0:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
