#!/usr/bin/env python3
"""Experimental periodic BeH doublet with real SKALA-1.1 and BIPOLE.

This small asymmetric cell exercises unequal spin populations at Gamma or a
complete Gamma-centred k mesh. It is not a model of bulk BeH or an accuracy
benchmark. Validate Gamma before multi-k and converge the lattice cutoff.
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="plan without loading PyTorch or the checkpoint")
    parser.add_argument("--output", type=Path,
                        help="output stem with an existing parent directory")
    parser.add_argument("--kpoints", type=int, nargs=3, default=[1, 1, 1],
                        metavar=("NX", "NY", "NZ"),
                        help="complete Gamma-centred mesh (default: 1 1 1)")
    parser.add_argument("--bipole-cutoff-bohr", type=float, required=True,
                        help="positive finite lattice cutoff; converge explicitly")
    args = parser.parse_args()
    if any(n <= 0 for n in args.kpoints):
        parser.error("--kpoints requires three positive integers")
    if not math.isfinite(args.bipole_cutoff_bohr) or args.bipole_cutoff_bohr <= 0:
        parser.error("--bipole-cutoff-bohr must be positive and finite")
    return args


def require_supported_runtime() -> None:
    if not ((3, 11) <= sys.version_info[:2] <= (3, 13)):
        raise SystemExit(
            "SKALA requires Python 3.11 through 3.13; this interpreter is "
            f"{platform.python_version()}."
        )
    if sys.platform != "linux":
        raise SystemExit("Real SKALA evaluation requires Linux.")


def reserve_output_stem(requested: Path | None, mesh: list[int]) -> Path:
    if requested is not None:
        stem = requested.expanduser().resolve()
        if not stem.parent.is_dir():
            raise FileNotFoundError(f"output parent does not exist: {stem.parent}")
        return stem
    label = "x".join(str(n) for n in mesh)
    root = Path(os.environ.get("VQ_WORKDIR", ".")).resolve()
    destination = root / f"beh-uks-bipole-k{label}-results"
    destination.mkdir(exist_ok=False)
    return destination / "beh-uks"


def spin_occupation_counts(result) -> tuple[float, float]:
    """Check occupation bookkeeping, independently of grid-density accuracy."""
    import numpy as np

    weights = np.asarray(result.kpoint_weights, dtype=float)
    alpha = np.asarray(result.occupations_alpha, dtype=float)
    beta = np.asarray(result.occupations_beta, dtype=float)
    if (weights.ndim != 1 or weights.size == 0 or alpha.ndim != 2
            or beta.shape != alpha.shape or alpha.shape[0] != weights.size
            or alpha.shape[1] == 0):
        raise RuntimeError("Missing or incompatible per-k spin occupations")
    if (not all(np.isfinite(x).all() for x in (weights, alpha, beta))
            or np.any(weights < 0) or np.any(alpha < 0) or np.any(alpha > 1)
            or np.any(beta < 0) or np.any(beta > 1)
            or abs(float(weights.sum()) - 1.0) > 1.0e-8):
        raise RuntimeError("Invalid k weights or spin occupations")
    counts = float(weights @ alpha.sum(axis=1)), float(weights @ beta.sum(axis=1))
    if abs(counts[0] - 3.0) > 1.0e-8 or abs(counts[1] - 2.0) > 1.0e-8:
        raise RuntimeError(f"BeH doublet requires alpha/beta occupations 3/2, got {counts}")
    return counts


def main() -> None:
    args = parse_args()
    if not args.dry_run:
        require_supported_runtime()

    import numpy as np
    import vibeqc as vq

    output = reserve_output_stem(args.output, args.kpoints)
    # Same 12-bohr cell and 2.5-bohr Be-H separation as the asymmetric
    # doublet in tests/test_bipole_gradient.py. This campaign has no accepted
    # SKALA reference for this fixture; it is a route and spin-population check.
    system = vq.PeriodicSystem(
        dim=3, lattice=np.eye(3) * 12.0,
        unit_cell=[vq.Atom(4, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 2.5])],
        multiplicity=2,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    result = vq.run_periodic_job(
        system, basis, method="UKS", functional="skala-1.1", jk_method="bipole",
        kpoints=args.kpoints, symmetry=False, use_exchange_ewald_split=True,
        smearing_temperature=0.0,
        bipole_cutoff_bohr=args.bipole_cutoff_bohr,
        bipole_nuclear_cutoff_bohr=args.bipole_cutoff_bohr,
        max_iter=128, conv_tol_energy=1.0e-8,
        output=output, dry_run=args.dry_run,
        write_molden_file=True, write_population_file=False, write_density=False,
        output_qvf=True, progress=False, record_hostname=False,
    )
    print(f"periodic route: BIPOLE UKS, Gamma-centred mesh {args.kpoints}")
    if args.dry_run:
        print(f"dry-run complete: {output}.system")
        return
    if result is None or not result.converged or not math.isfinite(result.energy):
        raise RuntimeError("BeH SKALA did not converge to a finite energy")
    alpha, beta = spin_occupation_counts(result)
    print(f"output stem: {output}")
    print(f"E(SKALA-1.1)/cell = {result.energy:.12f} hartree")
    print(f"k-weighted occupations: alpha={alpha:.8f}, beta={beta:.8f}")
    print("periodic SKALA status: experimental; occupation checks are not accuracy validation")


if __name__ == "__main__":
    main()
