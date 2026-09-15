#!/usr/bin/env python
"""Emit the benchmark job batch for AICCM-2026 (driven by ``testset.py``).

Generates the per-(system, route) commands for the whole test set, with a
tractable route plan per tier and the target machine for each. Review, then pipe
to a shell.

    python make_jobs.py                 # vq submit batch, grouped by machine
    python make_jobs.py --local         # plain `python run_case.py ...` (tier A)
    python make_jobs.py --tier A        # filter by tier
    python make_jobs.py --system mgo    # filter to one system
    python make_jobs.py --crystal-basis # periodic refs (gdf/bipole) at pob-tzvp-rev2
    python make_jobs.py | sh            # (review first) launch on vq

Route plan per tier (cost-aware; the four-center rebuilds each SCF iteration, the
post-HF correlation is O(N^5/N^6), and the multi-k GDF/bipole references
dominate):

    A (1-D / tiny):  aiccm-hf aiccm-ks aiccm-ri aiccm-ks-ri aiccm-rijcosx
                     aiccm-mp2 aiccm-ccsd aiccm-viz aiccm-localize aiccm-symmetry
                     aiccm-pao gdf bipole                       (full feature matrix + demos)
    B (3-D 2-atom):  aiccm-hf aiccm-ks aiccm-ri aiccm-ks-ri aiccm-rijcosx aiccm-mp2
                     aiccm-viz aiccm-localize aiccm-symmetry aiccm-pao gdf bipole
    C (oxides/AFM):  aiccm-ri aiccm-ks-ri aiccm-symmetry gdf bipole

The aiccm-localize / aiccm-symmetry / aiccm-pao / aiccm-dlpno-mp2 /
aiccm-dlpno-ccsd / aiccm-properties routes are demonstration cases: each runs a
built feature (Wannier localization / space-group symmetry / DLPNO PAO /
DLPNO-MP2 / DLPNO-CCSD(T) / gap+populations+dipole+forces, the latter with
vibe-view .qvf output) on the system and writes its metrics to JSON — full inputs
that show the feature works on real test-set crystals. The DLPNO correlation
routes are tier A (1-D) because the neutral cderi is a small-cluster build;
aiccm-properties (scalable SCF) runs on every tier.

The default runs everything at STO-3G (same-basis AICCM-vs-periodic comparison).
``--crystal-basis`` reruns the periodic refs at pob-tzvp-rev2 (vs CRYSTAL23) into
a separate results dir.
"""

from __future__ import annotations

import site_settings

import argparse

import testset

# Default targets come from an explicitly selected private site profile.
TIER_MACHINE = site_settings.host_map("tier_hosts")
# The DLPNO / UCCSD correlation routes reach 3-D via the neutral **fold** cderi
# (ccm_neutral_cderi_fold — per-q unit-cell fits, no supercell-FFT wall), so they
# now run on 3-D bulk too (a configured compute host). aiccm-properties uses the scalable SCF
# (gap / populations / dipole + vibe-view .qvf), so it runs on every tier.
# `-cosx` = neutral fitted-torus RIJCOSX control (GDF RI-J + COSX-K); COSX
# needs a 3-D multi-k grid, so those routes are emitted only for dim==3
# systems (gated in the loop).
TIER_ROUTES = {
    "A": ["aiccm-hf", "aiccm-hf-direct", "aiccm-ks", "aiccm-ri", "aiccm-ks-ri",
          "aiccm-ri-cosx", "aiccm-ks-ri-cosx", "aiccm-rijcosx",
          "aiccm-mp2", "aiccm-ccsd", "aiccm-dlpno-mp2", "aiccm-dlpno-ccsd",
          "aiccm-properties", "aiccm-viz",
          "aiccm-localize", "aiccm-symmetry", "aiccm-pao", "gdf", "bipole"],
    "B": ["aiccm-hf", "aiccm-hf-direct", "aiccm-ks", "aiccm-ri", "aiccm-ks-ri",
          "aiccm-ri-cosx", "aiccm-ks-ri-cosx",
          "aiccm-mp2", "aiccm-dlpno-ccsd", "aiccm-properties", "aiccm-viz",
          "aiccm-localize", "aiccm-symmetry", "aiccm-pao", "gdf", "bipole"],
    "C": ["aiccm-ri", "aiccm-ks-ri", "aiccm-ri-cosx", "aiccm-ks-ri-cosx",
          "aiccm-properties", "aiccm-symmetry", "gdf", "bipole"],
}
_COSX_ROUTES = {"aiccm-ri-cosx", "aiccm-ks-ri-cosx"}   # dim==3 only (COSX grid)
CRYSTAL_BASIS = "pob-tzvp-rev2"
PERIODIC_ROUTES = {"gdf", "bipole"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true",
                    help="emit plain `python run_case.py` for tier A (no vq)")
    ap.add_argument("--tier", default=None, choices=["A", "B", "C"])
    ap.add_argument("--system", default=None, choices=sorted(testset.SYSTEMS))
    ap.add_argument("--crystal-basis", action="store_true",
                    help="periodic refs at pob-tzvp-rev2 (CRYSTAL23 comparison)")
    ap.add_argument("--machine", default=None,
                    help="override the tier→machine map (e.g. compute-large) for the whole batch")
    ap.add_argument("--functional", default=None,
                    help="functional for the KS routes (e.g. pbe0 for hybrids); "
                         "tags the route dir so pure/hybrid don't collide")
    ap.add_argument("--dim", type=int, default=None, choices=[1, 2, 3],
                    help="only emit jobs for systems of this periodic dimension")
    args = ap.parse_args()

    # A vq batch must never land on this box: the 3-D crystals are memory-heavy and
    # the shared checkout has a 128 GiB ceiling. `--local` is the deliberate opt-out
    # and is tier-A only (1-D, seconds).
    if not args.local and site_settings.is_local_host(args.machine):
        ap.error(f"--machine {args.machine!r} would run the batch on this laptop; "
                 "pass a configured remote host or use --local for tier A")

    systems = [args.system] if args.system else list(testset.SYSTEMS)
    ks_routes = {"aiccm-ks", "aiccm-ks-ri", "aiccm-ks-ri-cosx"}
    by_machine = {}
    n = 0
    for name in systems:
        meta = testset.SYSTEMS[name]
        tier = meta["tier"]
        if args.tier and tier != args.tier:
            continue
        if args.dim and int(meta["dim"]) != args.dim:
            continue
        if args.local and tier != "A":
            continue
        machine = ("local" if args.local else site_settings.explicit_remote_host(args.machine) if args.machine else TIER_MACHINE[tier])
        for route in TIER_ROUTES[tier]:
            if route in _COSX_ROUTES and int(meta["dim"]) != 3:
                continue  # COSX needs the 3-D multi-k grid
            if route in testset.NEEDS_3D_COULOMB and int(meta["dim"]) != 3:
                # dim<3 fails closed in vibeqc (transverse-collapsed reciprocal
                # mesh); don't burn a cluster slot to record that. run_case.py
                # still emits an "unavailable" row if invoked directly.
                continue
            func_opt = ""
            if args.functional and route in ks_routes:
                func_opt = f" --functional {args.functional}"
            basis_opt = ""
            outdir = "results"
            if args.crystal_basis and route in PERIODIC_ROUTES:
                basis_opt = f" --basis {CRYSTAL_BASIS}"
                outdir = f"results-{CRYSTAL_BASIS}"
            elif args.crystal_basis:
                continue  # crystal-basis pass only reruns the periodic refs
            if args.local:
                line = f"python run_case.py {name} {route}{basis_opt}{func_opt} --out {outdir}/"
            else:
                # -d dir submit: encode the vibeqc-dev venv via run.sh (the
                # `--branch main` target). --branch is single-file-only, and the
                # host default python lacks vibeqc.
                line = f"vq submit {machine}  -d studies/aiccm-2026/ -- bash run.sh {name} {route}{basis_opt}{func_opt}"
            by_machine.setdefault(machine, []).append(line)
            n += 1

    if args.local:
        print(f"# {n} local jobs (tier A). Review, then: python make_jobs.py --local | sh")
        for m in by_machine:
            for line in by_machine[m]:
                print(line)
    else:
        print(f"# AICCM-2026 benchmark: {n} jobs. Review, then pipe to sh.")
        print("# Each writes <system>__<route>.json to its $VQ_WORKDIR; fetch + aggregate")
        print("# with: python compare.py <results-dir>  (single-line diagnostics only).")
        print("# D89 disables route-name-based cross-approach --vs deltas.")
        for m in sorted(by_machine):
            print(f"\n# ---- {m} ({len(by_machine[m])} jobs) ----")
            for line in by_machine[m]:
                print(line)


if __name__ == "__main__":
    main()
