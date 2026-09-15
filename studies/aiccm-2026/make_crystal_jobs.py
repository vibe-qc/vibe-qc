#!/usr/bin/env python
"""Emit the CRYSTAL23 *reference-value* job batch for the AICCM-2026 test set.

Generates, per system, an external CRYSTAL23 single-point at the test-set geometry
(the value the AICCM is compared against in the paper). CRYSTAL23 is an external
program — run **out of process** via vq (CLAUDE.md §10): we never import it; we
spawn it on its own `.d12` and parse the output (`crun.sh`).

For each system this:
  1. resolves the system's reference `.d12` from ``testset.SYSTEMS[name]["crystal_ref"]``
     under the QC input library (``--library``, default ``~/gitlab/qc-input-library``),
  2. stages a self-contained payload ``crystal23/jobs/<system>/`` = that INPUT.d12
     + crun.sh (the staging dir is gitignored — built on demand, not committed),
  3. emits the vq line ``vq submit <host> -d <payload>/ -- bash crun.sh INPUT.d12 <system>``.

    python make_crystal_jobs.py                 # stage payloads + print vq lines
    python make_crystal_jobs.py --host reference-small   # pin the host (default: per-tier)
    python make_crystal_jobs.py --system mgo    # one system
    python make_crystal_jobs.py | sh            # (review first) launch on vq

Each job writes ``<system>__crystal23.json`` (energy_ha, converged) to its
``$VQ_WORKDIR``; ``vq fetch`` + feed into ``compare.py`` alongside the AICCM runs.

The library `.d12` levels differ per system (PBE / B3LYP / RHF / r2SCAN /
PW1PW — see references.md); they are the established per-system references. For a
*uniform*-level reference set (e.g. all PBE/pob-TZVP-REV2), regenerate the `.d12`
method block first — tracked as a follow-up in references.md. Two systems have no
library `.d12`: ``tio2-anatase`` (literature geometry) and any future spec-only
crystal — they are listed as SKIPPED with the reason.
"""

from __future__ import annotations

import site_settings

import argparse
import os
import shutil

import testset

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))   # HERE is <checkout>/studies/aiccm-2026
TIER_MACHINE = site_settings.host_map("crystal_tier_hosts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", default=os.path.expanduser("~/gitlab/qc-input-library"),
                    help="QC input library root (holds crystal/<sys>/<level>/INPUT.d12)")
    ap.add_argument("--host", default=None, help="pin vq host (default: per-tier)")
    ap.add_argument("--system", default=None, choices=sorted(testset.SYSTEMS))
    args = ap.parse_args()

    systems = [args.system] if args.system else list(testset.SYSTEMS)
    targets = {name: (site_settings.explicit_remote_host(args.host) if args.host else TIER_MACHINE[testset.SYSTEMS[name]["tier"]]) for name in systems}
    jobs_root = os.path.join(HERE, "crystal23", "jobs")
    os.makedirs(jobs_root, exist_ok=True)
    crun = os.path.join(HERE, "crun.sh")

    lines, skipped = [], []
    for name in systems:
        meta = testset.SYSTEMS[name]
        ref = meta.get("crystal_ref", "")
        if not ref.startswith("crystal/"):           # literature / spec-only: no .d12
            skipped.append((name, ref or "no crystal_ref"))
            continue
        d12 = os.path.join(args.library, ref)
        if not os.path.isfile(d12):
            skipped.append((name, f"missing {ref}"))
            continue
        payload = os.path.join(jobs_root, name)
        os.makedirs(payload, exist_ok=True)
        shutil.copyfile(d12, os.path.join(payload, "INPUT.d12"))
        shutil.copyfile(crun, os.path.join(payload, "crun.sh"))
        host = targets[name]
        rel = os.path.relpath(payload, REPO_ROOT)   # repo-root-relative
        lines.append(f"vq submit {host}  -d {rel}/ -- bash crun.sh INPUT.d12 {name}")

    print(f"# CRYSTAL23 reference batch: {len(lines)} jobs "
          f"({len(skipped)} skipped). Review, then pipe to sh.")
    print("# Each writes <system>__crystal23.json to its $VQ_WORKDIR (energy_ha, converged).")
    for line in lines:
        print(line)
    if skipped:
        print("\n# ---- SKIPPED (no library .d12) ----")
        for name, why in skipped:
            print(f"#   {name}: {why}")


if __name__ == "__main__":
    main()
